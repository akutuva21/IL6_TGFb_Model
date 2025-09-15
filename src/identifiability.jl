using LinearAlgebra
using ComponentArrays
using PEtab
using Printf
using DataFrames
using Pkg

export compute_S_FIM, eigen_and_coord_metric, run_identifiability

"""
    coerce_to_petab_order(petab_problem, θ)

Return a ComponentVector whose keys and order exactly match `petab_problem.xnames`.
If `θ` is already in the correct order, it is returned unchanged.
"""
function coerce_to_petab_order(petab_problem::PEtab.PEtabODEProblem, θ::ComponentVector)
    ordered_syms = petab_problem.xnames
    if try collect(keys(θ)) == ordered_syms catch; false end
        return θ
    end
    vals = [θ[s] for s in ordered_syms]
    return ComponentArray(; (ordered_syms .=> vals)...)
end

"""
    get_noise_mode_and_params(petab_problem, θ)

Best-effort detection of the noise model and parameters from the PEtab tables
and the current parameter vector. Returns (mode, sigma_add, sigma_mult).
Falls back to (:combined, 0.05, 0.10) if unavailable.
"""
function get_noise_mode_and_params(petab_problem::PEtab.PEtabODEProblem, θ::ComponentVector)
    mode = :unknown
    sigma_add = 0.05
    sigma_mult = 0.10
    
    # Detect noise from observables table
    try
        obs_tbl = petab_problem.model_info.model.petab_tables[:observables]
        cols = names(obs_tbl)
        
        if :noiseDistribution in cols
            dist_col = obs_tbl[!, :noiseDistribution]
            
            # Check for lognormal first (more specific)
            if any(lowercase(String(d)) in ["lognormal", "log-normal"] for d in dist_col)
                mode = :lognormal
                @info "Detected lognormal noise distribution"
            # Then check for combined normal noise
            elseif any(lowercase(String(d)) == "normal" for d in dist_col) && 
                   :noiseFormula in cols
                form_col = obs_tbl[!, :noiseFormula]
                if any(occursin("sigma_add", String(f)) for f in form_col)
                    mode = :combined
                    @info "Detected combined normal noise distribution"
                else
                    mode = :lognormal  # Default to lognormal for single normal
                    @info "Detected simple normal noise, treating as lognormal"
                end
            end
        end
    catch e
        @warn "Failed to detect noise model from observables table: $e"
    end
    
    # Get sigma value from parameters (for lognormal case)
    if mode == :lognormal || mode == :unknown
        if haskey(θ, :log10_sigma_log_shared)
            try
                sigma_log = 10.0 ^ Float64(θ[:log10_sigma_log_shared])
                @info "Using sigma from log10_sigma_log_shared: $sigma_log"
                return :lognormal, sigma_log, sigma_log
            catch
            end
        elseif haskey(θ, :sigma_log_shared)
            try
                sigma_log = Float64(θ[:sigma_log_shared])
                @info "Using sigma from sigma_log_shared: $sigma_log"
                return :lognormal, sigma_log, sigma_log
            catch
            end
        else
            # Try to get from parameters table
            try
                ptab = petab_problem.model_info.model.petab_tables[:parameters]
                if :parameterId in names(ptab) && :nominalValue in names(ptab)
                    for row in eachrow(ptab)
                        if String(row.parameterId) == "sigma_log_shared"
                            sigma_log = Float64(row.nominalValue)
                            @info "Using sigma from parameters table: $sigma_log"
                            return :lognormal, sigma_log, sigma_log
                        end
                    end
                end
            catch
            end
        end
        # Default for lognormal
        return :lognormal, 0.01, 0.01  # 1% noise as fallback
    end
    
    return mode == :unknown ? :lognormal : mode, sigma_add, sigma_mult
end

"""
    compute_S_FIM(petab_problem, θ_full; perturb_syms=nothing, eps=1e-4, ...)

Compute measurement-aligned sensitivity matrix S via central differences by
perturbing only `perturb_syms` while always simulating with the full parameter
vector `θ_full`. Builds the noise-weighted FIM F = S' W S. Returns
(S, F, y0, noiseinfo, used_syms).
"""
function compute_S_FIM(
    petab_problem::PEtab.PEtabODEProblem,
    θ_full::ComponentVector;
    perturb_syms::Union{Nothing,Vector{Symbol}}=nothing,
    h::Float64=1e-4,
    eps_rel::Float64=1e-4,
    noise::Union{Symbol,Nothing}=nothing,
    sigma_add::Union{Nothing,Float64}=nothing,
    sigma_mult::Union{Nothing,Float64}=nothing,
)
    # Skip coercion since we already have the correct names from parameter discovery
    # θ_full = coerce_to_petab_order(petab_problem, θ_full)
    
    # CRITICAL: Debug parameter mapping before simulation
    @info "=== PEtab Parameter Mapping Diagnostic ==="
    @info "θ_full contents:"
    for (k, v) in pairs(θ_full)
        @info "  $k = $v"
    end
    
    @info "PEtab expected parameters:"
    @info "  xnames: $(petab_problem.xnames)"
    @info "  lower_bounds keys: $(collect(keys(petab_problem.lower_bounds)))"
    @info "  upper_bounds keys: $(collect(keys(petab_problem.upper_bounds)))"
    
    # Check name matching
    θ_keys = collect(keys(θ_full))
    xnames = petab_problem.xnames
    bounds_keys = collect(keys(petab_problem.lower_bounds))
    
    @info "Parameter name analysis:"
    @info "  θ_full has $(length(θ_keys)) parameters"
    @info "  xnames has $(length(xnames)) parameters"  
    @info "  bounds have $(length(bounds_keys)) parameters"
    
    missing_in_theta = setdiff(xnames, θ_keys)
    extra_in_theta = setdiff(θ_keys, xnames)
    if !isempty(missing_in_theta)
        @warn "Parameters in xnames but missing from θ_full: $missing_in_theta"
    end
    if !isempty(extra_in_theta)
        @warn "Parameters in θ_full but not in xnames: $extra_in_theta"
    end
    
    simvec = p -> petab_problem.simulated_values(p)
    
    # Test multiple simulation approaches
    @info "=== Testing different simulation methods ==="
    try
        # Method 1: Direct ComponentVector
        y0 = simvec(θ_full)
        @info "✅ Method 1 (ComponentVector) successful"
        
        # Method 2: Try with collected values
        θ_vector = collect(θ_full)
        y0_vector = simvec(θ_vector)
        vec_diff = norm(y0 - y0_vector, Inf)
        @info "Method 2 (Vector) difference from ComponentVector: ||Δy||_∞ = $vec_diff"
        
        # Method 3: Try nllh call to verify parameter application
        if hasmethod(petab_problem.nllh, (typeof(θ_full),))
            nllh_val = petab_problem.nllh(θ_full; prior=false)
            @info "✅ nllh call successful: $nllh_val"
        else
            @warn "nllh method not available for ComponentVector"
        end
        
    catch e
        @error "Simulation call failed: $e"
        @error "This confirms the parameter application issue"
        rethrow()
    end
    
    # Add PEtab.jl version check
    try
        petab_status = Pkg.status("PEtab"; mode=Pkg.PKGMODE_MANIFEST)
        @info "PEtab.jl package info available"
    catch
        @info "Could not retrieve PEtab.jl version"
    end
    @info "Julia version: $(VERSION)"
    
    # CRITICAL: Minimal parameter change test before proceeding
    @info "=== Minimal Parameter Change Test ==="
    if !isempty(θ_full)
        first_param = collect(keys(θ_full))[1]
        θ_minimal = ComponentArray(copy(θ_full))
        
        # Ensure we have baseline y0 for comparison
        y0_baseline = simvec(θ_minimal)
        
        # Try a very small change first
        original_val = θ_minimal[first_param]
        θ_minimal[first_param] += 0.001  # 0.1% change
        
        y_minimal = simvec(θ_minimal)
        minimal_diff = norm(y_minimal - y0_baseline, Inf)
        @info "Minimal change test ($first_param: $original_val → $(θ_minimal[first_param])): ||Δy||_∞ = $minimal_diff"
        
        if minimal_diff == 0.0
            @error "CONFIRMED: Even tiny parameter changes produce zero output difference"
            @error "PEtab.jl is definitely ignoring the parameter vector"
        else
            @info "✅ Parameter changes do affect simulation"
        end
    end
    
    # Compute baseline simulation for main sensitivity analysis
    y0 = simvec(θ_full)
    n = length(y0)
    ps = isnothing(perturb_syms) ? collect(keys(θ_full)) : perturb_syms
    
    @info "Computing sensitivity for $(length(ps)) parameters"
    @info "Baseline simulation has $(n) measurements"
    @info "Measurement range: $(minimum(y0)) to $(maximum(y0))"
    
    # CRITICAL: Verify parameter values being used in simulation
    @info "=== Parameter values verification ==="
    param_summary = []
    for (i, sym) in enumerate(collect(keys(θ_full))[1:min(5, length(θ_full))])  # Show first 5 params
        val = θ_full[sym]
        if startswith(String(sym), "log10_")
            linear_val = 10^val
            push!(param_summary, "$(sym)=$(val) → $(String(sym)[7:end])=$(linear_val)")
        else
            push!(param_summary, "$(sym)=$(val)")
        end
    end
    @info "Key parameters: $(join(param_summary, ", "))"
    if length(θ_full) > 5
        @info "... and $(length(θ_full)-5) more parameters"
    end
    
    # Filter to focus on informative measurements (non-zero or above threshold)
    active_indices = findall(x -> x > 1e-6, y0)
    @info "Using $(length(active_indices))/$(length(y0)) non-zero measurements (threshold: 1e-6)"
    
    # Manual sensitivity test with large perturbation
    if !isempty(ps)
        test_param = ps[1]
        θ_test = ComponentArray(copy(θ_full))
        θ_test[test_param] += 0.1 * abs(θ_full[test_param])  # Large 10% perturbation
        y_test = simvec(θ_test)
        manual_sens = norm(y_test - y0, Inf)
        @info "Manual sensitivity test for $(test_param): ||Δy||_∞ = $(manual_sens)"
        if manual_sens < 1e-12
            @warn "Manual test shows negligible sensitivity - model may be fundamentally non-identifiable"
        end
        
        # CRITICAL: Test if θ is actually being used by PEtab simulation
        @info "=== Verifying θ parameter application ==="
        
        # Test different parameter types that might exist
        test_params = [:log10_k_on, :log10_k_off, :log10_k_cat, :k_on, :k_off, :k_cat]
        param_tested = false
        
        for test_param_sym in test_params
            if haskey(θ_full, test_param_sym)
                θ_big = ComponentArray(copy(θ_full))
                if startswith(String(test_param_sym), "log10_")
                    θ_big[test_param_sym] += 1.0  # 10x increase
                    @info "Testing $(test_param_sym): $(θ_full[test_param_sym]) → $(θ_big[test_param_sym]) (10x change)"
                else
                    θ_big[test_param_sym] *= 10.0  # 10x increase
                    @info "Testing $(test_param_sym): $(θ_full[test_param_sym]) → $(θ_big[test_param_sym]) (10x change)"
                end
                
                y_big = simvec(θ_big)
                θ_effect = norm(y_big - y0, Inf)
                @info "Parameter $(test_param_sym) effect: ||Δy||_∞ = $(θ_effect)"
                
                if θ_effect < 1e-10
                    @error "CRITICAL: Large $(test_param_sym) change shows no effect!"
                else
                    @info "✅ $(test_param_sym) changes are affecting simulation"
                end
                param_tested = true
                break  # Test only the first parameter found
            end
        end
        
        if !param_tested
            @warn "No recognizable kinetic parameters found for verification"
            @info "Available parameters: $(collect(keys(θ_full)))"
        end
        
        # Test condition-specific overrides
        @info "=== Verifying condition-specific parameter application ==="
        try
            # Get condition IDs from PEtab problem
            cond_table = petab_problem.model_info.model.petab_tables[:conditions]
            if !isnothing(cond_table) && :conditionId in names(cond_table)
                cond_ids = unique(cond_table.conditionId)
                @info "Found conditions: $(cond_ids)"
                
                # Simulate first condition and check if different conditions give different results
                if length(cond_ids) > 1
                    # This is tricky - we need to check if conditions are being applied
                    # For now, just report that we found multiple conditions
                    @info "Multiple conditions detected - condition overrides should be active"
                    @info "If all conditions produce identical trajectories, overrides may not be working"
                end
            else
                @info "No conditions table found or no conditionId column"
            end
        catch e
            @warn "Could not verify condition overrides: $e"
        end
    end
    
    S = Matrix{Float64}(undef, length(active_indices), length(ps))  # Use filtered measurements
    zero_cols = falses(length(ps))
    lb = petab_problem.lower_bounds
    ub = petab_problem.upper_bounds
    
    for (j, pj) in enumerate(ps)
        θp = ComponentArray(copy(θ_full))
        θm = ComponentArray(copy(θ_full))
        
        # Use larger step sizes for better sensitivity detection
        θ_val = θ_full[pj]
        δ = max(1e-2, 1e-2 * abs(θ_val))  # 1% perturbations instead of 0.01%
        
        # Check bounds more carefully
        δ_up = min(δ, (ub[pj] - θ_val) * 0.5)
        δ_down = min(δ, (θ_val - lb[pj]) * 0.5)
        δ_final = min(δ_up, δ_down)
        
        @info "Parameter $(pj): value=$(θ_val), δ=$(δ_final), bounds=[$(lb[pj]), $(ub[pj])]"
        
        if δ_final <= 1e-8  # More generous threshold
            @warn "Parameter $(pj) too close to bounds - zero sensitivity"
            S[:, j] .= 0.0
            zero_cols[j] = true
            continue
        end
        
        θp[pj] = θ_val + δ_final
        θm[pj] = θ_val - δ_final
        
        # Verify the parameter change for first few parameters
        if j <= 3  # Only log first 3 to avoid spam
            if startswith(String(pj), "log10_")
                linear_base = 10^θ_val
                linear_plus = 10^(θ_val + δ_final)
                linear_minus = 10^(θ_val - δ_final)
                @info "  $(pj): $(θ_val) → [$(θ_val-δ_final), $(θ_val+δ_final)]"
                @info "  Linear: $(linear_base) → [$(linear_minus), $(linear_plus)]"
            else
                @info "  $(pj): $(θ_val) → [$(θ_val-δ_final), $(θ_val+δ_final)]"
            end
        end
        
        yp = simvec(θp)
        ym = simvec(θm)
        
        # Verify simulation results changed
        if j <= 3
            yp_range = (minimum(yp), maximum(yp))
            ym_range = (minimum(ym), maximum(ym))
            @info "  y_plus range: $(yp_range)"
            @info "  y_minus range: $(ym_range)"
            @info "  y_baseline range: $((minimum(y0), maximum(y0)))"
        end
        
        # Compute sensitivity using only active measurements
        sens_full = (yp - ym) / (2 * δ_final)
        sens = sens_full[active_indices]  # Extract only informative measurements
        S[:, j] = sens
        
        sens_range = (minimum(sens), maximum(sens))
        sens_norm = norm(sens, Inf)
        @info "Parameter $(pj) sensitivity: range=$(sens_range), ||·||_∞=$(sens_norm)"
        
        if sens_norm < 1e-10
            @warn "Parameter $(pj) has negligible sensitivity (||·||_∞ < 1e-10)"
        end
    end

    # Resolve noise settings with debugging
    nm, sa, sm = get_noise_mode_and_params(petab_problem, θ_full)
    @info "Final noise model: $(nm), sa=$(sa), sm=$(sm)"

    # Build weights with debugging
    w = ones(length(active_indices))  # Weights only for active measurements
    if nm == :lognormal
        σ = sa  # Use the first value as sigma for lognormal
        @info "Using lognormal weights with σ=$(σ)"
        
        # Apply row scaling for log-normal: S[i,:] *= 1/y[i] (only for active measurements)
        y0_active = y0[active_indices]
        for i in 1:length(active_indices)
            scale_factor = 1.0 / max(abs(y0_active[i]), 1e-12)
            for j in 1:size(S, 2)
                S[i, j] *= scale_factor
            end
        end
        
        # Uniform weights for log-normal
        w .= 1.0 / max(σ^2, 10 * Base.eps(Float64))
        @info "Applied lognormal scaling and weights to $(length(active_indices)) measurements"
    else
        @info "Using uniform weights (noise model: $(nm))"
    end
    
    # Check for degenerate sensitivity matrix
    S_max = maximum(abs.(S))
    S_norm_F = norm(S, 2)  # Frobenius norm
    @info "Sensitivity matrix: max(|S|)=$(S_max), ||S||_F=$(S_norm_F)"
    
    if S_max < 1e-10
        @error "Sensitivity matrix is effectively zero - fundamental identifiability problem"
    end
    
    F = S' * Diagonal(w) * S
    
    # Enhanced FIM diagnostics
    F_cond = cond(F)
    F_eigvals = eigvals(F)
    F_rank = rank(F, 1e-12)
    @info "FIM diagnostics:"
    @info "  - Condition number: $(F_cond)"
    @info "  - Rank: $(F_rank)/$(size(F,1))"
    @info "  - Eigenvalue range: $(minimum(F_eigvals)) to $(maximum(F_eigvals))"
    @info "  - Det(F): $(det(F))"
    
    if F_cond > 1e12
        @warn "FIM is near-singular (cond=$(F_cond)) - poor identifiability"
    end
    return S, F, y0, (noise_mode = nm, sigma_add = sa, sigma_mult = sm), ps, zero_cols
end

"""
    eigen_and_coord_metric(S, F; tol_factor=1e-4)

Compute eigen-decomposition of F, infer rank with tol = tol_factor * λmax, and the
coordinate identifiability metric m_i = ||(I - A A^†) s_i||_∞ for each column.
Returns a NamedTuple of diagnostics.
"""
function eigen_and_coord_metric(S::AbstractMatrix, F::AbstractMatrix; tol_factor=1e-4)
    λ, U = eigen(Symmetric(F))
    λmax = maximum(λ)
    tol = tol_factor * λmax
    rank = count(>(tol), λ)
    null_dim = size(F, 1) - rank

    k = size(S, 2)
    m = zeros(k)
    for i in 1:k
        si = S[:, i]
        cols = setdiff(1:k, (i,))
        A = S[:, cols]
        # Projection onto orthogonal complement of range(A)
        P = I - A * pinv(A)
        r = P * si
        m[i] = norm(r, Inf)
    end
    return (λ = λ, U = U, tol = tol, rank = rank, null_dim = null_dim, metric = m)
end

"""
    run_identifiability(petab_problem, θ; eps=1e-4)

End-to-end diagnostic: computes S and F, runs eigen-analysis and coordinate
metrics, and prints a concise report. Returns a NamedTuple with all artifacts.
"""
function run_identifiability(petab_problem::PEtab.PEtabODEProblem, θ_full::ComponentVector; eps=1e-4)
    # Use the actual parameter names from θ_full
    all_param_keys = collect(keys(θ_full))
    
    # Filter to exclude noise parameters and initial conditions
    est_syms = filter(all_param_keys) do s
        s_str = String(s)
        return !occursin("sigma", s_str) && !endswith(s_str, "_0")
    end
    
    @info "Perturbing $(length(est_syms)) parameters: $est_syms"

    S, F, y0, noiseinfo, used_syms, zero_cols = compute_S_FIM(petab_problem, θ_full; perturb_syms=est_syms, h=eps)
    stats = eigen_and_coord_metric(S, F)

    names = string.(used_syms)
    println("\n=== Identifiability diagnostics ===")
    println("Noise: $(noiseinfo.noise_mode), sigma_add=$(round(noiseinfo.sigma_add, digits=4)), sigma_mult=$(round(noiseinfo.sigma_mult, digits=4))")
    println("FIM eigenvalues: min=$(round(minimum(stats.λ), digits=6)), max=$(round(maximum(stats.λ), digits=6))")
    println("Rank(F)=$(stats.rank) of $(length(names)); null-space size=$(stats.null_dim); tol=$(round(stats.tol, digits=6))")
    println("FIM invertible: ", stats.rank == length(names) ? "yes" : "no")
    
    # Enhanced interpretation
    if stats.rank == length(names)
        println("✅ All parameters appear to be identifiable (locally)")
    elseif stats.rank == 0
        println("❌ CRITICAL: No parameters are identifiable - fundamental problem")
    else
        println("⚠️  PARTIAL: $(stats.null_dim) parameter(s) not identifiable - check coordinate metrics")
    end

    # Optional: print eigenvalue table with null flag
    try
        println("\nEigenvalues (flagged if < tol):")
        println(rpad("#", 6), rpad("eigenvalue", 18), "flag")
        println("-"^34)
        for (i, λi) in enumerate(stats.λ)
            flag = λi < stats.tol ? "null" : "ok"
            @printf("%-6d%-18.6g%s\n", i, λi, flag)
        end
    catch
        # printing is best-effort only
    end

    # Coordinate metric table, sorted descending
    ord = sortperm(stats.metric, rev=true)
    println(rpad("Parameter", 28), " | ", lpad("m_i (inf-norm)", 14), " | Status")
    println("-"^55)
    for idx in ord
        mark = zero_cols[idx] ? " *bound" : ""
        metric_val = stats.metric[idx]
        status = if metric_val > 1e-3
            "Good"
        elseif metric_val > 1e-6
            "Weak"
        elseif metric_val > 1e-10
            "Poor"
        else
            "None"
        end
        @printf("%-28s | %14.6g%s | %s\n", names[idx], metric_val, mark, status)
    end
    if any(zero_cols)
        println("* Parameters marked 'bound' had zero sensitivities due to step clamping at bounds.")
    end
    
    # Summary recommendations
    println("\n=== Recommendations ===")
    poor_params = count(m -> m < 1e-6, stats.metric)
    if poor_params == 0
        println("✅ All parameters show reasonable identifiability")
    elseif poor_params < length(stats.metric) / 2
        println("⚠️  $(poor_params) parameter(s) poorly identifiable - consider:")
        println("   • Parameter fixing or prior constraints")
        println("   • Model reparameterization")
        println("   • Additional experimental data")
    else
        println("❌ Most parameters poorly identifiable - consider:")
        println("   • Model simplification")
        println("   • Different experimental design")
        println("   • Profile likelihood for verification")
    end
    return (S = S, F = F, eigen = stats, names = names, y0 = y0)
end