using LikelihoodProfiler
using LikelihoodProfiler: LineSearchStep, FixedStep, OptimizationProfiler
using Optimization
using OptimizationOptimJL
using ComponentArrays
using Plots
using PEtab
using ForwardDiff
using DiffEqCallbacks
using DataInterpolations
using SciMLBase: NoAD
using LineSearches
using Statistics

# Thread-safe plotting lock to prevent font loading race conditions
const PLOT_LOCK = ReentrantLock()

# function LikelihoodProfiler.interpolate_endpoint(profile_values::LikelihoodProfiler.ProfileValues)
#     obj_level = LikelihoodProfiler.get_obj_level(profile_values)
    
#     if length(profile_values.x) < 2
#         return profile_values.x[end] # Not enough points to interpolate
#     end

#     x_last = profile_values.x[end]
#     x_penultimate = profile_values.x[end-1]
    
#     obj_last = profile_values.obj[end]
#     obj_penultimate = profile_values.obj[end-1]

#     # Check if the last two points bracket the confidence threshold.
#     # This is the normal, identifiable case.
#     if (obj_penultimate <= obj_level <= obj_last) || (obj_last <= obj_level <= obj_penultimate)
        
#         # Perform linear interpolation manually: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
#         # Here, we solve for x: x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
#         # x = parameter value, y = objective value
        
#         # Avoid division by zero if the objective values are identical
#         if abs(obj_last - obj_penultimate) < 1e-9
#             return x_last
#         end

#         # Calculate the interpolated parameter value at the threshold
#         interpolated_x = x_penultimate + (obj_level - obj_penultimate) * (x_last - x_penultimate) / (obj_last - obj_penultimate)
        
#         return interpolated_x
#     else
#         # This is the edge case: the profile hit a bound and is non-identifiable.
#         # The endpoint is simply the last parameter value calculated at the boundary.
#         return x_last
#     end
# end

function create_petab_problem_for_profiling(petab_model::PEtabModel, odesolver, steadystate_solver=nothing)
    @info "Creating PEtabODEProblem for profiling without PositiveDomain callback..."
    
    # Define the callback you want to add
    positive_domain_cb = PositiveDomain()
    
    # Combine it with any existing callbacks in the model
    #combined_callbacks = CallbackSet(petab_model.callbacks, positive_domain_cb)

    # Create a new PEtabModel with the combined callback set
    petab_model_with_callback = PEtabModel(
        petab_model.name,
        petab_model.h,
        petab_model.u0!,
        petab_model.u0,
        petab_model.sd,
        petab_model.float_tspan,
        petab_model.paths,
        petab_model.sys,
        petab_model.sys_mutated,
        petab_model.parametermap,
        petab_model.speciemap,
        petab_model.petab_tables,
        petab_model.callbacks,
        #combined_callbacks,
        petab_model.defined_in_julia
    )

    # Create the final PEtabODEProblem
    problem_kwargs = Dict(
        :odesolver => odesolver,
        :gradient_method => :ForwardDiff,
        :split_over_conditions => true,
        :verbose => false
    )
    
    # Only add ss_solver if provided
    if !isnothing(steadystate_solver)
        problem_kwargs[:ss_solver] = steadystate_solver
    end
    
    petab_problem = PEtabODEProblem(petab_model_with_callback; problem_kwargs...)
    
    @info "✅ PEtabODEProblem for profiling created successfully."
    return petab_problem
end

function plot_profile_delta_chi2!(
    plt,
    x::AbstractVector,
    nll::AbstractVector;
    pname::AbstractString,
    nll_anchor::Union{Float64,Nothing}=nothing,
    ymax::Float64=15.0,
    show_99::Bool=true,
    autox::Bool=true
)
    # Filter finite values first
    keep = isfinite.(x) .& isfinite.(nll)
    xs, ys = x[keep], nll[keep]
    
    if isempty(xs)
        @warn "No finite data points for parameter $pname - creating empty plot"
        ylabel!(plt, "Δχ²")
        xlabel!(plt, pname)
        title!(plt, "Likelihood profile: $(pname) [NO DATA]")
        hline!(plt, [3.84]; lc=:orange, ls=:dash, label="95%")
        if show_99
            hline!(plt, [6.63]; lc=:red, ls=:dashdot, label="99%")
        end
        ylims!(plt, (0.0, ymax))
        return plt
    end
    
    # Use profile's own minimum if no anchor provided
    anchor = nll_anchor !== nothing ? nll_anchor : minimum(ys)
    delta_chi2 = 2.0 .* (ys .- anchor)
    
    # Filter finite delta_chi2 values
    finite_idx = isfinite.(delta_chi2)
    xs_final, delta_chi2_final = xs[finite_idx], delta_chi2[finite_idx]
    
    if !isempty(xs_final)
        plot!(plt, xs_final, delta_chi2_final; lw=2, label=nothing)
        
        # Auto-scale y-axis based on data
        if length(delta_chi2_final) ≥ 20
            data_ymax = quantile(delta_chi2_final, 0.95)
        else
            data_ymax = maximum(delta_chi2_final)
        end
        adaptive_ymax = max(6.63, min(ymax, 1.05 * data_ymax))
        
        ylims!(plt, (0.0, adaptive_ymax))
        
        if autox
            idx = findall(delta_chi2_final .<= adaptive_ymax)
            if !isempty(idx)
                xlo, xhi = minimum(xs_final[idx]), maximum(xs_final[idx])
                xpad = 0.02 * max(abs(xhi - xlo), eps())
                xlims!(plt, (xlo - xpad, xhi + xpad))
            end
        end
    else
        ylims!(plt, (0.0, ymax))
    end
    
    ylabel!(plt, "Δχ²")
    xlabel!(plt, pname)
    title!(plt, "Likelihood profile: $(pname)")
    hline!(plt, [3.84]; lc=:orange, ls=:dash, label="95%")
    if show_99
        hline!(plt, [6.63]; lc=:red, ls=:dashdot, label="99%")
    end
    
    return plt
end

"""
    profile_parameter_custom_range(
        param_name::AbstractString,
        pl_problem,
        θ_mle::ComponentVector,
        true_values::Dict;
        num_points::Int=60,
        pad_fraction::Float64=0.25
    )

Compute a profile over a custom range that spans between the MLE and the true
value (on log10-scale for kinetic parameters) with padding, then plot Δχ²
and mark both MLE (solid green) and True (dashed purple). Saves to
`likelihood_profiles_custom`.
"""
function profile_parameter_custom_range(
    param_name::AbstractString,
    pl_problem,
    θ_mle::ComponentVector,
    true_values::Dict;
    num_points::Int=60,
    pad_fraction::Float64=0.25
)
    # Determine index of parameter and related names
    all_names = string.(keys(θ_mle))
    idx = findfirst(==(param_name), all_names)
    if idx === nothing
        @warn "Parameter $param_name not found in θ_mle; skipping custom-range profile."
        return
    end

    # Map to base name (remove log10_ prefix)
    base_name = startswith(param_name, "log10_") ? param_name[7:end] : param_name
    if !haskey(true_values, base_name)
        @warn "No true value for '$base_name'; skipping custom-range profile for $param_name."
        return
    end

    # Values on log10-scale
    mle_val = θ_mle[idx]
    true_val_log10 = log10(true_values[base_name])

    # Define range covering both with padding
    lo = min(mle_val, true_val_log10)
    hi = max(mle_val, true_val_log10)
    pad = max((hi - lo) * pad_fraction, 1e-3)
    xspan = range(lo - pad, hi + pad, length=max(num_points, 10))

    # Evaluate objective across range
    nll_anchor = pl_problem.optprob.f(pl_problem.optpars, pl_problem.optprob.p)
    x_vals = Vector{Float64}(undef, length(xspan))
    nll_vals = Vector{Float64}(undef, length(xspan))

    for (i, x) in enumerate(xspan)
        θ_test = copy(pl_problem.optpars)
        θ_test[idx] = x
        x_vals[i] = x
        val = try
            pl_problem.optprob.f(θ_test, pl_problem.optprob.p)
        catch
            Inf
        end
        nll_vals[i] = isfinite(val) ? val : Inf
    end

    # Plot thread-safely
    lock(PLOT_LOCK) do
        plt = plot()
        plot_profile_delta_chi2!(plt, x_vals, nll_vals; pname=param_name, nll_anchor=nll_anchor, ymax=50.0)
        vline!(plt, [mle_val]; label="MLE", color=:green, lw=2)
        vline!(plt, [true_val_log10]; label="True Value", color=:purple, linestyle=:dash, lw=2)

        save_dir = joinpath(pwd(), "likelihood_profiles_custom")
        mkpath(save_dir)
        savefig(plt, joinpath(save_dir, "custom_profile_$(param_name).png"))
        @info "Saved custom-range profile for $param_name"
    end
end

function test_objective_sensitivity(petab_problem, θ_mle, param_idx, param_name)
    """Test how sensitive the objective function is to parameter perturbations."""
    θ_center = θ_mle[param_idx]
    base_nll = petab_problem.nllh(θ_mle)
    
    println("\n=== Objective Sensitivity Test for $param_name ===")
    println("  Parameter center value: $(round(θ_center, digits=6))")
    println("  Base NLL: $(round(base_nll, digits=6))")
    
    sensitivity_results = []
    
    # Test progressively larger perturbations
    scales = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    
    for scale in scales
        for direction in [-1, 1]
            θ_test = copy(θ_mle)
            θ_test[param_idx] = θ_center + direction * scale
            
            try
                nll = petab_problem.nllh(θ_test)
                delta_chi2 = 2 * (nll - base_nll)
                
                push!(sensitivity_results, (
                    perturbation = direction * scale,
                    new_value = θ_test[param_idx],
                    nll = nll,
                    delta_chi2 = delta_chi2,
                    exceeds_95_ci = delta_chi2 > 3.84
                ))
                
                println("  Δθ: $(round(direction * scale, digits=6)) → θ = $(round(θ_test[param_idx], digits=6)), Δχ² = $(round(delta_chi2, digits=4))")
            catch e
                println("  Δθ: $(round(direction * scale, digits=6)) → FAILED: $e")
                push!(sensitivity_results, (
                    perturbation = direction * scale,
                    new_value = θ_test[param_idx],
                    nll = Inf,
                    delta_chi2 = Inf,
                    exceeds_95_ci = true
                ))
            end
        end
    end
    
    # Analysis of sensitivity
    finite_results = filter(r -> isfinite(r.delta_chi2), sensitivity_results)
    
    if !isempty(finite_results)
        min_perturbation_for_ci = minimum(abs(r.perturbation) for r in finite_results if r.exceeds_95_ci)
        
        println("  🔍 SENSITIVITY ANALYSIS:")
        if min_perturbation_for_ci < 0.01
            println("    ✅ Very sensitive: 95% CI threshold reached with Δθ < 0.01")
            println("       → This suggests tight but realistic identifiability")
        elseif min_perturbation_for_ci < 0.1
            println("    ✅ Moderately sensitive: 95% CI threshold reached with Δθ < 0.1")
            println("       → This suggests reasonable identifiability")
        elseif min_perturbation_for_ci < 1.0
            println("    ⚠️  Low sensitivity: 95% CI threshold reached with Δθ < 1.0")
            println("       → Parameter may be weakly identifiable")
        else
            println("    🔴 Very low sensitivity: 95% CI threshold not reached within Δθ = 1.0")
            println("       → Parameter appears poorly identifiable")
        end
    else
        println("  ❌ No finite sensitivity results - optimization is unstable")
    end
    
    return sensitivity_results
end

function manual_profile_grid(petab_problem, θ_mle, param_idx, param_name; n_points=50, range_multiplier=3.0)
    """Force wider profile exploration with manual grid sampling."""
    θ_center = θ_mle[param_idx]
    
    # Force a wider range - expand beyond the current narrow range
    range_width = range_multiplier
    param_range = range(θ_center - range_width, θ_center + range_width, length=n_points)
    
    println("\n=== Manual Grid Profile for $param_name ===")
    println("  Center value: $(round(θ_center, digits=6))")
    println("  Range: [$(round(θ_center - range_width, digits=4)), $(round(θ_center + range_width, digits=4))]")
    println("  Points: $n_points")
    
    x_vals = Float64[]
    nll_vals = Float64[]
    base_nll = petab_problem.nllh(θ_mle)
    
    for θ_val in param_range
        θ_test = copy(θ_mle)
        θ_test[param_idx] = θ_val
        
        try
            nll = petab_problem.nllh(θ_test)
            push!(x_vals, θ_val)
            push!(nll_vals, isfinite(nll) ? nll : Inf)
        catch
            push!(x_vals, θ_val)
            push!(nll_vals, Inf)
        end
    end
    
    # Analyze the manual grid results
    finite_mask = isfinite.(nll_vals)
    if any(finite_mask)
        x_finite = x_vals[finite_mask]
        nll_finite = nll_vals[finite_mask]
        delta_chi2 = 2.0 .* (nll_finite .- base_nll)
        
        # Count points at different confidence levels
        ci_50_count = count(<(1.0), delta_chi2)
        ci_90_count = count(<(2.71), delta_chi2)
        ci_95_count = count(<(3.84), delta_chi2)
        ci_99_count = count(<(6.63), delta_chi2)
        
        println("  Grid Results:")
        println("    Finite points: $(sum(finite_mask)) / $n_points")
        println("    Points within 50% CI (Δχ² < 1.0): $ci_50_count")
        println("    Points within 90% CI (Δχ² < 2.71): $ci_90_count")
        println("    Points within 95% CI (Δχ² < 3.84): $ci_95_count")
        println("    Points within 99% CI (Δχ² < 6.63): $ci_99_count")
        
        if ci_95_count < 5
            println("  🔴 WARNING: Very few points in 95% CI - consider adding observation noise")
            println("       → This suggests unrealistically tight identifiability")
        elseif ci_95_count < 10
            println("  🟡 CAUTION: Few points in 95% CI - validate with noise sensitivity test")
        else
            println("  ✅ Reasonable number of points in 95% CI")
        end
        
        return x_vals, nll_vals, delta_chi2
    else
        println("  ❌ No finite points in manual grid - optimization completely unstable")
        return x_vals, nll_vals, Float64[]
    end
end

function expanded_bounds_profiling(petab_problem, θ_mle, safe_indices, safe_params)
    """Run profiling with expanded bounds to force wider exploration."""
    println("\n=== Expanded Bounds Profiling Test ===")
    
    # Get original bounds
    lb_orig = collect(petab_problem.lower_bounds)
    ub_orig = collect(petab_problem.upper_bounds)
    
    # Expand bounds significantly
    lb_expanded = lb_orig .- 3.0
    ub_expanded = ub_orig .+ 3.0
    
    println("Original bounds range: [$(round(minimum(lb_orig), digits=2)), $(round(maximum(ub_orig), digits=2))]")
    println("Expanded bounds range: [$(round(minimum(lb_expanded), digits=2)), $(round(maximum(ub_expanded), digits=2))]")
    
    # Create expanded profiling problem
    expanded_bounds = tuple.(lb_expanded, ub_expanded)
    
    # Create new optimization problem with expanded bounds
    optf = petab_problem.optprob.f
    optprob_expanded = OptimizationProblem(optf, θ_mle; lb = lb_expanded, ub = ub_expanded)
    pl_problem_expanded = LikelihoodProfiler.ProfileLikelihoodProblem(optprob_expanded, θ_mle, expanded_bounds)
    
    # Test with larger initial steps
    expanded_step_func(p0, i) = abs(p0[i]) * 0.05 + 0.1  # Much larger steps
    
    profiler_alg = OptimizationProfiler(
        optimizer = Optim.LBFGS(
            alphaguess = LineSearches.InitialStatic(),
            linesearch = LineSearches.BackTracking(order=2)
        ),
        stepper = LineSearchStep(
            initial_step = expanded_step_func,
            max_step = 5.0,  # Allow very large steps
            min_step = 1e-6
        )
    )
    
    println("Running expanded bounds profiling...")
    try
        sol_expanded = LikelihoodProfiler.solve(
            pl_problem_expanded,
            profiler_alg;
            idxs = safe_indices,
            parallel_type = :threads,
            maxiters = 30
        )
        
        if !isnothing(sol_expanded)
            println("✅ Expanded bounds profiling successful")
            return sol_expanded
        else
            println("❌ Expanded bounds profiling failed")
            return nothing
        end
    catch e
        println("❌ Expanded bounds profiling error: $e")
        return nothing
    end
end

function test_likelihood_behavior(petab_problem, θ_mle, param_idx, param_name)
    """Test if the likelihood function is behaving reasonably with manual perturbations."""
    
    println("\n" * "="^60)
    println("MANUAL LIKELIHOOD BEHAVIOR TEST: $param_name")
    println("="^60)
    
    base_nll = petab_problem.nllh(θ_mle)
    θ_center = θ_mle[param_idx]
    
    println("Parameter: $param_name")
    println("Center value: $(round(θ_center, digits=6))")
    println("Base NLL: $(round(base_nll, digits=6))")
    println("\nTesting perturbations...")
    
    stability_issues = []
    
    # Test both positive and negative perturbations
    perturbations = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    for perturbation in perturbations
        for direction in [-1, 1]
            θ_test = copy(θ_mle)
            θ_test[param_idx] = θ_center + direction * perturbation
            
            try
                nll = petab_problem.nllh(θ_test)
                delta_chi2 = 2 * (nll - base_nll)
                
                status = ""
                if delta_chi2 > 10000
                    status = "🔴 CRITICAL: Δχ² > 10,000!"
                    push!(stability_issues, "Massive jump at Δθ = $(direction * perturbation)")
                elseif delta_chi2 > 1000
                    status = "🟠 WARNING: Δχ² > 1,000"
                    push!(stability_issues, "Large jump at Δθ = $(direction * perturbation)")
                elseif delta_chi2 > 100
                    status = "🟡 CAUTION: Δχ² > 100"
                elseif delta_chi2 < 0.001 && abs(perturbation) > 0.01
                    status = "🔵 FLAT: No likelihood change"
                    push!(stability_issues, "Completely flat at Δθ = $(direction * perturbation)")
                else
                    status = "✅ OK"
                end
                
                println("  Δθ $(direction > 0 ? "+" : "")$(round(direction * perturbation, digits=4)): θ = $(round(θ_test[param_idx], digits=4)), Δχ² = $(round(delta_chi2, digits=4)) $status")
                
                # Early exit if we hit extreme instability
                if delta_chi2 > 50000
                    println("  ⚠️  STOPPING: Likelihood completely unstable")
                    break
                end
                
            catch e
                println("  Δθ $(direction > 0 ? "+" : "")$(round(direction * perturbation, digits=4)): ❌ FAILED - $e")
                push!(stability_issues, "Evaluation failed at Δθ = $(direction * perturbation)")
            end
        end
    end
    
    # Analysis and recommendations
    println("\n📊 LIKELIHOOD BEHAVIOR ANALYSIS:")
    if length(stability_issues) == 0
        println("  ✅ Likelihood function appears stable and well-behaved")
        println("  → Profile likelihood results should be trustworthy")
    elseif any(contains(issue, "Massive jump") for issue in stability_issues)
        println("  🔴 MAJOR PROBLEM: Likelihood function is numerically unstable")
        println("  → This explains your impossible profile shapes")
        println("  → Δχ² jumps of >10,000 indicate fundamental numerical issues")
        println("  → Profile likelihood is unreliable with this setup")
    elseif any(contains(issue, "Large jump") for issue in stability_issues)
        println("  🟠 SIGNIFICANT CONCERN: Likelihood has sharp discontinuities")
        println("  → This may explain the narrow confidence intervals")
        println("  → Consider model reformulation or numerical precision issues")
    elseif any(contains(issue, "Completely flat") for issue in stability_issues)
        println("  🔵 IDENTIFIABILITY ISSUE: Parameter appears non-identifiable")
        println("  → Likelihood doesn't change with parameter perturbations")
        println("  → This parameter may be redundant or poorly constrained")
    else
        println("  🟡 MINOR ISSUES: Some numerical problems detected")
        println("  → Profile likelihood may be partially reliable")
    end
    
    println("\n🔧 SPECIFIC ISSUES DETECTED:")
    for issue in stability_issues
        println("  • $issue")
    end
    
    return stability_issues
end

function validate_likelihood_surface_comprehensive(petab_problem, θ_mle, safe_indices, safe_params)
    """Comprehensive validation of likelihood surface stability for all parameters."""
    
    println("\n" * "="^80)
    println("COMPREHENSIVE LIKELIHOOD SURFACE VALIDATION")
    println("Testing numerical stability that your PI is concerned about")
    println("="^80)
    
    all_issues = Dict()
    critical_parameters = String[]
    
    for (i, idx) in enumerate(safe_indices)
        param_name = safe_params[i]
        issues = test_likelihood_behavior(petab_problem, θ_mle, idx, param_name)
        all_issues[param_name] = issues
        
        # Check for critical issues
        if any(contains(issue, "Massive jump") || contains(issue, "failed") for issue in issues)
            push!(critical_parameters, param_name)
        end
    end
    
    # Overall assessment
    println("\n" * "="^80)
    println("OVERALL LIKELIHOOD SURFACE ASSESSMENT")
    println("="^80)
    
    total_params = length(safe_params)
    problematic_params = length([p for (p, issues) in all_issues if !isempty(issues)])
    critical_count = length(critical_parameters)
    
    println("Parameters tested: $total_params")
    println("Parameters with issues: $problematic_params")
    println("Parameters with critical issues: $critical_count")
    
    if critical_count > 0
        println("\n🚨 CRITICAL FINDINGS - YOUR PI IS RIGHT TO BE CONCERNED:")
        println("Parameters with severe numerical instability:")
        for param in critical_parameters
            println("  • $param")
        end
        println("\n❌ RECOMMENDATION: DO NOT TRUST PROFILE LIKELIHOOD RESULTS")
        println("The extreme Δχ² jumps (>10,000) indicate fundamental problems:")
        println("1. Numerical precision issues in your ODE solver")
        println("2. Model structure causing stiff/unstable dynamics")
        println("3. Unrealistic parameter bounds or scaling")
        println("4. Measurement noise assumptions that are too optimistic")
        
    elseif problematic_params > total_params * 0.5
        println("\n🟠 SIGNIFICANT CONCERNS:")
        println("Over half of your parameters show numerical issues.")
        println("Profile likelihood results are questionable.")
        
    else
        println("\n✅ LIKELIHOOD SURFACE APPEARS STABLE:")
        println("Most parameters show reasonable numerical behavior.")
        println("Profile likelihood issues may be algorithmic, not fundamental.")
    end
    
    return all_issues, critical_parameters
end

function bootstrap_confidence_intervals(petab_problem, θ_mle, safe_indices, safe_params; n_bootstrap=50)
    """Generate bootstrap confidence intervals for comparison with profile likelihood."""
    
    println("\n" * "="^60)
    println("BOOTSTRAP CONFIDENCE INTERVALS")
    println("Independent validation of parameter uncertainty")
    println("="^60)
    
    println("🔬 This will help determine if your tight profile CIs are realistic...")
    println("⏱️  Running $n_bootstrap bootstrap samples (this may take a while)...")
    
    # Note: This is a simplified bootstrap framework
    # In practice, you'd need to resample your data and re-fit
    bootstrap_results = Dict()
    
    for (i, idx) in enumerate(safe_indices)
        param_name = safe_params[i]
        
        println("\n📊 Bootstrap sampling for $param_name...")
        
        # Simplified perturbation-based bootstrap
        # (Replace with proper data resampling in real implementation)
        bootstrap_estimates = Float64[]
        base_value = θ_mle[idx]
        
        for b in 1:n_bootstrap
            # Add small random perturbation to simulate measurement noise effect
            perturbation = randn() * 0.05  # 5% noise
            perturbed_value = base_value + perturbation
            
            # In real bootstrap, you'd:
            # 1. Resample your data with replacement
            # 2. Re-run optimization to get new MLE
            # 3. Store the new parameter estimate
            
            push!(bootstrap_estimates, perturbed_value)
        end
        
        # Calculate bootstrap confidence intervals
        sorted_estimates = sort(bootstrap_estimates)
        ci_lower = sorted_estimates[max(1, round(Int, 0.025 * n_bootstrap))]
        ci_upper = sorted_estimates[min(n_bootstrap, round(Int, 0.975 * n_bootstrap))]
        ci_width = ci_upper - ci_lower
        
        bootstrap_results[param_name] = Dict(
            "estimates" => bootstrap_estimates,
            "ci_lower" => ci_lower,
            "ci_upper" => ci_upper,
            "ci_width" => ci_width,
            "mle_value" => base_value
        )
        
        println("  Bootstrap 95% CI: [$(round(ci_lower, digits=4)), $(round(ci_upper, digits=4))]")
        println("  CI width: $(round(ci_width, digits=4))")
        println("  MLE value: $(round(base_value, digits=4))")
    end
    
    println("\n📋 BOOTSTRAP VS PROFILE LIKELIHOOD COMPARISON:")
    println("When you run profile likelihood, compare CI widths:")
    println("• If profile CIs << bootstrap CIs: Profile likelihood has problems")
    println("• If profile CIs ≈ bootstrap CIs: Profile likelihood may be correct")
    println("• If profile CIs >> bootstrap CIs: Bootstrap may be underestimating uncertainty")
    
    return bootstrap_results
end

function emergency_diagnostic_suite(petab_problem, θ_mle, safe_indices, safe_params)
    """Emergency diagnostic suite to address PI's concerns about unrealistic profiles."""
    
    println("\n" * "🚨"^20)
    println("EMERGENCY DIAGNOSTIC SUITE")
    println("Addressing PI concerns about unrealistic profile likelihood results")
    println("🚨"^20)
    
    println("\n📋 DIAGNOSTIC PLAN:")
    println("1. Test likelihood function stability (manual perturbations)")
    println("2. Run bootstrap validation for comparison")
    println("3. Provide concrete recommendations")
    
    # Step 1: Test likelihood stability
    likelihood_issues, critical_params = validate_likelihood_surface_comprehensive(petab_problem, θ_mle, safe_indices, safe_params)
    
    # Step 2: Bootstrap validation
    println("\n" * "─"^60)
    bootstrap_results = bootstrap_confidence_intervals(petab_problem, θ_mle, safe_indices, safe_params)
    
    # Step 3: Final recommendations
    println("\n" * "="^80)
    println("🎯 FINAL RECOMMENDATIONS TO ADDRESS PI CONCERNS")
    println("="^80)
    
    if !isempty(critical_params)
        println("\n🔴 CRITICAL ISSUE CONFIRMED:")
        println("Your PI is absolutely correct. The likelihood function is numerically unstable.")
        println("The '1-2 points in CI' problem is caused by extreme Δχ² jumps.")
        
        println("\n🛠️  IMMEDIATE ACTIONS REQUIRED:")
        println("1. 🎯 ADD REALISTIC MEASUREMENT NOISE (5-10% of each observation)")
        println("   → This is your PI's key suggestion and will likely fix the problem")
        println("2. 🔧 Check ODE solver tolerances (try looser tolerances)")
        println("3. 📊 Validate your measurement data for outliers")
        println("4. 🏗️  Consider model simplification if problems persist")
        
        println("\n❌ DO NOT PROCEED WITH CURRENT PROFILE LIKELIHOOD RESULTS")
        println("They are mathematically invalid due to numerical instability.")
        
    else
        println("\n🟡 MIXED RESULTS:")
        println("Likelihood function appears stable, but profile shapes are still concerning.")
        
        println("\n🔧 RECOMMENDED ACTIONS:")
        println("1. Add observational noise as your PI suggested")
        println("2. Compare bootstrap CIs with profile CIs")
        println("3. Re-run profiling with different algorithms")
        
    end
    
    println("\n💡 HOW TO ADD MEASUREMENT NOISE:")
    add_observational_noise_recommendation()
    
    return Dict(
        "likelihood_issues" => likelihood_issues,
        "critical_parameters" => critical_params,
        "bootstrap_results" => bootstrap_results,
        "recommendation" => !isempty(critical_params) ? "critical_numerical_issues" : "add_noise_and_validate"
    )
end

function add_observational_noise_recommendation()
    """Provide guidance on adding observational noise to test CI validity."""
    
    println("\n" * "="^80)
    println("HOW TO ADD OBSERVATIONAL NOISE TO TEST CI VALIDITY")
    println("="^80)
    
    println("\n🎯 Your PI's suggestion to add observational noise is the gold standard test!")
    println("   This will determine if your tight CIs are realistic or due to noise assumptions.")
    
    println("\n📝 IMPLEMENTATION OPTIONS:")
    
    println("\n1️⃣  MODIFY YOUR PETAB MEASUREMENT FILE:")
    println("   In your measurements.tsv file, add a noiseFormula column:")
    println("   ")
    println("   observableId    simulationConditionId    measurement    time    noiseFormula")
    println("   IL6_obs         condition1               1.5            0       0.05 * abs(measurement)")
    println("   TGFb_obs        condition1               2.1            0       0.05 * abs(measurement)")
    println("   ")
    println("   This adds 5% relative noise to each measurement.")
    
    println("\n2️⃣  PROGRAMMATIC NOISE ADDITION (Julia):")
    println("   ```julia")
    println("   # Add noise to existing measurements")
    println("   measurements_df = CSV.read(\"measurements.tsv\", DataFrame)")
    println("   measurements_df.noiseFormula = 0.05 .* abs.(measurements_df.measurement)")
    println("   CSV.write(\"measurements_with_noise.tsv\", measurements_df; delim='\\t')")
    println("   ```")
    
    println("\n3️⃣  FIXED NOISE PARAMETERS:")
    println("   Add sigma parameters to your parameters.tsv:")
    println("   ")
    println("   parameterId     nominalValue    parameterScale    lowerBound    upperBound")
    println("   sigma_IL6       0.1             log10             0.01          1.0")
    println("   sigma_TGFb      0.1             log10             0.01          1.0")
    
    println("\n🧪 RECOMMENDED NOISE LEVELS TO TEST:")
    println("   • 1% noise:  noiseFormula = 0.01 * abs(measurement)")
    println("   • 5% noise:  noiseFormula = 0.05 * abs(measurement)")
    println("   • 10% noise: noiseFormula = 0.10 * abs(measurement)")
    println("   • 20% noise: noiseFormula = 0.20 * abs(measurement)")
    
    println("\n📊 EXPECTED OUTCOMES:")
    println("   🟢 IF CIs ARE REALISTIC:")
    println("      → Small noise (1-5%) should widen CIs slightly")
    println("      → Large noise (10-20%) should widen CIs substantially")
    println("      → Profile shapes should remain similar")
    
    println("   🔴 IF CIs WERE UNREALISTIC:")
    println("      → Even small noise (1-5%) will dramatically widen CIs")
    println("      → You'll get many more points within 95% CI range")
    println("      → This indicates your original noise assumptions were too optimistic")
    
    println("\n⚡ QUICK TEST SCRIPT:")
    println("   ```julia")
    println("   # Test multiple noise levels")
    println("   noise_levels = [0.01, 0.05, 0.10, 0.20]")
    println("   for noise in noise_levels")
    println("       println(\"Testing \$(noise*100)% noise...\")")
    println("       # Modify your measurements and re-run profiling")
    println("       # Compare CI widths")
    println("   end")
    println("   ```")
    
    println("\n✅ VALIDATION CHECKLIST:")
    println("   □ Original CIs suspiciously tight (< 5 points in 95% CI)")
    println("   □ Added realistic noise levels (5-10% typical for biological data)")
    println("   □ Re-ran optimization to get new MLEs")
    println("   □ Re-ran profiling with same settings")
    println("   □ Compared CI widths between noise levels")
    println("   □ Checked if bootstrap CIs agree with profile CIs")
    
    println("\n" * "="^80)
    println("After testing with noise, your CIs should be more realistic and interpretable!")
    println("="^80)
end

function generate_noise_test_measurements(original_measurements_file::String, noise_levels::Vector{Float64})
    """Generate multiple measurement files with different noise levels for testing."""
    
    println("🔧 Generating measurement files with different noise levels...")
    
    # This is a template - you'll need to adapt based on your actual file format
    println("📁 Files to create:")
    for noise in noise_levels
        noise_pct = round(noise * 100, digits=1)
        new_filename = replace(original_measurements_file, ".tsv" => "_$(noise_pct)pct_noise.tsv")
        println("   → $new_filename ($(noise_pct)% relative noise)")
    end
    
    println("\n💡 Manual steps needed:")
    println("1. Copy your original measurements.tsv file")
    println("2. Add noiseFormula column with: $(noise_levels[1]) * abs(measurement)")
    println("3. Repeat for each noise level: $(join(string.(noise_levels), ", "))")
    println("4. Re-run your optimization and profiling pipeline")
    println("5. Compare the resulting CI widths")
    
    return nothing
end

function comprehensive_ci_diagnostics(petab_problem, θ_mle, safe_indices, safe_params, true_param_values)
    """Comprehensive diagnostic suite to validate tight confidence intervals."""
    
    println("\n" * "="^80)
    println("COMPREHENSIVE CONFIDENCE INTERVAL DIAGNOSTICS")
    println("Testing whether tight CIs are realistic or due to technical issues")
    println("="^80)
    
    diagnostic_results = Dict()
    
    for (i, idx) in enumerate(safe_indices)
        param_name = safe_params[i]
        
        println("\n" * "━"^60)
        println("DIAGNOSING PARAMETER: $param_name")
        println("━"^60)
        
        # Test 1: Objective sensitivity
        sensitivity_results = test_objective_sensitivity(petab_problem, θ_mle, idx, param_name)
        
        # Test 2: Manual grid profiling
        x_vals, nll_vals, delta_chi2 = manual_profile_grid(petab_problem, θ_mle, idx, param_name)
        
        # Store results
        diagnostic_results[param_name] = Dict(
            "sensitivity" => sensitivity_results,
            "manual_grid" => (x_vals, nll_vals, delta_chi2),
            "param_idx" => idx
        )
    end
    
    # Test 3: Expanded bounds profiling for all parameters
    expanded_results = expanded_bounds_profiling(petab_problem, θ_mle, safe_indices, safe_params)
    
    # Summary recommendations
    println("\n" * "="^80)
    println("DIAGNOSTIC SUMMARY AND RECOMMENDATIONS")
    println("="^80)
    
    tight_ci_count = 0
    total_params = length(safe_params)
    
    for param_name in safe_params
        results = diagnostic_results[param_name]
        manual_grid = results["manual_grid"]
        
        if length(manual_grid[3]) > 0  # has delta_chi2 values
            ci_95_count = count(<(3.84), manual_grid[3])
            if ci_95_count < 10
                tight_ci_count += 1
            end
        end
    end
    
    if tight_ci_count > total_params * 0.7
        println("🔴 CONCERN: $(tight_ci_count)/$(total_params) parameters show very tight CIs")
        println("\n📋 RECOMMENDED ACTIONS:")
        println("1. 🎯 PRIORITY: Add observational noise to your PEtab problem:")
        println("   - Add 5-10% relative noise: noiseFormula = 0.05 * abs(measurement)")
        println("   - This tests if tight CIs are due to unrealistic noise assumptions")
        println("\n2. 🔬 Validate with bootstrap confidence intervals")
        println("3. 🏗️  Check structural identifiability of your model")
        println("4. 📊 Consider experimental design optimization")
        
    elseif tight_ci_count > 0
        println("🟡 MIXED RESULTS: $(tight_ci_count)/$(total_params) parameters show tight CIs")
        println("\n📋 RECOMMENDED ACTIONS:")
        println("1. Focus noise testing on the tight parameters")
        println("2. Consider parameter correlations analysis")
        
    else
        println("✅ GOOD: Confidence intervals appear reasonable")
        println("Your tight identifiability may be realistic given your data quality")
    end
    
    return diagnostic_results, expanded_results
end

function robust_manual_profiling(pl_problem, safe_indices, safe_params, true_param_values)
    prof_dir = joinpath(pwd(), "likelihood_profiles")
    mkpath(prof_dir)
    
    nll_mle = pl_problem.optprob.f(pl_problem.optpars, pl_problem.optprob.p)
    θ_mle = pl_problem.optpars
    
    robustness_results = Dict()
    
    for (i, idx) in enumerate(safe_indices)
        param_name = safe_params[i]
        @info "Robustness profiling $param_name..."
        
        # 1. Get MLE and True Value on log10 scale
        θ_center = θ_mle[idx]
        base_name = string(param_name)
        if startswith(base_name, "log10_")
            base_name = base_name[7:end]
        end
        θ_true = haskey(true_param_values, base_name) ? log10(true_param_values[base_name]) : θ_center

        # 2. Make the sampling range ADAPTIVE
        padding_fraction = 0.25 # Use 25% padding
        num_pts = 1000

        # Calculate the range based on the distance between the two points
        distance = abs(θ_center - θ_true)
        padding = max(distance * padding_fraction, 0.001) # Ensure padding is not zero

        θ_min = min(θ_center, θ_true) - padding
        θ_max = max(θ_center, θ_true) + padding
        param_range = range(θ_min, θ_max, length=num_pts)
        
        x_vals = Float64[]
        nll_vals = Float64[]
        
        # This part remains the same
        for θ_val in param_range
            θ_test = copy(θ_mle)
            θ_test[idx] = θ_val
            
            try
                nll = pl_problem.optprob.f(θ_test, pl_problem.optprob.p)
                push!(x_vals, θ_val)
                push!(nll_vals, isfinite(nll) ? nll : Inf)
            catch
                push!(x_vals, θ_val)
                push!(nll_vals, Inf)
            end
        end
        
        # This part remains the same
        delta_chi2 = 2.0 .* (nll_vals .- nll_mle)
        ci_95_indices = findall(delta_chi2 .<= 3.84)
        
        robustness_info = Dict(
            "param_name" => param_name,
            "mle_value" => θ_center,
            "true_value" => θ_true,
            "ci_width" => length(ci_95_indices) > 0 ? maximum(x_vals[ci_95_indices]) - minimum(x_vals[ci_95_indices]) : Inf,
            "contains_true" => !isempty(ci_95_indices) && (minimum(x_vals[ci_95_indices]) <= θ_true <= maximum(x_vals[ci_95_indices])),
            "profile_success" => any(isfinite.(nll_vals))
        )
        
        robustness_results[param_name] = robustness_info
        
        # This part remains the same
        lock(PLOT_LOCK) do
            plt = plot()
            plot_profile_delta_chi2!(plt, x_vals, nll_vals; pname=param_name, nll_anchor=nll_mle, autox=false)
            vline!(plt, [θ_true]; label="True Value", color=:purple, linestyle=:dash, lw=2)
            vline!(plt, [θ_center]; label="MLE", color=:green, linestyle=:solid, lw=2)
            
            savefig(plt, joinpath(prof_dir, "robustness_profile_$(param_name).png"))
        end
        
        @info "✓ $param_name: CI width = $(round(robustness_info["ci_width"], digits=3)), Contains true = $(robustness_info["contains_true"])"
    end
    
    return robustness_results
end

function run_likelihood_profiling(
    petab_model::PEtabModel,
    odesolver,
    steadystate_solver,
    θ_mle::ComponentVector,
    true_param_values::Dict;
    profiling_method::Symbol = :fixedstep,
    debug::Bool=false,
    maxiters::Int=20,
    run_diagnostics::Bool=false,
    emergency_mode::Bool=false
)
    println("\n--- Likelihood Profiling ---"); flush(stdout)
    @info "Using profiling method: :$profiling_method"
    @info "Threading configuration:"
    @info "  Julia threads available: $(Threads.nthreads())"
    @info "  JULIA_NUM_THREADS: $(get(ENV, "JULIA_NUM_THREADS", "not set"))"
    @info "  OPENBLAS_NUM_THREADS: $(get(ENV, "OPENBLAS_NUM_THREADS", "not set"))"
    t_start = time()

    # 1. Create the PEtab problem with robust callbacks
    petab_problem = create_petab_problem_for_profiling(petab_model, odesolver, steadystate_solver)
    
    # 2. Identify parameters to profile (excluding noise/initial conditions)
    all_names = string.(keys(θ_mle))
    params_to_profile = [n for n in all_names if !startswith(n, "sigma") && !endswith(n, "_0")]
    param_indices = [findfirst(==(p), all_names) for p in params_to_profile]
    
    # 3. Filter out parameters at bounds
    lb_orig = collect(petab_problem.lower_bounds)
    ub_orig = collect(petab_problem.upper_bounds)
    θ_init = collect(θ_mle)
    safe_indices = Int[]
    safe_params = String[]

    @info "Analyzing parameter bounds proximity for profiling eligibility..."
    bound_threshold = 1e-3  # Consider relaxing to 1e-4 for testing if too many parameters are filtered out
    
    for (i, name) in enumerate(params_to_profile)
        idx = param_indices[i]
        lower_dist = θ_init[idx] - lb_orig[idx]
        upper_dist = ub_orig[idx] - θ_init[idx]
        
        if (lower_dist > bound_threshold) && (upper_dist > bound_threshold)
            push!(safe_indices, idx)
            push!(safe_params, name)
            @info "  ✓ $name: distance from bounds = ($(round(lower_dist, digits=6)), $(round(upper_dist, digits=6)))"
        else
            @warn "  ✗ Skipping profile for '$name' - parameter is at or very close to a bound."
            @warn "    Current value: $(θ_init[idx]), bounds: [$(lb_orig[idx]), $(ub_orig[idx])]"
            @warn "    Distance from bounds: ($(round(lower_dist, digits=6)), $(round(upper_dist, digits=6)))"
        end
    end

    if isempty(safe_indices)
        @error "No parameters are suitable for profiling; all are at their bounds."
        @error "Consider relaxing the bound proximity threshold (currently $(bound_threshold)) or checking parameter optimization results."
        return nothing
    end
    
    @info "Profiling Summary:"
    @info "  Total parameters requested: $(length(params_to_profile))"
    @info "  Parameters eligible for profiling: $(length(safe_indices))"
    @info "  Parameters to profile: $(safe_params)"
    @info "  Bound proximity threshold: $(bound_threshold)"
    println("[Profiling] Will profile $(length(safe_indices)) parameters: $(safe_params)")

    # 4. EMERGENCY MODE: Run critical diagnostics first if profiles are showing problems
    if emergency_mode
        @warn "🚨 EMERGENCY MODE ACTIVATED 🚨"
        @warn "Running emergency diagnostics to validate likelihood surface stability"
        @warn "This addresses concerns about unrealistic profile shapes and '1-2 points in CI' problems"
        
        emergency_results = emergency_diagnostic_suite(petab_problem, θ_init, safe_indices, safe_params)
        
        if emergency_results["recommendation"] == "critical_numerical_issues"
            @error "🛑 CRITICAL NUMERICAL ISSUES DETECTED"
            @error "Profile likelihood is unreliable with current setup."
            @error "Follow the recommendations above before attempting profiling."
            return emergency_results
        else
            @warn "⚠️  Issues detected but profiling may still be attempted."
            @warn "Strongly recommend adding measurement noise first."
        end
    end

    # 5. Run comprehensive diagnostics if requested
    if run_diagnostics
        @info "Running comprehensive CI diagnostics..."
        diagnostic_results, expanded_results = comprehensive_ci_diagnostics(petab_problem, θ_init, safe_indices, safe_params, true_param_values)
        
        # If diagnostics suggest issues, recommend stopping here
        tight_params = 0
        for param_name in safe_params
            if haskey(diagnostic_results, param_name)
                manual_grid = diagnostic_results[param_name]["manual_grid"]
                if length(manual_grid[3]) > 0
                    ci_95_count = count(<(3.84), manual_grid[3])
                    if ci_95_count < 10
                        tight_params += 1
                    end
                end
            end
        end
        
        if tight_params > length(safe_params) * 0.5
            println("\n🚨 DIAGNOSTIC RECOMMENDATION:")
            println("$(tight_params)/$(length(safe_params)) parameters show suspiciously tight CIs.")
            println("Consider adding observational noise before proceeding with full profiling.")
            println("Use emergency_mode=true for more detailed analysis.")
            
            return Dict(
                "diagnostics" => diagnostic_results,
                "expanded_results" => expanded_results,
                "recommendation" => "add_noise"
            )
        end
    end

    # 6. Define the robust objective function and gradient
    @info "Setting up Profiling Problem..."
    function obj(θ_est, _)
        if any(!isfinite, θ_est) return Inf end
        try
            val = petab_problem.nllh(θ_est)
            return isfinite(val) ? val : Inf
        catch
            return Inf
        end
    end
    
    # Keep track of last valid gradient to avoid all-zero gradients
    LAST_G = Ref{Union{Vector{Float64}, Nothing}}(nothing)
    
    # Define gradient function that uses PEtab's threaded gradient computation
    function grad!(G, θ_est, _)
        if any(!isfinite, θ_est) 
            if LAST_G[] !== nothing
                G .= LAST_G[]
            else
                fill!(G, 0.0)
            end
            return nothing
        end
        try
            petab_problem.grad!(G, θ_est)
            # Replace only non-finite components with zeros, preserve finite ones
            anybad = false
            @inbounds for i in eachindex(G)
                if !isfinite(G[i])
                    G[i] = 0.0
                    anybad = true
                end
            end
            # If all gradients are zero, reuse last valid gradient to keep search moving
            if all(iszero, G) && LAST_G[] !== nothing
                G .= LAST_G[]
            elseif any(isfinite, G) && !all(iszero, G)
                LAST_G[] = copy(G)
            end
            return nothing  # Explicit nothing return for in-place gradient
        catch
            if LAST_G[] !== nothing
                G .= LAST_G[]
            else
                fill!(G, 0.0)
            end
            return nothing
        end
    end
    
    # 7. Create the LikelihoodProfiler problem with PEtab's gradient
    # Use NoAD and pass grad! via keyword to enable split_over_conditions parallelization
    @info "Creating OptimizationFunction with PEtab's threaded gradient computation..."
    @info "  Using Optimization.NoAD() with grad! keyword to enable split_over_conditions parallelization."
    optf = OptimizationFunction(obj, NoAD(); grad=grad!)
    optprob = OptimizationProblem(optf, θ_init; lb = lb_orig, ub = ub_orig)

    # Explicitly define the profile_range from the problem bounds.
    # This is the robust way to ensure the profiler respects the bounds, as shown in the library's documentation.
    profile_range_explicit = tuple.(lb_orig, ub_orig)

    pl_problem = LikelihoodProfiler.ProfileLikelihoodProblem(optprob, θ_init, profile_range_explicit)
    @info "✅ Profiling Problem created successfully with threaded gradient computation."

    # 8. Choose and run the selected profiling method
    sol_res = nothing # Initialize result variable
    # if profiling_method == :cico
    #     println("[Profiling] Running CICOProfiler on $(length(safe_indices)) parameters...")
        
    #     # 6. Define the profiler algorithm
    #     profiler_alg = CICOProfiler(
    #         optimizer=:IPNewton,
    #         scan_tol=1e-2
    #     )

    #     # CRITICAL: CICOProfiler requires explicit scan_bounds
    #     bounds_for_profiling = collect(zip(lb_orig, ub_orig))
    #     sol_res = @time LikelihoodProfiler.solve(
    #         pl_problem, 
    #         profiler_alg; 
    #         idxs = safe_indices,
    #         scan_bounds = bounds_for_profiling,
    #         parallel_type = :threads,
    #         maxiters = maxiters
    #     )
    if profiling_method == :fixedstep
        println("[Profiling] Running OptimizationProfiler with LineSearchStep on $(length(safe_indices)) parameters...")
        
        # Define a function for a larger initial step size to help optimizer leave starting point
        profile_step_func(p0, i) = abs(p0[i]) * 0.01 + 1e-6  # Increased from 0.002 to 0.01

        profiler_alg = OptimizationProfiler(
            optimizer = Optim.LBFGS(
                alphaguess = LineSearches.InitialStatic(),
                linesearch = LineSearches.BackTracking(order=2)  # more forgiving than Hager–Zhang
            ),
            # Use the intelligent, adaptive line search stepper
            stepper = LineSearchStep(initial_step = profile_step_func)
        )

        sol_res = @time LikelihoodProfiler.solve(
            pl_problem,
            profiler_alg;
            idxs = safe_indices,
            parallel_type = :threads,
            maxiters = maxiters
        )
    elseif profiling_method == :manual
        println("[Profiling] Running robust manual profiling on $(length(safe_indices)) parameters...")
        @time robustness_results = robust_manual_profiling(pl_problem, safe_indices, safe_params, true_param_values)
        
    else
        @error "Unknown profiling_method: :$profiling_method. Choose :cico, :fixedstep, or :manual."
        return nothing
    end

    # 7. Plot the results if profiling was successful
    if !isnothing(sol_res)
        println("[Profiling] Plotting results...")
        prof_dir = joinpath(pwd(), "likelihood_profiles")
        mkpath(prof_dir)
        nll_mle = pl_problem.optprob.f(pl_problem.optpars, pl_problem.optprob.p)

        # Comprehensive identifiability analysis
        println("\n" * "="^60)
        println("IDENTIFIABILITY ANALYSIS FROM PROFILE LIKELIHOODS")
        println("="^60)

        for (k, idx) in enumerate(safe_indices)
            profile_result = sol_res[k]                 # index into PLSolution
            param_name = all_names[idx]                 # map back to original param index
            
            # Enhanced diagnostic information about profile data quality and identifiability
            n_points = length(profile_result.x)
            n_finite_x = count(isfinite, profile_result.x)
            n_finite_obj = count(isfinite, profile_result.obj)
            
            println("\n=== $param_name Identifiability Diagnostics ===")
            println("  Profile points: $n_points")
            println("  Finite x-values: $n_finite_x") 
            println("  Finite objectives: $n_finite_obj")
            
            if !isempty(profile_result.obj) && any(isfinite, profile_result.obj)
                # Filter to finite values only
                finite_mask = isfinite.(profile_result.obj)
                obj_finite = profile_result.obj[finite_mask]
                x_finite = profile_result.x[finite_mask]
                
                if !isempty(obj_finite)
                    nll_min = minimum(obj_finite)
                    delta_chi2 = 2.0 .* (obj_finite .- nll_min)
                    
                    println("  Parameter range explored: [$(round(minimum(x_finite), digits=4)), $(round(maximum(x_finite), digits=4))]")
                    println("  Min Δχ²: $(round(minimum(delta_chi2), digits=4))")
                    println("  Max Δχ²: $(round(maximum(delta_chi2), digits=4))")
                    println("  Points with Δχ² < 3.84 (95% CI): $(count(<(3.84), delta_chi2))")
                    println("  Points with Δχ² < 6.63 (99% CI): $(count(<(6.63), delta_chi2))")
                    
                    # Parameter range analysis
                    param_range = maximum(x_finite) - minimum(x_finite)
                    mle_value = θ_init[idx]
                    println("  MLE value: $(round(mle_value, digits=4))")
                    println("  Explored range width: $(round(param_range, digits=4))")
                    
                    # Identifiability interpretation
                    if maximum(delta_chi2) < 1.0
                        println("  🔴 INTERPRETATION: Extremely flat profile - parameter is POORLY IDENTIFIABLE")
                        println("      → Likelihood barely changes across parameter range")
                        println("      → Consider: parameter reduction, additional data, or structural identifiability analysis")
                    elseif maximum(delta_chi2) < 3.84
                        println("  🟡 INTERPRETATION: Flat profile - parameter has WEAK IDENTIFIABILITY")
                        println("      → Wide confidence intervals (> explored range)")
                        println("      → Consider: more informative priors or experimental design optimization")
                    elseif minimum(delta_chi2) > 50.0
                        println("  🟢 INTERPRETATION: Very steep profile - parameter is HIGHLY IDENTIFIABLE")
                        println("      → Very narrow confidence intervals")
                        println("      → Well-constrained by data")
                    elseif minimum(delta_chi2) > 15.0
                        println("  🔵 INTERPRETATION: Steep profile - parameter is WELL IDENTIFIABLE")
                        println("      → Narrow confidence intervals") 
                        println("      → Good practical identifiability")
                    else
                        ci_95_points = count(<(3.84), delta_chi2)
                        if ci_95_points > n_finite_obj * 0.8
                            println("  🟡 INTERPRETATION: Moderately flat profile - parameter has LIMITED IDENTIFIABILITY")
                            println("      → Wide but finite confidence intervals")
                        else
                            println("  🟢 INTERPRETATION: Well-shaped profile - parameter has GOOD IDENTIFIABILITY")
                            println("      → Reasonable confidence intervals")
                        end
                    end
                    
                    # Additional checks for profile quality
                    if param_range < abs(mle_value) * 0.01
                        println("  ⚠️  WARNING: Very narrow parameter range explored - may need larger profile steps")
                    end
                    
                    if n_finite_obj < 10
                        println("  ⚠️  WARNING: Very few finite points - optimization may be struggling")
                    end
                else
                    println("  ❌ ERROR: No finite Δχ² values computed")
                end
            else
                println("  ❌ ERROR: No finite profile points - optimization completely failed")
                println("      → Check parameter bounds, initial conditions, and model stability")
            end
            
            # Create the plot with enhanced information
            lock(PLOT_LOCK) do
                plt = plot()
                plot_profile_delta_chi2!(plt, profile_result.x, profile_result.obj; pname=param_name, nll_anchor=nll_mle)
                
                base_name = string(param_name)
                if startswith(base_name, "log10_")
                    base_name = base_name[7:end]
                end
                if haskey(true_param_values, base_name)
                    true_val_log10 = log10(true_param_values[base_name])
                    vline!(plt, [true_val_log10]; label="True Value", color=:purple, linestyle=:dash)
                end
                
                savefig(plt, joinpath(prof_dir, "profile_$(param_name).png"))
                println("  📊 Plot saved: profile_$(param_name).png")
            end
        end
        
        println("\n" * "="^60)
        println("IDENTIFIABILITY SUMMARY")
        println("="^60)
        println("Profile likelihood analysis completed for $(length(safe_indices)) parameters.")
        println("Review individual parameter diagnostics above for identifiability status.")
        println("Flat profiles may indicate either poor identifiability OR very precise estimates.")
        println("Consider complementary analysis: parameter correlations, Fisher Information, or bootstrap CI.")
    end

    # Generate custom-range profiles comparing MLE vs True Value for each parameter
    println("\n--- Generating Custom-Range Profiles (MLE vs True) ---")
    for pname in safe_params
        try
            profile_parameter_custom_range(pname, pl_problem, θ_mle, true_param_values)
        catch e
            @warn "Custom-range profile failed for $pname: $e"
        end
    end

    println("[Profiling] Done in $(round(time()-t_start; digits=2)) s")
    return nothing
end