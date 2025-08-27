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

# Thread-safe plotting lock to prevent font loading race conditions
const PLOT_LOCK = ReentrantLock()

function LikelihoodProfiler.interpolate_endpoint(profile_values::LikelihoodProfiler.ProfileValues)
    obj_level = LikelihoodProfiler.get_obj_level(profile_values)
    
    if length(profile_values.x) < 2
        return profile_values.x[end] # Not enough points to interpolate
    end

    x_last = profile_values.x[end]
    x_penultimate = profile_values.x[end-1]
    
    obj_last = profile_values.obj[end]
    obj_penultimate = profile_values.obj[end-1]

    # Check if the last two points bracket the confidence threshold.
    # This is the normal, identifiable case.
    if (obj_penultimate <= obj_level <= obj_last) || (obj_last <= obj_level <= obj_penultimate)
        
        # Perform linear interpolation manually: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
        # Here, we solve for x: x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        # x = parameter value, y = objective value
        
        # Avoid division by zero if the objective values are identical
        if abs(obj_last - obj_penultimate) < 1e-9
            return x_last
        end

        # Calculate the interpolated parameter value at the threshold
        interpolated_x = x_penultimate + (obj_level - obj_penultimate) * (x_last - x_penultimate) / (obj_last - obj_penultimate)
        
        return interpolated_x
    else
        # This is the edge case: the profile hit a bound and is non-identifiable.
        # The endpoint is simply the last parameter value calculated at the boundary.
        return x_last
    end
end

function create_petab_problem_for_profiling(petab_model::PEtabModel, odesolver, steadystate_solver=nothing)
    @info "Creating PEtabODEProblem for profiling with PositiveDomain callback..."
    
    # Define the callback you want to add
    positive_domain_cb = PositiveDomain()
    
    # Combine it with any existing callbacks in the model
    combined_callbacks = CallbackSet(petab_model.callbacks, positive_domain_cb)

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
        combined_callbacks,
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
    nll_anchor::Float64,
    ymax::Float64=15.0,
    show_99::Bool=true,
    autox::Bool=true
)
    delta_chi2 = 2.0 .* (nll .- nll_anchor)
    plot!(plt, x, delta_chi2; lw=2, label=nothing)
    ylabel!(plt, "Δχ²")
    xlabel!(plt, pname)
    title!(plt, "Likelihood profile: $(pname)")
    hline!(plt, [3.84]; lc=:orange, ls=:dash, label="95%")
    if show_99
        hline!(plt, [6.63]; lc=:red, ls=:dashdot, label="99%")
    end
    ylims!(plt, (0.0, ymax))
    if autox
        idx = findall(delta_chi2 .<= ymax)
        if !isempty(idx)
            xlo, xhi = minimum(x[idx]), maximum(x[idx])
            xpad = 0.02 * max(abs(xhi - xlo), eps())
            xlims!(plt, (xlo - xpad, xhi + xpad))
        end
    end
    return plt
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
        
        # Adaptive grid around MLE + true value
        θ_center = θ_mle[idx]
        
        # Get true value, handling log10_ prefixes correctly
        base_name = string(param_name)
        if startswith(base_name, "log10_")
            base_name = base_name[7:end]
        end
        θ_true = haskey(true_param_values, base_name) ? log10(true_param_values[base_name]) : θ_center
        
        # Grid covering both MLE and true value with margin
        θ_min = min(θ_center, θ_true) - 2.0
        θ_max = max(θ_center, θ_true) + 2.0
        param_range = range(θ_min, θ_max, length=200)
        
        x_vals = Float64[]
        nll_vals = Float64[]
        
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
        
        # Calculate robustness metrics
        delta_chi2 = 2.0 .* (nll_vals .- nll_mle)
        ci_95_indices = findall(delta_chi2 .<= 3.84)
        
        robustness_info = Dict(
            "param_name" => param_name,
            "mle_value" => θ_center,
            "true_value" => θ_true,
            "ci_width" => length(ci_95_indices) > 0 ? maximum(x_vals[ci_95_indices]) - minimum(x_vals[ci_95_indices]) : Inf,
            "contains_true" => θ_true in x_vals[ci_95_indices],
            "profile_success" => any(isfinite.(nll_vals))
        )
        
        robustness_results[param_name] = robustness_info
        
        # Plot results with thread-safe plotting
        lock(PLOT_LOCK) do
            plt = plot()
            plot_profile_delta_chi2!(plt, x_vals, nll_vals; pname=param_name, nll_anchor=nll_mle)
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
    maxiters::Int=200
)
    println("\n--- Likelihood Profiling ---"); flush(stdout)
    @info "Using profiling method: :$profiling_method"
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

    for (i, name) in enumerate(params_to_profile)
        idx = param_indices[i]
        if (θ_init[idx] - lb_orig[idx] > 1e-3) && (ub_orig[idx] - θ_init[idx] > 1e-3)
            push!(safe_indices, idx)
            push!(safe_params, name)
        else
            @warn "Skipping profile for '$name' - parameter is at or very close to a bound."
        end
    end

    if isempty(safe_indices)
        @error "No parameters are suitable for profiling; all are at their bounds."
        return nothing
    end
    println("[Profiling] Will profile $(length(safe_indices)) parameters: $(safe_params)")

    # 4. Define the robust objective function
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
    
    # 5. Create the LikelihoodProfiler problem
    optf = OptimizationFunction(obj, Optimization.AutoForwardDiff())
    optprob = OptimizationProblem(optf, θ_init; lb = lb_orig, ub = ub_orig)

    # Explicitly define the profile_range from the problem bounds.
    # This is the robust way to ensure the profiler respects the bounds, as shown in the library's documentation.
    profile_range_explicit = tuple.(lb_orig, ub_orig)

    pl_problem = LikelihoodProfiler.ProfileLikelihoodProblem(optprob, θ_init, profile_range_explicit)
    @info "✅ Profiling Problem created successfully."

    # 6. Choose and run the selected profiling method
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
        
        # Define a function for a robust initial step size (1% of parameter magnitude)
        profile_step_func(p0, i) = abs(p0[i]) * 0.01 + 1e-8

        profiler_alg = OptimizationProfiler(
            optimizer = Optim.LBFGS(), # A robust, first-order optimizer
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

        for (i, profile_result) in enumerate(sol_res)
            param_name = all_names[safe_indices[i]]
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
                println("[Profiling] Saved plot for $(param_name)")
            end
        end
    end

    println("[Profiling] Done in $(round(time()-t_start; digits=2)) s")
    return nothing
end