using LikelihoodProfiler
using LikelihoodProfiler: AdaptiveStep, FixedStep
using Optimization
using OptimizationOptimJL
using ComponentArrays
using Plots
using PEtab
using ForwardDiff
using DiffEqCallbacks

function create_petab_problem_for_profiling(petab_model, odesolver, steadystate_solver)
    @info "Creating PEtabODEProblem for profiling with PositiveDomain callback..."
    
    positive_domain_cb = PositiveDomain()
    combined_callbacks = CallbackSet(petab_model.callbacks, positive_domain_cb)

    petab_problem = PEtabODEProblem(
        petab_model,
        odesolver=odesolver,
        ss_solver=steadystate_solver,
        # Use the dedicated keyword argument for callbacks
        callback=combined_callbacks, 
        verbose=false
    )
    
    @info "✅ PEtabODEProblem for profiling created successfully."
    return petab_problem
end

function plot_profile_delta_chi2!(
    plt,
    x::AbstractVector,
    nll::AbstractVector;
    pname::AbstractString,
    nll_anchor::Float64,
    ymax::Float64=4.0,
    show_99::Bool=true,
    autox::Bool=true
)
    Δχ2 = 2.0 .* (nll .- nll_anchor)
    plot!(plt, x, Δχ2; lw=2, label=nothing)
    ylabel!(plt, "Δχ²")
    xlabel!(plt, pname)
    title!(plt, "Likelihood profile: $(pname)")
    hline!(plt, [3.84]; lc=:orange, ls=:dash, label="95%")
    if show_99
        hline!(plt, [6.63]; lc=:red, ls=:dashdot, label="99%")
    end
    ylims!(plt, (0.0, ymax))
    if autox
        idx = findall(Δχ2 .<= ymax)
        if !isempty(idx)
            xlo, xhi = minimum(x[idx]), maximum(x[idx])
            xpad = 0.02 * max(abs(xhi - xlo), eps())
            xlims!(plt, (xlo - xpad, xhi + xpad))
        end
    end
    return plt
end

"""
run_likelihood_profiling(petab_model, odesolver, steadystate_solver, θ_mle; debug=false, maxiters=20)

Idiomatic likelihood profiling using integrated PEtab.jl and LikelihoodProfiler.jl helpers:
  - Uses library-provided objective function (no manual AD handling)
  - Leverages get_pl_problem for robust setup
  - Saves Δχ² plots with confidence intervals
Returns profile results object.
"""
function run_likelihood_profiling(
    petab_model::PEtabModel,
    odesolver,
    steadystate_solver,
    θ_mle::ComponentVector,
    true_param_values::Dict;
    debug::Bool=false,
    maxiters::Int=20
)
    println("\n--- Likelihood Profiling (Idiomatic Version) ---"); flush(stdout)
    t_start = time()

    # 1. Create the PEtabProblem with the PositiveDomain callback
    petab_problem = create_petab_problem_for_profiling(petab_model, odesolver, steadystate_solver)
    
    # 2. Select parameters to profile (your logic here is great, no changes needed)
    all_names = string.(keys(θ_mle))
    params_to_profile = [n for n in all_names if !startswith(n, "sigma") && !endswith(n, "_0")]
    if isempty(params_to_profile)
        @warn "No dynamic parameters found for profiling, selecting from all parameters."
        params_to_profile = all_names
    end
    # Get the indices of the parameters to profile
    param_indices = [findfirst(==(p), all_names) for p in params_to_profile]
    println("[Profiling] Parameters to profile: $(params_to_profile)")

    # 3. Use the integrated helper to create the LikelihoodProfiler problem
    #    This automatically handles the objective function, bounds, and AD.
    #    No need for a manual `obj` function!
    @info "Setting up LikelihoodProfiler Problem..."
    pl_problem = LikelihoodProfiler.get_pl_problem(petab_problem, θ_mle)

    # 4. Define the profiler algorithm (your setup is good)
    profiler_alg = LikelihoodProfiler.OptimizationProfiler(
        optimizer = OptimizationOptimJL.IPNewton(),
        stepper = LikelihoodProfiler.FixedStep(; initial_step = 0.005)
    )

    # 5. Run the profiling
    println("[Profiling] Running threaded profiling on $(length(param_indices)) parameters...")
    @time profile_res = LikelihoodProfiler.profile(
        pl_problem, 
        profiler_alg; 
        idxs = param_indices,
        parallel_type = :threads,
        maxiters=maxiters # Use the function argument
    )

    # 6. Plot the results
    println("[Profiling] Plotting results...")
    prof_dir = joinpath(pwd(), "likelihood_profiles")
    mkpath(prof_dir)
    
    # The best NLL is stored in the results object
    nll_mle = profile_res.maximum_log_likelihood

    for (param_name, profile_result) in profile_res.profiles
        plt = plot()
        
        # The result object has clean, documented fields
        x_vals = [p[1] for p in profile_result.points]
        nll_vals = [p[2] for p in profile_result.points]
        
        plot_profile_delta_chi2!(plt, x_vals, nll_vals; pname=string(param_name), nll_anchor=nll_mle)
        
        # Add true value if it exists
        if haskey(true_param_values, string(param_name))
            vline!(plt, [true_param_values[string(param_name)]]; label="True Value", color=:purple, linestyle=:dash)
        end
        
        savefig(plt, joinpath(prof_dir, "profile_$(param_name).png"))
        println("[Profiling] Saved plot for $(param_name)")
    end

    println("[Profiling] Done in $(round(time()-t_start; digits=2)) s")
    return profile_res
end