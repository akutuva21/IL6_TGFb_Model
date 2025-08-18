using CICOBase
using LikelihoodProfiler
using LikelihoodProfiler: AdaptiveStep, FixedStep
using Optimization
using OptimizationOptimJL
using ComponentArrays
using Plots
using PEtab
using ForwardDiff
using DiffEqCallbacks
using DataInterpolations

function LikelihoodProfiler.interpolate_endpoint(profile_values::LikelihoodProfiler.ProfileValues)
    @info "--- EXECUTING ROBUST INTERPOLATION ---"

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

function create_petab_problem_for_profiling(petab_model::PEtabModel, odesolver, steadystate_solver)
    @info "Creating PEtabODEProblem for profiling with PositiveDomain callback..."
    
    # 1. Define the callback you want to add
    positive_domain_cb = PositiveDomain()
    
    # 2. Combine it with any existing callbacks in the model
    combined_callbacks = CallbackSet(petab_model.callbacks, positive_domain_cb)

    # 3. Manually create a *new* PEtabModel that is a copy of the original,
    #    but with our new, combined callback set. This is the key step.
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
        combined_callbacks, # <<< Here we insert the new callbacks
        petab_model.defined_in_julia
    )

    # 4. Create the final PEtabODEProblem from this new, modified PEtabModel.
    petab_problem = PEtabODEProblem(
        petab_model_with_callback,
        odesolver=odesolver,
        ss_solver=steadystate_solver,
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
    ymax::Float64=15.0,
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
    println("\n--- Likelihood Profiling (Corrected Order) ---"); flush(stdout)
    t_start = time()

    # Steps 1-3: Setup is all correct
    petab_problem = create_petab_problem_for_profiling(petab_model, odesolver, steadystate_solver)
    
    all_names = string.(keys(θ_mle))
    params_to_profile = [n for n in all_names if !startswith(n, "sigma") && !endswith(n, "_0")]
    param_indices = [findfirst(==(p), all_names) for p in params_to_profile]
    println("[Profiling] Parameters to profile: $(params_to_profile)")

    @info "Setting up Profiling Problem Manually..."
    function obj(θ_est, _)
        θ_work = eltype(θ_est) <: ForwardDiff.Dual ? ForwardDiff.value.(θ_est) : θ_est
        if any(!isfinite, θ_work) return Inf end
        try
            val = petab_problem.nllh(θ_work)
            return isfinite(val) ? val : Inf
        catch
            return Inf
        end
    end
    θ_init = collect(θ_mle)
    optf = OptimizationFunction(obj, Optimization.AutoForwardDiff())
    optprob = OptimizationProblem(optf, θ_init; lb=collect(petab_problem.lower_bounds), ub=collect(petab_problem.upper_bounds))
    pl_problem = LikelihoodProfiler.PLProblem(optprob, θ_init)
    @info "✅ Profiling Problem created successfully."

    # 4. Define the profiler algorithm
    # CICOProfiler is more robust for difficult landscapes as it directly seeks the endpoints.
    profiler_alg = CICOProfiler(scan_tol = 1e-2)

    # 5. Run the profiling
    println("[Profiling] Running threaded profiling on $(length(param_indices)) parameters...")
    @time profile_res = LikelihoodProfiler.profile(
        pl_problem, 
        profiler_alg; 
        idxs = param_indices,
        parallel_type = :threads,
        maxiters=maxiters
    )

    # 6. Plot the results SECOND, now that `profile_res` exists
    println("[Profiling] Plotting results...")
    prof_dir = joinpath(pwd(), "likelihood_profiles")
    mkpath(prof_dir)
    
    nll_mle = pl_problem.optprob.f(pl_problem.optpars, pl_problem.optprob.p)

    # We iterate from 1 to the number of profiles calculated
    for i in 1:length(profile_res)
        profile_result = profile_res[i]
        
        # Get the parameter name using the same index from our list
        param_name = params_to_profile[i]
        
        plt = plot()
        
        # --- THIS IS THE CORRECTED PART ---
        # The data is in the .x and .obj fields directly
        x_vals = profile_result.x
        nll_vals = profile_result.obj
        # ------------------------------------
        
        plot_profile_delta_chi2!(plt, x_vals, nll_vals; pname=param_name, nll_anchor=nll_mle)
        
        if haskey(true_param_values, param_name)
            vline!(plt, [true_param_values[param_name]]; label="True Value", color=:purple, linestyle=:dash)
        end
        
        savefig(plt, joinpath(prof_dir, "profile_$(param_name).png"))
        println("[Profiling] Saved plot for $(param_name)")
    end
    # --- END OF FIX ---

    println("[Profiling] Done in $(round(time()-t_start; digits=2)) s")
    return profile_res
end