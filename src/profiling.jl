using LikelihoodProfiler
using LikelihoodProfiler: AdaptiveStep, FixedStep
using Optimization
using OptimizationOptimJL
using ComponentArrays
using Plots
using PEtab
using ForwardDiff

"""
run_likelihood_profiling(petab_model, odesolver, steadystate_solver, θ_mle; debug=false, maxiters=20)

Minimal threaded likelihood profiling:
  - Selects a small set of non-initial, non-noise parameters (<=8; <=4 in debug)
  - Uses finite-difference gradients (AutoFiniteDiff)
  - Strips Dual numbers defensively
  - Saves one PNG per parameter under ./likelihood_profiles
Returns Dict{name => profile_result}.
"""
function run_likelihood_profiling(
    petab_model::PEtabModel,
    odesolver,
    steadystate_solver,
    θ_mle::ComponentVector;
    debug::Bool=false,
    maxiters::Int=20
)
    println("\n--- Likelihood Profiling (minimal) ---"); flush(stdout)
    t_start = time()

    petab_problem = PEtabODEProblem(petab_model; odesolver=odesolver, ss_solver=steadystate_solver, verbose=false)

    all_names = string.(keys(θ_mle))
    cand = [n for n in all_names if !startswith(n, "noiseParameter") && !endswith(n, "_0")]
    #target = 8 # always take up to 8
    #params = first(cand, min(length(cand), target))
    params = cand
    if isempty(params)
        @warn "No dynamic parameters found for profiling, selecting from all parameters."
        params = first(all_names, min(length(all_names), target))
    end
    println("[Profiling] Parameters: $(params)")

    θ_init = collect(θ_mle)
    lb = collect(petab_problem.lower_bounds)
    ub = collect(petab_problem.upper_bounds)

    idxs = [findfirst(==(p), all_names) for p in params if findfirst(==(p), all_names) !== nothing && (ub[findfirst(==(p), all_names)] - lb[findfirst(==(p), all_names)] > 1e-9)]
    if isempty(idxs)
        println("[Profiling] No variable parameters to profile; aborting.")
        return Dict{String,Any}()
    end

    function obj(θ_est, _)
        θ_work = eltype(θ_est) <: ForwardDiff.Dual ? ForwardDiff.value.(θ_est) : θ_est
        if any(!isfinite, θ_work)
            return 1e8
        end
        try
            val = petab_problem.nllh(θ_work; prior=false)
            return isfinite(val) ? val : 1e8
        catch
            return 1e8
        end
    end

    # Explicitly use AutoForwardDiff for the gradient to avoid second-order warnings with IPNewton
    optf = OptimizationFunction(obj, Optimization.AutoForwardDiff())
    optprob = OptimizationProblem(optf, θ_init; lb=lb, ub=ub)
    # Add threshold for 95% confidence intervals
    threshold = 1.92  # χ²(1, 0.05)/2 for 95% CI
    plprob = LikelihoodProfiler.PLProblem(optprob, θ_init; threshold=threshold)

    profiler = LikelihoodProfiler.OptimizationProfiler(
        optimizer = OptimizationOptimJL.IPNewton(),
        stepper = LikelihoodProfiler.FixedStep(; initial_step = 0.005)
    )
        println("[Profiling] Profiler constructed with IPNewton + FixedStep(0.01)")

    println("[Profiling] Running threaded profiling on $(length(idxs)) parameters...")
    res = LikelihoodProfiler.profile(plprob, profiler; idxs=idxs, maxiters=5000, parallel_type=:threads)

    prof_dir = joinpath(pwd(), "likelihood_profiles")
    mkpath(prof_dir)
    out = Dict{String,Any}()

    sol = res
    for (j, p) in enumerate(params)
        if j <= length(sol.profiles)
            param_sol = sol[j]
            out[p] = param_sol
            try
                # Use the built-in plotting
                plt = plot(param_sol; xlabel=p, ylabel="Δ Log-Likelihood", 
                          title="Profile: $p", legend=:topright)
                hline!(plt, [threshold], color=:red, linestyle=:dash, label="95% CI")
                savefig(plt, joinpath(prof_dir, "profile_$(p).png"))
                println("[Profiling] Saved plot for $p")
            catch e
                @warn "Plot failed for $p: $e"
                # Fallback to manual extraction if needed
                try
                    profile_points = param_sol.profile
                    if !isempty(profile_points)
                        xs = [pt[1] for pt in profile_points]
                        ys = [pt[2] for pt in profile_points]
                        ys_delta = ys .- minimum(ys)
                        plt = plot(xs, ys_delta, seriestype=:path, marker=:circle,
                                  xlabel=p, ylabel="Δ Log-Likelihood", title="Profile: $p")
                        hline!(plt, [threshold], color=:red, linestyle=:dash, label="95% CI")
                        savefig(plt, joinpath(prof_dir, "profile_$(p).png"))
                        println("[Profiling] Saved fallback plot for $p")
                    end
                catch e2
                    @warn "Fallback plot also failed for $p: $e2"
                end
            end
        end
    end

    println("[Profiling] Done in $(round(time()-t_start; digits=2)) s")
    return out
end