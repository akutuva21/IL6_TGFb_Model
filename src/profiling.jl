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

    # Build a fresh PEtab ODE problem with provided solvers
    petab_problem = PEtabODEProblem(petab_model; odesolver=odesolver, ss_solver=steadystate_solver)

    all_names = string.(keys(θ_mle))
    # Filter: exclude noise params and initial conditions (ending with _0)
    cand = [n for n in all_names if !startswith(n, "noiseParameter") && !endswith(n, "_0")]
    target = debug ? 4 : 8
    params = first(cand, min(length(cand), target))
    if isempty(params)
        params = first(all_names, min(length(all_names), target))
    end
    println("[Profiling] Parameters: $(params)")

    θ_init = collect(θ_mle)
    lb = collect(petab_problem.lower_bounds)
    ub = collect(petab_problem.upper_bounds)

    # Indices for selected params
    idxs = Int[]
    for p in params
        i = findfirst(==(p), all_names)
        if i !== nothing && ub[i] - lb[i] > 1e-9
            push!(idxs, i)
        else
            println("[Profiling] Skipping fixed param $p")
        end
    end
    if isempty(idxs)
        println("[Profiling] No variable parameters to profile; aborting.")
        return Dict{String,Any}()
    end

    # Objective
    function obj(θ_est, _)
        θ_work = eltype(θ_est) <: ForwardDiff.Dual ? ForwardDiff.value.(θ_est) : θ_est
        try
            petab_problem.nllh(θ_work; prior=false)
        catch
            1e6
        end
    end

    optf = OptimizationFunction(obj, Optimization.AutoFiniteDiff())
    optprob = OptimizationProblem(optf, θ_init; lb=lb, ub=ub)
    plprob = LikelihoodProfiler.PLProblem(optprob, θ_init)
    profiler = LikelihoodProfiler.OptimizationProfiler(
    optimizer = OptimizationOptimJL.LBFGS(),
    stepper = LikelihoodProfiler.FixedStep(; initial_step = 0.1)
    )
    println("[Profiling] Profiler constructed with LBFGS + FixedStep(0.1)")

    println("[Profiling] Running threaded profiling on $(length(idxs)) parameters...")
    res = LikelihoodProfiler.profile(plprob, profiler; idxs=idxs, maxiters=maxiters, parallel_type=:threads)

    prof_dir = joinpath(pwd(), "likelihood_profiles"); mkpath(prof_dir)
    out = Dict{String,Any}()
    profiles_array = getproperty(res, :profiles)
    for (j, p) in enumerate(params)
        if j <= length(profiles_array)
            pr = profiles_array[j]
            out[p] = pr
            try
                plt = plot(pr, 1; xlabel=p, ylabel="Δ Log-Likelihood", title="Profile: $p")
                hline!(plt, [1.92]; color=:red, linestyle=:dash, label="95% CI")
                savefig(plt, joinpath(prof_dir, "profile_$(p).png"))
                println("[Profiling] Saved $p")
            catch e
                println("[Profiling] Plot failed for $p: $e")
            end
        end
    end

    println("[Profiling] Done in $(round(time()-t_start; digits=2)) s")
    return out
end