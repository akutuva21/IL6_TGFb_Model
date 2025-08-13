using LikelihoodProfiler
using LikelihoodProfiler: AdaptiveStep, FixedStep
using Optimization
using OptimizationOptimJL
using ComponentArrays
using Plots
using PEtab
using ForwardDiff

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
        hline!(plt, [9.0]; lc=:red, ls=:dashdot, label="99%")
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

Minimal threaded likelihood profiling:
  - Uses finite-difference gradients (AutoFiniteDiff)
  - Strips Dual numbers defensively
  - Saves one PNG per parameter under ./likelihood_profiles
Returns Dict{name => profile_result}.
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
    println("\n--- Likelihood Profiling (with True Value for Robustness Testing) ---"); flush(stdout)
    t_start = time()

    petab_problem = PEtabODEProblem(petab_model; odesolver=odesolver, ss_solver=steadystate_solver, verbose=false)

    all_names = string.(keys(θ_mle))
    cand = [n for n in all_names if !startswith(n, "noiseParameter") && !endswith(n, "_0")]
    params = cand
    if isempty(params)
        @warn "No dynamic parameters found for profiling, selecting from all parameters."
        params = first(all_names, min(length(all_names), 8))
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

    optf = OptimizationFunction(obj, Optimization.AutoForwardDiff())
    optprob = OptimizationProblem(optf, θ_init; lb=lb, ub=ub)
    
    # Let the library calculate the threshold from a confidence level
    plprob = LikelihoodProfiler.PLProblem(optprob, θ_init; conf_level=0.95)

    profiler = LikelihoodProfiler.OptimizationProfiler(
        optimizer = OptimizationOptimJL.IPNewton(),
        stepper = LikelihoodProfiler.FixedStep(; initial_step = 0.005)
    )
    println("[Profiling] Profiler constructed with IPNewton + FixedStep(0.005)")

    println("[Profiling] Running threaded profiling on $(length(idxs)) parameters...")
    res = LikelihoodProfiler.profile(plprob, profiler; idxs=idxs, maxiters=200, parallel_type=:threads)

    # Compute global best NLL at the provided MLE to anchor Δχ²
    best_nll = begin
        θmle_vec = collect(θ_mle)
        try
            petab_problem.nllh(θmle_vec; prior=false)
        catch
            Inf
        end
    end

    prof_dir = joinpath(pwd(), "likelihood_profiles")
    mkpath(prof_dir)
    out = Dict{String,Any}()

    sol = res
    for (j, p) in enumerate(params)
        if j <= length(sol.profiles)
            param_sol = sol[j]
            out[p] = param_sol
            try
                xvals = getfield(param_sol, :params; default=nothing)
                nllvals = getfield(param_sol, :objective_values; default=nothing)
                if xvals === nothing || nllvals === nothing
                    xvals = getfield(param_sol, :theta; default=nothing)
                    nllvals = getfield(param_sol, :fun_values; default=nothing)
                end
                if xvals === nothing || nllvals === nothing
                    @warn "Unknown profile fields; cannot plot Δχ² for $p"
                    continue
                end
                anchor = isfinite(best_nll) ? best_nll : minimum(nllvals)
                plt = plot()
                plot_profile_delta_chi2!(plt, xvals, nllvals; pname=p, nll_anchor=anchor, ymax=4.0)
                if haskey(true_param_values, p)
                    vline!(plt, [true_param_values[p]]; label="True Value", color=:purple, linestyle=:dash)
                end
                savefig(plt, joinpath(prof_dir, "profile_$(p).png"))
                println("[Profiling] Saved plot for $p (Δχ², y≤4)")
            catch e
                @warn "Plot failed for $p: $e"
            end
        end
    end

    println("[Profiling] Done in $(round(time()-t_start; digits=2)) s")
    return out
end