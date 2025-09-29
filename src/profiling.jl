using PEtab
using LikelihoodProfiler
using Optimization
using OptimizationNLopt
using CICOBase
using Printf
using Logging
using ComponentArrays

function run_likelihood_profiling(
    petab_model::PEtabModel,
    odesolver,
    steadystate_solver,
    θ_mle::ComponentVector,
    true_param_values::Dict
)
    println("\n--- Finding Confidence Intervals with CICOBase (ONE_PASS) ---")
    t_start = time()

    petab_problem = PEtabODEProblem(petab_model, verbose=false)

    # Convert to vectors
    θ_mle_vector = collect(θ_mle)
    lb_vector = collect(petab_problem.lower_bounds)
    ub_vector = collect(petab_problem.upper_bounds)
    param_names = string.(petab_problem.xnames)

    # Sanity checks
    @assert length(θ_mle_vector) == length(lb_vector) == length(ub_vector) "bounds/θ mismatch"

    # Build CICOBase-compatible bounds with expansion for profiling
    expand = 2.0  # in log10 units if parameters are log-scaled
    lb_prof = lb_vector .- expand
    ub_prof = ub_vector .+ expand
    cico_bounds = [(lb_prof[i], ub_prof[i]) for i in eachindex(lb_prof)]

    # Baseline loss and absolute threshold for 95% confidence interval
    obj0 = petab_problem.nllh(θ_mle_vector)
    losscrit = obj0 + 3.84  # Chi-squared critical value for 95% CI with 1 DOF

    println("Baseline loss (MLE): ", obj0)
    println("Loss threshold (95% CI): ", losscrit)

    # Objective for CICOBase with defensive error handling
    function safe_nllh(θ::AbstractVector{<:Real})
        try
            val = petab_problem.nllh(θ)
            return isfinite(val) ? val : Inf
        catch err
            @warn "nllh evaluation failed during profiling; treating as Inf" err maxlog=10
            return Inf
        end
    end

    cico_objective = θ -> safe_nllh(θ)

    # Tight scan tolerance
    tight_scan_tol = 1e-4

    # Simple scan_bounds function using expanded parameter bounds
    function scan_bounds_for(i)
        θi, lbi, ubi = θ_mle_vector[i], lb_prof[i], ub_prof[i]
        return (lbi, ubi)
    end

    # Storage
    intervals = Vector{Union{Nothing, CICOBase.ParamInterval}}(undef, length(θ_mle_vector))

    println("\nComputing confidence intervals with CICOBase.get_interval ...")
    for i in eachindex(θ_mle_vector)
        println("Parameter $i: $(param_names[i]) bounds=($(lb_vector[i]), $(ub_vector[i])), MLE=$(θ_mle_vector[i])")
        
        # Use simple scan_bounds with original parameter bounds
        scan_bounds_tuple = scan_bounds_for(i)
        println("  Scan_bounds: ", scan_bounds_tuple)
        
        try
            intervals[i] = CICOBase.get_interval(
                θ_mle_vector, i, cico_objective, :CICO_ONE_PASS;
                loss_crit    = losscrit,
                theta_bounds = cico_bounds,
                scan_bounds  = scan_bounds_tuple,
                scan_tol     = tight_scan_tol,
                local_alg    = :LN_BOBYQA,
                silent       = true
            )
            println("  ✓ done")
        catch e
            println("  ✗ failed: ", e)
            intervals[i] = nothing
        end
    end

    # Print summary like the minimal script
    println("\n" * "="^50)
    println("Confidence Interval Results (95%) - CICOBase Direct")
    println("="^50)
    @printf("%-28s | %-18s | %-18s | %-14s | %s\n", "Parameter", "Lower Bound", "Upper Bound", "Status", "Endpoint codes")
    println("-"^100)

    for i in eachindex(intervals)
        if intervals[i] === nothing
            @printf("%-28s | %-18s | %-18s | %-14s | %s\n",
                param_names[i], "FAILED", "FAILED", "ERROR", "n/a")
            continue
        end
        interval = intervals[i]
        left_ep, right_ep = interval.result

        lb_str = isnothing(left_ep.value)  ? "none" : @sprintf("%.6f", left_ep.value)
        ub_str = isnothing(right_ep.value) ? "none" : @sprintf("%.6f", right_ep.value)

        rets = (left_ep.status, right_ep.status)
        status =
            rets == (:BORDER_FOUND_BY_SCAN_TOL, :BORDER_FOUND_BY_SCAN_TOL) ? "Identifiable" :
            (:SCAN_BOUND_REACHED in rets)                                  ? "Non-identifiable" :
            (:MAX_ITER_REACHED in rets)                                    ? "MaxIters" :
            "Partial"

        @printf("%-28s | %-18s | %-18s | %-14s | (%s, %s)\n",
            param_names[i], lb_str, ub_str, status, String(rets[0x01]), String(rets[0x02]))
    end
    println("="^100)
    println("[Profiling] Done in $(round(time() - t_start; digits=2)) s")
    return intervals
end