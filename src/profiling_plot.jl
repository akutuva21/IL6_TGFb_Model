# src/profiling_plot.jl
using PEtab
using LikelihoodProfiler
using Optimization, OptimizationOptimJL
using DataFrames
using ComponentArrays
using Plots
using Logging
using SciMLBase
using ForwardDiff
using LineSearches
using CSV

const BIG_PEN = 1e30
const EPS = 1e-6

# Ensure GR backend for saving plots in headless environments
gr()

"""
Continuation-based, warm-started profile of a single parameter.

- Fixes parameter i to each value v by setting lb[i]=ub[i]=v, re-optimizes others.
- Warm-starts each solve from previous θ to trace a continuous path.
- Returns a DataFrame with columns: value (the fixed parameter), objective (ΔNLLH),
  and retcode (Symbol), plus the sequence of θ solutions (optional).
"""
function manual_profile_param!(
    i::Int,
    θ_mle_vec::Vector{Float64},
    lb_prof::Vector{Float64},
    ub_prof::Vector{Float64},
    vgrid::AbstractVector{Float64},
    optfunc::OptimizationFunction,
    base_optimizer::Any;
    maxiters::Int=20_000,
    reltol::Float64=5e-3,
)
    rows = Vector{NamedTuple{(:value,:objective,:retcode),Tuple{Float64,Float64,Symbol}}}(undef, length(vgrid))
    θ_prev = copy(θ_mle_vec)

    for (k, v) in enumerate(vgrid)
        lb = copy(lb_prof); ub = copy(ub_prof)
        lb[i] = v; ub[i] = v  # fix parameter i at v

        θ0 = clamp.(θ_prev, lb, ub)
        θ0[i] = v

        prob = OptimizationProblem(optfunc, θ0; lb=lb, ub=ub)
        sol  = solve(prob, base_optimizer; maxiters=maxiters, reltol=reltol)

        # Record and warm-start next step from the best θ or fallback last θ if failed
        ret = sol.retcode isa Symbol ? sol.retcode : Symbol(string(sol.retcode))
        if sol.u isa AbstractVector && length(sol.u) == length(θ_prev)
            θ_best = clamp.(Array(sol.u), lb, ub)
        else
            θ_best = θ0
        end
        θ_prev .= θ_best

        # Evaluate objective at the returned point (finite by construction)
        fval = sol.objective

        rows[k] = (value=v, objective=fval, retcode=ret)
    end

    return DataFrame(rows)
end

function run_likelihood_profiling(
    petab_model::PEtabModel,
    odesolver,
    steadystate_solver,
    θ_mle::ComponentVector,
    true_param_values::Dict;
    num_points::Int = 60,
    expand::Float64 = 2.0,
    use_threads::Bool = true,
    min_distinct::Int = 8,               # threshold to trigger manual continuation
    manual_points::Int = 100,            # points for manual continuation when needed
    manual_step_cap::Float64 = 2e-3      # small per-step cap for manual pass
)
    println("\n--- Generating Likelihood Profiles with LikelihoodProfiler.jl ---")
    t_start = time()

    petab_problem = odesolver === nothing ?
        PEtabODEProblem(petab_model, verbose=false) :
        PEtabODEProblem(petab_model; verbose=false, odesolver=odesolver)

    param_syms  = collect(petab_problem.xnames)
    param_names = string.(param_syms)

    # Robust accessor that accepts :log10_x or :x and tries the alternate form if missing
    @inline function _get_key(x::ComponentVector, s::Symbol)
        if haskey(x, s)
            return x[s]
        else
            str = String(s)
            alt = startswith(str, "log10_") ? Symbol(str[7:end]) : Symbol("log10_" * str)
            return x[alt]  # will throw if neither exists, which reveals a real mismatch
        end
    end

    # Build MLE vector and lookups in the exact PEtab order (estimation scale)
    θ_mle_vector = [_get_key(θ_mle, s) for s in param_syms]
    θ_lookup     = Dict(param_syms[i] => θ_mle_vector[i] for i in eachindex(param_syms))

    lb_vector = collect(petab_problem.lower_bounds)
    ub_vector = collect(petab_problem.upper_bounds)

    lb_prof = lb_vector .- expand
    ub_prof = ub_vector .+ expand
    profile_ranges = [(lb_prof[i], ub_prof[i]) for i in eachindex(lb_prof)]

    # Assert feasibility to catch bound mistakes early
    @assert all(lb_prof .<= θ_mle_vector .<= ub_prof)

    @inline function unwrap_dual(x)
        while x isa ForwardDiff.Dual
            x = ForwardDiff.value(x)
        end
        return x
    end

    # Raw objective with finite penalties
    function raw_nllh(vec::AbstractVector)
        vec_primal = unwrap_dual.(vec)
        if any(vec_primal .< lb_prof) || any(vec_primal .> ub_prof)
            return BIG_PEN
        end
        try
            val = petab_problem.nllh(vec_primal)  # Vector in PEtab order, on estimation scale
            val_real = unwrap_dual(val)
            return isfinite(val_real) ? val_real : BIG_PEN
        catch err
            @warn "nllh evaluation failed during profiling; using BIG_PEN" err maxlog=10
            return BIG_PEN
        end
    end

    nllh_mle_raw = raw_nllh(θ_mle_vector)
    println("Baseline loss (MLE): ", nllh_mle_raw)

    # Shifted objective so MLE ~ 0 (strictly negative by EPS to satisfy strict inequalities downstream)
    objective(θ, p) = raw_nllh(θ) - nllh_mle_raw - EPS
    @assert objective(θ_mle_vector, nothing) < 3.84 "Shift failed: baseline is not below threshold"

    # Gradient callback on the estimation scale
    function grad!(G, θ, p)
        petab_problem.grad!(G, θ)            # ∂NLLH/∂θ; shift is constant so gradient unchanged
        return nothing
    end

    # Construct the OptimizationFunction correctly
    optfunc = OptimizationFunction(objective, Optimization.AutoZygote(); grad = grad!)

    # Optim configuration: Fminbox(LBFGS) + BackTracking + generous iteration budget
    lbfgs_bt = OptimizationOptimJL.LBFGS(linesearch = LineSearches.BackTracking())
    optimizer = OptimizationOptimJL.Fminbox(lbfgs_bt)

    # Build ProfileLikelihoodProblem for the first pass (package-driven scanning)
    plprob = LikelihoodProfiler.ProfileLikelihoodProblem(
        OptimizationProblem(optfunc, θ_mle_vector; lb = lb_prof, ub = ub_prof),
        θ_mle_vector,
        profile_ranges;
        threshold = 3.84
    )

    # Small fixed steps for the package-driven pass
    span = ub_prof .- lb_prof
    function initial_step(θ, i)
        denom = max(num_points - 1, 1)
        step = span[i] / denom
        return clamp(step, 1e-3, 4e-3)
    end
    stepper = LikelihoodProfiler.FixedStep(; initial_step)

    pkg_method = LikelihoodProfiler.OptimizationProfiler(
        optimizer = optimizer,
        stepper   = stepper,
        optimizer_opts = (
            reltol = 5e-3,
            maxiters = 20_000,
            iterations = 20_000,
            allow_f_increases = true,
            successive_f_tol = 3
        )
    )

    parallel_mode = use_threads ? :threads : :none
    pkg_sol = LikelihoodProfiler.solve(plprob, pkg_method; parallel_type = parallel_mode)

    # Helper to find columns, robust to String/Symbol names
    find_col(df, candidates::Vector{String}) = begin
        nms = names(df); nms_s = String.(nms)
        for cand in candidates
            i = findfirst(==(cand), nms_s)
            if i !== nothing; return nms[i]; end
        end
        return nothing
    end

    println("Saving profile likelihood plots...")
    plot_dir = "likelihood_profiles"
    mkpath(plot_dir)

    threshold_95 = 3.84
    threshold_99 = 6.63

    # Iterate parameters; if first pass is too sparse, run manual continuation and replot
    for (idx, pname) in enumerate(param_names)
        profile_values = pkg_sol[idx]
        df = DataFrame(profile_values)

        loss_col = find_col(df, ["loss","objective"])
        xcol     = find_col(df, ["x$(idx)","value"])
        if loss_col === nothing || xcol === nothing
            @warn "Profile missing expected columns; switching to manual continuation" parameter=pname columns=names(df)
            loss_col = :objective; xcol = :value
            df = DataFrame()  # will be replaced below
        end

        # Count distinct x in the package pass
        distinct_ok = false
        if nrow(df) > 0
            vals = Float64.(df[!, xcol])
            uniq = unique(round.(vals; digits=6))
            distinct_ok = length(uniq) >= min_distinct
        end

        # Manual continuation if sparse or columns missing
        if !distinct_ok
            # Build a dense grid around the MLE within profile bounds with capped steps
            v_mle = θ_lookup[param_syms[idx]]
            lo, hi = profile_ranges[idx]
            # symmetric expansion centered at MLE, capped by manual_step_cap spacing
            halfspan = min(v_mle - lo, hi - v_mle)
            # ensure at least a small window if halfspan is tiny
            halfspan = max(halfspan, 10*manual_step_cap)
            v_lo = max(lo, v_mle - halfspan)
            v_hi = min(hi, v_mle + halfspan)
            # step size capped
            raw_step = (v_hi - v_lo) / max(manual_points - 1, 1)
            step = min(raw_step, manual_step_cap)
            # build grid anchored at MLE and expand both directions
            left  = reverse(collect(v_mle:-step:v_lo))[2:end]
            right = collect(v_mle:step:v_hi)
            vgrid = vcat(left, right)
            vgrid = unique(round.(vgrid; digits=9))

            df = manual_profile_param!(
                idx,
                copy(θ_mle_vector),
                lb_prof,
                ub_prof,
                vgrid,
                optfunc,
                optimizer;
                maxiters=20_000,
                reltol=5e-3
            )
            loss_col = :objective
            xcol     = :value
        end

        # Plot
        loss_values = df[!, loss_col]
        valid_mask  = findall(i -> isfinite(loss_values[i]), eachindex(loss_values))
        if length(valid_mask) < 2
            @warn "Skipping plot; insufficient valid points" parameter=pname valid_points=length(valid_mask)
            continue
        end

        values    = Float64.(df[!, xcol][valid_mask])
        delta_ll  = Float64.(loss_values[valid_mask])   # already ΔNLLH (shifted objective)
        order     = sortperm(values)
        values    = values[order]
        delta_ll  = delta_ll[order]

        uniq = unique(round.(values; digits=6))
        if length(uniq) < min_distinct
            @warn "Profile still sparse after continuation" parameter=pname distinct_points=length(uniq)
        end

        plt = plot(
            values, delta_ll,
            seriestype = :line,
            linewidth = 2,
            label = "Profile Likelihood",
            xlabel = pname,
            ylabel = "ΔNLLH",
            title = "Profile for $pname",
            legend = :top,
            framestyle = :box,
            ylims = (0, 15)
        )
        hline!(plt, [threshold_95], linestyle = :dash, color = :red,    label = "95% CI")
        hline!(plt, [threshold_99], linestyle = :dash, color = :orange, label = "99% CI")

        mle_val = θ_lookup[param_syms[idx]]
        scatter!(plt, [mle_val], [0.0], color = :black, markersize = 5, label = "MLE")

        savefig(plt, joinpath(plot_dir, "profile_$(pname).png"))

        # Save raw profile data
        CSV.write(joinpath(plot_dir, "profile_$(pname)_raw.csv"), df[valid_mask, [xcol, loss_col]])

        # Compute smoothed profile (moving average)
        window_size = max(3, length(delta_ll) ÷ 10)
        smoothed_delta_ll = [mean(delta_ll[max(1, i-window_size÷2):min(length(delta_ll), i+window_size÷2)]) for i in eachindex(delta_ll)]
        
        # Plot smoothed profile
        plt_smooth = plot(
            values, smoothed_delta_ll,
            seriestype = :line,
            linewidth = 2,
            color = :blue,
            label = "Smoothed Profile",
            xlabel = pname,
            ylabel = "ΔNLLH",
            title = "Smoothed Profile for $pname",
            legend = :top,
            framestyle = :box,
            ylims = (0, 15)
        )
        hline!(plt_smooth, [threshold_95], linestyle = :dash, color = :red,    label = "95% CI")
        hline!(plt_smooth, [threshold_99], linestyle = :dash, color = :orange, label = "99% CI")
        scatter!(plt_smooth, [mle_val], [0.0], color = :black, markersize = 5, label = "MLE")
        savefig(plt_smooth, joinpath(plot_dir, "profile_$(pname)_smoothed.png"))

        # Compute confidence intervals numerically
        function find_ci_bounds(delta_ll_vals, threshold)
            # Find indices where profile crosses threshold
            crossings = findall(i -> delta_ll_vals[i] <= threshold, 1:length(delta_ll_vals))
            if isempty(crossings)
                return (NaN, NaN)
            end
            left_idx = minimum(crossings)
            right_idx = maximum(crossings)
            
            # Linear interpolation for more precise bounds
            left_bound = if left_idx > 1
                # Interpolate between left_idx-1 and left_idx
                x1, x2 = values[left_idx-1], values[left_idx]
                y1, y2 = delta_ll_vals[left_idx-1], delta_ll_vals[left_idx]
                if y1 > threshold && y2 <= threshold
                    x1 + (x2 - x1) * (threshold - y1) / (y2 - y1)
                else
                    values[left_idx]
                end
            else
                values[left_idx]
            end
            
            right_bound = if right_idx < length(values)
                # Interpolate between right_idx and right_idx+1
                x1, x2 = values[right_idx], values[right_idx+1]
                y1, y2 = delta_ll_vals[right_idx], delta_ll_vals[right_idx+1]
                if y1 <= threshold && y2 > threshold
                    x1 + (x2 - x1) * (threshold - y1) / (y2 - y1)
                else
                    values[right_idx]
                end
            else
                values[right_idx]
            end
            
            return (left_bound, right_bound)
        end

        ci95_left, ci95_right = find_ci_bounds(delta_ll, threshold_95)
        ci99_left, ci99_right = find_ci_bounds(delta_ll, threshold_99)

        # Save CI summary
        ci_df = DataFrame(
            parameter = [pname],
            mle = [mle_val],
            ci95_left = [ci95_left],
            ci95_right = [ci95_right],
            ci99_left = [ci99_left],
            ci99_right = [ci99_right]
        )
        CSV.write(joinpath(plot_dir, "profile_$(pname)_ci.csv"), ci_df)
    end

    println("Saved to: ", abspath(plot_dir))
    println("Plot count: ", length(readdir(plot_dir)))
    println("✅ All profile plots saved to the '$plot_dir' directory.")
    println("[Profiling] Done in $(round(time() - t_start; digits=2)) s")
    return nothing
end