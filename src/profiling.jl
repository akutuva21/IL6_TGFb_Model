# src/profiling.jl

using PEtab
using Optim
using ComponentArrays
using Printf
using Base.Threads

# Entry point called by main.jl
function run_likelihood_profiling(
    petab_model::PEtabModel,
    odesolver,
    steadystate_solver,
    θ_mle::ComponentVector,
    true_param_values::Dict; # unused but kept for compatibility
    profiling_method::Symbol = :from_scratch,
    kwargs...
)
    println("\n--- Finding Confidence Intervals From Scratch (Advanced) ---")
    t_start = time()

    # Build PEtab problem for cost evaluation
    problem_kwargs = Dict(:odesolver => odesolver, :verbose => false)
    if !isnothing(steadystate_solver)
        problem_kwargs[:ss_solver] = steadystate_solver
    end
    petab_problem = PEtabODEProblem(petab_model; problem_kwargs...)

    # Compute CI endpoints for all estimated parameters (skip noise/init by name)
    find_all_endpoints_advanced(petab_problem, θ_mle)

    println("[Profiling] Done in $(round(time() - t_start; digits=2)) s")
    return nothing
end


"""
    find_all_endpoints(petab_problem, θ_mle)

Compute and print 95% CI endpoints for each profiled parameter via bisection.
Skips parameters named like noise (prefix "sigma") and initial conditions (suffix "_0").
"""
function find_all_endpoints(petab_problem::PEtabODEProblem, θ_mle::ComponentVector)
    all_names = string.(keys(θ_mle))
    profiled_names = [n for n in all_names if !startswith(n, "sigma") && !endswith(n, "_0")]

    # Base NLL and target level for 95% CI (1 dof): Δχ² = 3.84 => ΔNLL = 1.92
    nll_mle = petab_problem.nllh(θ_mle)
    threshold = 3.84
    target_nll = nll_mle + threshold / 2.0

    # Pre-allocate a results array: (parameter, lower_bound, upper_bound)
    results = Vector{Tuple{String, Union{Float64, Nothing}, Union{Float64, Nothing}}}(undef, length(profiled_names))

    # Parallelize across parameters
    Threads.@threads for i in 1:length(profiled_names)
        pname = profiled_names[i]
        println("Thread $(threadid()): STARTING profile for $pname...")
        t_start_param = time()
        lb = find_endpoint(pname, petab_problem, θ_mle, target_nll, -1)
        ub = find_endpoint(pname, petab_problem, θ_mle, target_nll, +1)
        t_end_param = time()
        elapsed_time = round(t_end_param - t_start_param, digits=2)
        println("Thread $(threadid()): FINISHED profile for $pname in $elapsed_time seconds.")
        results[i] = (pname, lb, ub)
    end

    println("\n" * "="^50)
    println("Confidence Interval Results (95%)")
    println("="^50)
    @printf("%-28s | %-18s | %-18s\n", "Parameter", "Lower Bound", "Upper Bound")
    println("-"^50)
    for (pname, lb, ub) in results
        lb_str = isnothing(lb) ? "Non-identifiable" : @sprintf("%.6f", lb)
        ub_str = isnothing(ub) ? "Non-identifiable" : @sprintf("%.6f", ub)
        @printf("%-28s | %-18s | %-18s\n", pname, lb_str, ub_str)
    end
    println("="^50)
end


"""
    find_endpoint(param_name, petab_problem, θ_mle, target_nll, direction)

Find a single CI endpoint for `param_name` in `direction` (-1 left, +1 right)
by solving for NLL_profiled(val) = target_nll using bisection.
Returns `nothing` if non-identifiable (target not reached within bounds).
"""
function find_endpoint(param_name::String, petab_problem::PEtabODEProblem, θ_mle::ComponentVector, target_nll::Float64, direction::Int)
    # Map name to index/symbol
    all_syms = collect(keys(θ_mle))
    all_names = string.(all_syms)
    idx = findfirst(==(param_name), all_names)
    idx === nothing && return nothing
    psym = all_syms[idx]

    # Root function: f(val) = profiled_nll(val) - target
    f = (val) -> begin
        nll = calculate_profiled_nllh(val, psym, petab_problem, θ_mle)
        nll - target_nll
    end

    a = θ_mle[psym]
    lb_vec = collect(petab_problem.lower_bounds)
    ub_vec = collect(petab_problem.upper_bounds)
    b = direction == -1 ? lb_vec[idx] : ub_vec[idx]

    # If at boundary target not reached, non-identifiable in this direction
    fb = try f(b) catch; Inf end
    if !isfinite(fb) || fb < 0
        return nothing
    end

    # Bisection requires bracketing with opposite signs
    # At a (MLE), f(a) should be negative by construction
    if direction == -1
        return bisection_search(f, b, a)
    else
        return bisection_search(f, a, b)
    end
end


"""
    bisection_search(f, a, b; tol=1e-4, max_iters=60)

Find root of f(x)=0 on [a,b] assuming f(a) and f(b) have opposite signs.
"""
function bisection_search(f, a, b; tol=1e-4, max_iters=60)
    fa = f(a)
    fb = f(b)
    if !isfinite(fa) || !isfinite(fb) || sign(fa) == sign(fb)
        @warn "Bisection failed to bracket a root (bad signs or non-finite endpoints)."
        return nothing
    end

    for _ in 1:max_iters
        c = (a + b) / 2
        fc = f(c)
        if !isfinite(fc)
            # Nudge inward toward finite region
            c = nextfloat(c)
            fc = f(c)
            if !isfinite(fc)
                return nothing
            end
        end

        if abs(b - a) / 2 < tol || fc == 0
            return c
        end
        if sign(fc) == sign(fa)
            a = c; fa = fc
        else
            b = c; fb = fc
        end
    end
    return (a + b) / 2
end


"""
    calculate_profiled_nllh(val_to_profile, param_to_profile, petab_problem, θ_mle)

Given a fixed parameter value, re-optimizes all other parameters (box constraints)
and returns the profiled NLL.
"""
function calculate_profiled_nllh(
    val_to_profile::Float64,
    param_to_profile::Symbol,
    petab_problem::PEtabODEProblem,
    θ_mle::ComponentVector
)::Float64
    # Prepare indices and symbols (all done locally for thread safety)
    all_syms = collect(keys(θ_mle))
    idx_profile = findfirst(==(param_to_profile), all_syms)
    @assert idx_profile !== nothing "Parameter to profile not found in θ_mle"

    # Symbols to optimize (all except the profiled parameter)
    optim_syms = [s for s in all_syms if s != param_to_profile]
    name_to_idx = Dict(s => i for (i, s) in enumerate(all_syms))
    optim_idxs = [name_to_idx[s] for s in optim_syms]

    # Bounds and initial guess for the sub-problem
    lb_sub = Float64[petab_problem.lower_bounds[s] for s in optim_syms]
    ub_sub = Float64[petab_problem.upper_bounds[s] for s in optim_syms]

    p_init_sub = (lb_sub .+ ub_sub) ./ 2.0

    # Define a fully local objective (no shared mutable state)
    function sub_cost(p_free::AbstractVector)
        # Start from the current MLE and set the profiled parameter
        p_full = ComponentArray(θ_mle)
        p_full[param_to_profile] = val_to_profile
        # Fill optimized parameters from the free vector
        @inbounds for (i, s) in enumerate(optim_syms)
            p_full[s] = p_free[i]
        end
        val = try
            petab_problem.nllh(p_full)
        catch
            Inf
        end
        return isfinite(val) ? val : Inf
    end

    # Use a derivative-free optimizer (NelderMead), loose tolerances, and fewer iterations for speed
    res = Optim.optimize(
        sub_cost,
        lb_sub,
        ub_sub,
        p_init_sub,
        Optim.Fminbox(Optim.NelderMead()),
        Optim.Options(
            f_reltol=1e-2, # Looser tolerance
            iterations=100, # Fewer iterations
            show_trace=false
        )
    )
    return Optim.minimum(res)
end


"""
    find_all_endpoints_advanced(petab_problem, θ_mle)

Stateful, multithreaded CI computation that uses the last accepted point
as the initial guess for each sub-optimization to improve robustness.
"""
function find_all_endpoints_advanced(petab_problem::PEtabODEProblem, θ_mle::ComponentVector)
    all_names = string.(keys(θ_mle))
    profiled_names = [n for n in all_names if !startswith(n, "sigma") && !endswith(n, "_0")]

    nll_mle = petab_problem.nllh(θ_mle)
    target_nll = nll_mle + 3.84 / 2.0

    results = Vector{Tuple{String, Union{Float64, Nothing}, Union{Float64, Nothing}}}(undef, length(profiled_names))

    Threads.@threads for i in 1:length(profiled_names)
        pname = profiled_names[i]
        println("Thread $(threadid()): STARTING profile for $pname...")
        t_start_param = time()

        lb = find_endpoint_advanced(pname, petab_problem, θ_mle, target_nll, -1)
        ub = find_endpoint_advanced(pname, petab_problem, θ_mle, target_nll, +1)

        elapsed_time = round(time() - t_start_param, digits=2)
        println("Thread $(threadid()): FINISHED profile for $pname in $elapsed_time seconds.")
        results[i] = (pname, lb, ub)
    end

    println("\n" * "="^50)
    println("Confidence Interval Results (95%)")
    println("="^50)
    @printf("%-28s | %-18s | %-18s\n", "Parameter", "Lower Bound", "Upper Bound")
    println("-"^50)
    for (pname, lb, ub) in results
        lb_str = isnothing(lb) ? "Non-identifiable" : @sprintf("%.6f", lb)
        ub_str = isnothing(ub) ? "Non-identifiable" : @sprintf("%.6f", ub)
        @printf("%-28s | %-18s | %-18s\n", pname, lb_str, ub_str)
    end
    println("="^50)
end


"""
    find_endpoint_advanced(param_name, petab_problem, θ_mle, target_nll, direction)

Use a stateful walk with smart initialization to robustly find a single CI endpoint
in the given direction (-1 left, +1 right).
"""
function find_endpoint_advanced(param_name::String, petab_problem::PEtabODEProblem, θ_mle::ComponentVector, target_nll::Float64, direction::Int)
    # Resolve symbol and index for the parameter
    all_syms = collect(keys(θ_mle))
    all_names = string.(all_syms)
    idx = findfirst(==(param_name), all_names)
    idx === nothing && return nothing
    psym = all_syms[idx]

    # Stateful last-accepted full parameter vector for smart initialization
    last_accepted_p = ComponentArray(copy(θ_mle))

    # Root function: compute profiled NLL at fixed value and update last_accepted_p on success
    f = (val) -> begin
        nll, p_new = calculate_profiled_nllh_advanced(val, psym, petab_problem, last_accepted_p)
        if !isnothing(p_new)
            last_accepted_p = p_new
        end
        nll - target_nll
    end

    # Bracket between MLE and bound in the chosen direction
    a = θ_mle[psym]
    lb_vec = collect(petab_problem.lower_bounds)
    ub_vec = collect(petab_problem.upper_bounds)
    b = direction == -1 ? lb_vec[idx] : ub_vec[idx]

    fb = try f(b) catch; Inf end
    if !isfinite(fb) || fb < 0
        return nothing
    end

    return direction == -1 ? bisection_search(f, b, a) : bisection_search(f, a, b)
end


"""
    calculate_profiled_nllh_advanced(val_to_profile, param_to_profile, petab_problem, p_init_full)

Given a fixed parameter value, re-optimizes all other parameters using a smart initial
guess (last accepted point) and LBFGS. Returns (nll, new_full_params_or_nothing).
"""
function calculate_profiled_nllh_advanced(
    val_to_profile::Float64,
    param_to_profile::Symbol,
    petab_problem::PEtabODEProblem,
    p_init_full::ComponentVector
)::Tuple{Float64, Union{ComponentVector, Nothing}}
    # Symbols and indices
    optim_syms = [s for s in keys(p_init_full) if s != param_to_profile]
    all_syms = collect(keys(p_init_full))
    name_to_idx = Dict(s => i for (i, s) in enumerate(all_syms))
    optim_idxs = [name_to_idx[s] for s in optim_syms]

    # Set fixed param in the initial full vector
    p_init_full = ComponentArray(p_init_full)
    p_init_full[param_to_profile] = val_to_profile

    # Bounds and initial guess for sub-problem from last accepted point
    lb_all = collect(petab_problem.lower_bounds)
    ub_all = collect(petab_problem.upper_bounds)
    lb_sub = lb_all[optim_idxs]
    ub_sub = ub_all[optim_idxs]
    p_init_sub = Float64[p_init_full[s] for s in optim_syms]

    # Local objective
    function sub_cost(p_free::AbstractVector)
        p_full = ComponentArray(p_init_full)
        @inbounds for (i, s) in enumerate(optim_syms)
            p_full[s] = p_free[i]
        end
        val = try
            petab_problem.nllh(p_full)
        catch
            Inf
        end
        return isfinite(val) ? val : Inf
    end

    # Optimize with LBFGS (box constraints)
    res = try
        Optim.optimize(
            sub_cost,
            lb_sub,
            ub_sub,
            p_init_sub,
            Optim.Fminbox(Optim.LBFGS()),
            Optim.Options(g_tol=1e-4, f_reltol=1e-3, iterations=150, show_trace=false)
        )
    catch e
        return (Inf, nothing)
    end

    # If not converged, return high cost without updating state
    if !Optim.converged(res)
        return (Optim.minimum(res) + 100.0, nothing)
    end

    # Build the new full parameter vector with the minimizer inserted
    p_new = ComponentArray(p_init_full)
    minim = Optim.minimizer(res)
    @inbounds for (i, s) in enumerate(optim_syms)
        p_new[s] = minim[i]
    end

    return (Optim.minimum(res), p_new)
end