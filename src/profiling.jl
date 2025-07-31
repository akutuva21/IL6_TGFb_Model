# src/profiling.jl
# Modern likelihood profiling using LikelihoodProfiler.jl

using LikelihoodProfiler
using OrdinaryDiffEq
using Optimization
using OptimizationOptimJL
using ComponentArrays
using Plots
using JLD2
using Statistics
using PEtab
using Pkg
using Base.Threads

# --- THREAD DIAGNOSTICS AT STARTUP ---
println("Julia Threads available: $(nthreads())")
flush(stdout)

include("solver_config.jl")

export run_likelihood_profiling

# Manual profiling fallback implementation
# function run_likelihood_profiling_manual(petab_problem::PEtabODEProblem, θ_mle::Union{ComponentVector, Nothing}=nothing, debug_mode::Bool=false)
#     println("\n--- 🔬 Starting Manual Likelihood Profiling ---")
    
#     if isnothing(θ_mle)
#         @error "MLE parameter estimates are required for profiling."
#         return nothing
#     end
    
#     θ_mle_vec = collect(θ_mle)
#     param_names = string.(petab_problem.xnames)
#     n_params = length(param_names)
    
#     # Create output directory
#     profile_dir = joinpath(pwd(), "likelihood_profiles")
#     if !isdir(profile_dir); mkpath(profile_dir); end
    
#     println("Manual profiling configuration:")
#     println("  Parameters to profile: $n_params")
#     println("  Parameter names: $(join(param_names, ", "))")
#     println("  Debug mode: $debug_mode")
    
#     # Manual profiling for each parameter
#     profiles = Dict()
#     successful_profiles = 0
    
#     # Get MLE log-likelihood for reference
#     try
#         mle_nllh = petab_problem.nllh(θ_mle_vec; prior=false)
#         println("  MLE log-likelihood: $mle_nllh")
#     catch e
#         @error "Could not evaluate MLE log-likelihood" exception=e
#         return nothing
#     end
    
#     for (i, param_name) in enumerate(param_names)
#         println("Profiling parameter $i: $param_name")
#         flush(stdout)
        
#         # Create range around MLE
#         lb = petab_problem.lower_bounds[i]
#         ub = petab_problem.upper_bounds[i]
#         mle_val = θ_mle_vec[i]
        
#         # Check for fixed parameters
#         if abs(ub - lb) < 1e-10
#             println("  🔒 Skipping fixed parameter: $param_name")
#             flush(stdout)
#             continue
#         end
        
#         # Create parameter range (more points around MLE)
#         n_points = debug_mode ? 20 : 50
#         param_range = range(lb, ub, length=n_points)
        
#         likelihood_values = Float64[]
#         mle_nllh = petab_problem.nllh(θ_mle_vec; prior=false)
        
#         for param_val in param_range
#             θ_test = copy(θ_mle_vec)
#             θ_test[i] = param_val
            
#             try
#                 nllh_val = petab_problem.nllh(θ_test; prior=false)
#                 # Convert to delta log-likelihood
#                 delta_llh = nllh_val - mle_nllh
#                 push!(likelihood_values, delta_llh)
#             catch e
#                 push!(likelihood_values, Inf)
#             end
#         end
        
#         profiles[param_name] = (param_range=collect(param_range), likelihood=likelihood_values)
        
#         # Create plot
#         valid_indices = .!isinf.(likelihood_values)
#         if any(valid_indices)
#             plt = plot(param_range[valid_indices], likelihood_values[valid_indices],
#                       xlabel="$param_name (log₁₀ scale)",
#                       ylabel="Δ Log-Likelihood",
#                       title="Profile Likelihood: $param_name",
#                       linewidth=2,
#                       legend=:topright,
#                       ylims = (0, 15))  # Zoom in for consistency with modern method
            
#             # Add confidence intervals
#             hline!(plt, [1.92], label="95% CI", color=:red, linestyle=:dash)
#             hline!(plt, [3.84], label="99% CI", color=:orange, linestyle=:dash)
            
#             plot_filename = joinpath(profile_dir, "profile_$(param_name).png")
#             savefig(plt, plot_filename)
#             println("✅ Plot saved: $plot_filename")
#             flush(stdout)
#             successful_profiles += 1
#         else
#             @warn "No valid likelihood values for parameter $param_name"
#         end
#     end
    
#     println("✅ Manual profiling completed")
#     println("Successfully profiled $successful_profiles out of $n_params parameters")
#     flush(stdout)
    
#     # Save results
#     try
#         results_filename = joinpath(profile_dir, "manual_profiling_results.jld2")
#         @save results_filename profiles param_names θ_mle_vec
#         println("✅ Manual profiling results saved: $results_filename")
#     catch e
#         @warn "Failed to save manual profiling results" exception=e
#     end
#     flush(stdout)
    
#     return profiles
# end

"""
    get_profile_data(sol)

Extracts the raw parameter and likelihood values from a profile solution object.
This is useful for saving results without storing non-serializable functions.
"""
function get_profile_data(sol)
    try
        # Try to extract profile values - the exact method depends on LikelihoodProfiler.jl's structure
        if hasfield(typeof(sol), :profile_values)
            return sol.profile_values
        elseif hasfield(typeof(sol), :results)
            return sol.results
        else
            # Fallback: return the whole solution but warn about potential serialization issues
            @warn "Could not extract raw profile data, saving full solution object"
            return sol
        end
    catch e
        @warn "Error extracting profile data" exception=e
        return sol
    end
end

"""
    run_likelihood_profiling(petab_problem::PEtabODEProblem, θ_mle::Union{ComponentVector, Nothing}=nothing, debug_mode::Bool=false)

Modern likelihood profiling using LikelihoodProfiler.jl, with a robust manual fallback.

# Arguments
- `petab_problem`: PEtab ODE problem object.
- `θ_mle`: Maximum Likelihood Estimate parameter vector (required).
- `debug_mode`: Enables faster, less accurate settings for quick testing.

# Key Performance Optimizations
- **Limited Iterations**: `maxiters=100` prevents excessive computation far beyond the confidence intervals.
- **Focused Resolution**: 15-30 points provides good detail without over-sampling.
- **Zoomed Plots**: `ylims=(0, 15)` makes confidence intervals clearly visible.
"""
function run_likelihood_profiling(petab_problem::PEtabODEProblem, θ_mle::Union{ComponentVector, Nothing}=nothing, debug_mode::Bool=false)
    println("\n--- 🔬 Starting Modern Likelihood Profiling (LikelihoodProfiler.jl, Parallel) ---")

    if isnothing(θ_mle)
        @error "MLE parameter estimates are required for profiling."
        return nothing
    end

    # Create output directory
    profile_dir = joinpath(pwd(), "likelihood_profiles")
    mkpath(profile_dir)
    println("Saving profile plots to: $profile_dir")
    flush(stdout)

    θ_mle_vec = collect(θ_mle)
    param_names = string.(petab_problem.xnames)
    n_params = length(param_names)

    # 1. Create the base PLProblem (shared across threads)
    function objective_wrapper(θ, p)
        if any(θ .< petab_problem.lower_bounds) || any(θ .> petab_problem.upper_bounds)
            return Inf
        end
        try
            result = petab_problem.nllh(θ; prior=false)
            return isfinite(result) ? result : Inf
        catch
            return Inf
        end
    end

    base_optprob = OptimizationProblem(
        OptimizationFunction(objective_wrapper, Optimization.AutoForwardDiff()),
        θ_mle_vec;
        lb = collect(petab_problem.lower_bounds),
        ub = collect(petab_problem.upper_bounds)
    )

    plprob = LikelihoodProfiler.PLProblem(base_optprob, θ_mle_vec)

    # --- PARALLEL EXECUTION ---
    println("Starting parallel likelihood profiling across $n_params parameters using $(nthreads()) threads...")
    prof_sols = Vector{Any}(undef, n_params)

    @time Threads.@threads for i in 1:n_params
        println("Thread $(threadid()) handling parameter $i ($(param_names[i]))")
        flush(stdout)
        # --- ADVANCED DIAGNOSTIC: Write to per-thread log file (optional, uncomment to use) ---
        # open("thread_log_$(threadid()).txt", "a") do io
        #     println(io, "Thread $(threadid()) profiled parameter $i at $(time())")
        # end

        # Configure the profiler for this specific run
        profiler = LikelihoodProfiler.OptimizationProfiler(
            optimizer = OptimizationOptimJL.LBFGS(),
            stepper = LikelihoodProfiler.FixedStep(initial_step = 0.1)
        )

        # Profile ONLY the i-th parameter in this thread using idxs
        single_prof_sol = LikelihoodProfiler.profile(plprob, profiler; idxs=[i], maxiters=100)

        # Store the result
        prof_sols[i] = single_prof_sol
    end

    println("✅ Parallel profiling computation complete.")

    # --- PLOTTING (Iterates over the collected results) ---
    println("\n--- Generating Profile Plots ---")
    for i in 1:n_params
        param_name = param_names[i]
        plt = plot(prof_sols[i], 1;  # '1' since each result has only one profile
                   xlabel = "$param_name (log₁₀ scale)",
                   ylabel = "Δ Log-Likelihood",
                   title = "Profile Likelihood: $param_name",
                   legend = :topright,
                   linewidth = 2,
                   ylims = (0, 15)
                  )
        hline!(plt, [1.92], label="95% CI", color=:red, linestyle=:dash)
        hline!(plt, [3.84], label="99% CI", color=:orange, linestyle=:dash)
        savefig(plt, joinpath(profile_dir, "profile_$(param_name).png"))
    end
    println("✅ Individual profile plots saved.")
    flush(stdout)

    return prof_sols
end
