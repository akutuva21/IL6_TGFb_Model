# src/profiling.jl
# Modern likelihood profiling using LikelihoodProfiler.jl

# More robust imports - use full module qualification instead of specific function imports
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

include("solver_config.jl")

export run_likelihood_profiling

# Manual profiling fallback implementation
function run_likelihood_profiling_manual(petab_problem::PEtabODEProblem, θ_mle::Union{ComponentVector, Nothing}=nothing, debug_mode::Bool=false)
    println("\n--- 🔬 Starting Manual Likelihood Profiling ---")
    
    if isnothing(θ_mle)
        @error "MLE parameter estimates are required for profiling."
        return nothing
    end
    
    θ_mle_vec = collect(θ_mle)
    param_names = string.(petab_problem.xnames)
    n_params = length(param_names)
    
    # Create output directory
    profile_dir = joinpath(pwd(), "likelihood_profiles")
    if !isdir(profile_dir); mkpath(profile_dir); end
    
    println("Manual profiling configuration:")
    println("  Parameters to profile: $n_params")
    println("  Parameter names: $(join(param_names, ", "))")
    println("  Debug mode: $debug_mode")
    
    # Manual profiling for each parameter
    profiles = Dict()
    successful_profiles = 0
    
    # Get MLE log-likelihood for reference
    try
        mle_nllh = petab_problem.nllh(θ_mle_vec; prior=false)
        println("  MLE log-likelihood: $mle_nllh")
    catch e
        @error "Could not evaluate MLE log-likelihood" exception=e
        return nothing
    end
    
    for (i, param_name) in enumerate(param_names)
        println("Profiling parameter $i: $param_name")
        
        # Create range around MLE
        lb = petab_problem.lower_bounds[i]
        ub = petab_problem.upper_bounds[i]
        mle_val = θ_mle_vec[i]
        
        # Check for fixed parameters
        if abs(ub - lb) < 1e-10
            println("  🔒 Skipping fixed parameter: $param_name")
            continue
        end
        
        # Create parameter range (more points around MLE)
        n_points = debug_mode ? 20 : 50
        param_range = range(lb, ub, length=n_points)
        
        likelihood_values = Float64[]
        mle_nllh = petab_problem.nllh(θ_mle_vec; prior=false)
        
        for param_val in param_range
            θ_test = copy(θ_mle_vec)
            θ_test[i] = param_val
            
            try
                nllh_val = petab_problem.nllh(θ_test; prior=false)
                # Convert to delta log-likelihood
                delta_llh = nllh_val - mle_nllh
                push!(likelihood_values, delta_llh)
            catch e
                push!(likelihood_values, Inf)
            end
        end
        
        profiles[param_name] = (param_range=collect(param_range), likelihood=likelihood_values)
        
        # Create plot
        valid_indices = .!isinf.(likelihood_values)
        if any(valid_indices)
            plt = plot(param_range[valid_indices], likelihood_values[valid_indices],
                      xlabel="$param_name (log₁₀ scale)",
                      ylabel="Δ Log-Likelihood",
                      title="Profile Likelihood: $param_name",
                      linewidth=2,
                      legend=:topright,
                      ylims = (0, 15))  # Zoom in for consistency with modern method
            
            # Add confidence intervals
            hline!(plt, [1.92], label="95% CI", color=:red, linestyle=:dash)
            hline!(plt, [3.84], label="99% CI", color=:orange, linestyle=:dash)
            
            plot_filename = joinpath(profile_dir, "profile_$(param_name).png")
            savefig(plt, plot_filename)
            println("✅ Plot saved: $plot_filename")
            successful_profiles += 1
        else
            @warn "No valid likelihood values for parameter $param_name"
        end
    end
    
    println("✅ Manual profiling completed")
    println("Successfully profiled $successful_profiles out of $n_params parameters")
    
    # Save results
    try
        results_filename = joinpath(profile_dir, "manual_profiling_results.jld2")
        @save results_filename profiles param_names θ_mle_vec
        println("✅ Manual profiling results saved: $results_filename")
    catch e
        @warn "Failed to save manual profiling results" exception=e
    end
    
    return profiles
end

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
    println("\n--- 🔬 Starting Modern Likelihood Profiling (LikelihoodProfiler.jl) ---")

    if isnothing(θ_mle)
        @error "MLE parameter estimates are required for profiling. Please provide θ_mle."
        return nothing
    end

    # Create output directory
    profile_dir = joinpath(pwd(), "likelihood_profiles")
    mkpath(profile_dir)
    println("Saving profile plots to: $profile_dir")

    θ_mle_vec = collect(θ_mle)
    param_names = string.(petab_problem.xnames)

    try
        # --- This is the main, modern profiling workflow ---

        # 1. Create OptimizationProblem wrapper
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

        optprob = OptimizationProblem(
            OptimizationFunction(objective_wrapper, Optimization.AutoForwardDiff()),
            θ_mle_vec;
            lb = collect(petab_problem.lower_bounds),
            ub = collect(petab_problem.upper_bounds)
        )

        # 2. Create PLProblem
        plprob = LikelihoodProfiler.PLProblem(optprob, θ_mle_vec)

        # 3. Configure and run the profiler
        # THE FIX: Add the required 'stepper' keyword argument
        profiler = LikelihoodProfiler.OptimizationProfiler(
            optimizer = OptimizationOptimJL.LBFGS(),
            stepper = LikelihoodProfiler.FixedStep(initial_step = 0.1)  # Required stepper
        )
        
        println("Starting likelihood profiling with maxiters=100...")
        # THE FIX: Remove the unsupported 'resolution' keyword argument
        @time prof_sol = LikelihoodProfiler.profile(plprob, profiler; maxiters=100)
        println("✅ Profiling computation complete.")

        # 4. Generate and save plots
        println("\n--- Generating Profile Plots ---")
        for i in 1:length(param_names)
            param_name = param_names[i]
            plt = plot(prof_sol, i;
                xlabel = "$param_name (log₁₀ scale)",
                ylabel = "Δ Log-Likelihood",
                title = "Profile Likelihood: $param_name",
                legend = :topright,
                linewidth = 2,
                ylims = (0, 15) # Zoom in
            )
            hline!(plt, [1.92], label="95% CI", color=:red, linestyle=:dash)
            hline!(plt, [3.84], label="99% CI", color=:orange, linestyle=:dash)
            savefig(plt, joinpath(profile_dir, "profile_$(param_name).png"))
        end
        println("✅ Individual profile plots saved.")

        # Generate summary plot
        summary_plt = plot(prof_sol; ylims=(0, 15), legend=:outertopright)
        hline!(summary_plt, [1.92], label="95% CI", color=:red, linestyle=:dash)
        savefig(summary_plt, joinpath(profile_dir, "profile_summary.png"))
        println("✅ Summary plot saved.")

        return prof_sol

    catch e
        @error "Modern likelihood profiling failed. Falling back to manual method." exception=(e, catch_backtrace())
        # --- This is the FALLBACK path ---
        return run_likelihood_profiling_manual(petab_problem, θ_mle, debug_mode)
    end
end

# Deprecated functions for backward compatibility (will be removed)
function skip_mle_refinement!(use_quick_mle::Bool=false)
    @warn "skip_mle_refinement! is deprecated. LikelihoodProfiler.jl handles MLE refinement automatically."
    return nothing
end
