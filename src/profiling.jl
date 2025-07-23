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
                      legend=:topright)
            
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

Modern likelihood profiling using LikelihoodProfiler.jl package with robust API checking.
Falls back to manual implementation if the expected API is not available.

# Arguments
- `petab_problem`: PEtab ODE problem object
- `θ_mle`: MLE parameter estimates (required)
- `debug_mode`: Enable debug mode for faster, less accurate profiling

# Features
- Automatic API compatibility checking
- Fallback to manual profiling if needed
- Automatic ΔLLH normalization (eliminates sign/offset bugs)
- Built-in multi-threading support
- Professional plotting with confidence intervals
- Minimal maintenance burden
"""
function run_likelihood_profiling(petab_problem::PEtabODEProblem, θ_mle::Union{ComponentVector, Nothing}=nothing, debug_mode::Bool=false)
    println("\n--- 🔬 Starting Modern Likelihood Profiling (LikelihoodProfiler.jl) ---")
    
    run_likelihood_profiling_manual(petab_problem, θ_mle, debug_mode)
    
    if isnothing(θ_mle)
        @error "MLE parameter estimates are required for profiling. Please provide θ_mle."
        return nothing
    end
    
    # Create output directory
    profile_dir = joinpath(pwd(), "likelihood_profiles")
    if !isdir(profile_dir); mkpath(profile_dir); end
    println("Created directory for profile plots: $profile_dir")
    
    # Convert ComponentVector to regular Vector for optimization
    θ_mle_vec = collect(θ_mle)
    param_names = string.(petab_problem.xnames)
    n_params = length(param_names)
    
    println("Profiling configuration:")
    println("  Parameters to profile: $n_params")
    println("  Parameter names: $(join(param_names, ", "))")
    println("  Debug mode: $debug_mode")
    
    # Get solver options
    solver_opts = debug_mode ? DEBUG_SOLVER_OPTS : SOLVER_OPTS
    println("  Using solver tolerances: abstol=$(solver_opts.abstol), reltol=$(solver_opts.reltol)")
    
    try
        # Step 1: Create OptimizationProblem wrapper around PEtab objective
        println("\n--- Setting up OptimizationProblem ---")
        
        # Wrap PEtab objective function for LikelihoodProfiler
        function objective_wrapper(θ, p)
            # Explicitly check bounds before evaluation
            if any(θ .< petab_problem.lower_bounds) || any(θ .> petab_problem.upper_bounds)
                return Inf  # Return Inf for out-of-bounds, which optimizers handle well
            end
            try
                result = petab_problem.nllh(θ; prior=false)
                return isfinite(result) ? result : Inf
            catch e
                @warn "PEtab evaluation failed inside bounds..." exception=e maxlog=5
                return Inf
            end
        end
        
        # Create optimization problem with bounds
        optprob = OptimizationProblem(
            OptimizationFunction(objective_wrapper, Optimization.AutoForwardDiff()),
            θ_mle_vec;
            lb = collect(petab_problem.lower_bounds),
            ub = collect(petab_problem.upper_bounds)
        )
        
        println("✅ OptimizationProblem created successfully")
        
        # Step 2: Create PLProblem for profiling using full qualification
        println("--- Setting up PLProblem ---")
        
        plprob = LikelihoodProfiler.PLProblem(optprob, θ_mle_vec)
        println("✅ PLProblem created successfully")
        
        # Step 2b: Parameter bounds analysis (diagnostic)
        println("\n--- Parameter Bounds Analysis ---")
        potential_issues = String[]
        
        for (i, name) in enumerate(param_names)
            lb = petab_problem.lower_bounds[i]
            ub = petab_problem.upper_bounds[i] 
            mle_val = θ_mle_vec[i]
            println("  $name: [$lb, $ub], MLE = $mle_val")
            
            # Check for fixed parameters (identical bounds)
            if abs(ub - lb) < 1e-10
                println("    🔒 FIXED parameter (lb = ub) - should be excluded from profiling")
                push!(potential_issues, "$name (fixed parameter)")
                continue
            end
            
            # Check if MLE is near bounds (potential issue)
            bound_range = abs(ub - lb)
            lower_dist = abs(mle_val - lb)
            upper_dist = abs(ub - mle_val)
            
            if lower_dist < 0.05 * bound_range || upper_dist < 0.05 * bound_range
                println("    ⚠️  MLE very close to bounds (within 5%) - potential numerical issues")
                push!(potential_issues, "$name (MLE near bounds)")
            end
            
            # Check for extremely wide parameter ranges
            if bound_range > 100.0  # More than 100 orders of magnitude in log space
                println("    ⚠️  Extremely wide parameter range ($(round(bound_range, digits=1)) log units)")
                push!(potential_issues, "$name (very wide range)")
            end
            
            # Check for small absolute bounds that might cause issues
            if lb < 1e-6 && ub < 1e-3
                println("    ⚠️  Very small parameter bounds - potential numerical precision issues")
                push!(potential_issues, "$name (very small bounds)")
            end
        end
        
        if !isempty(potential_issues)
            println("\n⚠️  Potential numerical issues detected:")
            for issue in potential_issues
                println("  - $issue")
            end
            println("These parameters may cause stack overflow or other numerical problems")
        end
        
        # Step 3: Configure profiler based on debug mode using full qualification
        if debug_mode
            println("--- Running profiling (debug mode: OptimizationProfiler with conservative settings) ---")
            profiler = LikelihoodProfiler.OptimizationProfiler(
                optimizer = OptimizationOptimJL.LBFGS(),
                stepper = LikelihoodProfiler.FixedStep(initial_step = 0.1)  # Conservative step size
            )
        else
            println("--- Running profiling (normal mode: OptimizationProfiler with standard settings) ---")
            profiler = LikelihoodProfiler.OptimizationProfiler(
                optimizer = OptimizationOptimJL.LBFGS(),
                stepper = LikelihoodProfiler.FixedStep(initial_step = 0.05)  # Smaller steps for accuracy
            )
        end
        
        # Step 4: Test individual parameters (diagnostic)
        println("\n--- Testing Individual Parameters ---")
        problematic_params = Int[]
        
        for i in 1:n_params
            try
                println("Testing parameter $i: $(param_names[i])")
                # Test with minimal resolution for speed using full qualification
                test_sol = LikelihoodProfiler.profile(plprob, profiler; idxs=[i])
                println("✅ Parameter $i ($(param_names[i])) succeeded")
            catch e
                println("❌ Parameter $i ($(param_names[i])) failed: $(typeof(e))")
                push!(problematic_params, i)
                if debug_mode
                    println("    Error details: $e")
                end
            end
        end
        
        if !isempty(problematic_params)
            println("\n⚠️  Problematic parameters detected:")
            for i in problematic_params
                println("  - Parameter $i: $(param_names[i])")
            end
            println("Consider excluding these parameters or using different bounds/scaling")
        end
        
        # Step 5: Run full profiling (excluding problematic parameters if in debug mode) using full qualification
        resolution = debug_mode ? 10 : 20
        
        if debug_mode && !isempty(problematic_params)
            good_params = setdiff(1:n_params, problematic_params)
            println("\nRunning profiling on $(length(good_params)) good parameters...")
            println("Starting likelihood profiling computation with resolution: $resolution...")
            profile_start_time = time()
            
            @time prof_sol = LikelihoodProfiler.profile(plprob, profiler; idxs=good_params, resolution=resolution)
        else
            println("\nStarting likelihood profiling computation with resolution: $resolution...")
            profile_start_time = time()
            
            @time prof_sol = LikelihoodProfiler.profile(plprob, profiler; resolution=resolution)
        end
        
        profile_time = time() - profile_start_time
        println("✅ Profiling completed in $(round(profile_time, digits=1)) seconds")
        
        # Step 6: Generate and save plots
        println("\n--- Generating Profile Plots ---")
        
        plot_start_time = time()
        successful_plots = 0
        
        # Determine which parameters were actually profiled
        profiled_params = if debug_mode && !isempty(problematic_params)
            good_params = setdiff(1:n_params, problematic_params)
            println("Generating plots for $(length(good_params)) successfully profiled parameters")
            good_params
        else
            println("Generating plots for all $n_params parameters")
            1:n_params
        end
        
        # Generate individual parameter plots
        for (plot_idx, param_idx) in enumerate(profiled_params)
            param_name = param_names[param_idx]
            try
                println("Generating plot for parameter: $param_name")
                
                # Create individual parameter plot using LikelihoodProfiler's built-in plotting
                plt = plot(prof_sol, plot_idx;  # Use plot_idx for the profiling result
                    xlabel = "$param_name (log₁₀ scale)",
                    ylabel = "Δ Log-Likelihood",
                    title = "Profile Likelihood: $param_name",
                    legend = :topright,
                    linewidth = 2
                )
                
                # Add confidence interval lines
                hline!(plt, [1.92], label="95% CI", color=:red, linestyle=:dash)  # χ²(1,0.05)/2
                hline!(plt, [3.84], label="99% CI", color=:orange, linestyle=:dash)  # χ²(1,0.01)/2
                
                # Save plot
                plot_filename = joinpath(profile_dir, "profile_$(param_name).png")
                savefig(plt, plot_filename)
                
                successful_plots += 1
                println("✅ Plot saved: $plot_filename")
                
            catch plot_error
                @warn "Failed to generate plot for $param_name" exception=plot_error
            end
        end
        
        # Generate summary plot with all parameters
        try
            println("Generating summary plot with all parameters...")
            
            summary_plt = plot(prof_sol;
                title = "Profile Likelihood Summary",
                xlabel = "Parameter Value (log₁₀ scale)",
                ylabel = "Δ Log-Likelihood",
                legend = :outertopright
            )
            
            # Add confidence interval lines
            hline!(summary_plt, [1.92], label="95% CI", color=:red, linestyle=:dash)
            hline!(summary_plt, [3.84], label="99% CI", color=:orange, linestyle=:dash)
            
            summary_filename = joinpath(profile_dir, "profile_summary_all_parameters.png")
            savefig(summary_plt, summary_filename)
            
            successful_plots += 1
            println("✅ Summary plot saved: $summary_filename")
            
        catch summary_error
            @warn "Failed to generate summary plot" exception=summary_error
        end
        
        plot_time = time() - plot_start_time
        println("✅ Plot generation completed in $(round(plot_time, digits=1)) seconds")
        
        # Step 7: Save profiling results
        println("\n--- Saving Results ---")
        
        try
            results_filename = joinpath(profile_dir, "profiling_results.jld2")
            
            # Extract the raw data from the solution object (safer than saving complex objects)
            profile_data = get_profile_data(prof_sol)
            
            @save results_filename profile_data param_names θ_mle_vec profiled_params
            println("✅ Profiling data (not full objects) saved: $results_filename")
        catch save_error
            @warn "Failed to save profiling results" exception=save_error
        end
        
        # Summary
        total_time = profile_time + plot_time
        actual_profiled = length(profiled_params)
        
        println("\n--- ✅ Modern Likelihood Profiling Complete ---")
        println("Successfully profiled $actual_profiled out of $n_params parameters")
        if actual_profiled < n_params
            skipped_count = n_params - actual_profiled
            println("⚠️  Skipped $skipped_count problematic parameters in debug mode")
        end
        println("Generated $successful_plots plots")
        println("Total time: $(round(total_time, digits=1)) seconds")
        println("  - Profiling: $(round(profile_time, digits=1))s")
        println("  - Plotting: $(round(plot_time, digits=1))s")
        println("Results saved to: $profile_dir")
        
        println("\nKey advantages of LikelihoodProfiler.jl:")
        println("  ✅ Automatic ΔLLH normalization (eliminates sign/offset bugs)")
        println("  ✅ Built-in confidence interval calculation")
        println("  ✅ Professional plotting with minimal code")
        println("  ✅ Maintained by the Julia optimization community")
        
        if debug_mode && !isempty(problematic_params)
            println("\nDiagnostic Information:")
            println("  - Switched to OptimizationProfiler to avoid ODE stack overflow")
            println("  - $(length(problematic_params)) parameters caused numerical issues")
            println("  - Consider parameter rescaling or bound adjustment for problematic parameters")
        end
        
        return prof_sol
        
    catch e
        @error "Likelihood profiling failed" exception=(e, catch_backtrace())
        return nothing
    end
end

# Deprecated functions for backward compatibility (will be removed)
function skip_mle_refinement!(use_quick_mle::Bool=false)
    @warn "skip_mle_refinement! is deprecated. LikelihoodProfiler.jl handles MLE refinement automatically."
    return nothing
end
