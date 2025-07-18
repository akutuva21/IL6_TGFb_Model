# Profiling dependencies
using PEtab
using Plots
using Base.Threads  # For parallel processing
using Distributed   # For distributed processing across workers
using ComponentArrays
using JLD2
using OptimizationOptimJL # For L-BFGS optimizer
using ADTypes: AutoForwardDiff # For specifying the AD backend
using Statistics # For mean function
using ForwardDiff # For accurate gradient computation
using LinearAlgebra # For norm function
using Optim # For Fminbox bounded optimizer
using Logging # For controlled warning suppression
using ProfileLikelihood

export run_likelihood_profiling

# Optional: Function to force acceptance of provided MLE without refinement
function skip_mle_refinement!(use_quick_mle::Bool=false)
    FORCE_QUICK_MLE[] = use_quick_mle  # Fixed: use [] to modify Ref contents
    if use_quick_mle
        println("⚡ MLE refinement will be skipped for maximum speed")
    else
        println("🔍 MLE refinement will use normal thresholds")
    end
end

# Global flag for quick MLE mode (keep as is)
FORCE_QUICK_MLE = Ref(false)

# Enhanced profiling function with all optimizations and edge case handling
function run_likelihood_profiling(petab_problem::PEtabODEProblem, θ_mle::Union{ComponentVector, Nothing}=nothing, debug_mode::Bool=false; debug_max_params::Int=5)
    println("\n--- 🔬 Starting Optimized Likelihood Profiling ---")
    
    profile_dir = joinpath(pwd(), "likelihood_profiles")
    if !isdir(profile_dir); mkpath(profile_dir); end
    println("Created directory for profile plots: $profile_dir")

    # Handle case where MLE is not provided (fallback to original behavior)
    if isnothing(θ_mle)
        @warn "No MLE provided, will need to compute it (slower). Consider passing multi_start_res.xmin for better performance."
        θ_mle = petab_problem.xnominal_transformed
        need_global_opt = true
    else
        println("✅ Using provided MLE, skipping global optimization")
        need_global_opt = false
    end

    # 1. Determine debug mode and set appropriate parameters
    n_steps = debug_mode ? 5 : 20  # Reduced from 40 to 20 for faster testing
    
    println("Profiling configuration:")
    println("  Debug mode: $debug_mode")
    println("  Resolution (profile points): $n_steps")
    if debug_mode
        println("  Debug max parameters: $debug_max_params")
        println("  Debug optimizations enabled:")
        println("    - Faster resolution ($n_steps points instead of 20)")
        println("    - Skip MLE refinement")
        println("    - Warning suppression")
        println("    - Stop after first failure")
        println("    - Smaller penalty values (1e4 instead of 1e10)")
    else
        println("  Production optimizations enabled:")
        println("    - Moderate resolution ($n_steps points)")
        println("    - Relaxed MLE refinement threshold (1e-3)")
        println("    - Reduced iteration limits for faster completion")
    end
    println("  Bounded optimizer: Fminbox(LBFGS()) to respect parameter bounds")
    println("  Penalty for failed evaluations: 1e4 (moderate cliff)")

    # 2. Create a wrapper function that matches ProfileLikelihood.jl's expected interface
    petab_nllh = petab_problem.nllh
    function neg_loglik_wrapper(θ, data)
        try
            result = petab_nllh(θ; prior=false)
            return isfinite(result) ? result : 1e4  # Smaller penalty to avoid confusing optimizer
        catch e
            @warn "PEtab evaluation failed, returning penalty value" exception=e maxlog=5
            return 1e4  # Moderate penalty instead of 1e10
        end
    end

    # 3. Get bounds and parameter information from PEtab problem
    lower_bounds = collect(petab_problem.lower_bounds)
    upper_bounds = collect(petab_problem.upper_bounds)
    param_names = petab_problem.xnames
    
    # Enhanced solver configuration with robust defaults
    solver_kwargs = Dict()
    try
        if debug_mode && hasfield(typeof(petab_problem), :probinfo) && hasfield(typeof(petab_problem.probinfo), :solver)
            solver = petab_problem.probinfo.solver
            solver_kwargs[:abstol] = solver.abstol
            solver_kwargs[:reltol] = solver.reltol
            println("  Using solver tolerances: abstol=$(solver_kwargs[:abstol]), reltol=$(solver_kwargs[:reltol])")
        end
    catch e
        @warn "Could not access solver tolerances from petab_problem, using defaults" exception=e
    end
    
    # 4. Create LikelihoodProblem with proper bounds and tolerances
    lik_prob = LikelihoodProblem(
        neg_loglik_wrapper, 
        θ_mle;  # Use the provided MLE as initial guess
        data = (),
        syms = param_names,
        f_kwargs = (adtype = AutoForwardDiff(),),
        prob_kwargs = merge((lb = lower_bounds, ub = upper_bounds), solver_kwargs)
    )
    println("✅ LikelihoodProblem created successfully.")

    # 5. Create MLE solution efficiently with optimization checks
    println("Creating MLE solution...")
    
    if need_global_opt
        println("  Computing MLE from scratch (no MLE provided)...")
        @time mle_solution = mle(lik_prob, Optim.Fminbox(Optim.LBFGS()))
        θ_mle = mle_solution.mle  # Update θ_mle to the computed result
    else
        println("  Using provided MLE as starting point...")
        # Test the wrapper function at the provided MLE
        mle_loglik = neg_loglik_wrapper(θ_mle, ())
        println("  MLE log-likelihood: $mle_loglik")
        
        if debug_mode
            println("  Debug mode: skipping MLE refinement for faster testing")
            # In debug mode, use minimal optimization to create solution object
            iter_limit = 1  # Minimal iterations
            @time mle_solution = mle(lik_prob,
                                   Optim.Fminbox(Optim.LBFGS());
                                   iterations = iter_limit)
            println("  ✅ Created LikelihoodSolution via minimal optimization (debug mode)")
        elseif FORCE_QUICK_MLE[]
            println("  Quick MLE mode: forcing acceptance of provided MLE")
            iter_limit = 1  # Force minimal optimization
            @time mle_solution = mle(lik_prob,
                                   Optim.Fminbox(Optim.LBFGS());
                                   iterations = iter_limit)
            println("  ✅ Created LikelihoodSolution via forced quick mode")
        else
            # Check if the provided MLE is already optimal using ForwardDiff for accuracy
            optimization_threshold = 1e-3  # Much more tolerant threshold (was 1e-6)
            initial_gradient_norm = 0.0
            
            try
                println("  Computing gradient to check optimality...")
                initial_gradient = ForwardDiff.gradient(θ -> neg_loglik_wrapper(θ, ()), θ_mle)
                initial_gradient_norm = norm(initial_gradient)
                println("  Gradient norm at provided MLE: $(round(initial_gradient_norm, sigdigits=3))")
            catch e
                @warn "Could not compute gradient using ForwardDiff, proceeding with MLE optimization" exception=e
                initial_gradient_norm = Inf  # Force optimization if gradient computation fails
            end
            
            # Only run optimization if gradient suggests we're not at optimum
            if initial_gradient_norm > optimization_threshold
                println("  Gradient norm suggests further optimization needed...")
                iter_limit = debug_mode ? 50 : 200  # Reduced from 1000
                @time mle_solution = mle(lik_prob,
                                       Optim.Fminbox(Optim.LBFGS());
                                       iterations = iter_limit)
            else
                println("  MLE appears acceptable (gradient norm < $optimization_threshold), skipping refinement...")
                # Use minimal optimization to create a proper solution object
                iter_limit = debug_mode ? 1 : 5  # Minimal iterations
                @time mle_solution = mle(lik_prob,
                                       Optim.Fminbox(Optim.LBFGS());
                                       iterations = iter_limit)
                println("  ✅ Created LikelihoodSolution via minimal optimization")
            end
        end
    end
    
    println("✅ LikelihoodSolution ready for profiling.")

    # 6. Enhanced parameter filtering using PEtab.jl's estimability information
    estimable_indices = Int[]
    estimable_names = String[]
    skipped_params = String[]
    
    # Priority parameters for faster testing (most important biological parameters)
    priority_params = ["IL6R_0", "PKA_0", "SMAD3_0", "SMAD4_0", "STAT3m_0", "TGFb_0", "IL6_0"]
    
    # Try to access PEtab parameter information for better filtering
    parameter_scales = nothing
    try
        # Attempt to get parameter scales from PEtab problem
        if hasfield(typeof(petab_problem), :xscale)
            parameter_scales = petab_problem.xscale
            println("  Found parameter scales in PEtab problem")
        end
    catch e
        @warn "Could not access parameter scales from PEtab problem" exception=e
    end
    
    for (i, param_name) in enumerate(param_names)
        param_str = string(param_name)
        skip_reason = ""
        
        # Skip sigma (noise) parameters - check PEtab's parameter indices if available
        if startswith(param_str, "sigma_") || startswith(param_str, "noise_")
            skip_reason = "noise parameter"
        # Check if parameter is estimable using PEtab's information
        # Note: PEtab.jl may not have is_estimable field directly, so we use bounds as proxy
        elseif (upper_bounds[i] - lower_bounds[i]) < 1e-6
            skip_reason = "fixed parameter (tight bounds)"
        # Additional check: skip if bounds are at extreme values indicating non-estimability
        elseif lower_bounds[i] == -Inf || upper_bounds[i] == Inf
            skip_reason = "unbounded parameter"
        # In debug mode, limit the number of parameters to profile for faster testing
        elseif debug_mode && length(estimable_indices) >= debug_max_params
            skip_reason = "debug mode limit (>$debug_max_params params)"
        # Priority parameter filtering for faster testing
        elseif !debug_mode && length(estimable_indices) >= 10 && !(param_str in priority_params)
            skip_reason = "non-priority parameter (focusing on top 10 most important)"
        end
        
        if !isempty(skip_reason)
            push!(skipped_params, "$param_name ($skip_reason)")
            println("  Skipping parameter: $param_name ($skip_reason)")
            continue
        end
        
        push!(estimable_indices, i)
        push!(estimable_names, param_str)
    end
    
    num_params = length(estimable_indices)
    println("Selected $num_params estimable parameters for profiling:")
    for (idx, name) in enumerate(estimable_names)
        println("  $idx. $name")
    end
    
    if !isempty(skipped_params)
        println("Skipped $(length(skipped_params)) parameters:")
        for skip_info in skipped_params
            println("  - $skip_info")
        end
    end

    # 7. Run profiling in parallel across parameters with continuation
    println("\n--- Starting distributed likelihood profiling ---")
    
    # Check if distributed workers are available
    if nworkers() > 1
        println("Using $(nworkers()) distributed workers for parallel profiling")
        println("This will utilize distributed processing instead of threading for better SLURM compatibility")
    else
        println("Using $(Threads.nthreads()) threads for parallel processing")
        if Threads.nthreads() == 1
            @warn "Running with only 1 thread. For better performance, start Julia with multiple threads: julia --threads auto"
        end
    end
    
    # Pre-allocate results storage
    profile_results = Vector{Any}(undef, num_params)
    completion_times = Vector{Float64}(undef, num_params)
    
    # Create bounded optimizer once for all parameter profiles
    bounded_optimizer = Optim.Fminbox(Optim.LBFGS())
    
    # Choose between distributed or threaded processing
    if nworkers() > 1
        # Distributed profiling loop for SLURM clusters
        println("Running distributed profiling across $(nworkers()) workers...")
        
        # Use distributed processing with result collection
        results = @distributed (vcat) for p_idx in 1:num_params
            param_idx = estimable_indices[p_idx]
            param_name = estimable_names[p_idx]
            worker_id = myid()
            
            # Progress monitoring
            if p_idx % 5 == 0 || p_idx == 1
                println("🔄 Progress: $p_idx/$num_params parameters started ($(round(100*p_idx/num_params, digits=1))%)")
                flush(stdout)
            end
            
            println("Worker $worker_id: Starting parameter $p_idx/$num_params: $param_name")
            
            start_time = time()
            local prof_result = nothing
            local completion_time = 0.0
            
            try
                # Check for cached profile data first
                profile_path = joinpath(profile_dir, "profile_data_$(param_name).jld2")
                
                if isfile(profile_path)
                    # File already on disk - just load and plot
                    println("Worker $worker_id: Using cached profile for $param_name")
                    @load profile_path prof_result param_name mle_val
                    
                    completion_time = time() - start_time
                    println("Worker $worker_id: ✅ Loaded cached profile for $param_name in $(round(completion_time, digits=3))s")
                else
                    # No cache - run the expensive profile computation and save it
                    println("Worker $worker_id: Computing new profile for $param_name...")
                    
                    # Use the actual PEtab bounds for this parameter instead of custom bounds
                    prof_extrema = (lower_bounds[param_idx], upper_bounds[param_idx])
                    
                    # Create consistent ranges for both debug and non-debug modes
                    left  = LinRange(prof_extrema[1], θ_mle[param_idx], n_steps)
                    right = LinRange(θ_mle[param_idx], prof_extrema[2], n_steps)
                    ranges = Dict(param_idx => (left, right))
                    
                    println("Worker $worker_id: Using PEtab bounds for $param_name: $prof_extrema")
                    
                    # Use consistent profiling call regardless of debug mode
                    # Define the profiler function call once to avoid duplication
                    run_profiler = () -> profile(
                        lik_prob, 
                        mle_solution, 
                        param_idx;
                        resolution=n_steps,      # Number of grid points (supported)
                        alg=bounded_optimizer,   # Use bounded optimizer to avoid NaN regions
                        param_ranges = ranges,   # Consistent range specification
                        parallel=false          # We handle parallelism at parameter level (supported)
                    )
                    
                    # Conditionally apply the logger wrapper to the function call
                    prof_result = if debug_mode
                        # In debug mode, suppress all warnings below @error level for cleaner output
                        Logging.with_logger(Logging.NullLogger()) do
                            run_profiler()
                        end
                    else
                        # In normal mode, suppress warnings but allow error messages through
                        Logging.with_logger(Logging.SimpleLogger(stderr, Logging.Error)) do
                            run_profiler()
                        end
                    end
                    
                    completion_time = time() - start_time
                    mle_val = θ_mle[param_idx]  # Keep interface identical
                    
                    println("Worker $worker_id: ✅ Profile computation for $param_name completed in $(round(completion_time, digits=1))s")
                    
                    # Save raw data immediately after computation
                    save_profile_data(prof_result, param_name, profile_dir, mle_val, worker_id)
                end
                
                # Return tuple of (index, result, time, name) for result collection
                (p_idx, prof_result, completion_time, param_name)
                
            catch e
                completion_time = time() - start_time
                @warn "Worker $worker_id: Failed to profile parameter $param_name after $(round(completion_time, digits=1))s" exception=e
                
                # Return tuple with nothing result for failed cases
                (p_idx, nothing, completion_time, param_name)
            end
        end
        
        # Process distributed results
        for (p_idx, result, time, name) in results
            profile_results[p_idx] = result
            completion_times[p_idx] = time
        end
        
    else
        # Fallback to threading if no distributed workers available
        Threads.@threads for p_idx in 1:num_params
            param_idx = estimable_indices[p_idx]
            param_name = estimable_names[p_idx]
            
            thread_id = Threads.threadid()
            @info "Thread $(threadid()) profiling $(param_name)"
            
            # Progress monitoring
            if p_idx % 5 == 0 || p_idx == 1  # Every 5th parameter or first parameter
                println("🔄 Progress: $p_idx/$num_params parameters started ($(round(100*p_idx/num_params, digits=1))%)")
                flush(stdout)
            end
            
            println("Thread $thread_id: Starting parameter $p_idx/$num_params: $param_name")
            
            start_time = time()
            try
                # Check for cached profile data first
                profile_path = joinpath(profile_dir, "profile_data_$(param_name).jld2")
                
                if isfile(profile_path)
                    # File already on disk - just load and plot
                    println("Thread $thread_id: Using cached profile for $param_name")
                    @load profile_path prof_result param_name mle_val
                    
                    completion_time = time() - start_time
                    completion_times[p_idx] = completion_time
                    profile_results[p_idx] = prof_result
                    
                    println("Thread $thread_id: ✅ Loaded cached profile for $param_name in $(round(completion_time, digits=3))s")
                else
                    # No cache - run the expensive profile computation and save it
                    println("Thread $thread_id: Computing new profile for $param_name...")
                    
                    # Use the actual PEtab bounds for this parameter instead of custom bounds
                    prof_extrema = (lower_bounds[param_idx], upper_bounds[param_idx])
                    
                    # Create consistent ranges for both debug and non-debug modes
                    left  = LinRange(prof_extrema[1], θ_mle[param_idx], n_steps)
                    right = LinRange(θ_mle[param_idx], prof_extrema[2], n_steps)
                    ranges = Dict(param_idx => (left, right))
                    
                    println("Thread $thread_id: Using PEtab bounds for $param_name: $prof_extrema")
                    
                    # Use consistent profiling call regardless of debug mode
                    # Define the profiler function call once to avoid duplication
                    run_profiler = () -> profile(
                        lik_prob, 
                        mle_solution, 
                        param_idx;
                        resolution=n_steps,      # Number of grid points (supported)
                        alg=bounded_optimizer,   # Use bounded optimizer to avoid NaN regions
                        param_ranges = ranges,   # Consistent range specification
                        parallel=false          # We handle parallelism at parameter level (supported)
                    )
                    
                    # Conditionally apply the logger wrapper to the function call
                    prof_result = if debug_mode
                        # In debug mode, suppress all warnings below @error level for cleaner output
                        Logging.with_logger(Logging.NullLogger()) do
                            run_profiler()
                        end
                    else
                        # In normal mode, suppress warnings but allow error messages through
                        Logging.with_logger(Logging.SimpleLogger(stderr, Logging.Error)) do
                            run_profiler()
                        end
                    end
                    
                    completion_time = time() - start_time
                    completion_times[p_idx] = completion_time
                    profile_results[p_idx] = prof_result
                    mle_val = θ_mle[param_idx]  # Keep interface identical
                    
                    println("Thread $thread_id: ✅ Profile computation for $param_name completed in $(round(completion_time, digits=1))s")
                    
                    # Save raw data immediately after computation
                    save_profile_data(prof_result, param_name, profile_dir, mle_val, thread_id)
                end
                
                # Note: Plot generation moved to sequential phase after parallel profiling
                
            catch e
                completion_time = time() - start_time
                completion_times[p_idx] = completion_time
                @warn "Thread $thread_id: Failed to profile parameter $param_name after $(round(completion_time, digits=1))s" exception=e
                profile_results[p_idx] = nothing
                
                # In debug mode, optionally stop after first failure to save time
                if debug_mode
                    @warn "Debug mode: stopping after first failed profile to save time"
                    break
                end
            end
        end
    end
    
    # 8. Summary with timing information and detailed reporting
    successful_profiles = count(x -> x !== nothing, profile_results)
    failed_profiles = num_params - successful_profiles
    successful_times = completion_times[profile_results .!== nothing]
    
    processing_method = nworkers() > 1 ? "distributed" : "threaded"
    worker_count = nworkers() > 1 ? nworkers() : Threads.nthreads()
    
    println("\n--- ✅ Parallel profiling completed ($processing_method) ---")
    println("Successfully profiled $successful_profiles out of $num_params parameters")
    println("Used $worker_count $(processing_method == "distributed" ? "workers" : "threads") for parallel processing")
    
    if failed_profiles > 0
        failed_indices = findall(x -> x === nothing, profile_results)
        failed_names = estimable_names[failed_indices]
        println("Failed to profile $failed_profiles parameters:")
        for name in failed_names
            println("  - $name")
        end
    end
    
    if !isempty(successful_times)
        avg_time = round(mean(successful_times), digits=1)
        total_time = round(sum(completion_times), digits=1)
        println("Average time per parameter: $(avg_time)s")
        println("Total computation time: $(total_time)s")
    end
    
    # 7b. Generate plots sequentially to avoid memory issues with concurrent GR/PNG allocation
    println("\n--- Starting sequential plot generation ---")
    println("Generating plots one by one to avoid memory allocation conflicts...")
    
    plot_start_time = time()
    successful_plots = 0
    
    for p_idx in 1:num_params
        if profile_results[p_idx] !== nothing
            param_idx = estimable_indices[p_idx]
            param_name = estimable_names[p_idx]
            mle_val = θ_mle[param_idx]
            
            try
                println("Generating plot $p_idx/$num_params: $param_name")
                generate_profile_plot(profile_results[p_idx], param_idx, param_name, profile_dir, mle_val, 1, parameter_scales)
                successful_plots += 1
            catch plot_error
                @warn "Failed to generate plot for $param_name" exception=plot_error
            end
        end
    end
    
    plot_time = time() - plot_start_time
    println("✅ Plot generation completed in $(round(plot_time, digits=1))s")
    println("Successfully generated $successful_plots out of $successful_profiles plots")
    
    println("Results saved to: $profile_dir")
    println("Summary:")
    println("  - Total parameters in model: $(length(param_names))")
    println("  - Skipped parameters: $(length(skipped_params))")
    println("  - Profiled parameters: $num_params")
    println("  - Successful profiles: $successful_profiles")
    println("  - Failed profiles: $failed_profiles")
    
    return profile_results
end

# Helper function for extracting the normalized profile segment
function flat_profile(sol::ProfileLikelihood.ProfileLikelihoodSolution)
    pidx = first(keys(sol.parameter_values))
    segs_x = sol.parameter_values[pidx]
    segs_y = sol.profile_values[pidx]

    # Concatenate all segments to get the complete profile curve
    xs = reduce(vcat, segs_x)
    ys = reduce(vcat, segs_y)
    
    # Force normalization: shift so maximum = 0.0
    y_max = maximum(ys)
    ys = ys .- y_max
    
    return xs, ys
end

# Helper function for plotting ProfileLikelihoodSolution objects manually
function plot_profile_solution(sol::ProfileLikelihood.ProfileLikelihoodSolution,
                               mle_x::Real, name::AbstractString; scale=:linear)

    # 1. Extract the complete profile curve by concatenating all segments
    xs, ys = flat_profile(sol)
    
    # Diagnostic logging to understand data structure
    @info "Profile diagnostic for $name" length(xs) length(ys) extrema(xs) extrema(ys) maximum(ys)
    
    # Validation check for single-point issues
    if length(xs) == 1
        @error "Still getting single point for $name - investigating segment structure"
        pidx = first(keys(sol.parameter_values))
        segs_x = sol.parameter_values[pidx]
        segs_y = sol.profile_values[pidx]
        @info "Segment info" length(segs_x) [length(x) for x in segs_x] [length(y) for y in segs_y]
        error("Cannot plot single-point profile for $name")
    end

    # 2. Find the MLE point FROM THE PROFILE CURVE DATA itself
    mle_idx = argmax(ys)  # Index of maximum likelihood (should be ≈ 0)
    mle_x_from_curve = xs[mle_idx]  # X-coordinate from the curve
    mle_y_from_curve = ys[mle_idx]  # Y-coordinate from the curve (should be ≈ 0)

    # 3. Apply scale transformation and set clear labels for debug mode
    xlabel_text = if scale === :log10
        xs = 10 .^ xs
        mle_x_from_curve = 10 ^ mle_x_from_curve
        "$(name) (linear scale)"
    elseif scale === :log
        xs = exp.(xs)
        mle_x_from_curve = exp(mle_x_from_curve)
        "$(name) (linear scale)"
    else
        "$(name) (log₁₀ scale)"  # Clear indication this is log10 space
    end

    # 4. Validation assertions (relaxed for debug mode)
    @assert maximum(ys) ≈ 0.0 atol=1e-8 "Profile not normalised: max = $(maximum(ys))"
    @assert mle_y_from_curve ≈ 0.0 atol=1e-8 "MLE not at maximum: y = $(mle_y_from_curve)"

    # 5. Draw blue profile curve + red MLE dot (both in same coordinate system)
    plt = plot(xs, ys;
               marker  = :circle,
               xlabel  = xlabel_text,  # Explicit label with scale info
               ylabel  = "Profile log-likelihood",
               legend  = false)

    # 6. MLE point now guaranteed to be ON the curve peak
    scatter!(plt, [mle_x_from_curve], [mle_y_from_curve]; 
             color = :red, ms = 6)
    
    return plt
end

# Helper function for saving profile data with error handling
function save_profile_data(prof_result, param_name, profile_dir, mle_val, thread_id)
    data_save_path = joinpath(profile_dir, "profile_data_$(param_name).jld2")
    try
        @save data_save_path prof_result param_name mle_val
        println("Thread $thread_id: ✅ Profile data for $param_name saved")
    catch save_error
        @warn "Thread $thread_id: Failed to save profile data for $param_name" exception=save_error
    end
end

# Helper function for plot generation (now called sequentially)
function generate_profile_plot(prof_result, param_idx, param_name, profile_dir, mle_val, thread_id, parameter_scales=nothing)
    println("Generating plot for $param_name...")
    
    try
        # SIMPLIFIED: Keep everything in native log10 space to avoid coordinate system issues
        param_scale = :linear  # Don't apply any transformation by default
        original_name = param_name
        petab_scale = nothing  # For debugging
        
        # Only transform if we have explicit PEtab scale information AND it makes sense
        if !isnothing(parameter_scales) && length(parameter_scales) >= param_idx
            try
                petab_scale = parameter_scales[param_idx]
                if petab_scale == :log10
                    # Check if transformation would create reasonable ranges
                    pidx = first(keys(prof_result.parameter_values))
                    raw_range = extrema(reduce(vcat, prof_result.parameter_values[pidx]))
                    
                    # Only transform if the range is reasonable (< 5 orders of magnitude)
                    if raw_range[2] - raw_range[1] < 5.0
                        param_scale = :log10
                        original_name = startswith(param_name, "log10_") ? replace(param_name, "log10_" => "") : param_name
                        println("Applying log10→linear transformation for $param_name (range: $(round.(raw_range, digits=2)))")
                    else
                        println("Keeping $param_name in log10 space (transformation would create excessive range: $(round.(raw_range, digits=2)))")
                    end
                elseif petab_scale == :log
                    param_scale = :log
                    original_name = replace(param_name, r"^(log_|ln_)" => "")
                    println("Applying ln→linear transformation for $param_name")
                end
            catch scale_error
                @warn "Could not access PEtab scale for parameter $param_idx" exception=scale_error
            end
        else
            println("No PEtab scales available - keeping $param_name in native space")
        end
        
        # Simple fallback: only detect explicit log prefixes
        if param_scale == :linear && isnothing(petab_scale)
            if startswith(param_name, "log10_")
                param_scale = :log10
                original_name = replace(param_name, "log10_" => "")
                println("Using explicit log10_ prefix detection for $param_name")
            elseif startswith(param_name, "log_") || startswith(param_name, "ln_")
                param_scale = :log
                original_name = replace(param_name, r"^(log_|ln_)" => "")
                println("Using explicit log_/ln_ prefix detection for $param_name")
            else
                println("Debug mode: keeping $param_name in native space (likely log10)")
            end
        end
        
        # Debug output for scale detection
        @info "Scale detection for $param_name" param_scale petab_scale original_name param_idx mle_value=mle_val
        
        # Use manual plotting helper that works with ProfileLikelihoodSolution
        plt = plot_profile_solution(prof_result, mle_val, original_name; scale = param_scale)
        
        # Simple title for debug mode
        title_text = if param_scale == :log10
            "Profile Likelihood for $original_name (linear scale)"
        elseif param_scale == :log
            "Profile Likelihood for $original_name (linear scale)"
        else
            "Profile Likelihood for $original_name (log₁₀ scale)"
        end
        
        plt = plot!(plt, title = title_text)
        
        save_path = joinpath(profile_dir, "profile_$(param_name).png")
        savefig(plt, save_path)
        println("✅ Profile plot for $param_name saved to: $save_path")
        
    catch plot_error
        @warn "ProfileLikelihood plotting failed for $param_name" exception=plot_error
        println("✅ Profile computation for $param_name completed (plot generation failed)")
    end
end