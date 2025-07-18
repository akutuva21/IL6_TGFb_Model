using PEtab
using Optim
using JLD2
using ComponentArrays
using Sundials

# This dictionary is specific to the optimization process
const SUPPORTED_OPTIMIZERS = Dict(
    "LBFGS" => LBFGS(),
    "IPNewton" => IPNewton(),
    "BFGS" => BFGS(),
    "ConjugateGradient" => ConjugateGradient(),
    "GradientDescent" => GradientDescent()
)

# Optimizer recommendations for different scenarios
const OPTIMIZER_GUIDANCE = Dict(
    "LBFGS" => "Good for smooth problems, memory efficient",
    "BFGS" => "Often faster than LBFGS for medium-sized problems",
    "ConjugateGradient" => "Fast for large parameter spaces",
    "GradientDescent" => "Simple and fast, good for rough optimization",
    "IPNewton" => "Most robust but slowest"
)

function run_parameter_estimation(parsed_args, petab_problem)
    println("\n🧪 Testing cost function before optimization...")
    
    try
        x_test = get_startguesses(petab_problem, 1)
        println("Testing cost function with a single start-guess vector.")
        
        function safe_cost_function(x)
            try
                result = petab_problem.nllh(x)
                if result === nothing || isinf(result) || isnan(result)
                    println("⚠️  Cost function returned an invalid value ($result), using penalty.")
                    return 1e10
                else
                    return result
                end
            catch e
                println("⚠️  An error occurred in the cost function: $e. Using penalty.")
                return 1e10
            end
        end

        cost_test = safe_cost_function(x_test)
        println("✅ Safe cost function test successful. Initial cost: $cost_test")

        if cost_test >= 1e10
            @warn "Initial cost is a penalty value. Check model parameters and solver options."
        end
        println("✅ Cost function test passed. Proceeding with optimization using safe wrapper.")

    catch e
        println("❌ Cost function test or start-guess generation failed: $e")
        return nothing
    end

    # FIXED: Use threading instead of distributed processing
    use_threading = Threads.nthreads() > 1
    optimizer_choice_str = parsed_args["optimizer"]
    n_starts = parsed_args["n-starts"]
    
    # FIXED: Use thread count for determining default n_starts
    if n_starts == 0
        n_starts = use_threading ? min(Threads.nthreads() * 4, 100) : 10
    end

    if !haskey(SUPPORTED_OPTIMIZERS, optimizer_choice_str)
        @error "Unsupported optimizer '$optimizer_choice_str'."
        return nothing
    end
    
    if haskey(OPTIMIZER_GUIDANCE, optimizer_choice_str)
        println("INFO: Using $optimizer_choice_str - $(OPTIMIZER_GUIDANCE[optimizer_choice_str])")
    end
    optimizer = SUPPORTED_OPTIMIZERS[optimizer_choice_str]
    debug_mode = get(parsed_args, "debug", false)
    time_limit = debug_mode ? 30.0 : 7200.0
    max_iterations = debug_mode ? 200 : 20000
    
    optim_options = Optim.Options(
        time_limit=time_limit,
        iterations=max_iterations,
        g_tol=debug_mode ? 1e-2 : 1e-5,
        f_reltol=debug_mode ? 1e-4 : 1e-9,
        show_trace=false,
        allow_f_increases=true
    )
    
    if debug_mode
        println("🐛 ULTRA-DEBUG MODE: Very short time limit ($(time_limit)s), very loose tolerances, max $(max_iterations) iterations")
    end

    println("\n[Timing] Calibrating parameters..."); flush(stdout)
    
    # FIXED: Use threading instead of distributed processing
    if use_threading
        println("Mode: PARALLEL (THREADING) using $(Threads.nthreads()) threads, $n_starts starts, and optimizer $optimizer_choice_str")
        
        println("Getting start guesses...")
        local start_guesses
        try
            _start_guesses_raw = get_startguesses(petab_problem, n_starts)
            start_guesses = (n_starts == 1) ? [_start_guesses_raw] : _start_guesses_raw
            println("Got $(length(start_guesses)) start guess(es) from PEtab")
        catch e
            println("❌ Start guess generation failed!")
            println("Error: $e")
            return nothing
        end
        
        # FIXED: Use threading for parallel optimization starts
        all_runs = Vector{Union{PEtab.PEtabOptimisationResult, Nothing}}(undef, length(start_guesses))
        
        Threads.@threads for i in 1:length(start_guesses)
            x0 = start_guesses[i]
            thread_id = Threads.threadid()
            println("  Thread $thread_id: Start $i/$n_starts...")
            try
                println("    Thread $thread_id: Starting optimization with initial guess...")
                start_time = time()
                res = calibrate(petab_problem, x0, optimizer; options=optim_options)
                elapsed = time() - start_time
                if !isnothing(res) && isfinite(res.fmin)
                    println("    Thread $thread_id: ✅ Optimization completed successfully in $(round(elapsed, digits=1))s. Cost: $(res.fmin)")
                    all_runs[i] = res
                else
                    println("    Thread $thread_id: ⚠️  Optimization returned invalid result for start $i after $(round(elapsed, digits=1))s")
                    all_runs[i] = nothing
                end
            catch e
                error_msg = sprint(showerror, e)
                if contains(error_msg, "maxiters") || contains(error_msg, "Interrupted")
                    @warn "    Thread $thread_id: ∇ Optimization start $i hit solver maxiters limit - this is expected for very stiff systems"
                else
                    @error "    Thread $thread_id: ❌ Calibration for start $i failed with error: $(typeof(e))" exception=(e, catch_backtrace())
                end
                all_runs[i] = nothing
            end
        end
        
        # Filter out nothing results
        valid_runs = filter(r -> !isnothing(r) && isfinite(r.fmin), all_runs)
        
        if isempty(valid_runs)
            @error "All threaded optimization starts failed to produce a valid solution."
            return nothing
        end
        
        best_res = valid_runs[argmin([r.fmin for r in valid_runs])]
        return PEtab.PEtabMultistartResult(best_res.xmin, best_res.fmin, best_res.alg, n_starts, 
                                           "LatinHypercubeSample", nothing, collect(filter(!isnothing, all_runs)))
        
    else
        # Serial fallback for single-threaded execution
        println("Mode: SERIAL with optimizer $optimizer_choice_str, $n_starts start(s)")
        println("Getting start guesses...")
        
        local start_guesses
        try
            _start_guesses_raw = get_startguesses(petab_problem, n_starts)
            start_guesses = (n_starts == 1) ? [_start_guesses_raw] : _start_guesses_raw
            println("Got $(length(start_guesses)) start guess(es) from PEtab")
        catch e
            println("❌ Start guess generation failed!")
            println("Error: $e")
            return nothing
        end
        
        all_runs = PEtab.PEtabOptimisationResult[]
        for (i, x0) in enumerate(start_guesses)
            println("  Serial Start $i/$n_starts...")
            try
                println("    Starting optimization with initial guess...")
                start_time = time()
                res = calibrate(petab_problem, x0, optimizer; options=optim_options)
                elapsed = time() - start_time
                if !isnothing(res) && isfinite(res.fmin)
                    println("    ✅ Optimization completed successfully in $(round(elapsed, digits=1))s. Cost: $(res.fmin)")
                    push!(all_runs, res)
                else
                    println("    ⚠️  Optimization returned invalid result for start $i after $(round(elapsed, digits=1))s")
                end
            catch e
                error_msg = sprint(showerror, e)
                if contains(error_msg, "maxiters") || contains(error_msg, "Interrupted")
                    @warn "    ∇ Optimization start $i hit solver maxiters limit - this is expected for very stiff systems"
                    @warn "    Consider using a different optimizer or increasing solver maxiters further"
                elseif isa(e, InterruptException)
                    @error "    🛑 Optimization interrupted for start $i" 
                else
                    @error "    ❌ Calibration for start $i failed with error: $(typeof(e))" exception=(e, catch_backtrace())
                end
            end
        end

        if isempty(all_runs)
            @error "No optimization runs completed successfully."
            return nothing
        end

        valid_runs = filter(r -> !isnothing(r) && isfinite(r.fmin), all_runs)
        if isempty(valid_runs)
            @error "All optimization starts failed to produce a valid solution."
            return nothing
        end
        best_res = valid_runs[argmin([r.fmin for r in valid_runs])]

        return PEtab.PEtabMultistartResult(best_res.xmin, best_res.fmin, best_res.alg, n_starts, 
                                            "LatinHypercubeSample", nothing, all_runs)
    end
end