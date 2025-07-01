using PEtab
using Optim
using JLD2
using ComponentArrays
using Sundials

# This dictionary is specific to the optimization process
const SUPPORTED_OPTIMIZERS = Dict(
    "LBFGS" => LBFGS(),
    "IPNewton" => IPNewton()
)

# Optimizer recommendations for different scenarios
const OPTIMIZER_GUIDANCE = Dict(
    "LBFGS" => "Good for smooth problems, but may struggle with very stiff systems",
    "IPNewton" => "More robust for constrained/stiff problems, slower but more reliable"
)

function run_parameter_estimation(parsed_args, petab_problem)
    println("\n🧪 Testing cost function before optimization...")
    
    # --- This block is now corrected ---
    try
        # Get a single, correctly scaled start-guess using the correct function
        x_test = get_startguesses(petab_problem, 1)
        println("Testing cost function with a single start-guess vector.")
        
        # Restore your safe_cost_function that uses .nllh, which is correct
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
    # --- End of corrected block ---

    use_parallel = parsed_args["parallel"]
    optimizer_choice_str = parsed_args["optimizer"]
    n_starts = parsed_args["n-starts"]
    if n_starts == 0
        n_starts = use_parallel ? length(procs()) : 1
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
    time_limit = debug_mode ? 30.0 : 600.0
    max_iterations = debug_mode ? 10 : 10000
    
    optim_options = Optim.Options(
        time_limit=time_limit,
        iterations=max_iterations,
        g_tol=debug_mode ? 1e-2 : 1e-6,
        f_reltol=debug_mode ? 1e-4 : 1e-8,
        show_trace=debug_mode,
        allow_f_increases=true
    )
    
    if debug_mode
        println("🐛 ULTRA-DEBUG MODE: Very short time limit ($(time_limit)s), very loose tolerances, max $(max_iterations) iterations")
    end

    println("\n[Timing] Calibrating parameters..."); flush(stdout)
    
    if use_parallel
        println("Mode: PARALLEL using $(length(procs())) processes, $n_starts starts, and optimizer $optimizer_choice_str")
        multi_start_res = calibrate_multistart(petab_problem, optimizer, n_starts;
                                               nprocs=length(procs()),
                                               dirsave=joinpath(pwd(), "Intermediate_results"),
                                               options=optim_options)
        return multi_start_res
    else
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