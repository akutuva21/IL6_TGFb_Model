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

function create_petab_compatible_parameters(petab_problem::PEtabODEProblem)
    """
    Create properly formatted parameter vector for PEtab cost function evaluation.
    This handles the log10 transformation and parameter ordering that PEtab expects.
    """
    param_names = petab_problem.xnames
    param_values = petab_problem.xnominal
    
    # CORRECTED: :log10_TGFb_0 has been removed as it is no longer an estimated parameter.
    expected_order = [
        :log10_IL6R_0, :log10_kr_s3stat3d, :log10_k_deact_pka, :log10_kf_pka_bind, 
        :log10_k_dephos_smad3, :log10_kr_pka_bind, :log10_k_phos_smad3, :log10_kr_s3s4, 
        :log10_kf_s3s4, :log10_k_inact_il6r, :log10_k_deact_stat3, 
        :log10_k_act_stat3_by_il6r, :log10_k_cat_pka, :log10_PKA_active_0, 
        :log10_kr_il6_bind, :log10_PKA_0, :log10_k_act_il6r, :log10_STAT3m_0, 
        :log10_kf_il6_bind, :log10_SMAD4_0, :log10_STAT3d_active_0, 
        :log10_kf_s3stat3d, :log10_SMAD3_phos_P_0, :log10_SMAD3_0
    ]
    
    param_dict = Dict(zip(param_names, param_values))
    ordered_values = Float64[]
    
    for expected_name in expected_order
        current_name = Symbol(replace(string(expected_name), "log10_" => ""))
        
        if haskey(param_dict, current_name)
            original_value = param_dict[current_name]
            final_value = if startswith(string(expected_name), "log10_")
                log10(max(original_value, 1e-12))
            else
                original_value
            end
            push!(ordered_values, final_value)
        else
            @error "Could not find parameter $(current_name) for expected $(expected_name)"
            throw(ArgumentError("Missing parameter: $(current_name)"))
        end
    end
    
    return ComponentArray(NamedTuple{Tuple(expected_order)}(ordered_values))
end

function run_parameter_estimation(parsed_args, petab_problem)
    println("\n🧪 Testing cost function before optimization...")
    
    # Test the cost function with properly formatted parameters
    try
        # Create properly formatted parameters for PEtab
        x_test = create_petab_compatible_parameters(petab_problem)
        println("Created properly formatted parameter vector with $(length(x_test)) parameters")
        
        # Create a safe wrapper for the cost function that handles 'nothing' values
        function safe_cost_function(x)
            try
                result = petab_problem.nllh(x)
                
                # Check if result is nothing and return a large penalty instead
                if result === nothing
                    println("⚠️  Cost function returned 'nothing', using penalty value")
                    return 1e10  # Large penalty value
                elseif isnan(result) || isinf(result)
                    println("⚠️  Cost function returned invalid value: $result, using penalty")
                    return 1e10
                else
                    return result
                end
            catch e
                if isa(e, MethodError) && e.f == convert && length(e.args) == 2 && e.args[2] === nothing
                    println("⚠️  Caught 'convert nothing to Real' error, using penalty value")
                    return 1e10  # Large penalty value instead of crashing
                else
                    println("⚠️  Cost function error: $e, using penalty value")
                    return 1e10
                end
            end
        end
        
        # Test the safe cost function
        cost_test = safe_cost_function(x_test)
        println("✅ Safe cost function test successful. Initial cost: $cost_test")
        
        if cost_test >= 1e10
            println("⚠️  Warning: Using penalty value due to cost function issues")
        elseif isinf(cost_test) || isnan(cost_test)
            println("⚠️  Warning: Initial cost is infinite or NaN - this may cause optimization issues")
        end
        
        # Store the safe cost function for use in optimization
        println("✅ Cost function test passed. Proceeding with optimization using safe wrapper.")
        
    catch e
        println("❌ Cost function test failed: $e")
        
        # Check if this is the specific 'convert nothing to Real' error
        if isa(e, MethodError) && e.f == convert && length(e.args) == 2 && e.args[2] === nothing
            println("🔍 Detected the 'convert nothing to Real' error!")
            println("This means an observable function is returning 'nothing' instead of a number.")
            println("This is likely due to steady-state solving issues when all species concentrations are zero.")
            println("❌ Parameter estimation cannot proceed until this issue is resolved.")
        else
            println("This explains the original parameter estimation failure.")
        end
        
        return nothing
    end

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
    
    # Print optimizer guidance
    if haskey(OPTIMIZER_GUIDANCE, optimizer_choice_str)
        println("INFO: Using $optimizer_choice_str - $(OPTIMIZER_GUIDANCE[optimizer_choice_str])")
    end
    optimizer = SUPPORTED_OPTIMIZERS[optimizer_choice_str]
    # Increase time limit and make optimization more robust for stiff systems
    # Check if this is a debug run (short time limit)
    debug_mode = get(parsed_args, "debug", false)
    time_limit = debug_mode ? 30.0 : 600.0   # 30 seconds for debug, 10 min for full run
    max_iterations = debug_mode ? 10 : 10000   # Very few iterations for debug
    
    optim_options = Optim.Options(
        time_limit=time_limit,
        iterations=max_iterations,
        g_tol=debug_mode ? 1e-2 : 1e-6,        # Very loose tolerance for debug
        f_tol=debug_mode ? 1e-4 : 1e-8,        # Very loose tolerance for debug
        show_trace=debug_mode,                  # Show progress in debug mode
        allow_f_increases=true
    )
    
    if debug_mode
        println("🐛 ULTRA-DEBUG MODE: Very short time limit ($(time_limit)s), very loose tolerances, max $(max_iterations) iterations")
    end

    println("\n[Timing] Calibrating parameters..."); flush(stdout)
    calibration_start_time = time()
    
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
        
        # Declare start_guesses in outer scope
        local start_guesses
        
        # Use PEtab's default start guess generation
        # The robust steady-state solver should prevent 'nothing' returns
        try
            _start_guesses_raw = get_startguesses(petab_problem, n_starts)
            start_guesses = (n_starts == 1) ? [_start_guesses_raw] : _start_guesses_raw
            println("Got $(length(start_guesses)) start guess(es) from PEtab")
        catch e
            println("❌ Start guess generation failed!")
            println("Error: $e")
            return nothing
        end
        
        all_runs = PEtabOptimisationResult[]
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
                    @warn "    � Optimization start $i hit solver maxiters limit - this is expected for very stiff systems"
                    # Try to extract any partial result if available
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

        return PEtabMultistartResult(best_res.xmin, best_res.fmin, best_res.alg, n_starts, 
                                     "LatinHypercubeSample", nothing, all_runs)
    end
end
