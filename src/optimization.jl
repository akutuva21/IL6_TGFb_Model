# src/optimization.jl

using PEtab
using Optim
using JLD2
using ComponentArrays
using Base.Threads

# Import Latin Hypercube Sampling for robust start guess generation
using QuasiMonteCarlo: LatinHypercubeSample

try
    using PyCall
    global PYCALL_AVAILABLE = true
    println("✅ PyCall available - advanced optimizers enabled")
catch
    global PYCALL_AVAILABLE = false
    println("⚠️  PyCall not available - using fallback optimizers")
end

function select_optimizer(optimizer_choice::String, debug_mode::Bool=false)
    println("\n🎯 Selecting optimizer for parameter estimation...")
    if optimizer_choice == "auto"
        if PYCALL_AVAILABLE && !debug_mode
            println("🔬 AUTO-SELECTION: Using Fides for maximum robustness")
            optimizer_choice = "Fides"
        else
            println("🔬 AUTO-SELECTION: Using IPNewton (robust Julia native)")
            optimizer_choice = "IPNewton"
        end
    end

    if optimizer_choice == "Fides"
        if !PYCALL_AVAILABLE
            @warn "Fides optimizer requested but PyCall is not available. Falling back to IPNewton."
            return (:IPNewton, Optim.IPNewton(), Optim.Options())
        end
        println("✅ Using Fides: Most robust Newton-trust region")
        return (:Fides, :Fides, Dict("maxiter" => debug_mode ? 100 : 1000))
    end

    optimizer_map = Dict(
        "IPNewton" => (IPNewton(), "Robust interior-point Newton"),
        "LBFGS" => (LBFGS(), "Reliable quasi-Newton"),
        "BFGS" => (BFGS(), "Fast quasi-Newton")
    )

    if haskey(optimizer_map, optimizer_choice)
        optimizer_obj, description = optimizer_map[optimizer_choice]
        println("✅ Using $optimizer_choice: $description")
        options = Optim.Options(iterations = debug_mode ? 200 : 2000, g_tol = 1e-8)
        return (Symbol(optimizer_choice), optimizer_obj, options)
    else
        @error "Unknown optimizer: $optimizer_choice"
        return nothing
    end
end


"""
    generate_start_guesses_robustly(petab_problem, n_starts; max_retries=20)

A robust wrapper around PEtab.get_startguesses that retries if it fails to find
valid starting points.
"""
function generate_start_guesses_robustly(petab_problem::PEtabODEProblem, n_starts::Int; max_retries::Int=20)
    for i in 1:max_retries
        try
            # --- FIX: Explicitly use Latin Hypercube Sampling ---
            # This samples from the parameter bounds defined in the PEtab file,
            # which is the best practice for multi-start optimization and avoids
            # any potential bugs related to using the nominalValue.
            start_guesses = PEtab.get_startguesses(petab_problem, n_starts; sampling_method=LatinHypercubeSample())
            
            # Check if the result is valid
            if !isnothing(start_guesses) && !isempty(start_guesses) && all(sg -> !isnothing(sg), start_guesses)
                println("✅ Successfully generated $n_starts start guesses using Latin Hypercube Sampling.")
                return start_guesses
            end
            println("⚠️  Attempt $i/$max_retries: get_startguesses returned an invalid result. Retrying...")
        catch e
            println("⚠️  Attempt $i/$max_retries: get_startguesses failed with error: $e. Retrying...")
        end
        sleep(0.1) # Small delay before retrying
    end
    @error "❌ Failed to generate valid start guesses after $max_retries retries."
    return nothing
end


"""
Run the multi-start parameter estimation using a robust, threaded approach.
"""
function run_parameter_estimation(parsed_args, petab_problem)
    println("\n🧪 SCIENTIFIC PARAMETER ESTIMATION - Enhanced Strategy")
    println("="^70)

    # --- Step 1: Configure optimization strategy ---
    debug_mode = get(parsed_args, "debug", false)
    optimizer_choice = get(parsed_args, "optimizer", "auto")
    n_starts = get(parsed_args, "n-starts", Threads.nthreads())
    
    optimizer_setup = select_optimizer(optimizer_choice, debug_mode)
    if isnothing(optimizer_setup)
        @error "Failed to configure optimizer"
        return nothing
    end
    alg_symbol, optimizer_obj, options = optimizer_setup

    # --- Step 2: Validate the cost function with a robustly generated start guess ---
    println("\n🔍 Step 2: Cost Function Validation")
    x_test = generate_start_guesses_robustly(petab_problem, 1)
    if isnothing(x_test)
        @error "❌ Cost function validation failed because no valid start guess could be generated."
        return nothing
    end
    initial_cost = petab_problem.nllh(x_test)
    println("✅ Initial cost evaluation successful: $initial_cost")

    # --- Step 3: Generate all start guesses for the multi-start ---
    println("\n🚀 Step 3: Generating $n_starts Start Guesses for Multi-start")
    start_guesses = generate_start_guesses_robustly(petab_problem, n_starts)
    if isnothing(start_guesses)
        @error "❌ Failed to generate start guesses for multi-start run."
        return nothing
    end

    # --- Step 4: Run multi-start optimization ---
    println("\n⚡ Step 4: Executing Multi-Start Optimization")
    println("  • Optimizer: $alg_symbol")
    println("  • Threads: $(Threads.nthreads())")

    all_runs = Vector{Any}(undef, n_starts)
    start_time = time()

    Threads.@threads for i in 1:n_starts
        x0 = start_guesses[i]
        
        try
            # Use PEtab's single-shot calibrate function, which is robust
            result = PEtab.calibrate(petab_problem, optimizer_obj, x0; options=options, save_trace=false)
            all_runs[i] = result
        catch e
            println("⚠️  Run $i failed with error: $e")
            all_runs[i] = nothing
        end
    end

    total_elapsed = time() - start_time
    
    # --- Step 5: Process and return results ---
    valid_runs = filter(r -> !isnothing(r) && isfinite(r.fmin), all_runs)
    if isempty(valid_runs)
        @error "All optimization attempts failed."
        return nothing
    end

    best_run = valid_runs[argmin([r.fmin for r in valid_runs])]
    
    println("\n✅ Multi-start estimation completed!")
    println("   • Total time: $(round(total_elapsed/60, digits=1)) minutes")
    println("   • Successful runs: $(length(valid_runs))/$n_starts")
    println("   • Best cost: $(round(best_run.fmin, digits=3))")

    return PEtab.PEtabMultistartResult(
        best_run.xmin,
        best_run.fmin,
        best_run.alg,
        n_starts,
        "CustomThreaded",
        nothing,
        valid_runs
    )
end