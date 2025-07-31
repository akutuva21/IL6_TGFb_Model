# src/optimization.jl (REVISED AND SIMPLIFIED)

using PEtab
using Optim
using JLD2
using ComponentArrays

# Import PyCall to enable Fides optimizer for maximum robustness
try
    using PyCall
    using PEtab: Fides  # Import Fides from PEtab
    global PYCALL_AVAILABLE = true
    println("✅ PyCall available - advanced optimizers enabled")
catch
    global PYCALL_AVAILABLE = false
    println("⚠️  PyCall not available - using fallback optimizers")
end

"""
Select the optimal optimizer and its options based on user choice and availability.
"""
function select_optimizer(optimizer_choice::String, debug_mode::Bool=false)
    println("\n🎯 Selecting optimizer for parameter estimation...")
    
    # Auto-select the best available optimizer
    if optimizer_choice == "auto"
        if PYCALL_AVAILABLE  # Removed && !debug_mode condition
            println("🔬 AUTO-SELECTION: Using Fides for maximum robustness")
            optimizer_choice = "Fides"
        else
            println("🔬 AUTO-SELECTION: Using IPNewton (robust Julia native)")
            optimizer_choice = "IPNewton"
        end
    end

    # Setup for Fides (via PyCall)
    if optimizer_choice == "Fides"
        if !PYCALL_AVAILABLE
            @warn "Fides optimizer requested but PyCall is not available. Falling back to IPNewton."
            return (:IPNewton, Optim.IPNewton(), Optim.Options(iterations=1000))
        end
        
        # Test if Fides can be imported
        try
            fides_py = pyimport("fides")
            println("✅ Using Fides: Most robust Newton-trust region (Python Fides via PyCall)")
            
            # Return the actual Fides object with proper constructor
            fides_options = Dict("maxiter" => debug_mode ? 100 : 1000)
            return (:fides, PEtab.Fides(nothing; verbose=false), fides_options)  # Use PEtab.Fides() constructor
        catch e
            @warn "Fides import failed: $e. Falling back to IPNewton."
            return (:IPNewton, Optim.IPNewton(), Optim.Options(iterations=1000))
        end
    end

    # Setup for Julia-native Optim.jl optimizers
    optimizer_map = Dict(
        "IPNewton" => (IPNewton(), "Robust interior-point Newton (Julia native)"),
        "LBFGS" => (LBFGS(), "Reliable quasi-Newton, memory efficient"),
        "BFGS" => (BFGS(), "Fast quasi-Newton for medium problems")
    )

    if haskey(optimizer_map, optimizer_choice)
        optimizer_obj, description = optimizer_map[optimizer_choice]
        println("✅ Using $optimizer_choice: $description")
        
        options = Optim.Options(
            iterations = debug_mode ? 200 : 2000,
            g_tol = debug_mode ? 1e-6 : 1e-8,
            f_reltol = debug_mode ? 1e-6 : 1e-12,
            show_trace = false
        )
        return (Symbol(optimizer_choice), optimizer_obj, options)
    else
        @error "Unknown optimizer: $optimizer_choice"
        return nothing
    end
end


"""
Run the multi-start parameter estimation using PEtab.jl's built-in robust functionality.
"""
function run_parameter_estimation(parsed_args, petab_problem)
    println("\n🧪 SCIENTIFIC PARAMETER ESTIMATION - Enhanced Strategy")
    println("="^70)

    # Step 0: Check parameter setup
    println("\n📋 Parameter Setup Validation")
    println("Number of parameters to estimate: $(petab_problem.nparameters_estimate)")
    
    if petab_problem.nparameters_estimate == 0
        @error "❌ No parameters marked for estimation! Check your parameter file."
        @error "Make sure some parameters have 'estimate = 1' in the parameters table."
        return nothing
    end
    
    println("✅ Found $(petab_problem.nparameters_estimate) parameters to estimate")
    println("Parameter names: $(petab_problem.xnames)")

    # Step 1: Configure optimization strategy
    println("\n⚙️  Step 1: Optimization Strategy Configuration")
    
    debug_mode = get(parsed_args, "debug", false)
    optimizer_choice = get(parsed_args, "optimizer", "auto")
    n_starts = get(parsed_args, "n-starts", Threads.nthreads())
    
    optimizer_setup = select_optimizer(optimizer_choice, debug_mode)
    if isnothing(optimizer_setup)
        @error "Failed to configure optimizer"
        return nothing
    end
    
    alg_symbol, optimizer_obj, options = optimizer_setup
    
    # Step 2: Execute robust multi-start parameter estimation
    println("\n🚀 Step 2: Multi-Start Parameter Estimation")
    println("Configuration:")
    println("  • Optimizer: $alg_symbol")
    println("  • Multi-starts: $n_starts")
    println("  • Threading: Enabled via Julia --threads flag")
    
    start_time = time()

    # Initialize the variable BEFORE the try block to fix scoping issue
    multi_start_res = nothing

    # Use the official PEtab.jl function, which handles threading and failures internally
    try
        println("🔍 Debug: Testing cost function before multistart...")
        
        # Test the cost function first
        x_test = PEtab.get_startguesses(petab_problem, 1)
        println("   Test parameter vector: $x_test")
        
        cost_test = petab_problem.nllh(x_test)
        println("   Test cost result: $cost_test (type: $(typeof(cost_test)))")
        
        if cost_test === nothing
            @error "Cost function returns nothing - this will cause the MethodError"
            return nothing
        elseif !isa(cost_test, Real) || !isfinite(cost_test)
            @warn "Cost function returns non-finite value: $cost_test"
        else
            println("   ✅ Cost function test passed")
        end
        
        println("🔍 Debug: Starting multistart with validated cost function...")
        
        if alg_symbol == :fides
            # Use Fides via PEtab's Python interface with proper Fides object
            multi_start_res = PEtab.calibrate_multistart(
                petab_problem,
                optimizer_obj,  # This should now be the proper PEtab.Fides() object
                n_starts;
                options=options,
                save_trace=false,
                dirsave = "intermediate_results"  # Save intermediate results
            )
        else
            # Use Julia native optimizers
            multi_start_res = PEtab.calibrate_multistart(
                petab_problem,
                optimizer_obj,
                n_starts;
                options=options,
                save_trace=false,
                dirsave = "intermediate_results"  # Save intermediate results
            )
        end
    catch e
        @error "Multi-start parameter estimation failed: $e"
        println("Full error details:")
        showerror(stdout, e, catch_backtrace())
        return nothing  # Return early on error
    end

    total_elapsed = time() - start_time
    
    if isnothing(multi_start_res) || isempty(multi_start_res.runs)
        @error "All optimization attempts failed. Check model stability and parameter bounds."
        return nothing
    end

    println("\n✅ Multi-start estimation completed!")
    println("   • Total time: $(round(total_elapsed/60, digits=1)) minutes")
    println("   • Successful runs: $(length(multi_start_res.runs))/$n_starts")
    println("   • Best cost: $(round(multi_start_res.fmin, digits=3))")
    
    return multi_start_res
end
