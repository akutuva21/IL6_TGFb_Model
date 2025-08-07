# src/optimization.jl

using PEtab
using Optim
using PyCall

"""
    get_optimizer_and_options(optimizer_name::Symbol, debug_mode::Bool)

Returns a tuple containing the optimizer algorithm object and its corresponding options dictionary.
This function ensures all return values are consistent.
"""
function get_optimizer_and_options(optimizer_name::Symbol, debug_mode::Bool)
    
    if optimizer_name === :Fides
        # 1. Correctly create the Fides algorithm object.
        #    The Hessian update strategy is an argument to the constructor.
        fides_alg = PEtab.Fides(:BFGS; verbose=false)
        
        # 2. Options must be a Python dictionary. Do not put "hessian_update" here.
        fides_opts = py"{
            'maxiter': $(debug_mode ? 150 : 500),
            'fatol': 1e-5,
            'frtol': 1e-7,
            'gtol': 1e-6
        }"

        println("Using Fides via PEtab.jl's built-in wrapper with BFGS updates.")
        return fides_alg, fides_opts

    else
        # This part for Optim.jl solvers is correct and needs no changes.
        optim_options = Optim.Options(
            iterations = debug_mode ? 200 : 800,
            g_tol      = 1e-6,
            f_reltol   = debug_mode ? 1e-6 : 1e-8,
            show_trace = false
        )

        if optimizer_name === :IPNewton
            println("Using IPNewton: Robust interior-point Newton (Julia native)")
            return Optim.IPNewton(), optim_options
        elseif optimizer_name === :LBFGS
            println("Using LBFGS: Reliable quasi-Newton, memory efficient")
            return Optim.LBFGS(), optim_options
        elseif optimizer_name === :BFGS
            println("Using BFGS: Fast quasi-Newton for medium problems")
            return Optim.BFGS(), optim_options
        else
            @error "Unknown optimizer: $optimizer_name"
            throw(ArgumentError("Invalid optimizer specified."))
        end
    end
end


"""
Run the multi-start parameter estimation using PEtab.jl's built-in robust functionality.
"""
function run_parameter_estimation(parsed_args, petab_problem)
    println("\n🧪 SCIENTIFIC PARAMETER ESTIMATION")
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
    # Convert string from command line to Symbol for our function
    optimizer_name = Symbol(get(parsed_args, "optimizer", "Fides")) 
    n_starts = get(parsed_args, "n-starts", min(Threads.nthreads(), petab_problem.nparameters_estimate * 2))
    
    # This call is now robust and returns a consistent two-part tuple
    optimizer_alg, options = get_optimizer_and_options(optimizer_name, debug_mode)
    
    # Step 2: Execute robust multi-start parameter estimation
    println("\n🚀 Step 2: Multi-Start Parameter Estimation")
    println("Configuration:")
    println("  • Optimizer: $(typeof(optimizer_alg))")
    println("  • Multi-starts: $n_starts")
    println("  • Threading: Enabled via Julia --threads flag")
    
    start_time = time()
    multi_start_res = nothing

    try
        # This single call now works for Fides, IPNewton, etc. without needing an if/else
        multi_start_res = PEtab.calibrate_multistart(
            petab_problem,
            optimizer_alg, # Pass the algorithm object directly
            n_starts;
            options=options,
            save_trace=false,
            dirsave=(debug_mode ? "Intermediate_results" : nothing) # avoid heavy I/O unless debugging
        )
    catch e
        @error "Multi-start parameter estimation failed: $e"
        showerror(stdout, e, catch_backtrace())
        return nothing
    end

    total_elapsed = time() - start_time
    
    if isnothing(multi_start_res) || (hasproperty(multi_start_res, :runs) && isempty(multi_start_res.runs))
        @error "All optimization attempts failed. Check model stability and parameter bounds."
        return nothing
    end

    println("\n✅ Multi-start estimation completed!")
    println("   • Total time: $(round(total_elapsed/60, digits=1)) minutes")
    println("   • Successful runs: $(length(multi_start_res.runs))/$n_starts")
    println("   • Best cost: $(round(multi_start_res.fmin, digits=3))")
    
    return multi_start_res
end
