# main.jl

using Pkg
Pkg.activate("./bngl_julia")

include("src/model_param_est_robustness.jl")
include("src/visualization.jl")
include("src/optimization.jl")

# --- START OF DEBUGGING FUNCTION ---
function debug_petab_problem(prob::PEtab.PEtabODEProblem)
    println("\n\n--- 🕵️  STARTING PETAB PROBLEM DIAGNOSTICS 🕵️  ---")
    
    println("\n[1] Checking Parameter Names and Count...")
    n_params = length(prob.xnames)
    println("    Number of parameters to estimate: ", n_params)
    println("    Parameter names: ", prob.xnames)

    println("\n[2] Checking Parameter Bounds...")
    println("    Lower bounds type: ", typeof(prob.lower_bounds))
    println("    Upper bounds type: ", typeof(prob.upper_bounds))
    
    lb_ok = all(x -> isa(x, Float64), prob.lower_bounds)
    ub_ok = all(x -> isa(x, Float64), prob.upper_bounds)
    println("    All lower bounds are Float64: ", lb_ok)
    println("    All upper bounds are Float64: ", ub_ok)
    if !lb_ok
        println("    ⚠️  Problem in lower bounds: ", prob.lower_bounds)
    end
    if !ub_ok
        println("    ⚠️  Problem in upper bounds: ", prob.upper_bounds)
    end

    println("\n[3] Checking Nominal Transformed Values (Source of `similar`)...")
    println("    Nominal transformed type: ", typeof(prob.xnominal_transformed))
    println("    Nominal transformed element type: ", eltype(prob.xnominal_transformed))
    nominal_ok = all(x -> isa(x, Float64), prob.xnominal_transformed)
    println("    All nominal transformed values are Float64: ", nominal_ok)
    if !nominal_ok
        println("    ⚠️  Problem in nominal transformed values: ")
        show(stdout, "text/plain", prob.xnominal_transformed)
        println()
    end

    println("\n[4] Checking Parameter Scales (Crucial for `transform_x`)...")
    xscales = prob.model_info.xindices.xscale
    println("    Parameter scales dictionary: ")
    show(stdout, "text/plain", xscales)
    println()

    valid_scales = [:lin, :log, :log10, :log2]
    scales_ok = all(scale -> scale in valid_scales, values(xscales))
    println("    All parameter scales are valid: ", scales_ok)
    if !scales_ok
        for (param, scale) in xscales
            if !(scale in valid_scales)
                println("    ⚠️  INVALID SCALE FOUND for parameter '", param, "': ", scale)
            end
        end
    end

    println("\n--- 🕵️  END OF DIAGNOSTICS 🕵️  ---\n\n")
end
# --- END OF DEBUGGING FUNCTION ---

using LinearAlgebra
using ArgParse
using JLD2
using Base.Threads
using DiffEqCallbacks

if Threads.nthreads() > 1
    println("INFO: Running with $(Threads.nthreads()) threads")
else
    println("INFO: Running with only 1 thread. For better performance, start Julia with: julia --threads=24")
end

include("src/profiling.jl")

using ComponentArrays
using Plots
using PEtab
using SciMLSensitivity
using OrdinaryDiffEq
using Optim
using Sundials
using Optimization

const DEFAULT_YAML_PATH = "petab_problem.yml"

# --- DEFINE AND PARSE ARGUMENTS ---
function define_argument_parser()
    s = ArgParseSettings(description="Run parameter estimation and visualization using a PEtab YAML file.")
    @add_arg_table! s begin
        "--yaml"
            arg_type = String
            default = "petab_problem.yml"
        "--parallel"
            help = "Enable parallel processing (deprecated - use threading instead)."
            action = :store_true
        "--output", "-o"
            help = "Path to the JLD2 output file for saving/loading results."
            arg_type = String
            default = "estimation_output_small.jld"
        "--n-starts"
            help = "Number of multi-starts. Defaults to thread-based parallel execution."
            arg_type = Int
            default = 0
        "--optimizer"
            help = "Optimization algorithm. Options: auto, Fides, IPNewton, LBFGS, BFGS (auto=most robust available)"
            arg_type = String
            default = "auto"
        "--debug"
            help = "Enable debug mode for faster, less accurate testing."
            action = :store_true
        "--profile"
            help = "Run likelihood profiling on the best-fit parameters."
            action = :store_true
    end
    return s
end

const PARSED_ARGS = parse_args(ARGS, define_argument_parser())

# ===================================================================
# --- 2. THREADING SETUP ---
# ===================================================================

if PARSED_ARGS["parallel"]
    @warn "The --parallel flag is deprecated. This version uses threading instead."
    @warn "For optimal performance, start Julia with: julia --threads=24"
end

# Threading is handled automatically by Julia when started with --threads=N
println("INFO: Using threading for parallel processing")
println("INFO: Available threads: $(Threads.nthreads())")
println("INFO: This approach is more reliable on SLURM clusters than distributed processing")

const PROJECT_PATH = abspath(dirname(Base.active_project()))
println("INFO: Project path: $PROJECT_PATH")

println("INFO: Using threading-based approach - no worker setup required")
println("INFO: All packages loaded in main process only")

function run_analysis()
    parsed_args = PARSED_ARGS  # Use globally parsed arguments

    # Apply debug mode adjustments
    if parsed_args["debug"]
        println("INFO: Debug mode enabled - using faster, less accurate settings")
        # In debug mode, run only a few starts for faster testing
        parsed_args["n-starts"] = parsed_args["n-starts"] == 0 ? 5 : min(parsed_args["n-starts"], 10)
        println("INFO: Debug mode will use faster tolerances and fewer starts")
    end

    # Dynamically set n_starts based on available threads
    if parsed_args["n-starts"] == 0
        n_threads = Threads.nthreads()
        parsed_args["n-starts"] = n_threads
        println("INFO: --n-starts not provided, defaulting to $(parsed_args["n-starts"]) based on $(n_threads) threads")
    end

    # File paths from command line
    yaml_path = parsed_args["yaml"]
    output_filename = parsed_args["output"]

    println("--- Starting Full Analysis ---"); flush(stdout)
    println("Using PEtab YAML file: '$yaml_path'"); flush(stdout)
    println("Using output file: '$output_filename'"); flush(stdout)

    # --- 1. Load existing results if available ---
    local multi_start_res = nothing
    if isfile(output_filename)
        println("Found existing '$output_filename'. Attempting to load results..."); flush(stdout)
        try
            best_mle = nothing
            best_cost = nothing
            
            try
                JLD2.@load output_filename best_mle best_cost
                if !isnothing(best_mle) && !isnothing(best_cost)
                    println("✅ Successfully loaded essential estimation data!")
                    println("  - Best cost: $best_cost")
                    println("  - Parameter count: $(length(best_mle))")
                    
                    # --- START: CORRECTED OBJECT RECONSTRUCTION ---
                    
                    # 1. Create a minimal PEtabOptimisationResult to represent the loaded best fit.
                    #    We fill in the non-essential fields with placeholder values.
                    best_run_reconstructed = PEtab.PEtabOptimisationResult(
                        best_mle,         # xmin
                        best_cost,        # fmin
                        best_mle,         # x0 (can use xmin as a placeholder)
                        :LoadedFromFile,  # alg
                        0,                # niterations
                        0.0,              # runtime
                        [],               # xtrace
                        [],               # ftrace
                        true,             # converged
                        nothing           # original
                    )

                    # 2. Create the PEtabMultistartResult with a list containing our single reconstructed run.
                    multi_start_res = PEtab.PEtabMultistartResult(
                        best_mle,            # xmin (the overall best)
                        best_cost,           # fmin (the overall best)
                        :LoadedFromFile,     # alg (placeholder)
                        1,                   # nmultistarts (placeholder)
                        "LoadedFromFile",    # sampling_method (placeholder)
                        nothing,             # dirsave
                        [best_run_reconstructed] # runs <-- NOW CONTAINS ONE VALID RUN
                    )
                    # --- END: CORRECTED OBJECT RECONSTRUCTION ---

                else
                    @warn "Essential data is incomplete. Will re-run estimation."
                    multi_start_res = nothing
                end
            catch
                println("Attempting to load legacy format...")
                JLD2.@load output_filename multi_start_res
                
                if !isnothing(multi_start_res) && hasfield(typeof(multi_start_res), :xmin) && hasfield(typeof(multi_start_res), :fmin)
                    println("Successfully loaded legacy 'multi_start_res' object!")
                    println("  - Best cost: $(multi_start_res.fmin)")
                    println("  - Parameter count: $(length(multi_start_res.xmin))")
                else
                    @warn "Loaded object is not a valid multi-start result. Will re-run estimation."
                    multi_start_res = nothing
                end
            end
            
        catch e
            @warn "Could not load results from file. Will re-run estimation. Error: $e"
            multi_start_res = nothing
        end
    end

    # --- 2. Setup the core PEtabModel from YAML file ---
    println("INFO: Setting up PEtab Model from YAML file..."); flush(stdout)

    @time setup_results = setup_petab_problem(yaml_path)
    if isnothing(setup_results)
        @error "Failed to build PEtabModel from '$yaml_path'. Cannot proceed."
        return
    end

    petab_model = setup_results.petab_model
    true_param_values = setup_results.true_values
    println("INFO: Successfully loaded PEtab model with $(length(true_param_values)) parameters")

    # --- 3. Define robust solver options ---
    println("INFO: Defining robust solver for simulation and steady-state..."); flush(stdout)
        
    local odesol, gradient_method

    # === ENHANCED ODE SOLVER CONFIGURATION FOR SCIENTIFIC PARAMETER ESTIMATION ===
    # Following DifferentialEquations.jl best practices for maximum accuracy and robustness
    # Using TerminateSteadyState callback for robust steady-state detection
    
    if parsed_args["debug"]
        println("🐛 DEBUG MODE: Using ROBUST composite solver with loose tolerances for rapid iteration")
        println("📖 Using AutoVern7(Rodas5P()) with built-in steady-state detection")
        
        odesol = ODESolver(
            AutoVern7(Rodas5P()),
            abstol=1e-6,              # Relaxed absolute tolerance for speed
            reltol=1e-6,              # Relaxed relative tolerance for speed
            force_dtmin=true,         # Crucial for preventing failures
            maxiters=10000            # Lower maxiters for faster debug runs
        )
        
        gradient_method = :ForwardDiff
        
    else # PRODUCTION MODE: Maximum accuracy for publication-quality fits
        println("🔬 PRODUCTION MODE: High-accuracy composite solver for publication-quality fits")
        println("📖 Using AutoVern7(Rodas5P()) - adaptive algorithm selection with built-in steady-state")
        println("   • AutoVern7: High-order solver for smooth regions")
        println("   • Rodas5P: Specialized for stiff biochemical systems")
        println("   • TerminateSteadyState: Robust steady-state detection")
        
        # Composite algorithm following DifferentialEquations.jl recommendations
        # AutoVern7 handles smooth regions efficiently, Rodas5P handles stiff regions
        composite_solver = AutoVern7(Rodas5P())
        
        odesol = ODESolver(
            composite_solver,
            abstol=1e-9,              # Tighter absolute tolerance for precise gradients
            reltol=1e-9,              # Tighter relative tolerance for precise gradients  
            force_dtmin=true,         # Force minimum timestep to prevent NaN errors
            maxiters=10000000         # Allow more iterations for complex dynamics
        )
        
        gradient_method = :ForwardDiff  # Most reliable for biochemical models
    end

    # --- START OF THE FIX ---
    # We will now use a more explicit and robust steady-state solver that avoids
    # the TerminateSteadyState callback which is causing the bug.
    # By setting tmax=Inf, we instruct the solver to simulate until the system
    # naturally reaches a steady state.
    println("INFO: Using explicit steady-state simulation (tmax=Inf) to bypass callback bug.")
    local steadystate_solver = SteadyStateSolver(
        :Simulate,
        # The key change is that we are NOT providing a custom termination_check.
        # PEtab.jl will then default to simulating for a very long time (tmax=Inf),
        # which is a very robust way to find the steady state.
        abstol = odesol.abstol * 10,
        reltol = odesol.reltol * 10
    )
    # --- END OF THE FIX ---

    println("✅ Enhanced solver configuration completed:")
    println("   • ODE Solver: $(typeof(odesol.solver))")
    println("   • Absolute tolerance: $(odesol.abstol)")
    println("   • Relative tolerance: $(odesol.reltol)")
    println("   • Steady-state solver: :Simulate mode with tmax=Inf")
    println("   • Expected benefits: Bypasses internal callback bug, robust steady-state detection")

    # --- 4. Run estimation ONLY if no results were loaded ---
    local petab_problem
    if isnothing(multi_start_res)
        println("INFO: Building PEtabODEProblem for estimation..."); flush(stdout)

        # --- START: CORRECTED AND FINAL CALLBACK CODE ---
        println("INFO: Adding PositiveDomain callback to PEtabModel to enforce non-negative concentrations.")
        
        # 1. Create the PositiveDomain callback
        positive_domain_cb = PositiveDomain()

        # 2. Combine it with any callbacks that might already exist in the model
        combined_callbacks = CallbackSet(petab_model.callbacks, positive_domain_cb)

        # 3. Create a NEW PEtabModel instance, copying all fields from the original
        #    but replacing the .callbacks field. This is the correct way to handle
        #    immutable structs.
        petab_model_with_callback = PEtabModel(
            petab_model.name,
            petab_model.h,
            petab_model.u0!,
            petab_model.u0,
            petab_model.sd,
            petab_model.float_tspan,
            petab_model.paths,
            petab_model.sys,
            petab_model.sys_mutated,
            petab_model.parametermap,
            petab_model.speciemap,
            petab_model.petab_tables,
            combined_callbacks,
            petab_model.defined_in_julia
        )
        
        # 4. Now, create the PEtabODEProblem using the NEW model object.
        @time petab_problem = PEtabODEProblem(
            petab_model_with_callback, # <-- Use the new model
            odesolver = odesol,
            ss_solver = steadystate_solver,
            gradient_method = gradient_method,
            verbose = false
        )
        # --- END: CORRECTED AND FINAL CALLBACK CODE ---
        
        # --- ADD DIAGNOSTIC CALL ---
        debug_petab_problem(petab_problem)
        # ---------------------
        
        println("✅ PEtabODEProblem created successfully")

        multi_start_res = run_parameter_estimation(parsed_args, petab_problem)
         
        if isnothing(multi_start_res)
            @error "Parameter estimation failed. Cannot proceed."
            return
        end

        try
            best_mle = multi_start_res.xmin
            best_cost = multi_start_res.fmin
            
            @info "Best parameters found: $best_mle"
            @info "Best cost: $best_cost"
            
            # Save only the essential data, avoiding complex Optim objects
            JLD2.@save output_filename best_mle best_cost
            println("✅ Essential estimation data saved successfully to '$output_filename'")
            println("   (Saved MLE parameters and cost, avoiding JLD2 warnings)")
        catch e
            @error "Failed to save essential data to '$output_filename'. Error: $e"
        end
    end

    # --- 5. Build visualization problem only if needed, otherwise reuse ---
    if !@isdefined(petab_problem)
        println("INFO: Building PEtabODEProblem for visualization..."); flush(stdout)
        
        # --- START: CORRECTED AND FINAL CALLBACK CODE ---
        println("INFO: Adding PositiveDomain callback to PEtabModel for visualization.")
        positive_domain_cb = PositiveDomain()
        combined_callbacks = CallbackSet(petab_model.callbacks, positive_domain_cb)
        
        petab_model_with_callback = PEtabModel(
            petab_model.name,
            petab_model.h,
            petab_model.u0!,
            petab_model.u0,
            petab_model.sd,
            petab_model.float_tspan,
            petab_model.paths,
            petab_model.sys,
            petab_model.sys_mutated,
            petab_model.parametermap,
            petab_model.speciemap,
            petab_model.petab_tables,
            combined_callbacks,
            petab_model.defined_in_julia
        )
        
        @time petab_problem = PEtabODEProblem(
            petab_model_with_callback,
            odesolver = odesol,
            ss_solver = steadystate_solver,
            gradient_method = gradient_method,
            verbose = false
        )
        # --- END: CORRECTED AND FINAL CALLBACK CODE ---
    else
        println("INFO: Reusing existing PEtabODEProblem for visualization."); flush(stdout)
    end

    # --- 6. Generate Plots and Final Visualizations ---
    if !isnothing(multi_start_res)
        println("\n--- Diagnosing Multistart Data ---"); flush(stdout)
        diagnose_multistart_data(multi_start_res, petab_problem)
        
        println("\n--- Generating Waterfall Plot ---"); flush(stdout)
        try
            plot_waterfall(multi_start_res)
        catch e
            @warn "Primary waterfall plot failed: $e"
            println("Attempting fallback implementation...")
            try
                plot_waterfall_custom_fallback(multi_start_res)
            catch e2
                @warn "Fallback waterfall plot also failed: $e2"
                try
                    plot_waterfall_native_fallback(multi_start_res, petab_problem)
                catch e3
                    @error "All waterfall plot implementations failed. Last error: $e3"
                end
            end
        end
        
        println("\n--- Generating Parameter Distribution Plot ---"); flush(stdout)
        
        # Use the true parameter values from the BNGL model as reference
        println("INFO: Using true parameter values from BNGL model as reference")
        
        plot_parameter_distribution(multi_start_res, petab_problem, reference_values=true_param_values)
    end

    saved_results = (
        theta_optim=multi_start_res.xmin, 
        cost=multi_start_res.fmin,
        names_est_opt=string.(propertynames(multi_start_res.xmin))
    )
    
    println("\n--- Starting Visualization ---"); flush(stdout)
    println("\n[Timing] Running visualization..."); flush(stdout)
    @time try
        run_visualization(
            collect(saved_results.theta_optim),
            petab_problem,
            odesol
        )
        println("✅ Visualization completed successfully!"); flush(stdout)
    catch e
        @error "Failed to generate visualization plots." exception=(e, catch_backtrace())
    end

    # --- 7. Run Likelihood Profiling if Requested ---
    if parsed_args["profile"]
        # We no longer need to check for multistart_result, as profiling is now independent.
        println("INFO: Running modern likelihood profiling with LikelihoodProfiler.jl...")
        
        @time try
            # Pass the MLE and debug flag to the modernized profiling function
            prof_result = run_likelihood_profiling(petab_problem, multi_start_res.xmin, parsed_args["debug"])
            
            if !isnothing(prof_result)
                println("✅ Modern likelihood profiling completed successfully!")
                println("Profile result type: $(typeof(prof_result))")
            else
                @warn "Profiling returned no results"
            end
            flush(stdout)
        catch e
            @error "Failed to run modern likelihood profiling." exception=(e, catch_backtrace())
        end
    end

    println("\n--- Full Analysis Complete ---"); flush(stdout)
end

run_analysis()