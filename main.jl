# main.jl

using Pkg
Pkg.activate("./bngl_julia")

include("src/model_param_est_robustness.jl")
include("src/visualization.jl")
include("src/optimization.jl")

using LinearAlgebra
using ArgParse
using JLD2
using Base.Threads

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
            help = "Optimization algorithm to use. Options: LBFGS, BFGS, NelderMead"
            arg_type = String
            default = "LBFGS"
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
# --- 2. THREADING SETUP (DISTRIBUTED PROCESSING DISABLED) ---
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
                    
                    multi_start_res = PEtab.PEtabMultistartResult(
                        best_mle,            # xmin
                        best_cost,           # fmin
                        :LoadedFromFile,     # alg (placeholder)
                        1,                   # nmultistarts (placeholder)
                        "LoadedFromFile",    # sampling_method (placeholder)
                        nothing,             # dirsave
                        []                   # runs (empty vector)
                    )
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
    println("INFO: Defining solvers for simulation and steady-state..."); flush(stdout)
        
    local odesol, steadystate_solver, gradient_method

    if parsed_args["debug"]
        println("INFO: Debug mode - using Rodas5P with numerical Jacobian for better stiffness handling")
        odesol = ODESolver(Rodas5P(),
                            abstol=1e-4, 
                            reltol=1e-4,
                            dtmin=1e-12)
        steadystate_solver = SteadyStateSolver(:Simulate, abstol=1e-4, reltol=1e-4)
        gradient_method = :ForwardDiff
    else # Normal (non-debug) mode
        println("INFO: Normal mode - using QNDF with numerical Jacobian for maximum stiffness robustness")
        println("INFO: Rodas5P solver is specifically designed for very stiff biochemical systems")
        odesol = ODESolver(Rodas5P(),
                            abstol=1e-10,
                            reltol=1e-10, 
                            maxiters=1000000,
                            dtmin=1e-15)
        steadystate_solver = SteadyStateSolver(:Simulate, 
                                                abstol=1e-10,
                                                reltol=1e-10, 
                                                maxiters=400000)
        gradient_method = :ForwardDiff
    end

    println("INFO: Solver configured with domain safety checks and minimum timestep floor")
    println("INFO: This should significantly reduce 'dt ... NaN' warnings during optimization")

    # --- 4. Run estimation ONLY if no results were loaded ---
    local petab_problem
    if isnothing(multi_start_res)
        println("INFO: Building PEtabODEProblem for estimation..."); flush(stdout)
        
        @time petab_problem = PEtabODEProblem(
            petab_model,
            odesolver = odesol,
            ss_solver = steadystate_solver,
            gradient_method=gradient_method,
            verbose=false
        )
        
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
        @time petab_problem = PEtabODEProblem(
            petab_model,
            odesolver = odesol,
            ss_solver = steadystate_solver,
            gradient_method=gradient_method,
            verbose=false
        )
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