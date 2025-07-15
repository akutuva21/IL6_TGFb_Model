# main.jl

using Pkg
Pkg.activate("./bngl_julia")  # Explicit project path for main process

include("src/model_param_est_robustness.jl")
include("src/visualization.jl")
include("src/optimization.jl")
include("src/profiling.jl")

using Distributed
using LinearAlgebra
using ArgParse
using JLD2 
using ComponentArrays
using Plots
using PEtab
using SciMLSensitivity
using OrdinaryDiffEq
using Optim
using Sundials
using Optimization
using ProfileLikelihood

const DEFAULT_YAML_PATH = "petab_problem.yml"

# --- DEFINE AND PARSE ARGUMENTS (CLEANED FOR YAML-BASED WORKFLOW) ---
function define_argument_parser()
    s = ArgParseSettings(description="Run parameter estimation and visualization using a PEtab YAML file.")
    @add_arg_table! s begin
        "--yaml"
            help = "Path to the PEtab YAML problem file."
            arg_type = String
            default = "petab_problem.yml"
        "--parallel"
            help = "Run parameter estimation using multi-processing."
            action = :store_true
        "--output", "-o"
            help = "Path to the JLD2 output file for saving/loading results."
            arg_type = String
            default = "estimation_output_small.jld"
        "--n-starts"
            help = "Number of multi-starts. Defaults to processes in parallel, or 10 in serial."
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
# --- 2. TOP-LEVEL DISTRIBUTED SETUP ---
# ===================================================================
if PARSED_ARGS["parallel"]
    try
        # Use the SLURM variable to determine how many processes to add
        n_procs = haskey(ENV, "SLURM_CPUS_PER_TASK") ? parse(Int, ENV["SLURM_CPUS_PER_TASK"]) : Sys.CPU_THREADS
        # Add n_procs-1 workers. The main process is already running.
        addprocs(n_procs - 1)
        println("INFO: Successfully added $(nworkers()) worker processes."); flush(stdout)
    catch e
        @warn "Could not add processes. Running in serial. Error: $e"
        flush(stderr)
    end
end

const PROJECT_PATH = abspath(dirname(Base.active_project()))
println("INFO: Main process confirmed project path is $PROJECT_PATH")

# --- 3. LOAD PACKAGES ON ALL PROCESSES ---
if nworkers() > 0
    # Get the absolute project path from the main process
    const PROJECT_PATH = abspath(dirname(Base.active_project()))
    println("INFO: Main process project path is $(PROJECT_PATH). Broadcasting to workers."); flush(stdout)

    # This builds the code with the value of PROJECT_PATH before sending it.
    @everywhere @eval Main begin
        using Pkg
        Pkg.activate($PROJECT_PATH)
        
        using PEtab
        using Optim
        using Sundials
        using SciMLSensitivity
    end
    println("INFO: Packages loaded successfully on all workers."); flush(stdout)
end

if nworkers() > 0
    println("INFO: Verifying system image on all processes..."); flush(stdout)
    
    # Check main process
    main_sysimage = unsafe_string(Base.JLOptions().image_file)
    println("      From Main (ID 1): System image is '$(main_sysimage)'"); flush(stdout)

    # Check all workers
    @everywhere begin
        worker_sysimage = unsafe_string(Base.JLOptions().image_file)
        println("      From Worker (ID $(myid())): System image is '$(worker_sysimage)'")
    end
    flush(stdout)
end

function run_analysis()
    parsed_args = PARSED_ARGS  # Use globally parsed arguments

    # Apply debug mode adjustments
    if parsed_args["debug"]
        println("INFO: Debug mode enabled - using faster, less accurate settings")
        # In debug mode, run only a few starts for faster testing
        parsed_args["n-starts"] = parsed_args["n-starts"] == 0 ? 5 : min(parsed_args["n-starts"], 10)
        println("INFO: Debug mode will use faster tolerances and fewer starts")
    end

    # Dynamically set n_starts if not provided by the user
    if parsed_args["n-starts"] == 0
        if parsed_args["parallel"] && nworkers() > 0
            parsed_args["n-starts"] = nworkers()
            println("INFO: --n-starts not provided, defaulting to nworkers(): $(parsed_args["n-starts"])" )
        else
            parsed_args["n-starts"] = 10
            println("INFO: --n-starts not provided, defaulting to 10 for serial execution.")
        end
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
            JLD2.@load output_filename multi_start_res
            println("Successfully loaded 'multi_start_res' object!"); flush(stdout)
        catch e
            @warn "Could not load 'multi_start_res' from file. Will re-run estimation. Error: $e"
            multi_start_res = nothing # Ensure it's nothing on failure
        end
    end

    # --- 2. Setup the core PEtabModel from YAML file ---
    println("INFO: Setting up PEtab Model from YAML file..."); flush(stdout)
    
    @time setup_results = setup_petab_problem(yaml_path)
    if isnothing(setup_results)
        @error "Failed to build PEtabModel from '$yaml_path'. Cannot proceed."
        return
    end
    
    # Extract the PEtab model and true parameter values
    petab_model = setup_results.petab_model
    true_param_values = setup_results.true_values
    println("INFO: Successfully loaded PEtab model with $(length(true_param_values)) parameters")

    # --- 3. Define robust solver options ---
    println("INFO: Defining solvers for simulation and steady-state..."); flush(stdout)
    
    local odesol, steadystate_solver, gradient_method

    if parsed_args["debug"]
        println("INFO: Debug mode - using Rodas5P with ForwardDiff for faster compilation")
        odesol = ODESolver(Rodas5P(), abstol=1e-4, reltol=1e-4)
        steadystate_solver = SteadyStateSolver(:Simulate, abstol=1e-4, reltol=1e-4)
        gradient_method = :ForwardDiff
    else # Normal (non-debug) mode
        println("INFO: Normal mode - using Rodas5P with ForwardDiff for robust optimization")
        odesol = ODESolver(Rodas5P(), abstol=1e-8, reltol=1e-8, maxiters=200000)
        steadystate_solver = SteadyStateSolver(:Simulate, abstol=1e-8, reltol=1e-8, maxiters=200000)
        gradient_method = :ForwardDiff
    end

    # --- 4. Run estimation ONLY if no results were loaded ---
    local petab_problem # Declare here to have it in the outer scope
    if isnothing(multi_start_res)
        println("INFO: Building PEtabODEProblem for estimation..."); flush(stdout)
        
        @time petab_problem = PEtabODEProblem(
            petab_model, # Use the extracted petab_model
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
            JLD2.@save output_filename multi_start_res
            println("INFO: New estimation output saved to '$output_filename'"); flush(stdout)
        catch e
            @error "Failed to save new '$output_filename'. Error: $e"
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
        println("\n--- Generating Waterfall Plot ---"); flush(stdout)
        plot_waterfall(multi_start_res)
        
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
            odesol  # <-- PASS THE SOLVER HERE
        )
        println("✅ Visualization completed successfully!"); flush(stdout)
    catch e
        @error "Failed to generate visualization plots." exception=(e, catch_backtrace())
    end

    # --- 7. (NEW) Run Likelihood Profiling if Requested ---
    if parsed_args["profile"]
        # We no longer need to check for multistart_result, as profiling is now independent.
        println("INFO: Running likelihood profiling...")
        @time try
            # --- THE FIX: Call with only the petab_problem ---
            run_likelihood_profiling(petab_problem)
            println("✅ Likelihood profiling completed successfully!"); flush(stdout)
        catch e
            @error "Failed to run likelihood profiling." exception=(e, catch_backtrace())
        end
    end

    println("\n--- Full Analysis Complete ---"); flush(stdout)
end

run_analysis()