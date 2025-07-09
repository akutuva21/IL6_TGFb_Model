# main.jl

using Pkg
Pkg.activate("./bngl_julia")  # Explicit project path for main process

include("src/model_param_est_robustness.jl")
include("src/visualization.jl")
include("src/optimization.jl")

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

const DEFAULT_TIME_COURSE_MEASUREMENTS = "SimData/measurements_time_course.tsv"
const DEFAULT_DOSE_RESPONSE_MEASUREMENTS = "SimData/measurements_dose_response.tsv"
const DEFAULT_MODEL_NET = "model_even_smaller/2025_07_08__12_31_42/model_even_smaller.net"

# --- DEFINE AND PARSE ARGUMENTS (ONCE) ---
function define_argument_parser()
    s = ArgParseSettings(description="Run parameter estimation and visualization.")
    @add_arg_table! s begin
        "--mode"
            help = "Workflow mode. Options: 'time-course', 'dose-response'."
            arg_type = String
            default = "time-course"
        "--parallel"
            help = "Run parameter estimation using multi-processing."
            action = :store_true
        "--with-preeq"
            help = "Enable pre-equilibration before the main simulation."
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
            help = "Optimization algorithm to use. Options: " * join(keys(SUPPORTED_OPTIMIZERS), ", ")
            arg_type = String
            default = "LBFGS"
        "--abstol"
            help = "Absolute tolerance for the ODE solver."
            arg_type = Float64
            default = 1e-8
        "--reltol"
            help = "Relative tolerance for the ODE solver."
            arg_type = Float64
            default = 1e-8
        "--net-file"
            help = "Path to the BioNetGen .net file."
            arg_type = String
            default = DEFAULT_MODEL_NET
        "--config"
            help = "Path to the YAML config file for observable mapping and bounds."
            arg_type = String
            default = "config.yml"
        "--debug"
            help = "Enable debug mode for faster, less accurate testing."
            action = :store_true
        "--measurements-file"
            help = "Custom measurements file. Overrides defaults."
            arg_type = String
            default = ""
        "--conditions-file"
            help = "Custom conditions file. Required if using custom measurements."
            arg_type = String
            default = ""
        "--observables-file"
            help = "Custom observables file (optional)."
            arg_type = String
            default = ""
        "--parameters-file"
            help = "Custom parameters file (optional)."
            arg_type = String
            default = ""
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
        parsed_args["abstol"] = 1e-4
        parsed_args["reltol"] = 1e-4
        # In debug mode, run only a few starts
        parsed_args["n-starts"] = parsed_args["n-starts"] == 0 ? 5 : min(parsed_args["n-starts"], 50)  # Allow up to 10 starts
        println("INFO: Debug tolerances set to abstol=$(parsed_args["abstol"]), reltol=$(parsed_args["reltol"])")
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
    net_file = parsed_args["net-file"]
    config_file = parsed_args["config"]
    enable_preeq = parsed_args["with-preeq"]
    output_filename = parsed_args["output"]

    # Determine data file based on the selected mode
    mode = parsed_args["mode"]
    local data_file::String
    if !isempty(parsed_args["measurements-file"])
        data_file = parsed_args["measurements-file"]
        println("INFO: Using custom measurements file: '$data_file'")
    elseif mode == "time-course"
        data_file = DEFAULT_TIME_COURSE_MEASUREMENTS
        println("INFO: Running in 'time-course' mode with default data: '$data_file'")
    elseif mode == "dose-response"
        data_file = DEFAULT_DOSE_RESPONSE_MEASUREMENTS
        println("INFO: Running in 'dose-response' mode with default data: '$data_file'")
    else
        @error "Invalid mode: '$mode'. Must be 'time-course' or 'dose-response'."
        return
    end
    flush(stdout)


    println("INFO: The script will use the following output file: '$output_filename'")
    flush(stdout)

    println("--- Starting Full Analysis ---"); flush(stdout)
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

    # --- 2. Setup the core PEtabModel (do this only ONCE) ---
    println("INFO: Setting up PEtab Model programmatically..."); flush(stdout)
    @time setup_results = setup_petab_problem(enable_preeq, net_file, data_file, config_file)
    if isnothing(setup_results)
        @error "Failed to build PEtabModel. Cannot proceed."
        return
    end
    
    # Extract the PEtab model and true parameter values
    petab_model = setup_results.petab_model
    true_param_values = setup_results.true_values
    println("INFO: Extracted $(length(true_param_values)) true parameter values for reference plotting")

    # --- 3. Define robust solver options for simulation and steady-state ---
    println("INFO: Defining dedicated solvers for simulation and steady-state..."); flush(stdout)
    
    local odesol, steadystate_solver, gradient_method

    if parsed_args["debug"]
        println("INFO: Debug mode - using Rodas5P with ForwardDiff for faster compilation")
        # Use a pure-Julia solver for the main problem
        odesol = ODESolver(Rodas5P(), abstol=parsed_args["abstol"], reltol=parsed_args["reltol"])
        
        steadystate_solver = SteadyStateSolver(:Simulate,
                                               abstol=parsed_args["abstol"], 
                                               reltol=parsed_args["reltol"])
        gradient_method = :ForwardDiff

    else # Normal (non-debug) mode
        println("INFO: Normal mode - using Rodas5P with ForwardDiff for robust optimization")
        # Use the robust pure-Julia solver for the main problem
        odesol = ODESolver(Rodas5P(), 
            abstol=parsed_args["abstol"], 
            reltol=parsed_args["reltol"],
            maxiters=200000
        )
        
        # Use the :Simulate method for steady-state as well to avoid rootfinding algorithm issues
        steadystate_solver = SteadyStateSolver(:Simulate,
                                               abstol=parsed_args["abstol"], 
                                               reltol=parsed_args["reltol"],
                                               maxiters=200000)
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
            petab_problem
        )
        println("✅ Visualization completed successfully!"); flush(stdout)
    catch e
        @error "Failed to generate visualization plots." exception=(e, catch_backtrace())
    end

    println("\n--- Full Analysis Complete ---"); flush(stdout)
end

run_analysis()