# main.jl

using Pkg
Pkg.activate("bngl_julia/")

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
using ReverseDiff
using OrdinaryDiffEq
using Optim
using Sundials

# --- NEW: Define defaults for PEtab TSV files ---
const DEFAULT_TIME_COURSE_MEASUREMENTS = "SimData/measurements_time_course.tsv"
const DEFAULT_DOSE_RESPONSE_MEASUREMENTS = "SimData/measurements_dose_response.tsv"
const DEFAULT_MODEL_NET = "model_even_smaller/2025_06_29__23_19_55/model_even_smaller.net"


function setup_multiprocessing()
    # Helper to get CPU info from Slurm or the system
    function get_slurm_cpus()
        if haskey(ENV, "SLURM_CPUS_PER_TASK")
            try
                return parse(Int, ENV["SLURM_CPUS_PER_TASK"])
            catch
                @warn "Could not parse SLURM_CPUS_PER_TASK. Defaulting to system threads."
                return Sys.CPU_THREADS
            end
        else
            @info "Not running in a Slurm task. Defaulting to system threads."
            return Sys.CPU_THREADS
        end
    end

    try
        num_procs_to_add = get_slurm_cpus() - 1
        
        if num_procs_to_add > 0
            sysimage_path = unsafe_string(Base.JLOptions().image_file)
            
            if isempty(sysimage_path)
                @warn "Main process was not started with a system image. Workers will also start without one."
                addprocs(num_procs_to_add; exeflags=`--project=$(Base.active_project())`)
            else
                println("INFO: Main process using system image at '$sysimage_path'. Propagating to workers."); flush(stdout)
                addprocs(num_procs_to_add; exeflags=`--project=$(Base.active_project()) -J$(sysimage_path)`)
            end
            println("INFO: Added $(num_procs_to_add) worker processes."); flush(stdout)
        else
            println("INFO: Running on a single process."); flush(stdout)
        end
    catch e
        @warn "Could not add processes. Running in serial. Error: $e"
        flush(stderr)
    end
end

# Make sure all workers have the necessary packages (must be at top level)
@everywhere using PEtab
@everywhere using Optim
@everywhere using Sundials
@everywhere using SciMLSensitivity
@everywhere using ReverseDiff

function parse_commandline()
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
            help = "Number of multi-starts. Defaults to the number of available processes in parallel mode, or 1 in serial mode."
            arg_type = Int
            default = 0 # Will be dynamically set later
        "--optimizer"
            help = "Optimization algorithm to use. Options: " * join(keys(SUPPORTED_OPTIMIZERS), ", ")
            arg_type = String
            default = "LBFGS"
        "--abstol"
            help = "Absolute tolerance for the ODE solver."
            arg_type = Float64
            default = 1e-8 # Tighter default tolerance
        "--reltol"
            help = "Relative tolerance for the ODE solver."
            arg_type = Float64
            default = 1e-8 # Tighter default tolerance
        "--net-file"
            help = "Path to the BioNetGen .net file."
            arg_type = String
            default = DEFAULT_MODEL_NET
        "--config"
            help = "Path to the YAML config file for observable mapping and bounds."
            arg_type = String
            default = "config.yml"
        "--debug"
            help = "Enable debug mode with shorter time limits and looser tolerances for faster testing."
            action = :store_true
        "--measurements-file"
            help = "Path to the PEtab measurements file. Overrides the default for the selected mode."
            arg_type = String
            default = ""
        "--conditions-file"
            help = "Path to the PEtab conditions file. Required if using a custom measurements file."
            arg_type = String
            default = ""
        "--observables-file"
            help = "Path to the PEtab observables file. Optional."
            arg_type = String
            default = ""
        "--parameters-file"
            help = "Path to the PEtab parameters file. Optional."
            arg_type = String
            default = ""
    end
    return parse_args(s)
end

function run_analysis()
    parsed_args = parse_commandline()

    # Apply debug mode adjustments
    if parsed_args["debug"]
        println("INFO: Debug mode enabled - using faster, less accurate settings")
        parsed_args["abstol"] = 1e-4
        parsed_args["reltol"] = 1e-4
        # In debug mode, run only a few starts
        parsed_args["n-starts"] = parsed_args["n-starts"] == 0 ? 5 : min(parsed_args["n-starts"], 10)  # Allow up to 10 starts
        println("INFO: Debug tolerances set to abstol=$(parsed_args["abstol"]), reltol=$(parsed_args["reltol"])")
    end

    # Setup multiprocessing based on environment
    if parsed_args["parallel"]
        setup_multiprocessing()
    end

    # Dynamically set n_starts if not provided by the user
    if parsed_args["n-starts"] == 0
        if parsed_args["parallel"]
            parsed_args["n-starts"] = nprocs()
            println("INFO: --n-starts not provided, defaulting to nprocs(): $(parsed_args["n-starts"])")
        else
            parsed_args["n-starts"] = 10 # A reasonable number for a thorough serial run
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
    # The --data-file argument is now the primary way to specify the data
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
    # THIS IS THE KEY CHANGE: We are now calling your original setup function again.
    println("INFO: Setting up PEtab Model programmatically..."); flush(stdout)
    @time setup_results = setup_petab_problem(enable_preeq, net_file, data_file, config_file)
    if isnothing(setup_results)
        @error "Failed to build PEtabModel. Cannot proceed."
        return
    end

    # --- 3. Define robust solver options for simulation and steady-state ---
    println("INFO: Defining dedicated solvers for simulation and steady-state..."); flush(stdout)
    
    local odesol, steadystate_solver, gradient_method

    if parsed_args["debug"]
        println("INFO: Debug mode - using Rodas5P with ForwardDiff for faster compilation")
        # Use a pure-Julia solver for the main problem
        odesol = ODESolver(Rodas5P(), abstol=parsed_args["abstol"], reltol=parsed_args["reltol"])
        
        # THIS IS THE FIX: Use the :Simulate method for the steady-state problem in debug mode.
        # It is universally compatible and avoids the TypeError.
        steadystate_solver = SteadyStateSolver(:Simulate,
                                               abstol=parsed_args["abstol"], 
                                               reltol=parsed_args["reltol"])
        gradient_method = :ForwardDiff

    else # Normal (non-debug) mode
        println("INFO: Normal mode - using CVODE_BDF with Adjoint for accuracy")
        # Use the robust Sundials solver for the main problem
        odesol = ODESolver(CVODE_BDF(linear_solver=:GMRES), 
            abstol=parsed_args["abstol"], 
            reltol=parsed_args["reltol"],
            maxiters=200000
        )
        
        # Use the :Simulate method for steady-state as well to avoid rootfinding algorithm issues
        steadystate_solver = SteadyStateSolver(:Simulate,
                                               abstol=parsed_args["abstol"], 
                                               reltol=parsed_args["reltol"])
        gradient_method = :Adjoint
    end

    # --- 4. Run estimation ONLY if no results were loaded ---
    local petab_problem # Declare here to have it in the outer scope
    if isnothing(multi_start_res)
        println("INFO: Building PEtabODEProblem for estimation..."); flush(stdout)
        
        @time petab_problem = PEtabODEProblem(
            setup_results, # Use the result from setup_petab_problem
            odesolver = odesol,
            ss_solver = steadystate_solver,
            gradient_method=gradient_method,
            sensealg = gradient_method == :Adjoint ? InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)) : nothing,
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
            setup_results,
            odesolver = odesol,
            ss_solver = steadystate_solver,
            gradient_method=gradient_method,
            sensealg = gradient_method == :Adjoint ? InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)) : nothing,
            verbose=false
        )
    else
        println("INFO: Reusing existing PEtabODEProblem for visualization."); flush(stdout)
    end

    # --- 6. Generate Plots and Final Visualizations ---
    if !isnothing(multi_start_res)
        println("\n--- Generating Waterfall Plot ---"); flush(stdout)
        plot_waterfall(multi_start_res)
        println("\n-- Print Parameter Distribution Plot ---"); flush(stdout)
        plot_parameter_distribution(multi_start_res, petab_problem)
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