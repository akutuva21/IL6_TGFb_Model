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

# Define defaults for model and data files
const DEFAULT_MODEL_NET = "model_even_smaller/2025_06_26__19_02_01/model_even_smaller.net"
const DEFAULT_DATA_XLSX = "SimData/simulation_results.xlsx"

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

@everywhere using PEtab
@everywhere using Optim
@everywhere using Sundials

function parse_commandline()
    s = ArgParseSettings(description="Run parameter estimation and visualization.")
    @add_arg_table! s begin
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
        "--data-file"
            help = "Path to the experimental data XLSX file."
            arg_type = String
            default = DEFAULT_DATA_XLSX
        "--config"
            help = "Path to the YAML config file for observable mapping."
            arg_type = String
            default = "config.yml"
    end
    return parse_args(s)
end

function run_analysis()
    parsed_args = parse_commandline()
    # File paths from command line
    net_file = parsed_args["net-file"]
    data_file = parsed_args["data-file"]
    config_file = parsed_args["config"]
    enable_preeq = parsed_args["with-preeq"]
    output_filename = parsed_args["output"]

    # --- ADD THIS LINE FOR DEBUGGING ---
    println("INFO: The script will use the following output file: '", output_filename, "'")
    flush(stdout)

    println("--- Starting Full Analysis ---"); flush(stdout)
    println("Using output file: '$output_filename'"); flush(stdout)

    local multi_start_res = nothing

    if isfile(output_filename)
        println("Found existing '$output_filename'. Attempting to load results..."); flush(stdout)
        try
            # Load the entire multistart result object
            JLD2.@load output_filename multi_start_res
            println("Successfully loaded 'multi_start_res' object!"); flush(stdout)
        catch e
            @warn "Could not load 'multi_start_res' from file. Will re-run estimation. Error: $e"
        end
    end

    local setup_results
    
    if isnothing(multi_start_res)
        println("\n[Timing] Setting up PEtab Model..."); flush(stdout)
        @time setup_results = setup_petab_problem(enable_preeq, net_file, data_file, config_file)
        if isnothing(setup_results)
            @error "Failed to build PEtabModel. Cannot proceed."
            return
        end

        println("\n[Timing] Building PEtabODEProblem..."); flush(stdout)
        @time petab_problem = PEtabODEProblem(setup_results, verbose=false)

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

    if !isnothing(multi_start_res)
        println("\n--- Generating Waterfall Plot ---"); flush(stdout)
        plot_waterfall(multi_start_res)
        println("\n-- Print Parameter Distribution Plot ---"); flush(stdout)
        plot_parameter_distribution(multi_start_res)
    end

    saved_results = (
        theta_optim=multi_start_res.xmin, 
        cost=multi_start_res.fmin,
        names_est_opt=string.(propertynames(multi_start_res.xmin))
    )
    
    println("\n--- Setting up objects for Visualization ---"); flush(stdout)
    if !@isdefined(setup_results) || isnothing(setup_results)
        println("\n[Timing] Setting up PEtab Model for visualization..."); flush(stdout)
        @time setup_results = setup_petab_problem(enable_preeq, net_file, data_file, config_file)
        if isnothing(setup_results)
            @error "Failed to setup problem for visualization."
            return
        end
    end
    
    local vis_petab_problem
    println("\n[Timing] Building PEtabODEProblem for visualization..."); flush(stdout)
    @time vis_petab_problem = PEtabODEProblem(setup_results, verbose=false)

    println("\n--- Starting Visualization ---"); flush(stdout)
    println("\n[Timing] Running visualization..."); flush(stdout)
    @time try
        run_visualization(
            collect(saved_results.theta_optim),
            vis_petab_problem
        )
        println("✅ Visualization completed successfully!"); flush(stdout)
    catch e
        @error "Visualization failed" exception=(e, catch_backtrace())
    end

    println("\n--- Full Analysis Complete ---"); flush(stdout)
end

run_analysis()