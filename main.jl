# main.jl

using Pkg
Pkg.activate("bngl_julia/")

include("src/model_param_est.jl")
include("src/visualization.jl")

using Distributed
using LinearAlgebra
using ArgParse
using JLD2 
using ComponentArrays
using Plots

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

const SUPPORTED_OPTIMIZERS = Dict(
    "LBFGS" => LBFGS(),
    "IPNewton" => IPNewton()
)

function parse_commandline()
    s = ArgParseSettings(description="Run parameter estimation and visualization.")
    @add_arg_table! s begin
        "--parallel"
            help = "Run parameter estimation using multi-processing."
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
    end
    return parse_args(s)
end

function run_parameter_estimation(parsed_args)
    use_parallel = parsed_args["parallel"]
    optimizer_choice_str = parsed_args["optimizer"]
    abstol = parsed_args["abstol"]
    reltol = parsed_args["reltol"]

    n_starts = parsed_args["n-starts"]
    if n_starts == 0
        n_starts = use_parallel ? length(procs()) : 1
    end

    if !haskey(SUPPORTED_OPTIMIZERS, optimizer_choice_str)
        @error "Unsupported optimizer '$optimizer_choice_str'. Please choose from $(keys(SUPPORTED_OPTIMIZERS))"
        return nothing, nothing
    end
    optimizer = SUPPORTED_OPTIMIZERS[optimizer_choice_str]
    optim_options = Optim.Options(time_limit=600.0)

    println("No valid saved results found. Running parameter estimation..."); flush(stdout)
    
    local setup_results
    println("\n[Timing] Setting up PEtab Model..."); flush(stdout)
    @time setup_results = setup_petab_problem()
    if isnothing(setup_results)
        @error "Failed to build PEtabModel. Cannot proceed."
        return nothing, nothing
    end

    # --- FINAL RECOMMENDED CONFIGURATION ---
    # Reverting to the original :GMRES solver, but making it compatible by using a dense Jacobian.
    # This is the most robust combination and avoids all previously seen errors.
    println("\n[Timing] Building PEtabODEProblem..."); flush(stdout)
    local petab_problem
    @time petab_problem = PEtabODEProblem(setup_results.petab_model;
                                          gradient_method=:Adjoint,
                                          sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
                                          odesolver=ODESolver(CVODE_BDF(linear_solver=:GMRES), abstol=abstol, reltol=reltol),
                                          sparse_jacobian=false, # <-- The key fix
                                          verbose=false)
    # --- END FIX ---                                     

    local saved_results = nothing
    println("\n[Timing] Calibrating parameters..."); flush(stdout)
    @time if use_parallel
        println("Mode: PARALLEL using $(length(procs())) processes, $n_starts starts, and optimizer $optimizer_choice_str")
        multi_start_res = calibrate_multistart(petab_problem, optimizer, n_starts;
                                               nprocs=length(procs()),
                                               dirsave=joinpath(pwd(), "Intermediate_results"),
                                               options=optim_options)
        valid_runs = filter(r -> !isnothing(r) && !isnan(r.fmin) && isfinite(r.fmin), multi_start_res.runs)

        if isempty(valid_runs)
            @warn "All $n_starts optimization starts failed to produce a valid solution."
            return nothing, setup_results
        else
            println("INFO: Found $(length(valid_runs)) valid solution(s) out of $n_starts starts.")
            best_run = valid_runs[argmin([r.fmin for r in valid_runs])]
            saved_results = (theta_optim=best_run.xmin, cost=best_run.fmin,
                             names_est_opt=string.(propertynames(best_run.xmin)))
        end
    else
        println("Mode: SERIAL with optimizer $optimizer_choice_str, $n_starts start(s)")
        start_guesses = get_startguesses(petab_problem, n_starts)
        best_res = nothing
        for i in 1:n_starts
            println("  Serial Start $i/$n_starts...")
            res = calibrate(petab_problem, optimizer, start_guesses[i]; options=optim_options)
            if isnothing(best_res) || res.fmin < best_res.fmin
                best_res = res
            end
        end
        saved_results = (theta_optim=best_res.xmin, cost=best_res.fmin,
                         names_est_opt=string.(propertynames(best_res.xmin)))
    end

    return saved_results, setup_results
end

function run_analysis()
    parsed_args = parse_commandline()
    output_filename = parsed_args["output"]

    println("--- Starting Full Analysis ---"); flush(stdout)
    println("Using output file: '$output_filename'"); flush(stdout)

    local saved_results = nothing

    if isfile(output_filename)
        println("Found existing '$output_filename'. Attempting to load results..."); flush(stdout)
        try
            JLD2.@load output_filename saved_results
            println("Successfully loaded 'saved_results' key!"); flush(stdout)
        catch e
            if e isa KeyError
                println("Warning: File is old or malformed (KeyError). Will re-run estimation."); flush(stdout)
            else
                rethrow(e)
            end
        end
    end

    local setup_results
    
    if isnothing(saved_results)
        saved_results, setup_results = run_parameter_estimation(parsed_args)

        if isnothing(saved_results)
            @error "Parameter estimation failed. Cannot proceed."
            return
        end

        try
            JLD2.@save output_filename saved_results
            println("INFO: New estimation output saved to '$output_filename'"); flush(stdout)
        catch e
            @error "Failed to save new '$output_filename'. Error: $e"
        end
    end

    println("\n--- Setting up objects for Visualization ---"); flush(stdout)
    if !@isdefined(setup_results) || isnothing(setup_results)
        println("\n[Timing] Setting up PEtab Model for visualization..."); flush(stdout)
        @time setup_results = setup_petab_problem()
        if isnothing(setup_results)
            @error "Failed to setup problem for visualization."
            return
        end
    end
    
    local vis_petab_problem
    println("\n[Timing] Building PEtabODEProblem for visualization..."); flush(stdout)
    @time vis_petab_problem = PEtabODEProblem(setup_results.petab_model, verbose=false)

    println("\n--- Starting Visualization ---"); flush(stdout)
    println("\n[Timing] Running visualization..."); flush(stdout)
    @time try
        run_visualization(
            collect(saved_results.theta_optim),
            collect(saved_results.names_est_opt),
            setup_results.petab_params_list,
            setup_results.meas,
            vis_petab_problem,
            setup_results.observables_petab_dict,
            setup_results.sf_param_map,
            setup_results.rsys,
            setup_results.prn,
            setup_results.il6_species_name
        )
        println("✅ Visualization completed successfully!"); flush(stdout)
    catch e
        @error "Visualization failed" exception=(e, catch_backtrace())
    end

    println("\n--- Full Analysis Complete ---"); flush(stdout)
end

run_analysis()