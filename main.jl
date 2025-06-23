# main.jl

using Pkg
Pkg.activate("bngl_julia/")

include("src/parameter_utils.jl") 
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

# NEW MAIN WORKFLOW
function run_analysis()
    parsed_args = parse_commandline()
    output_filename = parsed_args["output"]
    use_parallel = parsed_args["parallel"]
    optimizer_choice_str = parsed_args["optimizer"]
    abstol = parsed_args["abstol"]
    reltol = parsed_args["reltol"]
    n_starts = parsed_args["n-starts"] == 0 ? (use_parallel ? length(procs()) : 1) : parsed_args["n-starts"]
    optimizer = SUPPORTED_OPTIMIZERS[optimizer_choice_str]
    optim_options = Optim.Options(time_limit=1800.0) # Increased time limit for stages

    println("--- Starting Staged Fitting Analysis ---"); flush(stdout)

    # ==========================
    # --- STAGE 1: Fit TREG ----
    # ==========================
    println("\n\n--- STAGE 1: FITTING TREG DATA ---"); flush(stdout)
    setup_stage1 = setup_petab_problem(stage=:TREG)
    petab_problem_stage1 = PEtabODEProblem(setup_stage1.petab_model;
                                           gradient_method=:Adjoint,
                                           sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
                                           odesolver=ODESolver(CVODE_BDF(linear_solver=:GMRES), abstol=abstol, reltol=reltol),
                                           sparse_jacobian=false, verbose=false)

    println("Calibrating TREG pathway parameters with $n_starts starts..."); flush(stdout)
    multi_start_res_s1 = calibrate_multistart(petab_problem_stage1, optimizer, n_starts; nprocs=length(procs()), options=optim_options)
    best_run_s1 = multi_start_res_s1.runs[argmin([r.fmin for r in multi_start_res_s1.runs])]
    
    # Untransform the best parameters from this stage to be used in the next
    params_s1_dict = untransform_parameters(best_run_s1.xmin, petab_problem_stage1.petab_model.parameter_map)
    println("✅ Stage 1 complete. Best cost: $(best_run_s1.fmin)")


    # ==========================
    # --- STAGE 2: Fit TH17 ---
    # ==========================
    println("\n\n--- STAGE 2: FITTING TH17 DATA (fixing TREG params) ---"); flush(stdout)
    # Pass the optimized TREG params to be fixed
    setup_stage2 = setup_petab_problem(stage=:TH17, params_to_fix=params_s1_dict)
    petab_problem_stage2 = PEtabODEProblem(setup_stage2.petab_model;
                                           gradient_method=:Adjoint,
                                           sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
                                           odesolver=ODESolver(CVODE_BDF(linear_solver=:GMRES), abstol=abstol, reltol=reltol),
                                           sparse_jacobian=false, verbose=false)

    println("Calibrating TH17 pathway parameters with $n_starts starts..."); flush(stdout)
    multi_start_res_s2 = calibrate_multistart(petab_problem_stage2, optimizer, n_starts; nprocs=length(procs()), options=optim_options)
    best_run_s2 = multi_start_res_s2.runs[argmin([r.fmin for r in multi_start_res_s2.runs])]
    
    params_s2_dict = untransform_parameters(best_run_s2.xmin, petab_problem_stage2.petab_model.parameter_map)
    println("✅ Stage 2 complete. Best cost: $(best_run_s2.fmin)")


    # =================================
    # --- STAGE 3: Global Fine-Tune ---
    # =================================
    println("\n\n--- STAGE 3: FITTING ALL DATA (global fine-tuning) ---"); flush(stdout)
    # Combine the best parameters from both stages to use as an excellent starting guess
    combined_params = merge(params_s1_dict, params_s2_dict)

    setup_stage3 = setup_petab_problem(stage=:ALL, params_to_fix=nothing) # estimate all
    petab_problem_stage3 = PEtabODEProblem(setup_stage3.petab_model;
                                            gradient_method=:Adjoint,
                                            sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
                                            odesolver=ODESolver(CVODE_BDF(linear_solver=:GMRES), abstol=abstol, reltol=reltol),
                                            sparse_jacobian=false, verbose=false)

    # Create the single best start-guess for the final run
    start_guess_s3 = get_startguess(combined_params, petab_problem_stage3.petab_model.parameter_map)

    println("Performing final fine-tuning calibration..."); flush(stdout)
    # For the final stage, a single run from our excellent start-guess is often sufficient
    final_res = calibrate(petab_problem_stage3, optimizer, start_guess_s3; options=optim_options)
    println("✅ Stage 3 complete. Final best cost: $(final_res.fmin)")

    # --- Save final results ---
    saved_results = (theta_optim=final_res.xmin, cost=final_res.fmin,
                     names_est_opt=string.(propertynames(final_res.xmin)))

    try
        JLD2.@save output_filename saved_results
        println("INFO: Final staged fitting output saved to '$output_filename'"); flush(stdout)
    catch e
        @error "Failed to save final results to '$output_filename'. Error: $e"
    end


    # --- Visualization ---
    println("\n\n--- Starting Final Visualization ---"); flush(stdout)
    
    println("Visualizing final results using the PEtab-aware simulation engine...")
    try
        # The `petab_problem_stage3` is the full problem definition used for the final fit.
        # We pass it along with the final optimized parameters to the new visualization function,
        # which correctly handles pre-equilibration.
        run_visualization(
            final_res.xmin,       # The final optimized parameter vector
            petab_problem_stage3  # The full problem definition for simulation
        )
        println("✅ Visualization completed successfully!"); flush(stdout)
    catch e
        @error "Visualization failed" exception=(e, catch_backtrace())
    end

    println("\n--- Staged Fitting Analysis Complete ---"); flush(stdout)
end

run_analysis()