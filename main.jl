# main.jl

# --- 1. INCLUDES and USING STATEMENTS ---
include("src/model_param_est_robustness.jl")
include("src/visualization.jl")
include("src/optimization.jl")
include("src/profiling.jl")
include("src/config_resolve.jl")

using ArgParse, JLD2, Logging, PEtab, SciMLSensitivity, ReverseDiff, DiffEqCallbacks, OrdinaryDiffEq, Sundials, LinearAlgebra, CSV, DataFrames, ComponentArrays, YAML
using .ConfigResolve

# --- 2. ARGUMENT PARSER DEFINITION ---
function define_argument_parser()
    s = ArgParseSettings(description="Run PEtab parameter estimation in batches.")
    @add_arg_table! s begin
        "--yaml", "-y"
            help = "Path to the PEtab YAML file."
            arg_type = String
            default = "petab_problem.yml"
        "--optimizer"
            help = "Optimization algorithm. Options: IPNewton, LBFGS, Fides"
            arg_type = String
            default = "IPNewton"
        "--n-starts"
            help = "The TOTAL number of multi-starts to run across all batches."
            arg_type = Int
            default = 500 # Default to your total
        "--n-batches"
            help = "The total number of batches the work is divided into."
            arg_type = Int
            default = 16 # e.g., 500 starts / 32 cpus_per_batch ≈ 16
        "--batch-id"
            help = "The ID of the current batch job (from SLURM_ARRAY_TASK_ID)."
            arg_type = Int
            default = 0 # A non-zero value triggers batch worker mode
        "--n-procs"
            help = "Number of processes to use for each batch via PEtab.jl's Distributed computing."
            arg_type = Int
            default = 32 # Should match --cpus-per-task
        "--collate"
            help = "Run in collation mode to find the best result from all batch outputs."
            action = :store_true
        "--profile"
            help = "Run likelihood profiling on a best-fit parameter set."
            action = :store_true
        "--load-fit"
            help = "Path to a .jld2 file containing best-fit parameters for profiling."
            arg_type = String
            default = "best_fit.jld2"
        "--debug"
            help = "Enable debug mode for faster, less accurate testing."
            action = :store_true
    end
    return s
end

# --- 3. MAIN CONTROLLER FUNCTION ---
function main()
    # Basic setup
    parsed_args = parse_args(ARGS, define_argument_parser())
    if parsed_args["debug"]
        global_logger(ConsoleLogger(stderr, Logging.Debug))
    else
        global_logger(ConsoleLogger(stderr, Logging.Info))
    end
    
    # --- CONFIG-DRIVEN PETAB RESOLUTION ---
    paths = resolve_petab_paths("config.yml")
    log_resolved_files(paths)
    
    # Create temporary YAML with resolved paths
    temp_yaml = ".petab_resolved.yaml"
    write_temp_petab_yaml(temp_yaml, paths)
    @info "Created temporary PEtab YAML: $temp_yaml"
    
    # --- MODE SELECTION ---
    if parsed_args["batch-id"] > 0
        # --- BATCH WORKER MODE ---
        @info "--- Running in BATCH WORKER mode for Batch ID: $(parsed_args["batch-id"]) ---"
        
        setup_results = setup_petab_problem(temp_yaml)
        odesolver = ODESolver(KenCarp47(autodiff=false), abstol=1e-8, reltol=1e-8)
        ss_solver = SteadyStateSolver(:Simulate, abstol=1e-8, reltol=1e-8)
        
        petab_problem = PEtabODEProblem(setup_results.petab_model, odesolver=odesolver, ss_solver=ss_solver,
                                        gradient_method=:ForwardDiff)
        
        # This new function will handle the batch logic
        run_batch_optimization(parsed_args, petab_problem)

    elseif parsed_args["collate"]
        # --- COLLATION & VISUALIZATION MODE ---
        @info "--- Running in COLLATION & VISUALIZATION mode ---"
        
        n_batches = parsed_args["n-batches"]
        results_dir = "results" # The parent directory for all batch results
        
        best_cost = Inf
        best_params = nothing
        all_runs = PEtab.PEtabOptimisationResult[] # Store all valid runs from all batches
        best_file_info = ""

        @info "Scanning $n_batches batch directories inside '$results_dir'..."

        # Loop through each expected batch directory
        for i in 1:n_batches
            batch_dir = joinpath(results_dir, "batch_$(i)")
            if !isdir(batch_dir)
                @warn "Batch directory not found: $batch_dir. Skipping."
                continue
            end

            # Scan the results.csv file within the batch directory
            results_csv_path = joinpath(batch_dir, "results1.csv")
            params_csv_path = joinpath(batch_dir, "xmins1.csv")

            if isfile(results_csv_path) && isfile(params_csv_path)
                try
                    batch_res_df = CSV.read(results_csv_path, DataFrame)
                    batch_params_df = CSV.read(params_csv_path, DataFrame)
                    
                    # Process each run within the batch
                    for j in 1:nrow(batch_res_df)
                        fmin = batch_res_df[j, :fmin]
                        if isfinite(fmin)
                            # Reconstruct a minimal PEtabOptimisationResult for collation
                            param_names = propertynames(batch_params_df)[1:end-1]
                            xmin_vals = collect(batch_params_df[j, param_names])
                            xmin = ComponentArray(; (param_names .=> xmin_vals)...)
                            
                            # Create a minimal result object for plotting and comparison
                            run_result = PEtab.PEtabOptimisationResult(xmin, fmin, xmin, # x0 is not critical here
                                                                       Symbol(batch_res_df[j, :alg]), 
                                                                       batch_res_df[j, :niterations], 
                                                                       batch_res_df[j, :runtime], 
                                                                       [], [], # Traces are not loaded
                                                                       batch_res_df[j, :converged], nothing)
                            
                            push!(all_runs, run_result)

                            if fmin < best_cost
                                best_cost = fmin
                                best_params = xmin
                                best_file_info = "Batch $i, Run $j"
                            end
                        end
                    end
                catch e
                    @warn "Could not process results for batch $i. Error: $e"
                end
            end
        end
        
        if isnothing(best_params)
            @error "Collation failed: No valid result files found across all batch directories."
            return
        end

        @info "✅ Collation complete. Found $(length(all_runs)) valid runs in total."
        @info "   - Best result from: $(best_file_info)"
        @info "   - Best cost (nllh): $(best_cost)"
        
        # Reconstruct the PEtabMultistartResult object for plotting functions
        n_total_valid_runs = length(all_runs)
        multistart_res = PEtab.PEtabMultistartResult(best_params, best_cost, :collate, n_total_valid_runs, "Batched", nothing, all_runs)
        
        # Save the single best result AND the full multistart object for diagnostics
        JLD2.save("best_fit.jld2", Dict("best_mle" => best_params, "best_cost" => best_cost, "multistart_result" => multistart_res))
        @info "   - Best parameters and full multistart results saved to best_fit.jld2"

        # --- Set up the PEtab problem once for all plotting ---
        setup_results = setup_petab_problem(temp_yaml)
        odesolver = ODESolver(KenCarp47(autodiff=false), abstol=1e-8, reltol=1e-8)
        petab_problem = PEtabODEProblem(setup_results.petab_model, odesolver=odesolver)

        # --- Generate Diagnostic Plots ---
        @info "--- Generating diagnostic plots ---"
        plot_waterfall(multistart_res)
        plot_parameter_distribution(multistart_res, petab_problem, reference_values=setup_results.true_values)
        
        # --- Visualization Part ---
        @info "--- Starting visualization for the best fit ---"
        run_visualization(
            collect(best_params),
            petab_problem,
            odesolver
        )

    elseif parsed_args["profile"]
        @info "--- Running in PROFILING mode ---"
        
        fit_data = JLD2.load(parsed_args["load-fit"])
        best_mle = fit_data["best_mle"]
        
        setup_results = setup_petab_problem(temp_yaml)
        profiling_odesol = ODESolver(KenCarp47(autodiff=false), abstol=1e-8, reltol=1e-8)
        profiling_ss_solver = SteadyStateSolver(:Simulate, abstol=1e-8, reltol=1e-8)

        run_likelihood_profiling(setup_results.petab_model, profiling_odesol, profiling_ss_solver, 
                                 best_mle, setup_results.true_values)
    else
        @error "No mode selected. Please specify --batch-id > 0 to run a batch."
    end
end

# --- 4. SCRIPT EXECUTION ---
BLAS.set_num_threads(1)
main()