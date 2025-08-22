# main.jl

# --- 1. INCLUDES and USING STATEMENTS ---
include("src/model_param_est_robustness.jl")
include("src/visualization.jl")
include("src/optimization.jl")
include("src/profiling.jl")

using ArgParse, JLD2, Logging, PEtab, SciMLSensitivity, ReverseDiff, DiffEqCallbacks, OrdinaryDiffEq, Sundials, LinearAlgebra, CSV, DataFrames, ComponentArrays

# --- 2. ARGUMENT PARSER DEFINITION ---
function define_argument_parser()
    s = ArgParseSettings(description="Run PEtab parameter estimation.")
    @add_arg_table! s begin
        "--yaml", "-y"
            help = "Path to the PEtab YAML file."
            arg_type = String
            default = "petab_problem.yml"
        "--optimizer"
            help = "Optimization algorithm."
            arg_type = String
            default = "Fides"
        "--n-starts"
            help = "Total number of multi-starts in the array."
            arg_type = Int
            default = 500
        "--task-id"
            help = "Worker ID for a job array task. Runs a single optimization."
            arg_type = Int
            default = 0
        "--collate"
            help = "Run in collation mode to find the best result."
            action = :store_true
        "--profile"
            help = "Run likelihood profiling on a best-fit parameter set."
            action = :store_true
        "--load-fit"
            help = "Path to a .jld2 file containing best-fit parameters for profiling."
            arg_type = String
            default = "best_fit.jld2"
        "--profiling-method"
            help = "Method for profiling. Options: cico, manual"
            arg_type = String
            default = "cico"
        "--debug"
            help = "Enable debug mode for faster, less accurate testing."
            action = :store_true
    end
    return s
end

# --- 3. MAIN CONTROLLER FUNCTION ---
function main()
    parsed_args = parse_args(ARGS, define_argument_parser())
    if parsed_args["debug"]
        global_logger(ConsoleLogger(stderr, Logging.Debug))
    else
        global_logger(ConsoleLogger(stderr, Logging.Info))
    end
    
    if parsed_args["task-id"] > 0
        # --- WORKER MODE ---
        @info "--- Running in WORKER mode for Task ID: $(parsed_args["task-id"]) ---"
        
        setup_results = setup_petab_problem(parsed_args["yaml"])
        odesolver = ODESolver(KenCarp47(autodiff=false), abstol=1e-8, reltol=1e-8)
        ss_solver = SteadyStateSolver(:Simulate, abstol=1e-8, reltol=1e-8)
        
        petab_problem = PEtabODEProblem(setup_results.petab_model, odesolver=odesolver, ss_solver=ss_solver,
                                        gradient_method=:Adjoint, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP()))
        
        run_single_optimization(parsed_args, petab_problem)

    elseif parsed_args["collate"]
        # --- COLLATION & VISUALIZATION MODE ---
        @info "--- Running in COLLATION & VISUALIZATION mode ---"
        
        n_starts = parsed_args["n-starts"]
        results_dir = "results"
        
        best_cost = Inf
        best_params = nothing
        all_runs = PEtab.PEtabOptimisationResult[]
        best_file_info = ""

        @info "Scanning up to $n_starts result files in '$results_dir'..."

        for i in 1:n_starts
            filepath = joinpath(results_dir, "run_$(i).jld2")
            if isfile(filepath)
                try
                    res_data = JLD2.load(filepath)
                    res = res_data["result"]
                    if !isnothing(res) && isfinite(res.fmin)
                        push!(all_runs, res)
                        if res.fmin < best_cost
                            best_cost = res.fmin
                            best_params = res.xmin
                            best_file_info = "File: $(basename(filepath))"
                        end
                    end
                catch e
                    @warn "Could not process result file $filepath. Error: $e"
                end
            end
        end
        
        if isnothing(best_params)
            @error "Collation failed: No valid result files found in '$results_dir'."
            return
        end

        @info "✅ Collation complete. Found $(length(all_runs)) valid runs."
        @info "   - Best result from: $(best_file_info)"
        @info "   - Best cost (nllh): $(best_cost)"
        
        multistart_res = PEtab.PEtabMultistartResult(best_params, best_cost, :collate, length(all_runs), "Array", nothing, all_runs)
        
        JLD2.save("best_fit.jld2", Dict("best_mle" => best_params, "best_cost" => best_cost, "multistart_result" => multistart_res))
        @info "   - Best parameters and full multistart results saved to best_fit.jld2"

        # --- Set up the PEtab problem once for all plotting ---
        setup_results = setup_petab_problem(parsed_args["yaml"])
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
        
        setup_results = setup_petab_problem(parsed_args["yaml"])
        profiling_odesol = ODESolver(KenCarp47(autodiff=false), abstol=1e-8, reltol=1e-8)
        profiling_ss_solver = SteadyStateSolver(:Simulate, abstol=1e-8, reltol=1e-8)

        # Pass the method as a Symbol
        prof_method = Symbol(parsed_args["profiling-method"])

        run_likelihood_profiling(
            setup_results.petab_model, 
            profiling_odesol, 
            profiling_ss_solver, 
            best_mle, 
            setup_results.true_values;
            profiling_method = prof_method # Pass the selected method
        )
    else
        @error "No mode selected. Please specify --task-id or --collate."
    end
end

# --- 4. SCRIPT EXECUTION ---
BLAS.set_num_threads(1)
main()