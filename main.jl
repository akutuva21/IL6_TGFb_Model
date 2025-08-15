# main.jl

# --- 1. INCLUDES and USING STATEMENTS ---
include("src/model_param_est_robustness.jl")
include("src/visualization.jl")
include("src/optimization.jl")
include("src/profiling.jl")

using ArgParse, JLD2, Logging, PEtab, SciMLSensitivity, ReverseDiff, DiffEqCallbacks, OrdinaryDiffEq, Sundials, LinearAlgebra

# --- 2. ARGUMENT PARSER DEFINITION ---
function define_argument_parser()
    s = ArgParseSettings(description="Run PEtab parameter estimation, collation, or profiling.")
    @add_arg_table! s begin
        "--yaml", "-y"
            help = "Path to the PEtab YAML file."
            arg_type = String
            default = "petab_problem.yml"
        "--output", "-o"
            help = "Path to the output file for the current task."
            arg_type = String
        "--n-starts"
            help = "Total number of multi-starts in the array."
            arg_type = Int
            default = 96
        "--optimizer"
            help = "Optimization algorithm. Options: IPNewton, LBFGS, Fides"
            arg_type = String
            default = "IPNewton"
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
    
    # --- MODE SELECTION ---
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
        
        # --- Collation Part ---
        n_starts = parsed_args["n-starts"]
        best_cost = Inf
        best_params = nothing
        best_file = ""

        for i in 1:n_starts
            filepath = joinpath("results", "run_$(i).jld2")
            if isfile(filepath)
                res = JLD2.load(filepath, "result")
                if !isnothing(res) && res.fmin < best_cost
                    best_cost = res.fmin
                    best_params = res.xmin
                    best_file = filepath
                end
            end
        end
        
        if isnothing(best_params)
            @error "Collation failed: No valid result files found in 'results/' directory."
            return
        end

        @info "✅ Collation complete. Best result found in: $(best_file)"
        @info "   - Best cost (nllh): $(best_cost)"
        
        JLD2.save("best_fit.jld2", Dict("best_mle" => best_params, "best_cost" => best_cost))
        @info "   - Best parameters saved to best_fit.jld2"

        # --- Visualization Part ---
        @info "--- Starting visualization for the best fit ---"
        # Set up the PEtab problem again to run the simulation
        setup_results = setup_petab_problem(parsed_args["yaml"])
        odesolver = ODESolver(KenCarp47(autodiff=false), abstol=1e-8, reltol=1e-8)
        petab_problem = PEtabODEProblem(setup_results.petab_model, odesolver=odesolver)

        # Call the visualization function with the best parameters we just found
        run_visualization(
            collect(best_params),
            petab_problem,
            odesolver
        )

    elseif parsed_args["profile"]
        # --- PROFILING MODE ---
        @info "--- Running in PROFILING mode ---"
        
        fit_data = JLD2.load(parsed_args["load-fit"])
        best_mle = fit_data["best_mle"]
        
        setup_results = setup_petab_problem(parsed_args["yaml"])
        profiling_odesol = ODESolver(KenCarp47(autodiff=false), abstol=1e-8, reltol=1e-8)
        profiling_ss_solver = SteadyStateSolver(:Simulate, abstol=1e-8, reltol=1e-8)

        run_likelihood_profiling(setup_results.petab_model, profiling_odesol, profiling_ss_solver, 
                                 best_mle, setup_results.true_values)
    else
        @error "No mode selected. Please specify --task-id, --collate, or --profile."
    end
end

# --- 4. SCRIPT EXECUTION ---
BLAS.set_num_threads(1)
main()