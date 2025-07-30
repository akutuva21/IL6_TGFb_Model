# main.jl

include("src/model_param_est_robustness.jl")
include("src/visualization.jl")
include("src/optimization.jl")
include("src/profiling.jl")

using LinearAlgebra
using ArgParse
using JLD2
using Base.Threads
using DiffEqCallbacks
using Logging

# --- Cluster/Batch Output Setup ---
if !isinteractive()
    global_logger(ConsoleLogger(stdout, Logging.Info))
    ENV["JULIA_DEBUG"] = "all"  # Optional: set to "all" for debug
end
ENV["PYTHONUNBUFFERED"] = "1"  # For PyCall/Python output

# Helper for periodic progress logging
function log_progress(i, total, task_name)
    if i % max(1, total ÷ 10) == 0
        percentage = round(Int, 100 * i / total)
        @info "$task_name: $percentage% complete ($i/$total)"
        flush(stdout); flush(stderr)
    end
end

function create_petab_problem_with_callbacks(petab_model, odesolver, steadystate_solver, gradient_method)
    @info "Adding PositiveDomain callback to PEtabModel to enforce non-negative concentrations."
    
    positive_domain_cb = PositiveDomain()
    combined_callbacks = CallbackSet(petab_model.callbacks, positive_domain_cb)

    petab_model_with_callback = PEtabModel(
        petab_model.name,
        petab_model.h,
        petab_model.u0!,
        petab_model.u0,
        petab_model.sd,
        petab_model.float_tspan,
        petab_model.paths,
        petab_model.sys,
        petab_model.sys_mutated,
        petab_model.parametermap,
        petab_model.speciemap,
        petab_model.petab_tables,
        combined_callbacks,
        petab_model.defined_in_julia
    )
    
    @time petab_problem = PEtabODEProblem(
            petab_model, # Use the extracted petab_model
            odesolver = odesolver,
            ss_solver = steadystate_solver,
            gradient_method=gradient_method,
            verbose=false
        )
    
    @info "✅ PEtabODEProblem created successfully"
    return petab_problem
end

function run_analysis()
    parsed_args = PARSED_ARGS  # Use globally parsed arguments

    # Setup logging level
    if parsed_args["debug"]
        global_logger(ConsoleLogger(stderr, Logging.Debug))
        @info "Debug mode enabled - using faster, less accurate settings and debug logging."
    else
        global_logger(ConsoleLogger(stderr, Logging.Info))
    end

    # Apply debug mode adjustments
    if parsed_args["debug"]
        # In debug mode, run only a few starts for faster testing
        parsed_args["n-starts"] = parsed_args["n-starts"] == 0 ? 5 : min(parsed_args["n-starts"], 10)
        @info "Debug mode will use faster tolerances and fewer starts"
    end

    # Dynamically set n_starts based on available threads
    if parsed_args["n-starts"] == 0
        n_threads = Threads.nthreads()
        parsed_args["n-starts"] = n_threads
        @info "--n-starts not provided, defaulting to $(parsed_args["n-starts"]) based on $(n_threads) threads"
    end

    # File paths from command line
    yaml_path = parsed_args["yaml"]
    output_filename = parsed_args["output"]

    @info "--- Starting Full Analysis ---"; flush(stdout); flush(stderr)
    @info "Using PEtab YAML file: '$yaml_path'"; flush(stdout); flush(stderr)
    @info "Using output file: '$output_filename'"; flush(stdout); flush(stderr)

    # --- 1. Load existing results if available ---
    local multi_start_res = nothing
    if isfile(output_filename)
        @info "Found existing '$output_filename'. Attempting to load results..."
        try
            best_mle = nothing
            best_cost = nothing
            
            try
                JLD2.@load output_filename best_mle best_cost
                if !isnothing(best_mle) && !isnothing(best_cost)
                    @info "✅ Successfully loaded essential estimation data!"
                    @info "  - Best cost: $best_cost"
                    @info "  - Parameter count: $(length(best_mle))"
                                        
                    # 1. Create a minimal PEtabOptimisationResult to represent the loaded best fit.
                    best_run_reconstructed = PEtab.PEtabOptimisationResult(
                        best_mle,         # xmin
                        best_cost,        # fmin
                        best_mle,         # x0 (can use xmin as a placeholder)
                        :LoadedFromFile,  # alg
                        0,                # niterations
                        0.0,              # runtime
                        [],               # xtrace
                        [],               # ftrace
                        true,             # converged
                        nothing           # original
                    )

                    # 2. Create the PEtabMultistartResult with a list containing our single reconstructed run.
                    multi_start_res = PEtab.PEtabMultistartResult(
                        best_mle,            # xmin (the overall best)
                        best_cost,           # fmin (the overall best)
                        :LoadedFromFile,     # alg (placeholder)
                        1,                   # nmultistarts (placeholder)
                        "LoadedFromFile",    # sampling_method (placeholder)
                        nothing,             # dirsave
                        [best_run_reconstructed] # runs <-- NOW CONTAINS ONE VALID RUN
                    )

                else
                    @warn "Essential data is incomplete. Will re-run estimation."
                    multi_start_res = nothing
                end
            catch
                @info "Attempting to load legacy format..."
                JLD2.@load output_filename multi_start_res
                
                if !isnothing(multi_start_res) && hasfield(typeof(multi_start_res), :xmin) && hasfield(typeof(multi_start_res), :fmin)
                    @info "Successfully loaded legacy 'multi_start_res' object!"
                    @info "  - Best cost: $(multi_start_res.fmin)"
                    @info "  - Parameter count: $(length(multi_start_res.xmin))"
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
    @info "Setting up PEtab Model from YAML file..."

    @time setup_results = setup_petab_problem(yaml_path)
    if isnothing(setup_results)
        @error "Failed to build PEtabModel from '$yaml_path'. Cannot proceed."
        return
    end

    petab_model = setup_results.petab_model
    true_param_values = setup_results.true_values
    @info "Successfully loaded PEtab model with $(length(true_param_values)) parameters"; flush(stdout); flush(stderr)

    # --- 3. Define robust solver options ---
    @info "Defining robust solver for simulation and steady-state..."
        
    local odesol, gradient_method

    if parsed_args["debug"]
        println("INFO: Debug mode - using Rodas5P with ForwardDiff for faster compilation")
        odesol = ODESolver(Rodas5P(), 
                          abstol=1e-4, 
                          reltol=1e-4,
                          dtmin=1e-12)
        steadystate_solver = SteadyStateSolver(:Simulate, abstol=1e-4, reltol=1e-4)
        gradient_method = :ForwardDiff
    else # Normal (non-debug) mode
        println("INFO: Normal mode - using Rodas5P with ForwardDiff for robust optimization")
        odesol = ODESolver(Rodas5P(), 
                          abstol=1e-10, 
                          reltol=1e-10, 
                          maxiters=400000,
                          dtmin=1e-12)
        steadystate_solver = SteadyStateSolver(:Simulate, 
                                             abstol=1e-10, 
                                             reltol=1e-10, 
                                             maxiters=400000)
        gradient_method = :ForwardDiff
    end

    @info "✅ Enhanced solver configuration completed:"; flush(stdout); flush(stderr)
    @info "   • ODE Solver: $(typeof(odesol.solver))"; flush(stdout); flush(stderr)
    @info "   • Absolute tolerance: $(odesol.abstol)"; flush(stdout); flush(stderr)
    @info "   • Relative tolerance: $(odesol.reltol)"; flush(stdout); flush(stderr)
    @info "   • Steady-state solver: Using PEtab.jl defaults"; flush(stdout); flush(stderr)
    @info "   • Expected benefits: Avoids CompositeAlgorithm conflicts, default steady-state handling"; flush(stdout); flush(stderr)

    # --- 4. Create PEtabProblem (centralized) ---
    local petab_problem
    @info "Building PEtabODEProblem..."
    petab_problem = create_petab_problem_with_callbacks(petab_model, odesol, steadystate_solver, gradient_method)

    # --- 5. Run estimation ONLY if no results were loaded ---
    if isnothing(multi_start_res)
        @info "Starting parameter estimation..."
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
            @info "✅ Essential estimation data saved successfully to '$output_filename'"
            @info "   (Saved MLE parameters and cost, avoiding JLD2 warnings)"
        catch e
            @error "Failed to save essential data to '$output_filename'. Error: $e"
        end
    end

    # --- 6. Generate Plots and Final Visualizations ---
    if !isnothing(multi_start_res)
        @info "\n--- Diagnosing Multistart Data ---"; flush(stdout); flush(stderr)
        diagnose_multistart_data(multi_start_res, petab_problem)
        
        @info "\n--- Generating Waterfall Plot ---"; flush(stdout); flush(stderr)
        try
            plot_waterfall(multi_start_res)
        catch e
            @warn "Primary waterfall plot failed: $e"; flush(stdout); flush(stderr)
            @info "Attempting fallback implementation..."; flush(stdout); flush(stderr)
            try
                plot_waterfall_custom_fallback(multi_start_res)
            catch e2
                @warn "Fallback waterfall plot also failed: $e2"; flush(stdout); flush(stderr)
                try
                    plot_waterfall_native_fallback(multi_start_res, petab_problem)
                catch e3
                    @error "All waterfall plot implementations failed. Last error: $e3"; flush(stdout); flush(stderr)
                end
            end
        end
        
        @info "\n--- Generating Parameter Distribution Plot ---"; flush(stdout); flush(stderr)
        
        # Use the true parameter values from the BNGL model as reference
        @info "Using true parameter values from BNGL model as reference"; flush(stdout); flush(stderr)
        
        plot_parameter_distribution(multi_start_res, petab_problem, reference_values=true_param_values)
    end

    saved_results = (
        theta_optim=multi_start_res.xmin, 
        cost=multi_start_res.fmin,
        names_est_opt=string.(propertynames(multi_start_res.xmin))
    )
    
    @info "\n--- Starting Visualization ---"; flush(stdout); flush(stderr)
    @info "\n[Timing] Running visualization..."; flush(stdout); flush(stderr)
    @time try
        run_visualization(
            collect(saved_results.theta_optim),
            petab_problem,
            odesol
        )
        @info "✅ Visualization completed successfully!"; flush(stdout); flush(stderr)
    catch e
        @error "Failed to generate visualization plots." exception=(e, catch_backtrace()); flush(stdout); flush(stderr)
    end

    # --- 7. Run Likelihood Profiling if Requested ---
    if parsed_args["profile"]
        @info "Running modern likelihood profiling with LikelihoodProfiler.jl..."; flush(stdout); flush(stderr)
        
        @time try
            # Pass the MLE and debug flag to the modernized profiling function
            prof_result = run_likelihood_profiling(petab_problem, multi_start_res.xmin, parsed_args["debug"])
            
            if !isnothing(prof_result)
                @info "✅ Modern likelihood profiling completed successfully!"; flush(stdout); flush(stderr)
                @info "Profile result type: $(typeof(prof_result))"; flush(stdout); flush(stderr)
            else
                @warn "Profiling returned no results"; flush(stdout); flush(stderr)
            end
        catch e
            @error "Failed to run modern likelihood profiling." exception=(e, catch_backtrace()); flush(stdout); flush(stderr)
        end
    end

    @info "\n--- Full Analysis Complete ---"; flush(stdout); flush(stderr)
end

# --- DEFINE AND PARSE ARGUMENTS ---
function define_argument_parser()
    s = ArgParseSettings(description="Run parameter estimation and visualization using a PEtab YAML file.")
    @add_arg_table! s begin
        "--yaml"
            arg_type = String
            default = "petab_problem.yml"
        "--output", "-o"
            help = "Path to the JLD2 output file for saving/loading results."
            arg_type = String
            default = "estimation_output_small.jld"
        "--n-starts"
            help = "Number of multi-starts. Defaults to thread-based parallel execution."
            arg_type = Int
            default = 0
        "--optimizer"
            help = "Optimization algorithm. Options: auto, Fides, IPNewton, LBFGS, BFGS (auto=most robust available)"
            arg_type = String
            default = "auto"
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
# --- 2. THREADING SETUP ---
# ===================================================================

println("INFO: Using threading for parallel processing")
println("INFO: Available threads: $(Threads.nthreads())")

const PROJECT_PATH = abspath(dirname(Base.active_project()))
println("INFO: Project path: $PROJECT_PATH")

run_analysis()