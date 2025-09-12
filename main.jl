# main.jl

# Headless plotting: avoid GKS display errors on headless environments
ENV["GKSwstype"] = "100"

# --- 1. INCLUDES and USING STATEMENTS ---
include("src/model_param_est_robustness.jl")
include("src/visualization.jl")
include("src/optimization.jl")
include("src/profiling.jl")
include("src/identifiability.jl")

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
        "--ident"
            help = "Run identifiability diagnostics (FIM, eigen, coordinate metric) after collation."
            action = :store_true
        "--load-fit"
            help = "Path to a .jld2 file containing best-fit parameters for profiling."
            arg_type = String
            default = "best_fit.jld2"
        "--profiling-method"
            help = "Method for profiling. Options: cico, fixedstep, manual"
            arg_type = String
            default = "fixedstep"
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
        odesolver = ODESolver(Rodas5P(), abstol=1e-8, reltol=1e-8)
        # Remove ss_solver since pre-equilibration is disabled
        
        petab_problem = PEtabODEProblem(
                        setup_results.petab_model, 
                        odesolver=odesolver,
                        split_over_conditions=true)
        
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
        odesolver = ODESolver(Rodas5P(), abstol=1e-8, reltol=1e-8)
        petab_problem = PEtabODEProblem(
                            setup_results.petab_model, 
                            odesolver=odesolver,
                            split_over_conditions=true)

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

        # --- Optional: Identifiability diagnostics ---
        if parsed_args["ident"]
            @info "--- Running identifiability diagnostics at best fit ---"
            
            # Force best_params to be a plain Vector of values
            param_values = if best_params isa ComponentVector
                collect(best_params)  # Extract just the values
            else
                collect(best_params)  # Ensure it's a Vector
            end
            
            # Systematically get the correct parameter names that PEtab expects
            # Method 1: Try to extract from the parameter table with scaling info
            function get_scaled_parameter_names(petab_problem)
                try
                    # Access the parameter table from the PEtab model
                    param_table = petab_problem.model_info.model.petab_tables[:parameters]
                    
                    # Get parameter IDs and their scales
                    param_ids = String.(param_table.parameterId)
                    
                    # Check if parameterScale column exists (PEtab 1.0 format)
                    if :parameterScale in names(param_table)
                        scales = String.(param_table.parameterScale)
                        scaled_names = Symbol[]
                        
                        for (id, scale) in zip(param_ids, scales)
                            if scale == "log10"
                                push!(scaled_names, Symbol("log10_$id"))
                            elseif scale == "log"
                                push!(scaled_names, Symbol("log_$id"))
                            else  # "lin" or other
                                push!(scaled_names, Symbol(id))
                            end
                        end
                        return scaled_names
                    else
                        # PEtab 2.0+ format or scale info not available
                        # Fall back to trying both scaled and unscaled names
                        return nothing
                    end
                catch
                    return nothing
                end
            end
            
            # Method 2: Use trial-and-error with a test call to discover the expected names
            function discover_parameter_names(petab_problem, param_values)
                # First try with base names
                base_names = petab_problem.xnames
                θ_test = ComponentArray(NamedTuple{Tuple(base_names)}(param_values))
                
                try
                    petab_problem.nllh(θ_test; prior=false)
                    return base_names  # Success with base names
                catch e
                    if e isa PEtab.PEtabInputError && occursin("must appear in the order of", string(e))
                        # Extract the expected names from the error message
                        error_msg = string(e)
                        # Find the part with the expected parameter list
                        start_idx = findfirst('[', error_msg)
                        end_idx = findlast(']', error_msg)
                        if start_idx !== nothing && end_idx !== nothing
                            names_str = error_msg[start_idx+1:end_idx-1]
                            # Parse the symbol names
                            expected_names = Symbol[]
                            for name_match in eachmatch(r":(\w+)", names_str)
                                push!(expected_names, Symbol(name_match.captures[1]))
                            end
                            return expected_names
                        end
                    end
                    rethrow(e)
                end
            end
            
            # Try systematic approach first, then fallback to discovery
            scaled_names = get_scaled_parameter_names(petab_problem)
            if scaled_names === nothing
                @info "Using parameter name discovery method..."
                scaled_names = discover_parameter_names(petab_problem, param_values)
            else
                @info "Using systematic parameter scaling method..."
            end
            
            @assert length(scaled_names) == length(param_values)
            
            # Construct ComponentArray with the correct names
            θ_full = ComponentArray(NamedTuple{Tuple(scaled_names)}(param_values))
            
            @info "Using parameter names: $(collect(keys(θ_full)))"
            
            # Sanity checks
            try
                _ = petab_problem.nllh(θ_full; prior=false)
                _ = petab_problem.simulated_values(θ_full)
                @info "Sanity checks passed"
            catch e
                @error "Sanity check failed before identifiability: $e"
                rethrow()
            end
            
            run_identifiability(petab_problem, θ_full; eps=1e-4)
        end

    elseif parsed_args["profile"]
        @info "--- Running in PROFILING mode ---"
        
        fit_data = JLD2.load(parsed_args["load-fit"])
        best_mle = fit_data["best_mle"]
        
        setup_results = setup_petab_problem(parsed_args["yaml"])
        profiling_odesol = ODESolver(Rodas5P(), abstol=1e-8, reltol=1e-8)
        # Remove profiling_ss_solver since pre-equilibration is disabled

        # Pass the method as a Symbol
        prof_method = Symbol(parsed_args["profiling-method"])

        run_likelihood_profiling(
            setup_results.petab_model, 
            profiling_odesol, 
            nothing,  # Pass nothing for steadystate_solver
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