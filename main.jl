# main.jl

# Headless plotting: avoid GKS display errors on headless environments
ENV["GKSwstype"] = "100"

# --- 1. INCLUDES and USING STATEMENTS ---
include("src/model_param_est_robustness.jl")
include("src/visualization.jl")
include("src/optimization.jl")
include("src/profiling.jl")
include("src/identifiability.jl")

using ArgParse, JLD2, Logging, PEtab, SciMLSensitivity, ReverseDiff, DiffEqCallbacks, OrdinaryDiffEq, Sundials, LinearAlgebra, CSV, DataFrames, ComponentArrays, Printf

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
        
        petab_problem = PEtabODEProblem(
                        setup_results.petab_model, 
                        odesolver=odesolver,
                        split_over_conditions=true)
        
        run_single_optimization(parsed_args, petab_problem)

    elseif parsed_args["collate"]
        # --- COLLATION & VISUALIZATION MODE ---
        @info "--- Running in COLLATION & VISUALIZATION mode ---"
        
        # --- MODIFICATION: Set up the PEtab problem once at the beginning ---
        setup_results = setup_petab_problem(parsed_args["yaml"])
        odesolver = ODESolver(Rodas5P(), abstol=1e-8, reltol=1e-8)
        petab_problem = PEtabODEProblem(
                            setup_results.petab_model, 
                            odesolver=odesolver,
                            split_over_conditions=true)

        # --- Collation Loop ---
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

        # --- START: CORRECTED PI's QUESTION SECTION ---
        @info "--- Compare NLLH of Best-Fit vs. True Parameters ---"
        
        true_values_linear = setup_results.true_values
        estimated_param_ids = petab_problem.xnames # This is a Vector{Symbol} of the correctly scaled names
        xindices = petab_problem.model_info.xindices
        
        @info "All estimated parameter IDs: $(estimated_param_ids)"
        @info "Available true values: $(keys(true_values_linear))"
        
        true_params_on_scale_vector = Float64[]
        
        for param_id_sym in estimated_param_ids
            base_name = replace(string(param_id_sym), r"^(log10_|log2_|log_)" => "")
            @info "Processing parameter: $param_id_sym -> base_name: $base_name"
            
            if haskey(true_values_linear, base_name)
                linear_val = true_values_linear[base_name]
                scale = xindices.xscale[param_id_sym]
                
                if scale == :log10
                    scaled_val = log10(linear_val)
                    push!(true_params_on_scale_vector, scaled_val)
                    @info "  Added log10($linear_val) = $scaled_val"
                elseif scale == :log
                    scaled_val = log(linear_val)
                    push!(true_params_on_scale_vector, scaled_val)
                    @info "  Added log($linear_val) = $scaled_val"
                elseif scale == :log2
                    scaled_val = log2(linear_val)
                    push!(true_params_on_scale_vector, scaled_val)
                    @info "  Added log2($linear_val) = $scaled_val"
                else # :lin
                    push!(true_params_on_scale_vector, linear_val)
                    @info "  Added linear value: $linear_val"
                end
            else
                @warn "Could not find true value for estimated parameter: $base_name. Skipping."
            end
        end
        
        if length(true_params_on_scale_vector) == length(estimated_param_ids)
            # Create the parameter names with log10_ prefixes that PEtab expects
            petab_expected_names = Symbol[]
            for param_id_sym in estimated_param_ids
                scale = xindices.xscale[param_id_sym]
                if scale == :log10
                    push!(petab_expected_names, Symbol("log10_", string(param_id_sym)))
                elseif scale == :log
                    push!(petab_expected_names, Symbol("log_", string(param_id_sym)))
                else # :lin - no prefix needed
                    push!(petab_expected_names, param_id_sym)
                end
            end
            
            @info "PEtab expected parameter names: $(petab_expected_names)"
            
            true_params_component_array = ComponentArray(NamedTuple{Tuple(petab_expected_names)}(true_params_on_scale_vector))
            @info "ComponentArray keys: $(collect(keys(true_params_component_array)))"
            
            nllh_true = petab_problem.nllh(true_params_component_array)
            
            println("\n" * "="^50)
            @printf("NLLH for Best Fit Parameters: %.6f\n", best_cost)
            @printf("NLLH for 'True' Parameters:   %.6f\n", nllh_true)
            println("="^50 * "\n")

            if best_cost < nllh_true
                @info "✅ As expected, the best-fit parameters provide a better fit to the noisy data."
                @printf("   The fit is better by %.4f units.\n", nllh_true - best_cost)
                @info "   This confirms the optimizer is working correctly and finding a solution that accounts for the specific noise in this dataset (overfitting)."
            elseif best_cost > nllh_true
                @warn "This is unexpected. The optimizer did not find a solution as good as the true parameters."
                @warn "   This could indicate an optimization issue (e.g., getting stuck in a local minimum)."
            else
                @info "The NLLH values are nearly identical, suggesting the true parameters are very close to the optimal fit."
            end
        else
            @error "Could not construct the full 'true' parameter vector due to missing values. Comparison skipped."
        end
        # --- END: CORRECTED SECTION ---

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

        # --- START: CORRECTED IDENTIFIABILITY SECTION ---
        if parsed_args["ident"]
            @info "--- Running identifiability diagnostics at best fit ---"
            
            param_values = collect(best_params)
            
            # Use the same parameter naming approach as the NLLH comparison
            estimated_param_ids = petab_problem.xnames
            xindices = petab_problem.model_info.xindices
            
            # Create the parameter names with log10_ prefixes that PEtab expects
            petab_expected_names = Symbol[]
            for param_id_sym in estimated_param_ids
                scale = xindices.xscale[param_id_sym]
                if scale == :log10
                    push!(petab_expected_names, Symbol("log10_", string(param_id_sym)))
                elseif scale == :log
                    push!(petab_expected_names, Symbol("log_", string(param_id_sym)))
                elseif scale == :log2
                    push!(petab_expected_names, Symbol("log2_", string(param_id_sym)))
                else # :lin - no prefix needed
                    push!(petab_expected_names, param_id_sym)
                end
            end
            
            @assert length(petab_expected_names) == length(param_values) "Parameter name count ($(length(petab_expected_names))) does not match parameter vector length ($(length(param_values)))"
            
            # Construct the ComponentArray with the correct log10_ prefixed names
            θ_full = ComponentArray(NamedTuple{Tuple(petab_expected_names)}(param_values))
            
            @info "Using parameter names for FIM: $(collect(keys(θ_full)))"
            
            # Sanity check before running the main function
            try
                _ = petab_problem.nllh(θ_full)
                @info "Sanity check for identifiability passed."
            catch e
                @error "Sanity check failed before running identifiability analysis. This indicates a persistent parameter name mismatch."
                rethrow(e)
            end

            run_identifiability(petab_problem, θ_full)
        end
        # --- END: CORRECTED IDENTIFIABILITY SECTION ---

    elseif parsed_args["profile"]
        @info "--- Running in PROFILING mode ---"
        
        fit_data = JLD2.load(parsed_args["load-fit"])
        best_mle = fit_data["best_mle"]
        
        setup_results = setup_petab_problem(parsed_args["yaml"])
        profiling_odesol = ODESolver(Rodas5P(), abstol=1e-8, reltol=1e-8)
        
        prof_method = Symbol(parsed_args["profiling-method"])

        run_likelihood_profiling(
            setup_results.petab_model, 
            profiling_odesol, 
            nothing,
            best_mle, 
            setup_results.true_values
        )
    else
        @error "No mode selected. Please specify --task-id or --collate."
    end
end

# --- 4. SCRIPT EXECUTION ---
BLAS.set_num_threads(1)
main()