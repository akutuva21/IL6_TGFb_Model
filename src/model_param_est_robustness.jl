# src/model_param_est_robustness.jl

using ReactionNetworkImporters, Catalyst
using DifferentialEquations, ModelingToolkit
using PEtab, DataFrames, CSV, YAML
using SymbolicUtils, Symbolics
using Sundials

# Only export the setup function
export setup_petab_problem

function safe_name_initializer(sym_or_var)
    s = string(sym_or_var)
    s_name = first(split(s, "(t)"))
    return Symbol(s_name)
end

function setup_petab_problem(enable_preeq::Bool, model_net_path::String, data_path::String, config_path::String)
    println("Using BNGL model: $model_net_path")
    println("Using data file: $data_path")
    println("Using config file: $config_path")
    println("\n--- Setting up PEtab Problem Programmatically ---")

    # --- 1. Load configuration and model ---
    config = YAML.load_file(config_path)
    observables_mapping = config["observables_mapping"]
    prn = loadrxnetwork(BNGNetwork(), model_net_path)
    rsys = complete(prn.rn)
    println("Loaded BNGL model with $(length(species(rsys))) species and $(length(parameters(rsys))) parameters.")

    # --- 2. Load PEtab tables from files ---
    # This now correctly reads the standard PEtab TSV files
    measurements_df = CSV.read(data_path, DataFrame)
    
    # Construct the path to the conditions file based on the measurements file path
    base_path = dirname(data_path)
    conditions_filename = "conditions_" * replace(basename(data_path), "measurements_" => "")
    conditions_path = joinpath(base_path, conditions_filename)
    if !isfile(conditions_path)
        @error "Conditions file not found at expected path: $conditions_path"
        return nothing
    end
    conditions_df = CSV.read(conditions_path, DataFrame)
    println("Loaded $(nrow(measurements_df)) measurements and $(nrow(conditions_df)) conditions.")

    # --- 3. Build PEtabParameter list ---
    p_map_defaults = Dict{Symbol, Float64}()
    if !isnothing(prn.p)
        for (k, v) in prn.p
            default_val = Symbolics.value(v)
            if default_val isa Number
                p_map_defaults[Symbolics.getname(k)] = Float64(default_val)
            end
        end
    end

    condition_params = Set([Symbol(col) for col in names(conditions_df) if col != "conditionId"])
    
    # Define parameters that should NOT be estimated (fixed external ligands)
    # These are controlled by experimental conditions, not estimated
    fixed_external_ligands = Set([:IL6_0, :TGFb_0])  # External stimuli controlled by conditions
    
    petab_params_list = PEtabParameter[]
    for (param_symbol, default_val) in p_map_defaults
        # Don't estimate external ligands - they are controlled by experimental conditions
        if param_symbol in fixed_external_ligands
            should_estimate = false  # External ligands are not estimated
            println("DEBUG: $param_symbol marked as EXTERNAL LIGAND - not estimated (controlled by conditions)")
        else
            should_estimate = !(param_symbol in condition_params)
        end
        
        # For now, always use automatic bounds calculation
        # This avoids all config file dependency issues
        bounds = Dict(
            "lb" => default_val / 100.0,
            "ub" => default_val * 100.0
        )
        println("DEBUG: Parameter $(param_symbol) = $(default_val), bounds: [$(bounds["lb"]), $(bounds["ub"])]")

        push!(petab_params_list, PEtabParameter(param_symbol; 
                                                value=default_val,
                                                scale=:log10, 
                                                lb=bounds["lb"],
                                                ub=bounds["ub"],
                                                estimate=should_estimate))
    end

    # --- 4. Build PEtabObservable dictionary ---
    observables_petab_dict = Dict{String, PEtabObservable}()
    unique_obs_ids = unique(measurements_df.observableId)

    for obs_id in unique_obs_ids
        bngl_group_name = get(observables_mapping, obs_id, obs_id) # Fallback to direct match
        
        catalyst_model_expr = nothing
        found_catalyst_obs = false
        for obs_eq in observed(rsys)
            if safe_name_initializer(obs_eq.lhs) == Symbol(bngl_group_name)
                catalyst_model_expr = obs_eq.rhs
                found_catalyst_obs = true
                break
            end
        end
        if !found_catalyst_obs
             @warn "Could not find observable mapping for '$bngl_group_name'. It will be treated as a parameter."
             # This allows noise parameters to be defined without being model observables
             catalyst_model_expr = 1.0
        end

        # Define noise parameter, assuming it's named sigma_<observableId>
        sigma_param_sym = Symbol("sigma_" * obs_id)
        if !any(p -> p.parameter == sigma_param_sym, petab_params_list)
            
            # --- THIS IS THE FIX ---
            # For robustness testing, noise is a known, fixed quantity.
            # We set estimate=false and provide a reasonable fixed value.
            # A value of 1.0 is a common default if the data is normalized,
            # or you could set it based on the noise level in your config.
            push!(petab_params_list, PEtabParameter(sigma_param_sym; 
                                                    value=1.0,      # A fixed, representative noise value
                                                    scale=:lin, 
                                                    estimate=false)) # Set to false!
        end

        observables_petab_dict[obs_id] = PEtabObservable(catalyst_model_expr, sigma_param_sym)
    end
    println("Defined $(length(petab_params_list)) PEtabParameters and $(length(observables_petab_dict)) PEtabObservables.")

    # --- 5. Build Simulation Conditions Dictionary ---
    simulation_conditions = Dict{String, Dict{Symbol, Float64}}() 
    for row in eachrow(conditions_df)
        condition_id = row.conditionId
        param_overrides = Dict{Symbol, Float64}()
        for col_name in names(conditions_df)
            if col_name != "conditionId"
                param_overrides[Symbol(col_name)] = row[col_name]
            end
        end
        simulation_conditions[condition_id] = param_overrides
    end

    # --- 6. Create the PEtabModel programmatically ---
    petab_model = PEtabModel(
    rsys,
    observables_petab_dict,
    measurements_df,
    petab_params_list;
    simulation_conditions=simulation_conditions, # Now a keyword argument
    verbose=false
    )

    println("--- PEtab Problem Setup Complete ---")
    
    # Extract true parameter values (log10 scale) for reference plotting
    true_param_values = Dict{String, Float64}()
    println("DEBUG: Extracting true parameter values...")
    for param in petab_params_list
        if param.estimate  # Only include estimated parameters
            # Get the original parameter name without any prefix
            original_param_name = string(param.parameter)
            
            # Create the log10-prefixed name that will be used in the plot
            plot_param_name = if param.scale == :log10
                "log10_" * original_param_name
            else
                original_param_name
            end
            
            if param.scale == :log10
                true_value = log10(param.value)
                true_param_values[plot_param_name] = true_value
                println("DEBUG: $original_param_name -> $plot_param_name = $(param.value) -> log10 = $true_value")
            else
                true_param_values[plot_param_name] = param.value
                println("DEBUG: $original_param_name -> $plot_param_name = $(param.value) (linear scale)")
            end
        end
    end
    println("DEBUG: Total true parameter values extracted: $(length(true_param_values))")
    println("DEBUG: True parameter keys: $(collect(keys(true_param_values)))")
    
    return (petab_model=petab_model, true_values=true_param_values)
end