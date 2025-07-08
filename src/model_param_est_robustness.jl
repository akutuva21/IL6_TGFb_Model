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
    
    petab_params_list = PEtabParameter[]
    bounds_settings = config["parameter_bounds"] # Assumes bounds are in config
    for (param_symbol, default_val) in p_map_defaults
        should_estimate = !(param_symbol in condition_params)
        
        local bounds
        if haskey(bounds_settings, "overrides") && haskey(bounds_settings["overrides"], string(param_symbol))
            bounds = bounds_settings["overrides"][string(param_symbol)]
        elseif endswith(string(param_symbol), "_0")
            bounds = bounds_settings["default_initial_conc"]
        else
            bounds = bounds_settings["default_kinetic"]
        end

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
    return petab_model
end