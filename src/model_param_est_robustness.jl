# src/model_param_est.jl

using ReactionNetworkImporters, Catalyst
using DifferentialEquations, ModelingToolkit
using PEtab, DataFrames, XLSX, CSV
using Optimization, Optim, OptimizationOptimJL
using DiffEqCallbacks
using SymbolicUtils, Symbolics
using ComponentArrays
using SciMLBase, SciMLSensitivity
using Logging
using Sundials
using YAML

# Only export the setup function, as the workflows are now in main.jl
export setup_petab_problem

function safe_name_initializer(sym_or_var)
    s = string(sym_or_var)
    s_name = first(split(s, "(t)"))
    return Symbol(s_name)
end

function declare_scaling_parameters(bngl_group_names::Vector{String})
    unique_sf_symbols = unique(Symbol("sf_", replace(name, r"[^A-Za-z0-9_]"=>"_")) for name in bngl_group_names)
    if !isempty(unique_sf_symbols)
        for s_sym in unique_sf_symbols
            if !isdefined(@__MODULE__, s_sym)
                Core.eval(@__MODULE__, :(ModelingToolkit.@parameters $(s_sym)))
            end
        end
    end
    return Dict(s => Base.eval(@__MODULE__, s) for s in unique_sf_symbols if isdefined(@__MODULE__, s))
end

function setup_petab_problem(enable_preeq::Bool, model_net_path::String, data_xlsx_path::String, config_path::String)
    println("Using BNGL model: $model_net_path")
    println("Using config file: $config_path")
    println("\n--- Setting up PEtab Problem (Robust Method) ---")

    # --- 0. Load configuration from YAML file ---
    config = YAML.load_file(config_path)
    
    # Choose the appropriate observables mapping based on data format
    if endswith(lowercase(data_xlsx_path), ".tsv") || endswith(lowercase(data_xlsx_path), ".csv")
        # For TSV/CSV dose-response data
        observables_mapping = get(config, "dose_response_observables_mapping", config["observables_mapping"])
        println("INFO: Using dose-response observables mapping for TSV/CSV data")
    else
        # For Excel time-course data
        observables_mapping = config["observables_mapping"]
        println("INFO: Using time-course observables mapping for Excel data")
    end

    # --- 1. Load BNGL model and parse experimental data ---
    prn = loadrxnetwork(BNGNetwork(), model_net_path)
    rsys = complete(prn.rn)
    println("Loaded BNGL model with $(length(species(rsys))) species and $(length(parameters(rsys))) parameters.")

    # Determine file format and load data accordingly
    if endswith(lowercase(data_xlsx_path), ".tsv") || endswith(lowercase(data_xlsx_path), ".csv")
        # Load TSV/CSV file (PEtab format)
        meas_df = CSV.read(data_xlsx_path, DataFrame, delim='\t')
        
        # Rename columns to match expected format
        if hasproperty(meas_df, :simulationConditionId)
            rename!(meas_df, :simulationConditionId => :simulation_id)
        end
        
        # Ensure required columns exist
        if !hasproperty(meas_df, :observableId) || !hasproperty(meas_df, :time) || !hasproperty(meas_df, :measurement)
            @error "TSV/CSV file missing required columns: observableId, time, measurement"
            return nothing
        end
        
        # Extract unique observable IDs from the data and map them to BNGL observables
        raw_observable_ids = String.(unique(meas_df.observableId))
        used_bng_groups = String[]
        
        # Map TSV observable IDs to BNGL observable names using the config
        for obs_id in raw_observable_ids
            if haskey(observables_mapping, obs_id)
                mapped_obs = observables_mapping[obs_id]
                if !(mapped_obs in used_bng_groups)
                    push!(used_bng_groups, mapped_obs)
                end
                println("INFO: Mapped TSV observable '$obs_id' to BNGL observable '$mapped_obs'")
            else
                @warn "No mapping found for TSV observable '$obs_id' in config file"
                # Still add it to try direct matching
                if !(obs_id in used_bng_groups)
                    push!(used_bng_groups, obs_id)
                end
            end
        end
        
        println("Loaded $(nrow(meas_df)) measurement data points.")
        
    else
        # Load Excel file (legacy format)
        wb = XLSX.readxlsx(data_xlsx_path)
        meas_rows_list = []
        used_bng_groups = String[] 
        for (sheet_key, bng_group_name) in observables_mapping
            if !(bng_group_name in used_bng_groups); push!(used_bng_groups, bng_group_name); end
            petab_obs_id = "obs_" * replace(bng_group_name, r"[^A-Za-z0-9_]" => "_")
            df_sheet = DataFrame(XLSX.gettable(wb[sheet_key]))
            if !hasproperty(df_sheet, :Time); @error "Sheet '$sheet_key' missing 'Time'. Skipping."; continue; end
            for col_name_excel in names(df_sheet)
                col_name_str = strip(String(col_name_excel))
                if lowercase(col_name_str) == "time"; continue; end
                condition_id_str = ""
                if occursin(r"(?i)^treg", col_name_str); condition_id_str = "TREG";
                elseif occursin(r"(?i)^th17", col_name_str); condition_id_str = "TH17";
                else continue; end
                for r_idx in 1:nrow(df_sheet)
                    measurement_val = df_sheet[r_idx, col_name_excel]
                    if !ismissing(measurement_val) 
                        push!(meas_rows_list, (simulation_id=condition_id_str, observableId=petab_obs_id,
                                            time=df_sheet[r_idx, :Time], measurement=measurement_val))
                    end
                end
            end
        end
        if isempty(meas_rows_list); @error "No measurement data parsed. Aborting."; return nothing; end
        meas_df = DataFrame(meas_rows_list)
        meas_df.time = Float64.(meas_df.time)
        meas_df.measurement = Float64.(meas_df.measurement)
        meas_df.simulation_id = String.(meas_df.simulation_id) 
        println("Loaded $(nrow(meas_df)) measurement data points.")
    end
    
    # --- 2. Build Correct Parameter and Initial Condition Maps ---
    
    # Build a complete map of all default parameters from the BNGL file
    p_map_defaults = Dict{Symbol, Float64}()
    if !isnothing(prn.p)
        for (k, v) in prn.p
            # Use Symbolics.value to extract the underlying value
            default_val = Symbolics.value(v)

            # Check if the extracted value is a number before assigning
            if default_val isa Number
                p_map_defaults[Symbolics.getname(k)] = Float64(default_val)
            else
                # If the default value is another parameter or expression, it cannot be
                # converted to a Float64. We skip it here.
                @warn "Parameter '$(Symbolics.getname(k))' has a symbolic default value ('$default_val') and will be skipped in the initial parameter map."
            end
        end
    end
    
    # Define parameters that specify experimental conditions and should not be estimated.
    # For dose-response mode, we vary IL6_0 as the condition but estimate TGFb_0 as background
          
    if endswith(lowercase(data_xlsx_path), ".tsv") || endswith(lowercase(data_xlsx_path), ".csv")
        # In dose-response mode, both IL6_0 (the dose) and TGFb_0 (the required co-stimulus)
        # are condition parameters and should NOT be estimated.
        condition_params = Set([:IL6_0, :TGFb_0])
        println("INFO: Dose-response mode - IL6_0 is the dose parameter, TGFb_0 is a fixed condition parameter")
    else
        # In time-course mode (Excel), both are condition parameters
        condition_params = Set([:IL6_0, :TGFb_0])
        println("INFO: Time-course mode - both IL6_0 and TGFb_0 are condition parameters")
    end

    # Build the list of PEtab parameters to be estimated
    petab_params_list = PEtabParameter[]
    
    # Add kinetic parameters and initial concentration parameters from your BNGL file
    for (param_symbol, default_val) in p_map_defaults
        # Determine if the parameter should be estimated.
        should_estimate = !(param_symbol in condition_params)
        
        # --- THIS IS THE FIX ---
        # We are setting uniform, biochemically plausible bounds for all parameters to ensure
        # that the initial guesses for the optimizer are in a numerically stable region.
        # This is the standard approach when default parameters are unknown or unstable.

        if endswith(string(param_symbol), "_0")
            # For initial concentrations, use the actual value from BNGL file
            # Handle special case where initial concentration is 0 (like TGFb_0, IL6_0)
            actual_value = default_val > 0 ? default_val : 0.01  # Use small positive value for zero initial conditions
            push!(petab_params_list, PEtabParameter(param_symbol; 
                                                    value=actual_value,
                                                    scale=:log10, 
                                                    lb=1e-3,     # Lower bound for small initial amounts
                                                    ub=1000.0,   # Upper bound of 1000 molecules
                                                    estimate=should_estimate))
            if !should_estimate
                println("INFO: Treating '$param_symbol' as a fixed condition parameter (not estimated).")
            end
        else
            # For kinetic rates, use the actual value from BNGL file
            push!(petab_params_list, PEtabParameter(param_symbol; 
                                                    value=default_val,  # Use actual value from BNGL
                                                    scale=:log10, 
                                                    lb=1e-6,     # Lower bound for very slow rates
                                                    ub=10.0,     # Upper bound for fast rates (increased from 1.0)
                                                    estimate=true))
        end
    end

    # Add observable-related parameters (scaling factors, sigmas)
    sf_param_map = declare_scaling_parameters(used_bng_groups) 
    observables_petab_dict = Dict{String, PEtabObservable}()

    for bng_group_name in used_bng_groups 
        petab_obs_id_for_df = "obs_" * replace(bng_group_name, r"[^A-Za-z0-9_]" => "_")
        sf_param_sym = Symbol("sf_", replace(bng_group_name, r"[^A-Za-z0-9_]" => "_")) 
        sigma_param_sym = Symbol("sigma_", petab_obs_id_for_df)
        catalyst_target_obs_symbol = Symbol(bng_group_name)
        catalyst_model_expr = nothing
        found_catalyst_obs = false
        for obs_eq in observed(rsys) 
            if safe_name_initializer(obs_eq.lhs) == catalyst_target_obs_symbol
                catalyst_model_expr = obs_eq.rhs; found_catalyst_obs = true
                println("INFO: Matched BNGL group '$bng_group_name' to Catalyst observed: $(obs_eq.lhs)"); break
            end
        end
        if !found_catalyst_obs 
            for spec_in_rsys in species(rsys)
                if safe_name_initializer(spec_in_rsys) == catalyst_target_obs_symbol
                    catalyst_model_expr = spec_in_rsys; found_catalyst_obs = true
                    println("INFO: Matched BNGL group '$bng_group_name' to Catalyst species: $spec_in_rsys"); break
                end
            end
        end
        if !found_catalyst_obs
            @warn "Could not find Catalyst mapping for '$bng_group_name'. Placeholder used."
            catalyst_model_expr = species(rsys)[1] 
        end
        
        # For robustness testing, scaling factors and noise are known and fixed.
        if !any(p -> p.parameter == sf_param_sym, petab_params_list)
            push!(petab_params_list, PEtabParameter(sf_param_sym; value=1.0, scale=:lin, estimate=false))
        end
        if !any(p -> p.parameter == sigma_param_sym, petab_params_list)
            # Set sigma to a reasonable value for biological data (5% of typical signal range)
            # For data ranging 0-100, sigma=5.0 represents ~5% measurement noise
            push!(petab_params_list, PEtabParameter(sigma_param_sym; value=5.0, scale=:lin, estimate=false))
        end

        numeric_catalyst_model_expr = catalyst_model_expr isa ModelingToolkit.Num ? catalyst_model_expr : 1.0 * catalyst_model_expr
        symbolic_sf_param = sf_param_map[sf_param_sym] 
        observables_petab_dict[petab_obs_id_for_df] = PEtabObservable(symbolic_sf_param * numeric_catalyst_model_expr, string(sigma_param_sym))
    end

    println("Defined $(length(petab_params_list)) PEtabParameters.")
    
    # --- 3. Define Simulation Conditions (Conditional pre-equilibration) ---
    il6_condition_key_symbol = :IL6_0
    tgfb_condition_key_symbol = :TGFb_0
    
    println("INFO: Using parameter-based condition overrides: IL6_0 and TGFb_0")

    unique_conditions = unique(meas_df.simulation_id)
    simconds = Dict{String, Dict{Symbol, Float64}}()
    
    # --- THIS IS THE FIX ---
    # We now correctly set the fixed value for TGFb_0 for all dose-response conditions.
    if any(startswith.(unique_conditions, "dose_"))
        # Get the constant TGFb_0 value from the config YAML
        tgfb_value = config["dose_response_settings"]["constant_parameters"]["TGFb_0"]

        for condition in unique_conditions
            if startswith(condition, "dose_")
                dose_str = replace(condition, "dose_" => "")
                il6_dose = parse(Float64, dose_str)
                # Set both IL6_0 and the constant TGFb_0 for each condition
                simconds[condition] = Dict(il6_condition_key_symbol => il6_dose,
                                           tgfb_condition_key_symbol => tgfb_value)
                println("INFO: Created condition '$condition' with IL6=$il6_dose and fixed TGFb=$tgfb_value")
            end
        end
    else
        # Time-course format - read conditions from config file
        config = YAML.load_file(config_path)
        tc_conditions = config["time_course_settings"]["conditions"]
        
        for (condition_name, condition_values) in tc_conditions
            il6_val = get(condition_values, "IL6_0", 0.0)
            tgfb_val = get(condition_values, "TGFb_0", 1.0)
            simconds[condition_name] = Dict(il6_condition_key_symbol => il6_val, 
                                          tgfb_condition_key_symbol => tgfb_val)
            println("INFO: Created condition '$condition_name' with IL6=$il6_val, TGFb=$tgfb_val from config file")
        end
    end

    if enable_preeq
        println("--- PEtab Setup: Pre-equilibration ENABLED ---")
        if any(startswith.(unique_conditions, "dose_"))
            # For pre-equilibration, IL6 is 0 and TGFb is at its constant background level.
            tgfb_value = config["dose_response_settings"]["constant_parameters"]["TGFb_0"]
            simconds["preeq_ss"] = Dict(il6_condition_key_symbol => 0.0,
                                       tgfb_condition_key_symbol => tgfb_value)
            println("INFO: Dose-response pre-equilibration set to IL6=0, TGFb=$tgfb_value")
        else
            # Time-course format - read baseline from config file
            config = YAML.load_file(config_path)
            tc_settings = config["time_course_settings"]
            variable_stimuli = Set(tc_settings["variable_stimuli"])
            constant_stimuli = tc_settings["constant_stimuli"]
            
            # Get baseline values from TREG condition (or first condition)
            baseline_condition = tc_settings["conditions"]["TREG"]
            
            # Create pre-equilibration condition: variable stimuli = 0, constant stimuli = baseline values
            preeq_condition = Dict{Symbol, Float64}()
            preeq_condition[il6_condition_key_symbol] = 0.0  # Variable stimulus set to 0
            
            for const_param in constant_stimuli
                if haskey(baseline_condition, const_param)
                    if const_param == "TGFb_0"
                        preeq_condition[tgfb_condition_key_symbol] = baseline_condition[const_param]
                    end
                end
            end
            
            simconds["preeq_ss"] = preeq_condition
            println("INFO: Time-course pre-equilibration configured from config file")
        end
        
        if hasproperty(meas_df, :preequilibrationConditionId)
            # Keep existing preequilibrationConditionId values
        else
            meas_df[!, :preequilibrationConditionId] .= "preeq_ss"
        end
        
        # Verification: Check that pre-equilibration conditions match experimental design
        if !any(startswith.(unique_conditions, "dose_"))
            treg_tgfb = simconds["TREG"][tgfb_condition_key_symbol]
            preeq_tgfb = simconds["preeq_ss"][tgfb_condition_key_symbol]
            if treg_tgfb != preeq_tgfb
                @warn "Inconsistency detected: TREG TGFb ($treg_tgfb) ≠ Pre-eq TGFb ($preeq_tgfb)"
            else
                println("✅ Verified: Pre-equilibration baseline matches experimental design")
            end
        end
        
        println("Set pre-equilibration condition for all $(nrow(meas_df)) measurements.")
    else
        println("--- PEtab Setup: Pre-equilibration DISABLED ---")
        if hasproperty(meas_df, :preequilibrationConditionId)
            select!(meas_df, Not(:preequilibrationConditionId))
        end
    end

    # --- 4. Create PEtabModel ---
    petab_model = PEtabModel(rsys, observables_petab_dict, meas_df, petab_params_list; 
                            simulation_conditions=simconds, verbose=false)

    println("--- PEtab Problem Setup Complete ---")
    return petab_model
end