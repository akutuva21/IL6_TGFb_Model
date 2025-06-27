# src/model_param_est.jl

using ReactionNetworkImporters, Catalyst
using DifferentialEquations, ModelingToolkit
using PEtab, DataFrames, XLSX
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
    observables_mapping = config["observables_mapping"]

    # --- 1. Load BNGL model and parse experimental data ---
    prn = loadrxnetwork(BNGNetwork(), model_net_path)
    rsys = complete(prn.rn)
    println("Loaded BNGL model with $(length(species(rsys))) species and $(length(parameters(rsys))) parameters.")

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
    
    # Build the list of PEtab parameters to be estimated
    petab_params_list = PEtabParameter[]
    
    # Add kinetic parameters and initial concentration parameters from your BNGL file
    for (param_symbol, default_val) in p_map_defaults
        should_estimate = true # Or your specific logic
        
        # --- THIS IS THE CORRECTED LOGIC ---
        if endswith(string(param_symbol), "_0")
            # Use wide, permissive bounds for initial concentrations that
            # encompass all possible experimental conditions (including zero).
            # A small non-zero value like 1e-9 is good for a log-scale lower bound.
            push!(petab_params_list, PEtabParameter(param_symbol; value=default_val, scale=:log10, lb=1e-9, ub=1e4, estimate=should_estimate))
            println("Setting wide bounds for initial condition parameter: $param_symbol")
        else
            # Bounds for kinetic rates can be tighter around their default values
            lower_bound = max(1e-9, default_val / 10.0)
            upper_bound = default_val * 10.0
            push!(petab_params_list, PEtabParameter(param_symbol; value=default_val, scale=:log10, lb=lower_bound, ub=upper_bound, estimate=should_estimate))
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
        initial_sf_val = 1.0
        if !any(p -> p.parameter == sf_param_sym, petab_params_list)
            push!(petab_params_list, PEtabParameter(sf_param_sym; value=initial_sf_val, scale=:log10, estimate=true, lb=1e-3, ub=1e3))
        end
        if !any(p -> p.parameter == sigma_param_sym, petab_params_list)
            push!(petab_params_list, PEtabParameter(sigma_param_sym; value=0.1, scale=:lin, estimate=true, lb=1e-3, ub=5.0))
        end
        numeric_catalyst_model_expr = catalyst_model_expr isa ModelingToolkit.Num ? catalyst_model_expr : 1.0 * catalyst_model_expr
        symbolic_sf_param = sf_param_map[sf_param_sym] 
        observables_petab_dict[petab_obs_id_for_df] = PEtabObservable(symbolic_sf_param * numeric_catalyst_model_expr, string(sigma_param_sym))
    end

    println("Defined $(length(petab_params_list)) PEtabParameters.")
    
    # --- 3. Define Simulation Conditions (Conditional pre-equilibration) ---
    il6_stimulus_bngl_name = "IL6(r)"
    il6_catalyst_internal_sym = nothing
    for (cat_sym, bngl_name_str) in prn.varstonames
        if string(bngl_name_str) == il6_stimulus_bngl_name
            il6_catalyst_internal_sym = cat_sym
            println("INFO: Identified IL6 species '$il6_stimulus_bngl_name' as Catalyst symbol: $il6_catalyst_internal_sym"); break
        end
    end
    if isnothing(il6_catalyst_internal_sym); @error "Critical: Could not find IL6 species '$il6_stimulus_bngl_name'."; return nothing; end
    il6_condition_key_symbol = Symbolics.getname(il6_catalyst_internal_sym)
    
    tgfb_signal_bngl_name = "TGFb(r)"
    tgfb_catalyst_internal_sym = nothing
    for (cat_sym, bngl_name_str) in prn.varstonames
        if string(bngl_name_str) == tgfb_signal_bngl_name
            tgfb_catalyst_internal_sym = cat_sym
            println("INFO: Identified TGFb signal species '$tgfb_signal_bngl_name' as Catalyst symbol: $tgfb_catalyst_internal_sym"); break
        end
    end
    if isnothing(tgfb_catalyst_internal_sym); @error "Critical: Could not find TGFb signal species '$tgfb_signal_bngl_name'."; return nothing; end
    tgfb_condition_key_symbol = Symbolics.getname(tgfb_catalyst_internal_sym)

    if enable_preeq
        println("--- PEtab Setup: Pre-equilibration ENABLED ---")
        simconds = Dict(
            "TREG" => Dict(il6_condition_key_symbol => 0.0, tgfb_condition_key_symbol => 1.0),
            "TH17" => Dict(il6_condition_key_symbol => 100.0, tgfb_condition_key_symbol => 1.0),
            "preeq_cond" => Dict(il6_condition_key_symbol => 0.0, tgfb_condition_key_symbol => 0.0)
        )
        meas_df[!, :preequilibrationConditionId] .= "preeq_cond"
        println("Set pre-equilibration condition for all $(nrow(meas_df)) measurements.")
    else
        println("--- PEtab Setup: Pre-equilibration DISABLED ---")
        simconds = Dict(
            "TREG" => Dict(il6_condition_key_symbol => 0.0, tgfb_condition_key_symbol => 1.0),
            "TH17" => Dict(il6_condition_key_symbol => 100.0, tgfb_condition_key_symbol => 1.0)
        )
        if :preequilibrationConditionId in names(meas_df)
            select!(meas_df, Not(:preequilibrationConditionId))
        end
    end

    # --- 4. Finalize PEtab Problem Setup ---
    petab_model = PEtabModel(
        rsys,
        observables_petab_dict,
        meas_df,
        petab_params_list;
        simulation_conditions=simconds
    )

    # --- 5. Display Summary of Setup ---
    println("\n--- PEtab Problem Setup Summary ---")
    println("Model file: $model_net_path")
    println("Data file: $data_xlsx_path")
    println("Pre-equilibration: $(enable_preeq ? "Enabled" : "Disabled")")
    println("Number of observables: $(length(observables_petab_dict))")
    println("Number of parameters: $(length(petab_params_list))")
    println("Number of measurement data points: $(nrow(meas_df))")
    println("----------------------------------------")
    
    return petab_model
end