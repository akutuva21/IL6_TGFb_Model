# src/model_param_est.jl (Corrected Version 2)

using Pkg
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

export setup_petab_problem

const MODEL_NET = "model_small/2025_06_23__12_51_41/model_small.net"
const DATA_XLSX = "Data/JBC_Synergy_TC_data.xlsx"
const SHEET2GROUP = Dict(
    "CSK_2D"   => "CSK_Active",
    "P85_3C"   => "P85_P",
    "AKT_3G"   => "AKT_pS473",
    "PKA_6H"   => "PKA_Active"
)

# double check these with the bng model later
const TREG_PATHWAY_PARAMS = Set([
    :kf_tgfb_bind, :kr_tgfb_bind, :k_act_tgfbr, :k_inact_tgfbr,
    :k_phos_smad3, :k_dephos_smad3,
    :kf_s3s4, :kr_s3s4,
    :kf_pka_s3s4, :kr_pka_s3s4, :k_act_pka, :k_inact_pka,
    :k_act_csk, :k_inact_csk,
    :k_phos_p85, :k_dephos_p85,
    :k_phos_pten, :k_dephos_pten
])
const TH17_PATHWAY_PARAMS = Set([
    :kf_il6_bind, :kr_il6_bind, :k_deg_il6,
    :k_act_il6r, :k_inact_il6r,
    :kf_il6r_jak2, :kr_il6r_jak2, :k_act_jak2, :k_inact_jak2,
    :k_phos_stat3, :k_dephos_stat3,
    :kf_stat3_dimer, :kr_stat3_dimer,
    :kf_st3dimer_s3, :kr_st3dimer_s3,
    :k_prod_socs3, :k_deg_socs3, :kf_socs3_bind, :kr_socs3_bind,
    :kf_pi3k_bind, :kr_pi3k_bind, :kf_pip2_pi3k, :kr_pip2_pi3k, :k_cat_pi3k,
    :kf_pip3_akt, :kr_pip3_akt, :k_act_akt, :k_dephos_akt,
    :k_pten
])

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

function setup_petab_problem(; stage::Symbol, params_to_fix::Union{Dict, Nothing}=nothing)
    println("\n--- Setting up PEtab Problem for STAGE: $stage ---")

    # --- Load BNGL model ---
    prn = loadrxnetwork(BNGNetwork(), MODEL_NET)

    string_to_sym_map = Dict(string(v) => k for (k, v) in prn.varstonames)

    rsys = complete(prn.rn)
    
    # parse data from the Excel file
    wb = XLSX.readxlsx(DATA_XLSX)
    meas_rows_list = []
    used_bng_groups = String[]
    for (sheet_key, bng_group_name) in SHEET2GROUP
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
    full_meas_df = DataFrame(meas_rows_list)

    # --- Filter measurement data based on the current stage ---
    local meas_df
    if stage == :TREG
        meas_df = filter(row -> row.simulation_id == "TREG", full_meas_df)
        println("STAGE 1: Using $(nrow(meas_df)) data points for TREG condition.")
    elseif stage == :TH17
        meas_df = filter(row -> row.simulation_id == "TH17", full_meas_df)
        println("STAGE 2: Using $(nrow(meas_df)) data points for TH17 condition.")
    else # stage == :ALL
        meas_df = full_meas_df
        println("STAGE 3: Using all $(nrow(meas_df)) data points.")
    end

    meas_df.time = Float64.(meas_df.time)
    meas_df.measurement = Float64.(meas_df.measurement)
    meas_df.simulation_id = String.(meas_df.simulation_id)

    # --- Build parameter list with stage-specific estimation flags ---
    p_map_defaults = Dict(Symbolics.getname(k) => Symbolics.value(v) for (k, v) in prn.p)
    petab_params_list = PEtabParameter[]
    
    n_estimated = 0
    for (param_symbol, default_val) in p_map_defaults
        final_value = default_val
        should_estimate = false
        if !isnothing(params_to_fix) && haskey(params_to_fix, param_symbol)
            final_value = params_to_fix[param_symbol]
            should_estimate = false
        else
            if stage == :TREG
                should_estimate = param_symbol in TREG_PATHWAY_PARAMS
            elseif stage == :TH17
                should_estimate = param_symbol in TH17_PATHWAY_PARAMS
            elseif stage == :ALL
                should_estimate = true
            end
        end
        if endswith(string(param_symbol), "_0")
            push!(petab_params_list, PEtabParameter(param_symbol; value=final_value, estimate=should_estimate, scale=:log10, lb=10.0, ub=1000.0))
        else
            push!(petab_params_list, PEtabParameter(param_symbol; value=final_value, estimate=should_estimate, scale=:log10, lb=max(1e-9, default_val / 100.0), ub=default_val * 100.0))
        end
        if should_estimate; n_estimated += 1; end
    end
    println("For STAGE $stage, will estimate $n_estimated kinetic/initial value parameters.")
    
    # --- Observable parameter logic ---
    sf_param_map = declare_scaling_parameters(used_bng_groups)
    observables_petab_dict = Dict{String, PEtabObservable}()
    for bng_group_name in used_bng_groups
        petab_obs_id_for_df = "obs_" * replace(bng_group_name, r"[^A-Za-z0-9_]" => "_")
        sf_param_sym = Symbol("sf_", replace(bng_group_name, r"[^A-Za-z0-9_]" => "_"))
        sigma_param_sym = Symbol("sigma_", petab_obs_id_for_df)
        push!(petab_params_list, PEtabParameter(sf_param_sym; value=1.0, scale=:log10, estimate=true, lb=1e-3, ub=1e4))
        push!(petab_params_list, PEtabParameter(sigma_param_sym; value=0.1, scale=:lin, estimate=true, lb=1e-3, ub=5.0))
        catalyst_target_obs_symbol = Symbol(bng_group_name)
        catalyst_model_expr = nothing
        found_catalyst_obs = false
        for obs_eq in observed(rsys)
            if safe_name_initializer(obs_eq.lhs) == catalyst_target_obs_symbol
                catalyst_model_expr = obs_eq.rhs; found_catalyst_obs = true; break
            end
        end
        if !found_catalyst_obs
            @warn "Could not find Catalyst mapping for '$bng_group_name'. Placeholder used."
            catalyst_model_expr = species(rsys)[1]
        end
        numeric_catalyst_model_expr = catalyst_model_expr isa ModelingToolkit.Num ? catalyst_model_expr : 1.0 * catalyst_model_expr
        symbolic_sf_param = sf_param_map[sf_param_sym]
        observables_petab_dict[petab_obs_id_for_df] = PEtabObservable(symbolic_sf_param * numeric_catalyst_model_expr, string(sigma_param_sym))
    end
    unique!(p -> p.parameter, petab_params_list)

    # --- Conditions setup ---
    il6_stimulus_bngl_name = "IL6(r)"
    il6_catalyst_internal_sym = string_to_sym_map[il6_stimulus_bngl_name]
    il6_condition_key_symbol = Symbolics.getname(il6_catalyst_internal_sym)

    tgfb_signal_bngl_name = "TGFb(r)"
    tgfb_catalyst_internal_sym = string_to_sym_map[tgfb_signal_bngl_name]
    tgfb_condition_key_symbol = Symbolics.getname(tgfb_catalyst_internal_sym)

    simconds = Dict(
        "TREG" => Dict(il6_condition_key_symbol => 0.0, tgfb_condition_key_symbol => 1.0),
        "TH17" => Dict(il6_condition_key_symbol => 3.0, tgfb_condition_key_symbol => 1.0),
        "preeq_cond" => Dict(il6_condition_key_symbol => 0.0, tgfb_condition_key_symbol => 0.0)
    )
    meas_df[!, :preequilibrationConditionId] .= "preeq_cond"

    # --- Create and return final PEtabModel ---
    odesys = structural_simplify(convert(ODESystem, rsys); simplify=true)
    petab_model = PEtabModel(odesys, observables_petab_dict, meas_df, petab_params_list;
                             simulation_conditions=simconds, verbose=false)

    # Return all objects needed by the main script
    return (
        petab_model=petab_model,
        prn=prn,
        meas=full_meas_df, # Pass the full measurement data for plotting
        observables_petab_dict=observables_petab_dict,
    )
end
