using Pkg
Pkg.activate("bngl_julia/")

# Load all necessary packages
using DifferentialEquations, PEtab, Sundials, ComponentArrays, Printf
using DataFrames, CSV, Plots, SymbolicUtils, Symbolics
using ModelingToolkit: species, parameters, observed, unknowns, get_iv # Ensure species is here

# Include your project's setup functions
include("src/model_param_est.jl")

println("--- Final Results Processing: Exporting All Observables and Generating Plots ---")

# --- CONFIGURATION ---
const NUM_SIMULATION_POINTS = 200 # Increase for smoother curves

# --- 1. Set up the PEtab Model and other required objects ---
println("Setting up PEtab model and objects...")
setup_results = setup_petab_problem()
if isnothing(setup_results)
    @error "Failed to build PEtabModel. Cannot proceed."
    exit()
end
rsys = setup_results.rsys
prn = setup_results.prn
petab_problem = PEtabODEProblem(setup_results.petab_model, verbose=false)
petab_params_list = setup_results.petab_params_list

# --- 2. Define the Best-Fit Parameter Set ---
p_best_log_scale = [ -0.908656942, -0.498172882, -0.549168641, 0.916488746, -0.165407503, -1.538898833, -1.045878257, -1.28768169, -0.784087207, -1.514099478, -3.287843263, -1.514777518, -1.982146141, -1.199526012, -3.531398578, -0.394343395, -1.328942895, -0.285914876, -2.222941125, -1.51940869, 0.16638606, -0.239329628, 0.153219079, -0.157003261, -1.219552638, -1.447062883, -2.045574622, 0.518155018, -1.797766311, 0.7466577, -1.380497539, -0.075935401, -1.546681449, 0.635328518, 0.911325616, -0.61015559, 0.236037482, -0.856543728, -1.274746143, 1.402059166, -0.406045311, -0.644497925, -2.648930414, -1.163724591, 0.664432285, -0.456700104, -1.388620924, 0.054727726, -1.717058812, -0.917722031, 0.35312969, 0.207839448, -0.490527369, 0.341680173, -0.601350517, 0.108062533, -2.205999938, 0.886678766]

# --- 3. Prepare Correct Parameter and Initial Condition Maps ---
println("Preparing consistent parameter and initial condition maps...")
param_def_map = Dict(Symbolics.getname(p.parameter) => p for p in petab_params_list)
optimized_params_map_linear = Dict{Symbol, Float64}()
for (p_name, val) in zip(petab_problem.xnames, p_best_log_scale)
    scale = param_def_map[p_name].scale
    optimized_params_map_linear[p_name] = PEtab.transform_x(val, scale, to_xscale=false)
end

full_parameter_map = Dict{Symbol, Float64}()
if !isnothing(prn.p); for (k, v) in prn.p; full_parameter_map[Symbolics.getname(Symbolics.unwrap(k))] = Float64(v); end; end
merge!(full_parameter_map, optimized_params_map_linear)

ode_parameter_map = Dict()
for p in parameters(rsys); ode_parameter_map[p] = full_parameter_map[Symbolics.getname(p)]; end

base_u0_map = Dict()
if !isnothing(prn.u0); for (k, v) in prn.u0; base_u0_map[Symbolics.unwrap(k)] = Float64(v); end; end
il6_symbol = getproperty(rsys, Symbol(first(split(setup_results.il6_species_name, "(t)"))))
println("✅ Maps prepared.")

# --- 4. Discover All Observables from the Model ---
all_rsys_observables = observed(rsys)
# Store base names (strings) for iteration and CSV headers
all_observable_base_names = [string(Symbolics.getname(obs_eq.lhs)) for obs_eq in all_rsys_observables]
# Keep the PEtab observables map for expressions that include scaling factors
petab_observables_map = setup_results.observables_petab_dict
println("INFO: Discovered $(length(all_rsys_observables)) total observables from model. $(length(petab_observables_map)) have PEtab definitions.")

# --- 5. Simulate and Export ---
csv_dir = "final_results_csv"
if !isdir(csv_dir) mkdir(csv_dir) end
plots_dir = "final_results_plots"
if !isdir(plots_dir) mkdir(plots_dir) end

# Build a map from parameter name (Symbol) to symbolic object (Num) for substitution
param_symbol_map = Dict(Symbolics.getname(p) => p for p in parameters(rsys))

for condition in ["TREG", "TH17"]
    println("Processing condition: $condition")
    
    condition_u0_map = copy(base_u0_map)
    if !isnothing(il6_symbol); condition_u0_map[il6_symbol] = (condition == "TH17" ? 3.0 : 0.0); end
    
    t_max = isempty(setup_results.meas) ? 45.0 : maximum(filter(row -> row.simulation_id == condition, setup_results.meas).time)
    t_save = range(0.0, t_max, length=NUM_SIMULATION_POINTS)

    ode_prob = ODEProblem(rsys, condition_u0_map, (0.0, t_max), ode_parameter_map)
    sol = solve(ode_prob, Tsit5(), saveat=t_save, reltol=1e-6, abstol=1e-8)
    
    df = DataFrame(time = sol.t)
    
    # Build substitution map for parameters and scaling factors (once per condition)
    param_sub_map = Dict{Any, Float64}()
    for p_sym_model in parameters(rsys) # Iterate over Num objects from rsys.parameters
        p_name_model = Symbolics.getname(p_sym_model) # Get Symbol name
        if haskey(full_parameter_map, p_name_model)
            param_sub_map[p_sym_model] = full_parameter_map[p_name_model]
        end
    end
    for (sf_name_symbol, sf_param_as_num) in setup_results.sf_param_map # sf_name_symbol is Symbol, sf_param_as_num is Num
        if haskey(full_parameter_map, sf_name_symbol)
            param_sub_map[sf_param_as_num] = full_parameter_map[sf_name_symbol]
        end
    end

    for obs_eq in all_rsys_observables
        obs_base_name = string(Symbolics.getname(obs_eq.lhs)) # e.g., "CSK_Active"
        petab_style_id = "obs_$(obs_base_name)" # e.g., "obs_CSK_Active"

        local obs_expr_to_evaluate
        if haskey(petab_observables_map, petab_style_id)
            # Use the PEtab-defined observable expression, which includes the symbolic scaling factor
            obs_expr_to_evaluate = petab_observables_map[petab_style_id].obs
        else
            # Use the raw model observable expression
            obs_expr_to_evaluate = obs_eq.rhs
        end
        
        obs_values = [Symbolics.value(SymbolicUtils.substitute(obs_expr_to_evaluate, merge(
                Dict(s_model => val_s for (s_model, val_s) in zip(species(rsys), u_timepoint)), # species(rsys) are Nums
                param_sub_map
            ))) for u_timepoint in sol.u]
        df[!, Symbol(obs_base_name)] = obs_values # Use base name for CSV column
    end
    
    CSV.write(joinpath(csv_dir, "$(condition)_observables.csv"), df)
    println("  ✅ Wrote $(condition)_observables.csv")
end

# --- 6. Generate Plots ---
println("\nGenerating plots from exported CSV data...")
df_measurements = DataFrame(setup_results.meas)

# Iterate using the base names collected earlier
for obs_base_name_str in all_observable_base_names 
    p = plot(title=obs_base_name_str, xlabel="Time", ylabel="Value", legend=:outertopright, framestyle=:box)

    for condition in ["TREG", "TH17"]
        df_sim = CSV.read(joinpath(csv_dir, "$(condition)_observables.csv"), DataFrame)
        # Plot using the base name (which is now the column name in CSV)
        if obs_base_name_str in names(df_sim) # Compare string with vector of strings
             plot!(p, df_sim.time, df_sim[!, Symbol(obs_base_name_str)], label="Model ($condition)", linewidth=2)
        else
            # MODIFIED WARNING:
            @warn "Observable '" * obs_base_name_str * "' (Symbol: " * string(Symbol(obs_base_name_str)) * ") not found in simulated CSV for condition '" * condition * "'. Available columns: " * string(names(df_sim))
        end
    end
    
    # For experimental data, construct the "obs_" prefixed ID
    obs_id_for_exp_data = "obs_$(obs_base_name_str)"
    data_for_obs = filter(row -> row.observableId == obs_id_for_exp_data, df_measurements)
    if !isempty(data_for_obs)
        for (cond_key, group) in pairs(groupby(data_for_obs, :simulation_id))
            scatter!(p, group.time, group.measurement, label="Data ($(cond_key.simulation_id))", markershape=:xcross)
        end
    end
    
    savefig(p, joinpath(plots_dir, "$(obs_base_name_str).png"))
    println("  ✅ Saved $(obs_base_name_str).png")
end

println("\n--- Processing Complete ---")