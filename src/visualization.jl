# src/visualization.jl

# 1. Import Dependencies
using Plots; gr()
using DataFrames
using DifferentialEquations, ModelingToolkit, Catalyst
using ModelingToolkit: isparameter, unknowns
using SymbolicUtils, Symbolics
using PEtab
using Printf
using Statistics

# Helper function
if !isdefined(@__MODULE__, :safe_name_initializer)
    function safe_name_initializer(sym_or_var)
        s = string(sym_or_var)
        s_name = first(split(s, "(t)"))
        return Symbol(s_name)
    end
end

# 2. Main Visualization Function (v3 - FINAL)
function run_visualization(
    theta_optim::Vector{Float64},
    names_est_opt_str::Vector{String},
    petab_params_list::Vector{PEtabParameter},
    meas::DataFrame,
    petab_prob, # This is the PEtabODEProblem
    observables_petab_dict::Dict{String, PEtabObservable},
    sf_param_map_from_est::Dict{Symbol, Num}, # Map from sf_Symbol to sf_Num object
    rsys::ReactionSystem,
    prn::ParsedReactionNetwork,
    il6_species_name_from_est::String
    )

    println("\n--- Starting Visualization ---")

    # 3. Untransform Optimized Parameters to Linear Scale
    println("Processing optimization results (untransforming parameters)...")
    optimized_params_map_by_name = Dict{Symbol, Float64}() # Maps SYMBOL NAME to its linear scale value
    param_def_map = Dict{Symbol, PEtabParameter}()
    for pp_def in petab_params_list
        param_sym_for_map = pp_def.parameter isa Symbol ? pp_def.parameter : Symbolics.getname(pp_def.parameter)
        param_def_map[param_sym_for_map] = pp_def
    end

    for (i, param_name_on_estimation_scale_str) in enumerate(names_est_opt_str)
        val_on_estimation_scale = theta_optim[i]
        local original_param_sym::Symbol

        if startswith(param_name_on_estimation_scale_str, "log10_"); original_param_sym = Symbol(replace(param_name_on_estimation_scale_str, "log10_" => ""));
        elseif startswith(param_name_on_estimation_scale_str, "log_"); original_param_sym = Symbol(replace(param_name_on_estimation_scale_str, "log_" => ""));
        elseif startswith(param_name_on_estimation_scale_str, "log2_"); original_param_sym = Symbol(replace(param_name_on_estimation_scale_str, "log2_" => ""));
        else original_param_sym = Symbol(param_name_on_estimation_scale_str); end

        defined_scale = get(param_def_map, original_param_sym, nothing)
        if isnothing(defined_scale)
            @error "Param $original_param_sym (from $param_name_on_estimation_scale_str) not in PEtabParam definitions. Skipping."
            continue
        end
        
        local val_on_linear_scale::Float64
        if defined_scale.scale == :log10; val_on_linear_scale = 10^val_on_estimation_scale;
        elseif defined_scale.scale == :ln || defined_scale.scale == :log; val_on_linear_scale = exp(val_on_estimation_scale);
        elseif defined_scale.scale == :log2; val_on_linear_scale = 2^val_on_estimation_scale;
        elseif defined_scale.scale == :lin; val_on_linear_scale = val_on_estimation_scale;
        else @warn "Unknown scale '$(defined_scale.scale)' for $original_param_sym."; val_on_linear_scale = val_on_estimation_scale; end
        optimized_params_map_by_name[original_param_sym] = val_on_linear_scale
    end
    println("✅ Processed $(length(optimized_params_map_by_name)) parameters to linear scale.")

    # 4. Prepare Symbolic Maps using symbolic objects (Num) as keys
    println("Preparing symbolic maps...")
    
    # This map holds the raw BNGL values (Num or Float64) with the true symbolic object as the key
    bngl_defaults_map_by_symbolic_obj = Dict{Num, Any}()
    if !isnothing(prn.p) && !isempty(prn.p)
        for (key_from_prn_p, param_val_any) in prn.p
            bngl_defaults_map_by_symbolic_obj[key_from_prn_p] = param_val_any
        end
    end
    println("Processed prn.p. Found $(length(bngl_defaults_map_by_symbolic_obj)) parameter defaults from BNG file.")

    # Create a helper map from a parameter's name (Symbol) to its symbolic object (Num)
    name_to_symbol_map = Dict(Symbolics.getname(k) => k for k in keys(bngl_defaults_map_by_symbolic_obj))
    
    # --- Parameter Mapping (using SYMBOLIC OBJECTS as keys) ---
    final_param_map_for_ODE = Dict{Num, Float64}()

    # 1. Add optimized parameters, converting their names to symbolic objects via the map
    for (param_name, linear_val) in optimized_params_map_by_name
        if haskey(name_to_symbol_map, param_name)
            symbolic_obj = name_to_symbol_map[param_name]
            final_param_map_for_ODE[symbolic_obj] = linear_val
        else
             @warn "Optimized parameter '$param_name' not found in the BNGL model's parameter list. It may be a scaling factor or noise parameter."
        end
    end
    println("Applied $(length(final_param_map_for_ODE)) optimized parameters to final map for ODE.")

    # 2. Iteratively resolve default parameters from the BNGL file
    all_bngl_symbols = keys(bngl_defaults_map_by_symbolic_obj)
    unresolved_symbols = [p_sym for p_sym in all_bngl_symbols if !haskey(final_param_map_for_ODE, p_sym)]
    println("Attempting to resolve $(length(unresolved_symbols)) parameters from BNGL defaults...")
    
    max_passes = length(unresolved_symbols) + 1 
    for pass in 1:max_passes
        resolved_count_this_pass = 0
        still_unresolved = Num[]
        
        for p_symbolic_obj in unresolved_symbols
            default_expr = bngl_defaults_map_by_symbolic_obj[p_symbolic_obj]
            
            resolved_expr = SymbolicUtils.substitute(default_expr, final_param_map_for_ODE)
            numeric_value = Symbolics.value(resolved_expr)

            if numeric_value isa Real
                final_param_map_for_ODE[p_symbolic_obj] = Float64(numeric_value)
                resolved_count_this_pass += 1
            else
                push!(still_unresolved, p_symbolic_obj)
            end
        end

        unresolved_symbols = still_unresolved
        if isempty(unresolved_symbols) || (resolved_count_this_pass == 0 && !isempty(still_unresolved))
            break
        end
    end
    
    if isempty(unresolved_symbols)
        println("All default parameters successfully resolved.")
    else
        @warn "Could not resolve the following parameters: $(Symbolics.getname.(unresolved_symbols))"
    end

    # 5. Create initial conditions map
    base_u0_map_sym = Dict{Num, Float64}()
    println("--- Populating base_u0_map_sym with symbolic resolution ---")
    
    for (species_obj, u0_expr) in prn.u0
        resolved_ic_expr = SymbolicUtils.substitute(u0_expr, final_param_map_for_ODE)
        numeric_ic_value = Symbolics.value(resolved_ic_expr)

        if numeric_ic_value isa Real
            base_u0_map_sym[species_obj] = Float64(numeric_ic_value)
        else
            @error "Initial condition for species $(Symbolics.getname(species_obj)) contains unresolved symbols after substitution: $resolved_ic_expr"
            base_u0_map_sym[species_obj] = 0.0 # Default to 0 to prevent crash
        end
    end
    println("--- Finished populating base_u0_map_sym. Size: $(length(base_u0_map_sym)) ---")
    
    # Remap to the system's states, which might have different object IDs
    u0_final = [base_u0_map_sym[state] for state in ModelingToolkit.unknowns(rsys)]

    # IL6 identification for condition override
    local il6_viz_symbol = nothing
    local il6_state_index = -1
    if il6_species_name_from_est != "NOT_FOUND_IN_ESTIMATION"
        target_il6_base_name_sym = Symbol(first(split(il6_species_name_from_est, "(t)")))
        for (i, s_symb) in enumerate(ModelingToolkit.unknowns(rsys))
            if Symbolics.getname(s_symb) == target_il6_base_name_sym
                il6_viz_symbol = s_symb
                il6_state_index = i
                println("Successfully identified IL6 species for condition override: $(il6_viz_symbol) at index $i")
                break
            end
        end
    end
    if isnothing(il6_viz_symbol) && il6_species_name_from_est != "NOT_FOUND_IN_ESTIMATION"
        @warn "IL6 species '$il6_species_name_from_est' for condition override not found in rsys states."
    end
    
    # 6. Plotting Loop
    cond_ids_for_plot = unique(meas.simulation_id)
    unique_obs_ids_plot = unique(meas.observableId) 

    all_simulations_df_list = []

    for (pid_idx, obs_id_str_plot) in enumerate(unique_obs_ids_plot) 
        base_name_plot = replace(obs_id_str_plot, "obs_" => "") 
        println("\n--- Plotting for observable: $base_name_plot ---")

        plt = plot(title="Biochemical Dynamics: $base_name_plot", xlabel="Time (min)", ylabel="Concentration (a.u.)", legend=:outertopright, framestyle=:box)
        plot!(plt, ymin=0) 

        if !haskey(observables_petab_dict, obs_id_str_plot)
            @error "Observable ID '$obs_id_str_plot' not found in observables_petab_dict. Skipping."
            continue
        end
        symbolic_observable_expr = observables_petab_dict[obs_id_str_plot].obs 

        sf_name_sym = Symbol("sf_", base_name_plot) 
        # Get scaling factor value from optimized_params_map_by_name (which contains linear scale values)
        scaling_factor_numerical_value = get(optimized_params_map_by_name, sf_name_sym, 1.0) 
        println("  Observable: $base_name_plot, Symbolic Expr: $symbolic_observable_expr, Numerical SF val: $scaling_factor_numerical_value")
        
        # Get the symbolic parameter for the scaling factor from the map created during PEtab setup
        symbolic_sf_param = get(sf_param_map_from_est, sf_name_sym, nothing)

        for ext_cond in cond_ids_for_plot
            println("  🔬 Simulating condition: $ext_cond for observable $base_name_plot")
            exp_data_subset = meas[(meas.observableId .== obs_id_str_plot) .& (meas.simulation_id .== ext_cond), :]

            if !isempty(exp_data_subset)
                plot!(plt, exp_data_subset.time, exp_data_subset.measurement,
                      label="Data: $ext_cond", seriestype=:scatter,
                      markersize=5, markerstrokewidth=0.5)

                try
                    # Create the ODEProblem for this condition
                    current_u0 = copy(u0_final)
                    if il6_state_index != -1
                        if ext_cond == "TREG"; current_u0[il6_state_index] = 0.0; end
                        if ext_cond == "TH17"; current_u0[il6_state_index] = 3.0; end
                    end
                    
                    t_max_data = isempty(exp_data_subset.time) ? 45.0 : maximum(exp_data_subset.time)
                    t_max_sim = t_max_data > 0 ? t_max_data : 45.0
                    tspan_sim = (0.0, t_max_sim)
                    t_fine_points = range(0.0, tspan_sim[2], length=100)

                    ode_prob = ODEProblem(rsys, current_u0, tspan_sim, final_param_map_for_ODE)
                    sol = solve(ode_prob, Tsit5(), saveat=t_fine_points, reltol=1e-6, abstol=1e-8)

                    if sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Terminated
                        println("  DEBUG: Raw simulation output for observable '$obs_id_str_plot', condition '$ext_cond':")

                        involved_species_symb = Symbolics.get_variables(symbolic_observable_expr)
                        
                        observed_rsys_species = []
                        for inv_spec_symb in involved_species_symb

                            local matched_rsys_spec = nothing
                            for rsys_s in species(rsys)
                                if Symbolics.getname(rsys_s) == Symbolics.getname(inv_spec_symb)
                                    matched_rsys_spec = rsys_s
                                    break
                                end
                            end

                            if !isnothing(matched_rsys_spec) # Ensure it's a state variable
                                if !isparameter(matched_rsys_spec)          # ← instead of the isa-check
                                    # Check if it's a PEtab scaling factor, if so, skip direct plotting as a "species"
                                    is_sf_param = false
                                    for sf_obj in values(sf_param_map_from_est) # sf_param_map_from_est maps Symbol to Num
                                        if isequal(matched_rsys_spec, sf_obj)
                                            is_sf_param = true
                                            break
                                        end
                                    end
                                    if !is_sf_param
                                        push!(observed_rsys_species, matched_rsys_spec)
                                    end
                                end
                            end
                        end
                        
                        unique!(observed_rsys_species) # Ensure we only print each species once

                        if isempty(observed_rsys_species)
                            println("    Could not identify specific Catalyst species from observable expression: $(symbolic_observable_expr). Check naming or how observables are defined.")
                        else
                            println("    Species involved in '$obs_id_str_plot': $([Symbolics.getname(s) for s in observed_rsys_species])")
                        end

                    # Print time series for these species (first few and last few points)
                    num_points_to_show = min(5, length(sol.t))
                    for s_obj in observed_rsys_species
                        try
                            # Check if s_obj is actually in the solution's state map
                            # sol[s_obj] should work if s_obj is a valid state from rsys
                            sim_values = sol[s_obj] 
                            println("    Raw values for $(Symbolics.getname(s_obj))(t) at first $(num_points_to_show) timepoints: ", round.(sim_values[1:num_points_to_show], digits=4))
                            if length(sol.t) > num_points_to_show
                                println("    Raw values for $(Symbolics.getname(s_obj))(t) at last $(num_points_to_show) timepoints: ", round.(sim_values[end-num_points_to_show+1:end], digits=4))
                            end
                        catch e_species
                            println("    Could not retrieve solution for species $(Symbolics.getname(s_obj)). Is it a valid state in the ODEProblem? Error: $e_species")
                        end
                    end
                else
                    @warn " ODE solution failed for $ext_cond when trying to get raw species values. Retcode: $(sol.retcode)."
                end

                    if sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Terminated
                        evaluated_observable_timeseries = Vector{Float64}(undef, length(sol.u)) 
                        for (j_sol, u_timepoint) in enumerate(sol.u) 
                            
                            substitution_map = Dict{Num,Float64}(s => sol[s][j_sol] for s in species(rsys))
                            
                            if !isnothing(symbolic_sf_param) 
                                substitution_map[symbolic_sf_param] = scaling_factor_numerical_value
                            elseif occursin(string(sf_name_sym), string(symbolic_observable_expr))
                                 @warn "Symbolic scaling factor $(sf_name_sym) (object: $(isnothing(symbolic_sf_param) ? "not found" : symbolic_sf_param)) appears in observable expression string but its symbolic object was not found in sf_param_map_from_est for substitution. The expression might already incorporate it or this could be an issue."
                            end
                            
                            evaluated_value = SymbolicUtils.substitute(symbolic_observable_expr, substitution_map)
                            numeric_value = Symbolics.value(evaluated_value) 

                            if numeric_value isa Real
                                try
                                    evaluated_observable_timeseries[j_sol] = Float64(numeric_value)
                                catch e
                                    @warn "Could not convert evaluated observable to Float64 at t=$(sol.t[j_sol]). Value: $numeric_value, Type: $(typeof(numeric_value)). Error: $e"
                                    evaluated_observable_timeseries[j_sol] = NaN
                                end
                            else
                                # This case implies the expression wasn't fully numeric
                                @warn "Observable expression $symbolic_observable_expr did not evaluate to a Real number at t=$(sol.t[j_sol]). Got: $numeric_value (type: $(typeof(numeric_value))). Free symbols: $(Symbolics.free_symbols(numeric_value))."
                                evaluated_observable_timeseries[j_sol] = NaN
                            end
                            
                        end
                        
                        final_obs_sim = max.(evaluated_observable_timeseries, 0.0) # Just ensure non-negativity

                        plot!(plt, sol.t, final_obs_sim, 
                              label="Model: $ext_cond (sf val=$(@sprintf("%.2e", scaling_factor_numerical_value)))", # Display the sf value for clarity
                              seriestype=:line, linewidth=2.5)
                    else
                        @warn "    ODE solution failed for $ext_cond with retcode $(sol.retcode)."
                    end
                catch e_solve
                    @error "    Error during simulation or plotting for $ext_cond, observable $base_name_plot: $e_solve"
                    showerror(stdout, e_solve, catch_backtrace()); println()
                end
            else
                 println("    No experimental data for $obs_id_str_plot, condition $ext_cond.")
            end
        end
        
        plot_filename = "Plots_Small/REAL_biochemical_$(base_name_plot).png"
        try savefig(plt, plot_filename); println("  Saved plot: $plot_filename")
        catch e_save; @error "Failed to save plot $plot_filename: $e_save"; end

        if !isempty(all_simulations_df_list)
            final_simulations_df = vcat(all_simulations_df_list...)
            csv_filename = "simulation_results.csv"
            try
                CSV.write(csv_filename, final_simulations_df)
                println("\nINFO: All simulated concentrations saved to '$csv_filename'")
            catch e
                @error "Failed to save simulation results to CSV. Error: $e"
            end
        end
    end 
    println("\n--- Visualization Complete ---")
end