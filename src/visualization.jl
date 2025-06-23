# src/visualization.jl

Pkg.add("CSV")

# 1. Import Dependencies
using Plots; gr()
using DataFrames
using DifferentialEquations # For ReturnCode
using PEtab
using Printf
using CSV

# Helper function for consistent naming
function safe_name_initializer(sym_or_var)
    s = string(sym_or_var)
    s_name = first(split(s, "(t)"))
    return Symbol(s_name)
end

# 2. Main Visualization Function
function run_visualization(
    theta_optim::Vector{Float64},
    petab_prob # This is the PEtabODEProblem
    )

    println("\n--- Starting Visualization (with PEtab pre-equilibration) ---")

    # Extract necessary components directly from the PEtabODEProblem
    meas = petab_prob.petab_model.measurement_df
    observables_petab_dict = petab_prob.petab_model.observables
    
    cond_ids_for_plot = unique(meas.simulation_id)
    unique_obs_ids_plot = unique(meas.observableId) 

    all_simulations_df_list = []

    # Loop over each observable to create a separate plot
    for obs_id_str_plot in unique_obs_ids_plot
        base_name_plot = replace(obs_id_str_plot, "obs_" => "") 
        println("\n--- Plotting for observable: $base_name_plot ---")

        plt = plot(title="Biochemical Dynamics: $base_name_plot", xlabel="Time (min)", ylabel="Concentration (a.u.)", legend=:outertopright, framestyle=:box)
        plot!(plt, ymin=0) 

        # PEtab's observable dictionary uses Symbols for keys
        obs_id_sym = Symbol(obs_id_str_plot)
        if !haskey(observables_petab_dict, obs_id_sym)
            @error "Observable ID '$obs_id_str_plot' not found in observables_petab_dict. Skipping."
            continue
        end
        
        # Loop over each experimental condition (e.g., TREG, TH17)
        for ext_cond in cond_ids_for_plot
            println(" 🔬 Simulating condition: $ext_cond for observable $base_name_plot")
            
            # Filter and plot the experimental data for the current condition
            df_exp_cond = filter(row -> row.simulation_id == ext_cond && row.observableId == obs_id_str_plot, meas)
            if !isempty(df_exp_cond)
                plot!(plt, df_exp_cond.Time, df_exp_cond.measurement, 
                      seriestype=:scatter, 
                      label="Data: $ext_cond",
                      markersize=4, markerstrokewidth=0.5)
            end

            try
                # This correctly handles pre-equilibration and condition-specific parameters.
                t_max_sim = isempty(df_exp_cond) ? 120.0 : ceil(maximum(df_exp_cond.Time))
                
                sol = petab_prob.compute_solution(
                    theta_optim;
                    condition_id=ext_cond,
                    tmax=t_max_sim
                )

                if sol.retcode == ReturnCode.Success
                    # Use PEtab's function to calculate observables from the solution object
                    observables = petab_prob.compute_observables(sol)
                    final_obs_sim = observables[obs_id_sym] 

                    # Plot the simulated trajectory
                    plot!(plt, sol.t, final_obs_sim,
                          label="Model: $ext_cond",
                          seriestype=:line, linewidth=2.5)
                    
                    # Store simulation results for later saving
                    df_sim = DataFrame(time=sol.t, measurement=final_obs_sim, 
                                       simulation_id=ext_cond, observableId=obs_id_str_plot)
                    push!(all_simulations_df_list, df_sim)
                else
                    @warn "   ODE solution failed for condition '$ext_cond' with retcode $(sol.retcode)."
                end

            catch e_solve
                @error "   Error during simulation for visualization: $e_solve" exception=(e_solve, catch_backtrace())
            end
        end

        # Save the completed plot for the current observable
        plot_path = joinpath(pwd(), "final_results_plots")
        if !isdir(plot_path); mkpath(plot_path); end
        plot_filename = joinpath(plot_path, "$base_name_plot.png")
        savefig(plt, plot_filename)
        println("✅ Plot saved to: $plot_filename")
    end

    # Save all simulated data to a single CSV file
    if !isempty(all_simulations_df_list)
        full_sim_df = vcat(all_simulations_df_list...)
        csv_path = joinpath(pwd(), "final_results_csv")
        if !isdir(csv_path); mkpath(csv_path); end
        csv_filename = joinpath(csv_path, "simulated_trajectories.csv")
        CSV.write(csv_filename, full_sim_df)
        println("✅ All simulation data saved to: $csv_filename")
    end

    println("\n--- Visualization Complete ---")
end