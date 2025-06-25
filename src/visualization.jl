# src/visualization.jl

# 1. Import Dependencies
using Plots; gr()
using DataFrames
using DifferentialEquations
using PEtab
using Printf
using CSV

# 2. Main Visualization Function
function run_visualization(
    theta_optim::Vector{Float64},
    petab_prob::PEtabODEProblem
)
    println("\n--- Starting Visualization ---")

    parameter_names_on_scale = petab_prob.model_info.xindices.xids[:estimate_ps]
    p_est = ComponentArray(; (parameter_names_on_scale .=> theta_optim)...)
    
    println("Calculating simulated values for all conditions...")
    simulated_vals = petab_prob.simulated_values(p_est)

    # Get the measurement data table, which now has the standardized column names
    results_df = deepcopy(petab_prob.model_info.model.petab_tables[:measurements])
    results_df[!, :simulated] = simulated_vals

    plot_path = joinpath(pwd(), "final_results_plots")
    if !isdir(plot_path); mkpath(plot_path); end

    # Create one plot for each observable
    for obs_id in unique(results_df.observableId)
        
        plt = plot(title="Observable: $obs_id", xlabel="Time", ylabel="Value", legend=:outertopright)
        
        # Filter the results for the current observable
        obs_df = filter(:observableId => ==(obs_id), results_df)

        # For each condition, plot the experimental data and the simulated model output
        for condition_id in unique(obs_df.simulationConditionId)
            cond_df = filter(:simulationConditionId => ==(condition_id), obs_df)
            
            # Plot experimental data as scatter points
            plot!(plt, cond_df.time, cond_df.measurement, seriestype=:scatter, label="Data: $condition_id", markersize=4)

            # Plot simulated data as a line
            sort!(cond_df, :time)
            plot!(plt, cond_df.time, cond_df.simulated, seriestype=:line, label="Model: $condition_id", linewidth=2)
        end
        
        # Save the plot for the current observable
        plot_filename = joinpath(plot_path, "$(obs_id).png")
        savefig(plt, plot_filename)
        println("✅ Plot saved to: $plot_filename")
    end

    println("\n--- Visualization Complete ---")
end

function plot_waterfall(multistart_result::PEtabMultistartResult)
    
    # Define the output directory and create it if it doesn't exist
    plot_dir = joinpath(pwd(), "final_results_plots")
    if !isdir(plot_dir); mkpath(plot_dir); end
    save_path = joinpath(plot_dir, "waterfall_plot.png")

    # Extract the final cost (fmin) from each successful run
    fmin_values = [run.fmin for run in multistart_result.runs if isfinite(run.fmin)]

    if isempty(fmin_values)
        @warn "No successful runs found to create a waterfall plot."
        return
    end

    # Sort the values from best to worst
    sort!(fmin_values)

    # Create the plot
    plt = plot(
        1:length(fmin_values),
        fmin_values,
        seriestype=:path,
        marker=:circle,
        title="Waterfall Plot",
        xlabel="Sorted Optimization Run Index",
        ylabel="Final Objective Value (log scale)",
        legend=false,
        yscale=:log10,
        framestyle=:box,
        grid=true
    )

    savefig(plt, save_path)
    println("✅ Waterfall plot saved to: $save_path")
end