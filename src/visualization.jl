# src/visualization.jl

# 1. Import Dependencies
using Plots; gr()
using DataFrames
using DifferentialEquations
using PEtab
using Printf
using CSV

export run_visualization, plot_waterfall, plot_parameter_distribution

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
    
    plot_dir = joinpath(pwd(), "final_results_plots")
    if !isdir(plot_dir); mkpath(plot_dir); end
    save_path = joinpath(plot_dir, "waterfall_plot.png")

    fmin_values = [run.fmin for run in multistart_result.runs if isfinite(run.fmin) && run.fmin > 0]

    if isempty(fmin_values)
        @warn "No positive, finite objective function values found to create a waterfall plot."
        return
    end

    sort!(fmin_values)

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


"""
    plot_parameter_distribution(multistart_result::PEtabMultistartResult)

Creates a parameter distribution plot (parallel coordinates) similar to
pyPESTO's `visualize.parameters`, based on the provided Julia multi-start result.

Each line represents a single optimization run. The best overall run is
highlighted in red.
"""
function plot_parameter_distribution(multistart_result::PEtabMultistartResult, petab_prob::PEtabODEProblem)
    println("\n--- Generating Parameter Distribution Plot (Julia) ---")
    plot_dir = joinpath(pwd(), "final_results_plots")
    if !isdir(plot_dir); mkpath(plot_dir); end
    save_path = joinpath(plot_dir, "parameter_distribution_plot.png")

    # --- 1. Extract necessary data from the result object ---
    param_names = string.(petab_prob.model_info.xindices.xids[:estimate_ps])
    n_params = length(param_names)
    
    lower_bounds = petab_prob.lower_bounds
    upper_bounds = petab_prob.upper_bounds

    # Get all parameter estimates and filter out any failed runs
    all_x_estimates = [run.xmin for run in multistart_result.runs if !isempty(run.xmin)]
    if isempty(all_x_estimates)
        @warn "No valid parameter estimates found to create a distribution plot."
        return
    end

    # Get the single best parameter vector
    best_x = multistart_result.xmin

    plot_height = max(400, n_params * 30) # 30 pixels per parameter, with a minimum of 400
    # --- 2. Create the plot canvas ---
    plt = plot(
        title="Estimated parameters",
        xlabel="Parameter value (log10)",
        ylabel="Parameter",
        legend=false,
        yticks=(1:n_params, param_names), # Set y-axis ticks to parameter names
        yflip=true, # Match pyPESTO style (first param at top)
        framestyle=:box,
        size=(800, plot_height)
    )

    # --- 3. Plot all optimization runs as faint gray lines ---
    # The y-axis values are just the integer indices of the parameters
    y_values = 1:n_params
    for x_vec in all_x_estimates
        # Check if the vector is the best one to avoid plotting it twice
        if x_vec != best_x
            plot!(plt, x_vec, y_values, seriestype=:path, color=:gray, alpha=0.3, linewidth=1)
        end
    end
    
    # --- 4. Plot the parameter bounds as black '+' markers ---
    # We plot these as a scatter plot for clarity, mimicking the pyPESTO look
    bounds_y = vcat(y_values, y_values)
    bounds_x = vcat(lower_bounds, upper_bounds)
    scatter!(plt, bounds_x, bounds_y, marker=:+, color=:black, markersize=4, label="")

    # --- 5. Highlight the single best run in red ---
    if !isempty(best_x)
        plot!(plt, best_x, y_values, 
              seriestype=:path, 
              color=:red, 
              alpha=0.9,
              linewidth=2,
              marker=:circle,
              markersize=3,
              label="Best Run") # Label for clarity, though legend is off
    end
    
    # Save the final plot
    savefig(plt, save_path)
    println("✅ Parameter distribution plot saved to: $save_path")
end