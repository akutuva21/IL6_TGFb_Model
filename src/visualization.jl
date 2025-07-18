# src/visualization.jl

# 1. Import Dependencies
using Plots; gr()
using DataFrames
using DifferentialEquations
using PEtab
using Printf
using CSV

export run_visualization, plot_waterfall, plot_parameter_distribution

"""
    run_visualization(
        theta_optim::Vector{Float64},
        petab_prob::PEtabODEProblem,
        odesolver::ODESolver
    )

Generates and saves plots comparing the model simulation against measurement data.

This function manually solves the ODE for all conditions and creates plots for each observable.
"""
function run_visualization(
    theta_optim::Vector{Float64},
    petab_prob::PEtabODEProblem,
    odesolver::ODESolver
)
    println("\n--- Starting Visualization (Manual Workaround) ---")

    # Step 1: Manually solve the ODE for all conditions
    println("Manually solving ODE for all conditions...")
    ode_solutions = PEtab.solve_all_conditions(theta_optim, petab_prob, odesolver.solver)
    println("✅ ODE solutions obtained.")

    plot_path = joinpath(pwd(), "final_results_plots")
    if !isdir(plot_path); mkpath(plot_path); end

    # Step 2: Manually plot each observable
    measurements_df = petab_prob.model_info.petab_measurements
    observable_ids = unique(measurements_df.observable_id)

    for obs_id in observable_ids
        println("Plotting observable: $obs_id")
        
        plt = plot(
            title=string(obs_id),
            xlabel="Time",
            ylabel="Value",
            legend=:outertopright
        )

        relevant_conditions = unique(measurements_df.simulation_condition_id[measurements_df.observable_id .== obs_id])

        for condition_id in relevant_conditions
            
            # Plot measurement data
            data_for_plot = measurements_df.measurement[
                (measurements_df.observable_id .== obs_id) .& 
                (measurements_df.simulation_condition_id .== condition_id)
            ]
            time_points = measurements_df.time[
                (measurements_df.observable_id .== obs_id) .& 
                (measurements_df.simulation_condition_id .== condition_id)
            ]
            scatter!(plt, time_points, data_for_plot, label="Data ($condition_id)")

            # Plot simulation results
            sol_key = nothing
            for key in keys(ode_solutions)
                if occursin(string(condition_id), string(key))
                    sol_key = key
                    break
                end
            end

            if !isnothing(sol_key)
                solution = ode_solutions[sol_key]
                
                # --- THE FIX: Use the correct path to the 'h' function ---
                simulated_values = [
                    petab_prob.model_info.model.h(sol_u, sol_t, solution.prob.p, [], [], [], obs_id, nothing) 
                    for (sol_u, sol_t) in zip(solution.u, solution.t)
                ]
                
                plot!(plt, solution.t, simulated_values, label="Model ($condition_id)", linewidth=2)
            else
                @warn "Could not find a simulation solution for condition $condition_id"
            end
        end
        
        plot_filename = joinpath(plot_path, "plot_observable_$(obs_id).png")
        savefig(plt, plot_filename)
        println("✅ Plot saved to: $plot_filename")
    end

    println("\n--- Visualization Complete ---")
end


"""
    plot_waterfall(multistart_result::PEtabMultistartResult)

Generates and saves a waterfall plot from a multi-start optimization result.

This refactored version is a lightweight wrapper around the native PEtab.jl
waterfall plot functionality (`plot_type=:waterfall`), which is the recommended
way to visualize the distribution of final objective function values.
"""
function plot_waterfall(multistart_result::PEtabMultistartResult)
    
    plot_dir = joinpath(pwd(), "final_results_plots")
    if !isdir(plot_dir); mkpath(plot_dir); end
    save_path = joinpath(plot_dir, "waterfall_plot.png")

    # Check if there are any valid runs to plot
    if isempty(multistart_result.runs) || all(run -> !isfinite(run.fmin), multistart_result.runs)
        @warn "No finite objective function values found. Cannot create a waterfall plot."
        return
    end

    # FIX A: Get the actual number of estimated parameters to prevent BoundsError
    n_est = length(keys(multistart_result.xmin))
    println("INFO: Creating waterfall plot for $n_est estimated parameters")

    # Generate the waterfall plot using the single, native PEtab.jl function call.
    # This automatically handles sorting, scaling (log or linear), and color-clustering.
    # The native function should handle the parameter count correctly, but we provide explicit info
    try
        plt = plot(multistart_result; plot_type=:waterfall)
        savefig(plt, save_path)
        println("✅ Waterfall plot saved to: $save_path")
    catch e
        @warn "Failed to create waterfall plot. This may be due to parameter count mismatch. Error: $e"
        @warn "Attempted to plot $n_est parameters from multistart result"
    end
end


"""
    plot_parameter_distribution(multistart_result::PEtabMultistartResult, petab_prob::PEtabODEProblem; reference_values=nothing)

Creates a parameter distribution plot (parallel coordinates) based on the provided
Julia multi-start result. Each line represents a single optimization run.

NOTE: This function's implementation is intentionally preserved. The native
`PEtab.jl` `:parallel_coordinates` plot normalizes all parameter values to a 0-1
range. This function, in contrast, plots the raw (log10-scaled) parameter values,
which can be more informative for assessing parameter magnitudes and bounds.
Additionally, this function supports a `reference_values` feature not available
in the native plot.
"""
function plot_parameter_distribution(multistart_result::PEtabMultistartResult, petab_prob::PEtabODEProblem; reference_values=nothing)
    println("\n--- Generating Parameter Distribution Plot (Custom Julia) ---")
    plot_dir = joinpath(pwd(), "final_results_plots")
    if !isdir(plot_dir); mkpath(plot_dir); end
    save_path = joinpath(plot_dir, "parameter_distribution_plot.png")

    # --- FIX A: Get the actual number of estimated parameters to prevent BoundsError ---
    n_est = length(keys(multistart_result.xmin))
    println("INFO: Creating parameter distribution plot for $n_est estimated parameters")

    # --- 1. Extract necessary data ---
    param_names = string.(petab_prob.xnames)
    n_params = length(param_names)
    
    # Verify that the multistart result matches the PEtab problem
    if n_est != n_params
        @warn "Parameter count mismatch: PEtab problem has $n_params parameters but multistart result has $n_est"
        @warn "This may indicate a configuration error in your parameter estimation setup"
    end
    
    # --- FIX: Ensure bounds vectors match the number of estimated parameters ---
    lower_bounds = collect(petab_prob.lower_bounds)
    upper_bounds = collect(petab_prob.upper_bounds)
    
    # Verify that bounds have the correct length
    if length(lower_bounds) != n_params || length(upper_bounds) != n_params
        @warn "Bounds length mismatch: expected $n_params, got $(length(lower_bounds)) and $(length(upper_bounds))"
        @warn "This may cause plotting issues. Bounds will be truncated or extended."
        # Truncate or extend bounds to match n_params
        lower_bounds = resize!(lower_bounds, n_params)
        upper_bounds = resize!(upper_bounds, n_params)
    end
    # --- END FIX ---

    # Convert ComponentVectors to standard Vectors using collect()
    all_x_estimates = [collect(run.xmin) for run in multistart_result.runs if !isempty(run.xmin)]
    if isempty(all_x_estimates)
        @warn "No valid parameter estimates found to create a distribution plot."
        return
    end

    best_x = collect(multistart_result.xmin)
    plot_height = max(400, n_params * 30)
    
    # --- 2. Create the plot canvas ---
    plt = plot(
        title="Estimated parameters",
        xlabel="Parameter value (log10)",
        ylabel="Parameter",
        legend=false,
        yticks=(1:n_params, param_names),
        yflip=true,
        framestyle=:box,
        size=(800, plot_height)
    )

    # --- 3. Plot all optimization runs ---
    y_values = 1:n_params
    for x_vec in all_x_estimates
        if x_vec != best_x
            plot!(plt, x_vec, y_values, seriestype=:path, color=:gray, alpha=0.3, linewidth=1)
        end
    end
    
    # --- 4. Plot parameter bounds ---
    bounds_y = vcat(y_values, y_values)
    bounds_x = vcat(lower_bounds, upper_bounds)
    scatter!(plt, bounds_x, bounds_y, marker=:+, color=:black, markersize=4, label="")

    # --- 5. Add reference values if provided ---
    if !isnothing(reference_values)
        ref_x_values = Float64[]
        ref_y_values = Int[]
        for (i, param_name) in enumerate(param_names)
            # Check for both String and Symbol keys for robustness
            if haskey(reference_values, param_name)
                push!(ref_x_values, reference_values[param_name])
                push!(ref_y_values, i)
            elseif haskey(reference_values, Symbol(param_name))
                push!(ref_x_values, reference_values[Symbol(param_name)])
                push!(ref_y_values, i)
            end
        end
        
        if !isempty(ref_x_values)
            scatter!(plt, ref_x_values, ref_y_values, 
                    marker=:star, 
                    color=:blue, 
                    markersize=8, 
                    markerstrokewidth=1,
                    markerstrokecolor=:blue,
                    label="Reference values")
        end
    end

    # --- 6. Highlight the single best run ---
    if !isempty(best_x)
        plot!(plt, best_x, y_values, 
              seriestype=:path, 
              color=:red, 
              alpha=0.9,
              linewidth=2,
              marker=:circle,
              markersize=3,
              label="Best Run")
    end
    
    savefig(plt, save_path)
    println("✅ Parameter distribution plot saved to: $save_path")
end