# src/visualization.jl

# 1. Import Dependencies
using Plots; gr()
using DataFrames
using DifferentialEquations
using PEtab
using Printf
using CSV

# Export the original function names
export run_visualization, plot_waterfall, plot_parameter_distribution

"""
    run_visualization(
        theta_optim::Vector{Float64},
        petab_prob::PEtabODEProblem
    )

Generates and saves plots comparing the model simulation against measurement data.

This refactored version uses the native `PEtab.plot` function. To do so, it
wraps the provided `theta_optim` vector in a minimal `PEtabOptimisationResult`
struct, which is the input type the plotting function expects.
"""
function run_visualization(
    theta_optim::Vector{Float64},
    petab_prob::PEtabODEProblem
)
    println("\n--- Starting Visualization (using native PEtab.jl) ---")

    # The native plot function requires a result struct. We create a minimal
    # PEtabOptimisationResult to wrap the provided parameter vector.
    # The other fields can be placeholders as they are not used for this plot type.
    result_for_plotting = PEtab.PEtabOptimisationResult(
        :fminbox,      # Placeholder algorithm
        0,             # n_opts
        NaN,           # fmin
        theta_optim,   # The optimal parameters to use for simulation
        0,             # f_calls
        0,             # n_iterations
        0.0,           # run_time
        nothing,       # converged
        :Success       # ret
    )

    plot_path = joinpath(pwd(), "final_results_plots")
    if !isdir(plot_path); mkpath(plot_path); end

    # Create one plot for each unique observable ID
    observable_ids = unique(petab_prob.petab_model.measurements_data.observableId)
    for obs_id in observable_ids
        
        # Use the native PEtab.jl plot function for model fits.
        # It automatically handles data and simulation plotting.
        plt = plot(petab_prob, result_for_plotting; obs_id=[obs_id])
        
        # Save the plot for the current observable
        plot_filename = joinpath(plot_path, "$(obs_id).png")
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

    # Generate the waterfall plot using the single, native PEtab.jl function call.
    # This automatically handles sorting, scaling (log or linear), and color-clustering.
    plt = plot(multistart_result; plot_type=:waterfall)

    savefig(plt, save_path)
    println("✅ Waterfall plot saved to: $save_path")
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

    # --- 1. Extract necessary data ---
    param_names = string.(petab_prob.model_info.xindices.xids[:estimate_ps])
    n_params = length(param_names)
    
    lower_bounds = petab_prob.lower_bounds
    upper_bounds = petab_prob.upper_bounds

    all_x_estimates = [run.xmin for run in multistart_result.runs if !isempty(run.xmin)]
    if isempty(all_x_estimates)
        @warn "No valid parameter estimates found to create a distribution plot."
        return
    end

    best_x = multistart_result.xmin
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
                    markerstrokecolor=:black,
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