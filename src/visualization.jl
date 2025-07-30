# src/visualization.jl

# 1. Import Dependencies
using Plots; gr()
using DataFrames
using DifferentialEquations
using PEtab
using Printf
using CSV
using RecipesBase
using Colors

export run_visualization, plot_waterfall, plot_parameter_distribution, diagnose_multistart_data, plot_waterfall_native_fallback, plot_waterfall_custom_fallback, handle_Inf_vector!, assign_clustered_colors, add_reference_lines!

# Helper functions for recipe-based plotting
"""
    best_runs(res_ms::PEtabMultistartResult, n::Int)

Returns indices of the n best runs (lowest objective values) from multistart results.
"""
function best_runs(res_ms::PEtabMultistartResult, n::Int)
    finite_runs = findall(run -> isfinite(run.fmin), res_ms.runs)
    if isempty(finite_runs)
        return Int[]
    end
    
    # Get objective values for finite runs
    obj_values = [res_ms.runs[i].fmin for i in finite_runs]
    
    # Sort and take best n
    n_take = min(n, length(obj_values))
    sorted_idxs = sortperm(obj_values)[1:n_take]
    
    return finite_runs[sorted_idxs]
end

"""
    determine_yaxis(runs, plot_type::Symbol)

Determines whether to use linear or log scale based on value range.
"""
function determine_yaxis(runs, plot_type::Symbol)
    plot_type ∉ [:objective, :best_objective, :waterfall, :runtime_eval] && return :identity
    
    obj_values = [run.fmin for run in runs if isfinite(run.fmin)]
    isempty(obj_values) && return :identity
    
    min_val, max_val = extrema(obj_values)
    return (max_val / min_val) > 100 ? :log10 : :identity
end

"""
    objective_shift(runs, plot_type::Symbol, yaxis_scale)

Calculates shift needed to avoid negative values on log scales.
"""
function objective_shift(runs, plot_type::Symbol, yaxis_scale)
    obj_values = [run.fmin for run in runs if isfinite(run.fmin)]
    isempty(obj_values) && return 0.0
    
    min_val = minimum(obj_values)
    if (yaxis_scale != :identity) && (min_val <= 0)
        return abs(min_val) + 1
    end
    return 0.0
end

"""
    handle_Inf!(y_vals::Vector{Float64})

Handles infinite values by replacing with appropriate marker shapes.
Returns a single markershape symbol for consistency.
"""
function handle_Inf!(y_vals::Vector{Float64})
    has_inf = false
    for (i, val) in enumerate(y_vals)
        if isinf(val)
            y_vals[i] = maximum(filter(isfinite, y_vals)) * 1.1  # Place above finite values
            has_inf = true
        end
    end
    # Return single markershape to avoid dimension mismatch warnings
    return has_inf ? :utriangle : :circle
end

"""
    handle_Inf_vector!(y_vals::Vector{Float64})

Handles infinite values and returns a vector of markershapes when needed.
"""
function handle_Inf_vector!(y_vals::Vector{Float64})
    shapes = fill(:circle, length(y_vals))
    for (i, val) in enumerate(y_vals)
        if isinf(val)
            y_vals[i] = maximum(filter(isfinite, y_vals)) * 1.1  # Place above finite values
            shapes[i] = :utriangle  # Triangle for infinite values
        end
    end
    return shapes
end

"""
    objective_value_clustering(runs; threshold_factor=0.01)

Groups optimization runs by similar objective values for color coding.
"""
function objective_value_clustering(runs; threshold_factor=0.01)
    n = length(runs)
    obj_values = [run.fmin for run in runs]
    
    # Create pairs of (index, objective_value) and sort by objective value
    idxs_v_sorted = sort([(i, v) for (i, v) in enumerate(obj_values) if isfinite(v)], by=p -> p[2])
    
    if isempty(idxs_v_sorted)
        return reshape(ones(Int, n), 1, n)
    end
    
    # Determine clustering threshold
    min_val, max_val = extrema([p[2] for p in idxs_v_sorted])
    thres = (max_val - min_val) * threshold_factor
    
    # Assign colors based on clustering
    colors = fill(-1, n)
    cur_val = idxs_v_sorted[1][2]
    cur_color = 1
    
    for (i, v) in idxs_v_sorted
        if v > cur_val + thres
            cur_val = v
            cur_color += 1
        end
        colors[i] = cur_color
    end
    
    # Set non-finite runs to a special color
    for i in 1:n
        if colors[i] == -1
            colors[i] = maximum(colors[colors .!= -1]) + 1
        end
    end
    
    return reshape(colors, 1, n)
end

# Recipe-based plotting for PEtabMultistartResult
"""
Custom recipe for plotting PEtabMultistartResult with waterfall plots.
This follows the PEtab.jl plotting convention while providing robust error handling.
"""
@recipe function f(res_ms::PEtabMultistartResult;
                   plot_type = :waterfall,
                   best_idxs_n = (plot_type in [:waterfall, :runtime_eval] ?
                                  length(res_ms.runs) : 10),
                   idxs = best_runs(res_ms, best_idxs_n),
                   clustering_function = objective_value_clustering,
                   yaxis_scale = determine_yaxis(res_ms.runs[idxs], plot_type),
                   obj_shift = objective_shift(res_ms.runs[idxs], plot_type, yaxis_scale))
    
    if plot_type == :waterfall
        # Fixed plot attributes
        label --> ""
        yaxis --> yaxis_scale
        xlabel --> "Optimization run index"
        yguide --> "Final objective value"
        seriestype --> :scatter
        title --> "Optimization Results (Waterfall Plot)"
        
        # Tunable visual attributes
        ms --> 8
        markerstrokewidth --> 1
        size --> (800, 500)
        dpi --> 300
        
        # Prepare data
        if isempty(idxs)
            @warn "No finite runs found for waterfall plot"
            return [], []
        end
        
        # Get objective values and sort for waterfall effect
        y_vals = [res_ms.runs[i].fmin for i in idxs]
        sorted_indices = sortperm(y_vals)
        x_vals = 1:length(idxs)
        y_vals_sorted = y_vals[sorted_indices]
        
        # Apply objective shift to avoid negative values on log scale
        if obj_shift > 0
            y_vals_sorted = y_vals_sorted .+ obj_shift
        end
        
        # Handle infinite values and get consistent markershape
        markershape --> handle_Inf!(y_vals_sorted)  # Returns single scalar to avoid dimension mismatch
        
        # Color by clustering
        colors = clustering_function(res_ms.runs[idxs])
        color --> colors[1, sorted_indices]
        
        # Add connecting line
        @series begin
            seriestype := :line
            color := :blue
            alpha := 0.6
            linewidth := 2
            label := ""
            x_vals, y_vals_sorted
        end
        
        # Highlight best result
        @series begin
            seriestype := :scatter
            color := :red
            markersize := 10
            markerstrokewidth := 2
            markerstrokecolor := :darkred
            label := "Best Result"
            [1], [y_vals_sorted[1]]
        end
        
        return x_vals, y_vals_sorted
    end
    
    # Default fallback for other plot types
    return [], []
end

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
    println("Manually solving ODE for all conditions with high precision...")
    # Pass the high-precision tolerances from the odesolver object to ensure
    # visualization uses the same numerical precision as parameter estimation
    ode_solutions = PEtab.solve_all_conditions(
        theta_optim, 
        petab_prob, 
        odesolver.solver;
        abstol=odesolver.abstol,
        reltol=odesolver.reltol,
        maxiters=odesolver.maxiters
    )
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
    diagnose_multistart_data(multistart_result::PEtabMultistartResult, petab_prob::PEtabODEProblem)

Diagnostic function to understand the structure of multistart results and identify
potential issues causing plotting errors, especially parameter count mismatches.
"""
function diagnose_multistart_data(multistart_result::PEtabMultistartResult, petab_prob::PEtabODEProblem)
    println("\n--- Multistart Result Diagnostics ---")
    
    total_starts_attempted = multistart_result.nmultistarts  # Fixed: removed underscore
    valid_runs = multistart_result.runs
    num_valid_runs = length(valid_runs)
    num_failed_runs = total_starts_attempted - num_valid_runs
    
    println("Total starts attempted: $(total_starts_attempted)")
    println("Number of valid (finite) runs returned: $(num_valid_runs)")
    println("Number of failed/discarded runs: $(num_failed_runs)")
    println("Best objective value found: $(multistart_result.fmin)")
    
    if !isempty(multistart_result.xmin)
        println("Number of parameters in best result: $(length(multistart_result.xmin))")
    end
    
    println("\n--- PEtab Problem Diagnostics ---")
    println("Number of estimated parameters in model: $(petab_prob.nparameters_estimate)")
    
    # Check for parameter count mismatches
    if !isempty(multistart_result.xmin) && length(multistart_result.xmin) != petab_prob.nparameters_estimate
        @warn "PARAMETER COUNT MISMATCH DETECTED!"
        @warn "Multistart result has $(length(multistart_result.xmin)) parameters"
        @warn "PEtab problem expects $(petab_prob.nparameters_estimate) parameters"
        @warn "This is likely causing the plotting error!"
    end
end


"""
    plot_waterfall(multistart_result::PEtabMultistartResult)

Generates a professional waterfall plot following PyPESTO design standards.
Creates a layered plot with connecting lines and individual colored points.
"""
function plot_waterfall(multistart_result::PEtabMultistartResult)
    plot_dir = joinpath(pwd(), "final_results_plots")
    if !isdir(plot_dir); mkpath(plot_dir); end
    save_path = joinpath(plot_dir, "waterfall_plot.png")

    # Extract and validate finite objective values
    objective_values = Float64[]
    for run in multistart_result.runs
        if isfinite(run.fmin) && !isnan(run.fmin)
            push!(objective_values, run.fmin)
        end
    end

    if isempty(objective_values)
        @warn "No finite objective function values found. Cannot create a waterfall plot."
        return
    end

    # Sort for waterfall effect
    sorted_values = sort(objective_values)
    n_runs = length(sorted_values)
    start_indices = 1:n_runs
    
    println("Creating PyPESTO-style waterfall plot with $n_runs finite runs")

    # Determine y-axis scaling (log vs linear)
    y_min, y_max = extrema(sorted_values)
    use_log_scale = (y_max / y_min) > 100
    
    # Create base plot
    plt = plot(
        title = "Waterfall plot",
        xlabel = "Ordered optimizer run",
        ylabel = "Function value",
        size = (800, 500),
        dpi = 300,
        legend = false,
        framestyle = :box
    )
    
    # Step 1: Plot connecting line (light gray, semi-transparent) - PyPESTO style
    if use_log_scale
        plot!(plt, start_indices, sorted_values,
              color = RGBA(0.7, 0.7, 0.7, 0.6),
              linewidth = 1,
              yscale = :log10,
              label = "")
    else
        plot!(plt, start_indices, sorted_values,
              color = RGBA(0.7, 0.7, 0.7, 0.6),
              linewidth = 1,
              label = "")
    end
    
    # Step 2: Plot individual points with color coding
    for (i, fval) in enumerate(sorted_values)
        # Color scheme: red for best, blue gradient for others
        if i == 1
            point_color = :red
            marker_shape = :star
            marker_size = 10
        elseif i <= 5
            point_color = :orange
            marker_shape = :circle
            marker_size = 8
        else
            point_color = :blue
            marker_shape = :circle
            marker_size = 6
        end
        
        scatter!(plt, [i], [fval],
                color = point_color,
                markershape = marker_shape,
                markersize = marker_size,
                markerstrokewidth = 1,
                markerstrokecolor = :black,
                alpha = 0.8,
                label = "")
    end
    
    # Step 3: Set appropriate x-axis ticks (integer values only)
    x_tick_spacing = max(1, n_runs ÷ 10)  # Approximate PyPESTO's MaxNLocator behavior
    x_ticks = 1:x_tick_spacing:n_runs
    plot!(plt, xticks = x_ticks)
    
    savefig(plt, save_path)
    println("✅ Enhanced waterfall plot saved to: $save_path")
    
    return plt
end

"""
    assign_clustered_colors(fvals::Vector{Float64}, n_clusters::Int=3)
    
Assign colors based on clustering of objective function values.
"""
function assign_clustered_colors(fvals::Vector{Float64}, n_clusters::Int=3)
    if length(fvals) < n_clusters
        return fill(:blue, length(fvals))
    end
    
    # Simple quantile-based clustering
    sorted_fvals = sort(fvals)
    n_vals = length(sorted_fvals)
    
    # Define cluster boundaries using quantiles
    cluster_boundaries = Float64[]
    for i in 1:(n_clusters-1)
        boundary_idx = round(Int, (i / n_clusters) * n_vals)
        push!(cluster_boundaries, sorted_fvals[boundary_idx])
    end
    push!(cluster_boundaries, Inf)  # Upper boundary
    
    # Define colors for each cluster
    cluster_colors = [:red, :orange, :blue, :green, :purple][1:n_clusters]
    
    # Assign colors based on cluster membership
    colors = Symbol[]
    for fval in fvals
        cluster_idx = findfirst(x -> fval <= x, cluster_boundaries)
        cluster_idx = min(cluster_idx, length(cluster_colors))
        push!(colors, cluster_colors[cluster_idx])
    end
    
    return colors
end

"""
    add_reference_lines!(plt, reference_values::Vector{Float64}; labels=nothing)
    
Add horizontal reference lines to waterfall plot.
"""
function add_reference_lines!(plt, reference_values::Vector{Float64}; labels=nothing)
    for (i, ref_val) in enumerate(reference_values)
        label = isnothing(labels) ? "Reference $i" : labels[i]
        hline!(plt, [ref_val], 
               linestyle = :dash,
               color = :black,
               alpha = 0.7,
               linewidth = 2,
               label = label)
    end
    return plt
end

"""
    plot_waterfall_custom_fallback(multistart_result::PEtabMultistartResult)

Fallback custom waterfall plot implementation that doesn't rely on recipes.
Uses scalar markershapes to avoid dimension mismatch warnings.
"""
function plot_waterfall_custom_fallback(multistart_result::PEtabMultistartResult)
    
    plot_dir = joinpath(pwd(), "final_results_plots")
    if !isdir(plot_dir); mkpath(plot_dir); end
    save_path = joinpath(plot_dir, "waterfall_plot_fallback.png")

    # Extract objective function values from all runs
    objective_values = Float64[]
    for run in multistart_result.runs
        if isfinite(run.fmin) && !isnan(run.fmin)
            push!(objective_values, run.fmin)
        end
    end

    if isempty(objective_values)
        @warn "No finite objective function values found. Cannot create a waterfall plot."
        return
    end

    # Sort objective values for waterfall effect
    sorted_values = sort(objective_values)
    n_runs = length(sorted_values)

    println("Creating custom fallback waterfall plot with $n_runs finite runs")

    # Handle infinite values and determine markershape
    markershape_to_use = handle_Inf!(sorted_values)  # Returns single scalar

    # Create the waterfall plot with consistent scalar attributes
    plt = plot(
        1:n_runs, sorted_values,
        seriestype=:scatter,
        title="Optimization Results (Waterfall Plot)",
        xlabel="Run Index (sorted by objective value)",
        ylabel="Objective Function Value",
        legend=false,
        markersize=8,
        markercolor=:blue,
        markerstrokewidth=2,
        markershape=markershape_to_use,  # Use scalar to avoid dimension mismatch
        size=(800, 500),
        dpi=300
    )

    # Add connecting line
    plot!(plt, 1:n_runs, sorted_values, 
          seriestype=:line, 
          color=:blue, 
          alpha=0.6, 
          linewidth=2)

    # Highlight the best result with explicit scalar attributes
    scatter!(plt, [1], [sorted_values[1]], 
             markersize=12, 
             markercolor=:red,
             markerstrokewidth=2,
             markerstrokecolor=:darkred,
             markershape=:star,  # Explicit scalar markershape
             label="Best Result")

    # Add summary statistics as text
    annotate!(plt, n_runs * 0.7, maximum(sorted_values) * 0.9,
              text("Best: $(round(minimum(sorted_values), digits=3))\n" *
                   "Worst: $(round(maximum(sorted_values), digits=3))\n" *
                   "Runs: $n_runs", 10, :left))

    savefig(plt, save_path)
    println("✅ Custom fallback waterfall plot saved to: $save_path")
end


"""
    plot_waterfall_native_fallback(multistart_result::PEtabMultistartResult, petab_prob::PEtabODEProblem)

Attempts to use the native PEtab.jl plotting with error handling and fallback to custom implementation.
"""
function plot_waterfall_native_fallback(multistart_result::PEtabMultistartResult, petab_prob::PEtabODEProblem)
    plot_dir = joinpath(pwd(), "final_results_plots")
    if !isdir(plot_dir); mkpath(plot_dir); end
    save_path = joinpath(plot_dir, "waterfall_plot_native.png")

    # Check data validity first
    if isempty(multistart_result.runs) || all(run -> !isfinite(run.fmin), multistart_result.runs)
        @warn "No finite objective function values found. Cannot create a waterfall plot."
        return
    end

    try
        # Try the native plotting with explicit parameter specification
        if hasmethod(plot, (typeof(multistart_result), typeof(petab_prob)))
            plt = plot(multistart_result, petab_prob, plot_type=:waterfall)
        else
            # Fallback to just multistart result
            plt = plot(multistart_result, plot_type=:waterfall)
        end
        
        savefig(plt, save_path)
        println("✅ Native waterfall plot saved to: $save_path")
    catch e
        @warn "Native plotting failed: $e"
        println("Falling back to custom implementation...")
        plot_waterfall(multistart_result)  # Use our robust custom implementation
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

    n_est = length(keys(multistart_result.xmin))
    println("INFO: Creating parameter distribution plot for $n_est estimated parameters")

    # --- MODIFICATION: Work with Symbols directly ---
    param_names_symbols = petab_prob.xnames # This is already a Vector{Symbol}
    param_names_strings = string.(param_names_symbols) # Use this for plot labels only
    n_params_plot = length(param_names_symbols)
    
    lower_bounds = collect(petab_prob.lower_bounds)
    upper_bounds = collect(petab_prob.upper_bounds)
    
    all_x_estimates = []
    for run in multistart_result.runs
        if !isempty(run.xmin)
            push!(all_x_estimates, collect(run.xmin))
        end
    end
    
    if isempty(all_x_estimates)
        @warn "No valid parameter estimates found to create a distribution plot."
        return
    end

    best_x = collect(multistart_result.xmin)
    plot_height = max(400, n_params_plot * 40) # Increased height for better readability
    
    plt = plot(
        title="Estimated Parameters",
        xlabel="Parameter Value (log10)",
        ylabel="Parameter",
        legend=:topright, # Moved legend inside
        # --- MODIFICATION: Use the string version for labels ---
        yticks=(1:n_params_plot, param_names_strings),
        yflip=true,
        framestyle=:box,
        size=(900, plot_height),
        dpi=300
    )

    y_values = 1:n_params_plot
    for x_vec in all_x_estimates
        if x_vec != best_x
            plot!(plt, x_vec, y_values, seriestype=:path, color=:gray, alpha=0.3, linewidth=1, label="")
        end
    end
    
    bounds_y = vcat(y_values, y_values)
    bounds_x = vcat(lower_bounds, upper_bounds)
    scatter!(plt, bounds_x, bounds_y, marker=:+, color=:black, markersize=4, label="Bounds")

    # --- START OF MODIFICATION ---
    # This section is rewritten to correctly plot the true values as a connected line.
    if !isnothing(reference_values)
        ref_x_values_log10 = Float64[]
        
        println("INFO: Matching and transforming reference (true) values...")
        println("DIAGNOSTIC: Available true parameter names: $(collect(keys(reference_values)))")
        println("DIAGNOSTIC: Available true parameter types: $(typeof.(collect(keys(reference_values))))")
        println("DIAGNOSTIC: Estimated parameter symbols: $param_names_symbols")
        
        # Try multiple matching strategies for robustness
        matched_count = 0
        for param_name_sym in param_names_symbols
            # Remove log10_ prefix if present for matching
            base_name = string(param_name_sym)
            if startswith(base_name, "log10_")
                base_name = base_name[7:end]  # Remove "log10_" prefix
            end
            
            # Try multiple matching strategies
            true_value = nothing
            match_method = ""
            
            # Strategy 1: Try base_name as Symbol
            base_name_sym = Symbol(base_name)
            if haskey(reference_values, base_name_sym)
                true_value = reference_values[base_name_sym]
                match_method = "Symbol match"
            # Strategy 2: Try base_name as String
            elseif haskey(reference_values, base_name)
                true_value = reference_values[base_name]
                match_method = "String match"
            # Strategy 3: Try original param_name_sym directly
            elseif haskey(reference_values, param_name_sym)
                true_value = reference_values[param_name_sym]
                match_method = "Direct symbol match"
            # Strategy 4: Try case-insensitive matching
            else
                for (ref_key, ref_val) in reference_values
                    ref_key_str = string(ref_key)
                    if lowercase(ref_key_str) == lowercase(base_name)
                        true_value = ref_val
                        match_method = "Case-insensitive match"
                        break
                    end
                end
            end
            
            if !isnothing(true_value)
                true_log10_value = log10(true_value)
                push!(ref_x_values_log10, true_log10_value)
                matched_count += 1
                println("  ✓ Found '$base_name' → true value = $true_value (log10: $true_log10_value) [$match_method]")
            else
                @warn "Could not find true value for parameter '$base_name'. Available keys: $(collect(keys(reference_values)))"
                push!(ref_x_values_log10, NaN) 
            end
        end
        
        println("INFO: Successfully matched $matched_count out of $(length(param_names_symbols)) parameters")
        
        if !isempty(filter(!isnan, ref_x_values_log10))
            # Plot the true values as a distinct blue line with star markers
            plot!(plt, ref_x_values_log10, y_values, 
                    seriestype=:path, 
                    color=:blue, 
                    linewidth=2.5,
                    markershape=:star5,
                    markersize=8,
                    markerstrokecolor=:blue,
                    label="True Values")
        end
    end

    if !isempty(best_x)
        plot!(plt, best_x, y_values, 
              seriestype=:path, 
              color=:red, 
              alpha=0.9,
              linewidth=2.5, # Increased line width
              marker=:circle,
              markersize=4,
              label="Best Fit")
    end
    
    savefig(plt, save_path)
    println("✅ Parameter distribution plot saved to: $save_path")
end