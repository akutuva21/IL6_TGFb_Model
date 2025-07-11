using Pkg
Pkg.activate("bngl_julia/")

# Load all necessary packages
using DifferentialEquations, PEtab, Sundials, ComponentArrays, Printf
using DataFrames, CSV, Plots, SymbolicUtils, Symbolics
using ModelingToolkit: species, parameters, observed, unknowns, get_iv # Ensure species is here

# Include your project's setup functions
include("src/model_param_est_robustness.jl")
include("src/visualization.jl")

println("--- Final Results Processing: Exporting All Observables and Generating Plots ---")

# --- CONFIGURATION ---
const NUM_SIMULATION_POINTS = 200 # Increase for smoother curves

# --- 1. Set up the PEtab Model and other required objects ---
println("Setting up PEtab model and objects...")

# Use the current setup function with required parameters
enable_preeq = true
model_net_path = "model_even_smaller/2025_07_09__16_30_17/model_even_smaller.net"
data_path = "SimData/measurements_time_course.tsv"
config_path = "config.yml"

setup_results = setup_petab_problem(enable_preeq, model_net_path, data_path, config_path)
if isnothing(setup_results)
    @error "Failed to build PEtabModel. Cannot proceed."
    exit()
end

# Extract the PEtab model and create the problem
petab_model = setup_results.petab_model
petab_problem = PEtabODEProblem(petab_model, verbose=false)

# --- 2. Define the Best-Fit Parameter Set ---
p_best_log_scale = [2.000009014770089,-1.000591771818341,-0.0004005251525854358,-1.3080306173525476,-0.0003249141633903247,-1.3009220060659592,-1.0004255238121293,-0.9998167884972,-1.0005536541510653,-0.00025741962192025943,-1.3010185499486489,-1.3014470111366694,-1.6993663520955826,-0.6927858012582797,-2.0003906199452524,1.7377420288665353,-0.3010193317607902,2.0085996770857073,-2.0000428232970324,1.69896781955379,-0.0003385816387185429,2.0211805081222685]

# --- 3. Prepare Parameter Maps ---
println("Preparing parameter maps...")

# Get parameter names from the petab problem (these will have log10_ prefix)
param_names = petab_problem.xnames
println("Parameter names: ", param_names)
println("Number of parameters expected: ", length(param_names))

# The parameter values we have are for the base parameter names, but PEtab expects log10_ prefixed names
# So we need to create the ComponentArray with the correct log10_ prefixed names
expected_names = petab_problem.model_info.xindices.xids[:estimate_ps]
println("Expected parameter names from PEtab: ", expected_names)

# Create ComponentArray with the best-fit parameters using the correct names
if length(p_best_log_scale) != length(expected_names)
    @error "Parameter vector length ($(length(p_best_log_scale))) doesn't match expected length ($(length(expected_names)))"
    exit()
end

p_est = ComponentArray(; (expected_names .=> p_best_log_scale)...)
println("✅ Parameter maps prepared with correct naming.")

# --- 4. Use PEtab's Built-in Simulation ---
println("Using PEtab's simulation capabilities...")

# Calculate simulated values for all conditions using PEtab
simulated_vals = petab_problem.simulated_values(p_est)

# Get the measurement data table
measurements_df = petab_problem.model_info.model.petab_tables[:measurements]
results_df = deepcopy(measurements_df)
results_df[!, :simulated] = simulated_vals

println("INFO: Simulated $(length(simulated_vals)) data points across all conditions and observables.")

# --- 5. Export Results and Generate Plots ---
csv_dir = "final_results_csv"
if !isdir(csv_dir) mkdir(csv_dir) end
plots_dir = "final_results_plots"
if !isdir(plots_dir) mkdir(plots_dir) end

# Export combined results to CSV
CSV.write(joinpath(csv_dir, "simulation_results.csv"), results_df)
println("✅ Wrote simulation_results.csv with all conditions and observables")

# Generate plots for each observable
for obs_id in unique(results_df.observableId)
    p = plot(title="Observable: $obs_id", xlabel="Time", ylabel="Value", legend=:outertopright, framestyle=:box)
    
    # Filter the results for the current observable
    obs_df = filter(:observableId => ==(obs_id), results_df)

    # For each condition, plot the experimental data and the simulated model output
    for condition_id in unique(obs_df.simulationConditionId)
        cond_df = filter(:simulationConditionId => ==(condition_id), obs_df)
        
        # Plot experimental data as scatter points
        scatter!(p, cond_df.time, cond_df.measurement, label="Data: $condition_id", markershape=:xcross, markersize=4)

        # Plot simulated data as a line
        sort!(cond_df, :time)
        plot!(p, cond_df.time, cond_df.simulated, label="Model: $condition_id", linewidth=2)
    end
    
    # Save the plot
    plot_filename = joinpath(plots_dir, "$(obs_id).png")
    savefig(p, plot_filename)
    println("✅ Saved $(obs_id).png")
end

println("\n--- Processing Complete ---")