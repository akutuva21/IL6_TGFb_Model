# Load all necessary packages
using DifferentialEquations, PEtab, Sundials, ComponentArrays, Printf, Logging
using DataFrames, CSV, Plots, SymbolicUtils, Symbolics
using ModelingToolkit: species, parameters, observed, unknowns, get_iv # Ensure species is here

# Include your project's setup functions
include("src/model_param_est_robustness.jl")
include("src/visualization.jl")

# Setup logging
global_logger(ConsoleLogger(stderr, Logging.Info))

@info "--- Final Results Processing: Exporting All Observables and Generating Plots ---"

# --- CONFIGURATION ---
const NUM_SIMULATION_POINTS = 200 # Increase for smoother curves

# --- 1. Set up the PEtab Model and other required objects ---
@info "Setting up PEtab model and objects..."

# Use the current setup function with required parameters
petab_problem_path = "petab_problem.yml"

setup_results = setup_petab_problem(petab_problem_path)
if isnothing(setup_results)
    @error "Failed to build PEtabModel. Cannot proceed."
    exit()
end

# Extract the PEtab model and create the problem
petab_model = setup_results.petab_model
petab_problem = PEtabODEProblem(petab_model, verbose=false)

# --- 2. Define the Best-Fit Parameter Set ---
p_best_log_scale = [94.19428463, 6.275391417, 0.032118074, 100, 0.021080193, 0.001, 0.042211486, 6.975481504, 0.085109555, 0.042944632, 5, 0.225434403, 0.027451239, 0.000138633, 86.0247591, 0.503223142, 101.2027305, 0.011931676, 56.86613769, 39.06075951, 102.4572824]

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