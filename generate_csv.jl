using Pkg
Pkg.activate("./bngl_julia")

# Include your project's setup functions
include("src/model_param_est_robustness.jl")
include("src/visualization.jl")

# Load all necessary packages
using DifferentialEquations, PEtab, Sundials, ComponentArrays, Printf, Logging
using DataFrames, CSV, Plots, SymbolicUtils, Symbolics
using ModelingToolkit: parameters, observed, unknowns, get_iv # Ensure species is here

# Setup logging
global_logger(ConsoleLogger(stderr, Logging.Info))

@info "--- Final Results Processing: Exporting All Observables and Generating Plots ---"

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
p_best_log_scale = [-1.7743705843462096, -1.1234600738866587, -0.3435329241473487, -1.3488234447488323, -0.853123804260984, -1.1830266356398744, -2.544336262257378, -2.230429594279439, -0.0823595950579657, -0.9310305790508867, -0.04242072107927665, -0.9563859188186535, 0.40636731283430394, -1.3346988050272552, -0.6086813338040827, -1.4835308877895663, 1.9658670567329115, 1.9969601302162072, 1.6999306463761956, 2.0003525552505526, 1.6562675905452415]

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
simulated_vals = petab_problem.simulated_values(p_est)[:]  # Apply [:] workaround for SBML bug

# Get the measurement data table
measurements_df = petab_problem.model_info.model.petab_tables[:measurements]
results_df = deepcopy(measurements_df)
results_df[!, :simulated] = simulated_vals

println("INFO: Simulated $(length(simulated_vals)) data points across all conditions and observables.")

# --- 5. Export Results and Generate Plots ---
csv_dir = "final_results_csv"
if !isdir(csv_dir) mkdir(csv_dir) end

# Export combined results to CSV
CSV.write(joinpath(csv_dir, "simulation_results.csv"), results_df)
println("✅ Wrote simulation_results.csv with all conditions and observables")

println("\n--- Processing Complete ---")