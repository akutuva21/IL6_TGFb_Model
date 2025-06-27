# Corrected "Normal" Simulation Script
import matplotlib.pyplot as plt
import bionetgen
import pandas as pd
import os

# Create a directory for the simulation data and plots if it doesn't exist
if not os.path.exists("SimData"):
    os.makedirs("SimData")

model_path = "model_even_smaller.bngl"

# Define conditions more robustly, specifying all relevant parameters
# This now correctly reflects the experimental conditions you want to test.
conditions = {
    "Treg": {'IL6_0': 0.0,   'TGFb_0': 1.0},
    "TH17": {'IL6_0': 100.0, 'TGFb_0': 1.0},
}

simulation_results = {}
start_time = 0
end_time = 20
num_steps = 3000

# --- Running Simulations for Each Condition ---
try:
    for condition_name, param_values in conditions.items():
        print(f"--- Running simulation for {condition_name} condition ---")
        # Load the model fresh for each simulation
        model = bionetgen.bngmodel(model_path)

        # Set all parameters for the current condition
        for param_name, value in param_values.items():
            if param_name in model.parameters:
                print(f"  Setting parameter '{param_name}' to {value} for condition '{condition_name}'")
                model.parameters[param_name].value = value
                print(f"  Set parameter '{param_name}' to {value}")
            else:
                raise ValueError(f"Parameter '{param_name}' not found in the model.")
        # print all parameters values
        print("  Current model parameters:")
        for param in model.parameters:
            print(model.parameters[param])

        # Set up and run the simulation
        simulator = model.setup_simulator()
        result = simulator.simulate(end=end_time, steps=num_steps)

        # Store the results
        simulation_results[condition_name] = result
        print(f"Simulation for {condition_name} completed.")

except FileNotFoundError:
    print(f"Error: '{model_path}' not found.")
    simulation_results = None
except ValueError as ve:
    print(f"Error: {ve}")
    simulation_results = None

# --- Saving and Plotting (This part of your code was already good) ---
if simulation_results:
    output_filename = "SimData/normal.xlsx"
    with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
        print(f"\nSaving combined simulation results to '{output_filename}'...")
        all_species = set()
        for res in simulation_results.values():
            all_species.update(res.colnames)
        if 'time' in all_species:
            all_species.remove('time')

        for species in sorted(list(all_species)):
            combined_df = pd.DataFrame({'Time': simulation_results['TH17']['time']})
            for condition, result_df in simulation_results.items():
                if species in result_df.colnames:
                    combined_df[condition] = result_df[species]
                else:
                    combined_df[condition] = 0.0
            combined_df.to_excel(writer, sheet_name=species, index=False)
    print("Excel file saved successfully.")