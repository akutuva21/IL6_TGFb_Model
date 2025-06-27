import bionetgen
import pandas as pd
import numpy as np
import os

# --- Main Configuration ---
MODEL_PATH = "model_even_smaller.bngl"
OUTPUT_FILENAME_BASE = "SimData/preeq" # Base name for the output file
MAIN_SIM_TIME = 100.0
MAIN_SIM_STEPS = 1000

# --- Workflow Control Flags ---
# Set these flags to control the script's behavior
ADD_NOISE = False            # Master switch for adding any noise
NOISE_LEVEL_PERCENT = 5.0  # The percentage of noise to add (e.g., 5.0, 10.0)

# A random seed ensures that the "random" noise is the same every time you run the script,
# which is crucial for reproducibility.
RANDOM_SEED = 42

# --- Helper Functions ---

def discover_species_map(model: bionetgen.bngmodel, tracer_map: dict) -> dict:
    """Uses the tracer method to find the mapping from BNGL parameters to simulator species IDs."""
    print("--- Discovering species mapping with tracer method ---")
    for param_name, tracer_val in tracer_map.items():
        model.parameters[param_name].value = tracer_val
    
    # Compile the model with the tracer values
    simulator = model.setup_simulator()
    
    all_species_ids = simulator.model.getFloatingSpeciesIds()
    all_initial_concs = simulator.model.getFloatingSpeciesConcentrations()

    param_to_sbml_id = {}
    for param_name, tracer_val in tracer_map.items():
        found = False
        for i, conc in enumerate(all_initial_concs):
            if abs(conc - tracer_val) < 1e-6:
                sbml_id = all_species_ids[i]
                param_to_sbml_id[param_name] = sbml_id
                print(f"  SUCCESS: Traced parameter '{param_name}' to simulator ID '{sbml_id}'")
                found = True
                break
        if not found:
            raise RuntimeError(f"FATAL ERROR: Could not find tracer for '{param_name}'.")
            
    print("-----------------------------------------------------")
    return param_to_sbml_id


def run_simulation_for_condition(model: bionetgen.bngmodel, condition_name: str, stimuli: dict) -> pd.DataFrame:
    """Runs the full pre-equilibration and main simulation for a single condition."""
    print(f"--- Running workflow for: {condition_name} ---")
    
    simulator = model.setup_simulator()

    # 1. Solve for pre-equilibration steady-state via a long simulation
    print("    1. Solving for pre-equilibration steady-state...")
    simulator.resetAll()
    # For pre-equilibration, set all stimuli to 0
    for species_id in stimuli.keys():
        simulator.model[species_id] = 0.0
    simulator.simulate(start=0, end=1e7, steps=2)
    ss_concentrations = simulator.model.getFloatingSpeciesConcentrations()
    print("       ...Steady-state vector saved.")
    
    # 2. Prepare and run the main simulation from the steady-state
    print("    2. Preparing main simulation...")
    simulator.resetAll()
    simulator.model.setFloatingSpeciesConcentrations(ss_concentrations)
    # Apply the actual stimuli for the condition
    for species_id, value in stimuli.items():
        simulator.model[species_id] = value
    
    # 3. Apply robust integrator settings
    print("    3. Applying robust integrator settings...")
    simulator.integrator.stiff = True
    simulator.integrator.absolute_tolerance = 1e-8
    simulator.integrator.relative_tolerance = 1e-6
    simulator.integrator.maximum_num_steps = 50000
    
    # 4. Simulate and return the results
    print("    4. Simulating dynamic response...")
    result = simulator.simulate(start=0, end=MAIN_SIM_TIME, steps=MAIN_SIM_STEPS)
    print(f"✅ Workflow for '{condition_name}' complete.")
    return pd.DataFrame(result, columns=result.colnames)


def add_multiplicative_noise(data_df: pd.DataFrame, noise_level: float, rng: np.random.Generator) -> pd.DataFrame:
    """Adds multiplicative noise to all data columns except 'Time'."""
    noisy_df = data_df.copy()
    data_cols = [col for col in noisy_df.columns if col != 'Time']
    
    for col in data_cols:
        # Noise is proportional to the signal's magnitude
        noise = rng.normal(loc=0.0, scale=noise_level * np.abs(noisy_df[col]))
        noisy_df[col] += noise
        # Ensure data doesn't become negative after adding noise
        noisy_df[col] = noisy_df[col].clip(lower=0)
        
    return noisy_df


def save_to_excel(time_course_results: dict, model: bionetgen.bngmodel, filename: str):
    """Saves the collected simulation results into a PEtab-formatted Excel file."""
    print(f"\n--- Formatting and saving data to '{filename}' ---")
    
    try:
        # Correctly get observable names from the model object
        all_observables = [obs.name for obs_key in model.observables for obs in [model.observables[obs_key]]]
        if not all_observables:
            print("WARNING: No observables found in the BNGL model. File will be empty.")
            return
        print(f"INFO: Found the following observables to save: {all_observables}")
    except Exception as e:
        print(f"ERROR: Could not get observable names from the model object. Error: {e}")
        return

    with pd.ExcelWriter(filename) as writer:
        for obs_name in sorted(all_observables):
            # Check if the observable is present in the simulation output
            if obs_name not in time_course_results["Treg"].columns:
                print(f"WARNING: Observable '{obs_name}' was defined but not found in output. Skipping.")
                continue
            
            # Create a DataFrame for the current observable's sheet
            sheet_df = pd.DataFrame({'Time': time_course_results["Treg"]['time']})
            for condition_name, result_df in time_course_results.items():
                sheet_df[condition_name] = result_df[obs_name]
            
            sheet_df.to_excel(writer, sheet_name=obs_name, index=False)
            
    print(f"✅ Data saved successfully to {filename}")


# --- Main Workflow ---

def main():
    """Main function to run the entire data generation workflow."""
    if not os.path.exists("SimData"):
        os.makedirs("SimData")
        
    # Set up the random number generator for reproducibility
    rng = np.random.default_rng(RANDOM_SEED)

    # Load the model just once
    main_model = bionetgen.bngmodel(MODEL_PATH)
    
    # 1. Discover the mapping from BNGL parameters to simulator IDs
    tracer_map = {'IL6_0': 9999.9, 'TGFb_0': 8888.8}
    param_to_sbml_id = discover_species_map(main_model, tracer_map)
    il6_id = param_to_sbml_id['IL6_0']
    tgfb_id = param_to_sbml_id['TGFb_0']

    # 2. Define experimental conditions using the discovered IDs
    main_conditions = {
        "Treg": {il6_id: 0.0, tgfb_id: 1.0},
        "TH17": {il6_id: 100.0, tgfb_id: 1.0}
    }
    
    # 3. Run simulations for all conditions
    time_course_results = {}
    for condition_name, stimuli in main_conditions.items():
        result_df = run_simulation_for_condition(main_model, condition_name, stimuli)
        
        # 4. (Optional) Add noise to the results
        if ADD_NOISE:
            print(f"    -> Adding {NOISE_LEVEL_PERCENT}% multiplicative noise.")
            noise_fraction = NOISE_LEVEL_PERCENT / 100.0
            result_df = add_multiplicative_noise(result_df, noise_fraction, rng)

        time_course_results[condition_name] = result_df

    # 5. Save the final results to an Excel file
    # Construct the final output filename based on the flags
    noise_str = f"_noise{int(NOISE_LEVEL_PERCENT)}" if ADD_NOISE else ""
    final_filename = f"{OUTPUT_FILENAME_BASE}{noise_str}.xlsx"
    save_to_excel(time_course_results, main_model, final_filename)

if __name__ == "__main__":
    main()