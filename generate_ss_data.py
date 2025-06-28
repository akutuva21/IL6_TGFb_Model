import pandas as pd
import numpy as np
import yaml
import os
import argparse
import bionetgen

# --------------------------------------------------------------------------
#                   COMMAND-LINE ARGUMENT PARSING
# --------------------------------------------------------------------------

def get_args():
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate or process data for PEtab.")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yml",
        help="Path to the YAML configuration file. Default: config.yml"
    )
    return parser.parse_args()

# --------------------------------------------------------------------------
#                   HELPER AND SIMULATION FUNCTIONS
# --------------------------------------------------------------------------

def discover_species_map(model: bionetgen.bngmodel, params_to_trace: list) -> dict:
    """Uses a tracer method to find the mapping from BNGL parameters to simulator species IDs."""
    print("--- Discovering species mapping with tracer method ---")
    tracer_map = {param: 9999.9 - i*1000 for i, param in enumerate(params_to_trace)}
    
    for param_name, tracer_val in tracer_map.items():
        model.parameters[param_name].value = tracer_val
    
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

def add_noise(data_series: pd.Series, noise_level: float, rng: np.random.Generator) -> pd.Series:
    """Adds multiplicative noise to a pandas Series."""
    noise = rng.normal(loc=0.0, scale=noise_level * np.abs(data_series))
    noisy_series = data_series + noise
    return noisy_series.clip(lower=0)

def run_simulation_for_condition(
    model: bionetgen.bngmodel, 
    stimuli: dict,
    sim_duration: float,
    sim_steps: int
) -> pd.DataFrame:
    """
    Runs the full pre-equilibration and main simulation for a single condition.
    Uses a long simulation to reliably find the steady state.
    """
    simulator = model.setup_simulator()

    print("    1. Solving for pre-equilibration steady-state via long simulation...")
    simulator.resetAll()
    for species_id in stimuli.keys():
        simulator.model[species_id] = 0.0
    
    simulator.simulate(start=0, end=1e8, steps=2)
    
    ss_concentrations = simulator.model.getFloatingSpeciesConcentrations()
    print("       ...Steady-state vector saved.")

    print("    2. Preparing main simulation...")
    simulator.resetAll()
    simulator.model.setFloatingSpeciesConcentrations(ss_concentrations)
    for species_id, value in stimuli.items():
        simulator.model[species_id] = value
    
    print("    3. Applying robust integrator settings...")
    simulator.integrator.stiff = True
    simulator.integrator.absolute_tolerance = 1e-8
    simulator.integrator.relative_tolerance = 1e-6
    simulator.integrator.maximum_num_steps = 50000
    
    print("    4. Simulating dynamic response...")
    result = simulator.simulate(start=0, end=sim_duration, steps=sim_steps)
    
    if result is None:
        raise RuntimeError("Simulation failed to produce results.")

    return pd.DataFrame(result, columns=result.colnames)

# --------------------------------------------------------------------------
#                   TIME-COURSE WORKFLOW WITH CORRECT SAVING
# --------------------------------------------------------------------------

def generate_time_course_excel(config):
    """
    Generates time-course data and saves it to a single Excel file with one
    sheet per observable, in the desired "wide" format.
    """
    print(f"--- Running Time-Course Data Generation ---")
    
    # 1. Load settings and model
    tc_settings = config['time_course_settings']
    output_dir = config['output_dir']
    model_path = config['model_path']
    rng = np.random.default_rng(config['random_seed'])
    
    print(f"Loading BNGL model from: {model_path}")
    bng_model = bionetgen.bngmodel(model_path)
    os.makedirs(output_dir, exist_ok=True)

    # 2. Discover mappings for all stimulus parameters
    all_stimuli_params = set(k for v in tc_settings['conditions'].values() for k in v.keys())
    param_to_sbml_id = discover_species_map(bng_model, list(all_stimuli_params))

    # 3. Run simulation for each condition and store results
    time_course_results = {}
    for condition_name, stimuli_values in tc_settings['conditions'].items():
        print(f"\n--- Processing Condition: {condition_name} ---")
        stimuli_with_ids = {param_to_sbml_id[p]: v for p, v in stimuli_values.items()}
        
        df_sim = run_simulation_for_condition(
            bng_model, 
            stimuli_with_ids,
            tc_settings['simulation']['duration'],
            tc_settings['simulation']['steps']
        )
        time_course_results[condition_name] = df_sim

    # 4. Save the results to a single Excel file in the correct "wide" format
    noise_conf = tc_settings['noise']
    noise_str = f"_noise{int(noise_conf['level_percent'])}" if noise_conf['add'] else ""
    filename = os.path.join(output_dir, f"preeq{noise_str}.xlsx")
    
    print(f"\n--- Formatting and saving data to '{filename}' ---")
    
    # Get observable names from the model object
    all_observables = [obs.name for obs_key in bng_model.observables for obs in [bng_model.observables[obs_key]]]
    print(f"INFO: Found observables to save: {all_observables}")

    with pd.ExcelWriter(filename) as writer:
        for obs_name in sorted(all_observables):
            if obs_name not in time_course_results[list(tc_settings['conditions'].keys())[0]].columns:
                print(f"WARNING: Observable '{obs_name}' not found in simulation output. Skipping.")
                continue

            # Create a new DataFrame for this observable's sheet
            sheet_df = pd.DataFrame()
            sheet_df['Time'] = time_course_results[list(tc_settings['conditions'].keys())[0]]['time']

            # Add a column for each condition
            for condition_name, result_df in time_course_results.items():
                data_col = result_df[obs_name]
                if noise_conf['add']:
                    noise_fraction = noise_conf['level_percent'] / 100.0
                    data_col = add_noise(data_col, noise_fraction, rng)
                sheet_df[condition_name] = data_col
            
            # Write this observable's DataFrame to a sheet in the Excel file
            sheet_df.to_excel(writer, sheet_name=obs_name, index=False)
            
    print(f"✅ Data saved successfully to {filename}")


# --------------------------------------------------------------------------
#                   MAIN EXECUTION LOGIC
# --------------------------------------------------------------------------

def main():
    """Main function to run the data generation/processing pipeline."""
    args = get_args()
    
    print(f"Loading configuration from: {args.config}")
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at '{args.config}'")
        return
    except yaml.YAMLError as e:
        print(f"ERROR: Could not parse YAML file: {e}")
        return

    run_mode = config.get("run_mode")
    print(f"Selected run mode: '{run_mode}'")

    if run_mode == "dose_response":
        # excel_to_petab_dose_response(config) # This function is still available if needed
        print("Dose-response mode selected, but no action is defined for it in this version.")
            
    elif run_mode == "time_course":
        generate_time_course_excel(config)
        
    else:
        print(f"ERROR: Unknown run_mode '{run_mode}'. Please choose 'dose_response' or 'time_course'.")

if __name__ == "__main__":
    main()