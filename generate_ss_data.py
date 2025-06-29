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

def get_true_parameters(model: bionetgen.bngmodel, exclude_params: set) -> dict:
    """Extracts the default parameter values from the model, excluding condition parameters."""
    print("--- Extracting true kinetic parameters from BNGL model ---")
    true_params = {}
    # Iterate over parameter NAMES (which are strings)
    for param_name in model.parameters:
        if param_name not in exclude_params:
            # Access the parameter object from the model using its name
            param_obj = model.parameters[param_name]
            true_params[param_name] = float(param_obj.value)
    print(f"  Found {len(true_params)} kinetic parameters.")
    return true_params

def calculate_preeq_steadystate(model: bionetgen.bngmodel, true_params: dict, stimuli_to_zero: dict) -> np.ndarray:
    """
    Calculates the pre-equilibration steady-state using true kinetic parameters
    and zero stimulus, via a long simulation.
    """
    print("--- Calculating single pre-equilibration steady-state ---")
    simulator = model.setup_simulator()

    # Set all kinetic parameters to their "true" default values
    for name, value in true_params.items():
        simulator.model[name] = value
    
    # Set all stimuli parameters to 0 for pre-equilibration
    for sbml_id in stimuli_to_zero.values():
        simulator.model[sbml_id] = 0.0

    print("    1. Solving for steady-state via long simulation...")
    simulator.simulate(start=0, end=1e8, steps=2)
    
    ss_concentrations = simulator.model.getFloatingSpeciesConcentrations()
    print("    ...Correct pre-equilibrium state calculated and saved.")
    return ss_concentrations

def run_simulation_from_preeq(
    model: bionetgen.bngmodel, 
    ss_concentrations: np.ndarray,
    true_params: dict,
    stimuli: dict,
    sim_duration: float,
    sim_steps: int
) -> pd.DataFrame:
    """
    Runs a single time-course simulation starting from a pre-calculated
    steady-state.
    """
    simulator = model.setup_simulator()

    # 1. Set all model parameters to their true values
    for name, value in true_params.items():
        simulator.model[name] = value

    # 2. Set the initial concentrations to the provided steady-state vector
    simulator.model.setFloatingSpeciesConcentrations(ss_concentrations)

    # 3. Apply the specific stimuli for the current experimental condition
    for species_id, value in stimuli.items():
        simulator.model[species_id] = value
    
    # 4. Apply robust integrator settings
    simulator.integrator.stiff = True
    simulator.integrator.absolute_tolerance = 1e-8
    simulator.integrator.relative_tolerance = 1e-6
    simulator.integrator.maximum_num_steps = 50000
    
    # 5. Simulate the dynamic response
    print("    Simulating dynamic response...")
    result = simulator.simulate(start=0, end=sim_duration, steps=sim_steps)
    
    if result is None:
        raise RuntimeError("Simulation failed to produce results.")

    return pd.DataFrame(result, columns=result.colnames)


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

# --------------------------------------------------------------------------
#                   TIME-COURSE WORKFLOW WITH CORRECT SAVING
# --------------------------------------------------------------------------

def generate_time_course_excel(config):
    """
    Generates time-course data with a consistent pre-equilibration step
    and saves it to a single Excel file in "wide" format.
    """
    print(f"--- Running Time-Course Data Generation (Consistent Preeq) ---")
    
    # 1. Load settings and model
    tc_settings = config['time_course_settings']
    output_dir = config['output_dir']
    model_path = config['model_path']
    rng = np.random.default_rng(config['random_seed'])
    
    print(f"Loading BNGL model from: {model_path}")
    bng_model = bionetgen.bngmodel(model_path)
    os.makedirs(output_dir, exist_ok=True)

    # 2. Discover mappings and get true parameters
    all_stimuli_params = set(k for v in tc_settings['conditions'].values() for k in v.keys())
    param_to_sbml_id = discover_species_map(bng_model, list(all_stimuli_params))
    true_kinetic_params = get_true_parameters(bng_model, all_stimuli_params)

    # 3. Calculate the single, shared pre-equilibration steady state
    ss_concentrations = calculate_preeq_steadystate(bng_model, true_kinetic_params, param_to_sbml_id)

    # 4. Run simulation for each condition from the shared steady state
    time_course_results = {}
    for condition_name, stimuli_values in tc_settings['conditions'].items():
        print(f"\n--- Processing Condition: {condition_name} ---")
        stimuli_with_ids = {param_to_sbml_id[p]: v for p, v in stimuli_values.items()}
        
        df_sim = run_simulation_from_preeq(
            bng_model, 
            ss_concentrations,
            true_kinetic_params,
            stimuli_with_ids,
            tc_settings['simulation']['duration'],
            tc_settings['simulation']['steps']
        )
        time_course_results[condition_name] = df_sim

    # 5. Save the results to a single Excel file in the correct "wide" format
    noise_conf = tc_settings['noise']
    noise_str = f"_noise{int(noise_conf['level_percent'])}" if noise_conf['add'] else ""
    filename = os.path.join(output_dir, f"preeq{noise_str}.xlsx")
    
    print(f"\n--- Formatting and saving data to '{filename}' ---")
    
    # Get observable names from the model object
    all_observables = [obs.name for obs_key in bng_model.observables for obs in [bng_model.observables[obs_key]]]
    print(f"INFO: Found observables to save: {all_observables}")

    with pd.ExcelWriter(filename) as writer:
        for obs_name in sorted(all_observables):
            # Check if the observable column exists in the first simulation result
            if obs_name not in time_course_results[list(tc_settings['conditions'].keys())[0]].columns:
                print(f"WARNING: Observable '{obs_name}' not found in simulation output. Skipping.")
                continue

            # Create a new DataFrame for this observable's sheet
            sheet_df = pd.DataFrame()
            # Use the time column from the first condition's results
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