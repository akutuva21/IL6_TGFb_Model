import pandas as pd
import numpy as np
import yaml
import os
import argparse
import bionetgen
import logging
import sys

# --------------------------------------------------------------------------
#                    LOGGING CONFIGURATION
# --------------------------------------------------------------------------

def setup_logging(level=logging.INFO):
    """Setup logging configuration for cluster compatibility."""
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )
    # Force immediate flushing
    for handler in logging.getLogger().handlers:
        handler.setStream(sys.stdout)
        handler.flush = lambda: sys.stdout.flush()

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
    parser.add_argument('--debug', action='store_true', help='Enable debug logging.')
    return parser.parse_args()

# --------------------------------------------------------------------------
#                   HELPER AND SIMULATION FUNCTIONS
# --------------------------------------------------------------------------

def get_true_parameters(model: bionetgen.bngmodel, exclude_params: set) -> dict:
    """Extracts the default parameter values from the model, excluding condition parameters."""
    logging.info("--- Extracting true kinetic parameters from BNGL model ---")
    true_params = {}
    # Iterate over parameter NAMES (which are strings)
    for param_name in model.parameters:
        if param_name not in exclude_params:
            # Access the parameter object from the model using its name
            param_obj = model.parameters[param_name]
            true_params[param_name] = float(param_obj.value)
    logging.info(f"  Found {len(true_params)} kinetic parameters.")
    return true_params

def calculate_preeq_steadystate(
    model: bionetgen.bngmodel, 
    true_params: dict, 
    stimuli_to_zero: dict,      # This should be a dict of {param_name: value}
    constant_stimuli: dict,     # This should be a dict of {param_name: value}
    param_to_sbml_id: dict      # Pass in the map
) -> np.ndarray:
    """
    Calculates the pre-equilibration steady-state.
    Sets variable stimuli to zero but maintains constant background stimuli.
    """
    logging.info("--- Calculating single pre-equilibration steady-state ---")
    simulator = model.setup_simulator()

    # Set all kinetic parameters to their "true" default values
    for name, value in true_params.items():
        simulator.model[name] = value
    
    # Set variable stimuli parameters to 0 for pre-equilibration
    for param_name, value in stimuli_to_zero.items():
        sbml_id = param_to_sbml_id[param_name] # Use the map to get the ID
        simulator.model[sbml_id] = value
        logging.debug(f"    Setting variable stimulus {param_name} ({sbml_id}) to {value} for pre-equilibration.")

    # Set constant background stimuli to their defined values
    for param_name, value in constant_stimuli.items():
        sbml_id = param_to_sbml_id[param_name] # Use the map to get the ID
        simulator.model[sbml_id] = value
        logging.debug(f"    Setting constant stimulus {param_name} ({sbml_id}) to {value} for pre-equilibration.")

    logging.info("    1. Solving for steady-state via long simulation...")
    # It's better to use the model's own steady state finder if available
    try:
        simulator.ss()
        logging.info("    ...Steady-state calculated via simulator's ss() method.")
    except:
        logging.warning("    ss() method failed, falling back to long simulation...")
        simulator.simulate(start=0, end=1e8, steps=2)

    ss_concentrations = simulator.model.getFloatingSpeciesConcentrations()
    logging.info("    ...Correct pre-equilibrium state calculated and saved.")
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

    # --- START DIAGNOSTIC LOGGING ---
    # Find the simulator ID for IL6(r) to check its value. 
    # This is inefficient to do every time, but fine for a quick debug.
    il6_species_id = None
    for species_id in simulator.model.getFloatingSpeciesIds():
        if "IL6(r)" in species_id: # Find the species ID for IL6
            il6_species_id = species_id
            break
    
    if il6_species_id:
        logging.debug(f"    [DEBUG] IL-6 concentration AFTER loading pre-eq state: {simulator.model[il6_species_id]}")
    # --- END DIAGNOSTIC LOGGING ---

    # 3. Apply the specific stimuli for the current experimental condition
    for species_id, value in stimuli.items():
        simulator.model[species_id] = value
    
    # --- START DIAGNOSTIC LOGGING ---
    if il6_species_id:
        logging.debug(f"    [DEBUG] IL-6 concentration AFTER applying stimulus: {simulator.model[il6_species_id]}")
    # --- END DIAGNOSTIC LOGGING ---

    # 4. Apply robust integrator settings
    simulator.integrator.stiff = True
    simulator.integrator.absolute_tolerance = 1e-8
    simulator.integrator.relative_tolerance = 1e-6
    simulator.integrator.maximum_num_steps = 50000
    
    # 5. Simulate the dynamic response
    logging.info("    Simulating dynamic response...")
    result = simulator.simulate(start=0, end=sim_duration, steps=sim_steps)
    
    if result is None:
        raise RuntimeError("Simulation failed to produce results.")

    return pd.DataFrame(result, columns=result.colnames)


def discover_species_map(model: bionetgen.bngmodel, params_to_trace: list) -> dict:
    """Uses a tracer method to find the mapping from BNGL parameters to simulator species IDs."""
    logging.info("--- Discovering species mapping with tracer method ---")
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
                logging.info(f"  SUCCESS: Traced parameter '{param_name}' to simulator ID '{sbml_id}'")
                found = True
                break
        if not found:
            raise RuntimeError(f"FATAL ERROR: Could not find tracer for '{param_name}'.")
    logging.info("-----------------------------------------------------")
    return param_to_sbml_id

def add_noise(data_series: pd.Series, noise_level: float, rng: np.random.Generator) -> pd.Series:
    """
    Adds unbiased log-normal multiplicative noise with constant CV = noise_level.
    noise_level is the fractional CV (e.g., 0.05 for 5%).
    """
    if noise_level <= 0:
        return data_series.copy()

    sigma = float(np.sqrt(np.log(1.0 + noise_level**2)))
    # Draw one factor per element
    eps = np.exp(rng.normal(loc=-0.5 * sigma**2, scale=sigma, size=len(data_series)))
    noisy_series = data_series.to_numpy(dtype=float) * eps
    # Log-normal factor is > 0, so no need to clip; keep clip for robustness if desired
    # noisy_series = np.clip(noisy_series, a_min=0.0, a_max=None)
    return pd.Series(noisy_series, index=data_series.index)

# --------------------------------------------------------------------------
#                   TIME-COURSE WORKFLOW WITH PEtab TSV OUTPUT
# --------------------------------------------------------------------------

def generate_time_course_petab(config):
    """
    Generates time-course data with consistent pre-equilibration,
    handles zero values with a Limit of Detection (LOD), adds noise,
    and saves it in PEtab-standard TSV format.
    """
    logging.info(f"--- Running Time-Course Data Generation (PEtab TSV Format) ---")
    
    # 1. Load settings and model
    tc_settings = config['time_course_settings']
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    model_path = config['model_path']
    model = bionetgen.bngmodel(model_path)
    logging.info(f"Loading BNGL model from: {model_path}")
    
    # 2. Extract parameters and discover species map
    variable_stimuli = set(tc_settings.get('variable_stimuli', []))
    constant_stimuli_names = set(tc_settings.get('constant_stimuli', []))
    condition_params = variable_stimuli.union(constant_stimuli_names)
    true_params = get_true_parameters(model, condition_params)
    temp_model_for_tracing = bionetgen.bngmodel(model_path)
    param_to_sbml_id = discover_species_map(temp_model_for_tracing, list(condition_params))
    
    # 3. Calculate pre-equilibration steady-state
    stimuli_to_zero = {param: 0.0 for param in variable_stimuli}
    baseline_condition = tc_settings['conditions']['TREG']
    constant_stimuli = {param: baseline_condition[param] for param in constant_stimuli_names if param in baseline_condition}
    preeq_ss = calculate_preeq_steadystate(model, true_params, stimuli_to_zero, constant_stimuli, param_to_sbml_id)
    logging.info(f"  Pre-equilibration steady-state calculated.")
    
    # 4. Run simulations and generate "perfect" data
    time_course_results = {}
    for condition_name, condition_values in tc_settings['conditions'].items():
        logging.info(f"  Simulating condition: {condition_name}")
        stimuli_with_ids = {param_to_sbml_id[p]: v for p, v in condition_values.items()}
        result_df = run_simulation_from_preeq(
            model, preeq_ss, true_params, stimuli_with_ids, 
            tc_settings['simulation']['duration'], tc_settings['simulation']['steps']
        )
        time_course_results[condition_name] = result_df
        
    # 5. Convert to long format DataFrame (still noise-free)
    measurement_rows = []
    all_observables = [obs_name for obs_name in model.observables]
    for condition_name, result_df in time_course_results.items():
        for _, row in result_df.iterrows():
            time_val = row['time']
            for obs_name in all_observables:
                if obs_name in row:
                    measurement_rows.append({
                        'observableId': obs_name,
                        'simulationConditionId': condition_name,
                        'time': time_val,
                        'measurement': row[obs_name],
                        'preequilibrationConditionId': 'preeq_ss'
                    })
    
    measurement_df = pd.DataFrame(measurement_rows)

    # 6. Apply Limit of Detection (LOD) and Noise
    noise_conf = tc_settings['noise']
    if noise_conf['add']:
        logging.info("  Applying Limit of Detection (LOD) to zero-valued measurements...")
        lod_map = {}
        for obs_id in measurement_df['observableId'].unique():
            non_zero_vals = measurement_df.loc[
                (measurement_df['observableId'] == obs_id) & (measurement_df['measurement'] > 0), 
                'measurement'
            ]
            if not non_zero_vals.empty:
                # Heuristic: LOD is half the smallest positive measurement
                lod = 0.5 * non_zero_vals.min()
                lod_map[obs_id] = max(lod, 1e-12) # Add a floor value
            else:
                # Fallback for observables that are always zero
                lod_map[obs_id] = 1e-12

        def apply_lod(row):
            if row['measurement'] <= 0:
                return lod_map[row['observableId']]
            return row['measurement']
        
        measurement_df['measurement'] = measurement_df.apply(apply_lod, axis=1)
        logging.info("  ...LOD applied successfully.")
        
        logging.info(f"  Adding {noise_conf['level_percent']}% lognormal noise...")
        seed = tc_settings.get('random_seed', 42)
        rng = np.random.default_rng(seed)
        noise_fraction = noise_conf['level_percent'] / 100.0
        
        # Apply noise to the entire measurement column at once
        measurement_df['measurement'] = add_noise(measurement_df['measurement'], noise_fraction, rng)
        logging.info("  ...Noise added successfully.")

    # 7. Create and save PEtab files
    condition_rows = []
    for condition_name, condition_values in tc_settings['conditions'].items():
        condition_row = {'conditionId': condition_name}
        condition_row.update(condition_values)
        condition_rows.append(condition_row)
    
    baseline_cond = tc_settings['conditions'].get('TREG', {})
    preeq_row = {'conditionId': 'preeq_ss'}
    preeq_row.update(baseline_cond)
    condition_rows.append(preeq_row)
    condition_df = pd.DataFrame(condition_rows)
    
    filename_suffix = f"_noise{int(noise_conf['level_percent'])}" if noise_conf['add'] else "_no_noise"
    measurement_path = os.path.join(output_dir, f"measurements_time_course{filename_suffix}.tsv")
    condition_path = os.path.join(output_dir, f"conditions_time_course{filename_suffix}.tsv")
    
    measurement_df.to_csv(measurement_path, index=False, sep='\t', float_format='%.8g')
    condition_df.to_csv(condition_path, index=False, sep='\t')
    
    logging.info(f"✅ PEtab time-course files created successfully:")
    logging.info(f"   - Measurements: {measurement_path}")
    logging.info(f"   - Conditions:   {condition_path}")
    
    return measurement_df, condition_df, model

# --------------------------------------------------------------------------
#                   PEtab FILE CREATION
# --------------------------------------------------------------------------

def create_observables_petab(config, measurements_df, observables_mapping, output_path):
    """Creates a PEtab observables file with a shared noise parameter."""
    logging.info("Creating observables PEtab file (log-normal with shared sigma)...")

    noise_conf = config['time_course_settings']['noise']
    is_noisy = noise_conf['add'] and noise_conf['level_percent'] > 0

    observable_ids = measurements_df['observableId'].unique()
    observables_data = []

    for obs_id in observable_ids:
        # Use one shared sigma for all observables for simplicity and robustness
        noise_formula = "sigma_log_shared" if is_noisy else "1e-8"
        noise_dist = "logNormal"

        observables_data.append({
            'observableId': obs_id,
            'observableName': observables_mapping.get(obs_id, obs_id),
            'observableFormula': obs_id,
            'noiseFormula': noise_formula,
            'noiseDistribution': noise_dist
        })

    observables_df = pd.DataFrame(observables_data)
    observables_df.to_csv(output_path, sep='\t', index=False)
    logging.info(f"✅ Observables file saved: {output_path}")


def create_parameters_petab(config, model, output_path):
    """
    Creates a parameters.tsv file with raw (linear) values, which is the
    format expected by PEtab.jl and other common toolboxes.
    """
    logging.info("Creating parameters PEtab file with raw (linear) values...")

    parameters_data = []
    stimulus_params = set(config['time_course_settings']['variable_stimuli']) | set(config['time_course_settings']['constant_stimuli'])

    for param_name in model.parameters:
        # Get the parameter OBJECT using the name as a key
        param_obj = model.parameters[param_name]
        # Access the .value attribute from the OBJECT
        nominal_value = float(param_obj.value)
        
        # Default bounds for kinetic rates
        lower_bound = nominal_value / 2.0
        upper_bound = nominal_value * 2.0
        
        if param_name in stimulus_params or param_name.endswith("_0"):
            estimate = 0
            parameter_scale = 'lin'
            # For fixed params, bounds are just the nominal value
            lower_bound = nominal_value
            upper_bound = nominal_value
        else:
            estimate = 1
            parameter_scale = 'log10'

        parameters_data.append({
            'parameterId': param_name,
            'parameterName': param_name,
            'parameterScale': parameter_scale,
            'lowerBound': lower_bound,
            'upperBound': upper_bound,
            'nominalValue': nominal_value,
            'estimate': estimate
        })

    # Add the shared noise parameter
    noise_conf = config['time_course_settings']['noise']
    is_noisy = noise_conf.get('add', False) and noise_conf.get('level_percent', 0) > 0
    
    if is_noisy:
        cv = noise_conf['level_percent'] / 100.0
        # Nominal value for sigma on linear scale, derived from CV
        sigma_nominal = float(np.sqrt(np.log(1.0 + cv**2)))
        parameters_data.append({
            'parameterId': 'sigma_log_shared', 'parameterName': 'sigma_log_shared',
            'parameterScale': 'log10', # The optimizer still sees this on a log scale
            'lowerBound': 1e-2,       # Linear bound
            'upperBound': 0.25,        # Linear bound
            'nominalValue': sigma_nominal, # Linear nominal value
            'estimate': 0
        })
    else:
        # Add a fixed noise parameter if data is noise-free
        parameters_data.append({
            'parameterId': 'sigma_log_shared', 'parameterName': 'sigma_log_shared',
            'parameterScale': 'log10', 'lowerBound': 1e-8, 'upperBound': 1e-4,
            'nominalValue': 1e-8, 'estimate': 0
        })

    pd.DataFrame(parameters_data).to_csv(output_path, sep='\t', index=False, float_format='%.16g')
    logging.info(f"✅ Parameters file saved: {output_path}")

# --------------------------------------------------------------------------
#                   MAIN EXECUTION BLOCK
# --------------------------------------------------------------------------

def main():
    """Main function to generate steady-state data and PEtab files."""
    args = get_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate measurements and conditions
    measurements_df, condition_df, model = generate_time_course_petab(config)
    
    # Create PEtab files
    output_dir = "petab_files"
    os.makedirs(output_dir, exist_ok=True)
    
    # Add noise suffix to filenames if noise is enabled
    noise_conf = config['time_course_settings']['noise']
    if noise_conf['add'] and noise_conf['level_percent'] > 0:
        noise_suffix = f"_noise{int(noise_conf['level_percent'])}"
    else:
        noise_suffix = ""
    
    observables_path = os.path.join(output_dir, f"observables{noise_suffix}.tsv")
    parameters_path = os.path.join(output_dir, f"parameters{noise_suffix}.tsv")
    
    create_observables_petab(
        config,
        measurements_df, 
        config['observables_mapping'], 
        observables_path
    )
    
    create_parameters_petab(
        config,
        model,
        parameters_path
    )
    
    logging.info("🎉 All PEtab files generated successfully!")
    sys.stdout.flush()


if __name__ == "__main__":
    main()