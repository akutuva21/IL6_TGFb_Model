import pandas as pd
import numpy as np
import yaml
import os
import argparse
import bionetgen
import logging
import sys

# --------------------------------------------------------------------------
#                   LOGGING CONFIGURATION
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

    # 3. Apply the specific stimuli for the current experimental condition
    for species_id, value in stimuli.items():
        simulator.model[species_id] = value
    
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
#                   TIME-COURSE WORKFLOW WITH CORRECT SAVING
# --------------------------------------------------------------------------

def generate_time_course_excel(config):
    """
    Generates time-course data with a consistent pre-equilibration step
    and saves it to a single Excel file in "wide" format.
    """
    logging.info(f"--- Running Time-Course Data Generation (Consistent Preeq) ---")
    
    # 1. Load settings and model
    tc_settings = config['time_course_settings']
    output_dir = config['output_dir']
    model_path = config['model_path']
    rng = np.random.default_rng(config['random_seed'])
    
    logging.info(f"Loading BNGL model from: {model_path}")
    bng_model = bionetgen.bngmodel(model_path)
    os.makedirs(output_dir, exist_ok=True)

    # 2. Discover mappings and get true parameters
    variable_stimuli = set(tc_settings.get('variable_stimuli', []))
    constant_stimuli_names = set(tc_settings.get('constant_stimuli', []))
    all_stimuli_params = variable_stimuli.union(constant_stimuli_names)

    # Use a temporary model instance for species mapping discovery to avoid contaminating the main model
    logging.info("  Creating temporary model instance for species mapping discovery...")
    temp_model_for_tracing = bionetgen.bngmodel(model_path)
    param_to_sbml_id = discover_species_map(temp_model_for_tracing, list(all_stimuli_params))
    # After this, temp_model_for_tracing can be discarded. The main 'bng_model' object is still clean.
    
    true_kinetic_params = get_true_parameters(bng_model, all_stimuli_params)

    # 3. Calculate the single, shared pre-equilibration steady state
    # Separate the stimuli into those that should be zero vs constant
    stimuli_to_zero_map = {p: i for p, i in param_to_sbml_id.items() if p in variable_stimuli}
    
    # Get the actual values for the constant stimuli from the TREG condition (or any baseline)
    baseline_condition = tc_settings['conditions']['TREG']
    constant_stimuli_map = {
        param_to_sbml_id[p]: baseline_condition[p] 
        for p in constant_stimuli_names if p in baseline_condition
    }

    ss_concentrations = calculate_preeq_steadystate(
        bng_model, true_kinetic_params, stimuli_to_zero_map, constant_stimuli_map, param_to_sbml_id
    )

    # 4. Run simulation for each condition from the shared steady state
    time_course_results = {}
    for condition_name, stimuli_values in tc_settings['conditions'].items():
        logging.info(f"\n--- Processing Condition: {condition_name} ---")
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
    
    logging.info(f"\n--- Formatting and saving data to '{filename}' ---")
    
    # Get observable names from the model object
    all_observables = [obs.name for obs_key in bng_model.observables for obs in [bng_model.observables[obs_key]]]
    logging.info(f"INFO: Found observables to save: {all_observables}")

    with pd.ExcelWriter(filename) as writer:
        for obs_name in sorted(all_observables):
            # Check if the observable column exists in the first simulation result
            if obs_name not in time_course_results[list(tc_settings['conditions'].keys())[0]].columns:
                logging.warning(f"WARNING: Observable '{obs_name}' not found in simulation output. Skipping.")
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
            
    logging.info(f"✅ Data saved successfully to {filename}")
    sys.stdout.flush()


# --------------------------------------------------------------------------
#                   DOSE-RESPONSE WORKFLOW
# --------------------------------------------------------------------------

def excel_to_petab_dose_response(config):
    """
    Reads a dose-response Excel file, converts it to PEtab format,
    and saves the measurement and condition files.
    """
    logging.info("--- Running Dose-Response Data Processing ---")
    
    # 1. Load settings
    dr_settings = config['dose_response_settings']
    input_conf = dr_settings['input_data']
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    if not input_conf['load_from_file']:
        logging.info("This script is configured to process an existing file. Skipping.")
        return

    # 2. Read the Excel data
    filepath = input_conf['filepath']
    dose_col = input_conf['dose_column_name']
    col_map = input_conf['column_to_observable_map']
    
    logging.info(f"  Reading data from '{filepath}'...")
    try:
        df_wide = pd.read_excel(filepath)
    except FileNotFoundError:
        logging.error(f"ERROR: Data file not found at '{filepath}'")
        return

    # 3. Convert from wide to long format (PEtab measurements table)
    logging.info("  Converting data to PEtab long format...")
    
    # Melt the DataFrame to turn it into a long format
    df_long = df_wide.melt(
        id_vars=[dose_col],
        var_name="measurement_col",
        value_name="measurement"
    )
    
    # Map the original measurement columns to PEtab observableIds
    df_long['observableId'] = df_long['measurement_col'].map(col_map)
    
    # Drop rows where the mapping didn't exist (e.g., columns not in the map)
    df_long.dropna(subset=['observableId'], inplace=True)

    # 4. Create PEtab DataFrames
    
    # --- Measurement DataFrame ---
    measurement_df = pd.DataFrame()
    measurement_df['observableId'] = df_long['observableId']
    
    # Create a unique simulation condition for each dose level
    measurement_df['simulationConditionId'] = [f"dose_{d}" for d in df_long[dose_col]]
    
    # For steady-state, time is infinite
    measurement_df['time'] = np.inf
    measurement_df['measurement'] = df_long['measurement']
    
    # Add placeholder for preequilibration (can be defined later if needed)
    measurement_df['preequilibrationConditionId'] = 'preeq_ss'
    
    # --- Condition DataFrame ---
    dose_parameter = dr_settings['dose_parameter']
    constant_params = dr_settings['constant_parameters']
    
    unique_doses = df_wide[dose_col].unique()
    condition_ids = [f"dose_{d}" for d in unique_doses]
    
    condition_df = pd.DataFrame({
        'conditionId': condition_ids,
        dose_parameter: unique_doses
    })
    
    # Add any constant parameters
    for param, value in constant_params.items():
        condition_df[param] = value
        
    # Add the preequilibration condition
    preeq_cond = {'conditionId': 'preeq_ss', dose_parameter: 0.0}
    for param, value in constant_params.items():
        preeq_cond[param] = value
    condition_df = pd.concat([condition_df, pd.DataFrame([preeq_cond])], ignore_index=True)

    # 5. Save to CSV
    measurement_path = os.path.join(output_dir, "measurements_dose_response.tsv")
    condition_path = os.path.join(output_dir, "conditions_dose_response.tsv")
    
    measurement_df.to_csv(measurement_path, index=False, sep='\t')
    condition_df.to_csv(condition_path, index=False, sep='\t')
    
    logging.info(f"✅ PEtab files created successfully:")
    logging.info(f"   - Measurements: {measurement_path}")
    logging.info(f"   - Conditions:   {condition_path}")
    sys.stdout.flush()


# --------------------------------------------------------------------------
#                   TIME-COURSE WORKFLOW WITH PEtab TSV OUTPUT
# --------------------------------------------------------------------------

def generate_time_course_petab(config):
    """
    Generates time-course data with consistent pre-equilibration step
    and saves it in PEtab-standard TSV format (long format).
    This is the recommended method for PEtab compliance.
    """
    logging.info(f"--- Running Time-Course Data Generation (PEtab TSV Format) ---")
    
    # 1. Load settings and model
    tc_settings = config['time_course_settings']
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = config['model_path']
    model = bionetgen.bngmodel(model_path)

    # indicate what file you are loading
    logging.info(f"Loading BNGL model from: {model_path}")
    
    # 2. Extract parameters
    conditions_list = tc_settings['conditions'].keys()
    logging.info(f"  Identified conditions: {list(conditions_list)}")
    
    # Get stimulus parameters from the variable_stimuli and constant_stimuli lists
    variable_stimuli = set(tc_settings.get('variable_stimuli', []))
    constant_stimuli_names = set(tc_settings.get('constant_stimuli', []))
    condition_params = variable_stimuli.union(constant_stimuli_names)
    logging.info(f"  Identified stimulus (condition) parameters: {list(condition_params)}")
    
    true_params = get_true_parameters(model, condition_params)
    
    # 5. Discover the species map using a temporary model instance.
    #    This prevents the main model object from being modified with tracer values.
    logging.info("  Creating temporary model instance for species mapping discovery...")
    temp_model_for_tracing = bionetgen.bngmodel(model_path)
    param_to_sbml_id = discover_species_map(temp_model_for_tracing, list(condition_params))
    # After this, temp_model_for_tracing can be discarded. The main 'model' object is still clean.
    
    # 3. Get pre-equilibration steady-state
    variable_stimuli = set(tc_settings.get('variable_stimuli', []))
    constant_stimuli_names = set(tc_settings.get('constant_stimuli', []))
    
    stimuli_to_zero = {}
    for param in variable_stimuli:
        stimuli_to_zero[param] = 0.0
    
    # Get constant stimuli values from baseline condition
    baseline_condition = tc_settings['conditions']['TREG']
    constant_stimuli = {}
    for param in constant_stimuli_names:
        if param in baseline_condition:
            constant_stimuli[param] = baseline_condition[param]
    
    # Now this call will work because param_to_sbml_id is defined.
    preeq_ss = calculate_preeq_steadystate(model, true_params, stimuli_to_zero, constant_stimuli, param_to_sbml_id)
    logging.info(f"  Pre-equilibration steady-state calculated.")
    
    # 4. Simulation settings
    sim_confs = tc_settings['simulation']
    t_end = sim_confs['duration']
    n_points = sim_confs['steps']
    
    # 6. Run simulations for each condition
    time_course_results = {}
    
    for condition_name, condition_values in tc_settings['conditions'].items():
        logging.info(f"  Simulating condition: {condition_name}")
        
        # Create the dictionary with the correct simulator IDs
        stimuli_with_ids = {param_to_sbml_id[p]: v for p, v in condition_values.items()}
        
        result_df = run_simulation_from_preeq(
            model, preeq_ss, true_params, stimuli_with_ids, t_end, n_points
        )
        time_course_results[condition_name] = result_df
    
    # 7. Convert to PEtab long format
    logging.info("  Converting data to PEtab long format...")
    
    # Create noise generator
    noise_conf = tc_settings['noise']
    seed = tc_settings.get('random_seed', 42)
    rng = np.random.default_rng(seed)
    # Precompute sigma on natural log scale for a constant coefficient of variation
    # sigma_logn = sqrt(log(1 + CV^2))
    noise_fraction = noise_conf['level_percent'] / 100.0
    sigma_logn = (
        float(np.sqrt(np.log(1.0 + noise_fraction**2)))
        if noise_conf['add'] else 0.0
    )
    
    # Prepare measurement and condition DataFrames
    measurement_rows = []
    condition_rows = []
    
    # Get observable names from the model object
    all_observables = [obs.name for obs_key in model.observables for obs in [model.observables[obs_key]]]
    logging.info(f"INFO: Found observables to save: {all_observables}")

    # NOTE: We don't add preeq_ss to the main conditions table because it's only used
    # for pre-equilibration and doesn't have corresponding measurements. PEtab will
    # handle pre-equilibration internally using the preequilibrationConditionId column.
    
    # Process each condition
    for condition_name, condition_values in tc_settings['conditions'].items():
        result_df = time_course_results[condition_name]
        
        # Add condition to conditions table - use the actual stimulus values
        condition_row = {'conditionId': condition_name}
        condition_row.update(condition_values)  # This contains IL6_0, TGFb_0, etc.
        condition_rows.append(condition_row)
        
        # Add measurements for each observable and time point
        for _, row in result_df.iterrows():
            time_val = row['time']
            for obs_name in all_observables:
                if obs_name in row:
                    measurement_val = row[obs_name]
                    
                    # Add noise if configured: log-normal (natural log) for constant CV
                    if noise_conf['add']:
                        # Unbiased multiplicative log-normal noise: E[exp(N(-σ²/2, σ²))] = 1
                        eps = np.exp(rng.normal(loc=-0.5 * sigma_logn**2, scale=sigma_logn))
                        measurement_val = measurement_val * eps
                    
                    measurement_rows.append({
                        'observableId': obs_name,
                        'simulationConditionId': condition_name,
                        'time': time_val,
                        'measurement': measurement_val,
                        'preequilibrationConditionId': 'preeq_ss'
                    })
    
    # 8. Create DataFrames and save
    measurement_df = pd.DataFrame(measurement_rows)

    noise_conf = tc_settings['noise']
    is_noisy = noise_conf['add'] and noise_conf['level_percent'] > 0

    if is_noisy:
        lod_map = {}
        for obs_id in measurement_df['observableId'].unique():
            pos_vals = measurement_df.loc[
                (measurement_df['observableId'] == obs_id) & (measurement_df['measurement'] > 0.0),
                'measurement'
            ].to_numpy()
            if len(pos_vals) > 0:
                lod = float(0.5 * np.min(pos_vals))  # conservative LOD below smallest positive
            else:
                lod = 1e-12  # fallback if all zero
            lod_map[obs_id] = lod

        # Replace zeros and negatives with the LOD for that observable
        def _apply_lod(row):
            return lod_map[row['observableId']] if row['measurement'] <= 0.0 else row['measurement']

        mask = measurement_df['measurement'] <= 0.0
        if mask.any():
            measurement_df.loc[mask, 'measurement'] = measurement_df.loc[mask].apply(_apply_lod, axis=1)
    
    # Ensure the 'preeq_ss' condition is included in the conditions file
    # Use the baseline (TREG) stimuli values for pre-equilibration
    baseline_cond = tc_settings['conditions'].get('TREG', {})
    preeq_row = {'conditionId': 'preeq_ss'}
    preeq_row.update(baseline_cond)
    condition_rows.append(preeq_row)

    condition_df = pd.DataFrame(condition_rows)
    
    # Create filename suffix based on noise configuration
    if noise_conf['add']:
        noise_percent = int(noise_conf['level_percent'])
        filename_suffix = f"_noise{noise_percent}"
    else:
        filename_suffix = "_no_noise"
        
    measurement_path = os.path.join(output_dir, f"measurements_time_course{filename_suffix}.tsv")
    condition_path = os.path.join(output_dir, f"conditions_time_course{filename_suffix}.tsv")
    
    measurement_df.to_csv(measurement_path, index=False, sep='\t')
    condition_df.to_csv(condition_path, index=False, sep='\t')
    
    logging.info(f"✅ PEtab time-course files created successfully:")
    logging.info(f"   - Measurements: {measurement_path}")
    logging.info(f"   - Conditions:   {condition_path}")
    sys.stdout.flush()
    
    return measurement_df, condition_df, model


# --------------------------------------------------------------------------
#                   PEtab FILE CREATION
# --------------------------------------------------------------------------

def create_observables_petab(config, measurements_df, observables_mapping, output_path):
    logging.info("Creating observables PEtab file (log-normal with shared sigma)...")

    noise_conf = config['time_course_settings']['noise']
    is_noisy = noise_conf['add'] and noise_conf['level_percent'] > 0

    observable_ids = measurements_df['observableId'].unique()
    observables_data = []

    for obs_id in observable_ids:
        # Use one shared sigma for all observables
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

def create_parameters_petab(config, model, measurements_df, output_path):
    """Create parameters.tsv with a single shared sigma_log for log-normal noise."""
    logging.info("Creating parameters PEtab file (log-normal, shared sigma)...")

    parameters_data = []

    # Define which parameters should NOT be estimated (they are controlled by the conditions file)
    stimulus_params = {'IL6_0', 'TGFb_0'}

    # Define custom bounds for initial concentration parameters if needed
    initial_concentration_params = {'IL6R_0', 'SMAD3_0', 'SMAD4_0', 'STAT3m_0', 'PKA_0'}

    # Add model parameters (unchanged)
    for param_name in model.parameters:
        param_obj = model.parameters[param_name]
        nominal_value = float(param_obj.value)

        if param_name in stimulus_params:
            should_estimate = 0
        else:
            should_estimate = 1

        if param_name in initial_concentration_params:
            lower_bound = 1.0
            upper_bound = 200.0
        else:
            lower_bound = nominal_value / 100.0 if nominal_value != 0 else 1e-4
            upper_bound = nominal_value * 100.0 if nominal_value != 0 else 1e4

        if should_estimate == 0:
            epsilon = abs(nominal_value) * 1e-10 + 1e-10
            lower_bound = nominal_value - epsilon
            upper_bound = nominal_value + epsilon

        parameters_data.append({
            'parameterId': param_name,
            'parameterName': param_name,
            'parameterScale': 'log10',
            'lowerBound': lower_bound,
            'upperBound': upper_bound,
            'nominalValue': nominal_value,
            'estimate': should_estimate
        })

    # Shared sigma for log-normal noise
    noise_conf = config['time_course_settings']['noise']
    is_noisy = noise_conf['add'] and noise_conf['level_percent'] > 0
    noise_fraction = noise_conf['level_percent'] / 100.0 if is_noisy else 0.0

    if is_noisy:
        sigma_log_nominal = float(np.sqrt(np.log(1.0 + noise_fraction**2)))  # CV hint
        sigma_estimate_flag = 1  # estimate shared sigma
        sigma_lower = 1e-4    # Convert to log10: log10(0.0001) = -4
        sigma_upper = 1.0     # Convert to log10: log10(1.0) = 0
    else:
        sigma_log_nominal = 1e-8
        sigma_estimate_flag = 0
        sigma_lower = 1e-8   # Convert to log10: log10(1e-8) = -8
        sigma_upper = 1e-4    # Convert to log10: log10(1e-4) = -4

    parameters_data.append({
        'parameterId': 'sigma_log_shared',
        'parameterName': 'sigma_log_shared',
        'parameterScale': 'log10',
        'lowerBound': sigma_lower,
        'upperBound': sigma_upper,
        'nominalValue': sigma_log_nominal,
        'estimate': sigma_estimate_flag
    })

    parameters_df = pd.DataFrame(parameters_data)
    parameters_df.to_csv(output_path, sep='\t', index=False)
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
        measurements_df, 
        parameters_path
    )
    
    logging.info("🎉 All PEtab files generated successfully!")
    sys.stdout.flush()


if __name__ == "__main__":
    # excel_to_petab_dose_response(config)
    # generate_time_course_excel(config)
    main()