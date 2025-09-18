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
            try:
                true_params[param_name] = float(param_obj.value)
            except ValueError:
                logging.debug(f"  Skipping parameter '{param_name}' with expression value '{param_obj.value}'.")
                continue
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

    # Find the simulator ID for IL6(r) to check its value. 
    # This is inefficient to do every time, but fine for a quick debug.
    il6_species_id = None
    for species_id in simulator.model.getFloatingSpeciesIds():
        if "IL6(r)" in species_id: # Find the species ID for IL6
            il6_species_id = species_id
            break
    
    if il6_species_id:
        logging.debug(f"    [DEBUG] IL-6 concentration AFTER loading pre-eq state: {simulator.model[il6_species_id]}")

    # 3. Apply the specific stimuli for the current experimental condition
    for species_id, value in stimuli.items():
        simulator.model[species_id] = value
    
    if il6_species_id:
        logging.debug(f"    [DEBUG] IL-6 concentration AFTER applying stimulus: {simulator.model[il6_species_id]}")

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


def run_simulation_no_preeq(
    model: bionetgen.bngmodel,
    true_params: dict,
    stimuli: dict,
    sim_duration: float,
    sim_steps: int
) -> pd.DataFrame:
    """
    Runs a time-course simulation starting directly from model seed species at t=0
    with condition-specific IL6_0/TGFb_0 values applied.
    """
    simulator = model.setup_simulator()
    
    # Set all kinetic parameters to their true values
    for name, value in true_params.items():
        simulator.model[name] = value
    
    # Apply stimuli at t=0 (use SBML IDs discovered earlier)
    for species_id, value in stimuli.items():
        simulator.model[species_id] = value
        
    # Apply robust integrator settings
    simulator.integrator.stiff = True
    simulator.integrator.absolute_tolerance = 1e-8
    simulator.integrator.relative_tolerance = 1e-6
    simulator.integrator.maximum_num_steps = 50000
    
    # Simulate directly from seed species with condition stimuli at t=0
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
    eps = np.exp(rng.normal(loc=0, scale=sigma, size=len(data_series)))
    noisy_series = data_series.to_numpy(dtype=float) * eps
    # Log-normal factor is > 0, so no need to clip; keep clip for robustness if desired
    # noisy_series = np.clip(noisy_series, a_min=0.0, a_max=None)
    return pd.Series(noisy_series, index=data_series.index)

def add_combined_noise(data_series: pd.Series, sigma_add: float, sigma_mult: float, rng: np.random.Generator) -> pd.Series:
    """
    Adds combined additive and multiplicative Gaussian noise on linear scale.
    noise ~ N(0, (sigma_add + sigma_mult * |true|)^2)

    Also truncates results to a small positive floor to avoid negative values.
    """
    if (sigma_add is None or sigma_add <= 0) and (sigma_mult is None or sigma_mult <= 0):
        return data_series.copy()

    true_values = data_series.to_numpy(dtype=float)
    s_add = float(max(0.0, sigma_add if sigma_add is not None else 0.0))
    s_mult = float(max(0.0, sigma_mult if sigma_mult is not None else 0.0))
    std_devs = s_add + s_mult * np.abs(true_values)
    noise = rng.normal(loc=0.0, scale=std_devs, size=len(true_values))
    noisy = true_values + noise
    # Apply small positive floor
    floor_value = 1e-8
    noisy = np.maximum(noisy, floor_value)
    return pd.Series(noisy, index=data_series.index)

# --------------------------------------------------------------------------
#                   TIME-COURSE WORKFLOW WITH PEtab TSV OUTPUT
# --------------------------------------------------------------------------

def generate_time_course_petab(config):
    """
    Generates time-course data starting from t=0 with condition-specific stimuli,
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
    
    # Discover species mapping for stimulus parameters
    temp_model_for_tracing = bionetgen.bngmodel(model_path)
    param_to_sbml_id = discover_species_map(temp_model_for_tracing, list(condition_params))
    
    # 2.5 Simulation configuration: support either uniform duration/steps or explicit time_points + replicates
    sim_conf = tc_settings.get('simulation', {})
    time_points_mode = 'time_points' in sim_conf and isinstance(sim_conf.get('time_points'), (list, tuple)) and len(sim_conf.get('time_points')) > 0
    if time_points_mode:
        time_points_to_sample = list(sim_conf['time_points'])
        n_replicates = int(sim_conf.get('replicates', 1))
        # Simulate densely up to max time to ensure we can sample requested points reliably
        t_end = float(max(time_points_to_sample))
        # Heuristic for dense steps: at least 100 steps or 5 per time unit, whichever is larger
        sim_duration = t_end
        sim_steps = int(max(100, 5 * t_end))
        logging.info(f"Using explicit time points mode with {len(time_points_to_sample)} points, {n_replicates} replicates each.")
    else:
        # Default/uniform sampling mode for backward compatibility
        sim_duration = float(sim_conf.get('duration', 60.0))
        sim_steps = int(sim_conf.get('steps', 10))
        logging.info(f"Using uniform sampling: duration={sim_duration}, steps={sim_steps}.")
    
    # 3. Check if pre-equilibration is enabled
    use_preeq = tc_settings.get('preequilibration', True)  # Default to True for backward compatibility
    
    if use_preeq:
        # OLD PATH: Calculate pre-equilibration steady-state
        stimuli_to_zero = {param: 0.0 for param in variable_stimuli}
        baseline_condition = tc_settings['conditions']['TREG']
        constant_stimuli = {param: baseline_condition[param] for param in constant_stimuli_names if param in baseline_condition}
        preeq_ss = calculate_preeq_steadystate(model, true_params, stimuli_to_zero, constant_stimuli, param_to_sbml_id)
        logging.info(f"  Pre-equilibration steady-state calculated.")
        
        # Run simulations from pre-equilibrated state
        time_course_results = {}
        for condition_name, condition_values in tc_settings['conditions'].items():
            logging.info(f"  Simulating condition: {condition_name}")
            stimuli_with_ids = {param_to_sbml_id[p]: v for p, v in condition_values.items()}
            result_df = run_simulation_from_preeq(
                model, preeq_ss, true_params, stimuli_with_ids, 
                sim_duration, sim_steps
            )
            time_course_results[condition_name] = result_df
    else:
        # NEW PATH: Simulate directly from t=0 with condition stimuli
        logging.info(f"  Pre-equilibration disabled. Simulating directly from t=0.")
        time_course_results = {}
        for condition_name, condition_values in tc_settings['conditions'].items():
            logging.info(f"  Simulating condition: {condition_name}")
            stimuli_with_ids = {param_to_sbml_id[p]: v for p, v in condition_values.items()}
            result_df = run_simulation_no_preeq(
                model, true_params, stimuli_with_ids,
                sim_duration, sim_steps
            )
            time_course_results[condition_name] = result_df

    # 4/5. Build measurement table
    all_observables = [obs_name for obs_name in model.observables]
    # Optional: allow selecting a subset of observables to include
    obs_include = tc_settings.get('observables_to_include')
    if isinstance(obs_include, (list, tuple)) and len(obs_include) > 0:
        include_set = set(obs_include)
        all_observables = [o for o in all_observables if o in include_set]
        logging.info(f"Including a subset of observables ({len(all_observables)}): {sorted(all_observables)}")
    noise_conf = tc_settings['noise']
    # Determine noise model
    noise_add = bool(noise_conf.get('add', False))
    combined_mode = noise_add and (
        str(noise_conf.get('model', '')).lower().startswith('combined') or
        ('sigma_add' in noise_conf and 'sigma_mult' in noise_conf)
    )

    if time_points_mode:
        # Filter simulated trajectories to the requested time points, then create replicates with noise
        logging.info("Filtering simulation output to specified time points and generating replicates...")
        time_tolerance = 1e-6
        seed = tc_settings.get('random_seed', 42)
        rng = np.random.default_rng(seed)
        noise_fraction = noise_conf['level_percent'] / 100.0 if (noise_add and not combined_mode) else 0.0
        sigma_add = float(noise_conf.get('sigma_add', 0.0)) if combined_mode else 0.0
        sigma_mult = float(noise_conf.get('sigma_mult', 0.0)) if combined_mode else 0.0

        measurement_rows = []
        for condition_name, result_df in time_course_results.items():
            # Keep only rows whose time matches any requested time point (within tolerance)
            mask = result_df['time'].apply(lambda t: any(abs(t - tp) < time_tolerance for tp in time_points_to_sample))
            filtered_df = result_df.loc[mask].copy()
            # Snap times to the exact requested values for clean output
            filtered_df.loc[:, 'time'] = filtered_df['time'].apply(lambda t: min(time_points_to_sample, key=lambda tp: abs(t - tp)))

            for _, row in filtered_df.iterrows():
                time_val = float(row['time'])
                for obs_name in all_observables:
                    if obs_name in row:
                        true_val = float(row[obs_name])
                        # Create N replicates for each observable/time/condition
                        for _ in range(n_replicates):
                            meas_val = true_val
                            # Apply LOD floor first if noise is enabled
                            if noise_add and meas_val <= 1e-12:
                                meas_val = 1e-8
                            # Add multiplicative log-normal noise per replicate if requested
                            if noise_add:
                                if combined_mode:
                                    meas_val = add_combined_noise(pd.Series([meas_val]), sigma_add, sigma_mult, rng).iloc[0]
                                elif noise_fraction > 0:
                                    meas_val = add_noise(pd.Series([meas_val]), noise_fraction, rng).iloc[0]

                            measurement_row = {
                                'observableId': obs_name,
                                'simulationConditionId': condition_name,
                                'time': time_val,
                                'measurement': meas_val,
                            }
                            if use_preeq:
                                measurement_row['preequilibrationConditionId'] = 'preeq_ss'
                            measurement_rows.append(measurement_row)

        measurement_df = pd.DataFrame(measurement_rows)
        logging.info(f"  Generated a total of {len(measurement_df)} measurement points including replicates.")
    else:
        # Uniform sampling path (backward compatible): build long table then apply noise once
        measurement_rows = []
        for condition_name, result_df in time_course_results.items():
            for _, row in result_df.iterrows():
                time_val = row['time']
                for obs_name in all_observables:
                    if obs_name in row:
                        measurement_row = {
                            'observableId': obs_name,
                            'simulationConditionId': condition_name,
                            'time': time_val,
                            'measurement': row[obs_name],
                        }
                        if use_preeq:
                            measurement_row['preequilibrationConditionId'] = 'preeq_ss'
                        measurement_rows.append(measurement_row)

        measurement_df = pd.DataFrame(measurement_rows)

        # Apply LOD and noise in bulk as before
        if noise_add:
            def apply_floor(row):
                if row['measurement'] <= 1e-12:
                    return 1e-8
                return row['measurement']
            measurement_df['measurement'] = measurement_df.apply(apply_floor, axis=1)
            logging.info("  ...Floor value applied to non-positive measurements.")

            seed = tc_settings.get('random_seed', 42)
            rng = np.random.default_rng(seed)
            if combined_mode:
                sigma_add = float(noise_conf.get('sigma_add', 0.0))
                sigma_mult = float(noise_conf.get('sigma_mult', 0.0))
                logging.info(f"  Adding combined normal noise: sigma_add={sigma_add}, sigma_mult={sigma_mult}...")
                measurement_df['measurement'] = add_combined_noise(measurement_df['measurement'], sigma_add, sigma_mult, rng)
                logging.info("  ...Combined noise added successfully.")
            else:
                logging.info(f"  Adding {noise_conf['level_percent']}% lognormal noise...")
                noise_fraction = noise_conf['level_percent'] / 100.0
                measurement_df['measurement'] = add_noise(measurement_df['measurement'], noise_fraction, rng)
                logging.info("  ...Noise added successfully.")

    # 7. Create and save PEtab files
    condition_rows = []
    for condition_name, condition_values in tc_settings['conditions'].items():
        condition_row = {'conditionId': condition_name}
        condition_row.update(condition_values)
        condition_rows.append(condition_row)
    
    # Only add preeq_ss condition if using pre-equilibration
    if use_preeq:
        baseline_cond = tc_settings['conditions'].get('TREG', {})
        preeq_row = {'conditionId': 'preeq_ss'}
        preeq_row.update(baseline_cond)
        condition_rows.append(preeq_row)
        
    condition_df = pd.DataFrame(condition_rows)

    # Filename suffix reflects noise model
    if noise_add:
        if combined_mode:
            filename_suffix = "_noise_combined"
        elif 'level_percent' in noise_conf:
            # Replace dots with underscores in noise level for filesystem compatibility
            noise_level_str = str(noise_conf['level_percent']).replace('.', '_')
            filename_suffix = f"_noise{noise_level_str}"
        else:
            filename_suffix = "_noise"
    else:
        filename_suffix = "_no_noise"
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
    logging.info("Creating observables PEtab file (supports log-normal or combined normal noise)...")

    noise_conf = config['time_course_settings']['noise']
    noise_add = bool(noise_conf.get('add', False))
    combined_mode = noise_add and (
        str(noise_conf.get('model', '')).lower().startswith('combined') or
        ('sigma_add' in noise_conf and 'sigma_mult' in noise_conf)
    )
    is_noisy = noise_add and ((not combined_mode and noise_conf.get('level_percent', 0) > 0) or combined_mode)

    observable_ids = measurements_df['observableId'].unique()
    # Respect optional observables_to_include (already applied to measurements, but keep consistent here)
    obs_include = config.get('time_course_settings', {}).get('observables_to_include')
    if isinstance(obs_include, (list, tuple)) and len(obs_include) > 0:
        observable_ids = [oid for oid in observable_ids if oid in set(obs_include)]
    observables_data = []

    for obs_id in observable_ids:
        # Noise model selection
        if combined_mode:
            # Combined additive + multiplicative on linear scale, normal distribution
            noise_formula = f"sigma_add + sigma_mult * {obs_id}"
            noise_dist = "normal"
        else:
            # Legacy log-normal with shared sigma
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
    # Compute per-stimulus max across defined conditions to derive sane bounds (esp. when nominal is 0)
    conds = config.get('time_course_settings', {}).get('conditions', {})
    stim_max_values = {p: 0.0 for p in stimulus_params}
    for _cname, cvals in conds.items():
        for p in stimulus_params:
            if p in cvals:
                try:
                    stim_max_values[p] = max(float(cvals[p]), stim_max_values[p])
                except Exception:
                    pass

    for param_name in model.parameters:
        # Get the parameter OBJECT using the name as a key
        param_obj = model.parameters[param_name]
        # Access the .value attribute from the OBJECT
        try:
            nominal_value = float(param_obj.value)
        except ValueError:
            logging.debug(f"  Skipping expression '{param_name}' during parameters.tsv creation.")
            continue
        
        # Default bounds for kinetic rates
        lower_bound = nominal_value / 100.0
        upper_bound = nominal_value * 100.0
        
        if param_name in stimulus_params:
            estimate = 0
            parameter_scale = 'lin'
            # For fixed stimulus parameters, set bounds to cover condition values safely
            max_val = max(stim_max_values.get(param_name, 0.0), nominal_value)
            lower_bound = 0.0
            upper_bound = max(10.0 * max_val, 1.0)  # at least 1.0
        elif param_name.endswith("_0"):
            estimate = 1
            parameter_scale = 'log10'
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

    # Add noise parameter(s)
    noise_conf = config['time_course_settings']['noise']
    noise_add = bool(noise_conf.get('add', False))
    combined_mode = noise_add and (
        str(noise_conf.get('model', '')).lower().startswith('combined') or
        ('sigma_add' in noise_conf and 'sigma_mult' in noise_conf)
    )

    if combined_mode:
        # Add sigma_add and sigma_mult parameters (estimated on log10 scale)
        sigma_add_nom = float(noise_conf.get('sigma_add', 0.1))
        sigma_mult_nom = float(noise_conf.get('sigma_mult', 0.1))
        parameters_data.append({
            'parameterId': 'sigma_add', 'parameterName': 'sigma_add',
            'parameterScale': 'log10', 'lowerBound': 1e-8, 'upperBound': 10.0,
            'nominalValue': sigma_add_nom, 'estimate': 1
        })
        parameters_data.append({
            'parameterId': 'sigma_mult', 'parameterName': 'sigma_mult',
            'parameterScale': 'log10', 'lowerBound': 1e-4, 'upperBound': 1.0,
            'nominalValue': sigma_mult_nom, 'estimate': 1
        })
    else:
        # Single shared log-normal sigma parameter
        is_noisy = noise_add and noise_conf.get('level_percent', 0) > 0
        if is_noisy:
            cv = noise_conf['level_percent'] / 100.0
            sigma_nominal = float(np.sqrt(np.log(1.0 + cv**2)))
            parameters_data.append({
                'parameterId': 'sigma_log_shared', 'parameterName': 'sigma_log_shared',
                'parameterScale': 'log10',
                'lowerBound': 1e-5, 'upperBound': 10,
                'nominalValue': sigma_nominal, 'estimate': 1
            })
        else:
            parameters_data.append({
                'parameterId': 'sigma_log_shared', 'parameterName': 'sigma_log_shared',
                'parameterScale': 'log10', 'lowerBound': 1e-10, 'upperBound': 1.0,
                'nominalValue': 1e-8, 'estimate': 1
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
    # Allow overriding the PEtab output folder via config (defaults to 'petab_files')
    petab_output_dir = config.get('petab_output_dir', 'petab_files')
    os.makedirs(petab_output_dir, exist_ok=True)
    
    # Add noise suffix to filenames if noise is enabled
    noise_conf = config['time_course_settings']['noise']
    noise_add = bool(noise_conf.get('add', False))
    combined_mode = noise_add and (
        str(noise_conf.get('model', '')).lower().startswith('combined') or
        ('sigma_add' in noise_conf and 'sigma_mult' in noise_conf)
    )
    if combined_mode:
        noise_suffix = "_noise_combined"
    elif noise_add and noise_conf.get('level_percent', 0) > 0:
        # Replace dots with underscores in noise level for filesystem compatibility
        noise_level_str = str(noise_conf['level_percent']).replace('.', '_')
        noise_suffix = f"_noise{noise_level_str}"
    else:
        noise_suffix = ""
    
    observables_path = os.path.join(petab_output_dir, f"observables{noise_suffix}.tsv")
    parameters_path = os.path.join(petab_output_dir, f"parameters{noise_suffix}.tsv")
    
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