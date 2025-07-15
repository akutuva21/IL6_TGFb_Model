import pandas as pd
import yaml

# --- Configuration ---
CONFIG_PATH = "config.yml"
MEASUREMENTS_PATH = "SimData/measurements_time_course_default.tsv"
OUTPUT_PATH = "SimData/observables_time_course_default.tsv"

print("--- Generating PEtab observables file ---")

# 1. Load the YAML config to get the observable formulas
print(f"Loading observable mapping from: {CONFIG_PATH}")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
observables_mapping = config['observables_mapping']

# 2. Load the measurements file to get all unique observable IDs
print(f"Reading observable IDs from: {MEASUREMENTS_PATH}")
measurements_df = pd.read_csv(MEASUREMENTS_PATH, sep='\t')
unique_observable_ids = measurements_df['observableId'].unique()

# 3. Build the observables DataFrame
print("Building observables table...")
petab_observables = []
for obs_id in unique_observable_ids:
    if obs_id not in observables_mapping:
        print(f"WARNING: Observable ID '{obs_id}' from measurements file is not in config.yml. Skipping.")
        continue
    
    # The formula is the corresponding value from the config mapping
    formula = observables_mapping[obs_id]
    
    # The standard PEtab noise formula is 'sigma_' + observableId
    noise_formula = f"sigma_{obs_id}"
    
    petab_observables.append({
        'observableId': obs_id,
        'observableFormula': formula,
        'noiseFormula': noise_formula,
        'observableTransformation': 'lin'  # Assuming linear scale, change if needed
    })

observables_df = pd.DataFrame(petab_observables)

# 4. Save the new observables file
observables_df.to_csv(OUTPUT_PATH, sep='\t', index=False)

print(f"✅ Successfully created observables file at: {OUTPUT_PATH}")
print(f"   - Number of observables: {len(observables_df)}")
print("   - Observable IDs:", list(observables_df['observableId']))
