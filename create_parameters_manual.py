#!/usr/bin/env python3
"""
Manual complete PEtab parameters file generator.
Creates parameters based on known kinetic parameters and observable noise parameters.
"""

import pandas as pd
import os

OUTPUT_PATH = "SimData/parameters_time_course_default.tsv"

print("--- Generating Complete PEtab Parameters File (Manual) ---")

# Define all parameters manually
petab_params = []

# 1. Condition-controlled parameters (fixed)
condition_params = [
    {'parameterId': 'TGFb_0', 'parameterScale': 'log10', 'lowerBound': 1.0, 'upperBound': 1.0, 'nominalValue': 1.0, 'estimate': 0},
    {'parameterId': 'IL6_0', 'parameterScale': 'log10', 'lowerBound': 0.0, 'upperBound': 0.0, 'nominalValue': 0.0, 'estimate': 0}
]

# 2. Kinetic parameters (estimable)
kinetic_params = [
    {'parameterId': 'kf_il6_bind', 'parameterScale': 'log10', 'lowerBound': 0.0001, 'upperBound': 1.0, 'nominalValue': 0.01, 'estimate': 1},
    {'parameterId': 'kr_il6_bind', 'parameterScale': 'log10', 'lowerBound': 0.0001, 'upperBound': 1.0, 'nominalValue': 0.01, 'estimate': 1},
    {'parameterId': 'k_act_il6r', 'parameterScale': 'log10', 'lowerBound': 0.005, 'upperBound': 50.0, 'nominalValue': 0.5, 'estimate': 1},
    {'parameterId': 'k_inact_il6r', 'parameterScale': 'log10', 'lowerBound': 0.0005, 'upperBound': 5.0, 'nominalValue': 0.05, 'estimate': 1},
    {'parameterId': 'k_phos_smad3', 'parameterScale': 'log10', 'lowerBound': 0.001, 'upperBound': 10.0, 'nominalValue': 0.1, 'estimate': 1},
    {'parameterId': 'k_dephos_smad3', 'parameterScale': 'log10', 'lowerBound': 0.0005, 'upperBound': 5.0, 'nominalValue': 0.05, 'estimate': 1},
    {'parameterId': 'k_act_stat3_by_il6r', 'parameterScale': 'log10', 'lowerBound': 0.0002, 'upperBound': 2.0, 'nominalValue': 0.02, 'estimate': 1},
    {'parameterId': 'k_deact_stat3', 'parameterScale': 'log10', 'lowerBound': 0.0005, 'upperBound': 5.0, 'nominalValue': 0.05, 'estimate': 1},
    {'parameterId': 'kf_s3s4', 'parameterScale': 'log10', 'lowerBound': 0.01, 'upperBound': 100.0, 'nominalValue': 1.0, 'estimate': 1},
    {'parameterId': 'kr_s3s4', 'parameterScale': 'log10', 'lowerBound': 0.001, 'upperBound': 10.0, 'nominalValue': 0.1, 'estimate': 1},
    {'parameterId': 'kf_s3stat3d', 'parameterScale': 'log10', 'lowerBound': 0.01, 'upperBound': 100.0, 'nominalValue': 1.0, 'estimate': 1},
    {'parameterId': 'kr_s3stat3d', 'parameterScale': 'log10', 'lowerBound': 0.001, 'upperBound': 10.0, 'nominalValue': 0.1, 'estimate': 1},
    {'parameterId': 'kf_pka_bind', 'parameterScale': 'log10', 'lowerBound': 0.01, 'upperBound': 100.0, 'nominalValue': 1.0, 'estimate': 1},
    {'parameterId': 'kr_pka_bind', 'parameterScale': 'log10', 'lowerBound': 0.001, 'upperBound': 10.0, 'nominalValue': 0.1, 'estimate': 1},
    {'parameterId': 'k_cat_pka', 'parameterScale': 'log10', 'lowerBound': 0.002, 'upperBound': 20.0, 'nominalValue': 0.2, 'estimate': 1},
    {'parameterId': 'k_deact_pka', 'parameterScale': 'log10', 'lowerBound': 0.0005, 'upperBound': 5.0, 'nominalValue': 0.05, 'estimate': 1},
    {'parameterId': 'IL6R_0', 'parameterScale': 'log10', 'lowerBound': 1.0, 'upperBound': 10000.0, 'nominalValue': 100.0, 'estimate': 1},
    {'parameterId': 'SMAD3_0', 'parameterScale': 'log10', 'lowerBound': 1.0, 'upperBound': 10000.0, 'nominalValue': 100.0, 'estimate': 1},
    {'parameterId': 'SMAD4_0', 'parameterScale': 'log10', 'lowerBound': 0.5, 'upperBound': 5000.0, 'nominalValue': 50.0, 'estimate': 1},
    {'parameterId': 'STAT3m_0', 'parameterScale': 'log10', 'lowerBound': 1.0, 'upperBound': 10000.0, 'nominalValue': 100.0, 'estimate': 1},
    {'parameterId': 'PKA_0', 'parameterScale': 'log10', 'lowerBound': 0.5, 'upperBound': 5000.0, 'nominalValue': 50.0, 'estimate': 1}
]

# 3. Noise parameters for each observable (estimable)
observable_ids = [
    'Free_TGFb_obs', 'IL6R_Active', 'Free_IL6_obs', 'pSMAD3_obs',
    'STAT3d_active_obs', 'S3S4_complex_obs', 'S3STAT3d_complex_obs', 'PKA_active'
]

noise_params = []
for obs_id in observable_ids:
    noise_param_id = f"sigma_{obs_id}"
    noise_params.append({
        'parameterId': noise_param_id,
        'parameterScale': 'lin',  # Noise parameters are typically on linear scale
        'lowerBound': 1.0,        # For fixed params, bounds can be the same as nominal
        'upperBound': 1.0,        # For fixed params, bounds can be the same as nominal
        'nominalValue': 1.0,      # Standard nominal value
        'estimate': 0             # <-- THE FIX: Set to 0 to make it a fixed parameter
    })

# Combine all parameters
petab_params = condition_params + kinetic_params + noise_params

print(f"Creating parameters file with {len(petab_params)} total parameters:")
print(f"  - {len(condition_params)} condition-controlled parameters")
print(f"  - {len(kinetic_params)} kinetic parameters")
print(f"  - {len(noise_params)} noise parameters")

# Create DataFrame and save
parameters_df = pd.DataFrame(petab_params)

# Sort by parameterId for better organization
parameters_df = parameters_df.sort_values('parameterId')

try:
    parameters_df.to_csv(OUTPUT_PATH, sep='\t', index=False)
    print(f"✅ Successfully created complete parameters file at: {OUTPUT_PATH}")
    
    # Summary statistics
    estimable_count = len([p for p in petab_params if p['estimate'] == 1])
    fixed_count = len([p for p in petab_params if p['estimate'] == 0])
    print(f"   - {estimable_count} parameters to estimate")
    print(f"   - {fixed_count} fixed parameters")
    
    # Verify file exists and has content
    verify_df = pd.read_csv(OUTPUT_PATH, sep='\t')
    print(f"✅ Verification: File contains {len(verify_df)} parameters")
    
except Exception as e:
    print(f"ERROR: Failed to save parameters file: {e}")
    exit(1)

print("Complete parameters file generation finished!")
