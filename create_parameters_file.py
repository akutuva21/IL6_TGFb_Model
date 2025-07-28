import bionetgen
import pandas as pd
import os

# --- Configuration ---
# Point this to your original BioNetGen model file
BNGL_MODEL_PATH = "model_even_smaller.bngl"  # Use the .bngl file since .net might not exist
# Define where to save the new parameters file
OUTPUT_PATH = "SimData/parameters_time_course_default.tsv"

print("--- Generating PEtab parameters file with sigma variables ---")

# 1. Load the BioNetGen model
print(f"Loading model from: {BNGL_MODEL_PATH}")
model = bionetgen.bngmodel(BNGL_MODEL_PATH)

# 2. Define which parameters are controlled by the conditions file
# These will be marked as NOT estimated (estimate = 0)
condition_controlled_params = {'IL6_0', 'TGFb_0'}

# 2.1. Define the specific parameters to estimate
parameters_to_estimate = {
    'k_act_stat3_by_il6r', 
    'k_cat_pka', 
    'kf_pka_bind', 
    'kf_s3s4', 
    'kf_s3stat3d'
}

# 3. Define the observables that need sigma parameters
# These are from your measurements file and config.yml
observable_ids = [
    'Free_IL6_obs',
    'Free_TGFb_obs', 
    'IL6R_Active',
    'PKA_active',
    'S3S4_complex_obs',
    'S3STAT3d_complex_obs',
    'STAT3d_active_obs',
    'pSMAD3_obs'
]

# 4. Build the parameters DataFrame
print("Building parameters table...")
print(f"DEBUG: model.parameters type = {type(model.parameters)}")

petab_params = []

# First, add all the kinetic parameters from the BNGL model
try:
    # Get parameter names from the ParameterBlock
    for param_name in model.parameters:
        # Access the parameter object using the name
        param_obj = model.parameters[param_name]
        
        # Get the parameter value
        nominal_value = float(param_obj.value)
            
        # Check if this parameter is controlled by the experiment conditions
        if param_name in condition_controlled_params:
            should_estimate = 0
            # For non-estimated parameters, bounds are often set to the nominal value
            lower_bound = nominal_value
            upper_bound = nominal_value
            print(f"  Parameter '{param_name}': NOT estimated (controlled by conditions)")
        elif param_name in parameters_to_estimate:
            # Only these 5 specific parameters are estimated
            should_estimate = 1
            # Set bounds for estimated parameters
            lower_bound = nominal_value / 10.0
            upper_bound = nominal_value * 10.0
            print(f"  Parameter '{param_name}': ESTIMATED with bounds [{lower_bound:.3e}, {upper_bound:.3e}]")
        else:
            # All other kinetic parameters are fixed at their nominal values
            should_estimate = 0
            lower_bound = nominal_value
            upper_bound = nominal_value
            print(f"  Parameter '{param_name}': NOT estimated (fixed kinetic parameter)")

        petab_params.append({
            'parameterId': param_name,
            'parameterScale': 'log10',  # log10 is standard for kinetic parameters
            'lowerBound': lower_bound,
            'upperBound': upper_bound,
            'nominalValue': nominal_value,
            'estimate': should_estimate
        })
        
except Exception as e:
    print(f"ERROR: Could not process model.parameters: {e}")
    print(f"       Type: {type(model.parameters)}")
    raise

# Second, add sigma parameters for each observable (these will NOT be estimated)
print("\nAdding sigma (noise) parameters for observables...")
for obs_id in observable_ids:
    sigma_param_id = f"sigma_{obs_id}"
    print(f"  Adding noise parameter: {sigma_param_id} (NOT estimated)")
    
    petab_params.append({
        'parameterId': sigma_param_id,
        'parameterScale': 'lin',     # Noise parameters are typically on linear scale
        'lowerBound': 1.0,          # Fixed at nominal value (not estimated)
        'upperBound': 1.0,          # Fixed at nominal value (not estimated)  
        'nominalValue': 1.0,        # Standard default for noise
        'estimate': 0               # NOT estimated
    })

parameters_df = pd.DataFrame(petab_params)

# 5. Save the new parameters file
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
parameters_df.to_csv(OUTPUT_PATH, sep='\t', index=False)

print(f"\n✅ Successfully created parameters file at: {OUTPUT_PATH}")
print(f"   - Kinetic parameters: {len([p for p in petab_params if not p['parameterId'].startswith('sigma_')])}")
print(f"   - Sigma parameters: {len([p for p in petab_params if p['parameterId'].startswith('sigma_')])}")
print(f"   - Total parameters: {len(parameters_df)}")
print(f"   - Estimated parameters: {len(parameters_df[parameters_df['estimate'] == 1])}")
print(f"   - Fixed parameters: {len(parameters_df[parameters_df['estimate'] == 0])}")

# 6. Display the sigma parameters for verification
print("\n--- Sigma Parameters Added ---")
sigma_params = parameters_df[parameters_df['parameterId'].str.startswith('sigma_')]
print(sigma_params[['parameterId', 'parameterScale', 'nominalValue', 'estimate']])
