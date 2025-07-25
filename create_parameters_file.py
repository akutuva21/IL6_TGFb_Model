import bionetgen
import pandas as pd
import os

# --- Configuration ---
BNGL_MODEL_PATH = "model_even_smaller.bngl"
OUTPUT_PATH = "SimData/parameters_time_course_default.tsv"

print("--- Generating PEtab parameters file with TIGHT bounds ---")

# 1. Load the BioNetGen model
model = bionetgen.bngmodel(BNGL_MODEL_PATH)

# 2. Define which parameters should be estimated
parameters_to_estimate = {
    'k_act_stat3_by_il6r', 
    'k_cat_pka', 
    'kf_pka_bind', 
    'kf_s3s4', 
    'kf_s3stat3d'
}
condition_controlled_params = {'IL6_0', 'TGFb_0'}

# 3. Build the parameters DataFrame
petab_params = []
for param_name in model.parameters:
    param_obj = model.parameters[param_name]
    nominal_value = float(param_obj.value)
    
    if param_name in condition_controlled_params:
        should_estimate = 0
        lower_bound = nominal_value
        upper_bound = nominal_value
    
    elif param_name in parameters_to_estimate:
        should_estimate = 1
        # --- DEFINITIVE FIX: Enforce strict 10x bounds ---
        lower_bound = nominal_value / 10.0
        upper_bound = nominal_value * 10.0
        print(f"  Parameter '{param_name}': ESTIMATED with TIGHT bounds [{lower_bound:.4e}, {upper_bound:.4e}]")
        # -----------------------------------------------

    else: # All other parameters are fixed
        should_estimate = 0
        lower_bound = nominal_value
        upper_bound = nominal_value

    petab_params.append({
        'parameterId': param_name,
        'parameterScale': 'log10',
        'lowerBound': lower_bound,
        'upperBound': upper_bound,
        'nominalValue': nominal_value,
        'estimate': should_estimate
    })

parameters_df = pd.DataFrame(petab_params)

# 4. Save the new parameters file
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
parameters_df.to_csv(OUTPUT_PATH, sep='\t', index=False)

print(f"\n✅ Successfully created new parameters file at: {OUTPUT_PATH}")
print(f"   - Total parameters: {len(parameters_df)}")
print(f"   - Estimated parameters: {len(parameters_df[parameters_df['estimate'] == 1])}")

# 5. Display the final bounds for estimated parameters for verification
print("\n--- Final Bounds for Estimated Parameters ---")
print(parameters_df[parameters_df['estimate'] == 1])