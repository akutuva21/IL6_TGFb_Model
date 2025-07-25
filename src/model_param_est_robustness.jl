# src/model_param_est_robustness.jl

using PEtab
using DataFrames

# Only export the setup function
export setup_petab_problem

function setup_petab_problem(path_to_yaml::String)
    println("--- Setting up PEtab Problem from YAML: $path_to_yaml ---")

    # This single line creates the PEtabModel from the YAML file.
    # This is the intended, robust workflow.
    petab_model = PEtabModel(path_to_yaml, verbose=true)
    
    println("--- PEtab Problem Setup Complete ---")
    
    # --- START: DEFINITIVE FIX FOR TRUE VALUES ---
    true_param_values = Dict{String, Float64}()
    
    if hasfield(typeof(petab_model), :petab_tables) && haskey(petab_model.petab_tables, :parameters)
        param_table = petab_model.petab_tables[:parameters]
        
        # Extract nominal values for ALL parameters (both estimated and fixed)
        for row in eachrow(param_table)
            # Use STRING keys to match the plotting function
            param_name_str = string(row.parameterId)
            true_param_values[param_name_str] = row.nominalValue
        end

        println("INFO: Extracted $(length(true_param_values)) true parameter values for visualization")
    else
        @warn "Could not extract parameter table for true values"
    end
    # --- END: DEFINITIVE FIX ---

    return (petab_model=petab_model, true_values=true_param_values)
end