# src/model_param_est_robustness.jl

using PEtab

# Only export the setup function
export setup_petab_problem

function setup_petab_problem(path_to_yaml::String)
    println("--- Setting up PEtab Problem from YAML: $path_to_yaml ---")

    # This single line creates the PEtabModel from the YAML file.
    # This is the intended, robust workflow.
    petab_model = PEtabModel(path_to_yaml, verbose=true)
    
    println("--- PEtab Problem Setup Complete ---")
    
    # For now, we return an empty dictionary for true_values.
    # This can be implemented later if needed for plotting.
    true_param_values = Dict{String, Float64}()

    return (petab_model=petab_model, true_values=true_param_values)
end