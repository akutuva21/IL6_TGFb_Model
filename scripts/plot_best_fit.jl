using Pkg
Pkg.activate("bngl_julia")

include("../src/model_param_est_robustness.jl")
include("../src/optimization.jl")
include("../src/visualization.jl")

using JLD2
using PEtab

function main()
    println("Loading PEtab problem and best fit result...")
    
    # Setup problem
    setup_results = setup_petab_problem("petab_problem.yml")
    petab_problem = PEtabODEProblem(
        setup_results.petab_model, 
        split_over_conditions=true
    )

    # Load fit data
    fit_data = JLD2.load("results/best_fit.jld2")
    multistart_res = fit_data["multistart_result"]

    println("Generating parameter distribution plot WITHOUT true parameters...")
    # Call the plot function without reference_values (it defaults to nothing)
    plot_parameter_distribution(multistart_res, petab_problem)
    
    println("Plotting complete.")
end

main()
