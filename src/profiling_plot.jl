# src/profiling.jl

using PEtab
using LikelihoodProfiler
using Optimization
using OptimizationNLopt # Provides the LBFGS optimizer
using Printf
using ComponentArrays
using Plots

# Ensure GR backend for saving plots in headless environments
gr() 

function run_likelihood_profiling(
    petab_model::PEtabModel,
    odesolver,
    steadystate_solver,
    θ_mle::ComponentVector,
    true_param_values::Dict
)
    println("\n--- Generating Full Likelihood Profiles (Developer Recommended Method) ---")
    t_start = time()

    # Use robust solver settings for the profiling, as recommended
    petab_problem = PEtabODEProblem(petab_model, verbose=false,
                                    odesolver=ODESolver(Rodas5P(), abstol=1e-8, reltol=1e-8))
    
    # LikelihoodProfiler requires an OptimizationProblem
    optprob = OptimizationProblem(petab_problem)

    profile_bounds = collect(tuple.(petab_problem.lower_bounds, petab_problem.upper_bounds))
    plprob = ProfileLikelihoodProblem(optprob, θ_mle, profile_bounds)

    # --- IMPLEMENTING DEVELOPER'S ADVICE ---

    # 1. Use a gradient-based optimizer for the sub-problems.
    #    NLopt's LBFGS is a good and robust choice.
    sub_optimizer = NLopt.LD_LBFGS()

    # 2. Use an adaptive step size that is proportional to the parameter's magnitude.
    #    This is a much more stable stepping strategy.
    adaptive_stepper = FixedStep(; initial_step = (θ, i) -> 0.01 * abs(θ[i]))

    # 3. Configure the OptimizationProfiler with these robust settings.
    method = OptimizationProfiler(optimizer = sub_optimizer, 
                                  stepper = adaptive_stepper,
                                  optimizer_opts = (reltol=1e-2, maxeval=5000))

    # --- END OF IMPLEMENTATION ---

    println("Starting profile calculation for all parameters...")
    sol = solve(plprob, method; parallel_type=:threads)

    # --- Save the plots ---
    println("Saving profile likelihood plots...")
    plot_dir = "likelihood_profiles"
    mkpath(plot_dir)

    p_all = plot(sol, layout=(6, 3), size=(1800, 2400))
    savefig(p_all, joinpath(plot_dir, "all_profiles.png"))

    param_names = string.(petab_problem.xnames)
    for i in 1:length(sol)
        p_single = plot(sol[i], title=param_names[i])
        savefig(p_single, joinpath(plot_dir, "profile_$(param_names[i]).png"))
    end
    println("✅ All profile plots saved to the '$plot_dir' directory.")

    println("[Profiling] Done in $(round(time() - t_start; digits=2)) s")
    return nothing
end