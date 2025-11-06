using PEtab
using LikelihoodProfiler
using Optimization
using OptimizationOptimJL
using OrdinaryDiffEq
using SciMLSensitivity
using ReverseDiff
using SciMLBase
using ComponentArrays
using Plots
using Logging

function run_likelihood_profiling(
    petab_model::PEtabModel,
    odesolver,
    steadystate_solver,
    θ_mle::ComponentVector,
    true_param_values::Dict
)
    println("\n--- Running LikelihoodProfiler workflows (Integration & Optimization) ---")
    t_start = time()

    # Solver configuration with sensible defaults inspired by upstream guidance.
    solver = isnothing(odesolver) ?
        ODESolver(KenCarp47(autodiff=false), abstol=1e-8, reltol=1e-8) :
        odesolver
    ss_solver = isnothing(steadystate_solver) ?
        SteadyStateSolver(:Simulate, abstol=1e-8, reltol=1e-8) :
        steadystate_solver

    petab_problem = PEtabODEProblem(
        petab_model;
        odesolver = solver,
        ss_solver = ss_solver,
        gradient_method = :ForwardDiff,
        sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP()),
        verbose = false,
    )

    param_names = string.(petab_problem.xnames)
    println("Parameter count: ", length(param_names))
    if !isempty(true_param_values)
        matches = count(name -> haskey(true_param_values, name), param_names)
        println("True parameter values available for $(matches) / $(length(param_names)) parameters")
    end

    # Convert incoming θ_mle to a dense vector and optionally re-optimise.
    θ_guess = collect(θ_mle)
    optprob = OptimizationProblem(petab_problem)
    θ_best = θ_guess
    optsol = nothing
    try
        optsol = solve(optprob, Optimization.LBFGS(); maxiters = 10_000)
        if optsol.u isa AbstractVector && all(isfinite, optsol.u)
            θ_best = collect(optsol.u)
        else
            @warn "LBFGS optimisation returned non-finite solution; falling back to provided θ_mle"
        end
    catch err
        @warn "LBFGS optimisation failed; using provided θ_mle" err
    end

    println("Baseline loss (objective at θ_best): ", petab_problem.nllh(θ_best))

    plprob = ProfileLikelihoodProblem(optprob, θ_best)

    integration_profiler = IntegrationProfiler(
        integrator = Tsit5(),
        integrator_opts = (dtmax = 0.01, reltol = 1e-3, abstol = 1e-6),
        matrix_type = :identity,
        gamma = 0.0,
        reoptimize = true,
        optimizer = Optimization.LBFGS(),
        optimizer_opts = (maxiters = 10_000,),
    )

    println("\n--- Running IntegrationProfiler ---")
    prof_sol_ode = LikelihoodProfiler.solve(plprob, integration_profiler; verbose = true)
    endpoints_integration = endpoints(prof_sol_ode)

    optimization_profiler = OptimizationProfiler(
        optimizer = Optimization.LBFGS(),
        optimizer_opts = (maxiters = 10_000,),
        stepper = FixedStep(; initial_step = (p, i) -> 0.05),
    )

    println("\n--- Running OptimizationProfiler ---")
    prof_sol_opt = LikelihoodProfiler.solve(plprob, optimization_profiler; verbose = true)
    endpoints_optimization = endpoints(prof_sol_opt)

    # Persist per-parameter plots for the integration profiler results.
    plot_dir = joinpath("likelihood_profiles", "integration_profiler")
    mkpath(plot_dir)
    plots = Vector{Any}(undef, length(prof_sol_ode))
    for i in eachindex(prof_sol_ode)
        plt = plot(
            prof_sol_ode[i],
            xguide = param_names[i],
            yguide = "Objective",
            margins = 5mm,
            legend = false,
        )
        savefig(plt, joinpath(plot_dir, "profile_$(param_names[i]).png"))
        plots[i] = plt
    end

    # Grid plot for quick inspection
    grid_layout_rows = ceil(Int, length(plots) / 3)
    combined = plot(
        plots...;
        size = (900, max(600, 250 * grid_layout_rows)),
        layout = (grid_layout_rows, 3),
        legend = false,
    )
    savefig(combined, joinpath(plot_dir, "profiles_grid.png"))

    println("Integration endpoints:")
    for (name, ep) in zip(param_names, endpoints_integration)
        println("  $(name): left=$(ep.left), right=$(ep.right)")
    end

    println("\nOptimization endpoints:")
    for (name, ep) in zip(param_names, endpoints_optimization)
        println("  $(name): left=$(ep.left), right=$(ep.right)")
    end

    println("\n[Profiling] Done in $(round(time() - t_start; digits = 2)) s")

    return (
        optimization_problem = optprob,
        mle_vector = θ_best,
        integration_solution = prof_sol_ode,
        optimization_solution = prof_sol_opt,
        integration_endpoints = endpoints_integration,
        optimization_endpoints = endpoints_optimization,
    )
end