# precompile_workload.jl
module __PrecompileWorkload__
using Random
Random.seed!(1)

# Core imports that should succeed in the build env
using PEtab, OrdinaryDiffEq, Optim, ModelingToolkit, JLD2
using ComponentArrays, LinearAlgebra, SciMLSensitivity, Optimization, OptimizationOptimJL
using CSV, DataFrames, YAML, ArgParse, ReverseDiff, DiffEqCallbacks
using LikelihoodProfiler, QuasiMonteCarlo, ForwardDiff
using DataInterpolations

# Guard optional/fragile bits
try
    # Headless GR already set in the builder; this is a second line of defense
    get!(ENV, "GKSwstype", "100")
    @eval using Plots, Colors
    println("  ✓ Plots loaded successfully")
catch e
    @warn "Skipping Plots during precompile: $e"
end

try
    @eval using PyCall
    println("  ✓ PyCall loaded successfully")
catch e
    @warn "Skipping PyCall during precompile: $e"
end

println("  Including project source files at module level...")
project_path = get(ENV, "BNGL_JULIA_PROJECT_PATH", abspath(joinpath(@__DIR__, "..")))
include(joinpath(project_path, "src", "model_param_est_robustness.jl"))
include(joinpath(project_path, "src", "visualization.jl"))
include(joinpath(project_path, "src", "optimization.jl"))
include(joinpath(project_path, "src", "profiling.jl"))
println("  ✓ Project source files included.")

log_file_path = joinpath(get(ENV, "HOME", get(ENV, "USERPROFILE", ".")), "precompile_error.log")

function run_workload()
    # Declare variables that need to be available across try-catch blocks
    x0_param = nothing
    cost_val = Inf
    
    try
        println("--- Running PEtab-specific precompilation workload ---")
        println("  Project path: ", project_path)

        yaml_file = joinpath(project_path, "petab_problem.yml")
        isfile(yaml_file) || error("YAML file not found at $yaml_file")

        println("  Step 1: PEtabModel creation...")
        setup_results = setup_petab_problem(yaml_file)
        petab_model = setup_results.petab_model
        println("  ✓ Model compiled")

        println("  Step 2: PEtabODEProblem creation...")
        odesolver = ODESolver(KenCarp47(autodiff=false), abstol=1e-8, reltol=1e-8)
        steadystate_solver = SteadyStateSolver(:Simulate, abstol=1e-8, reltol=1e-8)
        petab_problem = PEtabODEProblem(
            petab_model;
            odesolver=odesolver,
            ss_solver=steadystate_solver,
            gradient_method=:ForwardDiff,
            sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP()),
            verbose=false,
        )
        println("  ✓ Problem compiled")

        println("  Step 3: Parameter init...")
        try
            x0 = get_startguesses(petab_problem, 1; sampling_method=LatinHypercubeSample())
            println("    Start guess type: $(typeof(x0)), length: $(length(x0))")
            
            # Get the first parameter vector and ensure it's the right format
            x0_param = if length(x0) > 0
                first_guess = x0[1]
                println("    First guess type: $(typeof(first_guess))")
                first_guess
            else
                # Fallback to nominal parameters
                println("    Using nominal parameters as fallback")
                collect(petab_problem.lower_bounds .+ 0.5 .* (petab_problem.upper_bounds .- petab_problem.lower_bounds))
            end
            
            # Test cost function evaluation
            cost_val = petab_problem.nllh(x0_param; prior=false)
            println("  ✓ Cost compiled (value: $(cost_val))")
        catch e
            @warn "Parameter initialization failed, using minimal test: $(typeof(e)): $(e)"
            # Ultra-minimal test with nominal bounds
            n_params = length(petab_problem.lower_bounds)
            x0_param = collect(petab_problem.lower_bounds .+ 0.1 .* (petab_problem.upper_bounds .- petab_problem.lower_bounds))
            cost_val = petab_problem.nllh(x0_param; prior=false)
            println("  ✓ Minimal cost test compiled")
        end

        println("  Step 4: Lightweight optimization...")
        try
            optimizer_alg, options = get_optimizer_and_options(:LBFGS, true)
            options = merge(options, Dict(:maxiters => 2, :maxtime => 5.0))
            result = calibrate(petab_problem, x0_param, optimizer_alg; options=options)
            println("  ✓ Optimization compiled")
        catch e
            @warn "Optimization failed during precompile: $(typeof(e)): $(e)"
            # Create a dummy result for the remaining steps
            result = (xmin = x0_param, fmin = cost_val)
            println("  ✓ Optimization compilation skipped")
        end

        println("  Step 5: Visualization solve (guarded)...")
        try
            PEtab.solve_all_conditions(
                result.xmin, petab_problem, odesolver.solver;
                abstol=odesolver.abstol, reltol=odesolver.reltol,
            )
            println("  ✓ Visualization solve compiled")
        catch e
            @warn "Skipping visualization solve during precompile: $e"
        end

        println("  Step 6: Profiling helper...")
        try
            θ_mle = ComponentArray(result.xmin)
            all_names = string.(keys(θ_mle))
            params_to_profile = [n for n in all_names if !startswith(n, "sigma") && !endswith(n, "_0")]
            isempty(params_to_profile) || println("  ✓ Profiling filter compiled")
        catch e
            println("  ⚠️ Profiling precompile skipped: $(typeof(e))")
        end

        println("  Step 7: I/O smoke tests...")
        temp_file = "temp_precompile_test.jld2"
        JLD2.jldsave(temp_file; result)
        rm(temp_file; force=true)
        temp_csv = "temp_precompile_test.csv"
        CSV.write(temp_csv, DataFrame(param=["test"], value=[1.0]))
        CSV.read(temp_csv, DataFrame)
        rm(temp_csv; force=true)
        println("  ✓ I/O compiled")

        println("✅ Complete PEtab workflow precompilation successful!")
    catch e
        println("⚠️  FATAL ERROR in precompilation workload. Writing details to log file.")
        open(log_file_path, "w") do f
            println(f, "Precompilation script failed.")
            println(f, "Error Type: ", typeof(e))
            println(f, "Error Message: ", sprint(showerror, e))
            println(f, "\n--- Stacktrace ---")
            Base.showerror(f, e, catch_backtrace())
        end
        println("   Log file created at: ", log_file_path)
        rethrow(e)
    end
end

end # module

__PrecompileWorkload__.run_workload()