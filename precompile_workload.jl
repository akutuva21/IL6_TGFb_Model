# precompile_workload.jl (FINAL, WORLD-AGE-SAFE VERSION)

# Wrap the entire workload in a module to create a single, consistent scope.
module __PrecompileWorkload__

# All necessary packages are imported here, at the top of our module.
using PEtab, OrdinaryDiffEq, Optim, ModelingToolkit, JLD2
using ComponentArrays, LinearAlgebra, SciMLSensitivity, Optimization, OptimizationOptimJL

# --- START OF FIX ---
# Get the project path and include source files at the module's top level.
# This ensures that all functions (like `setup_petab_problem`) are defined
# BEFORE the `run_workload` function is compiled, solving the world-age issue.
println("  Including project source files at the module level...")
project_path = ENV["BNGL_JULIA_PROJECT_PATH"]
include(joinpath(project_path, "src", "model_param_est_robustness.jl"))
include(joinpath(project_path, "src", "visualization.jl"))
include(joinpath(project_path, "src", "optimization.jl"))
include(joinpath(project_path, "src", "profiling.jl"))
println("  ✓ Project source files included.")
# --- END OF FIX ---

# Keep the reliable error logging for debugging on the cluster.
log_file_path = joinpath(get(ENV, "HOME", get(ENV, "USERPROFILE", ".")), "precompile_error.log")

# Now, define the main function to contain the logic.
# It will inherit all the included functions from the module's scope.
function run_workload()
    try
        println("--- Running PEtab-specific precompilation workload ---")

        println("  Project path found: ", project_path)

        yaml_file = joinpath(project_path, "petab_problem.yml")
        if !isfile(yaml_file)
            error("YAML file not found at $yaml_file. Aborting workload.")
        end

        println("  Step 1: Precompiling PEtabModel creation...")
        # This call is now safe because the function was defined before run_workload was compiled.
        setup_results = setup_petab_problem(yaml_file)
        petab_model = setup_results.petab_model
        println("  ✓ PEtabModel creation compiled")

        println("  Step 2: Precompiling PEtabODEProblem creation...")
        odesolver = ODESolver(Rodas5P(), abstol=1e-10, reltol=1e-10)
        steadystate_solver = SteadyStateSolver(:Simulate, abstol=1e-10, reltol=1e-10)
        petab_problem = PEtabODEProblem(petab_model, odesolver=odesolver, ss_solver=steadystate_solver, gradient_method=:ForwardDiff, verbose=false)
        println("  ✓ PEtabODEProblem creation compiled")

        println("  Step 3: Precompiling parameter estimation components...")
        x0 = get_startguesses(petab_problem, 1)
        println("  ✓ Starting guess generation compiled")
        cost_val = petab_problem.nllh(x0; prior=false)
        println("  ✓ Cost function evaluation compiled (cost: $(round(cost_val, digits=2)))")

        println("  Step 4: Precompiling optimization...")
        result = calibrate(petab_problem, x0, LBFGS(); options=Optim.Options(iterations=2))
        println("  ✓ Optimization stack compiled (final cost: $(round(result.fmin, digits=2)))")

        println("  Step 5: Precompiling visualization components...")
        PEtab.solve_all_conditions(result.xmin, petab_problem, odesolver.solver; abstol=odesolver.abstol, reltol=odesolver.reltol)
        println("  ✓ Visualization ODE solving compiled")

        temp_file = "temp_precompile_test.jld2"
        JLD2.jldsave(temp_file; result)
        rm(temp_file)
        println("  ✓ JLD2 saving/loading compiled")

        println("✅ Complete PEtab workflow precompilation successful!")
        println("--- Precompilation workload finished ---")

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

end # end of __PrecompileWorkload__ module

# Execute the workload by calling the main function.
__PrecompileWorkload__.run_workload()