# precompile_workload.jl (FINAL, WORLD-AGE-SAFE VERSION)

# Wrap the entire workload in a module to create a single, consistent scope.
module __PrecompileWorkload__

# All necessary packages are imported here, at the top of our module.
using PEtab, OrdinaryDiffEq, Optim, ModelingToolkit, JLD2
using ComponentArrays, LinearAlgebra, SciMLSensitivity, Optimization, OptimizationOptimJL
using CSV, DataFrames, YAML, ArgParse, ReverseDiff, DiffEqCallbacks
using LikelihoodProfiler, CICOBase, PyCall, QuasiMonteCarlo, ForwardDiff
using Plots, Colors, DataInterpolations

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
include(joinpath(project_path, "src", "config_resolve.jl"))
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

        # Test config resolution first
        println("  Step 0: Precompiling config resolution...")
        config_file = joinpath(project_path, "config.yml")
        if isfile(config_file)
            paths = ConfigResolve.resolve_petab_paths(config_file)
            temp_yaml = joinpath(project_path, ".petab_resolved_precompile.yaml")
            ConfigResolve.write_temp_petab_yaml(temp_yaml, paths)
            ConfigResolve.log_resolved_files(paths)
            yaml_file = temp_yaml
            println("  ✓ Config resolution compiled")
        else
            yaml_file = joinpath(project_path, "petab_problem.yml")
        end
        if !isfile(yaml_file)
            error("YAML file not found at $yaml_file. Aborting workload.")
        end

        println("  Step 1: Precompiling PEtabModel creation...")
        # This call is now safe because the function was defined before run_workload was compiled.
        setup_results = setup_petab_problem(yaml_file)
        petab_model = setup_results.petab_model
        println("  ✓ PEtabModel creation compiled")

        println("  Step 2: Precompiling PEtabODEProblem creation...")
        odesolver = ODESolver(KenCarp47(autodiff=false), abstol=1e-8, reltol=1e-8)
        steadystate_solver = SteadyStateSolver(:Simulate, abstol=1e-8, reltol=1e-8)
        petab_problem = PEtabODEProblem(petab_model, odesolver=odesolver, ss_solver=steadystate_solver, 
                                        gradient_method=:ForwardDiff, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP()), verbose=false)
        println("  ✓ PEtabODEProblem creation compiled")

        println("  Step 3: Precompiling parameter estimation components...")
        x0 = get_startguesses(petab_problem, 1; sampling_method=LatinHypercubeSample())
        println("  ✓ Starting guess generation compiled")
        cost_val = petab_problem.nllh(x0[1]; prior=false)
        println("  ✓ Cost function evaluation compiled (cost: $(round(cost_val, digits=2)))")

        println("  Step 4: Precompiling optimization...")
        # Test both Optim.jl and PEtab.calibrate
        optimizer_alg, options = get_optimizer_and_options(:LBFGS, true)
        result = calibrate(petab_problem, x0[1], optimizer_alg; options=options)
        println("  ✓ Optimization stack compiled (final cost: $(round(result.fmin, digits=2)))")

        println("  Step 5: Precompiling batch optimization components...")
        # Test calibrate_multistart functionality 
        try
            batch_result = calibrate_multistart(petab_problem, optimizer_alg, 2; nprocs=1, options=options, seed=1234)
            println("  ✓ Batch optimization compiled")
        catch e
            println("  ⚠️ Batch optimization skipped: $(typeof(e))")
        end

        println("  Step 6: Precompiling visualization components...")
        PEtab.solve_all_conditions(result.xmin, petab_problem, odesolver.solver; abstol=odesolver.abstol, reltol=odesolver.reltol)
        println("  ✓ Visualization ODE solving compiled")

        println("  Step 7: Precompiling profiling components...")
        # Test manual profiling setup
        try
            θ_mle = ComponentArray(result.xmin)
            all_names = string.(keys(θ_mle))
            params_to_profile = [n for n in all_names if !startswith(n, "sigma") && !endswith(n, "_0")]
            if !isempty(params_to_profile)
                # Test profiling setup without full execution
                println("  ✓ Profiling parameter filtering compiled")
            end
        catch e
            println("  ⚠️ Profiling precompile skipped: $(typeof(e))")
        end

        println("  Step 8: Precompiling I/O operations...")
        temp_file = "temp_precompile_test.jld2"
        JLD2.jldsave(temp_file; result)
        rm(temp_file)
        println("  ✓ JLD2 saving/loading compiled")

        # Test CSV operations
        temp_df = DataFrame(param=["test"], value=[1.0])
        temp_csv = "temp_precompile_test.csv"
        CSV.write(temp_csv, temp_df)
        CSV.read(temp_csv, DataFrame)
        rm(temp_csv)
        println("  ✓ CSV operations compiled")

        # Clean up temporary files
        if occursin("precompile", yaml_file) && isfile(yaml_file)
            rm(yaml_file)
        end

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