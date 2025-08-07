# src/profiling.jl

using LikelihoodProfiler
using Optimization
using OptimizationOptimJL
using ComponentArrays
using Plots
using JLD2
using Statistics
using PEtab

# ==============================================================================
# METHOD 1: Modern, AD-based Profiling (Thread-Safe Version)
# ==============================================================================
function run_modern_likelihood_profiling(
    petab_model::PEtabModel, # Pass the base model
    odesolver,
    steadystate_solver,
    θ_mle::ComponentVector,
    debug_mode::Bool
)
    println("\n--- 🔬 Attempting Thread-Safe Modern Likelihood Profiling ---")

    profile_dir = joinpath(pwd(), "likelihood_profiles")
    mkpath(profile_dir)

    param_names = string.(keys(θ_mle))
    n_params = length(param_names)
    prof_sols = Vector{Any}(undef, n_params)

    println("Starting parallel profiling across $n_params parameters...")
    Threads.@threads for i in 1:n_params
        println("Thread $(threadid()) starting parameter $i ($(param_names[i]))")
        
        # --- KEY CHANGE: THREAD-LOCAL PROBLEM CREATION ---
        # Each thread builds its own PEtabODEProblem to avoid race conditions.
        local petab_problem_local = PEtabODEProblem(
            petab_model,
            odesolver=odesolver,
            ss_solver=steadystate_solver,
            verbose=false
        )

        function local_objective(θ_est, p_not_used)
            return petab_problem_local.nllh(θ_est; prior=false)
        end

        local_optprob = OptimizationProblem(
            OptimizationFunction(local_objective, Optimization.AutoForwardDiff()),
            collect(θ_mle);
            lb = collect(petab_problem_local.lower_bounds),
            ub = collect(petab_problem_local.upper_bounds)
        )

        local_plprob = LikelihoodProfiler.PLProblem(local_optprob, collect(θ_mle))
        
        profiler = LikelihoodProfiler.OptimizationProfiler(
            optimizer = OptimizationOptimJL.LBFGS(),
            stepper = LikelihoodProfiler.FixedStep(initial_step = 0.1)
        )
        
        single_prof_sol = LikelihoodProfiler.profile(local_plprob, profiler; idxs=[i], maxiters=100)
        prof_sols[i] = single_prof_sol
        println("✅ Thread $(threadid()) completed parameter $i")
    end

    println("\n--- Generating Modern Profile Plots ---")
    for i in 1:n_params
        param_name = param_names[i]
        
        # Check if the profile solution for this parameter exists and is valid
        if isassigned(prof_sols, i) && !isnothing(prof_sols[i])
            try
                plt = plot(prof_sols[i], 1; # The '1' is needed as each result has only one profile
                           xlabel = "$param_name (log₁₀ scale)",
                           ylabel = "Δ Log-Likelihood",
                           title = "Profile Likelihood: $param_name",
                           legend = :topright,
                           linewidth = 2,
                           ylims = (0, 15) # Zoom in on the relevant confidence region
                          )
                
                # Add confidence interval lines
                hline!(plt, [1.92], label="95% CI", color=:red, linestyle=:dash)
                hline!(plt, [3.84], label="99% CI", color=:orange, linestyle=:dash)
                
                savefig(plt, joinpath(profile_dir, "profile_$(param_name).png"))

            catch e
                @warn "Could not generate plot for parameter '$param_name'. Error: $e"
            end
        else
            @warn "No valid profile solution found for parameter '$param_name'. Skipping plot."
        end
    end
    println("✅ Modern profile plots saved.")

    println("✅ Modern profiling computation complete.")
    return prof_sols
end

# ==============================================================================
# METHOD 2: Manual, Grid-based Profiling (Robust Fallback)
# ==============================================================================
function run_manual_likelihood_profiling(petab_problem::PEtabODEProblem, θ_mle::ComponentVector, debug_mode::Bool)
    println("\n--- 🔬 Running Robust Manual Likelihood Profiling (Fallback) ---")
    
    profile_dir = joinpath(pwd(), "likelihood_profiles_manual")
    mkpath(profile_dir)

    param_names = string.(petab_problem.xnames)
    n_params = length(param_names)
    θ_mle_vec = collect(θ_mle)
    
    mle_nllh = petab_problem.nllh(θ_mle_vec; prior=false)
    println("  MLE negative log-likelihood: $mle_nllh")
    
    profiles = Dict()
    
    for (i, param_name) in enumerate(param_names)
        println("Manually profiling parameter $i: $param_name")
        
        lb = petab_problem.lower_bounds[i]
        ub = petab_problem.upper_bounds[i]
        
        n_points = debug_mode ? 20 : 50
        param_range = range(lb, ub, length=n_points)
        
        likelihood_values = Float64[]
        
        for param_val in param_range
            θ_test = copy(θ_mle_vec)
            θ_test[i] = param_val
            
            try
                nllh_val = petab_problem.nllh(θ_test; prior=false)
                push!(likelihood_values, nllh_val - mle_nllh)
            catch e
                push!(likelihood_values, Inf)
            end
        end
        
        profiles[param_name] = (param_range=collect(param_range), likelihood=likelihood_values)
        
        # Plotting
        valid_indices = .!isinf.(likelihood_values)
        if any(valid_indices)
            plt = plot(param_range[valid_indices], likelihood_values[valid_indices],
                      xlabel="$param_name (log₁₀ scale)",
                      ylabel="Δ Log-Likelihood",
                      title="Manual Profile: $param_name",
                      linewidth=2, legend=false, ylims=(0, 15))
            hline!(plt, [1.92], color=:red, linestyle=:dash)
            hline!(plt, [3.84], color=:orange, linestyle=:dash)
            savefig(plt, joinpath(profile_dir, "profile_$(param_name).png"))
        end
    end
    
    println("✅ Manual profiling completed.")
    return profiles
end

# ==============================================================================
# MAIN WRAPPER FUNCTION WITH FALLBACK LOGIC (Updated Call)
# ==============================================================================
function run_likelihood_profiling_with_fallback(petab_problem::PEtabODEProblem, petab_model, profiling_odesol, profiling_steadystate_solver, θ_mle::ComponentVector, debug_mode::Bool)
    
    try
        @info "Attempting modern, thread-safe likelihood profiling..."
        
        # Call the updated modern profiler with the necessary ingredients
        prof_result = run_modern_likelihood_profiling(
            petab_model, 
            profiling_odesol, 
            profiling_steadystate_solver, 
            θ_mle, 
            debug_mode
        )
        return prof_result

    catch e
        @warn "Modern likelihood profiling failed."
        @warn "Error was: $e"
        @info "Switching to robust manual profiling as a fallback..."
        
        # The manual method is not parallel and can use the pre-built problem
        prof_result = run_manual_likelihood_profiling(petab_problem, θ_mle, debug_mode)
        return prof_result
    end
end

# Export the main wrapper function
export run_likelihood_profiling_with_fallback