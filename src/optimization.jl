# src/optimization.jl

using PEtab
using Optim
using PyCall
using JLD2
using QuasiMonteCarlo
using Random

# This helper function for setting optimizer options remains useful
function get_optimizer_and_options(optimizer_name::Symbol, debug_mode::Bool)
    max_run_time = 3600.0  # 1 hour per run

    if optimizer_name === :Fides
        fides_alg = PEtab.Fides(:BFGS; verbose=false)
        
        # Fides options are passed as a Python dictionary
        fides_opts = py"{
            'maxiter': $(debug_mode ? 200 : 1000),
            'fatol': 1e-6,
            'frtol': 1e-8,
            'gtol': 1e-6,
            'maxtime': $(max_run_time)
        }"

        @info "Using Fides optimizer with BFGS hessian approximation."
        return fides_alg, fides_opts

    else
        # This section for Optim.jl solvers remains as a fallback
        optim_options = Optim.Options(
            iterations = debug_mode ? 200 : 1000,
            g_tol      = 1e-6,
            f_reltol   = 1e-8,
            time_limit = max_run_time,
            show_trace = false
        )
        if optimizer_name === :IPNewton
            @info "Using Optim.jl Optimizer: IPNewton"
            return Optim.IPNewton(), optim_options
        elseif optimizer_name === :LBFGS
            @info "Using Optim.jl Optimizer: LBFGS"
            return Optim.LBFGS(), optim_options
        else
            @error "Unknown optimizer: $optimizer_name. Defaulting to IPNewton."
            return Optim.IPNewton(), optim_options
        end
    end
end

# This is now the main function for a worker job. It runs ONE optimization.
function run_single_optimization(parsed_args, petab_problem)
    println("\n🧪 Running Single Parameter Estimation for Task ID: $(parsed_args["task-id"])")
    
    n_starts_total = parsed_args["n-starts"]
    task_id = parsed_args["task-id"]

    if !(1 <= task_id <= n_starts_total)
        @error "Invalid --task-id provided. Must be between 1 and $(n_starts_total)."
        return nothing
    end

    optimizer_alg, options = get_optimizer_and_options(Symbol(parsed_args["optimizer"]), parsed_args["debug"])

    # Generate all potential start-guesses but only select the one for this task
    # A seed ensures that every job generates the same list and picks its unique start
    Random.seed!(1234)
    x_starts_all = get_startguesses(petab_problem, n_starts_total; sampling_method=LatinHypercubeSample())
    x_start_this_job = x_starts_all[task_id]
    
    println("Starting optimization from guess #$(task_id)...")
    
    # Calibrate performs a single optimization run
    result = calibrate(petab_problem, x_start_this_job, optimizer_alg; options=options)

    # Save the result of this single run to its unique output file
    output_filename = parsed_args["output"]
    JLD2.save(output_filename, Dict("result" => result, "task_id" => task_id))
    
    println("✅ Task $(task_id) finished. Cost=$(result.fmin). Saved to $(output_filename).")
    return result
end
