# src/optimization.jl

using PEtab
using Optim
using PyCall
using JLD2
using QuasiMonteCarlo
using Random

# This helper function is still perfect, no changes needed.
function get_optimizer_and_options(optimizer_name::Symbol, debug_mode::Bool)
    max_run_time = 3600.0  # 1 hour per run

    if optimizer_name === :Fides
        fides_alg = PEtab.Fides(:BFGS; verbose=false)
        fides_opts = py"{
            'maxiter': $(debug_mode ? 200 : 1000),
            'fatol': 1e-6, 'frtol': 1e-8, 'gtol': 1e-6,
            'maxtime': $(max_run_time)
        }"
        @info "Using Fides optimizer with BFGS hessian approximation."
        return fides_alg, fides_opts
    else
        # Fallback for Optim.jl solvers
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


# NEW function to run a batch of optimizations in parallel
function run_batch_optimization(parsed_args, petab_problem)
    # 1. Extract batch information from arguments
    n_starts_total = parsed_args["n-starts"]
    n_batches = parsed_args["n-batches"]
    batch_id = parsed_args["batch-id"]
    n_procs = parsed_args["n-procs"]

    @info "Starting Batch #$batch_id of $n_batches"

    # 2. Determine which slice of start-guesses this batch is responsible for
    starts_per_batch = ceil(Int, n_starts_total / n_batches)
    start_index = (batch_id - 1) * starts_per_batch + 1
    end_index = min(batch_id * starts_per_batch, n_starts_total)
    n_starts_this_batch = end_index - start_index + 1

    if n_starts_this_batch <= 0
        @warn "Batch #$batch_id has no starts to run. Exiting."
        return
    end
    @info "This batch will run $n_starts_this_batch optimizations (indices $start_index to $end_index)."

    # 3. Generate ALL start guesses but only use the slice for this batch
    # The fixed seed ensures every batch job generates the same master list
    Random.seed!(1234)
    x_starts_all = get_startguesses(petab_problem, n_starts_total; sampling_method=LatinHypercubeSample())
    x_starts_this_batch = x_starts_all[start_index:end_index]

    # 4. Get optimizer settings
    optimizer_alg, options = get_optimizer_and_options(Symbol(parsed_args["optimizer"]), parsed_args["debug"])

    # 5. Define where to save intermediate results for this batch
    dir_save = joinpath("results", "batch_$(batch_id)")
    mkpath(dir_save)
    @info "Intermediate results for this batch will be saved in: $dir_save"

    # 6. Run the multi-start optimization for this batch in parallel
    @info "Launching calibrate_multistart with nprocs = $n_procs..."
    batch_result = calibrate_multistart(
        petab_problem,
        optimizer_alg,
        x_starts_this_batch; # Pass the specific list of start guesses
        nprocs=n_procs,
        dirsave=dir_save,
        options=options
    )

    @info "✅ Batch #$batch_id finished. Best cost in this batch: $(batch_result.fmin)."
end
