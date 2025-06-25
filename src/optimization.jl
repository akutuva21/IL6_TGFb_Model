using PEtab
using Optim
using JLD2
using ComponentArrays
using Sundials

# This dictionary is specific to the optimization process
const SUPPORTED_OPTIMIZERS = Dict(
    "LBFGS" => LBFGS(),
    "IPNewton" => IPNewton()
)

function run_parameter_estimation(parsed_args, petab_problem)
    use_parallel = parsed_args["parallel"]
    optimizer_choice_str = parsed_args["optimizer"]
    n_starts = parsed_args["n-starts"]
    if n_starts == 0
        n_starts = use_parallel ? length(procs()) : 1
    end

    if !haskey(SUPPORTED_OPTIMIZERS, optimizer_choice_str)
        @error "Unsupported optimizer '$optimizer_choice_str'."
        return nothing
    end
    optimizer = SUPPORTED_OPTIMIZERS[optimizer_choice_str]
    optim_options = Optim.Options(time_limit=600.0)

    println("\n[Timing] Calibrating parameters..."); flush(stdout)
    
    if use_parallel
        println("Mode: PARALLEL using $(length(procs())) processes, $n_starts starts, and optimizer $optimizer_choice_str")
        multi_start_res = calibrate_multistart(petab_problem, optimizer, n_starts;
                                               nprocs=length(procs()),
                                               dirsave=joinpath(pwd(), "Intermediate_results"),
                                               options=optim_options)
        return multi_start_res
    else
        println("Mode: SERIAL with optimizer $optimizer_choice_str, $n_starts start(s)")
        _start_guesses_raw = get_startguesses(petab_problem, n_starts)
        start_guesses = (n_starts == 1) ? [_start_guesses_raw] : _start_guesses_raw
        
        all_runs = PEtabOptimisationResult[]
        for (i, x0) in enumerate(start_guesses)
            println("  Serial Start $i/$n_starts...")
            try
                res = calibrate(petab_problem, x0, optimizer; options=optim_options)
                push!(all_runs, res)
            catch e
                @error "Calibration for start $i failed! Skipping this start." exception=(e, catch_backtrace())
            end
        end

        if isempty(all_runs)
            @error "No optimization runs completed successfully."
            return nothing
        end

        valid_runs = filter(r -> !isnothing(r) && isfinite(r.fmin), all_runs)
        if isempty(valid_runs)
            @error "All optimization starts failed to produce a valid solution."
            return nothing
        end
        best_res = valid_runs[argmin([r.fmin for r in valid_runs])]

        return PEtabMultistartResult(best_res.xmin, best_res.fmin, best_res.alg, n_starts, 
                                     "LatinHypercubeSample", nothing, all_runs)
    end
end
