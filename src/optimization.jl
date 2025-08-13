# src/optimization.jl

using PEtab
using Optim
using PyCall
using Base.Threads
using CSV
using DataFrames
using QuasiMonteCarlo

"""
    get_optimizer_and_options(optimizer_name::Symbol, debug_mode::Bool)

Returns a tuple containing the optimizer algorithm object and its corresponding options dictionary.
This function ensures all return values are consistent.
"""
function get_optimizer_and_options(optimizer_name::Symbol, debug_mode::Bool)
    
    if optimizer_name === :Fides
        # 1. Correctly create the Fides algorithm object.
        #    The Hessian update strategy is an argument to the constructor.
        fides_alg = PEtab.Fides(:BFGS; verbose=false)
        
        # 2. Options must be a Python dictionary. Do not put "hessian_update" here.
        fides_opts = py"{
            'maxiter': $(debug_mode ? 150 : 500),
            'fatol': 1e-5,
            'frtol': 1e-7,
            'gtol': 1e-6
        }"

        println("Using Fides via PEtab.jl's built-in wrapper with BFGS updates.")
        return fides_alg, fides_opts

    else
        # This part for Optim.jl solvers is correct and needs no changes.
        optim_options = Optim.Options(
            iterations = debug_mode ? 200 : 800,
            g_tol      = 1e-6,
            f_reltol   = debug_mode ? 1e-6 : 1e-8,
            show_trace = false
        )

        if optimizer_name === :IPNewton
            println("Using IPNewton: Robust interior-point Newton (Julia native)")
            return Optim.IPNewton(), optim_options
        elseif optimizer_name === :LBFGS
            println("Using LBFGS: Reliable quasi-Newton, memory efficient")
            return Optim.LBFGS(), optim_options
        elseif optimizer_name === :BFGS
            println("Using BFGS: Fast quasi-Newton for medium problems")
            return Optim.BFGS(), optim_options
        else
            @error "Unknown optimizer: $optimizer_name"
            throw(ArgumentError("Invalid optimizer specified."))
        end
    end
end

function calibrate_multistart_threaded(prob::PEtabODEProblem, alg, nmultistarts::Integer; 
                                     dirsave=nothing, 
                                     sampling_method=LatinHypercubeSample(),
                                     sample_prior::Bool=true,
                                     save_trace::Bool=false,
                                     seed=nothing,
                                     options=nothing)::PEtabMultistartResult
    
    # Set up paths for saving intermediate results (same logic as original)
    paths_save = Dict{Symbol, String}()
    if !isnothing(dirsave)
        !isdir(dirsave) && mkpath(dirsave)
        i = 1
        while true
            path_x0 = joinpath(dirsave, "startguesses$i.csv")
            !isfile(path_x0) && break
            i += 1
        end
        paths_save[:x0] = joinpath(dirsave, "startguesses" * string(i) * ".csv")
        paths_save[:res] = joinpath(dirsave, "results" * string(i) * ".csv")
        paths_save[:xmin] = joinpath(dirsave, "xmins" * string(i) * ".csv")
        if save_trace == true
            paths_save[:trace] = joinpath(dirsave, "trace" * string(i) * ".csv")
        end
    end

    # Generate starting guesses
    if !isnothing(seed)
        Random.seed!(seed)
    end
    xstarts = get_startguesses(prob, nmultistarts; sampling_method = sampling_method,
                               sample_prior = sample_prior)
    
    # Save starting guesses to file
    if !isempty(paths_save)
        xnames = propertynames(xstarts[1]) |> collect
        xstarts_df = DataFrame(vcat(reduce(vcat, xstarts')), xnames)
        xstarts_df[!, "startguess"] = 1:nrow(xstarts_df)
        CSV.write(paths_save[:x0], xstarts_df)
    end

    # Use ReentrantLock instead of RemoteChannel for thread synchronization
    mutex = ReentrantLock()
    
    # Preallocate results vector
    runs = Vector{Union{Nothing, PEtabOptimisationResult}}(undef, nmultistarts)

    # Run calibrations in parallel using threads instead of processes
    Threads.@threads for i in 1:nmultistarts
        runs[i] = _calibrate_startguess_threaded(xstarts[i], i, prob, alg, save_trace,
                                               options, paths_save, mutex)
    end

    # Filter out failed runs and find best result
    valid_runs = filter(!isnothing, runs)
    if isempty(valid_runs)
        error("All optimization runs failed")
    end

    bestrun = valid_runs[argmin([isnan(r.fmin) ? Inf : r.fmin for r in valid_runs])]
    fmin = bestrun.fmin
    xmin = bestrun.xmin
    
    # Format sampling method string (same as original)
    sampling_method_str = string(sampling_method)[1:findfirst(x -> x == '(',
                                                              string(sampling_method))][1:(end - 1)]
    
    return PEtabMultistartResult(xmin, fmin, bestrun.alg, nmultistarts, sampling_method_str,
                                 dirsave, runs)
end

function _calibrate_startguess_threaded(xstart, i, prob::PEtabODEProblem, alg, save_trace::Bool,
                                       options, paths_save, mutex::ReentrantLock)
    if !isempty(xstart)
        try
            res = calibrate(prob, xstart, alg; save_trace = save_trace, options = options)
        catch e
            @warn "Calibration failed for start $i: $e"
            return nothing
        end
    else
        # Handle edge case where no parameters to estimate
        xstart, xmin = ComponentArray{Float64}(), ComponentArray{Float64}()
        xtrace, ftrace = Vector{Vector{Float64}}(undef, 0), Vector{Float64}(undef, 0)
        fmin = prob.nllh(xstart)
        res = PEtabOptimisationResult(xmin, fmin, xstart, :alg, 0, 0.0, xtrace, ftrace,
                                      true, nothing)
    end
    
    # Thread-safe saving of intermediate results
    if !isempty(paths_save)
        lock(mutex) do
            _save_multistart_results_threaded(paths_save, res, i)
        end
    end
    return res
end

function _save_multistart_results_threaded(paths_save::Dict{Symbol, String},
                                          res::PEtabOptimisationResult, i::Int64)::Nothing
    xnames = propertynames(res.xmin) |> collect
    res_df = DataFrame(fmin = res.fmin, alg = res.alg, runtime = res.runtime,
                       niterations = res.niterations, converged = res.converged,
                       startguess = i)
    x_df = DataFrame(Matrix(res.xmin'), xnames)
    x_df[!, "startguess"] = [i]
    
    CSV.write(paths_save[:res], res_df, append = isfile(paths_save[:res]))
    CSV.write(paths_save[:xmin], x_df, append = isfile(paths_save[:xmin]))
    
    if haskey(paths_save, :trace) && !isnothing(res.ftrace) && !isempty(res.ftrace)
        trace_df = DataFrame(Matrix(reduce(vcat, res.xtrace')), xnames)
        trace_df[!, "ftrace"] = res.ftrace
        trace_df[!, "startguess"] = repeat([i], length(res.ftrace))
        CSV.write(paths_save[:trace], trace_df, append = isfile(paths_save[:trace]))
    end
    return nothing
end


"""
Run the multi-start parameter estimation using PEtab.jl's built-in robust functionality.
"""
function run_parameter_estimation(parsed_args, petab_problem)
    println("\n🧪 SCIENTIFIC PARAMETER ESTIMATION")
    println("="^70)

    # Step 0: Check parameter setup
    println("\n📋 Parameter Setup Validation")
    println("Number of parameters to estimate: $(petab_problem.nparameters_estimate)")
    
    if petab_problem.nparameters_estimate == 0
        @error "❌ No parameters marked for estimation! Check your parameter file."
        @error "Make sure some parameters have 'estimate = 1' in the parameters table."
        return nothing
    end
    
    println("✅ Found $(petab_problem.nparameters_estimate) parameters to estimate")
    println("Parameter names: $(petab_problem.xnames)")

    # Step 1: Configure optimization strategy
    println("\n⚙️  Step 1: Optimization Strategy Configuration")
    
    debug_mode = get(parsed_args, "debug", false)
    # Convert string from command line to Symbol for our function
    optimizer_name = Symbol(get(parsed_args, "optimizer", "Fides")) 
    n_starts = get(parsed_args, "n-starts", min(Threads.nthreads(), petab_problem.nparameters_estimate * 2))
    
    # This call is now robust and returns a consistent two-part tuple
    optimizer_alg, options = get_optimizer_and_options(optimizer_name, debug_mode)
    
    # Step 2: Execute robust multi-start parameter estimation
    println("\n🚀 Step 2: Multi-Start Parameter Estimation")
    println("Configuration:")
    println("  • Optimizer: $(typeof(optimizer_alg))")
    println("  • Multi-starts: $n_starts")
    println("  • Threading: CUSTOM multithreaded implementation")
    println("  • Available threads: $(Threads.nthreads())")
    
    start_time = time()
    multi_start_res = nothing

    try
        multi_start_res = calibrate_multistart_threaded(
            petab_problem,
            optimizer_alg,
            n_starts;
            dirsave="Intermediate_results",
            options=options
        )
    catch e
        @error "Multi-start parameter estimation failed: $e"
        showerror(stdout, e, catch_backtrace())
        return nothing
    end

    total_elapsed = time() - start_time
    
    if isnothing(multi_start_res) || (hasproperty(multi_start_res, :runs) && isempty(multi_start_res.runs))
        @error "All optimization attempts failed. Check model stability and parameter bounds."
        return nothing
    end

    println("\n✅ Multi-start estimation completed!")
    println("   • Total time: $(round(total_elapsed/60, digits=1)) minutes")
    println("   • Successful runs: $(length(multi_start_res.runs))/$n_starts")
    println("   • Best cost: $(round(multi_start_res.fmin, digits=3))")
    
    return multi_start_res
end
