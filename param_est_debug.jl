# param_est_debug.jl

using Pkg
Pkg.activate("bngl_julia/")

using DifferentialEquations, PEtab, Sundials, Plots, SciMLSensitivity

include("src/model_param_est.jl")
const abstol = 1e-8
const reltol = 1e-8

println("Setting up PEtab problem...")
setup_results = setup_petab_problem()
if isnothing(setup_results)
    @error "Failed to build PEtabModel. Cannot proceed."
    exit()
end

println("Building PEtabODEProblem..."); flush(stdout)
petab_problem = PEtabODEProblem(setup_results.petab_model;
                                          gradient_method=:Adjoint,
                                          sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
                                          odesolver=ODESolver(CVODE_BDF(linear_solver=:GMRES), abstol=abstol, reltol=reltol),
                                          sparse_jacobian=false, # <-- The key fix
                                          verbose=false)

p_fail = [ -0.908656942, -0.498172882, -0.549168641, 0.916488746, -0.165407503, -1.538898833, -1.045878257, -1.28768169, -0.784087207, -1.514099478, -3.287843263, -1.514777518, -1.982146141, -1.199526012, -3.531398578, -0.394343395, -1.328942895, -0.285914876, -2.222941125, -1.51940869, 0.16638606, -0.239329628, 0.153219079, -0.157003261, -1.219552638, -1.447062883, -2.045574622, 0.518155018, -1.797766311, 0.7466577, -1.380497539, -0.075935401, -1.546681449, 0.635328518, 0.911325616, -0.61015559, 0.236037482, -0.856543728, -1.274746143, 1.402059166, -0.406045311, -0.644497925, -2.648930414, -1.163724591, 0.664432285, -0.456700104, -1.388620924, 0.054727726, -1.717058812, -0.917722031, 0.35312969, 0.207839448, -0.490527369, 0.341680173, -0.601350517, 0.108062533, -2.205999938, 0.886678766]
x_fail = ComponentArray(; zip(petab_problem.xnames, p_fail)...)

println("Attempting to compute the gradient with the parameter set...")
gradient_vector = similar(p_fail)
try
    @time petab_problem.grad!(gradient_vector, x_fail)
    println("\nSUCCESS: Gradient computation succeeded.")
    println("Gradient norm: ", norm(gradient_vector))

catch e
    println("\nSUCCESSFULLY REPRODUCED THE FAILURE!")
    println("Gradient computation failed with error:")
    showerror(stdout, e, catch_backtrace())
    println()
end