# src/identifiability.jl

using LinearAlgebra
using ComponentArrays
using PEtab
using Printf
using DataFrames

export run_identifiability

function run_identifiability(petab_problem::PEtab.PEtabODEProblem, θ_full::ComponentVector)
    
    @info "--- Computing Fisher Information Matrix (FIM) using PEtab.jl's built-in method ---"
    
    param_values = collect(θ_full)
    est_syms = petab_problem.xnames
    
    # Use PEtab's dynamic parameter group instead of manual filtering
    mi = petab_problem.model_info
    dyn_names = mi.xindices.xids[:dynamic]                     # Symbols in PEtab order
    # Map dynamic names into the full xnames index space
    ix_dyn = [findfirst(==(n), petab_problem.xnames) for n in dyn_names]
    
    @info "Analyzing identifiability for $(length(dyn_names)) dynamic parameters."

    F_full = zeros(Float64, length(param_values), length(param_values))
    try
        petab_problem.FIM!(F_full, param_values)
    catch e
        @error "Failed to compute the Fisher Information Matrix. Error: $e"
        return
    end

    # Extract the sub-matrix corresponding to only the dynamic parameters
    F = F_full[ix_dyn, ix_dyn]

    try
        λ, U = eigen(Symmetric(F))
        λ_sorted = sort(λ, rev=true)
        U_sorted = U[:, sortperm(λ, rev=true)]

        λmax = λ_sorted[1]
        tol = 1e-4 * λmax
        rank_F = count(x -> x > tol, λ)
        null_dim = length(λ) - rank_F

        println("\n=== Identifiability diagnostics ===")
        println("FIM eigenvalues: min=$(round(minimum(λ), digits=6)), max=$(round(maximum(λ), digits=6))")
        println("Rank(F)=$(rank_F) of $(length(dyn_names)); null-space size=$(null_dim); tol=$(round(tol, digits=6))")
        
        if null_dim > 0
            println("⚠️  PARTIAL: $(null_dim) parameter combinations are not identifiable.")
            
            # --- START: NEW EIGENVECTOR ANALYSIS SECTION ---
            println("\n--- Eigenvectors of the Null-Space ---")
            println("These vectors show which parameter combinations are non-identifiable.")
            
            for i in 1:null_dim
                eigenvector = U_sorted[:, end-i+1]
                println("\n--- Non-identifiable Combination #$i (Eigenvalue: $(round(λ_sorted[end-i+1], digits=9))) ---")
                
                # Find the most significant components of the eigenvector
                components = []
                for j in 1:length(dyn_names)
                    # Show parameters with a significant contribution (e.g., >10% of the max component)
                    if abs(eigenvector[j]) > 0.1 * maximum(abs.(eigenvector))
                        push!(components, (string(dyn_names[j]), round(eigenvector[j], digits=3)))
                    end
                end
                
                # Sort by magnitude for readability
                sort!(components, by = x -> abs(x[2]), rev=true)
                
                for (pname, val) in components
                    @printf("%-30s: % .3f\n", pname, val)
                end
            end
            # --- END: NEW EIGENVECTOR ANALYSIS SECTION ---
        else
            println("✅ All dynamic parameters appear to be locally identifiable.")
        end

    catch e
        @error "Eigen-decomposition of the FIM failed. Error: $e"
    end

    return (FIM = F, names = dyn_names)
end