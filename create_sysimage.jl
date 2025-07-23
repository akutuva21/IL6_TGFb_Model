# create_sysimage.jl (The Correct, Isolated Build Environment Solution)
using Pkg, PackageCompiler

# 1. Define the final, desired list of packages. JuliaFormatter is NOT on this list.
final_pkgs = [
    "DifferentialEquations", "OrdinaryDiffEq", "Sundials", "SciMLBase",
    "SciMLSensitivity", "DiffEqCallbacks", "ModelingToolkit", "Catalyst",
    "Symbolics", "SymbolicUtils", "ReactionNetworkImporters", "PEtab",
    "Optimization", "OptimizationOptimJL", "Optim", "ADTypes",
    "LikelihoodProfiler", "ReverseDiff", "DataFrames", "CSV", "JLD2",
    "XLSX", "YAML", "Plots", "Colors", "RecipesBase", "ComponentArrays",
    "ArgParse", "PackageCompiler"
]
unique!(sort!(final_pkgs))

println("📦 System image will be built with these $(length(final_pkgs)) packages.")

# 2. Create a temporary, clean project directory.
tmp_project_dir = mktempdir()
println("\nCreating a clean build environment at: ", tmp_project_dir)

try
    # 3. Activate the clean environment and add ONLY the desired packages.
    # This creates a new, clean Manifest.toml that will not contain JuliaFormatter.
    Pkg.activate(tmp_project_dir)
    println("Adding packages to the clean environment...")
    Pkg.add(final_pkgs)

    # 4. Build the system image using the clean environment.
    mkpath("SysImage")
    sysimage_path = joinpath("SysImage", "bngl_full.so")
    println("\n🛠 Creating FULL system image from the clean environment...")
    println("📍 Output: $sysimage_path")

    create_sysimage(
        final_pkgs;
        sysimage_path  = sysimage_path,
        project        = tmp_project_dir, # This is the crucial instruction
        incremental    = false
    )

    println("\n✅ Full system image created successfully.")
finally
    # 5. Clean up the temporary directory.
    println("Cleaning up the temporary build environment...")
    rm(tmp_project_dir; force=true, recursive=true)
end

println("\nDone.")