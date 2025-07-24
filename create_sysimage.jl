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

    # Set an environment variable to tell the workload script where the project root is.
    project_path = abspath(@__DIR__)
    ENV["BNGL_JULIA_PROJECT_PATH"] = project_path
    println("🗂️  Setting BNGL_JULIA_PROJECT_PATH = $project_path")

    # 4. Build the system image using the clean environment AND the workload script.
    mkpath("SysImage")
    sysimage_path = joinpath("SysImage", "bngl_full.so")
    println("\n🛠 Creating FULL system image from the clean environment...")
    println("📍 Output: $sysimage_path")
    println("🚀 Including precompilation workload to eliminate 'time-to-first-X' costs...")

    create_sysimage(
        final_pkgs;
        sysimage_path  = sysimage_path,
        project        = tmp_project_dir, # This is the crucial instruction
        precompile_execution_file = joinpath(project_path, "precompile_workload.jl"), # Execute workload during build
        incremental    = false,
        cpu_target     = "x86-64-v2"
    )

    println("\n✅ Full system image created successfully.")
finally
    # 5. Clean up the temporary directory and environment variable.
    println("Cleaning up the temporary build environment...")
    delete!(ENV, "BNGL_JULIA_PROJECT_PATH")
    rm(tmp_project_dir; force=true, recursive=true)
end

println("\nDone.")