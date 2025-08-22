# In create_sysimage.jl

using Pkg, PackageCompiler

# 1. Define the list of packages to be included in the sysimage.
# CRITICAL: Fides and PyCall have been REMOVED as they are not compatible with sysimage creation.
final_pkgs = [
    # Core differential equations and scientific computing
    "DifferentialEquations", "OrdinaryDiffEq", "Sundials", "SciMLBase",
    "SciMLSensitivity", "DiffEqCallbacks", "ModelingToolkit", "Catalyst",
    "Symbolics", "SymbolicUtils", "ReactionNetworkImporters", 
    
    # PEtab ecosystem and optimization (Fides removed)
    "PEtab", "Optimization", "OptimizationOptimJL", "Optim", "ADTypes",
    "LikelihoodProfiler", "CICOBase", "QuasiMonteCarlo",
    
    # Automatic differentiation
    "ReverseDiff", "ForwardDiff",
    
    # Data handling and I/O
    "DataFrames", "CSV", "JLD2", "XLSX", "YAML", "DataInterpolations",
    
    # Plotting and visualization
    "Plots", "Colors", "RecipesBase",
    
    # Utilities and arrays
    "ComponentArrays", "ArgParse",
    
    # Build tools
    "PackageCompiler"
]
unique!(sort!(final_pkgs))

# Define the standard libraries that PackageCompiler needs to see explicitly.
std_libs = ["Dates", "LinearAlgebra", "Pkg", "Printf", "Random", "Logging"]

println("📦 System image will be built with $(length(final_pkgs)) packages and $(length(std_libs)) standard libraries.")

# 2. Define project paths
project_path = abspath(@__DIR__)
bngl_julia_project_path = joinpath(project_path, "bngl_julia")

# 3. Create a temporary, clean project directory.
tmp_project_dir = mktempdir()
println("\nCreating a clean build environment at: ", tmp_project_dir)

try
    # 4. Copy your project's Manifest.toml to ensure consistent versions.
    println("Copying project Manifest.toml to the build directory...")
    cp(joinpath(bngl_julia_project_path, "Manifest.toml"), joinpath(tmp_project_dir, "Manifest.toml"))

    # 5. Activate the temporary environment and add the packages.
    Pkg.activate(tmp_project_dir)
    println("Adding main packages to the clean environment...")
    Pkg.add(final_pkgs)
    println("Adding standard libraries to the clean environment...")
    Pkg.add(std_libs) # Add the standard libraries

    # Set environment variable for the workload script
    ENV["BNGL_JULIA_PROJECT_PATH"] = project_path
    println("🗂️  Setting BNGL_JULIA_PROJECT_PATH = $project_path")

    # 6. Build the system image.
    mkpath("SysImage")
    sysimage_path = joinpath("SysImage", "bngl_full.so")
    println("\n🛠 Creating FULL system image...")
    
    create_sysimage(
        final_pkgs; # Pass the list of non-standard-library packages here
        sysimage_path  = sysimage_path,
        project        = tmp_project_dir,
        precompile_execution_file = joinpath(project_path, "precompile_workload.jl"),
        incremental    = false,
        cpu_target     = "x86-64-v2"
    )

    println("\n✅ Full system image created successfully.")
finally
    # 7. Clean up.
    println("Cleaning up the temporary build environment...")
    delete!(ENV, "BNGL_JULIA_PROJECT_PATH")
    rm(tmp_project_dir; force=true, recursive=true)
end

println("\nDone.")