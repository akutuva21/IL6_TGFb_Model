# create_sysimage.jl
using Pkg, PackageCompiler

# Paths
env_path   = abspath(joinpath(@__DIR__, "bngl_julia"))   # where Project/Manifest live
data_path  = abspath(@__DIR__)                           # project root with precompile_workload.jl, src/, YAML
precomp_js = joinpath(data_path, "precompile_workload.jl")

# Sanity checks
isfile(precomp_js) || error("Precompile file not found at: $precomp_js")
isfile(joinpath(env_path, "Project.toml")) || error("Project.toml not found at: $env_path")

# Activate exact environment (keeps PEtab 3.10.0 you resolved)
Pkg.activate(env_path)
Pkg.instantiate()

# Environment for workload
ENV["BNGL_JULIA_PROJECT_PATH"] = data_path   # where src/ and petab_problem.yml are
ENV["GKSwstype"] = "100"                     # headless GR for Plots on clusters

# Packages to bake in
final_pkgs = [
    "DifferentialEquations","OrdinaryDiffEq","Sundials","SciMLBase",
    "SciMLSensitivity","DiffEqCallbacks","ModelingToolkit","Catalyst",
    "Symbolics","SymbolicUtils","ReactionNetworkImporters",
    "PEtab","Optimization","OptimizationOptimJL","Optim","ADTypes",
    "LikelihoodProfiler","QuasiMonteCarlo",
    "ReverseDiff","ForwardDiff",
    "DataFrames","CSV","JLD2","XLSX","YAML","DataInterpolations",
    "Plots","Colors","RecipesBase",
    "ComponentArrays","ArgParse","PackageCompiler",
]
unique!(sort!(final_pkgs))

mkpath("SysImage")
create_sysimage(
    final_pkgs;
    project = env_path,                    # use your exact resolved env
    #precompile_execution_file = precomp_js,# the file in project root
    sysimage_path = joinpath("SysImage","bngl_full.so"),
    incremental = true,                    # more forgiving; switch if needed
    cpu_target = "generic;sandybridge;znver2;cascadelake",
)