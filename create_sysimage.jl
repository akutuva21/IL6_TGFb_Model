# create_sysimage.jl

using Pkg
Pkg.activate("bngl_julia/")

using PackageCompiler

pkgs = [
    "DataFrames",
    "DifferentialEquations",
    "ModelingToolkit",
    "Catalyst",
    "SymbolicUtils",
    "Symbolics",
    "PEtab",
    "Plots",
    "ReactionNetworkImporters",
    "Optimization",
    "Optim",
    "OptimizationOptimJL",
    "DiffEqCallbacks",
    "SciMLBase",
    "SciMLSensitivity",
    "ArgParse",
    "JLD2", 
    "ComponentArrays",
    "XLSX"
]

sysimage_path = "bngl_sysimage.so"

println("--- Creating custom system image (package-only mode) ---")
create_sysimage(pkgs;
                sysimage_path=sysimage_path
                )

println("--- System image created at '$sysimage_path' ---")
