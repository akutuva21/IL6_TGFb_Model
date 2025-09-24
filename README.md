# IL6–TGFβ PEtab Calibration and Profiling

This repository provides a workflow to generate PEtab-compliant problems from a BNGL model, run multi-start parameter estimation in Julia, collate and visualize results, and compute profile-likelihood confidence intervals. It supports local and cluster execution, batching across many starts, and produces diagnostics and plots for interpretation.

## Features

- PEtab model setup from BNGL/SBML with multiple conditions and steady-state handling via PEtab.jl's ODE problem interface.
- Batch-oriented multi-start calibration with flexible optimizer backends, reproducible start sets, and robust result collation.
- Likelihood profiling using CICO-based endpoint search to summarize practical identifiability and confidence intervals, with plotting and tabular outputs.
- Optional Fisher Information Matrix diagnostics at the best fit for complementary local analysis.

## Repository layout

- config.yml — experiment, noise, and export settings for data generation and PEtab tables.
- petab_problem.yml and petab_files/ — PEtab problem specification and generated TSVs for parameters, measurements, observables, and conditions.
- src/ — Julia modules for optimization, profiling, identifiability, visualization, and robustness utilities used by the main CLI.
- model_even_smaller.bngl — example model and exported SBML used to build the ODESystem.
- Data/ and SimData/ — example inputs/outputs and artifacts for validation.
- bngl_julia/Project.toml — Julia project environment for running the pipeline.

## Requirements

- Julia 1.10+ with the bngl_julia environment instantiated via Pkg.
- Optional Python 3 for generating PEtab TSVs from BNGL using the included script.

## Quick start

Instantiate the Julia environment under bngl_julia, run multi-start calibration with main.jl, collate to recover the best fit, and optionally compute profile-likelihood intervals.

Example
```bash
# Environment
julia --project=bngl_julia -e "using Pkg; Pkg.instantiate()"

# Single batch locally
julia --project=bngl_julia main.jl \
  --yaml petab_problem.yml --optimizer Fides \
  --n-starts 500 --n-batches 16 --batch-id 1 --n-procs 32

# Collate after batches complete
julia --project=bngl_julia main.jl --yaml petab_problem.yml --n-batches 16 --collate

# Profile the best fit saved at collation
julia --project=bngl_julia main.jl --yaml petab_problem.yml --profile --load-fit best_fit.jld2
```
These commands reflect the intended workflow: distributed starts, centralized collation, and optional profiling at the best fit.

## Workflow summary

1) Data and PEtab creation — configure config.yml and optionally generate PEtab tables used by petab_problem.yml.
2) Multi-start calibration — run starts locally or in batches; each start is saved and subject to the chosen optimizer's settings.
3) Collation and visualization — select the best solution, compare against reference values on the internal scale, and generate diagnostic figures.
4) Profiling and identifiability — compute profile-likelihood endpoints to summarize confidence intervals and complement FIM-based local analysis.

## Model notes (example)

The example BNGL model includes reversible ligand–receptor binding and activation/deactivation steps to yield proper dose–response behavior. The exported SBML and PEtab problem are used to build the ODE problem for calibration and profiling.

## Likelihood profiling (high level)

The profiling step searches for where each parameter's profile intersects a confidence threshold suitable for negative log-likelihood objectives. The method integrates with the PEtab ODE problem and reports left/right endpoints per parameter, including one‑sided outcomes when the profile intersects domain limits before the threshold.

## Outputs

- best_fit.jld2 — best parameters and metadata from collation for downstream profiling and visualization.
- results/ — CSV summaries and plots (multi-start diagnostics, parameter distributions, and profile-based confidence intervals).

## Identifiability diagnostics

FIM and related local analyses at the best fit complement profile-based intervals. Together they distinguish local curvature at the optimum from bounded-domain profile behavior, informing model updates or experimental design.

## Reproducibility

Use the included Julia project for all commands, keep seeds consistent where configured, and archive best_fit.jld2 and the environment files. For batch runs on clusters, maintain a consistent directory layout so that collation and profiling operate deterministically across runs.
