# IL6/TGFβ Model – PEtab-based Calibration Pipeline

This repository contains a complete workflow to generate PEtab-compliant data from a BNGL model and perform robust multi-start parameter estimation in Julia using PEtab.jl.

## What’s included

- Python data generation and PEtab file creation: `generate_ss_data.py`
- Julia multi-start calibration and visualization: `main.jl` and `src/`
- Precompilation workload (optional): `precompile_workload.jl`

## Requirements

- Python 3.8+
- Julia 1.9+
- Python packages: `pandas`, `numpy`, `pyyaml`, `bionetgen`, `openpyxl`

## Setup

```bash
git clone <repository-url>
cd IL6_TGFB

# Python deps
pip install -r requirements.txt  # or install packages listed above

# Julia deps (from project root)
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

Optional: build a sysimage for faster Julia startup

```bash
julia precompile_workload.jl
# or your existing sysimage creation script
```

## Basic workflow

1) Generate PEtab data (Python)

```bash
python generate_ss_data.py --config config.yml
```

2) Point your PEtab YAML to the generated TSVs (in `petab_files/`).

3) Run multi-start calibration (Julia)

```bash
julia --threads=<N> --project=. main.jl --yaml petab_problem.yml --n-starts 24 --optimizer Fides --output results.jld
```

4) Visualize and optional profiling

```bash
julia --threads=<N> --project=. main.jl --yaml petab_problem.yml --n-starts 24 --optimizer Fides --output results.jld --profile
```

## Generate PEtab data (Python)

Edit `config.yml` to define your model and experiment configuration. Minimal example of time-course settings and noise:

```yaml
model_path: "model_even_smaller.bngl"
output_dir: "SimData"

time_course_settings:
  variable_stimuli: ["IL6_0"]
  constant_stimuli: ["TGFb_0"]
  conditions:
    TREG: {IL6_0: 0.0, TGFb_0: 1.0}
    TH17: {IL6_0: 100.0, TGFb_0: 1.0}
  simulation:
    duration: 100.0
    steps: 20
  noise:
    add: true            # set false for 0% noise
    level_percent: 5     # 0, 5, 10, ... CV
  random_seed: 42

observables_mapping:
  Free_IL6_obs: Free_IL6_obs
  Free_TGFb_obs: Free_TGFb_obs
  IL6R_Active: IL6R_Active
  PKA_active: PKA_active
  S3S4_complex_obs: S3S4_complex_obs
  S3STAT3d_complex_obs: S3STAT3d_complex_obs
  STAT3d_active_obs: STAT3d_active_obs
  pSMAD3_obs: pSMAD3_obs
```

Then run:

```bash
python generate_ss_data.py --config config.yml
```

This creates PEtab TSVs in `petab_files/`:

- `observables.tsv` (uses `noiseDistribution=logNormal` and per-observable `noiseParameter1_*`)
- `parameters.tsv` (adds per-observable sigma on linear scale; nominal value computed from level_percent)
- `measurements_time_course[_noiseX].tsv`
- `conditions_time_course[_noiseX].tsv`

Make sure your PEtab YAML (e.g., `petab_problem.yml`) points to these TSVs.

## Run parameter estimation (Julia)

`main.jl` expects a PEtab YAML and runs a robust multistart calibration.

Common flags:

```bash
julia --threads=<N> \
      --project=. \
      [--sysimage=<path_to_sysimage>] \
      main.jl \
      --yaml petab_problem.yml \
      --output results.jld \
      --n-starts 24 \
      --optimizer Fides
```

Useful options:

- `--debug`: faster, looser settings (fewer starts, lower maxiter)
- `--profile`: also run likelihood profiling after calibration

Defaults and tuning in this repo:

- ODE solver: Rodas5P
- Grad/Hess (small-model defaults): `gradient_method=:ForwardDiff`, `hessian_method=:ForwardDiff`
- Fides options (multistart-friendly):
  - debug: `maxiter=150`, `fatol=1e-5`, `frtol=1e-6`, `gtol=1e-6`
  - non-debug: `maxiter=500`, `fatol=1e-5`, `frtol=1e-7`, `gtol=1e-6`
- Optim options (IPNewton/LBFGS/BFGS):
  - debug: `iterations=200`, `g_tol=1e-6`, `f_reltol=1e-6`
  - non-debug: `iterations=800`, `g_tol=1e-6`, `f_reltol=1e-8`
- Multistart I/O: intermediate saving disabled unless `--debug`

## Noise model

- Controlled in `config.yml` via `time_course_settings.noise` (e.g., `add: true`, `level_percent: 0|5|10|...`).
- The generator writes matching PEtab files:
  - `observables.tsv`: sets `noiseDistribution = logNormal` and `noiseFormula = noiseParameter1_<obs>`
  - `parameters.tsv`: per‑observable sigma parameter fixed to the selected noise level (set `estimate=1` to fit it)
- To change noise level, update the config and re-run the Python generator.

## Tips

- Prefer multistart threading.
- Save only essential results (`best_mle`, `best_cost`).
- Use `--debug` for quick checks; switch to non-debug for final runs.

## Repository layout

- `generate_ss_data.py`: BNGL simulation, pre-equilibration, PEtab TSV generation (log-normal noise, configurable CV)
- `main.jl`: CLI entry, loads PEtab, builds ODE/SS problems, runs multistart, saves results, produces plots
- `src/optimization.jl`: Optimizer selection and multistart call (Fides/Optim options tuned)
- `src/profiling.jl`: Likelihood profiling utilities
- `src/visualization.jl`: Diagnostic and result plots
- `precompile_workload.jl`: Optional PEtab-specific precompilation workload
