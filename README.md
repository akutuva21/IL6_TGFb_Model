# IL6/TGFβ Model – PEtab-based Calibration Pipeline

This repository contains a complete workflow to generate PEtab-compliant data from a BNGL model and perform robust multi-start parameter estimation in Julia using PEtab.jl.

## What’s included

- Python data generation and PEtab file creation: `generate_ss_data.py`
- Julia batch-based multistart calibration, collation, profiling, and visualization: `main.jl` and `src/`
- Precompilation workload (optional): `precompile_workload.jl`

## Requirements

- Python 3.8+
- Julia 1.10+
- Python packages: `pandas`, `numpy`, `pyyaml`, `bionetgen`, `openpyxl`

## Setup

```powershell
# Clone
git clone <repository-url>
cd IL6_TGFB

# Python deps (install listed packages)
pip install pandas numpy pyyaml bionetgen openpyxl

# Julia deps (project lives under bngl_julia/)
julia --project=bngl_julia -e "using Pkg; Pkg.instantiate()"
```

Optional: build a sysimage for faster Julia startup

```powershell
julia --project=bngl_julia precompile_workload.jl
# or your existing sysimage creation script
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

```powershell
python generate_ss_data.py --config config.yml
```

This creates PEtab TSVs in `petab_files/`:

- `observables.tsv` (uses `noiseDistribution=logNormal` and a single shared `noiseFormula = sigma_log_shared`)
- `parameters.tsv` (adds `sigma_log_shared` fixed to the selected noise level; initial-condition parameters `*_0` are not estimated)
- `measurements_time_course[_noiseX].tsv`
- `conditions_time_course[_noiseX].tsv`

Additional data-generation details:

- A small floor is applied to non-positive measurements before noise: values ≤ 1e-12 are set to 1e-8, then log-normal noise is applied.
- Parameters ending with `_0` are treated as fixed (not estimated) in the generated `parameters.tsv`.

Make sure your PEtab YAML (e.g., `petab_problem.yml`) points to these TSVs.

## Batch-based parameter estimation (Julia)

`main.jl` now supports a batch-oriented workflow suitable for HPC job arrays. Each batch runs a slice of the total multi-starts in parallel processes using `calibrate_multistart`.

Key CLI flags:

- `--yaml <path>`: Path to PEtab YAML.
- `--optimizer {IPNewton|LBFGS|Fides}`: Optimizer.
- `--n-starts <N>`: Total number of starts across all batches (e.g., 500).
- `--n-batches <B>`: Total number of batches (e.g., 16).
- `--batch-id <i>`: 1-based batch index for the current job. If > 0, runs in batch worker mode.
- `--n-procs <P>`: Number of processes used within a batch by `calibrate_multistart` (e.g., 32).
- `--collate`: Collate results from all batch directories and produce diagnostics/plots.
- `--profile`: Run likelihood profiling (after collation).
- `--debug`: Faster, looser settings.

### Run a single batch locally

```powershell
julia --project=bngl_julia main.jl `
  --yaml petab_problem.yml `
  --optimizer Fides `
  --n-starts 500 `
  --n-batches 16 `
  --batch-id 1 `
  --n-procs 32
```

### Collate all batches

After all batches have finished, collate results across `results/batch_*/`:

```powershell
julia --project=bngl_julia main.jl `
  --yaml petab_problem.yml `
  --n-batches 16 `
  --collate
```

This scans each `results/batch_i/` for `results1.csv` and `xmins1.csv`, reconstructs results, finds the global best, saves `best_fit.jld2`, and generates diagnostic plots.

### Profile the best fit

```powershell
julia --project=bngl_julia main.jl `
  --yaml petab_problem.yml `
  --profile `
  --load-fit best_fit.jld2
```

## Optimizers

- Fides via PEtab.jl: `PEtab.Fides(:BFGS)` with Python options, including `maxtime` and robust tolerances.
- Optim.jl fallbacks: `IPNewton`, `LBFGS` with tuned `Optim.Options`.

## Likelihood profiling

Profiling is handled in `src/profiling.jl` and runs after collation if `--profile` is provided. Defaults:

- Parameter selection: all parameters except noise (`sigma*`) and initial conditions (`*_0`).
- Profiler backend: `CICOProfiler` (from `CICOBase`) for robust endpoint-finding.
- Parallelization: threaded across parameters.
- Output: `likelihood_profiles/profile_<param>.png` plus Δχ² overlays.

## Tips

- Use batches to scale across the cluster; each batch can use many processes via `--n-procs`.
- Set a consistent RNG seed (built-in) so batches slice the same global start list deterministically.
- Keep `--debug` for quick smoke tests; omit it for final runs.

## Repository layout

- `generate_ss_data.py`: BNGL simulation, pre-equilibration, PEtab TSV generation (log-normal noise with shared sigma, LOD floor, fixed `*_0` params)
- `main.jl`: CLI entry; batch worker, collation, profiling, visualization
- `src/optimization.jl`: Optimizer selection and batch runner (`run_batch_optimization`)
- `src/profiling.jl`: Likelihood profiling utilities (CICOProfiler)
- `src/visualization.jl`: Diagnostic and result plots
- `precompile_workload.jl`: Optional PEtab-specific precompilation workload

Julia environment files live under `bngl_julia/` (this is the active project for running the Julia code in this repo).
