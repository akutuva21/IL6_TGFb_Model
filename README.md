# IL6/TGFβ Model for Th17 and Treg Differentiation

This repository provides a complete workflow for performing parameter estimation on a BioNetGen (BNGL) model using the PEtab standard. The pipeline uses Python for data generation and preprocessing, and Julia with the `PEtab.jl` ecosystem for robust, multi-start parameter estimation.

## Overview

The core idea is to calibrate a biochemical model, defined in a `.bngl` file, against experimental data. This project is structured to handle two common types of experiments: **time-course data** and **dose-response data**.

The workflow is divided into three main stages:

1. **Configuration**: A central `config.yml` file controls all aspects of the analysis, from data sources to simulation settings.
2. **Data Preparation (Python)**: A Python script (`generate_ss_data.py`) either generates synthetic data from the model or converts existing experimental data into the PEtab format.
3. **Parameter Estimation (Julia)**: A Julia script (`main.jl`) takes the model and the PEtab-formatted data to perform parameter estimation, leveraging multi-processing and pre-compilation for high performance.

## Key Features

* **BNGL Integration**: Directly uses a BioNetGen model as the basis for analysis.
* **Dual-Mode Operation**: Switch between `time-course` and `dose-response` analysis.
* **Data Generation**: Capable of generating synthetic time-course data with optional noise for model testing.
* **PEtab Conversion**: Converts standard wide-format Excel data into PEtab-compliant TSV files.
* **High-Performance Estimation**: Utilizes Julia, `PEtab.jl`, and multi-processing for efficient and robust parameter calibration.
* **System Image Caching**: Includes a script to pre-compile all Julia dependencies into a system image, dramatically reducing startup times.
* **Rich Visualization**: Automatically generates plots for model fits, parameter distributions, and optimization performance (waterfall plot).
* **Robust Solver Configuration**: Automatically selects appropriate ODE and steady-state solvers based on debug vs. production mode.

## Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.8+**
* **Julia 1.9+**
* **BioNetGen**: The `bionetgen` Python package is required for model simulation and parsing.

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd IL6_TGFB
```

### 2. Set Up Python Environment

Create a `requirements.txt` file with the necessary packages:

```text
# requirements.txt
pandas
numpy
pyyaml
bionetgen
openpyxl
```

Then, install them using pip:

```bash
pip install -r requirements.txt
```

### 3. Set Up Julia Environment

The Julia environment is managed through the `bngl_julia/` project. Activate and instantiate it:

```bash
cd bngl_julia/
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

### 4. Optional: Create Julia System Image

For faster startup times, create a precompiled system image:

```bash
julia create_sysimage.jl
```

This will create a system image that dramatically reduces Julia compilation time for subsequent runs.

## Quick Start

### Step 1: Configure Your Analysis

Edit `config.yml` to specify your analysis parameters:

```yaml
# Main configuration
model_path: "model_even_smaller.bngl"
output_dir: "SimData"
run_mode: "time_course_petab"  # or "dose_response"

# Parameter bounds for estimation
parameter_bounds:
  default_kinetic: {lb: 1.0e-6, ub: 10.0}
  default_initial_conc: {lb: 1.0e-3, ub: 1000.0}

# Time-course settings
time_course_settings:
  variable_stimuli: ["IL6_0"]
  constant_stimuli: ["TGFb_0"]
  conditions:
    TREG: {IL6_0: 0.0, TGFb_0: 1.0}
    TH17: {IL6_0: 100.0, TGFb_0: 1.0}
  simulation:
    duration: 100.0
    steps: 20

# Observable mappings (unified for all modes)
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

### Step 2: Generate PEtab Data

Run the Python script to generate PEtab-formatted data:

```bash
python generate_ss_data.py --config config.yml
```

This will create:
- `SimData/measurements_time_course.tsv`: Measurement data
- `SimData/conditions_time_course.tsv`: Experimental conditions

### Step 3: Run Parameter Estimation

Run the Julia pipeline:

```bash
julia main.jl --mode time-course --output my_results.jld
```

Or with additional options:

```bash
julia main.jl \
  --mode time-course \
  --parallel \
  --n-starts 20 \
  --optimizer LBFGS \
  --with-preeq \
  --output robust_results.jld
```

### Step 4: Debug Mode (Optional)

For faster testing during development:

```bash
julia main.jl --mode time-course --debug --n-starts 5
```

Debug mode uses:
- Faster, pure-Julia solvers (Rodas5P)
- Looser tolerances for quick convergence
- Shorter time limits
- ForwardDiff for gradient computation

## Command Line Options

### Main Julia Script (`main.jl`)

```bash
julia main.jl [OPTIONS]
```

**Required Arguments:**
- `--mode`: Analysis mode (`time-course` or `dose-response`)

**Optional Arguments:**
- `--parallel`: Enable multi-processing for parameter estimation
- `--n-starts INT`: Number of multi-start optimizations (default: 10 serial, nprocs() parallel)
- `--optimizer STR`: Optimization algorithm (default: `LBFGS`)
- `--output/-o FILE`: Output file path (default: `estimation_output_small.jld`)
- `--with-preeq`: Enable pre-equilibration before main simulation
- `--debug`: Enable debug mode for faster testing
- `--abstol FLOAT`: Absolute tolerance for ODE solver (default: 1e-8)
- `--reltol FLOAT`: Relative tolerance for ODE solver (default: 1e-8)
- `--net-file FILE`: Path to BioNetGen .net file
- `--config FILE`: Path to config.yml file
- `--measurements-file FILE`: Custom measurements file path
- `--profile`: Run likelihood profiling on the best-fit parameters after estimation. This is computationally intensive and requires a results file (`.jld`) to exist.

### Python Data Generation Script (`generate_ss_data.py`)

```bash
python generate_ss_data.py [OPTIONS]
```

**Optional Arguments:**
- `--config/-c FILE`: Path to YAML configuration file (default: `config.yml`)

## Configuration Reference

### Run Modes

1. **`time_course_petab`**: Generates time-course data in PEtab TSV format
2. **`time_course`**: Generates time-course data in Excel format (legacy)
3. **`dose_response`**: Processes dose-response data from Excel to PEtab format

### Parameter Bounds

```yaml
parameter_bounds:
  default_kinetic: {lb: 1.0e-6, ub: 10.0}
  default_initial_conc: {lb: 1.0e-3, ub: 1000.0}
  overrides:
    specific_param: {lb: 1.0e-4, ub: 100.0}
```

### Solver Configuration

The pipeline automatically selects appropriate solvers:

- **Debug Mode**: Rodas5P (pure Julia) + :Simulate steady-state + ForwardDiff
- **Production Mode**: CVODE_BDF (Sundials) + :Simulate steady-state + Adjoint

## Output Files

### Parameter Estimation Results

- `*.jld`: Julia binary file containing optimization results
- `final_results_plots/`: Visualization plots
- `final_results_csv/`: Parameter estimates in CSV format

### Generated Plots

- **Waterfall Plot**: Shows convergence across multiple starts
- **Parameter Distribution**: Distribution of estimated parameters
- **Model Fit Plots**: Comparison of model predictions vs. data

## Troubleshooting

### Common Issues

1. **Observable Mapping Errors**: Ensure all observables in your data are mapped in `observables_mapping`
2. **Solver Failures**: Try debug mode first, then adjust tolerances
3. **Memory Issues**: Reduce `n-starts` or use serial mode
4. **Compilation Time**: Use system image for faster startup

### Debug Mode

Always test with debug mode first:

```bash
julia main.jl --mode time-course --debug --n-starts 2
```

This provides:
- Faster compilation
- Shorter run times
- Better error messages
- Looser tolerances for numerical stability

## Performance Tips

1. **Use System Image**: Run `julia create_sysimage.jl` for faster startup
2. **Parallel Processing**: Use `--parallel` for multi-start optimization
3. **Appropriate Tolerances**: Use tighter tolerances for production runs
4. **Solver Selection**: The pipeline automatically selects optimal solvers

## Advanced Usage

### Custom Measurements File

```bash
julia main.jl \
  --measurements-file custom_data.tsv \
  --conditions-file custom_conditions.tsv
```

### SLURM Integration

```bash
sbatch --job-name=param_est \
       --partition=compute \
       --nodes=1 \
       --cpus-per-task=20 \
       --wrap="julia main.jl --mode time-course --parallel --n-starts 50"
```

### Multiple Optimizers

```bash
# Test different optimizers
julia main.jl --optimizer LBFGS --output lbfgs_results.jld
julia main.jl --optimizer NelderMead --output nelder_results.jld
julia main.jl --optimizer IPNewton --output ipnewton_results.jld
```

### Running Likelihood Profiling

After obtaining a good set of best-fit parameters from a multi-start run, you can analyze parameter identifiability using likelihood profiling.

1. **First, run the parameter estimation** to generate a results file (e.g., `my_results.jld`).
   ```bash
   julia -J bngl_sysimage.so main.jl --mode time-course --parallel --n-starts 50 --output my_results.jld
   ```

2. **Then, run the script again with the `--profile` flag.** It will load the results from the specified output file and begin the profiling analysis.
   ```bash
   julia -J bngl_sysimage.so main.jl --mode time-course --output my_results.jld --profile
   ```

The results will be saved as individual PNG files in the `likelihood_profiles/` directory. Steep, V-shaped profiles indicate identifiable parameters, while flat profiles suggest non-identifiability.

## Setup Instructions

    This will create a `bngl_sysimage.so` file. The main script will automatically use it if available.

## Running the Pipeline

The entire analysis is controlled by `config.yml` and executed via the Python and Julia scripts.

### Step 1: Configure Your Analysis (`config.yml`)

This is the main control panel. Open `config.yml` and edit it for your needs.

  * **`run_mode`**: The most important setting.
      * `"time_course"`: To generate and fit time-course data.
      * `"dose_response"`: To process and fit steady-state dose-response data.
  * **`model_path`**: Path to your `.bngl` model file.
  * **`output_dir`**: Where the generated data files will be saved (default: `SimData/`).
  * **`time_course_settings`**:
      * `conditions`: Define experimental conditions (e.g., `Treg`, `TH17`) and their corresponding stimulus levels (`IL6_0`, `TGFb_0`).
      * `simulation`: Set the duration and number of steps for the simulation.
  * **`dose_response_settings`**:
      * `input_data`: Specify the path to your Excel data file and map its columns (`column_to_observable_map`) to the observable names you want to use in PEtab.
      * `dose_parameter`: The parameter being varied in the experiment (e.g., `"IL6_0"`).

### Step 2: Prepare the Data (Python)

Run the Python script to process your data according to the `config.yml` settings.

```bash
python generate_ss_data.py --config config.yml
```

  * If `run_mode` is `"time_course"`, this will generate a `preeq.xlsx` file in the `output_dir`.
  * If `run_mode` is `"dose_response"`, this will read your Excel file and create two PEtab files: `measurements_dose_response.tsv` and `conditions_dose_response.tsv`.

### Step 3: Run Parameter Estimation (Julia)

Execute the main Julia script to perform the parameter estimation. Use command-line arguments to control its behavior.

```bash
# Example for time-course analysis with pre-equilibration and parallelism
julia -J bngl_sysimage.so main.jl --mode time-course --with-preeq --parallel

# Example for dose-response analysis
julia -J bngl_sysimage.so main.jl --mode dose-response --with-preeq --parallel --n-starts 20
```

*The `-J bngl_sysimage.so` flag tells Julia to use the pre-compiled system image.*

#### Key Command-Line Arguments:

  * `--mode`: `time-course` or `dose-response`. **Must match the data you prepared in Step 2.**
  * `--with-preeq`: **(Recommended)** Enables pre-equilibration to find a steady state before applying stimuli.
  * `--parallel`: Runs the multi-start optimization across multiple CPU cores.
  * `--n-starts <Int>`: The number of independent optimization runs to perform. Defaults to the number of available cores.
  * `--output <filename>`: Name for the `.jld` file that saves the estimation results (e.g., `my_results.jld`).
  * `--optimizer <String>`: Choose the optimization algorithm. `LBFGS` (default) or `IPNewton`.
  * `--debug`: Runs in a fast debug mode with looser tolerances and shorter time limits.


## Output

After a successful run of `main.jl`, you will find:

1.  **Estimation Results (`.jld` file)**: A binary file (e.g., `[filename].jld`) containing the complete multi-start optimization result object, including the best-fit parameters (`xmin`) and the minimum cost (`fmin`).
2.  **Visualization Plots (`final_results_plots/`)**:
      * **Model Fit Plots**: A `.png` for each observable, showing the experimental data points overlaid with the simulated model curve using the best-fit parameters.
      * **Waterfall Plot**: Shows the final objective function value for each optimization run, sorted to visualize convergence.
      * **Parameter Distribution Plot**: A parallel coordinates plot showing the final values for each estimated parameter from all optimization runs, highlighting the best run.
