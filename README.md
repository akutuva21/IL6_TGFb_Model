# IL6/TGFb Model for Th17 and Treg Differentation

This repository provides a complete workflow for performing parameter estimation on a BioNetGen (BNGL) model using the PEtab standard via robustness testing currently. The pipeline uses Python for data generation and preprocessing and Julia with the `PEtab.jl` ecosystem for robust, multi-start parameter estimation.

## Overview

The core idea is to calibrate a biochemical model, defined in a `.bngl` file, against experimental data. This project is structured to handle two common types of experiments: **time-course data** and **dose-response data**.

The workflow is divided into three main stages:

1.  **Configuration**: A central `config.yml` file controls all aspects of the analysis, from data sources to simulation settings.
2.  **Data Preparation (Python)**: A Python script (`generate_ss_data.py`) either generates synthetic data from the model or converts existing experimental data into the PEtab format.
3.  **Parameter Estimation (Julia)**: A Julia script (`main.jl`) takes the model and the PEtab-formatted data to perform parameter estimation, leveraging multi-processing and pre-compilation for high performance.

## Key Features

  * **BNGL Integration**: Directly uses a BioNetGen model as the basis for analysis.
  * **Dual-Mode Operation**: Switch between `time-course` and `dose-response` analysis.
  * **Data Generation**: Capable of generating synthetic time-course data with optional noise for model testing.
  * **PEtab Conversion**: Converts standard wide-format Excel data into PEtab-compliant TSV files.
  * **High-Performance Estimation**: Utilizes Julia, `PEtab.jl`, and multi-processing for efficient and robust parameter calibration.
  * **System Image Caching**: Includes a script to pre-compile all Julia dependencies into a system image, dramatically reducing startup times.
  * **Rich Visualization**: Automatically generates plots for model fits, parameter distributions, and optimization performance (waterfall plot).

## Prerequisites

Before you begin, ensure you have the following installed:

  * **Python 3.8+**
  * **Julia 1.9+**
  * **BioNetGen**: The `bionetgen` Python package is required for model simulation and parsing.

## Installation & Setup

1.  **Clone the Repository**

2.  **Set Up Python Environment**
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

3.  **Set Up Julia Environment**
    Start Julia in the project directory and activate the local environment.

    ```julia
    # Press ']' to enter Pkg mode
    pkg> activate .
    pkg> instantiate
    ```

    This will install all the Julia packages listed in `Project.toml`.

4.  **(Highly Recommended) Create the Julia System Image**
    To significantly speed up the startup of the estimation script, pre-compile the dependencies. Run this command from your terminal:

    ```bash
    julia create_sysimage.jl
    ```

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
  * `--output <filename>`: Name for the `.jld2` file that saves the estimation results (e.g., `my_results.jld2`).
  * `--optimizer <String>`: Choose the optimization algorithm. `LBFGS` (default) or `IPNewton`.
  * `--debug`: Runs in a fast debug mode with looser tolerances and shorter time limits.


## Output

After a successful run of `main.jl`, you will find:

1.  **Estimation Results (`.jld2` file)**: A binary file (e.g., `[filename].jld`) containing the complete multi-start optimization result object, including the best-fit parameters (`xmin`) and the minimum cost (`fmin`).
2.  **Visualization Plots (`final_results_plots/`)**:
      * **Model Fit Plots**: A `.png` for each observable, showing the experimental data points overlaid with the simulated model curve using the best-fit parameters.
      * **Waterfall Plot**: Shows the final objective function value for each optimization run, sorted to visualize convergence.
      * **Parameter Distribution Plot**: A parallel coordinates plot showing the final values for each estimated parameter from all optimization runs, highlighting the best run.
