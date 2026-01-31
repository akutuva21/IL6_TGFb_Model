ENV["GKSwstype"] = "100"
using Plots
include("../src/visualization.jl")
plot_dose_response("SimData/measurements_real_data.tsv", "SimData/conditions_real_data.tsv"; endpoint_time=60.0, output_dir="final_results_plots_test")
