module ConfigResolve

using YAML
using Dates

export resolve_petab_paths, write_temp_petab_yaml, log_resolved_files

"""
    resolve_petab_paths(config_path::String)
    
Reads config.yml and returns resolved PEtab file paths based on noise settings.
"""
function resolve_petab_paths(config_path::String)
    cfg = YAML.load_file(config_path)
    
    # Extract noise settings from time_course_settings
    tc = cfg["time_course_settings"]
    noise = tc["noise"]
        
    # Suffix for parameter and observable files
    petab_file_suffix = if get(noise, "add", false) && get(noise, "level_percent", 0) > 0
        "_noise$(Int(noise["level_percent"]))"
    else
        "" # No suffix for noise-free
    end

    # Suffix for measurement and condition data files
    data_file_suffix = if get(noise, "add", false) && get(noise, "level_percent", 0) > 0
        "_noise$(Int(noise["level_percent"]))"
    else
        "_no_noise" # Suffix is '_no_noise' for data files
    end
    
    # Build resolved file paths using the correct suffix for each file type
    parameters_tsv   = "petab_files/parameters$(petab_file_suffix).tsv"
    observables_tsv  = "petab_files/observables$(petab_file_suffix).tsv"
    measurements_tsv = "SimData/measurements_time_course$(data_file_suffix).tsv"
    conditions_tsv   = "SimData/conditions_time_course$(data_file_suffix).tsv"
    
    # Fail-fast if any files are missing
    missing_files = String[]
    for (label, path) in [("parameters", parameters_tsv),
                          ("observables", observables_tsv), 
                          ("measurements", measurements_tsv),
                          ("conditions", conditions_tsv)]
        if !isfile(path)
            push!(missing_files, "$label: $path")
        end
    end
    
    if !isempty(missing_files)
        # Create a more informative suffix for the error message
        error_suffix = isempty(petab_file_suffix) ? "'no_noise'" : "'$petab_file_suffix'"
        error("Missing PEtab files for configuration $error_suffix:\n" * 
              join(missing_files, "\n") * 
              "\nRun 'python generate_ss_data.py' with the correct noise settings to regenerate data.")
    end
    
    # Return the main suffix for logging purposes
    return (suffix=petab_file_suffix,
            parameters_tsv=parameters_tsv,
            observables_tsv=observables_tsv,
            measurements_tsv=measurements_tsv,
            conditions_tsv=conditions_tsv)
end

"""
    write_temp_petab_yaml(out_yaml::String, paths; sbml_file="model_even_smaller_sbml.xml")
    
Creates a temporary PEtab YAML file pointing to resolved TSV files.
"""
function write_temp_petab_yaml(out_yaml::String, paths; sbml_file="model_even_smaller_sbml.xml")
    petab_doc = Dict{String,Any}()
    
    # Basic PEtab structure
    petab_doc["format_version"] = 1
    petab_doc["parameter_file"] = paths.parameters_tsv
    
    petab_doc["problems"] = [Dict(
        "condition_files"   => [paths.conditions_tsv],
        "measurement_files" => [paths.measurements_tsv],
        "observable_files"  => [paths.observables_tsv],
        "sbml_files"        => [sbml_file]
    )]
    
    # Write temporary YAML
    open(out_yaml, "w") do io
        YAML.write(io, petab_doc)
    end
end

"""
    log_resolved_files(paths)
    
Pretty-print resolved files and their modification times.
"""
function log_resolved_files(paths)
    @info "Resolved PEtab files (suffix: $(paths.suffix)):"
    for (label, path) in [("parameters", paths.parameters_tsv),
                          ("observables", paths.observables_tsv),
                          ("measurements", paths.measurements_tsv),
                          ("conditions", paths.conditions_tsv)]
        mtime = Dates.unix2datetime(stat(path).mtime)
        @info "  $label -> $path ($(Dates.format(mtime, "yyyy-mm-dd HH:MM")))"
    end
end

end # module
