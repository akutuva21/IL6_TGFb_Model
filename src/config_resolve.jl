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
    
    # Determine suffix based on noise settings
    suffix = if get(noise, "add", false) && get(noise, "level_percent", 0) > 0
        "_noise$(Int(noise["level_percent"]))"
    else
        "_no_noise"
    end
    
    # Build resolved file paths (matching your current structure)
    parameters_tsv   = "petab_files/parameters$(suffix).tsv"
    observables_tsv  = "petab_files/observables$(suffix).tsv"
    measurements_tsv = "SimData/measurements_time_course$(suffix).tsv"
    conditions_tsv   = "SimData/conditions_time_course$(suffix).tsv"
    
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
        error("Missing PEtab files for suffix '$suffix':\n" * 
              join(missing_files, "\n") * 
              "\nRun 'python generate_ss_data.py' to regenerate data.")
    end
    
    return (suffix=suffix,
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
