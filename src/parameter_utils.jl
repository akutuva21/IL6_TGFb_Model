# src/parameter_utils.jl

using PEtab
using ComponentArrays

"""
    untransform_parameters(theta_optim::ComponentVector, parameter_map)

Takes the optimized parameter vector (which is on the scale defined in PEtab, e.g., log10)
and returns a dictionary mapping parameter symbols to their UNTRANSFORMED (linear) scale values.
"""
function untransform_parameters(theta_optim::ComponentVector, parameter_map)
    untransformed_dict = Dict{Symbol, Float64}()
    param_names = propertynames(theta_optim)

    for p_id in param_names
        # PEtab parameter maps use the base name (without log10_ prefix)
        param_info = parameter_map[p_id]
        val_on_scale = theta_optim[p_id]
        
        if param_info.scale == :log10
            untransformed_dict[p_id] = 10^val_on_scale
        elseif param_info.scale == :ln
            untransformed_dict[p_id] = exp(val_on_scale)
        else # :lin
            untransformed_dict[p_id] = val_on_scale
        end
    end
    return untransformed_dict
end


"""
    get_startguess(param_dict_linear::Dict, parameter_map)

Takes a dictionary of parameters on the LINEAR scale and returns a vector
on the correct PEtab scale, to be used as a start guess for `calibrate`.
"""
function get_startguess(param_dict_linear::Dict, parameter_map)
    start_guess = Float64[]
    for param_info in values(parameter_map)
        if param_info.estimate
            val_linear = param_dict_linear[param_info.parameter]
            if param_info.scale == :log10
                push!(start_guess, log10(val_linear))
            elseif param_info.scale == :ln
                push!(start_guess, log(val_linear))
            else # :lin
                push!(start_guess, val_linear)
            end
        end
    end
    return start_guess
end
