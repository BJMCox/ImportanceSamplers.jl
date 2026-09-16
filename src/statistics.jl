@kernel function _evaluate_functional_kernel!(values, samples, f)
    sample_index = @index(Global, Linear)
    @inbounds values[sample_index] = f(_sample_at(samples, sample_index))
end

"""
    Statistics.mean(result)
    Statistics.mean(f, result)

Return a self-normalized weighted expectation. The function form evaluates one
scalar value per sample without transferring a device-resident result to CPU.
"""
function Statistics.mean(result::_AbstractWeightedSamples)
    return _with_backend_device(_storage_device(result.logweights)) do
        _weighted_sum(result.samples, normalized_weights(result))
    end
end

function Statistics.mean(f, result::_AbstractWeightedSamples)
    return _with_backend_device(_storage_device(result.logweights)) do
        weights = normalized_weights(result)
        _is_host_storage(result.logweights) &&
            return _weighted_functional_mean(f, result, weights)
        values = _device_functional_values(f, result)
        _weighted_sum(values, weights)
    end
end

"""
    Statistics.var(result; corrected=false)
    Statistics.var(f, result; corrected=false)

Return the uncorrected weighted target variance. The function form computes
the variance of one scalar value per sample. `corrected=true` is unsupported
because importance weights have no general sample-variance correction.
"""
function Statistics.var(result::_AbstractWeightedSamples; corrected::Bool=false)
    _require_uncorrected(corrected)
    return _with_backend_device(_storage_device(result.logweights)) do
        weights = normalized_weights(result)
        center = _weighted_sum(result.samples, weights)
        _weighted_variance(result.samples, weights, center)
    end
end

function Statistics.var(f, result::_AbstractWeightedSamples; corrected::Bool=false)
    _require_uncorrected(corrected)
    return _with_backend_device(_storage_device(result.logweights)) do
        weights = normalized_weights(result)
        if _is_host_storage(result.logweights)
            center = _weighted_functional_mean(f, result, weights)
            return _weighted_functional_variance(f, result, weights, center)
        end
        values = _device_functional_values(f, result)
        center = _weighted_sum(values, weights)
        _weighted_variance(values, weights, center)
    end
end

"""
    Statistics.std(result; corrected=false)
    Statistics.std(f, result; corrected=false)

Return the square root of the corresponding weighted target variance.
"""
function Statistics.std(result::_AbstractWeightedSamples; corrected::Bool=false)
    return _with_backend_device(_storage_device(result.logweights)) do
        _sqrt_summary(Statistics.var(result; corrected=corrected))
    end
end

function Statistics.std(f, result::_AbstractWeightedSamples; corrected::Bool=false)
    return _with_backend_device(_storage_device(result.logweights)) do
        _sqrt_summary(Statistics.var(f, result; corrected=corrected))
    end
end

"""
    Statistics.cov(result; corrected=false)

Return the uncorrected weighted covariance matrix for vector-valued samples.
`corrected=true` is unsupported.
"""
function Statistics.cov(result::_AbstractWeightedSamples; corrected::Bool=false)
    _require_uncorrected(corrected)
    return _with_backend_device(_storage_device(result.logweights)) do
        weights = normalized_weights(result)
        center = _weighted_sum(result.samples, weights)
        _weighted_covariance(result.samples, weights, center)
    end
end

"""
    Statistics.quantile(result, probability)
    Statistics.quantile(result, probabilities)

Return component-wise weighted quantiles using the StatsBase generic-weight
interpolation rule. Transfer device-resident results to CPU before calling.
"""
function Statistics.quantile(
    result::_AbstractWeightedSamples,
    probability::Real,
)
    _require_host_result_access(result)
    weights = normalized_weights(result)
    return _weighted_quantile(result.samples, weights, probability)
end

function Statistics.quantile(
    result::_AbstractWeightedSamples,
    probabilities::AbstractVector{<:Real},
)
    _require_host_result_access(result)
    weights = normalized_weights(result)
    return _weighted_quantile(result.samples, weights, probabilities)
end

"""
    Statistics.median(result)

Return `Statistics.quantile(result, 0.5)`.
"""
Statistics.median(result::_AbstractWeightedSamples) = Statistics.quantile(result, 0.5)

function _require_uncorrected(corrected)
    corrected && throw(
        ArgumentError("corrected moments are undefined for importance weights"),
    )
    return nothing
end

function _device_functional_values(
    f,
    result::Union{_AbstractWeightedSamples,UnweightedSamples},
)
    sample_type = _functional_sample_type(result)
    value_type = Base.promote_op(f, sample_type)
    value_type <: Number && isconcretetype(value_type) || throw(
        ArgumentError("device functionals must return one concrete number per sample"),
    )
    values = similar(_functional_prototype(result), value_type, length(result))
    backend = KernelAbstractions.get_backend(values)
    kernel = _evaluate_functional_kernel!(backend)
    kernel(values, result.samples, f; ndrange=length(values))
    KernelAbstractions.synchronize(backend)
    return values
end

_functional_sample_type(::UnweightedSamples{R}) where {R} = R
_functional_sample_type(::_AbstractWeightedSamples{R}) where {R} = fieldtype(R, :sample)
_functional_prototype(result::UnweightedSamples) = _first_sample_leaf(result.samples)
_functional_prototype(result::_AbstractWeightedSamples) = result.logweights

function _weighted_functional_mean(f, result, weights)
    return mapreduce(
        sample_index -> weights[sample_index] *
                        f(_sample_at(result.samples, sample_index)),
        +,
        eachindex(weights),
    )
end

function _weighted_functional_variance(f, result, weights, center)
    return mapreduce(
        sample_index -> weights[sample_index] *
                        abs2(f(_sample_at(result.samples, sample_index)) - center),
        +,
        eachindex(weights),
    )
end

_weighted_sum(samples::AbstractVector, weights) = LinearAlgebra.dot(weights, samples)
_weighted_sum(samples::AbstractMatrix, weights) = samples * weights

function _weighted_sum(samples::NamedTuple, weights)
    summaries = map(sample -> _weighted_sum(sample, weights), values(samples))
    return NamedTuple{keys(samples)}(summaries)
end

function _weighted_variance(samples::AbstractVector, weights, center)
    return LinearAlgebra.dot(weights, abs2.(samples .- center))
end

function _weighted_variance(samples::AbstractMatrix, weights, center)
    centered = samples .- center
    return vec(sum(abs2.(centered) .* reshape(weights, 1, :); dims=2))
end

function _weighted_variance(samples::NamedTuple, weights, center)
    summaries = map(
        (sample, sample_center) -> _weighted_variance(sample, weights, sample_center),
        values(samples),
        values(center),
    )
    return NamedTuple{keys(samples)}(summaries)
end

function _weighted_covariance(samples::AbstractMatrix, weights, center)
    centered = samples .- center
    return (centered .* reshape(weights, 1, :)) * adjoint(centered)
end

function _weighted_covariance(samples, weights, center)
    throw(ArgumentError("covariance requires vector-valued samples"))
end

function _weighted_quantile(samples::AbstractVector{<:Real}, weights, probability::Real)
    return only(_weighted_quantile(samples, weights, [probability]))
end

function _weighted_quantile(
    samples::AbstractVector{<:Real},
    weights,
    probabilities::AbstractVector{<:Real},
)
    isempty(probabilities) && throw(ArgumentError("quantile probabilities cannot be empty"))
    all(probability -> 0 <= probability <= 1, probabilities) || throw(
        ArgumentError("quantile probabilities must lie in [0, 1]"),
    )

    T = promote_type(float(eltype(samples)), eltype(weights), eltype(probabilities))
    any(isnan, samples) && return fill(T(NaN), length(probabilities))

    positive_indices = findall(!iszero, weights)
    order = positive_indices[sortperm(view(samples, positive_indices))]
    values = samples[order]
    sorted_weights = weights[order]
    cumulative_weights = cumsum(sorted_weights)
    total_weight = last(cumulative_weights)
    first_weight = first(sorted_weights)
    zero_value = zero(T)

    return map(probabilities) do probability
        threshold = probability * (total_weight - first_weight) + first_weight
        upper = searchsortedlast(cumulative_weights, threshold) + 1
        upper > length(values) && return oftype(zero_value, last(values))
        lower = upper - 1
        lower_weight = cumulative_weights[lower]
        fraction = (threshold - lower_weight) /
                   (cumulative_weights[upper] - lower_weight)
        return oftype(zero_value, values[lower] + fraction * (values[upper] - values[lower]))
    end
end

function _weighted_quantile(samples::AbstractMatrix{<:Real}, weights, probability::Real)
    return map(
        row -> _weighted_quantile(view(samples, row, :), weights, probability),
        axes(samples, 1),
    )
end

function _weighted_quantile(
    samples::AbstractMatrix{<:Real},
    weights,
    probabilities::AbstractVector{<:Real},
)
    rows = map(
        row -> _weighted_quantile(view(samples, row, :), weights, probabilities),
        axes(samples, 1),
    )
    return reduce(vcat, transpose.(rows))
end

function _weighted_quantile(samples::NamedTuple, weights, probabilities)
    summaries = map(
        sample -> _weighted_quantile(sample, weights, probabilities),
        values(samples),
    )
    return NamedTuple{keys(samples)}(summaries)
end

function _weighted_quantile(samples, weights, probabilities)
    throw(ArgumentError("quantiles require real-valued sample storage"))
end

_sqrt_summary(value::Number) = sqrt(value)
_sqrt_summary(value::AbstractArray) = sqrt.(value)

function _sqrt_summary(value::NamedTuple)
    summaries = map(_sqrt_summary, values(value))
    return NamedTuple{keys(value)}(summaries)
end

"""
    Statistics.mean(samples::UnweightedSamples)
    Statistics.mean(f, samples::UnweightedSamples)

Return an ordinary unweighted expectation over logical samples.
"""
function Statistics.mean(samples::UnweightedSamples)
    device = _storage_device(_first_sample_leaf(samples.samples))
    return _with_backend_device(device) do
        _unweighted_mean(samples.samples)
    end
end

function Statistics.mean(f, samples::UnweightedSamples)
    device = _storage_device(_first_sample_leaf(samples.samples))
    return _with_backend_device(device) do
        if _is_host_storage(_first_sample_leaf(samples.samples))
            total = mapreduce(
                index -> f(_sample_at(samples.samples, index)),
                +,
                eachindex(Base.OneTo(length(samples))),
            )
            return total / length(samples)
        end
        return Statistics.mean(_device_functional_values(f, samples))
    end
end

"""
    Statistics.var(samples::UnweightedSamples; corrected=true)
    Statistics.var(f, samples::UnweightedSamples; corrected=true)

Return the ordinary unweighted sample variance.
"""
function Statistics.var(samples::UnweightedSamples; corrected::Bool=true)
    device = _storage_device(_first_sample_leaf(samples.samples))
    return _with_backend_device(device) do
        _unweighted_variance(samples.samples, corrected)
    end
end

function Statistics.var(f, samples::UnweightedSamples; corrected::Bool=true)
    device = _storage_device(_first_sample_leaf(samples.samples))
    return _with_backend_device(device) do
        if _is_host_storage(_first_sample_leaf(samples.samples))
            center = Statistics.mean(f, samples)
            total = mapreduce(
                index -> abs2(f(_sample_at(samples.samples, index)) - center),
                +,
                eachindex(Base.OneTo(length(samples))),
            )
            return total / (length(samples) - Int(corrected))
        end
        return Statistics.var(
            _device_functional_values(f, samples);
            corrected=corrected,
        )
    end
end

function Statistics.std(samples::UnweightedSamples; corrected::Bool=true)
    return _sqrt_summary(Statistics.var(samples; corrected=corrected))
end

function Statistics.std(f, samples::UnweightedSamples; corrected::Bool=true)
    return sqrt(Statistics.var(f, samples; corrected=corrected))
end

function Statistics.cov(samples::UnweightedSamples; corrected::Bool=true)
    device = _storage_device(_first_sample_leaf(samples.samples))
    return _with_backend_device(device) do
        _unweighted_covariance(samples.samples, corrected)
    end
end

function Statistics.quantile(samples::UnweightedSamples, probability)
    _is_host_storage(_first_sample_leaf(samples.samples)) || throw(
        ArgumentError(
            "quantiles are unavailable for device-resident unweighted samples; " *
            "transfer the samples to CPU first",
        ),
    )
    return _unweighted_quantile(samples.samples, probability)
end

Statistics.median(samples::UnweightedSamples) = Statistics.quantile(samples, 0.5)

_unweighted_mean(samples::AbstractVector) = Statistics.mean(samples)
_unweighted_mean(samples::AbstractMatrix) = vec(Statistics.mean(samples; dims=2))
function _unweighted_mean(samples::NamedTuple)
    summaries = map(_unweighted_mean, values(samples))
    return NamedTuple{keys(samples)}(summaries)
end

_unweighted_variance(samples::AbstractVector, corrected) =
    Statistics.var(samples; corrected=corrected)
_unweighted_variance(samples::AbstractMatrix, corrected) =
    vec(Statistics.var(samples; dims=2, corrected=corrected))
function _unweighted_variance(samples::NamedTuple, corrected)
    summaries = map(
        sample -> _unweighted_variance(sample, corrected),
        values(samples),
    )
    return NamedTuple{keys(samples)}(summaries)
end

_unweighted_covariance(samples::AbstractMatrix, corrected) =
    Statistics.cov(samples; dims=2, corrected=corrected)
_unweighted_covariance(samples, corrected) =
    throw(ArgumentError("covariance requires vector-valued samples"))

_unweighted_quantile(samples::AbstractVector{<:Real}, probability) =
    Statistics.quantile(samples, probability)
function _unweighted_quantile(samples::AbstractMatrix{<:Real}, probability)
    rows = map(
        row -> Statistics.quantile(view(samples, row, :), probability),
        axes(samples, 1),
    )
    probability isa Real && return rows
    return reduce(vcat, transpose.(rows))
end
function _unweighted_quantile(samples::NamedTuple, probability)
    summaries = map(
        sample -> _unweighted_quantile(sample, probability),
        values(samples),
    )
    return NamedTuple{keys(samples)}(summaries)
end
_unweighted_quantile(samples, probability) =
    throw(ArgumentError("quantiles require real-valued sample storage"))
