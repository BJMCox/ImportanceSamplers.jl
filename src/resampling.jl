"""
    AbstractResamplingMethod

Abstract supertype for concrete resampling algorithms.
"""
abstract type AbstractResamplingMethod end

"""
    MultinomialResampling()

Independent weighted sampling with replacement.
"""
struct MultinomialResampling <: AbstractResamplingMethod end

mutable struct _ValidatedUnweightedSamplesToken end
const _VALIDATED_UNWEIGHTED_SAMPLES_TOKEN = _ValidatedUnweightedSamplesToken()

"""
    UnweightedSamples(samples)

An owned batch of unweighted samples. Logical samples occupy the final array
axis and support scalar indexing and iteration on CPU storage.
"""
struct UnweightedSamples{R,S}
    samples::S

    function UnweightedSamples{R}(
        samples::S,
        token::_ValidatedUnweightedSamplesToken,
    ) where {R,S}
        token === _VALIDATED_UNWEIGHTED_SAMPLES_TOKEN || throw(
            ArgumentError("invalid internal unweighted-sample construction token"),
        )
        return new{R,S}(samples)
    end
end

UnweightedSamples(samples) = _adopt_unweighted_samples(deepcopy(samples))

function _adopt_unweighted_samples(samples)
    _validate_numeric_storage(samples, "sample")
    _sample_count(samples) > 0 || throw(
        ArgumentError("unweighted results must contain samples"),
    )
    R = Base.promote_op(_sample_at, typeof(samples), Int)
    isconcretetype(R) || error("could not infer the logical sample type")
    return UnweightedSamples{R}(samples, _VALIDATED_UNWEIGHTED_SAMPLES_TOKEN)
end

function (device::MLDataDevices.AbstractDevice)(samples::UnweightedSamples)
    return _adopt_unweighted_samples(
        _transfer_result_storage(device, samples.samples),
    )
end

Base.length(samples::UnweightedSamples) = _sample_count(samples.samples)
Base.firstindex(::UnweightedSamples) = 1
Base.lastindex(samples::UnweightedSamples) = length(samples)
Base.IteratorSize(::Type{<:UnweightedSamples}) = Base.HasLength()
Base.IteratorEltype(::Type{<:UnweightedSamples}) = Base.HasEltype()
Base.eltype(::Type{<:UnweightedSamples{R}}) where {R} = R

function Base.getindex(samples::UnweightedSamples{R}, index::Integer)::R where {R}
    _is_host_storage(_first_sample_leaf(samples.samples)) || throw(
        ArgumentError(
            "scalar indexing and iteration are unavailable for device-resident " *
            "unweighted samples; transfer the samples to CPU first",
        ),
    )
    checkbounds(Base.OneTo(length(samples)), index)
    return _sample_at(samples.samples, index)
end

Base.iterate(samples::UnweightedSamples) = (samples[1], 2)
function Base.iterate(samples::UnweightedSamples, state::Int)
    state > length(samples) && return nothing
    return (samples[state], state + 1)
end

_first_sample_leaf(samples::AbstractArray) = samples
_first_sample_leaf(samples::NamedTuple) = _first_sample_leaf(first(values(samples)))

_allocate_resampled_storage(samples::AbstractVector, count) =
    similar(samples, eltype(samples), count)
_allocate_resampled_storage(samples::AbstractMatrix, count) =
    similar(samples, eltype(samples), size(samples, 1), count)
function _allocate_resampled_storage(samples::NamedTuple, count)
    leaves = map(sample -> _allocate_resampled_storage(sample, count), values(samples))
    return NamedTuple{keys(samples)}(leaves)
end

@kernel function _finalize_resampling_cdf_kernel!(cdf, last_index)
    @inbounds cdf[last_index] = one(eltype(cdf))
end

@kernel function _select_resampling_ancestors_kernel!(ancestors, uniforms, cdf, last_index)
    output_index = @index(Global, Linear)
    uniform = @inbounds uniforms[output_index]
    first = 1
    last = last_index
    while first < last
        middle = first + ((last - first) >> 1)
        if uniform < @inbounds(cdf[middle])
            last = middle
        else
            first = middle + 1
        end
    end
    @inbounds ancestors[output_index] = first
end

@inline function _gather_resampled_sample!(
    destination::AbstractVector,
    source::AbstractVector,
    output_index,
    ancestor,
)
    @inbounds destination[output_index] = source[ancestor]
    return nothing
end

@inline function _gather_resampled_sample!(
    destination::AbstractMatrix,
    source::AbstractMatrix,
    output_index,
    ancestor,
)
    for coordinate in axes(destination, 1)
        @inbounds destination[coordinate, output_index] = source[coordinate, ancestor]
    end
    return nothing
end


@generated function _gather_resampled_sample!(
    destination::NamedTuple{Names},
    source::NamedTuple{Names},
    output_index,
    ancestor,
) where {Names}
    calls = map(Names) do name
        :(_gather_resampled_sample!(
            getfield(destination, $(QuoteNode(name))),
            getfield(source, $(QuoteNode(name))),
            output_index,
            ancestor,
        ))
    end
    return Expr(:block, calls..., :(nothing))
end

@kernel function _gather_resampled_samples_kernel!(destination, source, ancestors)
    output_index = @index(Global, Linear)
    ancestor = @inbounds ancestors[output_index]
    _gather_resampled_sample!(destination, source, output_index, ancestor)
end

function _resample_and_gather!(
    cdf,
    uniforms,
    ancestors,
    source,
    destination,
    execution,
)
    backend = KernelAbstractions.get_backend(cdf)
    last_index = length(cdf)
    finalize = _finalize_resampling_cdf_kernel!(backend)
    finalize(cdf, last_index; ndrange=1, workgroupsize=1)
    select = _select_resampling_ancestors_kernel!(backend)
    select(
        ancestors,
        uniforms,
        cdf,
        last_index;
        ndrange=length(ancestors),
        workgroupsize=_native_workgroupsize(execution, length(ancestors)),
    )
    gather = _gather_resampled_samples_kernel!(backend)
    gather(
        destination,
        source,
        ancestors;
        ndrange=length(ancestors),
        workgroupsize=_native_workgroupsize(execution, length(ancestors)),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

function _resampling_cdf!(
    cdf,
    logweights,
    transfers::_ResultTransferCounter=_ResultTransferCounter(0, 0),
)
    maximum_logweight = maximum(logweights)
    _record_device_scalar_transfer!(
        transfers,
        logweights,
        eltype(logweights),
        Val(:cdf_maximum),
    )
    maximum_logweight == -Inf && throw(AllZeroWeightsError())
    cdf .= exp.(logweights .- maximum_logweight)
    total = sum(cdf)
    _record_device_scalar_transfer!(
        transfers,
        cdf,
        eltype(cdf),
        Val(:cdf_sum),
    )
    isfinite(total) && total > zero(total) || throw(AllZeroWeightsError())
    cdf ./= total
    cumsum!(cdf, cdf)
    return cdf
end

_resampling_rng(rng, ::MLDataDevices.AbstractCPUDevice) = rng
function _resampling_rng(rng, device::MLDataDevices.AbstractAcceleratorDevice)
    seed = try
        Random.rand(rng, UInt64)
    catch
        throw(SamplerDeviceError(device, :accelerator_rng_unavailable))
    end
    return _owned_backend_rng(device, seed)
end

"""
    resample(rng, result, count; method=MultinomialResampling())
    resample(rng, result; method=MultinomialResampling())

Draw unweighted samples with replacement from normalized importance weights.
The count-free form returns `length(result)` draws. Output stays on the input
device and carries no importance weights.
"""
function resample(
    rng::Random.AbstractRNG,
    result::_AbstractWeightedSamples,
    count::Int;
    method::AbstractResamplingMethod=MultinomialResampling(),
)
    count > 0 || throw(ArgumentError("resampling count must be positive"))
    return _resample(rng, result, count, method)
end

function resample(
    rng::Random.AbstractRNG,
    result::_AbstractWeightedSamples;
    method::AbstractResamplingMethod=MultinomialResampling(),
)
    return resample(rng, result, length(result); method=method)
end

function _resample(rng, result, count, ::MultinomialResampling)
    device = _storage_device(result.logweights)
    return _with_backend_device(device) do
        cdf = similar(result.logweights)
        uniforms = similar(result.logweights, count)
        ancestors = similar(result.logweights, Int, count)
        destination = _allocate_resampled_storage(result.samples, count)
        _resampling_cdf!(cdf, result.logweights, _result_transfers(result))
        Random.rand!(_resampling_rng(rng, device), uniforms)
        _resample_and_gather!(
            cdf,
            uniforms,
            ancestors,
            result.samples,
            destination,
            _KernelExecution(_ThreadedCPUExecution()),
        )
        return _adopt_unweighted_samples(destination)
    end
end
