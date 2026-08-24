abstract type _AbstractWeightedSamples{R} end

mutable struct _ValidatedResultToken end
const _VALIDATED_RESULT_TOKEN = _ValidatedResultToken()

mutable struct _ResultTransferCounter
    count::Int
    bytes::Int
end

struct _LogSumExpAccumulator{T<:AbstractFloat}
    maximum::T
    scaled_sum::T
end

@inline _logsumexp_accumulator(value::T) where {T<:AbstractFloat} =
    _LogSumExpAccumulator(value, value == -Inf ? zero(T) : one(T))

@inline function _merge_logsumexp_accumulators(
    left::_LogSumExpAccumulator{T},
    right::_LogSumExpAccumulator{T},
) where {T}
    if left.maximum >= right.maximum
        left.maximum == -Inf && return left
        return _LogSumExpAccumulator(
            left.maximum,
            left.scaled_sum + right.scaled_sum * exp(right.maximum - left.maximum),
        )
    end
    return _LogSumExpAccumulator(
        right.maximum,
        right.scaled_sum + left.scaled_sum * exp(left.maximum - right.maximum),
    )
end

@inline function _finish_logsumexp(accumulator::_LogSumExpAccumulator)
    accumulator.maximum == -Inf && return accumulator.maximum
    return accumulator.maximum + log(accumulator.scaled_sum)
end

function _record_scalar_transfer!(counter::_ResultTransferCounter, ::Type{T}) where {T}
    counter.count += 1
    counter.bytes += sizeof(T)
    return nothing
end

_record_device_scalar_transfer!(counter, storage, type::Type) =
    _is_host_storage(storage) ? nothing : _record_scalar_transfer!(counter, type)

function _result_transfer_counter(diagnostics::NamedTuple)
    if !hasproperty(diagnostics, :transfers)
        return _ResultTransferCounter(0, 0)
    end
    transfers = diagnostics.transfers
    transfers isa _ResultTransferCounter && return transfers
    transfers isa NamedTuple &&
        hasproperty(transfers, :count) &&
        hasproperty(transfers, :bytes) || throw(
        ArgumentError("diagnostic transfers must contain count and bytes"),
    )
    count = transfers.count
    bytes = transfers.bytes
    count isa Int && bytes isa Int && count >= 0 && bytes >= 0 || throw(
        ArgumentError("diagnostic transfer count and bytes must be nonnegative Int values"),
    )
    return _ResultTransferCounter(count, bytes)
end

function _with_result_transfer_counter(diagnostics::NamedTuple)
    return merge(diagnostics, (transfers=_result_transfer_counter(diagnostics),))
end

"""
    WeightedSamples(samples, logweights; provenance=NamedTuple(), diagnostics=NamedTuple())

Construct an owned, complete weighted-sample result.

`samples` is a numeric vector, numeric matrix, or nonempty named-tuple tree of
those arrays. Its final logical axis must align with the raw, unnormalized
`logweights`. Plain importance sampling normally constructs this type through
[`importance_sample`](@ref) or [`importance_sample!`](@ref).

Scalar indexing and iteration return aligned named tuples with `sample`,
`logweight`, and `provenance` fields. Range, integer-vector, and Boolean-mask
indexing return a [`WeightedSampleView`](@ref).

Apply an `MLDataDevices.AbstractDevice` to a complete result to create an
independent copy on that device. Device-resident results must be transferred to
CPU before scalar indexing, iteration, quantiles, or medians.
"""
struct WeightedSamples{R,S,W<:AbstractVector,P<:NamedTuple,D<:NamedTuple} <:
       _AbstractWeightedSamples{R}
    samples::S
    logweights::W
    provenance::P
    diagnostics::D

    function WeightedSamples{R}(
        samples::S,
        logweights::W,
        provenance::P,
        diagnostics::D,
        token::_ValidatedResultToken,
    ) where {R,S,W<:AbstractVector,P<:NamedTuple,D<:NamedTuple}
        token === _VALIDATED_RESULT_TOKEN || throw(
            ArgumentError("invalid internal result-construction token"),
        )
        return new{R,S,W,P,D}(samples, logweights, provenance, diagnostics)
    end
end

"""
    WeightedSampleView

An aligned descriptive view of a subset of weighted samples.

Views support indexing, iteration, and [`normalized_weights`](@ref), but not
[`lognormalizer`](@ref), because an arbitrary subset is not a complete
estimator result. Construct views by indexing a [`WeightedSamples`](@ref).
Device-resident views retain device storage. Apply an
`MLDataDevices.AbstractDevice` directly to create an independent aligned copy;
transfer to CPU before scalar indexing or iteration.
"""
struct WeightedSampleView{R,S,W<:AbstractVector,P<:NamedTuple} <:
       _AbstractWeightedSamples{R}
    samples::S
    logweights::W
    provenance::P
    transfers::_ResultTransferCounter

    function WeightedSampleView{R}(
        samples::S,
        logweights::W,
        provenance::P,
        transfers::_ResultTransferCounter,
        token::_ValidatedResultToken,
    ) where {R,S,W<:AbstractVector,P<:NamedTuple}
        token === _VALIDATED_RESULT_TOKEN || throw(
            ArgumentError("invalid internal view-construction token"),
        )
        return new{R,S,W,P}(samples, logweights, provenance, transfers)
    end
end

"""
    AllZeroWeightsError

Exception thrown when normalized weights are requested but every raw log
weight is `-Inf`. The samples remain available and [`lognormalizer`](@ref)
returns `-Inf`.
"""
struct AllZeroWeightsError <: Exception end

function Base.showerror(io::IO, ::AllZeroWeightsError)
    print(io, "normalized weights are undefined because all raw weights are zero")
end

function WeightedSamples(
    samples,
    logweights::AbstractVector;
    provenance=NamedTuple(),
    diagnostics=NamedTuple(),
)
    owned_samples = deepcopy(samples)
    owned_logweights = deepcopy(logweights)
    owned_provenance = deepcopy(provenance)
    owned_diagnostics = deepcopy(diagnostics)
    return _adopt_weighted_samples(
        owned_samples,
        owned_logweights;
        provenance=owned_provenance,
        diagnostics=owned_diagnostics,
    )
end

function _adopt_weighted_samples(
    samples,
    logweights::AbstractVector;
    provenance=NamedTuple(),
    diagnostics=NamedTuple(),
)
    return _adopt_weighted_samples(
        samples,
        logweights,
        Val(false);
        provenance=provenance,
        diagnostics=diagnostics,
    )
end

function _adopt_validated_weighted_samples(
    samples,
    logweights::AbstractVector;
    provenance=NamedTuple(),
    diagnostics=NamedTuple(),
)
    return _adopt_weighted_samples(
        samples,
        logweights,
        Val(true);
        provenance=provenance,
        diagnostics=diagnostics,
    )
end

function _adopt_weighted_samples(
    samples,
    logweights::AbstractVector,
    logweights_validated::Val;
    provenance,
    diagnostics,
)
    provenance isa NamedTuple || throw(
        ArgumentError("provenance must be a named tuple of aligned arrays"),
    )
    diagnostics isa NamedTuple || throw(
        ArgumentError("diagnostics must be a named tuple"),
    )

    sample_device = _validate_numeric_storage(samples, "sample")
    sample_count = _sample_count(samples)
    sample_count > 0 || throw(ArgumentError("weighted results must contain samples"))
    length(logweights) == sample_count || throw(
        ArgumentError("sample and log-weight counts must match"),
    )
    logweight_device = _storage_device(logweights)
    provenance_device = _validate_provenance(provenance, sample_count)
    _require_aligned_storage_device(
        sample_device,
        logweight_device,
        provenance_device,
    )
    diagnostics = _with_result_transfer_counter(diagnostics)
    _validate_logweights(logweights, diagnostics.transfers, logweights_validated)
    _validate_diagnostics(diagnostics)

    return _new_weighted_samples(samples, logweights, provenance, diagnostics)
end

function (device::MLDataDevices.AbstractDevice)(result::WeightedSamples)
    samples = _transfer_result_storage(device, result.samples)
    logweights = _transfer_result_storage(device, result.logweights)
    provenance = _transfer_result_storage(device, result.provenance)
    diagnostics = _transfer_result_storage(device, result.diagnostics)
    return _new_weighted_samples(samples, logweights, provenance, diagnostics)
end

function (device::MLDataDevices.AbstractDevice)(result::WeightedSampleView)
    samples = _transfer_result_storage(device, result.samples)
    logweights = _transfer_result_storage(device, result.logweights)
    provenance = _transfer_result_storage(device, result.provenance)
    transfers = _transfer_result_storage(device, result.transfers)
    return _new_weighted_sample_view(samples, logweights, provenance, transfers)
end

function _new_weighted_sample_view(samples, logweights, provenance, transfers)
    R = _result_record_type(samples, logweights, provenance)
    return WeightedSampleView{R}(
        samples,
        logweights,
        provenance,
        transfers,
        _VALIDATED_RESULT_TOKEN,
    )
end

function _transfer_result_storage(device, storage::AbstractArray)
    transferred = device(storage)
    return transferred === storage ? copy(transferred) : transferred
end

function _transfer_result_storage(device, storage::NamedTuple)
    leaves = map(value -> _transfer_result_storage(device, value), values(storage))
    return NamedTuple{keys(storage)}(leaves)
end

_transfer_result_storage(device, storage::Tuple) =
    map(value -> _transfer_result_storage(device, value), storage)
_transfer_result_storage(device, counter::_ResultTransferCounter) =
    _ResultTransferCounter(counter.count, counter.bytes)
_transfer_result_storage(device, value) = value

function _new_weighted_samples(samples, logweights, provenance, diagnostics)
    R = _result_record_type(samples, logweights, provenance)
    return WeightedSamples{R}(
        samples,
        logweights,
        provenance,
        diagnostics,
        _VALIDATED_RESULT_TOKEN,
    )
end

function _validate_numeric_storage(storage::AbstractArray{T,N}, label) where {T,N}
    Base.require_one_based_indexing(storage)
    N in (1, 2) || throw(
        ArgumentError("$label array leaves must be vectors or matrices"),
    )
    T <: Number && isconcretetype(T) || throw(
        ArgumentError("$label array leaves must have a concrete Number element type"),
    )
    return _storage_device(storage)
end

function _validate_numeric_storage(storage::NamedTuple, label)
    isempty(storage) && throw(
        ArgumentError("$label named tuples must contain numeric leaves"),
    )
    devices = map(leaf -> _validate_numeric_storage(leaf, label), values(storage))
    return _combine_storage_devices(devices)
end

function _validate_numeric_storage(storage, label)
    throw(
        ArgumentError(
            "$label storage must be a numeric vector, numeric matrix, or nonempty named-tuple tree",
        ),
    )
end

function _validate_logweights(logweights, transfers::_ResultTransferCounter, ::Val{V}) where {V}
    Base.require_one_based_indexing(logweights)
    T = eltype(logweights)
    T <: AbstractFloat || throw(
        ArgumentError("log weights must have an AbstractFloat element type"),
    )
    isconcretetype(T) || throw(
        ArgumentError("log-weight element type $T must be concrete"),
    )
    V && return nothing
    has_invalid = mapreduce(
        logweight -> isnan(logweight) || logweight == Inf,
        |,
        logweights;
        init=false,
    )
    _record_device_scalar_transfer!(transfers, logweights, Bool)
    has_invalid && throw(ArgumentError("log weights may not contain NaN or +Inf"))
    return nothing
end

_validate_provenance(::NamedTuple{(),Tuple{}}, sample_count) = nothing

function _validate_provenance(provenance::NamedTuple, sample_count)
    device = _validate_numeric_storage(provenance, "provenance")
    _sample_count(provenance) == sample_count || throw(
        ArgumentError("provenance and sample counts must match"),
    )
    return device
end

function _require_aligned_storage_device(sample_device, logweight_device, provenance_device)
    devices = isnothing(provenance_device) ?
              (sample_device, logweight_device) :
              (sample_device, logweight_device, provenance_device)
    all(==(first(devices)), devices) || throw(
        ArgumentError("samples, log weights, and provenance must use the same device"),
    )
    return nothing
end

function _validate_diagnostics(diagnostics::NamedTuple)
    foreach(_validate_diagnostic_value, values(diagnostics))
    return nothing
end

function _validate_diagnostic_value(value::NamedTuple)
    return _validate_diagnostics(value)
end

function _validate_diagnostic_value(value::Tuple)
    foreach(_validate_diagnostic_value, value)
    return nothing
end

function _validate_diagnostic_value(value::AbstractArray)
    Base.require_one_based_indexing(value)
    T = eltype(value)
    _safe_diagnostic_leaf_type(T) || throw(
        ArgumentError(
            "diagnostic arrays must have a concrete immutable scalar element type",
        ),
    )
    return nothing
end

function _validate_diagnostic_value(value::_ResultTransferCounter)
    value.count >= 0 && value.bytes >= 0 || throw(
        ArgumentError("diagnostic transfer count and bytes must be nonnegative"),
    )
    return nothing
end

function _validate_diagnostic_value(value)
    _safe_diagnostic_leaf_type(typeof(value)) || throw(
        ArgumentError(
            "diagnostic values must be immutable scalars, arrays of such scalars, tuples, or named tuples",
        ),
    )
    return nothing
end

function _safe_diagnostic_leaf_type(::Type{T}) where {T}
    isconcretetype(T) || return false
    T <: AbstractString || T <: Symbol || T <: Char || T <: Missing ||
        T <: Nothing || return T <: Number && !ismutabletype(T)
    return true
end

Base.length(result::_AbstractWeightedSamples) = length(result.logweights)
Base.firstindex(::_AbstractWeightedSamples) = 1
Base.lastindex(result::_AbstractWeightedSamples) = length(result)
Base.IteratorSize(::Type{<:_AbstractWeightedSamples}) = Base.HasLength()
Base.IteratorEltype(::Type{<:_AbstractWeightedSamples}) = Base.HasEltype()
Base.eltype(::Type{<:_AbstractWeightedSamples{R}}) where {R} = R

@inline function _result_record(samples, logweights, provenance, sample_index)
    return (
        sample=_sample_at(samples, sample_index),
        logweight=logweights[sample_index],
        provenance=_provenance_at(provenance, sample_index),
    )
end

function _result_record_type(samples, logweights, provenance)
    R = Base.promote_op(
        _result_record,
        typeof(samples),
        typeof(logweights),
        typeof(provenance),
        Int,
    )
    isconcretetype(R) || error("could not infer the logical result record type")
    return R
end

function Base.getindex(
    result::_AbstractWeightedSamples{R},
    sample_index::Integer,
)::R where {R}
    _require_host_result_access(result)
    checkbounds(result.logweights, sample_index)
    return _result_record(
        result.samples,
        result.logweights,
        result.provenance,
        sample_index,
    )
end

function _require_host_result_access(result::_AbstractWeightedSamples)
    _is_host_storage(result.logweights) || throw(
        ArgumentError(
            "scalar indexing and iteration are unavailable for device-resident " *
            "weighted samples; transfer the result to CPU first",
        ),
    )
    return nothing
end

_provenance_at(::NamedTuple{(),Tuple{}}, sample_index) = NamedTuple()
_provenance_at(provenance::NamedTuple, sample_index) =
    _sample_at(provenance, sample_index)

function Base.iterate(result::WeightedSamples)
    return (result[1], 2)
end

function Base.iterate(result::WeightedSampleView)
    if length(result) == 0
        _require_host_result_access(result)
        return nothing
    end
    return (result[1], 2)
end

function Base.iterate(result::_AbstractWeightedSamples, state::Int)
    _require_host_result_access(result)
    state > length(result) && return nothing
    return (result[state], state + 1)
end

function Base.getindex(
    result::_AbstractWeightedSamples,
    sample_indices::AbstractRange{<:Integer},
)
    return _weighted_sample_view(result, sample_indices)
end

function Base.getindex(
    result::_AbstractWeightedSamples,
    sample_indices::AbstractRange{Bool},
)
    throw(ArgumentError("Boolean ranges are not valid result selectors"))
end

function Base.getindex(
    result::_AbstractWeightedSamples,
    sample_indices::AbstractVector{<:Integer},
)
    Base.require_one_based_indexing(sample_indices)
    frozen_indices = copy(sample_indices)
    return _weighted_sample_view(result, frozen_indices)
end

function Base.getindex(
    result::_AbstractWeightedSamples,
    sample_mask::AbstractVector{Bool},
)
    length(sample_mask) == length(result) || throw(BoundsError(result, sample_mask))
    Base.require_one_based_indexing(sample_mask)
    frozen_mask = copy(sample_mask)
    return _weighted_sample_view(result, frozen_mask)
end

function _weighted_sample_view(result, sample_indices)
    viewed_samples = _sample_view(result.samples, sample_indices)
    viewed_logweights = view(result.logweights, sample_indices)
    viewed_provenance = _provenance_view(result.provenance, sample_indices)
    return _new_weighted_sample_view(
        viewed_samples,
        viewed_logweights,
        viewed_provenance,
        _result_transfers(result),
    )
end

_result_transfers(result::WeightedSamples) = result.diagnostics.transfers
_result_transfers(result::WeightedSampleView) = result.transfers

_sample_view(samples::AbstractVector, sample_indices) = view(samples, sample_indices)
_sample_view(samples::AbstractMatrix, sample_indices) = view(samples, :, sample_indices)

function _sample_view(samples::NamedTuple, sample_indices)
    viewed_values = map(
        samples_leaf -> _sample_view(samples_leaf, sample_indices),
        values(samples),
    )
    return NamedTuple{keys(samples)}(viewed_values)
end

_provenance_view(::NamedTuple{(),Tuple{}}, sample_indices) = NamedTuple()
_provenance_view(provenance::NamedTuple, sample_indices) =
    _sample_view(provenance, sample_indices)

"""
    lognormalizer(result::WeightedSamples)

Return `logsumexp(result.logweights) - log(length(result))`, the log of the
plain-importance-sampling normalizer estimate.

This is Bayesian log evidence only if the supplied target retains every
required normalizing constant. It returns `-Inf` when all raw weights are zero.
It is unavailable on [`WeightedSampleView`](@ref), which is not a complete
estimator. Device-resident complete results use a device reduction and transfer
only its two-scalar log-sum-exp accumulator.
"""
function lognormalizer(result::WeightedSamples)
    logweight_sum = _logweight_sum(result)
    logweight_sum == -Inf && return logweight_sum
    T = eltype(result.logweights)
    return logweight_sum - log(T(length(result)))
end

function lognormalizer(::WeightedSampleView)
    throw(
        ArgumentError(
            "lognormalizer is unavailable on a view because it is not a complete estimator",
        ),
    )
end

"""
    normalized_weights(result)

Derive an array of weights that sums to one from a complete result or
descriptive view.

The raw `result.logweights` remain unchanged. If every raw log weight is
`-Inf`, throw [`AllZeroWeightsError`](@ref) rather than inventing uniform
weights. For a device-resident result, the returned weight array stays on the
same device.
"""
function normalized_weights(result::_AbstractWeightedSamples)
    logweight_sum = _logweight_sum(result)
    logweight_sum == -Inf && throw(AllZeroWeightsError())
    return exp.(result.logweights .- logweight_sum)
end

function _logweight_sum(result::_AbstractWeightedSamples)
    _is_host_storage(result.logweights) &&
        return LogExpFunctions.logsumexp(result.logweights)
    T = eltype(result.logweights)
    accumulator = mapreduce(
        _logsumexp_accumulator,
        _merge_logsumexp_accumulators,
        result.logweights;
        init=_LogSumExpAccumulator(T(-Inf), zero(T)),
    )
    _record_device_scalar_transfer!(
        _result_transfers(result),
        result.logweights,
        typeof(accumulator),
    )
    return _finish_logsumexp(accumulator)
end
