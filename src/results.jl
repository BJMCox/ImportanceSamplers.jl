abstract type _AbstractWeightedSamples{R} end

mutable struct _ValidatedResultToken end
const _VALIDATED_RESULT_TOKEN = _ValidatedResultToken()

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
"""
struct WeightedSampleView{R,S,W<:AbstractVector,P<:NamedTuple} <:
       _AbstractWeightedSamples{R}
    samples::S
    logweights::W
    provenance::P

    function WeightedSampleView{R}(
        samples::S,
        logweights::W,
        provenance::P,
        token::_ValidatedResultToken,
    ) where {R,S,W<:AbstractVector,P<:NamedTuple}
        token === _VALIDATED_RESULT_TOKEN || throw(
            ArgumentError("invalid internal view-construction token"),
        )
        return new{R,S,W,P}(samples, logweights, provenance)
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
    provenance isa NamedTuple || throw(
        ArgumentError("provenance must be a named tuple of aligned arrays"),
    )
    diagnostics isa NamedTuple || throw(
        ArgumentError("diagnostics must be a named tuple"),
    )

    _validate_numeric_storage(samples, "sample")
    sample_count = _sample_count(samples)
    sample_count > 0 || throw(ArgumentError("weighted results must contain samples"))
    length(logweights) == sample_count || throw(
        ArgumentError("sample and log-weight counts must match"),
    )
    _validate_logweights(logweights)
    _validate_provenance(provenance, sample_count)
    _validate_diagnostics(diagnostics)

    R = _result_record_type(samples, logweights, provenance)
    return WeightedSamples{R}(
        samples,
        logweights,
        provenance,
        diagnostics,
        _VALIDATED_RESULT_TOKEN,
    )
end

function _validate_numeric_storage(storage::AbstractVector, label)
    Base.require_one_based_indexing(storage)
    T = eltype(storage)
    T <: Number && isconcretetype(T) || throw(
        ArgumentError("$label vector leaves must have a concrete Number element type"),
    )
    return nothing
end

function _validate_numeric_storage(storage::AbstractMatrix, label)
    Base.require_one_based_indexing(storage)
    T = eltype(storage)
    T <: Number && isconcretetype(T) || throw(
        ArgumentError("$label matrix leaves must have a concrete Number element type"),
    )
    return nothing
end

function _validate_numeric_storage(storage::NamedTuple, label)
    isempty(storage) && throw(
        ArgumentError("$label named tuples must contain numeric leaves"),
    )
    for leaf in values(storage)
        _validate_numeric_storage(leaf, label)
    end
    return nothing
end

function _validate_numeric_storage(storage, label)
    throw(
        ArgumentError(
            "$label storage must be a numeric vector, numeric matrix, or nonempty named-tuple tree",
        ),
    )
end

function _validate_logweights(logweights)
    Base.require_one_based_indexing(logweights)
    T = eltype(logweights)
    T <: AbstractFloat || throw(
        ArgumentError("log weights must have an AbstractFloat element type"),
    )
    isconcretetype(T) || throw(
        ArgumentError("log-weight element type $T must be concrete"),
    )
    for logweight in logweights
        (isnan(logweight) || logweight == Inf) && throw(
            ArgumentError("log weights may not contain NaN or +Inf"),
        )
    end
    return nothing
end

_validate_provenance(::NamedTuple{(),Tuple{}}, sample_count) = nothing

function _validate_provenance(provenance::NamedTuple, sample_count)
    _validate_numeric_storage(provenance, "provenance")
    _sample_count(provenance) == sample_count || throw(
        ArgumentError("provenance and sample counts must match"),
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
    checkbounds(result.logweights, sample_index)
    return _result_record(
        result.samples,
        result.logweights,
        result.provenance,
        sample_index,
    )
end

_provenance_at(::NamedTuple{(),Tuple{}}, sample_index) = NamedTuple()
_provenance_at(provenance::NamedTuple, sample_index) =
    _sample_at(provenance, sample_index)

Base.iterate(result::WeightedSamples) = (result[1], 2)

function Base.iterate(result::WeightedSampleView)
    length(result) == 0 && return nothing
    return (result[1], 2)
end

function Base.iterate(result::_AbstractWeightedSamples, state::Int)
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
    R = _result_record_type(viewed_samples, viewed_logweights, viewed_provenance)
    return WeightedSampleView{R}(
        viewed_samples,
        viewed_logweights,
        viewed_provenance,
        _VALIDATED_RESULT_TOKEN,
    )
end

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
estimator.
"""
function lognormalizer(result::WeightedSamples)
    logweight_sum = LogExpFunctions.logsumexp(result.logweights)
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
weights.
"""
function normalized_weights(result::_AbstractWeightedSamples)
    logweight_sum = LogExpFunctions.logsumexp(result.logweights)
    logweight_sum == -Inf && throw(AllZeroWeightsError())
    return exp.(result.logweights .- logweight_sum)
end
