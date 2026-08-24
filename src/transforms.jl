"""
    AbstractSampleTransform

Abstract supertype for transforms from unconstrained coordinates to a logical
sample value. Forward log Jacobians belong to the proposal's change-of-variables
density.
"""
abstract type AbstractSampleTransform end

"""
    IdentityTransform()

Leave an unconstrained scalar or vector unchanged with zero log Jacobian. A
named product layout inserts this transform for omitted known fields.
"""
struct IdentityTransform <: AbstractSampleTransform end

"""
    PositiveTransform()

Map an unconstrained scalar `z` to `exp(z)` on the positive real line, with
forward log Jacobian `z`.
"""
struct PositiveTransform <: AbstractSampleTransform end

"""
    SoftplusTransform()

Map an unconstrained scalar smoothly to the positive real line with `softplus`
and its stable forward log Jacobian.
"""
struct SoftplusTransform <: AbstractSampleTransform end

struct IntervalTransform{T,L,U} <: AbstractSampleTransform
    lower::L
    upper::U
end

"""
    SimplexTransform(K)

Transform `K - 1` unconstrained coordinates into `K` positive weights that
sum to one. The coordinates use an orthonormal embedding into the sum-zero
logit subspace. The forward Jacobian is measured against the first `K - 1`
simplex coordinates, `dx₁⋯dxₖ₋₁`, and includes the full constant:
`log|J| = 0.5log(K) + sum(log, x)`.
"""
struct SimplexTransform <: AbstractSampleTransform
    dimension::Int

    function SimplexTransform(dimension::Int)
        dimension >= 2 || throw(ArgumentError("simplex dimension must be at least two"))
        return new(dimension)
    end
end

function SimplexTransform(dimension)
    throw(ArgumentError("simplex dimension must be an Int"))
end

"""
    InvalidTransformError(reason, location)

Raised when a finite scalar transform input cannot produce a valid logical
value. `location` identifies the logical scalar or block without retaining a
sample array.
"""
struct InvalidTransformError <: Exception
    reason::Symbol
    location::Union{Nothing,Int,Symbol,UnitRange{Int}}
end

function InvalidTransformError(reason::Symbol, location=nothing)
    location isa Union{Nothing,Int,Symbol,UnitRange{Int}} || throw(
        ArgumentError(
            "transform location must be nothing, an integer, a symbol, or an integer range",
        ),
    )
    return InvalidTransformError(reason, location)
end

function Base.showerror(io::IO, error::InvalidTransformError)
    print(io, "invalid transform: ", error.reason)
    isnothing(error.location) || print(io, " at ", error.location)
end

const _TransformFloat = Union{Float32,Float64}
const _NativeScalarTransform = Union{
    IdentityTransform,
    PositiveTransform,
    SoftplusTransform,
    IntervalTransform,
}

const _NATIVE_TRANSFORM_NONFINITE_INPUT = UInt16(0x0001)
const _NATIVE_TRANSFORM_NONFINITE_OUTPUT = UInt16(0x0002)
const _NATIVE_TRANSFORM_NONFINITE_LOGJAC = UInt16(0x0004)
const _NATIVE_TRANSFORM_OUTSIDE_SUPPORT = UInt16(0x0008)

@inline _native_transform_success(x, logabsjac) = (x, logabsjac, UInt16(0))

@inline function _native_checked_transform_result(x::T, logabsjac::T) where {T}
    isfinite(x) || return (x, logabsjac, _NATIVE_TRANSFORM_NONFINITE_OUTPUT)
    isfinite(logabsjac) || return (x, logabsjac, _NATIVE_TRANSFORM_NONFINITE_LOGJAC)
    return _native_transform_success(x, logabsjac)
end

@inline function _native_inverse_with_logjac(::IdentityTransform, x::T) where {T<:_TransformFloat}
    isfinite(x) || return (x, zero(T), _NATIVE_TRANSFORM_NONFINITE_INPUT)
    return _native_transform_success(x, zero(T))
end

@inline function _native_inverse_with_logjac(::PositiveTransform, x::T) where {T<:_TransformFloat}
    isfinite(x) || return (x, x, _NATIVE_TRANSFORM_NONFINITE_INPUT)
    x > zero(T) || return (x, x, _NATIVE_TRANSFORM_OUTSIDE_SUPPORT)
    z = log(x)
    return _native_checked_transform_result(z, z)
end

@inline function _native_inverse_with_logjac(::SoftplusTransform, x::T) where {T<:_TransformFloat}
    isfinite(x) || return (x, x, _NATIVE_TRANSFORM_NONFINITE_INPUT)
    x > zero(T) || return (x, x, _NATIVE_TRANSFORM_OUTSIDE_SUPPORT)
    z = LogExpFunctions.logexpm1(x)
    return _native_checked_transform_result(z, z - x)
end

@inline function _native_inverse_with_logjac(
    transform::IntervalTransform{T,T,Nothing},
    x::T,
) where {T<:_TransformFloat}
    isfinite(x) || return (x, x, _NATIVE_TRANSFORM_NONFINITE_INPUT)
    x > transform.lower || return (x, x, _NATIVE_TRANSFORM_OUTSIDE_SUPPORT)
    z = log(x - transform.lower)
    return _native_checked_transform_result(z, z)
end

@inline function _native_inverse_with_logjac(
    transform::IntervalTransform{T,Nothing,T},
    x::T,
) where {T<:_TransformFloat}
    isfinite(x) || return (x, x, _NATIVE_TRANSFORM_NONFINITE_INPUT)
    x < transform.upper || return (x, x, _NATIVE_TRANSFORM_OUTSIDE_SUPPORT)
    z = log(transform.upper - x)
    return _native_checked_transform_result(z, z)
end

@inline function _native_transform_with_logjac(::IdentityTransform, z::T) where {T<:_TransformFloat}
    isfinite(z) || return (z, zero(T), _NATIVE_TRANSFORM_NONFINITE_INPUT)
    return _native_transform_success(z, zero(T))
end

@inline function _native_transform_with_logjac(::PositiveTransform, z::T) where {T<:_TransformFloat}
    isfinite(z) || return (z, z, _NATIVE_TRANSFORM_NONFINITE_INPUT)
    x = exp(z)
    isfinite(x) || return (x, z, _NATIVE_TRANSFORM_NONFINITE_OUTPUT)
    x > zero(T) || return (x, z, _NATIVE_TRANSFORM_OUTSIDE_SUPPORT)
    return _native_transform_success(x, z)
end

@inline function _native_transform_with_logjac(::SoftplusTransform, z::T) where {T<:_TransformFloat}
    isfinite(z) || return (z, z, _NATIVE_TRANSFORM_NONFINITE_INPUT)
    x = _softplus(z)
    logabsjac = z - x
    isfinite(x) || return (x, logabsjac, _NATIVE_TRANSFORM_NONFINITE_OUTPUT)
    isfinite(logabsjac) || return (x, logabsjac, _NATIVE_TRANSFORM_NONFINITE_LOGJAC)
    x > zero(T) || return (x, logabsjac, _NATIVE_TRANSFORM_OUTSIDE_SUPPORT)
    return _native_transform_success(x, logabsjac)
end

@inline function _native_transform_with_logjac(
    transform::IntervalTransform{T,T,Nothing},
    z::T,
) where {T<:_TransformFloat}
    distance, logabsjac, reason = _native_transform_with_logjac(PositiveTransform(), z)
    iszero(reason) || return (distance, logabsjac, reason)
    x = transform.lower + distance
    isfinite(x) || return (x, logabsjac, _NATIVE_TRANSFORM_NONFINITE_OUTPUT)
    x > transform.lower || return (x, logabsjac, _NATIVE_TRANSFORM_OUTSIDE_SUPPORT)
    return _native_transform_success(x, logabsjac)
end

@inline function _native_inverse_with_logjac(
    transform::IntervalTransform{T,T,T},
    x::T,
) where {T<:_TransformFloat}
    isfinite(x) || return (x, x, _NATIVE_TRANSFORM_NONFINITE_INPUT)
    transform.lower < x < transform.upper ||
        return (x, x, _NATIVE_TRANSFORM_OUTSIDE_SUPPORT)
    lower_logdistance = _log_positive_difference(x, transform.lower)
    upper_logdistance = _log_positive_difference(transform.upper, x)
    span_logdistance = _log_positive_difference(transform.upper, transform.lower)
    z = lower_logdistance - upper_logdistance
    logabsjac = lower_logdistance + upper_logdistance - span_logdistance
    return _native_checked_transform_result(z, logabsjac)
end

@inline function _native_transform_with_logjac(
    transform::IntervalTransform{T,Nothing,T},
    z::T,
) where {T<:_TransformFloat}
    distance, logabsjac, reason = _native_transform_with_logjac(PositiveTransform(), z)
    iszero(reason) || return (distance, logabsjac, reason)
    x = transform.upper - distance
    isfinite(x) || return (x, logabsjac, _NATIVE_TRANSFORM_NONFINITE_OUTPUT)
    x < transform.upper || return (x, logabsjac, _NATIVE_TRANSFORM_OUTSIDE_SUPPORT)
    return _native_transform_success(x, logabsjac)
end

@inline function _native_transform_with_logjac(
    transform::IntervalTransform{T,T,T},
    z::T,
) where {T<:_TransformFloat}
    isfinite(z) || return (z, z, _NATIVE_TRANSFORM_NONFINITE_INPUT)
    probability = _logistic(z)
    x = _bounded_interval_value(transform.lower, transform.upper, probability)
    logabsjac = _log_positive_difference(transform.upper, transform.lower) +
                _logsigmoid(z) + _logsigmoid(-z)
    x, logabsjac, reason = _native_checked_transform_result(x, logabsjac)
    iszero(reason) || return (x, logabsjac, reason)
    transform.lower < x < transform.upper ||
        return (x, logabsjac, _NATIVE_TRANSFORM_OUTSIDE_SUPPORT)
    return _native_transform_success(x, logabsjac)
end

function _validated_interval_endpoint(bound::Nothing, name)
    return nothing
end

function _validated_interval_endpoint(bound::T, name) where {T<:_TransformFloat}
    isfinite(bound) || throw(ArgumentError("$name must be finite"))
    return bound
end

function _validated_interval_endpoint(bound, name)
    throw(ArgumentError("$name must be nothing or a finite Float32 or Float64"))
end

"""
    IntervalTransform(lower, upper)

Construct a scalar transform onto the open interval defined by finite
`Float32` or `Float64` endpoints. Use `nothing` for the unbounded endpoint;
at least one endpoint is required. `(lower, nothing)`, `(nothing, upper)`, and
`(lower, upper)` represent lower-bounded, upper-bounded, and bounded support.
"""
function IntervalTransform(lower, upper)
    validated_lower = _validated_interval_endpoint(lower, "lower bound")
    validated_upper = _validated_interval_endpoint(upper, "upper bound")
    isnothing(validated_lower) && isnothing(validated_upper) && throw(
        ArgumentError("at least one interval endpoint must be finite"),
    )
    if !isnothing(validated_lower) && !isnothing(validated_upper)
        typeof(validated_lower) === typeof(validated_upper) || throw(
            ArgumentError("interval endpoints must have the same Float32 or Float64 type"),
        )
        validated_lower < validated_upper || throw(
            ArgumentError("lower bound must be strictly less than upper bound"),
        )
    end
    T = isnothing(validated_lower) ? typeof(validated_upper) : typeof(validated_lower)
    return IntervalTransform{T,typeof(validated_lower),typeof(validated_upper)}(
        validated_lower,
        validated_upper,
    )
end

@inline function _throw_invalid_transform(reason::Symbol)
    throw(InvalidTransformError(reason))
end

@inline function _native_transform_failure_reason(reason_bits)
    reason_bits & _NATIVE_TRANSFORM_NONFINITE_INPUT != 0 && return :nonfinite_input
    reason_bits & _NATIVE_TRANSFORM_NONFINITE_OUTPUT != 0 && return :nonfinite_output
    reason_bits & _NATIVE_TRANSFORM_NONFINITE_LOGJAC != 0 && return :nonfinite_logabsjac
    return :outside_support
end

@inline function _checked_native_transform_result(result)
    value, logabsjac, reason = result
    iszero(reason) || _throw_invalid_transform(_native_transform_failure_reason(reason))
    return value, logabsjac
end

@inline function _softplus(z::T) where {T<:_TransformFloat}
    return max(z, zero(T)) + log1p(exp(-abs(z)))
end

@inline function _logsigmoid(z::T) where {T<:_TransformFloat}
    return -_softplus(-z)
end

@inline function _logistic(z::T) where {T<:_TransformFloat}
    if z >= zero(T)
        exponent = exp(-z)
        return inv(one(T) + exponent)
    else
        exponent = exp(z)
        return exponent / (one(T) + exponent)
    end
end

@inline function _log_positive_difference(upper::T, lower::T) where {T<:_TransformFloat}
    if lower < zero(T) && upper > zero(T)
        lower_magnitude = -lower
        upper_magnitude = upper
        larger = max(lower_magnitude, upper_magnitude)
        smaller = min(lower_magnitude, upper_magnitude)
        return log(larger) + log1p(smaller / larger)
    end
    return log(upper - lower)
end

@inline function _bounded_interval_value(lower::T, upper::T, probability::T) where {T<:_TransformFloat}
    return (one(T) - probability) * lower + probability * upper
end

@inline _transform_with_logjac(transform::_NativeScalarTransform, z::_TransformFloat) =
    _checked_native_transform_result(_native_transform_with_logjac(transform, z))

@inline _inverse_with_logjac(transform::_NativeScalarTransform, x::_TransformFloat) =
    _checked_native_transform_result(_native_inverse_with_logjac(transform, x))

@inline function _simplex_embedding_constants(::Type{T}, dimension::Int) where {T}
    inverse_root_dimension = inv(sqrt(T(dimension)))
    shared_coefficient = (one(T) + inverse_root_dimension) / T(dimension - 1)
    return inverse_root_dimension, shared_coefficient
end

@inline function _compensated_add(total::T, correction::T, value::T) where {T}
    corrected_value = value - correction
    updated_total = total + corrected_value
    updated_correction = (updated_total - total) - corrected_value
    return updated_total, updated_correction
end

@inline function _native_simplex_inverse!(coordinate, offset, transform, x)
    T = eltype(x)
    dimension = transform.dimension
    weight_sum = zero(T)
    weight_sum_correction = zero(T)
    log_weight_sum = zero(T)
    log_weight_sum_correction = zero(T)
    last_log_weight = zero(T)
    for index in 1:dimension
        weight = @inbounds x[index]
        isfinite(weight) || return zero(T), _NATIVE_TRANSFORM_NONFINITE_INPUT
        weight > zero(T) || return zero(T), _NATIVE_TRANSFORM_OUTSIDE_SUPPORT
        weight_sum, weight_sum_correction =
            _compensated_add(weight_sum, weight_sum_correction, weight)
        log_weight = log(weight)
        log_weight_sum, log_weight_sum_correction =
            _compensated_add(log_weight_sum, log_weight_sum_correction, log_weight)
        if index < dimension
            @inbounds coordinate[offset + index - 1] = log_weight
        else
            last_log_weight = log_weight
        end
    end
    isfinite(weight_sum) || return zero(T), _NATIVE_TRANSFORM_NONFINITE_INPUT
    abs(weight_sum - one(T)) <= _simplex_sum_tolerance(T) ||
        return zero(T), _NATIVE_TRANSFORM_OUTSIDE_SUPPORT

    mean_log_weight = log_weight_sum / T(dimension)
    inverse_root_dimension, _ = _simplex_embedding_constants(T, dimension)
    transpose_coefficient = inverse_root_dimension / (one(T) - inverse_root_dimension)
    last_centered_log_weight = last_log_weight - mean_log_weight
    for index in 1:(dimension - 1)
        @inbounds coordinate[offset + index - 1] =
            coordinate[offset + index - 1] - mean_log_weight +
            transpose_coefficient * last_centered_log_weight
    end

    logabsjac = T(0.5) * log(T(dimension)) + log_weight_sum
    isfinite(logabsjac) || return logabsjac, _NATIVE_TRANSFORM_NONFINITE_LOGJAC
    return logabsjac, UInt16(0)
end

@inline function _simplex_sum_tolerance(::Type{T}) where {T}
    # Compensated accumulation plus rounded stored weights stays within a few ulps.
    return T(8) * eps(T)
end

function _transform_with_logjac(
    transform::SimplexTransform,
    z::AbstractVector{T},
) where {T<:_TransformFloat}
    dimension = transform.dimension
    length(z) == dimension - 1 || throw(
        DimensionMismatch(
            "SimplexTransform($dimension) requires $(dimension - 1) unconstrained coordinates",
        ),
    )

    x = similar(z, dimension)
    coordinate_sum = zero(T)
    coordinate_sum_correction = zero(T)
    for index in eachindex(z)
        coordinate = z[index]
        isfinite(coordinate) || _throw_invalid_transform(:nonfinite_input)
        coordinate_sum, coordinate_sum_correction =
            _compensated_add(coordinate_sum, coordinate_sum_correction, coordinate)
    end
    isfinite(coordinate_sum) || _throw_invalid_transform(:nonfinite_output)

    inverse_root_dimension, shared_coefficient =
        _simplex_embedding_constants(T, dimension)
    maximum_logit = inverse_root_dimension * coordinate_sum
    x[dimension] = maximum_logit
    for index in eachindex(z)
        logit = z[index] - shared_coefficient * coordinate_sum
        isfinite(logit) || _throw_invalid_transform(:nonfinite_output)
        x[index] = logit
        maximum_logit = max(maximum_logit, logit)
    end

    exponential_sum = zero(T)
    exponential_sum_correction = zero(T)
    for index in eachindex(x)
        weight = exp(x[index] - maximum_logit)
        x[index] = weight
        exponential_sum, exponential_sum_correction =
            _compensated_add(exponential_sum, exponential_sum_correction, weight)
    end
    isfinite(exponential_sum) || _throw_invalid_transform(:nonfinite_output)

    logabsjac = T(0.5) * log(T(dimension))
    logabsjac_correction = zero(T)
    for index in eachindex(x)
        weight = x[index] / exponential_sum
        weight > zero(T) || _throw_invalid_transform(:outside_support)
        x[index] = weight
        logabsjac, logabsjac_correction =
            _compensated_add(logabsjac, logabsjac_correction, log(weight))
    end
    isfinite(logabsjac) || _throw_invalid_transform(:nonfinite_logabsjac)
    return x, logabsjac
end

function _inverse_with_logjac(
    transform::SimplexTransform,
    x::AbstractVector{T},
) where {T<:_TransformFloat}
    dimension = transform.dimension
    length(x) == dimension || throw(
        DimensionMismatch("SimplexTransform($dimension) requires $dimension simplex weights"),
    )

    z = similar(x, dimension - 1)
    logabsjac, reason = _native_simplex_inverse!(z, firstindex(z), transform, x)
    iszero(reason) || _throw_invalid_transform(_native_transform_failure_reason(reason))
    return z, logabsjac
end

function _transform_with_logjac(
    ::IdentityTransform,
    z::AbstractVector{T},
) where {T<:_TransformFloat}
    for coordinate in z
        isfinite(coordinate) || _throw_invalid_transform(:nonfinite_input)
    end
    return z, zero(T)
end

_inverse_with_logjac(transform::IdentityTransform, x::AbstractVector{<:_TransformFloat}) =
    _transform_with_logjac(transform, x)
