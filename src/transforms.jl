"""
    AbstractSampleTransform

Abstract supertype for transforms from an unconstrained scalar to a logical
sample value.
"""
abstract type AbstractSampleTransform end

struct IdentityTransform <: AbstractSampleTransform end

struct PositiveTransform <: AbstractSampleTransform end

struct SoftplusTransform <: AbstractSampleTransform end

struct IntervalTransform{T,L,U} <: AbstractSampleTransform
    lower::L
    upper::U
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
at least one endpoint is required.
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

@inline function _checked_transform_input(z::T) where {T<:_TransformFloat}
    isfinite(z) || _throw_invalid_transform(:nonfinite_input)
    return z
end

@inline function _checked_forward_value(x::T, logabsjac::T) where {T<:_TransformFloat}
    isfinite(x) || _throw_invalid_transform(:nonfinite_output)
    isfinite(logabsjac) || _throw_invalid_transform(:nonfinite_logabsjac)
    return x, logabsjac
end

@inline function _checked_positive_value(x::T, logabsjac::T) where {T<:_TransformFloat}
    _checked_forward_value(x, logabsjac)
    x > zero(T) || _throw_invalid_transform(:outside_support)
    return x, logabsjac
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

@inline function _transform_with_logjac(::IdentityTransform, z::T) where {T<:_TransformFloat}
    _checked_transform_input(z)
    return z, zero(T)
end

@inline function _transform_with_logjac(::PositiveTransform, z::T) where {T<:_TransformFloat}
    _checked_transform_input(z)
    return _checked_positive_value(exp(z), z)
end

@inline function _transform_with_logjac(::SoftplusTransform, z::T) where {T<:_TransformFloat}
    _checked_transform_input(z)
    x = _softplus(z)
    return _checked_positive_value(x, z - x)
end

@inline function _transform_with_logjac(
    transform::IntervalTransform{T,T,Nothing},
    z::T,
) where {T<:_TransformFloat}
    _checked_transform_input(z)
    distance, logabsjac = _transform_with_logjac(PositiveTransform(), z)
    x = transform.lower + distance
    _checked_forward_value(x, logabsjac)
    x > transform.lower || _throw_invalid_transform(:outside_support)
    return x, logabsjac
end

@inline function _transform_with_logjac(
    transform::IntervalTransform{T,Nothing,T},
    z::T,
) where {T<:_TransformFloat}
    _checked_transform_input(z)
    distance, logabsjac = _transform_with_logjac(PositiveTransform(), z)
    x = transform.upper - distance
    _checked_forward_value(x, logabsjac)
    x < transform.upper || _throw_invalid_transform(:outside_support)
    return x, logabsjac
end

@inline function _transform_with_logjac(
    transform::IntervalTransform{T,T,T},
    z::T,
) where {T<:_TransformFloat}
    _checked_transform_input(z)
    probability = _logistic(z)
    x = _bounded_interval_value(transform.lower, transform.upper, probability)
    logabsjac = _log_positive_difference(transform.upper, transform.lower) +
                 _logsigmoid(z) + _logsigmoid(-z)
    _checked_forward_value(x, logabsjac)
    transform.lower < x < transform.upper || _throw_invalid_transform(:outside_support)
    return x, logabsjac
end

@inline function _inverse_with_logjac(::IdentityTransform, x::T) where {T<:_TransformFloat}
    _checked_transform_input(x)
    return x, zero(T)
end

@inline function _inverse_with_logjac(::PositiveTransform, x::T) where {T<:_TransformFloat}
    isfinite(x) || _throw_invalid_transform(:nonfinite_input)
    x > zero(T) || _throw_invalid_transform(:outside_support)
    z = log(x)
    return _checked_forward_value(z, z)
end

@inline function _inverse_with_logjac(::SoftplusTransform, x::T) where {T<:_TransformFloat}
    isfinite(x) || _throw_invalid_transform(:nonfinite_input)
    x > zero(T) || _throw_invalid_transform(:outside_support)
    z = LogExpFunctions.logexpm1(x)
    return _checked_forward_value(z, z - x)
end

@inline function _inverse_with_logjac(
    transform::IntervalTransform{T,T,Nothing},
    x::T,
) where {T<:_TransformFloat}
    isfinite(x) || _throw_invalid_transform(:nonfinite_input)
    x > transform.lower || _throw_invalid_transform(:outside_support)
    z = log(x - transform.lower)
    return _checked_forward_value(z, z)
end

@inline function _inverse_with_logjac(
    transform::IntervalTransform{T,Nothing,T},
    x::T,
) where {T<:_TransformFloat}
    isfinite(x) || _throw_invalid_transform(:nonfinite_input)
    x < transform.upper || _throw_invalid_transform(:outside_support)
    z = log(transform.upper - x)
    return _checked_forward_value(z, z)
end

@inline function _inverse_with_logjac(
    transform::IntervalTransform{T,T,T},
    x::T,
) where {T<:_TransformFloat}
    isfinite(x) || _throw_invalid_transform(:nonfinite_input)
    transform.lower < x < transform.upper || _throw_invalid_transform(:outside_support)
    lower_logdistance = _log_positive_difference(x, transform.lower)
    upper_logdistance = _log_positive_difference(transform.upper, x)
    span_logdistance = _log_positive_difference(transform.upper, transform.lower)
    z = lower_logdistance - upper_logdistance
    logabsjac = lower_logdistance + upper_logdistance - span_logdistance
    return _checked_forward_value(z, logabsjac)
end
