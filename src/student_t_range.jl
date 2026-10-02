# Exceptional Student-t solves extend exponent range, not float precision.
struct _RangeSolveScratch{V,E}
    values::V
    exponents::E
end
Adapt.@adapt_structure _RangeSolveScratch

_solve_values(scratch) = scratch
_solve_values(scratch::_RangeSolveScratch) = scratch.values
_solve_exponents(scratch) = nothing
_solve_exponents(scratch::_RangeSolveScratch) = scratch.exponents
_range_scratch(values, ::Nothing) = values
_range_scratch(values, exponents) = _RangeSolveScratch(values, exponents)
_solve_view(scratch, indices...) = view(scratch, indices...)
_solve_view(scratch::_RangeSolveScratch, indices...) =
    _RangeSolveScratch(view(scratch.values, indices...), view(scratch.exponents, indices...))

_range_exponents(prototype, ::GaussianFamily, dims...) = nothing

# Binade endpoints bound factor magnitudes across Float32/Float64 transfers.
function _factor_exponent_bound(factor)
    bound = 0
    for column in axes(factor, 2), row in column:size(factor, 1)
        value = factor[row, column]
        iszero(value) && continue
        isfinite(value) || return typemax(Int)
        _, exponent = frexp(value)
        bound = max(bound, abs(exponent), abs(exponent - 1))
    end
    return bound
end

_factor_range_risk(bound, ::Type{T}) where {T} =
    bound >= (-exponent(floatmin(T))) ÷ 2
_exponent_bound_at(::Nothing, slot) = 0
_exponent_bound_at(bounds, slot) = bounds[slot]
_exponent_bounds_view(::Nothing, indices) = nothing
_exponent_bounds_view(bounds, indices) = view(bounds, indices)
_allocate_exponent_bounds(prototype, ::GaussianFamily, count) = nothing
_allocate_exponent_bounds(prototype, family, count) = similar(prototype, Int, count)
_set_factor_exponent_bound!(::Nothing, slot, factor) = nothing
_set_factor_exponent_bound!(bounds, slot, factor) =
    (bounds[slot] = _factor_exponent_bound(factor); nothing)
_copy_exponent_bounds!(::Nothing, ::Nothing) = nothing
_copy_exponent_bounds!(destination, source) = copyto!(destination, source)

function _factor_exponent_bounds(factors, family)
    bounds = _allocate_exponent_bounds(factors, family, size(factors, 3))
    for slot in axes(factors, 3)
        _set_factor_exponent_bound!(bounds, slot, view(factors, :, :, slot))
    end
    return bounds
end

@inline function _range_subtract(left, right)
    iszero(left[1]) && return (-right[1], right[2])
    iszero(right[1]) && return left
    exponent = max(left[2], right[2])
    mantissa, shift = frexp(ldexp(left[1], left[2] - exponent) -
                            ldexp(right[1], right[2] - exponent))
    return mantissa, iszero(mantissa) ? 0 : exponent + shift
end

@inline function _range_multiply(value, factor)
    mantissa, exponent = frexp(factor)
    product, shift = frexp(value[1] * mantissa)
    return product, iszero(product) ? 0 : value[2] + exponent + shift
end

@inline function _range_divide(value, divisor)
    mantissa, exponent = frexp(divisor)
    quotient, shift = frexp(value[1] / mantissa)
    return quotient, iszero(quotient) ? 0 : value[2] - exponent + shift
end

@inline function _logabsdiff(left, right)
    difference = left - right
    if isfinite(difference) || !isfinite(left) || !isfinite(right)
        return log(abs(difference))
    end
    largest = max(abs(left), abs(right))
    return log(largest) + log1p(min(abs(left), abs(right)) / largest)
end

function _diagonal_range_logradius(dimension, sample_at, location_at, scale_at)
    total = oftype(sample_at(1), -Inf)
    for coordinate in 1:dimension
        value = _logabsdiff(sample_at(coordinate), location_at(coordinate)) -
                log(scale_at(coordinate))
        total = LogExpFunctions.logaddexp(total, oftype(value, 2) * value)
    end
    return total
end

function _factor_range_logradius!(values, exponents, dimension, sample_at, location_at, factor_at)
    total = oftype(sample_at(1), -Inf)
    for row in 1:dimension
        value = _range_subtract(frexp(sample_at(row)), frexp(location_at(row)))
        for column in 1:(row - 1)
            previous = (values[column], exponents[column])
            value = _range_subtract(value, _range_multiply(previous, factor_at(row, column)))
        end
        mantissa, exponent = _range_divide(value, factor_at(row, row))
        values[row] = mantissa
        exponents[row] = exponent
        logabs = log(abs(mantissa)) + oftype(mantissa, exponent) * log(oftype(mantissa, 2))
        total = LogExpFunctions.logaddexp(total, oftype(logabs, 2) * logabs)
    end
    return total
end
