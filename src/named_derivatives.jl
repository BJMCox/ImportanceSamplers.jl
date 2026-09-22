struct _BoundNamedGradient{T,S} <: _BoundGradient
    target::T
    scratch::S
end

Adapt.@adapt_structure _BoundNamedGradient

_logical_gradient_view(scratch, block::_LocatedTransform{Int}, offset) = view(scratch, offset)
_logical_gradient_view(scratch, block::_LocatedTransform{<:UnitRange}, offset) =
    view(scratch, offset:(offset + _logical_block_length(block) - 1))

_logical_gradient_views(scratch, ::Tuple{}, offset) = ()
function _logical_gradient_views(scratch, blocks::Tuple, offset)
    block = first(blocks)
    return (_logical_gradient_view(scratch, block, offset),
        _logical_gradient_views(scratch, Base.tail(blocks), offset + _logical_block_length(block))...)
end

function _logical_gradient(layout, scratch)
    return NamedTuple{keys(layout.blocks)}(
        _logical_gradient_views(scratch, values(layout.blocks), 1))
end

function _prepare_bound_gradient(target::_NamedPreparedTarget, sample, worker_count)
    target = _scalar_target(target)
    _validate_gradient_worker_count(worker_count)
    inner = target.target
    inner isa _PreparedLogTarget || return _prepare_bound_gradient(inner, sample, worker_count)
    if isnothing(inner.gradient)
        return _prepare_bound_gradient(_named_ad_target(target), sample, worker_count)
    end
    dimension = sum(_logical_block_length, values(target.layout.blocks))
    scratch = _prepare_gradient_pool(worker_count) do
        similar(sample, dimension)
    end
    _validate_named_gradient(target, sample, first(scratch.preparations))
    return _BoundNamedGradient(target, scratch)
end

function _validate_named_gradient(target, sample, scratch)
    logical = _logical_binding_sample(target.layout, sample)
    gradient = _logical_gradient(target.layout, scratch)
    inner = target.target
    args = inner.context isa _NoTargetContext ? (gradient, logical) :
        (gradient, logical, inner.context)
    applicable(inner.gradient, args...) || throw(ArgumentError(
        "named explicit gradients require grad!(g, theta) or grad!(g, theta, p)"))
    return nothing
end

function _prepare_accelerator_gradient(target::_NamedPreparedTarget, locations, values)
    target = _scalar_target(target)
    inner = target.target
    inner isa _PreparedLogTarget || return _prepare_accelerator_gradient(inner, locations, values)
    isnothing(inner.gradient) && return _prepare_accelerator_gradient(
        _named_ad_target(target; materialized=false), locations, values)
    dimension = sum(_logical_block_length, Base.values(target.layout.blocks))
    scratch = similar(locations, dimension, size(locations, 2))
    _validate_named_gradient(target, view(locations, :, 1), view(scratch, :, 1))
    return _BoundNamedGradient(target, scratch)
end

_named_gradient_scratch(pool::_GradientWorkerPool, slot) = _gradient_workspace(pool)
_named_gradient_scratch(scratch::AbstractMatrix, slot) = view(scratch, :, slot)

# The explicit slot gives each GPU proposal its own logical gradient storage.
_gradient!(destination, bound, sample, slot) = _gradient!(destination, bound, sample)
function _gradient!(destination, bound::_BoundNamedGradient, sample, slot=1)
    target = bound.target
    logical, _, reason, block = _coordinate_to_logical(target.layout, sample)
    iszero(reason) || _throw_named_transform_failure(target.layout, reason, block)
    scratch = _named_gradient_scratch(bound.scratch, slot)
    gradient = _logical_gradient(target.layout, scratch)
    _explicit_inplace_gradient!(gradient, target.target, target.target.context, logical)
    _pullback_named_blocks!(destination, sample, values(target.layout.blocks),
        values(logical), values(gradient))
    return destination
end

_pullback_named_blocks!(destination, sample, ::Tuple{}, ::Tuple{}, ::Tuple{}) = nothing
function _pullback_named_blocks!(destination, sample, blocks::Tuple, logical::Tuple, gradients::Tuple)
    _pullback_named_block!(destination, sample, first(blocks), first(logical), first(gradients))
    return _pullback_named_blocks!(destination, sample, Base.tail(blocks), Base.tail(logical), Base.tail(gradients))
end

function _pullback_named_block!(destination, sample, block::_LocatedTransform{Int}, value, gradient)
    index = block.location
    slope, jacobian_gradient = _scalar_transform_derivative(block.transform, sample[index], value)
    destination[index] = slope * gradient[] + jacobian_gradient
    return nothing
end

function _pullback_named_block!(destination, sample,
    block::_LocatedTransform{<:UnitRange,IdentityTransform}, value, gradient)
    for (index, coordinate) in enumerate(block.location)
        destination[coordinate] = gradient[index]
    end
    return nothing
end

function _pullback_named_block!(destination, sample,
    block::_LocatedTransform{<:UnitRange,SimplexTransform}, weights, gradient)
    T = eltype(sample)
    dimension = length(weights)
    weighted_gradient = zero(T)
    for i in eachindex(weights)
        weighted_gradient += weights[i] * gradient[i]
    end
    head_sum = zero(T)
    for i in 1:(dimension - 1)
        term = weights[i] * (gradient[i] - weighted_gradient) + one(T) - dimension * weights[i]
        destination[block.location[i]] = term
        head_sum += term
    end
    tail = weights[dimension] * (gradient[dimension] - weighted_gradient) +
        one(T) - dimension * weights[dimension]
    a, c = _simplex_embedding_constants(T, dimension)
    for index in block.location
        destination[index] += a * tail - c * head_sum
    end
    return nothing
end

_scalar_transform_derivative(::IdentityTransform, z, value) = (one(z), zero(z))
_scalar_transform_derivative(::PositiveTransform, z, value) = (value, one(z))
function _scalar_transform_derivative(::SoftplusTransform, z, value)
    probability = _logistic(z)
    return probability, _logistic(-z)
end
_scalar_transform_derivative(::IntervalTransform{T,T,Nothing}, z, value) where {T<:_TransformFloat} = (exp(z), one(z))
_scalar_transform_derivative(::IntervalTransform{T,Nothing,T}, z, value) where {T<:_TransformFloat} = (-exp(z), one(z))
function _scalar_transform_derivative(transform::IntervalTransform{T,T,T}, z, value) where {T<:_TransformFloat}
    _, logjac, _ = _native_transform_with_logjac(transform, z)
    return exp(logjac), one(z) - 2 * _logistic(z)
end

struct _NamedADLogDensity{F,L,M}
    logdensity::F
    layout::L
    materialized::M
end

Adapt.@adapt_structure _NamedADLogDensity

function _named_ad_target(target; materialized=true)
    inner = target.target
    density = _NamedADLogDensity(inner.logdensity, target.layout, Val(materialized))
    return _PreparedLogTarget(density, inner.context, inner.adtype, nothing)
end

function (density::_NamedADLogDensity)(coordinate, context...)
    logical, logjac = _ad_coordinate_to_logical(density.layout, coordinate, density.materialized)
    return density.logdensity(logical, context...) + logjac
end

# Keep borrowed views in the differentiated kernel frame. An outlined aggregate
# return causes invalid device accesses with Enzyme's GPU reverse pass.
@inline function _ad_coordinate_to_logical(layout, coordinate, ::Val{false})
    logical, logjac, reason, block = _coordinate_to_logical(layout, coordinate)
    iszero(reason) || _throw_named_transform_failure(layout, reason, block)
    return logical, logjac
end

function _ad_coordinate_to_logical(layout, coordinate, ::Val{true})
    blocks = map(values(layout.blocks)) do block
        _materialized_transform_block(coordinate, block)
    end
    return NamedTuple{keys(layout.blocks)}(map(first, blocks)), sum(last, blocks)
end

function _materialized_transform_block(coordinate, block::_LocatedTransform{Int})
    value, logjac, reason = _native_transform_with_logjac(block.transform, coordinate[block.location])
    iszero(reason) || _throw_invalid_transform(_native_transform_failure_reason(reason))
    return value, logjac
end

function _materialized_transform_block(coordinate,
    block::_LocatedTransform{<:UnitRange,IdentityTransform})
    values = coordinate[block.location]
    all(isfinite, values) || _throw_invalid_transform(:nonfinite_input)
    return values, zero(eltype(coordinate))
end

function _materialized_transform_block(coordinate,
    block::_LocatedTransform{<:UnitRange,SimplexTransform})
    coordinates = coordinate[block.location]
    all(isfinite, coordinates) || _throw_invalid_transform(:nonfinite_input)
    total = sum(coordinates)
    dimension = block.transform.dimension
    a, c = _simplex_embedding_constants(typeof(total), dimension)
    logits = vcat(coordinates .- c * total, a * total)
    all(isfinite, logits) || _throw_invalid_transform(:nonfinite_output)
    exponentials = exp.(logits .- maximum(logits))
    weights = exponentials ./ sum(exponentials)
    all(>(zero(eltype(weights))), weights) || _throw_invalid_transform(:outside_support)
    logjac = oftype(total, 0.5) * log(oftype(total, dimension)) + sum(log, weights)
    isfinite(logjac) || _throw_invalid_transform(:nonfinite_logabsjac)
    return weights, logjac
end
