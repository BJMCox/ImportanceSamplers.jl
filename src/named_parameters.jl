struct _NamedPreparedTarget{T,L}
    target::T
    layout::L
end

struct _BoundNamedTarget{T,L} <: _BoundTarget
    target::T
    layout::L
end

Adapt.@adapt_structure _NamedPreparedTarget
Adapt.@adapt_structure _BoundNamedTarget

struct _ScalarCoordinateView{T} <: AbstractVector{T}
    value::T
end

Base.IndexStyle(::Type{<:_ScalarCoordinateView}) = IndexLinear()
Base.size(::_ScalarCoordinateView) = (1,)

@inline function Base.getindex(values::_ScalarCoordinateView, index::Int)
    @boundscheck checkbounds(values, index)
    return values.value
end

struct _SimplexCoordinateView{T,C} <: AbstractVector{T}
    coordinate::C
    first::Int
    dimension::Int
    coordinate_sum::T
    maximum_logit::T
    exponential_sum::T
    inverse_root_dimension::T
    shared_coefficient::T
end

Base.IndexStyle(::Type{<:_SimplexCoordinateView}) = IndexLinear()
Base.size(values::_SimplexCoordinateView) = (values.dimension,)

@inline function Base.getindex(values::_SimplexCoordinateView, index::Int)
    @boundscheck checkbounds(values, index)
    logit = if index == values.dimension
        values.inverse_root_dimension * values.coordinate_sum
    else
        values.coordinate[values.first + index - 1] -
        values.shared_coefficient * values.coordinate_sum
    end
    return exp(logit - values.maximum_logit) / values.exponential_sum
end

@inline _selected_coordinate_view(coordinate::AbstractVector, selector::UnitRange{Int}) =
    @view coordinate[selector]
@inline _selected_coordinate_view(coordinate::T, selector::UnitRange{Int}) where
    {T<:_TransformFloat} = _ScalarCoordinateView(coordinate)

@inline function _map_flat_block(coordinate, block::_LocatedTransform{Int})
    value = coordinate[block.location]
    return _native_transform_with_logjac(block.transform, value)
end

@inline function _map_flat_block(
    coordinate,
    block::_LocatedTransform{<:UnitRange{Int},IdentityTransform},
)
    values = _selected_coordinate_view(coordinate, block.location)
    reason = UInt16(0)
    for value in values
        reason |= isfinite(value) ? UInt16(0) : _NATIVE_TRANSFORM_NONFINITE_INPUT
    end
    return values, zero(eltype(coordinate)), reason
end

@inline function _map_flat_block(
    coordinate,
    block::_LocatedTransform{<:UnitRange{Int},SimplexTransform},
)
    transform = block.transform
    selector = block.location
    T = eltype(coordinate)
    coordinate_sum = zero(T)
    coordinate_sum_correction = zero(T)
    reason = UInt16(0)
    for index in selector
        value = coordinate[index]
        reason |= isfinite(value) ? UInt16(0) : _NATIVE_TRANSFORM_NONFINITE_INPUT
        coordinate_sum, coordinate_sum_correction =
            _compensated_add(coordinate_sum, coordinate_sum_correction, value)
    end
    isfinite(coordinate_sum) || (reason |= _NATIVE_TRANSFORM_NONFINITE_OUTPUT)

    inverse_root_dimension, shared_coefficient =
        _simplex_embedding_constants(T, transform.dimension)
    maximum_logit = inverse_root_dimension * coordinate_sum
    for index in selector
        logit = coordinate[index] -
                shared_coefficient * coordinate_sum
        reason |= isfinite(logit) ? UInt16(0) : _NATIVE_TRANSFORM_NONFINITE_OUTPUT
        maximum_logit = max(maximum_logit, logit)
    end

    exponential_sum = zero(T)
    exponential_sum_correction = zero(T)
    for output_index in 1:transform.dimension
        logit = output_index == transform.dimension ?
                inverse_root_dimension * coordinate_sum :
                coordinate[first(selector) + output_index - 1] -
                shared_coefficient * coordinate_sum
        weight = exp(logit - maximum_logit)
        exponential_sum, exponential_sum_correction = _compensated_add(
            exponential_sum,
            exponential_sum_correction,
            weight,
        )
    end
    isfinite(exponential_sum) || (reason |= _NATIVE_TRANSFORM_NONFINITE_OUTPUT)

    values = _SimplexCoordinateView{T,typeof(coordinate)}(
        coordinate,
        first(selector),
        transform.dimension,
        coordinate_sum,
        maximum_logit,
        exponential_sum,
        inverse_root_dimension,
        shared_coefficient,
    )
    logabsjac = T(0.5) * log(T(transform.dimension))
    logabsjac_correction = zero(T)
    for index in 1:transform.dimension
        weight = values[index]
        reason |= weight > zero(T) ? UInt16(0) : _NATIVE_TRANSFORM_OUTSIDE_SUPPORT
        logabsjac, logabsjac_correction =
            _compensated_add(logabsjac, logabsjac_correction, log(weight))
    end
    isfinite(logabsjac) || (reason |= _NATIVE_TRANSFORM_NONFINITE_LOGJAC)
    return values, logabsjac, reason
end

@inline _map_flat_blocks(coordinate, ::Tuple{}, block_index) =
    ((), zero(eltype(coordinate)), UInt16(0), 0)

@inline function _map_flat_blocks(coordinate, blocks::Tuple, block_index)
    logical, logabsjac, reason = _map_flat_block(coordinate, first(blocks))
    tail, tail_logabsjac, tail_reason, tail_block =
        _map_flat_blocks(coordinate, Base.tail(blocks), block_index + 1)
    failed_block = !iszero(reason) ? block_index : tail_block
    combined_reason = !iszero(reason) ? reason : tail_reason
    return (
        (logical, tail...),
        logabsjac + tail_logabsjac,
        combined_reason,
        failed_block,
    )
end

@inline function _coordinate_to_logical(
    layout::_FlatTransformLayout{B},
    coordinate,
) where {Names,B<:NamedTuple{Names}}
    logical, logabsjac, reason, block =
        _map_flat_blocks(coordinate, values(layout.blocks), 1)
    return NamedTuple{Names}(logical), logabsjac, reason, block
end

@inline function _logical_binding_block(coordinate, block::_LocatedTransform{Int})
    sample = zero(eltype(coordinate))
    _validate_known_transform_input(block.transform, sample)
    return sample
end

@inline function _logical_binding_block(
    coordinate,
    block::_LocatedTransform{<:UnitRange{Int},IdentityTransform},
)
    return _selected_coordinate_view(coordinate, block.location)
end


@inline function _logical_binding_block(
    coordinate,
    block::_LocatedTransform{<:UnitRange{Int},SimplexTransform},
)
    T = eltype(coordinate)
    dimension = block.transform.dimension
    inverse_root_dimension, shared_coefficient =
        _simplex_embedding_constants(T, dimension)
    return _SimplexCoordinateView{T,typeof(coordinate)}(
        coordinate,
        first(block.location),
        dimension,
        zero(T),
        zero(T),
        T(dimension),
        inverse_root_dimension,
        shared_coefficient,
    )
end

function _logical_binding_sample(layout::_FlatTransformLayout, coordinate)
    return map(block -> _logical_binding_block(coordinate, block), layout.blocks)
end

@inline _throw_named_transform_failure(layout::_FlatTransformLayout, reason, block) =
    _throw_named_transform_failure(values(layout.blocks), reason, block)

@inline function _throw_named_transform_failure(blocks::Tuple, reason, block)
    block == 1 && throw(InvalidTransformError(
        _native_transform_failure_reason(reason), first(blocks).location))
    return _throw_named_transform_failure(Base.tail(blocks), reason, block - 1)
end

_throw_named_transform_failure(::Tuple{}, reason, block) =
    throw(InvalidTransformError(_native_transform_failure_reason(reason)))

function _named_target_layout(proposal, specification::NamedTuple)
    return _prepare_flat_transform_layout(proposal, specification)
end

function _named_target_layout(proposal, layout::_FlatTransformLayout)
    dimension = _proposal_dimension(proposal)
    dimension == layout.dimension || throw(
        DimensionMismatch(
            "named target transform requires $(layout.dimension) sampling coordinates",
        ),
    )
    return layout
end

function _named_target_layout(proposal, transform)
    throw(ArgumentError("transform must be nothing or a named flat selector layout"))
end

function _prepare_named_target(target, proposal, specification)
    layout = _named_target_layout(proposal, specification)
    target isa _BoundLogDensityProblemsTarget && throw(
        ArgumentError(
            "a bare LogDensityProblems target cannot be combined with a named transform layout; wrap a callable in LogTarget",
        ),
    )
    return _NamedPreparedTarget(target, layout)
end

function _bind_resolved_target(target::_NamedPreparedTarget, coordinate)
    logical = _logical_binding_sample(target.layout, coordinate)
    bound = _bind_resolved_target(target.target, logical)
    return _BoundNamedTarget(bound, target.layout)
end

@inline function (target::_BoundNamedTarget)(coordinate)
    logical, logabsjac, reason, block =
        _coordinate_to_logical(target.layout, coordinate)
    iszero(reason) ||
        _throw_named_transform_failure(target.layout, reason, block)
    value = target.target(logical)
    _validate_target_logdensity(value)
    return value + logabsjac
end

function _transfer_prepared_target(device, target::_NamedPreparedTarget)
    return _NamedPreparedTarget(
        _transfer_prepared_target(device, target.target),
        _copy_to_device(device, target.layout),
    )
end

_target_has_opaque_host_closure(target::_NamedPreparedTarget) =
    _target_has_opaque_host_closure(target.target)
_target_has_opaque_host_closure(target::_NamedPreparedTarget, device) =
    _target_has_opaque_host_closure(target.target, device)
_target_transfer_rewrites_opaque_closure(target::_NamedPreparedTarget) =
    _target_transfer_rewrites_opaque_closure(target.target)

_prepared_target_layout(target) = nothing
_prepared_target_layout(target::_NamedPreparedTarget) = target.layout

@inline function _native_device_target_result(target::_BoundNamedTarget, coordinate, ::Type{L}) where {L}
    logical, logabsjac, transform_reason, _ =
        _coordinate_to_logical(target.layout, coordinate)
    iszero(transform_reason) ||
        return zero(L), _NATIVE_TARGET_NAN, true
    value = convert(L, target.target(logical) + logabsjac)
    reason = _native_target_reason(value)
    return value, reason, !iszero(reason)
end

@inline _store_named_result_value!(output::AbstractVector, value, sample_index) =
    (output[sample_index] = value)

@inline function _store_named_result_value!(
    output::AbstractMatrix,
    value,
    sample_index,
)
    for index in eachindex(value)
        output[index, sample_index] = value[index]
    end
    return nothing
end

@inline _store_named_result_values!(::Tuple{}, ::Tuple{}, sample_index) = nothing

@inline function _store_named_result_values!(outputs, values, sample_index)
    _store_named_result_value!(first(outputs), first(values), sample_index)
    return _store_named_result_values!(
        Base.tail(outputs),
        Base.tail(values),
        sample_index,
    )
end

@kernel function _map_named_result_kernel!(
    outputs,
    samples,
    layout,
    failure_storage,
)
    sample_index = @index(Global, Linear)
    coordinate = _native_sample_at(samples, sample_index)
    logical, _, reason, block = _coordinate_to_logical(layout, coordinate)
    if iszero(reason)
        _store_named_result_values!(outputs, values(logical), sample_index)
    else
        _record_native_failure!(
            failure_storage,
            sample_index,
            block,
            reason,
        )
    end
end

_map_result_samples(target, samples, failure_scratch, transfers, threaded) = samples
_map_owned_result_samples(target, samples, failure_scratch, transfers, threaded) = copy(samples)

function _map_result_samples(target::_NamedPreparedTarget, samples, failure_scratch, transfers, threaded)
    layout = target.layout
    count = _sample_count(samples)
    mapped = _allocate_flat_samples(samples, eltype(samples), layout, count)
    backend = KernelAbstractions.get_backend(samples)
    if backend isa KernelAbstractions.CPU
        iterate = threaded ? _threaded_foreach : foreach
        iterate(1:count) do sample_index
            logical, _, reason, block = _coordinate_to_logical(layout, _sample_at(samples, sample_index))
            iszero(reason) || _throw_named_result_failure(layout, reason, sample_index, block)
            _store_named_result_values!(values(mapped), values(logical), sample_index)
        end
    else
        record = failure_scratch.record
        fill!(record.storage, zero(UInt64))
        kernel = _map_named_result_kernel!(backend)
        kernel(values(mapped), samples, layout, record.storage; ndrange=count)
        KernelAbstractions.synchronize(backend)
        snapshot = _device_failure_snapshot(record)
        failure = snapshot.failure
        iszero(failure.count) || _throw_named_result_failure(
            layout, failure.reason_bits, failure.first_logical_index, failure.first_block)
        _record_reported_transfer!(transfers, snapshot.transfers.count,
            snapshot.transfers.bytes, Val(:failure_snapshot))
    end
    return mapped
end

_map_owned_result_samples(target::_NamedPreparedTarget, samples, failure_scratch, transfers, threaded) =
    _map_result_samples(target, samples, failure_scratch, transfers, threaded)

function _throw_named_result_failure(layout, reason, sample_index, block)
    cause = InvalidTransformError(
        _native_transform_failure_reason(reason), _native_failure_location(layout, block))
    throw(SamplerExecutionError(
        :result_construction, sample_index, CapturedException(cause, backtrace())))
end
