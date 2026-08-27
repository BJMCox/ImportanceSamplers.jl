@inline function _native_inverse_sample!(
    coordinates,
    offset,
    sample::T,
    ::_NoSampleTransform,
) where {T<:_NativeGaussianFloat}
    @inbounds coordinates[offset] = sample
    return zero(T), UInt16(0)
end

@inline function _native_copy_coordinates!(coordinates, offset, sample::AbstractVector{T}) where {T}
    for index in eachindex(sample)
        @inbounds coordinates[offset + index - 1] = sample[index]
    end
    return zero(T), UInt16(0)
end

@inline _native_inverse_sample!(
    coordinates,
    offset,
    sample::AbstractVector,
    ::Union{_NoSampleTransform,IdentityTransform},
) = _native_copy_coordinates!(coordinates, offset, sample)

@inline function _native_inverse_sample!(
    coordinates,
    offset,
    sample::T,
    transform::_NativeScalarTransform,
) where {T<:_NativeGaussianFloat}
    coordinate, logabsjac, reason = _native_inverse_with_logjac(transform, sample)
    @inbounds coordinates[offset] = coordinate
    return logabsjac, reason
end

@inline function _native_inverse_sample!(
    coordinates,
    offset,
    sample::AbstractVector,
    transform::SimplexTransform,
)
    return _native_simplex_inverse!(coordinates, offset, transform, sample)
end

@inline function _native_inverse_flat_block!(coordinates, offset, sample, block::_LocatedTransform)
    return _native_inverse_sample!(
        coordinates,
        offset + first(_selector_indices(block.location)) - 1,
        sample,
        block.transform,
    )
end

@inline _native_inverse_flat_blocks!(coordinates, offset, ::Tuple{}, ::Tuple{}) =
    (zero(eltype(coordinates)), UInt16(0))

@inline function _native_inverse_flat_blocks!(coordinates, offset, samples, blocks)
    logabsjac, reason = _native_inverse_flat_block!(
        coordinates,
        offset,
        first(samples),
        first(blocks),
    )
    iszero(reason) || return logabsjac, reason
    tail_logabsjac, tail_reason = _native_inverse_flat_blocks!(
        coordinates,
        offset,
        Base.tail(samples),
        Base.tail(blocks),
    )
    return logabsjac + tail_logabsjac, tail_reason
end

@inline function _native_inverse_sample!(
    coordinates,
    offset,
    sample::NamedTuple,
    layout::_FlatTransformLayout,
)
    return _native_inverse_flat_blocks!(
        coordinates,
        offset,
        values(sample),
        values(layout.blocks),
    )
end

@inline function _native_generated_logdensity(
    base,
    transform,
    sample,
    coordinates,
    offset,
)
    logabsjac, reason = _native_inverse_sample!(coordinates, offset, sample, transform)
    iszero(reason) || return base.lognormalizer, _NATIVE_PROPOSAL_INVALID
    return _native_gaussian_logdensity!(base, coordinates, offset) - logabsjac, UInt16(0)
end

@inline function _native_gaussian_coordinate(
    base::_GaussianProposal,
    normals,
    offset,
    coordinate,
    proposal_slot,
)
    return _gaussian_coordinate(
        base.location,
        base.scale,
        normals,
        offset,
        coordinate,
    )
end

@inline function _native_gaussian_coordinate(
    bank::_PackedDiagonalGaussianBank,
    normals,
    offset,
    coordinate,
    proposal_slot,
)
    return _gaussian_affine_coordinate(
        @inbounds(bank.locations[coordinate, proposal_slot]),
        @inbounds(bank.scales[coordinate, proposal_slot]),
        @inbounds(normals[offset + coordinate - 1]),
    )
end

@inline function _native_gaussian_coordinate(
    bank::_PackedFactorGaussianBank,
    normals,
    offset,
    coordinate,
    proposal_slot,
)
    value = @inbounds bank.locations[coordinate, proposal_slot]
    for column in 1:coordinate
        value += @inbounds(
            bank.factors[coordinate, column, proposal_slot] *
            normals[offset + column - 1]
        )
    end
    return value
end

@inline _packed_sample_coordinate(sample::Real, coordinate) = sample
@inline _packed_sample_coordinate(sample::AbstractVector, coordinate) =
    @inbounds sample[coordinate]

@inline _packed_gaussian_location(bank::_PackedFactorGaussianBank, row, slot) =
    @inbounds bank.locations[row, slot]
@inline _packed_gaussian_location(history::_AMISFactorHistory, row, slot) =
    @inbounds history.means[row, slot]
@inline _packed_gaussian_factor(bank::_PackedFactorGaussianBank, row, column, slot) =
    @inbounds bank.factors[row, column, slot]
@inline _packed_gaussian_factor(history::_AMISFactorHistory, row, column, slot) =
    @inbounds history.factors[row, column, slot]
@inline _packed_gaussian_lognormalizer(bank::_PackedFactorGaussianBank, slot) =
    @inbounds bank.lognormalizers[slot]
@inline _packed_gaussian_lognormalizer(history::_AMISFactorHistory, slot) =
    @inbounds history.lognormalizers[slot]

@inline function _packed_gaussian_logdensity!(
    bank::Union{_PackedFactorGaussianBank,_AMISFactorHistory},
    sample,
    proposal_slot,
    solve_scratch,
    sample_index,
)
    T = eltype(bank.lognormalizers)
    squared_radius = zero(T)
    for row in 1:_mis_dimension(bank)
        standardized = _packed_sample_coordinate(sample, row) -
                       _packed_gaussian_location(bank, row, proposal_slot)
        for column in 1:(row - 1)
            standardized -= @inbounds(
                _packed_gaussian_factor(bank, row, column, proposal_slot) *
                solve_scratch[column, sample_index]
            )
        end
        standardized /= _packed_gaussian_factor(
            bank,
            row,
            row,
            proposal_slot,
        )
        @inbounds solve_scratch[row, sample_index] = standardized
        squared_radius += abs2(standardized)
    end
    return _packed_gaussian_lognormalizer(bank, proposal_slot) -
           T(0.5) * squared_radius
end

@inline _packed_gaussian_location(bank::_PackedDiagonalGaussianBank, coordinate, slot) =
    @inbounds bank.locations[coordinate, slot]
@inline _packed_gaussian_location(history::_AMISScalarHistory, coordinate, slot) =
    @inbounds history.means[slot]
@inline _packed_gaussian_scale(bank::_PackedDiagonalGaussianBank, coordinate, slot) =
    @inbounds bank.scales[coordinate, slot]
@inline _packed_gaussian_scale(history::_AMISScalarHistory, coordinate, slot) =
    @inbounds history.scales[slot]
@inline _packed_gaussian_lognormalizer(bank::_PackedDiagonalGaussianBank, slot) =
    @inbounds bank.lognormalizers[slot]
@inline _packed_gaussian_lognormalizer(history::_AMISScalarHistory, slot) =
    @inbounds history.lognormalizers[slot]

@inline function _packed_gaussian_logdensity(
    bank::Union{_PackedDiagonalGaussianBank,_AMISScalarHistory},
    sample,
    proposal_slot,
)
    T = eltype(bank.lognormalizers)
    squared_radius = zero(T)
    for coordinate in 1:_mis_dimension(bank)
        standardized = (
            _packed_sample_coordinate(sample, coordinate) -
            _packed_gaussian_location(bank, coordinate, proposal_slot)
        ) / _packed_gaussian_scale(bank, coordinate, proposal_slot)
        squared_radius += abs2(standardized)
    end
    return _packed_gaussian_lognormalizer(bank, proposal_slot) -
           T(0.5) * squared_radius
end

@inline _mis_proposal_logdensity(
    ::Type{T},
    bank::_PackedDiagonalGaussianBank,
    sample,
    proposal_slot,
) where {T} = convert(T, _packed_gaussian_logdensity(bank, sample, proposal_slot))

@inline _native_sample_at(samples::AbstractVector, slot) = @inbounds samples[slot]
@inline _native_sample_at(samples::AbstractMatrix, slot) = @view samples[:, slot]

@generated function _native_sample_at(samples::NamedTuple{Names}, slot) where {Names}
    leaves = map(Names) do name
        :(_native_sample_at(getfield(samples, $(QuoteNode(name))), slot))
    end
    return :(NamedTuple{$Names}(($(leaves...),)))
end

@inline function _native_store_gaussian!(
    samples::AbstractVector,
    slot,
    gaussian,
    normals,
    offset,
    proposal_slot,
)
    value = _native_gaussian_coordinate(
        gaussian,
        normals,
        offset,
        1,
        proposal_slot,
    )
    @inbounds samples[slot] = value
    return isfinite(value)
end

@inline function _native_store_gaussian!(
    samples::AbstractMatrix,
    slot,
    gaussian,
    normals,
    offset,
    proposal_slot,
)
    valid = true
    for coordinate in axes(samples, 1)
        value = _native_gaussian_coordinate(
            gaussian,
            normals,
            offset,
            coordinate,
            proposal_slot,
        )
        @inbounds samples[coordinate, slot] = value
        valid &= isfinite(value)
    end
    return valid
end

@inline function _native_generate_sample!(samples, slot, base, ::_NoSampleTransform, normals, offset)
    valid = _native_store_gaussian!(samples, slot, base, normals, offset, 0)
    reason = valid ? UInt16(0) : _NATIVE_GENERATED_NONFINITE
    return zero(base.lognormalizer), reason, 0
end

@inline function _native_generate_sample!(
    samples,
    slot,
    base,
    transform::_NativeFusedNonidentityScalarTransform,
    normals,
    offset,
)
    coordinate = _gaussian_coordinate(base.location, base.scale, normals, offset, 1)
    sample, logabsjac, reason = _native_transform_with_logjac(transform, coordinate)
    @inbounds samples[slot] = sample
    return logabsjac, reason, 1
end

@inline function _native_generate_sample!(samples, slot, base, ::IdentityTransform, normals, offset)
    valid = _native_store_gaussian!(samples, slot, base, normals, offset, 0)
    reason = valid ? UInt16(0) : _NATIVE_GENERATED_NONFINITE
    return zero(base.lognormalizer), reason, 1
end

@inline function _native_simplex_transform!(
    output,
    slot,
    transform::SimplexTransform,
    base,
    normals,
    offset,
    selector,
)
    input_dimension = transform.dimension - 1
    coordinate_sum = zero(base.lognormalizer)
    coordinate_sum_correction = zero(base.lognormalizer)
    for output_index in 1:input_dimension
        coordinate_index = first(selector) + output_index - 1
        coordinate = _gaussian_coordinate(
            base.location,
            base.scale,
            normals,
            offset,
            coordinate_index,
        )
        isfinite(coordinate) ||
            return zero(coordinate), _NATIVE_TRANSFORM_NONFINITE_INPUT
        @inbounds output[output_index, slot] = coordinate
        coordinate_sum, coordinate_sum_correction = _compensated_add(
            coordinate_sum,
            coordinate_sum_correction,
            coordinate,
        )
    end
    isfinite(coordinate_sum) ||
        return coordinate_sum, _NATIVE_TRANSFORM_NONFINITE_OUTPUT

    inverse_root_dimension, shared_coefficient =
        _simplex_embedding_constants(typeof(coordinate_sum), transform.dimension)
    maximum_logit = inverse_root_dimension * coordinate_sum
    @inbounds output[transform.dimension, slot] = maximum_logit
    for output_index in 1:input_dimension
        logit = @inbounds(output[output_index, slot]) -
                shared_coefficient * coordinate_sum
        isfinite(logit) || return logit, _NATIVE_TRANSFORM_NONFINITE_OUTPUT
        @inbounds output[output_index, slot] = logit
        maximum_logit = max(maximum_logit, logit)
    end

    exponential_sum = zero(coordinate_sum)
    exponential_sum_correction = zero(coordinate_sum)
    for output_index in 1:transform.dimension
        weight = exp(@inbounds(output[output_index, slot]) - maximum_logit)
        @inbounds output[output_index, slot] = weight
        exponential_sum, exponential_sum_correction = _compensated_add(
            exponential_sum,
            exponential_sum_correction,
            weight,
        )
    end
    isfinite(exponential_sum) ||
        return exponential_sum, _NATIVE_TRANSFORM_NONFINITE_OUTPUT

    logabsjac = typeof(coordinate_sum)(0.5) * log(typeof(coordinate_sum)(transform.dimension))
    logabsjac_correction = zero(coordinate_sum)
    for output_index in 1:transform.dimension
        weight = @inbounds(output[output_index, slot]) / exponential_sum
        weight > zero(weight) || return logabsjac, _NATIVE_TRANSFORM_OUTSIDE_SUPPORT
        @inbounds output[output_index, slot] = weight
        logabsjac, logabsjac_correction = _compensated_add(
            logabsjac,
            logabsjac_correction,
            log(weight),
        )
    end
    isfinite(logabsjac) || return logabsjac, _NATIVE_TRANSFORM_NONFINITE_LOGJAC
    return logabsjac, UInt16(0)
end

@inline function _native_generate_sample!(
    samples,
    slot,
    base,
    transform::SimplexTransform,
    normals,
    offset,
)
    logabsjac, reason = _native_simplex_transform!(
        samples,
        slot,
        transform,
        base,
        normals,
        offset,
        1:(transform.dimension - 1),
    )
    return logabsjac, reason, 1
end

@inline function _native_transform_flat_block!(
    output,
    slot,
    block::_LocatedTransform{Int},
    base,
    normals,
    offset,
)
    coordinate = _gaussian_coordinate(
        base.location,
        base.scale,
        normals,
        offset,
        block.location,
    )
    sample, logabsjac, reason =
        _native_transform_with_logjac(block.transform, coordinate)
    @inbounds output[slot] = sample
    return logabsjac, reason
end

@inline function _native_transform_flat_block!(
    output,
    slot,
    block::_LocatedTransform{<:UnitRange},
    base,
    normals,
    offset,
)
    if block.transform isa SimplexTransform
        return _native_simplex_transform!(
            output,
            slot,
            block.transform,
            base,
            normals,
            offset,
            block.location,
        )
    end
    for output_index in eachindex(block.location)
        coordinate_index = first(block.location) + output_index - 1
        coordinate = _gaussian_coordinate(
            base.location,
            base.scale,
            normals,
            offset,
            coordinate_index,
        )
        @inbounds output[output_index, slot] = coordinate
        isfinite(coordinate) ||
            return zero(coordinate), _NATIVE_TRANSFORM_NONFINITE_INPUT
    end
    return zero(base.lognormalizer), UInt16(0)
end

@inline _native_transform_flat_blocks!(::Tuple{}, ::Tuple{}, slot, base, normals, offset, block) =
    (zero(base.lognormalizer), UInt16(0), 0)

@inline function _native_transform_flat_blocks!(
    outputs,
    blocks,
    slot,
    base,
    normals,
    offset,
    block_index,
)
    logabsjac, reason = _native_transform_flat_block!(
        first(outputs),
        slot,
        first(blocks),
        base,
        normals,
        offset,
    )
    iszero(reason) || return logabsjac, reason, block_index
    tail_logabsjac, tail_reason, failed_block = _native_transform_flat_blocks!(
        Base.tail(outputs),
        Base.tail(blocks),
        slot,
        base,
        normals,
        offset,
        block_index + 1,
    )
    return logabsjac + tail_logabsjac, tail_reason, failed_block
end

@inline function _native_generate_sample!(
    samples::NamedTuple,
    slot,
    base,
    layout::_FlatTransformLayout,
    normals,
    offset,
)
    return _native_transform_flat_blocks!(
        values(samples),
        values(layout.blocks),
        slot,
        base,
        normals,
        offset,
        1,
    )
end

@kernel function _native_gaussian_fused_kernel!(
    samples,
    logweights,
    failure_storage,
    normal_buffer,
    target,
    base,
    transform,
)
    slot = @index(Global, Linear)
    dimension = _gaussian_dimension(base.location)
    normal_offset = (slot - 1) * dimension + 1
    _, reason, block = _native_generate_sample!(
        samples,
        slot,
        base,
        transform,
        normal_buffer,
        normal_offset,
    )
    if !iszero(reason)
        _record_native_failure!(failure_storage, slot, block, reason)
    else
        sample = _native_sample_at(samples, slot)
        target_log, target_reason, target_failed = target(sample, slot)
        if target_failed
            iszero(target_reason) ||
                _record_native_failure!(failure_storage, slot, 0, target_reason)
        else
            proposal_log, density_reason = _native_generated_logdensity(
                base,
                transform,
                sample,
                normal_buffer,
                normal_offset,
            )
            proposal_reason = iszero(density_reason) ?
                              _native_proposal_reason(proposal_log) : density_reason
            if !iszero(proposal_reason)
                _record_native_failure!(failure_storage, slot, 0, proposal_reason)
            else
                logweight, logweight_reason = _subtract_logweight(
                    target_log,
                    proposal_log,
                )
                if iszero(logweight_reason)
                    @inbounds logweights[slot] = logweight
                else
                    _record_native_failure!(
                        failure_storage,
                        slot,
                        0,
                        logweight_reason,
                    )
                end
            end
        end
    end
end

_native_workgroupsize(::_SerialCPUExecution, nsamples) = nsamples
_native_workgroupsize(::_ThreadedCPUExecution, nsamples) = nothing

_native_failure_location(::_NoSampleTransform, block) = nothing
_native_failure_location(::_NativeScalarTransform, block) = nothing
_native_failure_location(transform::SimplexTransform, block) = 1:(transform.dimension - 1)
_native_failure_location(layout::_FlatTransformLayout, block) =
    getfield(values(layout.blocks), block).location

function _throw_native_failure(snapshot, transform)
    iszero(snapshot.count) && return nothing
    bits = snapshot.reason_bits
    index = snapshot.first_logical_index
    if bits & _NATIVE_TRANSFORM_REASONS != 0
        cause = InvalidTransformError(
            _native_transform_failure_reason(bits),
            _native_failure_location(transform, snapshot.first_block),
        )
        throw(SamplerExecutionError(:proposal_draw, index, CapturedException(cause, backtrace())))
    elseif bits & _NATIVE_GENERATED_NONFINITE != 0
        cause = DomainError(bits, "generated proposal samples must contain only finite values")
        throw(SamplerExecutionError(:proposal_draw, index, CapturedException(cause, backtrace())))
    elseif bits & (_NATIVE_TARGET_NAN | _NATIVE_TARGET_POSITIVE_INFINITY) != 0
        cause = DomainError(bits, "target log density may not be NaN or +Inf")
        throw(SamplerExecutionError(:target, index, CapturedException(cause, backtrace())))
    elseif bits & _NATIVE_LOGWEIGHT_INVALID != 0
        cause = DomainError(bits, "derived log weight may not be NaN or +Inf")
        throw(SamplerExecutionError(:logweight, index, CapturedException(cause, backtrace())))
    end
    cause = DomainError(bits, "proposal log density at a generated sample may not be NaN or -Inf")
    throw(
        SamplerExecutionError(
            :proposal_logdensity,
            index,
            CapturedException(cause, backtrace()),
        ),
    )
end

_take_first_native_target_failure!(::_NoNativeTargetFailures) = nothing

function _take_first_native_target_failure!(failures::_NativeCPUTargetFailures)
    index = failures.first_index[]
    index == typemax(Int) && return nothing
    failure = checkbounds(Bool, failures.slots, index) ?
              @inbounds(failures.slots[index]) : nothing
    fill!(failures.slots, nothing)
    failures.first_index[] = typemax(Int)
    return failure
end

function _throw_native_failures(snapshot, draw_snapshot, target_failures, transform)
    target_failure = _take_first_native_target_failure!(target_failures)
    _throw_native_failure(draw_snapshot, transform)
    isnothing(target_failure) || throw(target_failure)
    return _throw_native_failure(snapshot, transform)
end

function _launch_native_fused!(
    samples,
    logweights,
    failure_record,
    normal_buffer,
    target,
    base,
    transform,
    execution,
)
    backend = KernelAbstractions.get_backend(normal_buffer)
    kernel = _native_gaussian_fused_kernel!(backend)
    kernel(
        samples,
        logweights,
        failure_record.storage,
        normal_buffer,
        target,
        base,
        transform;
        ndrange=length(logweights),
        workgroupsize=_native_workgroupsize(execution, length(logweights)),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end
