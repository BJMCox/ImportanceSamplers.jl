struct _SerialCPUExecution end
struct _ThreadedCPUExecution end

struct _KernelExecution{E}
    cpu_execution::E
end

struct _NoSampleTransform end
struct _NoRandomBuffers end

struct _RandomBuffers{U,N,F}
    uniform::U
    normal::N
    failure_scratch::F
end

struct _DeviceFailureRecord{A}
    storage::A
end

struct _NoNativeFailureScratch end

struct _NativeFailureScratch{R,F}
    record::R
    target_failures::F
end

struct _NoNativeTargetFailures end

struct _NativeDeviceTarget{L,T}
    target::T
end

function Adapt.adapt_structure(to, evaluator::_NativeDeviceTarget{L}) where {L}
    target = Adapt.adapt(to, evaluator.target)
    return _NativeDeviceTarget{L,typeof(target)}(target)
end

struct _NativeCPUTarget{L,T,F}
    target::T
    failures::F
end

const _NativeFusedNonidentityScalarTransform = Union{
    PositiveTransform,
    SoftplusTransform,
    IntervalTransform,
}

_execution_name(::_SerialCPUExecution) = :serial
_execution_name(::_ThreadedCPUExecution) = :threaded
_execution_name(execution::_KernelExecution) = _execution_name(execution.cpu_execution)

_supports_native_fused_cpu(proposal) = false

function _supports_native_fused_cpu(proposal::_GaussianProposal)
    location = proposal.location
    scale = proposal.scale
    if location isa _NativeGaussianFloat
        return scale isa _SphericalGaussianScale{typeof(location)}
    elseif location isa AbstractVector{<:_NativeGaussianFloat}
        T = eltype(location)
        return scale isa _SphericalGaussianScale{T} ||
               scale isa _DiagonalGaussianScale{<:AbstractVector{T}} ||
               scale isa _FactorGaussianScale{<:AbstractMatrix{T}}
    end
    return false
end

_supports_native_transform(::IdentityTransform, dimension) = true
_supports_native_transform(::SimplexTransform, dimension) = true
_supports_native_transform(::_NativeScalarTransform, dimension) = dimension == 1

function _supports_native_transform(layout::_FlatTransformLayout, dimension)
    layout.dimension == dimension || return false
    return all(values(layout.blocks)) do block
        selected_dimension = length(_selector_indices(block.location))
        _supports_native_transform(block.transform, selected_dimension)
    end
end

_supports_native_transform(transform, dimension) = false

function _supports_native_fused_cpu(proposal::TransformedProposal)
    base = proposal.base
    transform = proposal.transform
    base isa _GaussianProposal || return false
    _supports_native_fused_cpu(base) || return false
    dimension = _gaussian_dimension(base.location)
    _supports_native_transform(transform, dimension) || return false
    transform isa IntervalTransform || return true
    T = _native_fused_float_type(base)
    return transform isa IntervalTransform{T,T,Nothing} ||
           transform isa IntervalTransform{T,Nothing,T} ||
           transform isa IntervalTransform{T,T,T}
end

function _sampling_execution(proposal, threaded::Bool)
    cpu_execution = threaded ? _ThreadedCPUExecution() : _SerialCPUExecution()
    return _supports_native_fused_cpu(proposal) ?
           _KernelExecution(cpu_execution) : cpu_execution
end

function _allocate_random_buffers(device, proposal, nsamples)
    _supports_native_fused_cpu(proposal) || return _NoRandomBuffers()
    T = _native_fused_float_type(proposal)
    prototype = device(Vector{T}(undef, 0))
    uniform = similar(prototype, T, 0)
    normal = similar(prototype, T, _native_fused_dimension(proposal) * nsamples)
    failure_scratch = _allocate_native_failure_scratch(normal, nsamples)
    return _RandomBuffers(uniform, normal, failure_scratch)
end

_native_failure_scratch(::_NoRandomBuffers) = _NoNativeFailureScratch()
_native_failure_scratch(buffers::_RandomBuffers) = buffers.failure_scratch

function _fill_random_buffers!(rng, buffers::_RandomBuffers)
    isempty(buffers.uniform) || Random.rand!(rng, buffers.uniform)
    isempty(buffers.normal) || Random.randn!(rng, buffers.normal)
    return buffers
end

_owned_backend_rng(
    device::MLDataDevices.AbstractAcceleratorDevice,
    ::UInt64,
) = throw(SamplerDeviceError(device, :accelerator_rng_unavailable))

function _importance_sample!(sampler, ::_SerialCPUExecution)
    samples, logweights = _importance_sample_generic_cpu!(sampler, false)
    return samples, logweights, (count=0, bytes=0)
end

function _importance_sample!(sampler, ::_ThreadedCPUExecution)
    samples, logweights = _importance_sample_generic_cpu!(sampler, true)
    return samples, logweights, (count=0, bytes=0)
end

function _importance_sample!(sampler, execution::_KernelExecution)
    proposal = sampler.algorithm.proposal
    nsamples = sampler.algorithm.nsamples
    buffers = _capture_sampler_failure(:proposal_draw, 1) do
        _fill_random_buffers!(sampler.rng, sampler.random_buffers)
    end
    normal_buffer = buffers.normal
    base, transform = _native_fused_components(proposal)
    samples = _allocate_native_samples(normal_buffer, proposal, nsamples)
    binding_sample = _native_binding_sample(samples)
    target = _capture_sampler_failure(:target, 1) do
        _bind_resolved_target(sampler.target, binding_sample)
    end
    log_type = _resolve_native_logweight_type(target, base, typeof(binding_sample))
    logweights = similar(normal_buffer, log_type, nsamples)
    failure_scratch = sampler.random_buffers.failure_scratch
    failure_record = failure_scratch.record
    target_failures = failure_scratch.target_failures
    target_evaluator, target_failures = _native_target_evaluator(
        KernelAbstractions.get_backend(normal_buffer),
        target,
        log_type,
        target_failures,
    )
    _launch_native_fused!(
        samples,
        logweights,
        failure_record,
        normal_buffer,
        target_evaluator,
        base,
        transform,
        execution.cpu_execution,
    )
    snapshot = _device_failure_snapshot(failure_record)
    _throw_native_failures(snapshot.failure, target_failures, transform)
    return samples, logweights, snapshot.transfers
end

function _native_target_evaluator(
    ::KernelAbstractions.CPU,
    target,
    ::Type{L},
    failures::Vector{Union{Nothing,SamplerExecutionError}},
) where {L}
    return _NativeCPUTarget{L,typeof(target),typeof(failures)}(target, failures), failures
end

function _native_target_evaluator(
    backend,
    target,
    ::Type{L},
    failures::_NoNativeTargetFailures,
) where {L}
    return _NativeDeviceTarget{L,typeof(target)}(target), failures
end

function _preflight_native_kernel_target(
    device,
    target,
    proposal,
    buffers::_RandomBuffers,
)
    samples = _allocate_native_samples(buffers.normal, proposal, 1)
    binding_sample = _native_binding_sample(samples)
    bound_target = _bind_resolved_target(target, binding_sample)
    base, _ = _native_fused_components(proposal)
    log_type = _resolve_native_logweight_type(
        bound_target,
        base,
        typeof(binding_sample),
    )
    target_argument =
        _NativeDeviceTarget{log_type,typeof(bound_target)}(bound_target)
    backend = KernelAbstractions.get_backend(buffers.normal)
    kernel = _native_gaussian_fused_kernel!(backend)
    converted = try
        KernelAbstractions.argconvert(kernel, target_argument)
    catch
        throw(SamplerDeviceError(device, :kernel_argument_unsupported))
    end
    isbits(converted) || throw(
        SamplerDeviceError(device, :kernel_argument_unsupported),
    )
    return nothing
end

_native_fused_components(proposal::_GaussianProposal) =
    (proposal, _NoSampleTransform())
_native_fused_components(proposal::TransformedProposal) =
    (proposal.base, proposal.transform)

function _native_fused_float_type(proposal)
    base, _ = _native_fused_components(proposal)
    return base.location isa _NativeGaussianFloat ?
           typeof(base.location) : eltype(base.location)
end

function _native_fused_dimension(proposal)
    base, _ = _native_fused_components(proposal)
    return _gaussian_dimension(base.location)
end

function _resolve_native_logweight_type(target, base, sample_type::Type)
    target_type = _capture_sampler_failure(:target, 1) do
        _canonical_inferred_log_type(Base.promote_op(target, sample_type), "target")
    end
    proposal_type = _canonical_inferred_log_type(typeof(base.lognormalizer), "proposal")
    return target_type === Float64 || proposal_type === Float64 ? Float64 : Float32
end

_native_binding_sample(samples::AbstractVector{T}) where {T} = zero(T)
_native_binding_sample(samples::AbstractMatrix) = view(samples, :, 1)

@generated function _native_binding_sample(samples::NamedTuple{Names}) where {Names}
    leaves = map(Names) do name
        :(_native_binding_sample(getfield(samples, $(QuoteNode(name)))))
    end
    return :(NamedTuple{$Names}(($(leaves...),)))
end

function _allocate_native_samples(
    prototype,
    base::_GaussianProposal{F,T},
    nsamples,
) where {F,T<:_NativeGaussianFloat}
    return similar(prototype, T, nsamples)
end

function _allocate_native_samples(
    prototype,
    base::_GaussianProposal{F,<:AbstractVector{T}},
    nsamples,
) where {F,T<:_NativeGaussianFloat}
    return similar(prototype, T, _gaussian_dimension(base.location), nsamples)
end

function _allocate_native_samples(prototype, proposal::TransformedProposal, nsamples)
    base = proposal.base
    transform = proposal.transform
    T = _native_fused_float_type(base)
    if transform isa _NativeScalarTransform &&
       base.location isa _NativeGaussianFloat
        return similar(prototype, T, nsamples)
    elseif transform isa _FlatTransformLayout
        return map(transform.blocks) do block
            if block.location isa Int
                similar(prototype, T, nsamples)
            else
                output_dimension = block.transform isa SimplexTransform ?
                                   block.transform.dimension : length(block.location)
                similar(prototype, T, output_dimension, nsamples)
            end
        end
    end
    output_dimension = transform isa SimplexTransform ?
                       transform.dimension : _gaussian_dimension(base.location)
    return similar(prototype, T, output_dimension, nsamples)
end

function _allocate_native_failure_scratch(normal_buffer, nsamples)
    backend = KernelAbstractions.get_backend(normal_buffer)
    target_failures = _allocate_native_target_failures(backend, nsamples)
    storage = similar(normal_buffer, UInt64, 2)
    record = _DeviceFailureRecord(storage)
    return _NativeFailureScratch(record, target_failures)
end

_allocate_native_target_failures(::KernelAbstractions.CPU, nsamples) =
    Vector{Union{Nothing,SamplerExecutionError}}(nothing, nsamples)
_allocate_native_target_failures(backend, nsamples) = _NoNativeTargetFailures()

Base.@noinline _reset_native_failure_scratch!(::_NoNativeFailureScratch)::Nothing =
    nothing

Base.@noinline function _reset_native_failure_scratch!(
    scratch::_NativeFailureScratch,
)::Nothing
    fill!(scratch.record.storage, zero(UInt64))
    _reset_native_target_failures!(scratch.target_failures)
    return nothing
end

_reset_native_target_failures!(::_NoNativeTargetFailures) = nothing

function _reset_native_target_failures!(failures)
    fill!(failures, nothing)
    return nothing
end

@inline function _record_native_failure!(
    storage,
    logical_index::Int,
    block::Int,
    reason_bits::UInt16,
)
    inverse_index = typemax(UInt32) - UInt32(logical_index) + one(UInt32)
    inverse_block = typemax(UInt16) - UInt16(block) + one(UInt16)
    packed = UInt64(inverse_index) << 32 |
             UInt64(inverse_block) << 16 |
             UInt64(reason_bits)
    KernelAbstractions.@atomic storage[1] += UInt64(1)
    KernelAbstractions.@atomic storage[2] max packed
    return nothing
end

function _device_failure_snapshot(record::_DeviceFailureRecord)
    values = Array(record.storage)
    transfers = _is_host_storage(record.storage) ?
                (count=0, bytes=0) : (count=1, bytes=sizeof(values))
    count = values[1]
    packed = values[2]
    failure = iszero(count) ? (
        count=count,
        first_logical_index=0,
        first_block=0,
        reason_bits=UInt16(0),
    ) : let
        inverse_index = UInt32(packed >> 32)
        inverse_block = UInt16((packed >> 16) & 0xffff)
        (
            count=count,
            first_logical_index=Int(typemax(UInt32) - inverse_index + one(UInt32)),
            first_block=Int(typemax(UInt16) - inverse_block + one(UInt16)),
            reason_bits=UInt16(packed & 0xffff),
        )
    end
    return (
        failure=failure,
        transfers=transfers,
    )
end

const _NATIVE_TARGET_NAN = UInt16(0x0100)
const _NATIVE_TARGET_POSITIVE_INFINITY = UInt16(0x0200)
const _NATIVE_PROPOSAL_INVALID = UInt16(0x0400)
const _NATIVE_TRANSFORM_REASONS = UInt16(0x000f)

@inline function _native_target_reason(value)
    isnan(value) && return _NATIVE_TARGET_NAN
    value == Inf && return _NATIVE_TARGET_POSITIVE_INFINITY
    return UInt16(0)
end

@inline function (evaluator::_NativeDeviceTarget{L})(sample, slot) where {L}
    value = convert(L, evaluator.target(sample))
    reason = _native_target_reason(value)
    return value, reason, !iszero(reason)
end

@inline function _record_cpu_target_failure!(
    evaluator::_NativeCPUTarget{L},
    slot,
    cause,
    trace,
) where {L}
    @inbounds evaluator.failures[slot] = SamplerExecutionError(
        :target,
        slot,
        CapturedException(cause, trace),
    )
    return zero(L), UInt16(0), true
end

@inline function (evaluator::_NativeCPUTarget{L})(sample, slot) where {L}
    try
        value = convert(L, evaluator.target(sample))
        reason = _native_target_reason(value)
        if !iszero(reason)
            cause = DomainError(value, "target log density may not be NaN or +Inf")
            return _record_cpu_target_failure!(evaluator, slot, cause, backtrace())
        end
        return value, UInt16(0), false
    catch error
        return _record_cpu_target_failure!(evaluator, slot, error, catch_backtrace())
    end
end

@inline function _native_proposal_reason(value)
    (isnan(value) || value == -Inf) && return _NATIVE_PROPOSAL_INVALID
    return UInt16(0)
end

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

@inline _native_sample_at(samples::AbstractVector, slot) = @inbounds samples[slot]
@inline _native_sample_at(samples::AbstractMatrix, slot) = @view samples[:, slot]

@generated function _native_sample_at(samples::NamedTuple{Names}, slot) where {Names}
    leaves = map(Names) do name
        :(_native_sample_at(getfield(samples, $(QuoteNode(name))), slot))
    end
    return :(NamedTuple{$Names}(($(leaves...),)))
end

@inline function _native_store_gaussian!(samples::AbstractVector, slot, base, normals, offset)
    value = _gaussian_coordinate(base.location, base.scale, normals, offset, 1)
    @inbounds samples[slot] = value
    return isfinite(value)
end

@inline function _native_store_gaussian!(samples::AbstractMatrix, slot, base, normals, offset)
    valid = true
    for coordinate in 1:_gaussian_dimension(base.location)
        value = _gaussian_coordinate(base.location, base.scale, normals, offset, coordinate)
        @inbounds samples[coordinate, slot] = value
        valid &= isfinite(value)
    end
    return valid
end

@inline function _native_generate_sample!(samples, slot, base, ::_NoSampleTransform, normals, offset)
    _native_store_gaussian!(samples, slot, base, normals, offset)
    return zero(base.lognormalizer), UInt16(0), 0
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
    valid = _native_store_gaussian!(samples, slot, base, normals, offset)
    reason = valid ? UInt16(0) : _NATIVE_TRANSFORM_NONFINITE_INPUT
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
                @inbounds logweights[slot] = target_log - proposal_log
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
    elseif bits & (_NATIVE_TARGET_NAN | _NATIVE_TARGET_POSITIVE_INFINITY) != 0
        cause = DomainError(bits, "target log density may not be NaN or +Inf")
        throw(SamplerExecutionError(:target, index, CapturedException(cause, backtrace())))
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

_throw_first_native_target_failure(::_NoNativeTargetFailures) = nothing

function _throw_first_native_target_failure(failures)
    for failure in failures
        isnothing(failure) || throw(failure)
    end
    return nothing
end

function _throw_native_failures(snapshot, target_failures, transform)
    if !iszero(snapshot.count) && snapshot.reason_bits & _NATIVE_TRANSFORM_REASONS != 0
        _throw_native_failure(snapshot, transform)
    end
    _throw_first_native_target_failure(target_failures)
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
