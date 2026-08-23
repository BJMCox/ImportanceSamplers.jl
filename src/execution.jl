struct _SerialCPUExecution end
struct _ThreadedCPUExecution end

struct _KernelExecution{E}
    cpu_execution::E
end

struct _NoSampleTransform end
struct _NoRandomBuffers end

struct _RandomBuffers{U,N}
    uniform::U
    normal::N
end

struct _DeviceFailureRecord{A}
    storage::A
end

const _NativeFusedScalarTransform = Union{
    IdentityTransform,
    PositiveTransform,
    SoftplusTransform,
    IntervalTransform,
}
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
_supports_native_transform(::_NativeFusedScalarTransform, dimension) = dimension == 1

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
    return _RandomBuffers(uniform, normal)
end

function _fill_random_buffers!(rng, buffers::_RandomBuffers)
    isempty(buffers.uniform) || Random.rand!(rng, buffers.uniform)
    isempty(buffers.normal) || Random.randn!(rng, buffers.normal)
    return buffers
end

_owned_backend_rng(
    device::MLDataDevices.AbstractAcceleratorDevice,
    ::UInt64,
) = throw(SamplerDeviceError(device, :accelerator_rng_unavailable))

function _owned_backend_rng(device::MLDataDevices.CUDADevice, seed::UInt64)
    # MLDataDevices delegates to CUDA.default_rng(), a mutable task-local cache
    # shared by samplers created on the same task. Use it only as a concrete type
    # witness; never retain or seed that shared object.
    shared = try
        MLDataDevices.default_device_rng(device)
    catch
        throw(SamplerDeviceError(device, :accelerator_rng_unavailable))
    end
    R = typeof(shared)
    isconcretetype(R) &&
        ismutabletype(R) &&
        fieldnames(R) == (:seed, :counter) &&
        fieldtypes(R) == (UInt64, UInt64) || throw(
        SamplerDeviceError(device, :accelerator_rng_unavailable),
    )
    owned = try
        R(seed)
    catch
        throw(SamplerDeviceError(device, :accelerator_rng_unavailable))
    end
    independent = try
        R(seed)
    catch
        throw(SamplerDeviceError(device, :accelerator_rng_unavailable))
    end
    owned isa Random.AbstractRNG && owned !== shared && owned !== independent || throw(
        SamplerDeviceError(device, :accelerator_rng_unavailable),
    )
    return owned
end

_importance_sample!(sampler, ::_SerialCPUExecution) =
    _importance_sample_generic_cpu!(sampler, false)
_importance_sample!(sampler, ::_ThreadedCPUExecution) =
    _importance_sample_generic_cpu!(sampler, true)

function _importance_sample!(sampler, execution::_KernelExecution)
    proposal = sampler.algorithm.proposal
    nsamples = sampler.algorithm.nsamples
    T = _native_fused_float_type(proposal)
    buffers = _capture_sampler_failure(:proposal_draw, 1) do
        _fill_random_buffers!(sampler.rng, sampler.random_buffers)
    end
    normal_buffer = buffers.normal
    base, transform = _native_fused_components(proposal)
    sample_template = _native_sample_template(proposal)
    target = _capture_sampler_failure(:target, 1) do
        _bind_resolved_target(sampler.target, sample_template)
    end
    log_type = _resolve_logweight_type(target, proposal, typeof(sample_template))
    samples = _allocate_native_samples(normal_buffer, proposal, nsamples)
    logweights = similar(normal_buffer, log_type, nsamples)
    failure_record = _allocate_device_failure_record(normal_buffer)
    _launch_native_fused!(
        samples,
        logweights,
        failure_record,
        normal_buffer,
        target,
        base,
        transform,
        execution.cpu_execution,
    )
    _throw_native_failure(_device_failure_snapshot(failure_record), transform)
    return samples, logweights
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

_native_sample_template(base::_GaussianProposal{F,T}) where {F,T<:_NativeGaussianFloat} =
    zero(T)
_native_sample_template(base::_GaussianProposal{F,<:AbstractVector{T}}) where {F,T} =
    zeros(T, _gaussian_dimension(base.location))

function _native_sample_template(proposal::TransformedProposal)
    base = proposal.base
    transform = proposal.transform
    T = _native_fused_float_type(base)
    if transform isa IdentityTransform && base.location isa AbstractVector
        return zeros(T, _gaussian_dimension(base.location))
    elseif transform isa _NativeFusedScalarTransform
        return zero(T)
    elseif transform isa SimplexTransform
        return zeros(T, transform.dimension)
    end
    return _native_flat_transform_template(transform, T)
end

function _native_flat_transform_template(layout::_FlatTransformLayout, ::Type{T}) where {T}
    return map(layout.blocks) do block
        block.transform isa SimplexTransform ?
        zeros(T, block.transform.dimension) :
        block.location isa Int ? zero(T) : zeros(T, length(block.location))
    end
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
    if transform isa _NativeFusedScalarTransform &&
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

function _allocate_device_failure_record(prototype)
    storage = similar(prototype, UInt64, 2)
    fill!(storage, zero(UInt64))
    return _DeviceFailureRecord(storage)
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
    KernelAbstractions.@atomic storage[2] = max(storage[2], packed)
    return nothing
end

function _device_failure_snapshot(record::_DeviceFailureRecord)
    values = Array(record.storage)
    count = values[1]
    packed = values[2]
    iszero(count) && return (
        count=count,
        first_logical_index=0,
        first_block=0,
        reason_bits=UInt16(0),
    )
    inverse_index = UInt32(packed >> 32)
    inverse_block = UInt16((packed >> 16) & 0xffff)
    return (
        count=count,
        first_logical_index=Int(typemax(UInt32) - inverse_index + one(UInt32)),
        first_block=Int(typemax(UInt16) - inverse_block + one(UInt16)),
        reason_bits=UInt16(packed & 0xffff),
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

@inline function _native_proposal_reason(value)
    (isnan(value) || value == -Inf) && return _NATIVE_PROPOSAL_INVALID
    return UInt16(0)
end

@inline function _native_generated_logdensity(
    base::_GaussianProposal{F,T},
    transform::IntervalTransform{T,T,T},
    sample,
    normals,
    normal_offset,
    forward_logabsjac,
) where {F,T<:_NativeGaussianFloat}
    coordinate, logabsjac, reason = _native_inverse_with_logjac(transform, sample)
    iszero(reason) || return base.lognormalizer, reason
    standardized = (coordinate - base.location) / base.scale.scale
    base_logdensity = base.lognormalizer -
                      oftype(base.lognormalizer, 0.5) * abs2(standardized)
    return base_logdensity - logabsjac, UInt16(0)
end

@inline function _native_generated_logdensity(
    base,
    transform,
    sample,
    normals,
    normal_offset,
    forward_logabsjac,
)
    return _gaussian_logdensity_from_normal(base, normals, normal_offset) -
           forward_logabsjac, UInt16(0)
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
    logabsjac, reason, block = _native_generate_sample!(
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
        target_log = target(sample)
        target_reason = _native_target_reason(target_log)
        if !iszero(target_reason)
            _record_native_failure!(failure_storage, slot, 0, target_reason)
        else
            proposal_log, density_reason = _native_generated_logdensity(
                base,
                transform,
                sample,
                normal_buffer,
                normal_offset,
                logabsjac,
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

function _native_transform_failure_reason(reason_bits)
    reason_bits & _NATIVE_TRANSFORM_NONFINITE_INPUT != 0 && return :nonfinite_input
    reason_bits & _NATIVE_TRANSFORM_NONFINITE_OUTPUT != 0 && return :nonfinite_output
    reason_bits & _NATIVE_TRANSFORM_NONFINITE_LOGJAC != 0 && return :nonfinite_logabsjac
    return :outside_support
end

_native_failure_location(::_NoSampleTransform, block) = nothing
_native_failure_location(::_NativeFusedScalarTransform, block) = nothing
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
