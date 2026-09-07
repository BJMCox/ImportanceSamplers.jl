struct _SerialCPUExecution end
struct _ThreadedCPUExecution end

function _threaded_foreach(f, indices)
    item_count = length(indices)
    iszero(item_count) && return nothing
    task_count = min(Threads.nthreads(:default), item_count)
    chunk_size = cld(item_count, task_count)
    @sync for first_position in 1:chunk_size:item_count
        last_position = min(first_position + chunk_size - 1, item_count)
        Threads.@spawn for position in first_position:last_position
            @inbounds f(indices[position])
        end
    end
    return nothing
end

struct _ReportedTransfer
    count::Int
    bytes::Int
end

mutable struct _ReportedTransferReasons
    failure_snapshot::_ReportedTransfer
    cdf_maximum::_ReportedTransfer
    cdf_sum::_ReportedTransfer
    local_resampling_validity::_ReportedTransfer
    local_mean_validity::_ReportedTransfer
    logweight_maximum::_ReportedTransfer
    logweight_scaled_sum::_ReportedTransfer
    logweight_scaled_square_sum::_ReportedTransfer
    covariance_diagnostic::_ReportedTransfer
end

function _ReportedTransferReasons()
    return _ReportedTransferReasons(
        _ReportedTransfer(0, 0),
        _ReportedTransfer(0, 0),
        _ReportedTransfer(0, 0),
        _ReportedTransfer(0, 0),
        _ReportedTransfer(0, 0),
        _ReportedTransfer(0, 0),
        _ReportedTransfer(0, 0),
        _ReportedTransfer(0, 0),
        _ReportedTransfer(0, 0),
    )
end

mutable struct _ResultTransferCounter
    count::Int
    bytes::Int
    reasons::_ReportedTransferReasons
end

_ResultTransferCounter(count::Int, bytes::Int) =
    _ResultTransferCounter(count, bytes, _ReportedTransferReasons())

function _record_reported_transfer!(
    counter::_ResultTransferCounter,
    count,
    bytes,
    ::Val{R},
) where {R}
    reason = getfield(counter.reasons, R)
    counter.count += count
    counter.bytes += bytes
    setfield!(
        counter.reasons,
        R,
        _ReportedTransfer(reason.count + count, reason.bytes + bytes),
    )
    return nothing
end

function _record_scalar_transfer!(counter::_ResultTransferCounter, ::Type{T}) where {T}
    counter.count += 1
    counter.bytes += sizeof(T)
    return nothing
end

function _record_scalar_transfer!(
    counter::_ResultTransferCounter,
    ::Type{T},
    reason::Val,
) where {T}
    return _record_reported_transfer!(counter, 1, sizeof(T), reason)
end

_record_device_scalar_transfer!(counter, storage, type::Type) =
    _is_host_storage(storage) ? nothing : _record_scalar_transfer!(counter, type)

_record_device_scalar_transfer!(counter, storage, type::Type, reason::Val) =
    _is_host_storage(storage) ?
    nothing : _record_scalar_transfer!(counter, type, reason)

function _logweight_summary(
    logweights,
    transfers::_ResultTransferCounter=_ResultTransferCounter(0, 0),
)
    maximum_logweight = maximum(logweights)
    _record_device_scalar_transfer!(
        transfers,
        logweights,
        eltype(logweights),
        Val(:logweight_maximum),
    )
    maximum_logweight == eltype(logweights)(-Inf) && return (
        ess=zero(eltype(logweights)),
        lognormalizer=eltype(logweights)(-Inf),
    )
    scaled_sum = mapreduce(
        value -> exp(value - maximum_logweight),
        +,
        logweights;
        init=zero(eltype(logweights)),
    )
    _record_device_scalar_transfer!(
        transfers,
        logweights,
        eltype(logweights),
        Val(:logweight_scaled_sum),
    )
    scaled_square_sum = mapreduce(
        value -> abs2(exp(value - maximum_logweight)),
        +,
        logweights;
        init=zero(eltype(logweights)),
    )
    _record_device_scalar_transfer!(
        transfers,
        logweights,
        eltype(logweights),
        Val(:logweight_scaled_square_sum),
    )
    return _logweight_summary(
        maximum_logweight,
        scaled_sum,
        scaled_square_sum,
        length(logweights),
    )
end

@inline function _logweight_summary(
    maximum_logweight,
    scaled_sum,
    scaled_square_sum,
    sample_count,
)
    T = typeof(maximum_logweight)
    return (
        ess=abs2(scaled_sum) / scaled_square_sum,
        lognormalizer=maximum_logweight + log(scaled_sum) - log(T(sample_count)),
    )
end

@inline function _scale_aware_ridge(
    previous_trace,
    dimension,
    regularization::T,
) where {T}
    return regularization * previous_trace / T(dimension)
end

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

struct _PackedStaticMISRandomBuffers{U,N,A,S,F}
    uniform::U
    normal::N
    assignments::A
    solve_scratch::S
    failure_scratch::F
end

Adapt.@adapt_structure _PackedStaticMISRandomBuffers

struct _DMPMCRandomBuffers{N,U,F}
    normals::N
    resampling_uniforms::U
    failure_scratch::F
end

struct _PopulationNormalBuffers{N,F}
    normals::N
    failure_scratch::F
end

struct _DeviceFailureRecord{A}
    storage::A
end

Adapt.@adapt_structure _DeviceFailureRecord

struct _NoNativeFailureScratch end

struct _NativeFailureScratch{R,F}
    record::R
    target_failures::F
end

Adapt.@adapt_structure _NativeFailureScratch

struct _NoNativeTargetFailures end

struct _NativeCPUTargetFailures
    slots::Vector{Union{Nothing,SamplerExecutionError}}
    first_index::Threads.Atomic{Int}
end

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

function _supports_native_radial_storage(proposal)
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
_supports_native_fused_cpu(proposal::_GaussianProposal) =
    _supports_native_radial_storage(proposal)
_supports_native_fused_cpu(proposal::_StudentTProposal) =
    _supports_native_radial_storage(proposal)

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
    base isa Union{_GaussianProposal,_StudentTProposal} || return false
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
    uniform = similar(prototype, T, _native_uniform_count(proposal, nsamples))
    normal = similar(prototype, T, _native_normal_count(proposal, nsamples))
    failure_scratch = _allocate_native_failure_scratch(normal, nsamples)
    return _RandomBuffers(uniform, normal, failure_scratch)
end

function _allocate_random_buffers(
    device,
    proposal,
    ::_SingleProposalMethodState,
    nsamples,
)
    return _allocate_random_buffers(device, proposal, nsamples)
end

_native_failure_scratch(::_NoRandomBuffers) = _NoNativeFailureScratch()
_native_failure_scratch(
    buffers::Union{
        _RandomBuffers,
        _PackedStaticMISRandomBuffers,
        _DMPMCRandomBuffers,
        _PopulationNormalBuffers,
    },
) = buffers.failure_scratch

function _fill_random_buffers!(
    rng,
    buffers::Union{_RandomBuffers,_PackedStaticMISRandomBuffers},
)
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
    if _use_native_factor_batch_path(
        sampler.device,
        base,
        transform,
        sampler.factor_execution,
    )
        _launch_native_factor_batch!(
            samples,
            logweights,
            failure_record,
            normal_buffer,
            target_evaluator,
            base,
            execution.cpu_execution,
        )
    else
        _launch_native_fused!(
            samples,
            logweights,
            failure_record,
            buffers.uniform,
            normal_buffer,
            target_evaluator,
            base,
            transform,
            execution.cpu_execution,
        )
    end
    snapshot = _device_failure_snapshot(failure_record)
    _throw_native_failures(
        snapshot.failure,
        snapshot.draw_failure,
        target_failures,
        transform,
    )
    return samples, logweights, snapshot.transfers
end

function _native_target_evaluator(
    ::KernelAbstractions.CPU,
    target,
    ::Type{L},
    failures::_NativeCPUTargetFailures,
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
    factor_execution,
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
    kernel = _native_fused_kernel!(backend)
    _preflight_kernel_argument(device, kernel, target_argument)
    if _use_native_factor_batch_path(
        device,
        base,
        _native_fused_components(proposal)[2],
        factor_execution,
    )
        batch_kernel = _native_factor_batch_finish_kernel!(backend)
        _preflight_kernel_argument(device, batch_kernel, target_argument)
    end
    return nothing
end

function _preflight_kernel_argument(device, kernel, argument)
    converted = try
        KernelAbstractions.argconvert(kernel, argument)
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
_native_fused_components(proposal::_StudentTProposal) =
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
const _STUDENT_T_GAMMA_ATTEMPTS = 8

_native_normal_stride(base::_GaussianProposal) = _gaussian_dimension(base.location)
_native_uniform_stride(::_GaussianProposal) = 0

function _native_normal_stride(base::_StudentTProposal)
    dimension = _gaussian_dimension(base.location)
    return dimension + (isone(base.family.dof) ? 1 : _STUDENT_T_GAMMA_ATTEMPTS)
end

function _native_uniform_stride(base::_StudentTProposal)
    isone(base.family.dof) && return 0
    return _STUDENT_T_GAMMA_ATTEMPTS + (base.family.dof < typeof(base.family.dof)(2))
end

function _native_normal_count(proposal, nsamples)
    base, _ = _native_fused_components(proposal)
    return _native_normal_stride(base) * nsamples
end

function _native_uniform_count(proposal, nsamples)
    base, _ = _native_fused_components(proposal)
    return _native_uniform_stride(base) * nsamples
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
    base::_NativeRadialProposal,
    nsamples,
)
    T = _native_fused_float_type(base)
    return base.location isa _NativeGaussianFloat ?
           similar(prototype, T, nsamples) :
           similar(prototype, T, _gaussian_dimension(base.location), nsamples)
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
    storage = similar(normal_buffer, UInt64, 3)
    record = _DeviceFailureRecord(storage)
    return _NativeFailureScratch(record, target_failures)
end

function _allocate_native_target_failures(::KernelAbstractions.CPU, nsamples)
    slots = Vector{Union{Nothing,SamplerExecutionError}}(nothing, nsamples)
    return _NativeCPUTargetFailures(slots, Threads.Atomic{Int}(typemax(Int)))
end
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

function _reset_native_target_failures!(failures::_NativeCPUTargetFailures)
    failures.first_index[] == typemax(Int) && return nothing
    fill!(failures.slots, nothing)
    failures.first_index[] = typemax(Int)
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
    if reason_bits & _NATIVE_PROPOSAL_DRAW_REASONS != 0
        KernelAbstractions.@atomic storage[3] max packed
    end
    return nothing
end

@inline function _decode_native_failure(count, packed)
    return iszero(count) ? (
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
end

function _device_failure_snapshot(record::_DeviceFailureRecord)
    values = Array(record.storage)
    transfers = _is_host_storage(record.storage) ?
                (count=0, bytes=0) : (count=1, bytes=sizeof(values))
    count = values[1]
    failure = _decode_native_failure(count, values[2])
    draw_failure = _decode_native_failure(
        iszero(values[3]) ? zero(count) : one(count),
        values[3],
    )
    return (
        failure=failure,
        draw_failure=draw_failure,
        transfers=transfers,
    )
end

const _NATIVE_TARGET_NAN = UInt16(0x0100)
const _NATIVE_TARGET_POSITIVE_INFINITY = UInt16(0x0200)
const _NATIVE_PROPOSAL_INVALID = UInt16(0x0400)
const _NATIVE_GENERATED_NONFINITE = UInt16(0x0800)
const _NATIVE_LOGWEIGHT_INVALID = UInt16(0x1000)
const _NATIVE_PROPOSAL_DRAW_EXHAUSTED = UInt16(0x0010)
const _NATIVE_TRANSFORM_REASONS = UInt16(0x000f)
const _NATIVE_PROPOSAL_DRAW_REASONS =
    _NATIVE_TRANSFORM_REASONS | _NATIVE_GENERATED_NONFINITE |
    _NATIVE_PROPOSAL_DRAW_EXHAUSTED

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
    failures = evaluator.failures
    @inbounds failures.slots[slot] = SamplerExecutionError(
        :target,
        slot,
        CapturedException(cause, trace),
    )
    Threads.atomic_min!(failures.first_index, slot)
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
