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

const _NativeFusedScalarTransform = Union{
    IdentityTransform,
    PositiveTransform,
    SoftplusTransform,
    IntervalTransform,
}

_execution_name(::_SerialCPUExecution) = :serial
_execution_name(::_ThreadedCPUExecution) = :threaded
_execution_name(execution::_KernelExecution) = _execution_name(execution.cpu_execution)

_supports_native_fused_cpu(proposal) = false

function _supports_native_fused_cpu(proposal::_GaussianProposal)
    T = typeof(proposal.location)
    return proposal.location isa _NativeGaussianFloat &&
           proposal.scale isa _SphericalGaussianScale{T}
end

function _supports_native_fused_cpu(proposal::TransformedProposal)
    base = proposal.base
    transform = proposal.transform
    base isa _GaussianProposal || return false
    transform isa _NativeFusedScalarTransform || return false
    _supports_native_fused_cpu(base) || return false
    transform isa IntervalTransform || return true
    T = typeof(base.location)
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
    normal = similar(prototype, T, nsamples)
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
    isconcretetype(R) && all(isbitstype, fieldtypes(R)) || throw(
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
    logical_indices = Base.OneTo(nsamples)
    T = _native_fused_float_type(proposal)
    buffers = _capture_sampler_failure(:proposal_draw, 1) do
        _fill_random_buffers!(sampler.rng, sampler.random_buffers)
    end
    normal_buffer = buffers.normal
    base, transform = _native_fused_components(proposal)
    target = _capture_sampler_failure(:target, 1) do
        _bind_resolved_target(sampler.target, zero(T))
    end
    log_type = _resolve_logweight_type(target, proposal, T)
    samples = Vector{T}(undef, nsamples)
    logweights = Vector{log_type}(undef, nsamples)
    failures = Vector{Union{Nothing,SamplerExecutionError}}(undef, nsamples)
    fill!(failures, nothing)
    _launch_native_fused!(
        samples,
        logweights,
        failures,
        logical_indices,
        normal_buffer,
        target,
        proposal,
        base,
        transform,
        execution.cpu_execution,
    )
    _throw_first_native_failure(failures)
    return samples, logweights
end

_native_fused_components(proposal::_GaussianProposal) =
    (proposal, _NoSampleTransform())
_native_fused_components(proposal::TransformedProposal) =
    (proposal.base, proposal.transform)

function _native_fused_float_type(proposal)
    base, _ = _native_fused_components(proposal)
    return typeof(base.location)
end

@inline _native_transform_with_logjac(::_NoSampleTransform, coordinate) =
    (coordinate, zero(coordinate))
@inline _native_transform_with_logjac(transform, coordinate) =
    _transform_with_logjac(transform, coordinate)

function _native_generated_sample(base, transform, normal, sample_index)
    return _capture_sampler_failure(:proposal_draw, sample_index) do
        coordinate = base.location + base.scale.scale * normal
        sample, _ = _native_transform_with_logjac(transform, coordinate)
        sample
    end
end

function _native_generated_logdensity(proposal, sample, sample_index)
    return _capture_sampler_failure(:proposal_logdensity, sample_index) do
        value = DensityInterface.logdensityof(proposal, sample)
        _validate_proposal_logdensity(value)
        value
    end
end

@inline function _native_fused_sample!(
    samples,
    logweights,
    failures,
    slot,
    sample_index,
    normal,
    target,
    proposal,
    base,
    transform,
)
    sample = try
        _native_generated_sample(base, transform, normal, sample_index)
    catch error
        @inbounds failures[slot] = error::SamplerExecutionError
        return nothing
    end
    @inbounds samples[slot] = sample
    target_log = try
        _lossless_log_convert(
            eltype(logweights),
            _target_logdensity(target, sample, sample_index),
            :target,
            sample_index,
        )
    catch error
        @inbounds failures[slot] = error::SamplerExecutionError
        return nothing
    end
    proposal_log = try
        _lossless_log_convert(
            eltype(logweights),
            _native_generated_logdensity(proposal, sample, sample_index),
            :proposal_logdensity,
            sample_index,
        )
    catch error
        @inbounds failures[slot] = error::SamplerExecutionError
        return nothing
    end
    @inbounds logweights[slot] = target_log - proposal_log
    return nothing
end

@kernel function _native_scalar_fused_kernel!(
    samples,
    logweights,
    failures,
    logical_indices,
    normal_buffer,
    target,
    proposal,
    base,
    transform,
)
    slot = @index(Global, Linear)
    sample_index = @inbounds logical_indices[slot]
    _native_fused_sample!(
        samples,
        logweights,
        failures,
        slot,
        sample_index,
        @inbounds(normal_buffer[slot]),
        target,
        proposal,
        base,
        transform,
    )
end

_native_workgroupsize(::_SerialCPUExecution, nsamples) = nsamples
_native_workgroupsize(::_ThreadedCPUExecution, nsamples) = nothing

function _throw_first_native_failure(failures)
    for phase in (:proposal_draw, :target, :proposal_logdensity)
        for failure in failures
            if !isnothing(failure) && failure.phase === phase
                throw(failure)
            end
        end
    end
    return nothing
end

function _launch_native_fused!(
    samples,
    logweights,
    failures,
    logical_indices,
    normal_buffer,
    target,
    proposal,
    base,
    transform,
    execution,
)
    backend = KernelAbstractions.get_backend(samples)
    kernel = _native_scalar_fused_kernel!(backend)
    kernel(
        samples,
        logweights,
        failures,
        logical_indices,
        normal_buffer,
        target,
        proposal,
        base,
        transform;
        ndrange=length(normal_buffer),
        workgroupsize=_native_workgroupsize(execution, length(normal_buffer)),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end
