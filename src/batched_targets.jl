_call_batch_target!(batch, values, samples, context) = batch(values, samples, context)
_call_batch_target!(batch, values, samples, ::_NoTargetContext) = batch(values, samples)

function _evaluate_generic_batch_logs!(logs, target, samples)
    isempty(logs) && return logs
    record = _DeviceFailureRecord(zeros(UInt64, 3))
    _invoke_batch_target!(target, logs, samples, record.storage, _SerialCPUExecution())
    transform = target isa _BoundNamedTarget ? target.layout : _NoSampleTransform()
    _throw_native_failure(_device_failure_snapshot(record).failure, transform)
    return logs
end

_has_batch_target(target) = false
_has_batch_target(::_BoundBatchTarget) = true
_has_batch_target(target::_PreparedLogTarget) = !isnothing(target.batch)
_has_batch_target(target::_NamedPreparedTarget) = _has_batch_target(target.target)
_has_batch_target(target::_BoundNamedTarget) = _has_batch_target(target.target)

_with_batch_transfers(target, transfers) = target
_with_batch_transfers(target::_BoundBatchTarget, transfers) =
    _BoundBatchTarget(target.scalar, target.batch, target.context, transfers)
_with_batch_transfers(target::_BoundNamedTarget, transfers) =
    _BoundNamedTarget(_with_batch_transfers(target.target, transfers), target.layout)
_bind_resolved_target(target, sample, transfers) =
    _with_batch_transfers(_bind_resolved_target(target, sample), transfers)
_batch_transfers(target::_BoundBatchTarget) = target.transfers
_batch_transfers(target::_BoundNamedTarget) = _batch_transfers(target.target)

function _batch_failure_snapshot(target, storage)
    snapshot = _device_failure_snapshot(_DeviceFailureRecord(storage))
    _record_reported_transfer!(_batch_transfers(target), snapshot.transfers.count,
        snapshot.transfers.bytes, Val(:failure_snapshot))
    return snapshot
end

_check_batch_failure!(target, ::Nothing, transform=_NoSampleTransform()) = nothing
function _check_batch_failure!(target, storage, transform=_NoSampleTransform())
    snapshot = _batch_failure_snapshot(target, storage)
    _throw_native_failure(snapshot.draw_failure, transform)
    _throw_native_failure(snapshot.failure, transform)
    return nothing
end

function _check_batch_mapping!(target, storage, capture)
    failure = _batch_failure_snapshot(target, storage).failure
    iszero(failure.count) && return nothing
    cause = InvalidTransformError(_native_transform_failure_reason(failure.reason_bits),
        _native_failure_location(target.layout, failure.first_block))
    capture || throw(cause)
    throw(SamplerExecutionError(:target, failure.first_logical_index,
        CapturedException(cause, backtrace())))
end

struct _NativeBatchTarget{L,T}
    target::T
end
Adapt.@adapt_structure _NativeBatchTarget

# Derivative kernels borrow the same scalar callable and context, without
# retaining an unused host batch callback in their device arguments.
_scalar_target(target) = target
_scalar_target(target::_PreparedLogTarget) = isnothing(target.batch) ? target :
    _PreparedLogTarget(target.logdensity, target.context, target.adtype, target.gradient)
_scalar_target(target::_NamedPreparedTarget) =
    _NamedPreparedTarget(_scalar_target(target.target), target.layout)
_scalar_target(target::_BoundBatchTarget) = target.scalar
_scalar_target(target::_BoundNamedTarget) =
    _BoundNamedTarget(_scalar_target(target.target), target.layout)

# Draws and proposal denominators do not depend on target values. The existing
# kernels compute their contribution with a zero target, then the batch adds
# the actual log target. This keeps one implementation of each sampling law.
struct _DeferredTargetValues{L} end
@inline (::_DeferredTargetValues{L})(sample) where {L} = zero(L)
@inline (::_DeferredTargetValues{L})(sample, slot) where {L} =
    (zero(L), UInt16(0), false)
_deferred_target(::_NativeBatchTarget{L}) where {L} = _DeferredTargetValues{L}()

struct _CachedTargetValues{V}
    values::V
end
Adapt.@adapt_structure _CachedTargetValues
@inline function (target::_CachedTargetValues)(sample, slot)
    value = target.values[slot]
    reason = _native_target_reason(value)
    return value, reason, !iszero(reason)
end

_native_device_evaluator(target, ::Type{L}) where {L} = _has_batch_target(target) ?
    _DeferredTargetValues{L}() : _NativeDeviceTarget{L,typeof(target)}(target)

function _invoke_batch_target!(target::_BoundBatchTarget, logs, samples, failure_storage, execution;
    transform=_NoSampleTransform())
    isempty(logs) && return nothing
    _check_batch_failure!(target, failure_storage, transform)
    backend = KernelAbstractions.get_backend(logs)
    fill!(logs, eltype(logs)(NaN))
    _capture_sampler_failure(:target_batch, 0) do
        try
            _call_batch_target!(target.batch, logs, samples, target.context)
        catch
            # A host callback can enqueue device writes before throwing.
            KernelAbstractions.synchronize(backend)
            rethrow()
        end
    end
    if failure_storage !== nothing
        _validate_batch_values_kernel!(backend)(logs, failure_storage;
            ndrange=length(logs), workgroupsize=_native_workgroupsize(execution, length(logs)))
    end
    KernelAbstractions.synchronize(backend)
    _check_batch_failure!(target, failure_storage, transform)
    return nothing
end

@kernel function _map_named_batch_kernel!(outputs, jacobians, samples, layout, failure_storage)
    index = @index(Global, Linear)
    logical, jacobian, reason, block = _coordinate_to_logical(layout, _native_sample_at(samples, index))
    jacobians[index] = jacobian
    if iszero(reason)
        _store_named_result_values!(outputs, values(logical), index)
    else
        jacobians[index] = eltype(jacobians)(NaN)
        failure_storage === nothing || _record_native_failure!(failure_storage, index, block, reason)
    end
end

function _invoke_batch_target!(target::_BoundNamedTarget, logs, samples, failure_storage, execution;
    transform=_NoSampleTransform())
    _check_batch_failure!(target, failure_storage, transform)
    mapped = _allocate_flat_samples(samples, eltype(samples), target.layout, length(logs))
    jacobians = similar(logs)
    backend = KernelAbstractions.get_backend(logs)
    mapping_failures = failure_storage
    if mapping_failures === nothing
        mapping_failures = _allocate_failure_storage(backend, logs, length(logs))
        fill!(mapping_failures, zero(eltype(mapping_failures)))
    end
    _map_named_batch_kernel!(backend)(values(mapped), jacobians, samples, target.layout, mapping_failures;
        ndrange=length(logs), workgroupsize=_native_workgroupsize(execution, length(logs)))
    KernelAbstractions.synchronize(backend)
    _check_batch_mapping!(target, mapping_failures, failure_storage !== nothing)
    _invoke_batch_target!(target.target, logs, mapped, failure_storage, execution)
    logs .+= jacobians
    if failure_storage !== nothing
        _validate_batch_values_kernel!(backend)(logs, failure_storage;
            ndrange=length(logs), workgroupsize=_native_workgroupsize(execution, length(logs)))
    end
    KernelAbstractions.synchronize(backend)
    _check_batch_failure!(target, failure_storage, target.layout)
    return nothing
end

@kernel function _validate_batch_values_kernel!(logs, failure_storage)
    index = @index(Global, Linear)
    reason = _native_target_reason(logs[index])
    iszero(reason) || _record_native_failure!(failure_storage, index, 0, reason)
end

_batch_logweights(output::AbstractVector) = output
_batch_logweights(output::Union{_MISRoundOutput,_GaussianRoundOutput}) = output.logweights
_batch_value_buffer(output) = similar(_batch_logweights(output))
_batch_value_buffer(output::_GaussianRoundOutput) = output.logtargets
_batch_value_buffer(output::_MISRoundOutput{W,I,<:_MISAdaptationOutput}) where {W,I} =
    output.adaptation.logtargets

@inline _finish_batch_weight!(output::AbstractVector, value, index) =
    _subtract_logweight(value, -output[index])
@inline function _finish_batch_weight!(output::_MISRoundOutput, value, index)
    _store_mis_adaptation!(output.adaptation, index, value, Val(:target))
    return _subtract_logweight(value, -output.logweights[index])
end

function _launch_gaussian_target!(
    logtargets, lognumerators, logweights, round_ids, samples, target, round, failure_storage, execution,
)
    backend = KernelAbstractions.get_backend(logweights)
    evaluator = target isa _NativeBatchTarget ? _deferred_target(target) : target
    _gaussian_batch_target_kernel!(backend)(logtargets, lognumerators, logweights,
        round_ids, samples, evaluator, round, failure_storage;
        ndrange=length(logweights),
        workgroupsize=_native_workgroupsize(execution, length(logweights)))
    if target isa _NativeBatchTarget
        _invoke_batch_target!(target.target, logtargets, samples, failure_storage, execution)
    end
    return nothing
end
@inline _finish_batch_weight!(output::_GaussianRoundOutput, value, index) =
    _logweight_from_logmixture(value, output.lognumerators[index], output.logtotal)

@kernel function _finish_batch_weights_kernel!(output, logs, failure_storage)
    index = @index(Global, Linear)
    value = logs[index]
    if iszero(_native_target_reason(value))
        weight, reason = _finish_batch_weight!(output, value, index)
        if iszero(reason)
            _batch_logweights(output)[index] = weight
        else
            _record_native_failure!(failure_storage, index, 0, reason)
        end
    end
end

function _finish_batch_weights!(output, target, samples, failure_storage, execution;
    transform=_NoSampleTransform())
    logs = _batch_value_buffer(output)
    samples = _sample_count(samples) == length(logs) ? samples : _sample_view(samples, 1:length(logs))
    _invoke_batch_target!(target, logs, samples, failure_storage, execution; transform)
    backend = KernelAbstractions.get_backend(logs)
    _finish_batch_weights_kernel!(backend)(output, logs, failure_storage;
        ndrange=length(logs), workgroupsize=_native_workgroupsize(execution, length(logs)))
    KernelAbstractions.synchronize(backend)
    return nothing
end
