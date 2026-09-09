_gaussian_round_error(::NPMC) = NPMCRoundError
_gaussian_method_name(::NPMC) = :npmc
_gaussian_proposal_evaluations(::NPMC, n) = n
_allocate_gaussian_adaptation_ess(::NPMC, round_ess) = similar(round_ess)
_gaussian_extra_diagnostics(::NPMC, adaptation_ess, schedule) =
    (; adaptation_ess, clipped_counts=isqrt.(collect(schedule)))

_gaussian_summary_indices(::NPMC, state, round) =
    state.offsets[round]:(state.offsets[round + 1] - 1)

function _sort_clipping_weights!(::MLDataDevices.AbstractCPUDevice, scratch, threshold_index)
    partialsort!(scratch, threshold_index)
    return nothing
end

function _sort_clipping_weights!(device, scratch, threshold_index)
    AcceleratedKernels.sort!(scratch)
    return nothing
end

function _gaussian_adaptation_workspace!(algorithm::NPMC, device, state, round)
    workspace = state.workspace
    indices = _gaussian_summary_indices(algorithm, state, round)
    raw = view(workspace.logweights, indices)
    scratch = view(workspace.lognumerators, indices)
    clipped = view(workspace.logtargets, indices)
    copyto!(scratch, raw)
    threshold_index = length(indices) - isqrt(length(indices)) + 1
    _sort_clipping_weights!(device, scratch, threshold_index)
    # A one-element device view keeps the order statistic on its device.
    clipped .= min.(raw, view(scratch, threshold_index:threshold_index))
    return _GaussianMomentWorkspace(
        _sample_view(workspace.samples, indices),
        clipped,
        scratch,
        clipped,
        view(workspace.normalized_weights, indices),
        _sample_view(workspace.centered_scaled, indices),
        workspace.covariance,
        workspace.candidate_mean,
        workspace.candidate_scale,
        workspace.candidate_lognormalizer,
    )
end

struct _GeneratingGaussianDenominator end
@inline _mis_term_bounds(history, ::_GeneratingGaussianDenominator, slot) = (slot, slot)
@inline _mis_denominator_term(::Type{T}, history, ::_GeneratingGaussianDenominator, slot) where {T} =
    (slot, zero(T))

function _launch_adaptive_gaussian_round!(
    ::NPMC, samples, logtargets, scratch, logweights, round_ids,
    failure_storage, normal_buffer, target, history, logcounts, offsets,
    round, solve_scratch, execution, device, factor_execution,
)
    indices = offsets[round]:(offsets[round + 1] - 1)
    new_samples = _sample_view(samples, indices)
    output = _GaussianRoundOutput(
        view(logtargets, indices), view(scratch, indices),
        view(logweights, indices), view(round_ids, indices),
        zero(eltype(logweights)), round,
    )
    if history isa _GaussianFactorHistory &&
       _use_factor_batch_path(device, history, factor_execution)
        _launch_npmc_factor_batch!(
            new_samples, output, failure_storage, normal_buffer,
            target, history, _sample_view(solve_scratch, indices), execution,
        )
    else
        _launch_mis_round!(
            new_samples, output, failure_storage, normal_buffer, target,
            history, _FixedMISAssignments(round, length(indices)),
            _GeneratingGaussianDenominator(), solve_scratch, execution,
        )
    end
    return nothing
end

@kernel function _npmc_factor_weights_kernel!(
    logweights, logtargets, solved, lognormalizers, slot, failure_storage,
)
    sample_index = @index(Global, Linear)
    T = eltype(logweights)
    square_norm = zero(eltype(solved))
    @inbounds for coordinate in axes(solved, 1)
        square_norm += abs2(solved[coordinate, sample_index])
    end
    logproposal = convert(T, @inbounds(lognormalizers[slot]) - square_norm / 2)
    reason = _native_proposal_reason(logproposal)
    if iszero(reason)
        weight, reason = _subtract_logweight(@inbounds(logtargets[sample_index]), logproposal)
        iszero(reason) && (@inbounds logweights[sample_index] = weight)
    end
    iszero(reason) || _record_native_failure!(failure_storage, sample_index, 0, reason)
end

function _launch_npmc_factor_batch!(
    samples, output, failure_storage, normals, target, history, solve_scratch, execution,
)
    count = length(output.logweights)
    dimension = size(samples, 1)
    backend = KernelAbstractions.get_backend(normals)
    _factor_batch_draw!(
        samples, reshape(view(normals, 1:(dimension * count)), dimension, count),
        history, output.round,
    )
    kernel = _gaussian_batch_target_kernel!(backend)
    kernel(
        output.logtargets, output.lognumerators, output.logweights, output.round_ids,
        samples, target, output.round, failure_storage;
        ndrange=count, workgroupsize=_native_workgroupsize(execution, count),
    )
    _factor_batch_solve!(solve_scratch, samples, history, output.round)
    weight_kernel = _npmc_factor_weights_kernel!(backend)
    weight_kernel(
        output.logweights, output.logtargets, solve_scratch, history.lognormalizers,
        output.round, failure_storage;
        ndrange=count, workgroupsize=_native_workgroupsize(execution, count),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

function _preflight_accelerator_method(
    device, target, algorithm::NPMC, state::_PreparedAdaptiveGaussian,
    buffers::_RandomBuffers, factor_execution,
)
    workspace = state.workspace
    binding_sample = _native_binding_sample(workspace.samples)
    bound_target = _bind_resolved_target(target, binding_sample)
    T = eltype(workspace.logweights)
    target_argument = _NativeDeviceTarget{T,typeof(bound_target)}(bound_target)
    backend = KernelAbstractions.get_backend(buffers.normal)
    kernel = _gaussian_round_launch_kernel!(backend)
    round = findmax(state.schedule)[2]
    indices = _gaussian_summary_indices(algorithm, state, round)
    samples = _sample_view(workspace.samples, indices)
    round_ids = similar(workspace.logweights, Int, 1)
    for argument in (
        samples, view(workspace.logtargets, indices),
        view(workspace.lognumerators, indices), view(workspace.logweights, indices),
        round_ids, zero(T), round, buffers.failure_scratch.record.storage,
        buffers.normal, target_argument, state.history,
        _FixedMISAssignments(round, length(indices)),
        _GeneratingGaussianDenominator(), workspace.centered_scaled,
    )
        _preflight_kernel_argument(device, kernel, argument)
    end
    if state.history isa _GaussianFactorHistory &&
       _use_factor_batch_path(device, state.history, factor_execution)
        target_kernel = _gaussian_batch_target_kernel!(backend)
        for argument in (
            view(workspace.logtargets, indices), view(workspace.lognumerators, indices),
            view(workspace.logweights, indices), round_ids, samples, target_argument,
            round, buffers.failure_scratch.record.storage,
        )
            _preflight_kernel_argument(device, target_kernel, argument)
        end
        weight_kernel = _npmc_factor_weights_kernel!(backend)
        for argument in (
            view(workspace.logweights, indices), view(workspace.logtargets, indices),
            view(workspace.centered_scaled, :, indices), state.history.lognormalizers,
            round, buffers.failure_scratch.record.storage,
        )
            _preflight_kernel_argument(device, weight_kernel, argument)
        end
    end
    return _preflight_gaussian_fit_kernels(device, state, buffers)
end
