_gaussian_round_error(::AMIS) = AMISRoundError
_gaussian_method_name(::AMIS) = :amis
_gaussian_proposal_evaluations(algorithm::AMIS, n) = algorithm.rounds * n
_allocate_gaussian_adaptation_ess(::AMIS, round_ess) = nothing
_gaussian_extra_diagnostics(::AMIS, adaptation_ess, schedule) = (;)
_gaussian_summary_indices(::AMIS, state, round) =
    1:(state.offsets[round + 1] - 1)
_gaussian_adaptation_workspace!(::AMIS, device, state, round) = state.workspace
_launch_adaptive_gaussian_round!(::AMIS, args...) =
    _launch_prefilled_amis_round!(args...)

struct _AMISMixtureDenominator{L}
    logcounts::L
    round::Int
end

Adapt.@adapt_structure _AMISMixtureDenominator

@inline _mis_term_bounds(
    history,
    denominator::_AMISMixtureDenominator,
    generating_slot,
) = (1, denominator.round)

@inline function _mis_denominator_term(
    ::Type{T},
    history,
    denominator::_AMISMixtureDenominator,
    term_index,
) where {T}
    return term_index, convert(
        T,
        @inbounds(denominator.logcounts[term_index]),
    )
end

@kernel function _append_logmixture_kernel!(
    lognumerators,
    samples,
    history,
    slot,
    logcounts,
    failure_storage,
    solve_scratch,
)
    sample_index = @index(Global, Linear)
    logdensity = _mis_proposal_logdensity(
        eltype(lognumerators),
        history,
        _native_sample_at(samples, sample_index),
        slot,
        solve_scratch,
        sample_index,
    )
    reason = isnan(logdensity) ? _NATIVE_PROPOSAL_INVALID : UInt16(0)
    if iszero(reason)
        value = _append_logmixture(
            @inbounds(lognumerators[sample_index]),
            @inbounds(logcounts[slot]),
            logdensity,
        )
        reason = _native_proposal_reason(value)
        iszero(reason) && (@inbounds lognumerators[sample_index] = value)
    end
    iszero(reason) || _record_native_failure!(
        failure_storage,
        sample_index,
        0,
        reason,
    )
end

function _launch_append_logmixture!(
    lognumerators,
    samples,
    history,
    slot,
    logcounts,
    failure_storage,
    solve_scratch,
    execution,
)
    backend = KernelAbstractions.get_backend(lognumerators)
    launch_scratch = _fused_mis_solve_scratch(
        solve_scratch,
        backend,
    )
    kernel = _append_logmixture_kernel!(backend)
    kernel(
        lognumerators,
        samples,
        history,
        slot,
        logcounts,
        failure_storage,
        launch_scratch;
        ndrange=length(lognumerators),
        workgroupsize=_native_workgroupsize(
            execution,
            length(lognumerators),
        ),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function _form_amis_logweights_kernel!(
    logweights,
    logtargets,
    lognumerators,
    round_ids,
    logtotal,
    failure_storage,
)
    sample_index = @index(Global, Linear)
    # Earlier stages already recorded the failure for inactive samples.
    if !iszero(round_ids[sample_index])
        value, reason = _logweight_from_logmixture(
            @inbounds(logtargets[sample_index]),
            @inbounds(lognumerators[sample_index]),
            logtotal,
        )
        if iszero(reason)
            @inbounds logweights[sample_index] = value
        else
            _record_native_failure!(
                failure_storage,
                sample_index,
                0,
                reason,
            )
        end
    end
end

function _launch_form_amis_logweights!(
    logweights,
    logtargets,
    lognumerators,
    round_ids,
    logtotal,
    failure_storage,
    execution,
)
    backend = KernelAbstractions.get_backend(logweights)
    kernel = _form_amis_logweights_kernel!(backend)
    kernel(
        logweights,
        logtargets,
        lognumerators,
        round_ids,
        logtotal,
        failure_storage;
        ndrange=length(logweights),
        workgroupsize=_native_workgroupsize(execution, length(logweights)),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end


function _launch_prefilled_amis_factor_batch!(
    samples,
    logtargets,
    lognumerators,
    logweights,
    round_ids,
    failure_storage,
    normal_buffer,
    target,
    history,
    logcounts,
    offsets,
    round,
    solve_scratch,
    execution,
)
    first_sample = offsets[round]
    last_sample = offsets[round + 1] - 1
    old_sample_count = first_sample - 1
    new_indices = first_sample:last_sample
    new_sample_count = length(new_indices)
    dimension = _mis_dimension(history)
    backend = KernelAbstractions.get_backend(normal_buffer)
    phase = :denominator
    try
        phase = :sampling
        new_samples = _sample_view(samples, new_indices)
        normals = reshape(
            view(normal_buffer, 1:(dimension * new_sample_count)),
            dimension,
            new_sample_count,
        )
        _factor_batch_draw!(new_samples, normals, history, round)

        _launch_gaussian_target!(
            view(logtargets, new_indices),
            view(lognumerators, new_indices),
            view(logweights, new_indices),
            view(round_ids, new_indices),
            new_samples,
            target,
            round,
            failure_storage,
            execution,
        )

        new_round_ids = view(round_ids, new_indices)
        for proposal_slot in 1:(round - 1)
            _launch_factor_batch_logmixture!(
                view(lognumerators, new_indices),
                view(solve_scratch, :, new_indices),
                new_samples,
                history,
                proposal_slot,
                logcounts,
                failure_storage,
                execution,
                new_round_ids,
                new_round_ids,
                nothing,
            )
        end

        iszero(old_sample_count) || KernelAbstractions.synchronize(backend)
        phase = iszero(old_sample_count) ? :sampling : :denominator
        current_indices = 1:last_sample
        current_round_ids = view(round_ids, current_indices)
        _launch_factor_batch_logmixture!(
            view(lognumerators, current_indices),
            view(solve_scratch, :, current_indices),
            _sample_view(samples, current_indices),
            history,
            round,
            logcounts,
            failure_storage,
            execution,
            current_round_ids,
            current_round_ids,
            nothing,
        )

        KernelAbstractions.synchronize(backend)
        phase = :weight
        logtotal = log(eltype(logcounts)(last_sample))
        _launch_form_amis_logweights!(
            view(logweights, current_indices),
            view(logtargets, current_indices),
            view(lognumerators, current_indices),
            view(round_ids, current_indices),
            logtotal,
            failure_storage,
            execution,
        )
    catch cause
        _throw_gaussian_stage(phase, cause)
    end
    return nothing
end

function _launch_prefilled_amis_round!(
    samples,
    logtargets,
    lognumerators,
    logweights,
    round_ids,
    failure_storage,
    normal_buffer,
    target,
    history,
    logcounts,
    offsets,
    round,
    solve_scratch,
    execution,
    device=nothing,
    factor_execution=FusedFactorExecution(),
)
    first_sample = offsets[round]
    last_sample = offsets[round + 1] - 1
    old_sample_count = first_sample - 1
    phase = :sampling
    try
        if history isa _FactorProposalHistory &&
           _use_factor_batch_path(device, history, factor_execution)
            return _launch_prefilled_amis_factor_batch!(
                samples,
                logtargets,
                lognumerators,
                logweights,
                round_ids,
                failure_storage,
                normal_buffer,
                target,
                history,
                logcounts,
                offsets,
                round,
                solve_scratch,
                execution,
            )
        end
        if !iszero(old_sample_count)
            phase = :denominator
            old_indices = 1:old_sample_count
            _launch_append_logmixture!(
                view(lognumerators, old_indices),
                _sample_view(samples, old_indices),
                history,
                round,
                logcounts,
                failure_storage,
                solve_scratch,
                execution,
            )
        end

        phase = :sampling
        new_indices = first_sample:last_sample
        logtotal = log(eltype(logcounts)(last_sample))
        output = _GaussianRoundOutput(
            view(logtargets, new_indices),
            view(lognumerators, new_indices),
            view(logweights, new_indices),
            view(round_ids, new_indices),
            logtotal,
            round,
        )
        _launch_mis_round!(
            _sample_view(samples, new_indices),
            output,
            failure_storage,
            normal_buffer,
            target,
            history,
            _FixedMISAssignments(round, length(new_indices)),
            _AMISMixtureDenominator(logcounts, round),
            solve_scratch,
            execution,
        )

        phase = :weight
        current_indices = 1:last_sample
        _launch_form_amis_logweights!(
            view(logweights, current_indices),
            view(logtargets, current_indices),
            view(lognumerators, current_indices),
            view(round_ids, current_indices),
            logtotal,
            failure_storage,
            execution,
        )
    catch cause
        _throw_gaussian_stage(phase, cause)
    end
    return nothing
end

function _preflight_amis_kernels(
    device,
    target,
    method_state::_PreparedMomentSampler,
    buffers::_RandomBuffers,
    factor_execution,
)
    history = method_state.history
    workspace = method_state.workspace
    representative_round = findmax(method_state.schedule)[2]
    first_sample = method_state.offsets[representative_round]
    last_sample = method_state.offsets[representative_round + 1] - 1
    new_indices = first_sample:last_sample
    samples = _sample_view(workspace.samples, new_indices)
    binding_sample = _native_binding_sample(samples)
    bound_target = _bind_resolved_target(target, binding_sample)
    log_type = eltype(workspace.logweights)
    target_argument = _native_device_evaluator(bound_target, log_type)
    backend = KernelAbstractions.get_backend(buffers.normal)
    round_ids = similar(workspace.logweights, Int, 1)
    logtotal = log(eltype(method_state.logcounts)(last_sample))
    assignments = _FixedMISAssignments(
        representative_round,
        length(new_indices),
    )
    denominator = _AMISMixtureDenominator(
        method_state.logcounts,
        representative_round,
    )
    use_factor_batch = history isa _FactorProposalHistory &&
                       _use_factor_batch_path(device, history, factor_execution)
    solve_scratch = use_factor_batch ? workspace.centered_scaled :
                    _fused_mis_solve_scratch(workspace.centered_scaled, backend)

    round_kernel = _gaussian_round_launch_kernel!(backend)
    for argument in (
        samples,
        view(workspace.logtargets, new_indices),
        view(workspace.lognumerators, new_indices),
        view(workspace.logweights, new_indices),
        round_ids,
        logtotal,
        representative_round,
        buffers.failure_scratch.record.storage,
        buffers.normal,
        target_argument,
        history,
        assignments,
        denominator,
        solve_scratch,
    )
        _preflight_kernel_argument(device, round_kernel, argument)
    end

    append_kernel = _append_logmixture_kernel!(backend)
    old_count = max(first_sample - 1, 1)
    old_indices = 1:old_count
    for argument in (
        view(workspace.lognumerators, old_indices),
        _sample_view(workspace.samples, old_indices),
        history,
        representative_round,
        method_state.logcounts,
        buffers.failure_scratch.record.storage,
        solve_scratch,
    )
        _preflight_kernel_argument(device, append_kernel, argument)
    end

    weight_kernel = _form_amis_logweights_kernel!(backend)
    current_indices = 1:last_sample
    for argument in (
        view(workspace.logweights, current_indices),
        view(workspace.logtargets, current_indices),
        view(workspace.lognumerators, current_indices),
        round_ids,
        logtotal,
        buffers.failure_scratch.record.storage,
    )
        _preflight_kernel_argument(device, weight_kernel, argument)
    end

    if use_factor_batch
        target_kernel = _gaussian_batch_target_kernel!(backend)
        for argument in (
            view(workspace.logtargets, new_indices),
            view(workspace.lognumerators, new_indices),
            view(workspace.logweights, new_indices),
            round_ids,
            samples,
            target_argument,
            representative_round,
            buffers.failure_scratch.record.storage,
        )
            _preflight_kernel_argument(device, target_kernel, argument)
        end

        append_batch_kernel = _append_factor_batch_logmixture_kernel!(backend)
        for argument in (
            view(workspace.lognumerators, new_indices),
            view(workspace.centered_scaled, :, new_indices),
            history.lognormalizers,
            history.family,
            method_state.logcounts,
            representative_round,
            buffers.failure_scratch.record.storage,
        )
            _preflight_kernel_argument(device, append_batch_kernel, argument)
        end
    end

    return _preflight_gaussian_fit_kernels(device, method_state, buffers)
end
