@kernel function _transition_cache_kernel!(centres, logtargets, evaluator, failures)
    chain = @index(Global, Linear)
    if chain <= length(logtargets)
        value, reason, failed = evaluator(_native_sample_at(centres, chain), chain)
        logtargets[chain] = value
        if failed || !isfinite(value)
            _record_native_failure!(failures, chain, 0,
                iszero(reason) ? _NATIVE_GENERATED_NONFINITE : reason)
        end
    end
end

@kernel function _transition_step_kernel!(batch, evaluator, update, failures)
    chain = @index(Global, Linear)
    if chain <= length(batch.logtargets) && isfinite(batch.logtargets[chain])
        reason, failed = _transition_step!(batch, evaluator, update, chain)
        if failed
            # Latch per chain. Later queued moves must not use its damaged factor.
            batch.logtargets[chain] = oftype(batch.logtargets[chain], NaN)
            _record_native_failure!(failures, chain, 0, reason)
        end
    end
end

function _check_transition_failure!(state, transfers)
    snapshot = _device_failure_snapshot(state.failure_scratch.record)
    _record_reported_transfer!(transfers, snapshot.transfers.count,
        snapshot.transfers.bytes, Val(:failure_snapshot))
    failure = snapshot.failure
    if !iszero(failure.count)
        cause = DomainError(failure.reason_bits,
            "non-finite transition target/candidate or invalid RAM direction/factor")
        throw(SamplerExecutionError(:transition, failure.first_logical_index,
            CapturedException(cause, backtrace())))
    end
    return nothing
end

function _initialize_transition_batch!(backend, state, target, transfers)
    count = length(state.logtargets)
    L = eltype(state.logtargets)
    evaluator = _NativeDeviceTarget{L,typeof(target)}(target)
    record = state.failure_scratch.record.storage
    _reset_native_failure_scratch!(state.failure_scratch)
    if !state.cache_valid
        _transition_cache_kernel!(backend)(state.centres, state.logtargets, evaluator, record;
            ndrange=count)
        KernelAbstractions.synchronize(backend)
        _check_transition_failure!(state, transfers)
        state.cache_valid = true
        state.initial_evaluations += count
    end
    return evaluator
end

function _enqueue_transition_move!(backend, state, evaluator, rng, update)
    Random.randn!(rng, state.normals)
    Random.rand!(rng, state.uniforms)
    _transition_step_kernel!(backend)(_transition_arrays(state), evaluator, update,
        state.failure_scratch.record.storage; ndrange=length(state.logtargets))
    state.steps += 1
    return nothing
end

function _transition_batch!(backend, state, target, rng, execution, transfers, update)
    evaluator = _initialize_transition_batch!(backend, state, target, transfers)
    _enqueue_transition_move!(backend, state, evaluator, rng, update)
    KernelAbstractions.synchronize(backend)
    _check_transition_failure!(state, transfers)
    return nothing
end

function _warmup_transition!(backend, state, target, rng, execution, transfers)
    walk = state.walk
    evaluator = _initialize_transition_batch!(backend, walk, target, transfers)
    # Same-stream ordering protects reused random buffers and factors. Failed
    # chains latch NaN; the first round checks the whole sequence before drawing.
    while state.n_tuned < state.tuning.steps
        _enqueue_transition_move!(backend, walk, evaluator, rng, _transition_adaptation(state))
        state.n_tuned += 1
    end
    _enqueue_transition_move!(backend, walk, evaluator, rng, nothing)
    KernelAbstractions.synchronize(backend)
    _check_transition_failure!(walk, transfers)
    return nothing
end

_preflight_transition(device, state, target) =
    throw(SamplerDeviceError(device, :transition_unsupported))
_preflight_transition(device, state::_RAMState, target) =
    _preflight_transition(device, state.walk, target, _transition_adaptation(state))

function _preflight_transition(device, state::_RandomWalkState, target, update=nothing)
    backend = KernelAbstractions.get_backend(state.normals)
    L = eltype(state.logtargets)
    evaluator = _NativeDeviceTarget{L,typeof(target)}(target)
    kernel = _transition_step_kernel!(backend)
    for argument in (_transition_arrays(state), evaluator, update, state.failure_scratch.record.storage)
        _preflight_kernel_argument(device, kernel, argument)
    end
    return nothing
end

function _preflight_accelerator_method(device, target, ::LAIS, state::_PreparedLAIS,
    buffers::_PopulationNormalBuffers, factor_execution)
    bank = state.bank
    bound = _bind_resolved_target(target, _population_binding_sample(bank))
    _preflight_transition(device, state.run_transition, bound)
    L = eltype(state.workspace.round_logweights)
    evaluator = _NativeDeviceTarget{L,typeof(bound)}(bound)
    backend = KernelAbstractions.get_backend(buffers.normals)
    round = findmax(state.plan.schedule)[2]
    views = _population_round_views(state, round)
    denominator = _population_denominator(state, round)
    kernel = _mis_round_launch_kernel!(backend)
    for argument in (views.samples, views.logweights, views.proposal_ids,
        buffers.failure_scratch.record.storage, buffers.normals, evaluator,
        bank, views.assignments, denominator, state.workspace.solve_scratch, nothing)
        _preflight_kernel_argument(device, kernel, argument)
    end
    if _use_factor_batch_mis_path(device, bank, denominator, L, factor_execution)
        _preflight_kernel_argument(device, _factor_batch_mis_draw_target_kernel!(backend), evaluator)
    end
    return nothing
end
