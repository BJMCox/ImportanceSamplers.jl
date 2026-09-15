const _TRANSITION_WORKGROUP_SIZE = 32

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

@kernel function _cooperative_transition_warmup_kernel!(batch, evaluator, direction,
    normals, uniforms, first_step, steps, acceptance, decay, failures)
    chain = Int(@index(Group, Linear))
    lane = @index(Local, Linear)
    @uniform lanes = @groupsize()[1]
    @uniform dimension = size(normals, 1)
    @uniform T = eltype(normals)
    # Direction norm, adaptation coefficient, diagonal ratio, update ratio.
    parameters = @localmem T (4,)
    target_values = @localmem eltype(batch.logtargets) (2,)
    status = @localmem UInt16 (2,) # failure reason and acceptance flag
    invalid = @localmem Bool (_TRANSITION_WORKGROUP_SIZE,)
    for step in 1:steps
        isfinite(batch.logtargets[chain]) || break
        invalid[lane] = false
        if lane == 1
            status[1] = 0
            status[2] = 0
        end
        for row in lane:lanes:dimension
            increment = zero(T)
            for column in 1:row
                increment += batch.factors[row, column, chain] * normals[column, chain, step]
            end
            value = _population_sample_coordinate(batch.centres, row, chain) + increment
            invalid[lane] |= !isfinite(value)
            _store_population_location!(batch.candidates, row, chain, value)
        end
        @synchronize()
        if lane == 1
            if any(view(invalid, 1:lanes))
                status[1] = _NATIVE_GENERATED_NONFINITE
            else
                value, reason, failed = evaluator(_native_sample_at(batch.candidates, chain), chain)
                target_values[1] = value
                target_values[2] = min(zero(value), value - batch.logtargets[chain])
                status[1] = failed ? reason : UInt16(0)
                norm = zero(T)
                for row in 1:dimension
                    norm = hypot(norm, normals[row, chain, step])
                end
                parameters[1] = norm
                parameters[2] = T(first_step + step)^(-decay) *
                    (T(exp(target_values[2])) - acceptance)
                if iszero(status[1]) && !(isfinite(norm) && norm > zero(T))
                    status[1] = _NATIVE_PROPOSAL_INVALID
                end
            end
        end
        @synchronize()
        if iszero(status[1])
            for row in lane:lanes:dimension
                value = zero(T)
                for column in 1:row
                    value += batch.factors[row, column, chain] *
                        (normals[column, chain, step] / parameters[1])
                end
                direction[row, chain] = sqrt(abs(parameters[2])) * value
            end
        end
        @synchronize()
        for column in 1:dimension
            if lane == 1 && iszero(status[1])
                diagonal = batch.factors[column, column, chain]
                next, ratio = _transition_rankone_diagonal(
                    diagonal, direction[column, chain], parameters[2] < zero(T))
                if !(isfinite(next) && next > zero(T))
                    status[1] = _NATIVE_PROPOSAL_INVALID
                else
                    parameters[3] = next / diagonal
                    parameters[4] = ratio
                    batch.factors[column, column, chain] = next
                end
            end
            @synchronize()
            if iszero(status[1])
                for row in (column + lane):lanes:dimension
                    value = direction[row, chain]
                    factor, next_direction = _transition_rankone_entry(
                        batch.factors[row, column, chain], value,
                        parameters[3], parameters[4], parameters[2] < zero(T))
                    invalid[lane] |= !isfinite(factor)
                    batch.factors[row, column, chain] = factor
                    direction[row, chain] = next_direction
                end
            end
            @synchronize()
        end
        if lane == 1
            iszero(status[1]) && any(view(invalid, 1:lanes)) &&
                (status[1] = _NATIVE_PROPOSAL_INVALID)
            if iszero(status[1]) && log(uniforms[chain, step]) < target_values[2]
                status[2] = 1
                batch.logtargets[chain] = target_values[1]
                batch.accepted[chain] += 1
            end
        end
        @synchronize()
        if !iszero(status[2])
            for row in lane:lanes:dimension
                _store_population_location!(batch.centres, row, chain,
                    _population_sample_coordinate(batch.candidates, row, chain))
            end
        end
        @synchronize()
        if !iszero(status[1])
            if lane == 1
                batch.logtargets[chain] = oftype(batch.logtargets[chain], NaN)
                _record_native_failure!(failures, chain, 0, status[1])
            end
            break
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
    _launch_transition_move!(backend, state, evaluator, rng, update)
    state.steps += 1
    return nothing
end

function _launch_transition_move!(backend, state, evaluator, rng, update)
    Random.randn!(rng, state.normals)
    Random.rand!(rng, state.uniforms)
    _transition_step_kernel!(backend)(_transition_arrays(state), evaluator, update,
        state.failure_scratch.record.storage; ndrange=length(state.logtargets))
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
    remaining = state.tuning.steps - state.n_tuned
    if remaining > 0
        normals, uniforms = _transition_warmup_buffers(walk, remaining)
        capacity = size(normals, 3)
        while state.n_tuned < state.tuning.steps
            steps = min(capacity, state.tuning.steps - state.n_tuned)
            _launch_transition_warmup!(backend, state, evaluator, rng,
                normals, uniforms, state.n_tuned, steps)
            state.n_tuned += steps
            walk.steps += steps
        end
    end
    _enqueue_transition_move!(backend, walk, evaluator, rng, nothing)
    KernelAbstractions.synchronize(backend)
    _check_transition_failure!(walk, transfers)
    return nothing
end

function _transition_warmup_buffers(walk, remaining)
    # Bound scratch to 1 MiB (or one move) and sequential work to 256 moves.
    bytes_per_move = sizeof(eltype(walk.normals)) * (length(walk.normals) + length(walk.uniforms))
    capacity = min(remaining, 256, max(1, (1 << 20) ÷ bytes_per_move))
    return similar(walk.normals, size(walk.normals)..., capacity),
        similar(walk.uniforms, length(walk.uniforms), capacity)
end

function _launch_transition_warmup!(backend, state, evaluator, rng, normals, uniforms, first_step, steps)
    walk = state.walk
    Random.randn!(rng, view(normals, :, :, 1:steps))
    Random.rand!(rng, view(uniforms, :, 1:steps))
    lanes = min(_TRANSITION_WORKGROUP_SIZE, size(walk.normals, 1))
    _cooperative_transition_warmup_kernel!(backend)(
        _transition_arrays(walk), evaluator, state.direction, normals,
        uniforms, first_step, steps, state.target_acceptance, state.decay,
        walk.failure_scratch.record.storage;
        ndrange=lanes*length(walk.logtargets), workgroupsize=lanes)
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
    use_factor_batch = _use_factor_batch_mis_path(
        device,
        bank,
        denominator,
        L,
        factor_execution,
    )
    solve_scratch = use_factor_batch ? state.workspace.solve_scratch :
                    _fused_mis_solve_scratch(
        state.workspace.solve_scratch,
        backend,
    )
    kernel = _mis_round_launch_kernel!(backend)
    for argument in (views.samples, views.logweights, views.proposal_ids,
        buffers.failure_scratch.record.storage, buffers.normals, evaluator,
        bank, views.assignments, denominator, solve_scratch, nothing)
        _preflight_kernel_argument(device, kernel, argument)
    end
    if use_factor_batch
        _preflight_kernel_argument(device, _factor_batch_mis_draw_target_kernel!(backend), evaluator)
    end
    return nothing
end
