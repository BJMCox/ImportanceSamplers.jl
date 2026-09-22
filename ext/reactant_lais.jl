_prepare_execution(sampler, state::IS._PreparedLAIS{B,P,W,<:Union{
    IS._RandomWalkState,IS._RAMState,IS._SampleMetropolisHastingsState}}) where {B,P,W} =
    _prepare_population_execution(sampler, state)

_population_trace_state(state::IS._PreparedLAIS) = IS._PreparedLAIS(state.bank,
    state.run_bank, _population_trace_plan(state.plan), state.workspace,
    state.transition, state.run_transition)
_reset_population_arrays!(state::IS._PreparedLAIS) =
    IS._copy_transition_arrays!(state.run_transition, state.transition)

function _population_draw_state(state::IS._PreparedLAIS)
    # Scalar banks alias vector centres through reshape wrappers. Materialize
    # only the traced view before kernel conversion, not a copy of the centres.
    bank_view = bank -> IS._population_with_locations(bank,
        Reactant.TracedUtils.materialize_traced_array(bank.locations))
    return IS._PreparedLAIS(bank_view(state.bank), bank_view(state.run_bank),
        state.plan, state.workspace, state.transition, state.run_transition)
end

function IS._reset_prepared_population!(plan::_CompiledPopulationExecution, state::IS._PreparedLAIS, record)
    plan.reset(_population_trace_state(state), record)
    IS._copy_transition_counters!(state.run_transition, state.transition)
    return nothing
end

_transition_walk(state::IS._RandomWalkState) = state
_transition_walk(state::IS._RAMState) = state.walk
_transition_accepted(state) = state.accepted
_transition_accepted(state::IS._RAMState) = state.walk.accepted

# Plain Julia scalars are tracing constants. RAM's diminishing gain must remain
# a live input on the same device as its transition arrays.
_transition_number(array, value) = Reactant.to_rarray(value; track_numbers=true,
    client=Reactant.XLA.client(array), device=Reactant.XLA.device(array))
_compiled_transition_update(state) = nothing
function _compiled_transition_update(state::IS._RAMState{W,IS.ContinuousTuning}) where {W}
    update = IS._transition_adaptation(state)
    return merge(update, (; gain=_transition_number(state.walk.normals, update.gain)))
end

function _prepare_population_adaptation(sampler, state::IS._PreparedLAIS)
    target = IS._bind_resolved_target(sampler.target, IS._population_binding_sample(state.bank))
    transition = _prepare_compiled_transition(state.run_transition, target, sampler.rng.rng)
    summaries = map(eachindex(state.plan.schedule)) do round
        phase = state -> _logweight_moments(IS._population_round_views(state, round).logweights)
        Reactant.@compile phase(state)
    end
    accepted = _transition_accepted(state.run_transition)
    counts = Reactant.@compile sum(accepted)
    return (; transition, summaries, counts)
end

function IS._before_prepared_population_round!(plan::_CompiledPopulationExecution, sampler,
    state::IS._PreparedLAIS, target, round, execution, transfers)
    _run_compiled_transition!(plan.adaptation.transition, state.run_transition,
        target, sampler.rng.rng, transfers)
    return nothing
end

function IS._advance_prepared_population!(plan::_CompiledPopulationExecution, sampler,
    state::IS._PreparedLAIS, views, round, execution, transfers)
    values = Array(plan.adaptation.summaries[round](_population_trace_state(state)))
    IS._record_reported_transfer!(transfers, 1, sizeof(values), Val(:logweight_moments))
    return IS._logweight_summary(values..., views.round_size)
end

function IS._population_transition_diagnostics(plan::_CompiledPopulationExecution, state, transfers)
    accepted = _transition_accepted(state)
    count = Int(plan.adaptation.counts(accepted))
    IS._record_device_scalar_transfer!(transfers, accepted, eltype(accepted))
    return _compiled_transition_diagnostics(state, count)
end
_compiled_transition_diagnostics(state::IS._RandomWalkState, count) = IS._walk_diagnostics(state, count)
_compiled_transition_diagnostics(state::IS._RAMState, count) =
    IS._ram_diagnostics(state, IS._walk_diagnostics(state.walk, count))
_compiled_transition_diagnostics(state::IS._SampleMetropolisHastingsState, count) =
    IS._smh_diagnostics(state, count)

function _prepare_compiled_transition(state::Union{IS._RandomWalkState,IS._RAMState}, target, rng)
    walk = _transition_walk(state)
    L = eltype(walk.logtargets)
    clear_phase = walk -> IS._reset_native_failure_scratch!(walk.failure_scratch)
    cache_phase = function (walk, target)
        clear_phase(walk)
        evaluator, _ = IS._native_target_evaluator(KA.get_backend(walk.normals),
            target, L, walk.failure_scratch.target_failures)
        if evaluator isa IS._NativeBatchTarget
            IS._invoke_batch_target!(target, walk.logtargets, walk.centres,
                walk.failure_scratch.record.storage, IS._ThreadedCPUExecution())
            evaluator = IS._CachedTargetValues(walk.logtargets)
        end
        IS._transition_cache_kernel!(KA.get_backend(walk.normals))(
            walk.centres, walk.logtargets, evaluator, walk.failure_scratch.record.storage;
            ndrange=length(walk.logtargets))
        return nothing
    end
    move_phase = function (walk, target, rng, update)
        evaluator, _ = IS._native_target_evaluator(KA.get_backend(walk.normals),
            target, L, walk.failure_scratch.target_failures)
        IS._launch_transition_move!(KA.get_backend(walk.normals), walk, evaluator, rng, update)
        return nothing
    end
    update = _compiled_transition_update(state)
    clear = Reactant.@compile clear_phase(walk)
    cache = Reactant.@compile cache_phase(walk, target)
    move = Reactant.@compile move_phase(walk, target, rng, update)
    warmup = _prepare_compiled_warmup(state, target, rng)
    return (; clear, cache, move, warmup)
end

_prepare_compiled_warmup(state, target, rng) = nothing
function _prepare_compiled_warmup(state::IS._RAMState{W,<:IS.WarmupTuning}, target, rng) where {W}
    remaining = state.tuning.steps - state.n_tuned
    remaining > 0 || return nothing
    normals, uniforms = IS._transition_warmup_buffers(state.walk, remaining)
    capacity = size(normals, 3)
    first_step = _transition_number(normals, state.n_tuned)
    L = eltype(state.walk.logtargets)
    compile_chunk = function (steps)
        phase = function (state, target, rng, normals, uniforms, first_step)
            walk = state.walk
            evaluator, _ = IS._native_target_evaluator(KA.get_backend(walk.normals),
                target, L, walk.failure_scratch.target_failures)
            IS._launch_transition_warmup!(KA.get_backend(walk.normals), state,
                evaluator, rng, normals, uniforms, first_step, steps)
            return nothing
        end
        Reactant.@compile phase(state, target, rng, normals, uniforms, first_step)
    end
    full = compile_chunk(capacity)
    tail_size = remaining % capacity
    tail = iszero(tail_size) ? nothing : compile_chunk(tail_size)
    return (; normals, uniforms, capacity, full, tail)
end

_run_compiled_warmup!(::Nothing, state, target, rng, transfers) = nothing
function _run_compiled_warmup!(plan, state, target, rng, transfers)
    while state.n_tuned < state.tuning.steps
        steps = min(plan.capacity, state.tuning.steps - state.n_tuned)
        first_step = _transition_number(plan.normals, state.n_tuned)
        IS._record_scalar_transfer!(transfers, Int)
        phase = steps == plan.capacity ? plan.full : plan.tail
        phase(state, target, rng, plan.normals, plan.uniforms, first_step)
        state.n_tuned += steps
        state.walk.steps += steps
    end
    return nothing
end

_finish_compiled_transition!(state) = nothing
_finish_compiled_transition!(state::IS._RAMState{W,IS.ContinuousTuning}) where {W} = (state.n_tuned += 1)

function _run_compiled_transition!(plan, state::Union{IS._RandomWalkState,IS._RAMState}, target, rng, transfers)
    walk = _transition_walk(state)
    if walk.cache_valid
        plan.clear(walk)
    else
        plan.cache(walk, target)
        IS._check_transition_failure!(walk, transfers)
        walk.cache_valid = true
        walk.initial_evaluations += length(walk.logtargets)
    end
    _run_compiled_warmup!(plan.warmup, state, target, rng, transfers)
    update = _compiled_transition_update(state)
    isnothing(update) || IS._record_scalar_transfer!(transfers, eltype(walk.normals))
    plan.move(walk, target, rng, update)
    walk.steps += 1
    KA.synchronize(KA.get_backend(walk.normals))
    IS._check_transition_failure!(walk, transfers)
    _finish_compiled_transition!(state)
    return nothing
end

function _prepare_compiled_transition(state::IS._SampleMetropolisHastingsState, target, rng)
    execution = IS._ThreadedCPUExecution()
    L = eltype(state.logratios)
    cache_phase = function (state, target)
        evaluator, _ = IS._native_target_evaluator(KA.get_backend(state.normals),
            target, L, state.failure_scratch.target_failures)
        IS._launch_smh_cache!(state, evaluator, execution)
        return nothing
    end
    cache = Reactant.@compile cache_phase(state, target)
    compile_chunk = function (steps)
        phase = function (state, target, rng)
            IS._smh_candidate_batch!(state, target, rng, execution, steps, L)
            IS._launch_smh_ordered!(KA.get_backend(state.normals), state, steps)
            return nothing
        end
        Reactant.@compile phase(state, target, rng)
    end
    full = compile_chunk(state.capacity)
    tail_size = state.moves % state.capacity
    tail = iszero(tail_size) ? nothing : compile_chunk(tail_size)
    return (; cache, full, tail)
end

function _run_compiled_transition!(plan, state::IS._SampleMetropolisHastingsState, target, rng, transfers)
    if !state.cache_valid
        plan.cache(state, target)
        IS._check_smh_failures!(state, transfers)
        state.cache_valid = true
        state.initial_evaluations += length(state.logratios)
    end
    completed = 0
    while completed < state.moves
        steps = min(state.capacity, state.moves - completed)
        phase = steps == state.capacity ? plan.full : plan.tail
        phase(state, target, rng)
        KA.synchronize(KA.get_backend(state.normals))
        IS._check_smh_failures!(state, transfers)
        state.production_evaluations += steps
        completed += steps
    end
    return nothing
end
