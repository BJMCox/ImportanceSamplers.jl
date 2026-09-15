struct _CompiledGRAMISExecution{R,P,M}
    rounds::R
    phases::P
    map_result::M
end

# The gradient executable and host round list do not belong in a numerical
# trace. Borrow live arrays and fixed tuples, then reconstruct only while tracing.
_gramis_inputs(state) = (
    state.committed, state.run, state.candidate, _population_trace_plan(state.plan),
    state.repulsion_strength, state.covariance_rate, state.covariance_ess_threshold,
    state.covariance_regularization, state.tempering_tolerance, state.tempering_max_iterations,
    state.repulsion_softening, state.max_backtracking_trials, state.workspace,
)
_gramis_state(inputs) = IS._PreparedFirstOrderGRAMIS(
    inputs[1:12]..., nothing, nothing, Int[], inputs[13])

function _prepare_execution(sampler, state::IS._PreparedFirstOrderGRAMIS)
    inputs = _gramis_inputs(state)
    buffers, rng = sampler.random_buffers, sampler.rng.rng
    workspace = state.workspace
    record = buffers.failure_scratch.record
    execution = IS._KernelExecution(IS._ThreadedCPUExecution())
    device, factor_execution = sampler.device, sampler.factor_execution
    target = IS._bind_resolved_target(sampler.target, view(state.run.locations, :, 1))
    evaluator, _ = IS._native_target_evaluator(KA.get_backend(buffers.normal), target,
        eltype(workspace.round_logweights), buffers.failure_scratch.target_failures)
    output = IS._allocate_gramis_output(state)
    rounds = map(eachindex(state.plan.schedule)) do round
        draw_phase = (rng, buffers, inputs, evaluator) -> IS._launch_gramis_round!(
            rng, buffers, _gramis_state(inputs), evaluator, round, execution, device, factor_execution)
        fit_phase = inputs -> IS._fit_local_covariances!(_gramis_state(inputs), round, execution)
        covariance_phase = function (inputs, record)
            traced = _gramis_state(inputs)
            IS._launch_gramis_covariance!(nothing, traced, round, traced.workspace.factor_info, execution, record)
        end
        publish_phase = function (inputs, output)
            traced = _gramis_state(inputs)
            IS._publish_gramis_round!(traced, round, output)
            w = traced.workspace
            return _logweight_moments(IS._first_order_gramis_round_views(traced, round).logweights),
                _gramis_diagnostics(w.factor_status, w.steps, w.backtracking_trials)
        end
        draw = Reactant.@compile draw_phase(rng, buffers, inputs, evaluator)
        fit = Reactant.@compile fit_phase(inputs)
        covariance = Reactant.@compile covariance_phase(inputs, record)
        publish = Reactant.@compile publish_phase(inputs, output)
        force = round in state.active_repulsion_rounds ?
            _prepare_gramis_force(state, round, execution, device, record) : nothing
        return (; draw, fit, covariance, publish, force)
    end
    reset_phase = function (inputs, record)
        traced = _gramis_state(inputs)
        IS._copy_first_order_gramis_population!(traced.run, traced.committed)
        fill!(record.storage, zero(UInt64))
        return nothing
    end
    precondition_phase = (inputs, record) -> IS._launch_gramis_precondition!(
        nothing, _gramis_state(inputs), execution, device, record)
    backtrack_phase = (batch, record) -> IS._launch_gramis_backtracking!(nothing, batch, execution, device, record)
    add_phase = (candidate, repulsion, record, values) ->
        IS._launch_gramis_add_repulsion!(nothing, candidate, repulsion, execution, record, values)
    validation_phase = (candidate, status, record, values) ->
        IS._launch_gramis_factor_validation!(nothing, candidate, status, execution, record, values)
    copy_phase = inputs -> IS._copy_prepared_gramis_lognormalizers!(nothing, _gramis_state(inputs))
    clear_phase = workspace -> IS._clear_prepared_gramis_repulsion!(nothing, workspace)
    batch = (candidate_locations=state.candidate.locations, candidate_values=workspace.candidate_values,
        active_mask=workspace.active_mask, steps=workspace.steps, trials=workspace.backtracking_trials,
        target=target, frozen_values=workspace.frozen_values, locations=state.run.locations,
        moves=workspace.moves, max_trials=state.max_backtracking_trials)
    reset = Reactant.@compile reset_phase(inputs, record)
    precondition = Reactant.@compile precondition_phase(inputs, record)
    backtrack = Reactant.@compile backtrack_phase(batch, record)
    add = Reactant.@compile add_phase(state.candidate.locations, workspace.repulsion, record, workspace.candidate_values)
    validate = Reactant.@compile validation_phase(state.candidate, workspace.factor_status, record, workspace.candidate_values)
    copy_normalizers = Reactant.@compile copy_phase(inputs)
    clear = Reactant.@compile clear_phase(workspace)
    gradient = _prepare_gramis_gradient(inputs, target, state.serial_gradient, execution, device, record)
    repulsion = isempty(state.active_repulsion_rounds) ? nothing :
        _prepare_gramis_repulsion(state, execution, device, record)
    phases = (; reset, precondition, backtrack, add, validate, copy_normalizers, clear, gradient, repulsion)
    mapper = _prepare_result_map(sampler.target, output.samples, record)
    return _CompiledGRAMISExecution(rounds, phases, mapper)
end

function _prepare_gramis_gradient(inputs, target, bound, execution, device, record)
    phase = (inputs, target, bound, record) -> IS._launch_gramis_gradients!(
        nothing, _gramis_state(inputs), target, bound, execution, device, record)
    return (; evaluate=Reactant.@compile(phase(inputs, target, bound, record)), validate=nothing)
end

function _prepare_gramis_gradient(inputs, target, bound::IS._BoundBatchGradient, execution, device, record)
    phase = function (inputs, record)
        fill!(record.storage, zero(UInt64))
        IS._validate_gramis_gradients!(_gramis_state(inputs).workspace, record, execution)
        return nothing
    end
    return (; evaluate=nothing, validate=Reactant.@compile(phase(inputs, record)))
end

function _prepare_gramis_repulsion(state, execution, device, record)
    workspace = state.workspace
    pool_phase = (covariance, factors, family, record) -> IS._launch_gramis_pool!(
        nothing, covariance, factors, execution, family, device, record)
    distance_phase = function (means, output)
        IS._minimum_first_order_gramis_whitened_distance_kernel!(KA.get_backend(means))(
            output, means; ndrange=1, workgroupsize=1)
        return sum(view(output, 1:1))
    end
    pool = Reactant.@compile pool_phase(workspace.pooled_covariance, state.run.factors, state.run.family, record)
    factor = Reactant.@compile _factor!(workspace.pooled_covariance)
    distance = Reactant.@compile distance_phase(workspace.whitened_means, workspace.candidate_values)
    return (; pool, factor, distance)
end

function _prepare_gramis_force(state, round, execution, device, record)
    w = state.workspace
    phase = (repulsion, collisions, covariance, whitened, means, strength, softening, record, values) ->
        IS._launch_gramis_force!(nothing, repulsion, collisions, covariance, whitened,
            means, strength, round, softening, execution, device, record, values)
    return Reactant.@compile phase(w.repulsion, w.collision_counts, w.pooled_covariance,
        w.whitened_means, state.run.locations, state.repulsion_strength,
        state.repulsion_softening, record, w.candidate_values)
end

IS._execute_prepared_sampler!(sampler, ::_CompiledGRAMISExecution, threaded) =
    IS._importance_sample_cpu!(sampler, threaded)
IS._reset_prepared_gramis!(plan::_CompiledGRAMISExecution, state, record) =
    plan.phases.reset(_gramis_inputs(state), record)
IS._sample_prepared_gramis_round!(plan::_CompiledGRAMISExecution, sampler, state, target, round, execution) =
    plan.rounds[round].draw(sampler.rng.rng, sampler.random_buffers, _gramis_inputs(state), target)
IS._fit_prepared_gramis_covariances!(plan::_CompiledGRAMISExecution, state, round, execution) =
    plan.rounds[round].fit(_gramis_inputs(state))
IS._launch_gramis_precondition!(plan::_CompiledGRAMISExecution, state, execution, device, record) =
    plan.phases.precondition(_gramis_inputs(state), record)
IS._launch_gramis_backtracking!(plan::_CompiledGRAMISExecution, batch, execution, device, record) =
    plan.phases.backtrack(batch, record)
IS._launch_gramis_add_repulsion!(plan::_CompiledGRAMISExecution, candidate, repulsion, execution, record, values) =
    plan.phases.add(candidate, repulsion, record, values)
IS._launch_gramis_factor_validation!(plan::_CompiledGRAMISExecution, candidate, status, execution, record, values) =
    plan.phases.validate(candidate, status, record, values)
IS._copy_prepared_gramis_lognormalizers!(plan::_CompiledGRAMISExecution, state) =
    plan.phases.copy_normalizers(_gramis_inputs(state))
IS._clear_prepared_gramis_repulsion!(plan::_CompiledGRAMISExecution, workspace) =
    plan.phases.clear(workspace)
IS._launch_gramis_covariance!(plan::_CompiledGRAMISExecution, state, round, info, execution, record) =
    plan.rounds[round].covariance(_gramis_inputs(state), record)
IS._gramis_result_samples(plan::_CompiledGRAMISExecution, sampler, samples, transfers) =
    _map_compiled_result(plan.map_result, sampler, samples, transfers)

function IS._launch_gramis_gradients!(plan::_CompiledGRAMISExecution, state, target, bound, execution, device, record)
    phase = plan.phases.gradient
    if isnothing(phase.evaluate)
        # The AD executable owns its preparation. Call it outside the surrounding trace.
        IS._batch_value_and_gradient!(state.workspace.frozen_values, state.workspace.gradients, bound, state.run.locations)
        phase.validate(_gramis_inputs(state), record)
    else
        phase.evaluate(_gramis_inputs(state), target, bound, record)
    end
    return nothing
end

IS._launch_gramis_pool!(plan::_CompiledGRAMISExecution, covariance, factors, execution, family, device, record) =
    plan.phases.repulsion.pool(covariance, factors, family, record)
function IS._factor_gramis_pool!(plan::_CompiledGRAMISExecution, covariance)
    Bool(plan.phases.repulsion.factor(covariance)) || throw(IS._FirstOrderGRAMISRepulsionError(
        0, :pooled_factorization_failed, :nonfinite_result))
    return nothing
end
IS._launch_gramis_force!(plan::_CompiledGRAMISExecution, repulsion, collisions, covariance,
    whitened, means, strength, round, softening, execution, device, record, values) =
    plan.rounds[round].force(repulsion, collisions, covariance, whitened, means, strength, softening, record, values)
function IS._minimum_prepared_gramis_distance(plan::_CompiledGRAMISExecution, device, means, transfers, execution, output)
    distance = eltype(means)(plan.phases.repulsion.distance(means, output))
    IS._record_device_scalar_transfer!(transfers, output, eltype(output))
    return distance
end

function IS._publish_prepared_gramis!(plan::_CompiledGRAMISExecution, sampler, state, round, output, transfers, execution)
    moments, bounded = plan.rounds[round].publish(_gramis_inputs(state), output)
    weights, counts = Array(moments), Array(bounded)
    IS._record_reported_transfer!(transfers, 1, sizeof(weights), Val(:logweight_moments))
    transfers.count += 1
    transfers.bytes += sizeof(counts)
    return (weights=IS._logweight_summary(weights..., state.plan.schedule[round]),
        bounded=(all_zero=counts[1], tempering=counts[2], backtracking=counts[3], target_trials=counts[4]))
end
