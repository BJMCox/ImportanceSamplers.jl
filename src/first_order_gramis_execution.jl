@kernel function _first_order_gramis_local_weights_kernel!(
    local_logweights,
    generating_logdensities,
    proposal_ids,
    round_ids,
    round,
    failure_storage,
)
    sample_index = @index(Global, Linear)
    @inbounds round_ids[sample_index] = 0
    if !iszero(@inbounds(proposal_ids[sample_index]))
        local_logweight, reason = _subtract_logweight(
            @inbounds(local_logweights[sample_index]),
            @inbounds(generating_logdensities[sample_index]),
        )
        if iszero(reason)
            @inbounds local_logweights[sample_index] = local_logweight
            @inbounds round_ids[sample_index] = round
        else
            _record_native_failure!(failure_storage, sample_index, 0, reason)
        end
    end
end

function _first_order_gramis_sample_round!(
    samples,
    returned_logweights,
    local_logweights,
    generating_logdensities,
    proposal_ids,
    round_ids,
    failure_storage,
    normal_buffer,
    target,
    bank,
    assignments,
    denominator,
    solve_scratch,
    round,
    execution,
    device,
    factor_execution,
)
    adaptation = _MISAdaptationOutput(
        local_logweights,
        generating_logdensities,
    )
    output = _MISRoundOutput(
        returned_logweights,
        proposal_ids,
        adaptation,
    )
    launch = _use_factor_batch_mis_path(
        device,
        bank,
        denominator,
        eltype(returned_logweights),
        factor_execution,
    ) ? _launch_factor_batch_mis_round! : _launch_mis_round!
    launch(
        samples,
        output,
        failure_storage,
        normal_buffer,
        target,
        bank,
        assignments,
        denominator,
        solve_scratch,
        execution,
    )

    backend = KernelAbstractions.get_backend(local_logweights)
    kernel = _first_order_gramis_local_weights_kernel!(backend)
    kernel(
        local_logweights,
        generating_logdensities,
        proposal_ids,
        round_ids,
        round,
        failure_storage;
        ndrange=length(local_logweights),
        workgroupsize=_native_workgroupsize(execution, length(local_logweights)),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

struct _FirstOrderGRAMISDerivativeError{T} <: Exception
    proposal_slot::Int
    reason::Symbol
    value::T
end

function Base.showerror(io::IO, error::_FirstOrderGRAMISDerivativeError)
    print(
        io,
        "FirstOrderGRAMIS derivative failure for proposal ",
        error.proposal_slot,
        ": ",
        error.reason,
        " (",
        error.value,
        ')',
    )
end

@noinline function _throw_first_order_gramis_derivative_error(
    proposal_slot,
    reason,
    value,
)
    throw(_FirstOrderGRAMISDerivativeError(proposal_slot, reason, value))
end

@inline function _pooled_covariance_entry!(
    pooled_covariance,
    factors,
    entry,
)
    dimension = size(pooled_covariance, 1)
    row = (entry - 1) % dimension + 1
    column = (entry - 1) ÷ dimension + 1
    proposal_count = size(factors, 3)
    T = eltype(pooled_covariance)
    covariance = zero(T)
    for proposal_slot in axes(factors, 3)
        covariance += _gramis_current_covariance(
            factors,
            row,
            column,
            proposal_slot,
        )
    end
    @inbounds pooled_covariance[row, column] = covariance / T(proposal_count)
    return nothing
end

function _pooled_covariance!(
    pooled_covariance,
    factors,
    ::_SerialCPUExecution,
)
    @inbounds for entry in 1:length(pooled_covariance)
        _pooled_covariance_entry!(pooled_covariance, factors, entry)
    end
    return nothing
end

function _pooled_covariance!(
    pooled_covariance,
    factors,
    ::_ThreadedCPUExecution,
)
    Threads.@threads :dynamic for entry in 1:length(pooled_covariance)
        _pooled_covariance_entry!(pooled_covariance, factors, entry)
    end
    return nothing
end

function _whiten_means!(whitened_means, pooled_factor, means)
    copyto!(whitened_means, means)
    LinearAlgebra.ldiv!(
        LinearAlgebra.LowerTriangular(pooled_factor),
        whitened_means,
    )
    return nothing
end

const _GRAMIS_COLLISIONS_UNAVAILABLE = -1

struct _FirstOrderGRAMISRepulsionError{V} <: Exception
    proposal_slot::Int
    reason::Symbol
    value::V
end

function Base.showerror(io::IO, error::_FirstOrderGRAMISRepulsionError)
    print(
        io,
        "FirstOrderGRAMIS repulsion failure for proposal ",
        error.proposal_slot,
        ": ",
        error.reason,
        " (",
        error.value,
        ')',
    )
end

@noinline function _throw_first_order_gramis_repulsion_error(
    proposal_slot,
    reason,
    value,
)
    throw(_FirstOrderGRAMISRepulsionError(proposal_slot, reason, value))
end

function _factor_pooled_covariance!(pooled_covariance)
    @inbounds for entry in eachindex(pooled_covariance)
        value = pooled_covariance[entry]
        isfinite(value) || _throw_first_order_gramis_repulsion_error(
            0,
            :pooled_covariance_nonfinite,
            value,
        )
    end
    _, info = LinearAlgebra.LAPACK.potrf!('L', pooled_covariance)
    iszero(info) || _throw_first_order_gramis_repulsion_error(
        0,
        :pooled_factorization_failed,
        info,
    )
    return nothing
end

@inline function _repulsion_slot!(
    repulsion,
    collision_counts,
    means,
    whitened_means,
    strength,
    softening,
    proposal_slot,
)
    T = eltype(repulsion)
    dimension, proposal_count = size(means)
    @inbounds for row in axes(repulsion, 1)
        repulsion[row, proposal_slot] = zero(T)
    end
    collision_count = 0
    for peer_slot in axes(means, 2)
        peer_slot == proposal_slot && continue
        exact_collision = true
        softened_norm = abs(softening)
        for row in axes(means, 1)
            whitened_difference = @inbounds(
                whitened_means[row, proposal_slot] -
                whitened_means[row, peer_slot]
            )
            exact_collision &= iszero(whitened_difference)
            softened_norm = hypot(softened_norm, whitened_difference)
        end
        if exact_collision
            for row in axes(means, 1)
                exact_collision &= @inbounds(
                    means[row, proposal_slot] == means[row, peer_slot]
                )
            end
            if exact_collision
                collision_count += 1
                continue
            end
        end
        denominator = softened_norm ^ dimension
        for row in axes(means, 1)
            @inbounds repulsion[row, proposal_slot] +=
                (means[row, proposal_slot] - means[row, peer_slot]) /
                denominator
        end
    end
    scale = strength / T(proposal_count - 1)
    @inbounds for row in axes(repulsion, 1)
        repulsion[row, proposal_slot] *= scale
    end
    @inbounds collision_counts[proposal_slot] = collision_count
    return nothing
end

function _repulsion_force!(
    repulsion,
    collision_counts,
    means,
    whitened_means,
    strength,
    softening,
    ::_SerialCPUExecution,
)
    @inbounds for proposal_slot in axes(means, 2)
        _repulsion_slot!(
            repulsion,
            collision_counts,
            means,
            whitened_means,
            strength,
            softening,
            proposal_slot,
        )
    end
    return nothing
end

function _repulsion_force!(
    repulsion,
    collision_counts,
    means,
    whitened_means,
    strength,
    softening,
    ::_ThreadedCPUExecution,
)
    Threads.@threads :dynamic for proposal_slot in axes(means, 2)
        _repulsion_slot!(
            repulsion,
            collision_counts,
            means,
            whitened_means,
            strength,
            softening,
            proposal_slot,
        )
    end
    return nothing
end

function _validate_repulsion!(repulsion)
    @inbounds for proposal_slot in axes(repulsion, 2), row in axes(repulsion, 1)
        force = repulsion[row, proposal_slot]
        isfinite(force) || _throw_first_order_gramis_repulsion_error(
            proposal_slot,
            :force_nonfinite,
            force,
        )
    end
    return nothing
end

function _repulsion!(
    repulsion,
    collision_counts,
    pooled_covariance,
    whitened_means,
    means,
    factors,
    strength::T,
    softening::T,
    execution::Union{_SerialCPUExecution,_ThreadedCPUExecution},
) where {T}
    if iszero(strength)
        fill!(repulsion, zero(eltype(repulsion)))
        fill!(collision_counts, _GRAMIS_COLLISIONS_UNAVAILABLE)
        return nothing
    end

    _pooled_covariance!(pooled_covariance, factors, execution)
    _factor_pooled_covariance!(pooled_covariance)
    _whiten_means!(whitened_means, pooled_covariance, means)
    _repulsion_force!(
        repulsion,
        collision_counts,
        means,
        whitened_means,
        strength,
        softening,
        execution,
    )
    _validate_repulsion!(repulsion)
    return nothing
end

function _validate_frozen_derivatives!(values, gradients)
    @inbounds for proposal_slot in eachindex(values)
        value = values[proposal_slot]
        isfinite(value) || _throw_first_order_gramis_derivative_error(
            proposal_slot,
            :frozen_value_nonfinite,
            value,
        )
        for row in axes(gradients, 1)
            gradient = gradients[row, proposal_slot]
            isfinite(gradient) || _throw_first_order_gramis_derivative_error(
                proposal_slot,
                :gradient_nonfinite,
                gradient,
            )
        end
    end
    return nothing
end

function _validate_preconditioned_moves!(moves)
    @inbounds for proposal_slot in axes(moves, 2), row in axes(moves, 1)
        move = moves[row, proposal_slot]
        isfinite(move) || _throw_first_order_gramis_derivative_error(
            proposal_slot,
            :move_nonfinite,
            move,
        )
    end
    return nothing
end

function _validate_backtracking_candidates!(candidate_values, trials, trial)
    @inbounds for proposal_slot in eachindex(candidate_values, trials)
        trials[proposal_slot] == trial || continue
        value = candidate_values[proposal_slot]
        (isfinite(value) || value == -Inf) ||
            _throw_first_order_gramis_derivative_error(
                proposal_slot,
                :candidate_value_nonfinite,
                value,
            )
    end
    return nothing
end

@inline function _precondition_gradient_slot!(
    moves,
    gradients,
    factors,
    proposal_slot,
)
    dimension = size(gradients, 1)
    T = eltype(gradients)
    for factor_column in 1:dimension
        projected_gradient = zero(T)
        for gradient_row in factor_column:dimension
            projected_gradient += @inbounds(
                factors[gradient_row, factor_column, proposal_slot] *
                gradients[gradient_row, proposal_slot]
            )
        end
        @inbounds moves[factor_column, proposal_slot] = projected_gradient
    end
    for row in dimension:-1:1
        move = zero(T)
        for factor_column in 1:row
            move += @inbounds(
                factors[row, factor_column, proposal_slot] *
                moves[factor_column, proposal_slot]
            )
        end
        @inbounds moves[row, proposal_slot] = move
    end
    return nothing
end

function _precondition_gradients!(
    moves,
    gradients,
    factors,
    ::_SerialCPUExecution,
)
    @inbounds for proposal_slot in axes(gradients, 2)
        _precondition_gradient_slot!(moves, gradients, factors, proposal_slot)
    end
    _validate_preconditioned_moves!(moves)
    return nothing
end

function _precondition_gradients!(
    moves,
    gradients,
    factors,
    ::_ThreadedCPUExecution,
)
    Threads.@threads :dynamic for proposal_slot in axes(gradients, 2)
        _precondition_gradient_slot!(moves, gradients, factors, proposal_slot)
    end
    _validate_preconditioned_moves!(moves)
    return nothing
end

@inline function _evaluate_frozen_gradient_slot!(
    values,
    gradients,
    target,
    bound_gradient,
    locations,
    proposal_slot,
)
    location = view(locations, :, proposal_slot)
    gradient = view(gradients, :, proposal_slot)
    @inbounds values[proposal_slot] = target(location)
    _gradient!(gradient, bound_gradient, location)
    return nothing
end

function _evaluate_frozen_gradients!(
    values,
    gradients,
    target,
    bound_gradient,
    locations,
    ::_SerialCPUExecution,
)
    @inbounds for proposal_slot in axes(locations, 2)
        _evaluate_frozen_gradient_slot!(
            values,
            gradients,
            target,
            bound_gradient,
            locations,
            proposal_slot,
        )
    end
    _validate_frozen_derivatives!(values, gradients)
    return nothing
end

_first_order_gramis_bound_gradient(
    method_state::_PreparedFirstOrderGRAMIS,
    ::_SerialCPUExecution,
) = method_state.serial_gradient

_first_order_gramis_bound_gradient(
    method_state::_PreparedFirstOrderGRAMIS,
    ::_ThreadedCPUExecution,
) = method_state.threaded_gradient

function _evaluate_frozen_gradients!(
    method_state::_PreparedFirstOrderGRAMIS,
    target,
    execution::Union{_SerialCPUExecution,_ThreadedCPUExecution},
)
    workspace = method_state.workspace
    return _evaluate_frozen_gradients!(
        workspace.frozen_values,
        workspace.gradients,
        target,
        _first_order_gramis_bound_gradient(method_state, execution),
        method_state.run.locations,
        execution,
    )
end

function _precondition_gradients!(
    method_state::_PreparedFirstOrderGRAMIS,
    execution::Union{_SerialCPUExecution,_ThreadedCPUExecution},
)
    workspace = method_state.workspace
    return _precondition_gradients!(
        workspace.moves,
        workspace.gradients,
        method_state.run.factors,
        execution,
    )
end

function _evaluate_frozen_gradients!(
    values,
    gradients,
    target,
    bound_gradient,
    locations,
    ::_ThreadedCPUExecution,
)
    Threads.@threads :dynamic for proposal_slot in axes(locations, 2)
        _evaluate_frozen_gradient_slot!(
            values,
            gradients,
            target,
            bound_gradient,
            locations,
            proposal_slot,
        )
    end
    _validate_frozen_derivatives!(values, gradients)
    return nothing
end

@inline function _initialize_backtracking_slot!(
    candidate_locations,
    candidate_values,
    active_mask,
    steps,
    trials,
    frozen_values,
    locations,
    proposal_slot,
)
    copyto!(
        view(candidate_locations, :, proposal_slot),
        view(locations, :, proposal_slot),
    )
    @inbounds candidate_values[proposal_slot] = frozen_values[proposal_slot]
    @inbounds active_mask[proposal_slot] = true
    @inbounds steps[proposal_slot] = zero(eltype(steps))
    @inbounds trials[proposal_slot] = 0
    return nothing
end

@inline function _backtracking_trial_slot!(
    candidate_locations,
    candidate_values,
    active_mask,
    steps,
    trials,
    target,
    frozen_values,
    locations,
    moves,
    step,
    trial,
    proposal_slot,
)
    @inbounds active_mask[proposal_slot] || return nothing
    @inbounds for row in axes(locations, 1)
        candidate_locations[row, proposal_slot] =
            locations[row, proposal_slot] + step * moves[row, proposal_slot]
    end
    candidate = view(candidate_locations, :, proposal_slot)
    candidate_value = target(candidate)
    @inbounds candidate_values[proposal_slot] = candidate_value
    @inbounds trials[proposal_slot] = trial
    if isfinite(candidate_value) &&
       candidate_value >= @inbounds(frozen_values[proposal_slot])
        @inbounds steps[proposal_slot] = step
        @inbounds active_mask[proposal_slot] = false
    end
    return nothing
end

@inline function _finish_backtracking_slot!(
    candidate_locations,
    candidate_values,
    active_mask,
    frozen_values,
    locations,
    proposal_slot,
)
    @inbounds active_mask[proposal_slot] || return nothing
    copyto!(
        view(candidate_locations, :, proposal_slot),
        view(locations, :, proposal_slot),
    )
    @inbounds candidate_values[proposal_slot] = frozen_values[proposal_slot]
    @inbounds active_mask[proposal_slot] = false
    return nothing
end

function _initialize_backtracking!(
    candidate_locations,
    candidate_values,
    active_mask,
    steps,
    trials,
    frozen_values,
    locations,
    ::_SerialCPUExecution,
)
    @inbounds for proposal_slot in axes(locations, 2)
        _initialize_backtracking_slot!(
            candidate_locations,
            candidate_values,
            active_mask,
            steps,
            trials,
            frozen_values,
            locations,
            proposal_slot,
        )
    end
    return nothing
end

function _initialize_backtracking!(
    candidate_locations,
    candidate_values,
    active_mask,
    steps,
    trials,
    frozen_values,
    locations,
    ::_ThreadedCPUExecution,
)
    Threads.@threads :dynamic for proposal_slot in axes(locations, 2)
        _initialize_backtracking_slot!(
            candidate_locations,
            candidate_values,
            active_mask,
            steps,
            trials,
            frozen_values,
            locations,
            proposal_slot,
        )
    end
    return nothing
end

function _backtracking_trial!(
    candidate_locations,
    candidate_values,
    active_mask,
    steps,
    trials,
    target,
    frozen_values,
    locations,
    moves,
    step,
    trial,
    ::_SerialCPUExecution,
)
    @inbounds for proposal_slot in axes(locations, 2)
        _backtracking_trial_slot!(
            candidate_locations,
            candidate_values,
            active_mask,
            steps,
            trials,
            target,
            frozen_values,
            locations,
            moves,
            step,
            trial,
            proposal_slot,
        )
    end
    return nothing
end

function _backtracking_trial!(
    candidate_locations,
    candidate_values,
    active_mask,
    steps,
    trials,
    target,
    frozen_values,
    locations,
    moves,
    step,
    trial,
    ::_ThreadedCPUExecution,
)
    Threads.@threads :dynamic for proposal_slot in axes(locations, 2)
        _backtracking_trial_slot!(
            candidate_locations,
            candidate_values,
            active_mask,
            steps,
            trials,
            target,
            frozen_values,
            locations,
            moves,
            step,
            trial,
            proposal_slot,
        )
    end
    return nothing
end

function _finish_backtracking!(
    candidate_locations,
    candidate_values,
    active_mask,
    frozen_values,
    locations,
    ::_SerialCPUExecution,
)
    @inbounds for proposal_slot in axes(locations, 2)
        _finish_backtracking_slot!(
            candidate_locations,
            candidate_values,
            active_mask,
            frozen_values,
            locations,
            proposal_slot,
        )
    end
    return nothing
end

function _finish_backtracking!(
    candidate_locations,
    candidate_values,
    active_mask,
    frozen_values,
    locations,
    ::_ThreadedCPUExecution,
)
    Threads.@threads :dynamic for proposal_slot in axes(locations, 2)
        _finish_backtracking_slot!(
            candidate_locations,
            candidate_values,
            active_mask,
            frozen_values,
            locations,
            proposal_slot,
        )
    end
    return nothing
end

function _backtrack_means!(
    candidate_locations,
    candidate_values,
    active_mask,
    steps,
    trials,
    target,
    frozen_values,
    locations,
    moves,
    max_trials,
    execution::Union{_SerialCPUExecution,_ThreadedCPUExecution},
)
    _validate_preconditioned_moves!(moves)
    _initialize_backtracking!(
        candidate_locations,
        candidate_values,
        active_mask,
        steps,
        trials,
        frozen_values,
        locations,
        execution,
    )
    for trial in 1:max_trials
        step = ldexp(one(eltype(steps)), 1 - trial)
        _backtracking_trial!(
            candidate_locations,
            candidate_values,
            active_mask,
            steps,
            trials,
            target,
            frozen_values,
            locations,
            moves,
            step,
            trial,
            execution,
        )
        _validate_backtracking_candidates!(candidate_values, trials, trial)
    end
    _finish_backtracking!(
        candidate_locations,
        candidate_values,
        active_mask,
        frozen_values,
        locations,
        execution,
    )
    return nothing
end

function _backtrack_means!(
    method_state::_PreparedFirstOrderGRAMIS,
    target,
    execution::Union{_SerialCPUExecution,_ThreadedCPUExecution},
)
    workspace = method_state.workspace
    return _backtrack_means!(
        method_state.candidate.locations,
        workspace.candidate_values,
        workspace.active_mask,
        workspace.steps,
        workspace.backtracking_trials,
        target,
        workspace.frozen_values,
        method_state.run.locations,
        workspace.moves,
        method_state.max_backtracking_trials,
        execution,
    )
end

const _GRAMIS_COVARIANCE_READY = UInt8(0)
const _GRAMIS_ALL_ZERO_LOCAL = UInt8(1)
const _GRAMIS_TEMPERING_FALLBACK = UInt8(2)

@inline function _scale_aware_ridge(
    previous_trace,
    dimension,
    regularization::T,
) where {T}
    return regularization * previous_trace / T(dimension)
end

@kernel function _local_group_starts_kernel!(starts, counts, round)
    first_sample = 1
    for proposal_slot in eachindex(starts)
        @inbounds starts[proposal_slot] = first_sample
        first_sample += @inbounds counts[proposal_slot, round]
    end
end

function _local_group_starts!(starts, counts, round, execution)
    backend = KernelAbstractions.get_backend(starts)
    kernel = _local_group_starts_kernel!(backend)
    kernel(
        starts,
        counts,
        round;
        ndrange=1,
        workgroupsize=_native_workgroupsize(execution, 1),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

@inline function _gramis_local_group(starts, counts, proposal_slot, round)
    return @inbounds(starts[proposal_slot]), @inbounds(counts[proposal_slot, round])
end

@kernel function _local_weight_summary_kernel!(
    normalized_weights,
    local_ess,
    tempering_powers,
    status,
    local_logweights,
    starts,
    counts,
    round,
)
    proposal_slot = @index(Global, Linear)
    first_sample, sample_count = _gramis_local_group(
        starts,
        counts,
        proposal_slot,
        round,
    )
    last_sample = first_sample + sample_count - 1
    maximum_logweight = eltype(local_logweights)(-Inf)
    for sample_index in first_sample:last_sample
        maximum_logweight = max(
            maximum_logweight,
            @inbounds(local_logweights[sample_index]),
        )
    end

    T = eltype(normalized_weights)
    if maximum_logweight == -Inf
        @inbounds local_ess[proposal_slot] = zero(T)
        @inbounds tempering_powers[proposal_slot] = zero(T)
        @inbounds status[proposal_slot] = _GRAMIS_ALL_ZERO_LOCAL
    else
        total = zero(T)
        square_total = zero(T)
        for sample_index in first_sample:last_sample
            weight = T(exp(
                @inbounds(local_logweights[sample_index]) - maximum_logweight,
            ))
            @inbounds normalized_weights[sample_index] = weight
            total += weight
            square_total += abs2(weight)
        end
        inverse_total = inv(total)
        for sample_index in first_sample:last_sample
            @inbounds normalized_weights[sample_index] *= inverse_total
        end
        @inbounds local_ess[proposal_slot] = abs2(total) / square_total
        @inbounds tempering_powers[proposal_slot] = one(T)
        @inbounds status[proposal_slot] = _GRAMIS_COVARIANCE_READY
    end
end

function _local_weight_summary!(
    method_state::_PreparedFirstOrderGRAMIS,
    round,
    execution,
)
    workspace = method_state.workspace
    backend = KernelAbstractions.get_backend(workspace.normalized_weights)
    proposal_count = size(method_state.run.locations, 2)
    kernel = _local_weight_summary_kernel!(backend)
    kernel(
        workspace.normalized_weights,
        workspace.local_ess,
        workspace.tempering_powers,
        workspace.factor_status,
        workspace.local_logweights,
        workspace.local_starts,
        method_state.plan.counts,
        round;
        ndrange=proposal_count,
        workgroupsize=_native_workgroupsize(execution, proposal_count),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

@inline function _gramis_power_ess(
    local_logweights,
    first_sample,
    last_sample,
    power,
    ::Type{T},
) where {T}
    maximum_scaled = T(-Inf)
    for sample_index in first_sample:last_sample
        scaled = power * T(@inbounds(local_logweights[sample_index]))
        maximum_scaled = max(maximum_scaled, scaled)
    end
    total = zero(T)
    square_total = zero(T)
    for sample_index in first_sample:last_sample
        scaled = power * T(@inbounds(local_logweights[sample_index]))
        shifted_weight = exp(scaled - maximum_scaled)
        total += shifted_weight
        square_total += abs2(shifted_weight)
    end
    return abs2(total) / square_total
end

@kernel function _tempering_power_kernel!(
    normalized_weights,
    local_ess,
    tempering_powers,
    status,
    local_logweights,
    starts,
    counts,
    thresholds,
    round,
    tolerance,
    max_iterations,
)
    proposal_slot = @index(Global, Linear)
    T = eltype(normalized_weights)
    if @inbounds(status[proposal_slot]) == _GRAMIS_COVARIANCE_READY &&
       @inbounds(local_ess[proposal_slot]) <
       T(@inbounds(thresholds[proposal_slot, round]))
        first_sample, sample_count = _gramis_local_group(
            starts,
            counts,
            proposal_slot,
            round,
        )
        last_sample = first_sample + sample_count - 1
        threshold = T(@inbounds thresholds[proposal_slot, round])
        feasible_lower = zero(T)
        infeasible_upper = one(T)
        feasible_ess = zero(T)
        positive_feasible = false
        for _ in 1:max_iterations
            power = (feasible_lower + infeasible_upper) / T(2)
            ess = _gramis_power_ess(
                local_logweights,
                first_sample,
                last_sample,
                power,
                T,
            )
            if ess >= threshold
                feasible_lower = power
                feasible_ess = ess
                positive_feasible = true
            else
                infeasible_upper = power
            end
            infeasible_upper - feasible_lower <= tolerance && break
        end

        if positive_feasible
            maximum_scaled = T(-Inf)
            for sample_index in first_sample:last_sample
                scaled = feasible_lower * T(@inbounds(local_logweights[sample_index]))
                maximum_scaled = max(maximum_scaled, scaled)
            end
            total = zero(T)
            for sample_index in first_sample:last_sample
                scaled = feasible_lower * T(@inbounds(local_logweights[sample_index]))
                weight = exp(scaled - maximum_scaled)
                @inbounds normalized_weights[sample_index] = weight
                total += weight
            end
            inverse_total = inv(total)
            for sample_index in first_sample:last_sample
                @inbounds normalized_weights[sample_index] *= inverse_total
            end
            @inbounds local_ess[proposal_slot] = feasible_ess
            @inbounds tempering_powers[proposal_slot] = feasible_lower
        else
            @inbounds tempering_powers[proposal_slot] = zero(T)
            @inbounds status[proposal_slot] = _GRAMIS_TEMPERING_FALLBACK
        end
    end
end

function _tempering_power!(
    method_state::_PreparedFirstOrderGRAMIS,
    round,
    execution,
)
    workspace = method_state.workspace
    backend = KernelAbstractions.get_backend(workspace.normalized_weights)
    proposal_count = size(method_state.run.locations, 2)
    kernel = _tempering_power_kernel!(backend)
    kernel(
        workspace.normalized_weights,
        workspace.local_ess,
        workspace.tempering_powers,
        workspace.factor_status,
        workspace.local_logweights,
        workspace.local_starts,
        method_state.plan.counts,
        method_state.covariance_ess_threshold,
        round,
        method_state.tempering_tolerance,
        method_state.tempering_max_iterations;
        ndrange=proposal_count,
        workgroupsize=_native_workgroupsize(execution, proposal_count),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

@inline function _gramis_current_covariance(factors, row, column, proposal_slot)
    T = eltype(factors)
    covariance = zero(T)
    for factor_column in 1:min(row, column)
        covariance += @inbounds(
            factors[row, factor_column, proposal_slot] *
            factors[column, factor_column, proposal_slot]
        )
    end
    return covariance
end

@kernel function _fit_local_covariances_kernel!(
    covariances,
    normalized_weights,
    tempering_powers,
    status,
    samples,
    locations,
    factors,
    starts,
    counts,
    round,
)
    entry = @index(Global, Linear)
    dimension = size(samples, 1)
    entries_per_proposal = dimension * dimension
    proposal_slot = (entry - 1) ÷ entries_per_proposal + 1
    matrix_entry = (entry - 1) % entries_per_proposal
    row = matrix_entry % dimension + 1
    column = matrix_entry ÷ dimension + 1
    canonical_row = min(row, column)
    canonical_column = max(row, column)

    if @inbounds(status[proposal_slot]) == _GRAMIS_COVARIANCE_READY
        first_sample, sample_count = _gramis_local_group(
            starts,
            counts,
            proposal_slot,
            round,
        )
        last_sample = first_sample + sample_count - 1
        T = eltype(covariances)
        center_row = @inbounds locations[canonical_row, proposal_slot]
        center_column = @inbounds locations[canonical_column, proposal_slot]
        if @inbounds(tempering_powers[proposal_slot]) < one(T)
            center_row = zero(T)
            center_column = zero(T)
            for sample_index in first_sample:last_sample
                weight = @inbounds normalized_weights[sample_index]
                center_row += weight * @inbounds(samples[canonical_row, sample_index])
                center_column +=
                    weight * @inbounds(samples[canonical_column, sample_index])
            end
        end
        covariance = zero(T)
        for sample_index in first_sample:last_sample
            weight = @inbounds normalized_weights[sample_index]
            centered_row =
                @inbounds(samples[canonical_row, sample_index]) - center_row
            centered_column =
                @inbounds(samples[canonical_column, sample_index]) - center_column
            covariance += weight * centered_row * centered_column
        end
        @inbounds covariances[row, column, proposal_slot] = covariance
    else
        @inbounds covariances[row, column, proposal_slot] =
            _gramis_current_covariance(factors, row, column, proposal_slot)
    end
end

function _fit_local_covariances!(
    method_state::_PreparedFirstOrderGRAMIS,
    round,
    execution,
)
    _local_group_starts!(
        method_state.workspace.local_starts,
        method_state.plan.counts,
        round,
        execution,
    )
    _local_weight_summary!(method_state, round, execution)
    _tempering_power!(method_state, round, execution)

    workspace = method_state.workspace
    bank = method_state.run
    backend = KernelAbstractions.get_backend(workspace.covariances)
    covariance_entries = length(workspace.covariances)
    kernel = _fit_local_covariances_kernel!(backend)
    kernel(
        workspace.covariances,
        workspace.normalized_weights,
        workspace.tempering_powers,
        workspace.factor_status,
        workspace.samples,
        bank.locations,
        bank.factors,
        workspace.local_starts,
        method_state.plan.counts,
        round;
        ndrange=covariance_entries,
        workgroupsize=_native_workgroupsize(execution, covariance_entries),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function _blend_local_covariances_kernel!(
    covariances,
    factors,
    status,
    covariance_rate,
    round,
    regularization,
)
    proposal_slot = @index(Global, Linear)
    if @inbounds(status[proposal_slot]) == _GRAMIS_COVARIANCE_READY
        T = eltype(covariances)
        dimension = size(covariances, 1)
        previous_trace = zero(T)
        for column in 1:dimension, row in column:dimension
            previous_trace += abs2(@inbounds factors[row, column, proposal_slot])
        end
        ridge = _scale_aware_ridge(
            previous_trace,
            dimension,
            regularization,
        )
        rate = T(@inbounds covariance_rate[round])
        for column in 1:dimension, row in column:dimension
            estimate = (
                @inbounds(covariances[row, column, proposal_slot]) +
                @inbounds(covariances[column, row, proposal_slot])
            ) / T(2)
            old = _gramis_current_covariance(
                factors,
                row,
                column,
                proposal_slot,
            )
            blended = (one(T) - rate) * old + rate * estimate
            row == column && (blended += ridge)
            @inbounds covariances[row, column, proposal_slot] = blended
            @inbounds covariances[column, row, proposal_slot] = blended
        end
    end
end

function _blend_local_covariances!(
    method_state::_PreparedFirstOrderGRAMIS,
    round,
    execution,
)
    workspace = method_state.workspace
    proposal_count = size(method_state.run.locations, 2)
    backend = KernelAbstractions.get_backend(workspace.covariances)
    kernel = _blend_local_covariances_kernel!(backend)
    kernel(
        workspace.covariances,
        method_state.run.factors,
        workspace.factor_status,
        method_state.covariance_rate,
        round,
        method_state.covariance_regularization;
        ndrange=proposal_count,
        workgroupsize=_native_workgroupsize(execution, proposal_count),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

const _GRAMIS_COVARIANCE_NONFINITE_INFO = -1

@inline function _factor_population_slot!(
    factors,
    covariances,
    info,
    proposal_slot,
)
    factor = view(factors, :, :, proposal_slot)
    covariance = view(covariances, :, :, proposal_slot)
    copyto!(factor, covariance)
    finite = true
    @inbounds for entry in eachindex(factor)
        finite &= isfinite(factor[entry])
    end
    if !finite
        @inbounds info[proposal_slot] = _GRAMIS_COVARIANCE_NONFINITE_INFO
        return nothing
    end

    factor, factor_info = LinearAlgebra.LAPACK.potrf!('L', factor)
    @inbounds info[proposal_slot] = factor_info
    if iszero(factor_info)
        dimension = size(factor, 1)
        @inbounds for column in 1:dimension, row in 1:(column - 1)
            factor[row, column] = zero(eltype(factor))
        end
    end
    return nothing
end

function _factor_population!(
    ::MLDataDevices.AbstractCPUDevice,
    factors::StridedArray{T,3},
    covariances::StridedArray{T,3},
    info::StridedVector{Int},
    ::_SerialCPUExecution,
) where {T<:Union{Float32,Float64}}
    @inbounds for proposal_slot in axes(factors, 3)
        _factor_population_slot!(
            factors,
            covariances,
            info,
            proposal_slot,
        )
    end
    return nothing
end

function _factor_population!(
    ::MLDataDevices.AbstractCPUDevice,
    factors::StridedArray{T,3},
    covariances::StridedArray{T,3},
    info::StridedVector{Int},
    ::_ThreadedCPUExecution,
) where {T<:Union{Float32,Float64}}
    Threads.@threads :dynamic for proposal_slot in axes(factors, 3)
        _factor_population_slot!(
            factors,
            covariances,
            info,
            proposal_slot,
        )
    end
    return nothing
end

function _factor_population!(
    device::MLDataDevices.AbstractCPUDevice,
    factors::StridedArray{T,3},
    covariances::StridedArray{T,3},
    info::StridedVector{Int},
) where {T<:Union{Float32,Float64}}
    return _factor_population!(
        device,
        factors,
        covariances,
        info,
        _SerialCPUExecution(),
    )
end

function _factor_ready_population!(
    factors,
    covariances,
    info,
    status,
    ::_SerialCPUExecution,
)
    @inbounds for proposal_slot in axes(factors, 3)
        if status[proposal_slot] == _GRAMIS_COVARIANCE_READY
            _factor_population_slot!(
                factors,
                covariances,
                info,
                proposal_slot,
            )
        else
            info[proposal_slot] = 0
        end
    end
    return nothing
end

function _factor_ready_population!(
    factors,
    covariances,
    info,
    status,
    ::_ThreadedCPUExecution,
)
    Threads.@threads :dynamic for proposal_slot in axes(factors, 3)
        if @inbounds(status[proposal_slot]) == _GRAMIS_COVARIANCE_READY
            _factor_population_slot!(
                factors,
                covariances,
                info,
                proposal_slot,
            )
        else
            @inbounds info[proposal_slot] = 0
        end
    end
    return nothing
end

function _update_local_covariances!(
    ::MLDataDevices.AbstractCPUDevice,
    method_state::_PreparedFirstOrderGRAMIS,
    round,
    info::StridedVector{Int},
    execution,
)
    copyto!(method_state.candidate.factors, method_state.run.factors)
    _blend_local_covariances!(method_state, round, execution)
    workspace = method_state.workspace
    _factor_ready_population!(
        method_state.candidate.factors,
        workspace.covariances,
        info,
        workspace.factor_status,
        execution,
    )
    if any(!iszero, info)
        copyto!(method_state.candidate.factors, method_state.run.factors)
    end
    return nothing
end
