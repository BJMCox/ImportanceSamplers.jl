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

"""
    FirstOrderGRAMISRoundError

Exception thrown when a [`FirstOrderGRAMIS`](@ref) call cannot complete a
round. `round` and `phase` locate the failed operation, `cause` retains the
underlying exception, and `diagnostics.completed_rounds` reports how many
round transitions completed. Diagnostics also contain the requested round
size, bounded covariance or derivative details when applicable, and explicit
device-transfer counts.

The prepared sampler's proposal population is one call-level transaction: a
failure leaves the population committed before the call unchanged and returns
no partial result. Random numbers consumed before the failure remain consumed.
"""
struct FirstOrderGRAMISRoundError{E,D<:NamedTuple} <: Exception
    round::Int
    phase::Symbol
    cause::E
    diagnostics::D
end

function Base.showerror(io::IO, error::FirstOrderGRAMISRoundError)
    print(
        io,
        "FirstOrderGRAMIS failed in round ",
        error.round,
        " during ",
        error.phase,
        ": ",
    )
    showerror(io, error.cause)
end

struct _FirstOrderGRAMISCovarianceError <: Exception
    proposal_slot::Int
    info::Int
end

function Base.showerror(io::IO, error::_FirstOrderGRAMISCovarianceError)
    print(
        io,
        "FirstOrderGRAMIS covariance factorization failed for proposal ",
        error.proposal_slot,
        " (info=",
        error.info,
        ')',
    )
end

struct _FirstOrderGRAMISProposalError{V} <: Exception
    proposal_slot::Int
    reason::Symbol
    value::V
end

function Base.showerror(io::IO, error::_FirstOrderGRAMISProposalError)
    print(
        io,
        "FirstOrderGRAMIS candidate proposal failure for proposal ",
        error.proposal_slot,
        ": ",
        error.reason,
        " (",
        error.value,
        ')',
    )
end

function _first_order_gramis_round_phase(cause::SamplerExecutionError, default)
    cause.phase === :target && return :target
    cause.phase === :proposal_logdensity && return :denominator
    cause.phase === :logweight && return :weight
    cause.phase === :proposal_draw && return :sampling
    return default
end

_first_order_gramis_round_phase(::_FirstOrderGRAMISDerivativeError, default) =
    :derivative
_first_order_gramis_round_phase(::_FirstOrderGRAMISRepulsionError, default) =
    :repulsion
_first_order_gramis_round_phase(::_FirstOrderGRAMISCovarianceError, default) =
    :covariance
_first_order_gramis_round_phase(::_FirstOrderGRAMISProposalError, default) =
    :proposal
_first_order_gramis_round_phase(cause, default) = default

function _first_order_gramis_failure_details(cause)
    covariance = cause isa _FirstOrderGRAMISCovarianceError ?
                 (proposal_slot=cause.proposal_slot, info=cause.info) : nothing
    derivative = cause isa _FirstOrderGRAMISDerivativeError ?
                 (
        proposal_slot=cause.proposal_slot,
        reason=cause.reason,
        value=cause.value,
    ) : nothing
    return (; covariance, derivative)
end

@inline function _capture_first_order_gramis_round(
    f,
    method_state,
    transfers,
    round,
    phase,
    completed_rounds,
)
    try
        return f()
    catch cause
        cause isa FirstOrderGRAMISRoundError && rethrow()
        failure_phase = _first_order_gramis_round_phase(cause, phase)
        details = _first_order_gramis_failure_details(cause)
        throw(
            FirstOrderGRAMISRoundError(
                round,
                failure_phase,
                cause,
                (
                    round_size=method_state.plan.schedule[round],
                    completed_rounds=completed_rounds,
                    covariance=details.covariance,
                    derivative=details.derivative,
                    transfers=transfers,
                    pre_call_state_preserved=true,
                ),
            ),
        )
    end
end

function _copy_first_order_gramis_population!(destination, source)
    copyto!(destination.locations, source.locations)
    copyto!(destination.factors, source.factors)
    copyto!(destination.lognormalizers, source.lognormalizers)
    return nothing
end

function _first_order_gramis_round_views(method_state, round)
    workspace = method_state.workspace
    round_size = method_state.plan.schedule[round]
    return (
        round_size=round_size,
        samples=_sample_view(workspace.samples, 1:round_size),
        logweights=view(workspace.round_logweights, 1:round_size),
        local_logweights=view(workspace.local_logweights, 1:round_size),
        generating_logdensities=view(
            workspace.generating_logdensities,
            1:round_size,
        ),
        proposal_ids=view(workspace.round_proposal_ids, 1:round_size),
        round_ids=view(workspace.round_ids, 1:round_size),
        assignments=view(method_state.plan.assignments, 1:round_size, round),
    )
end

function _add_first_order_gramis_repulsion!(
    ::MLDataDevices.AbstractCPUDevice,
    candidate,
    repulsion,
    transfers,
    execution,
)
    @inbounds for proposal_slot in axes(candidate, 2), row in axes(candidate, 1)
        value = candidate[row, proposal_slot] + repulsion[row, proposal_slot]
        isfinite(value) || throw(
            _FirstOrderGRAMISProposalError(
                proposal_slot,
                :location_nonfinite,
                value,
            ),
        )
        candidate[row, proposal_slot] = value
    end
    return nothing
end

function _validate_first_order_gramis_candidate_factors!(
    ::MLDataDevices.AbstractCPUDevice,
    candidate,
    status,
    transfers,
    execution,
)
    T = eltype(candidate.factors)
    dimension = size(candidate.factors, 1)
    @inbounds for proposal_slot in axes(candidate.factors, 3)
        status[proposal_slot] == _GRAMIS_COVARIANCE_READY || continue
        logabsdet = zero(T)
        for column in axes(candidate.factors, 2), row in axes(candidate.factors, 1)
            value = candidate.factors[row, column, proposal_slot]
            isfinite(value) || throw(
                _FirstOrderGRAMISProposalError(
                    proposal_slot,
                    :factor_nonfinite,
                    value,
                ),
            )
            if row == column
                value > zero(T) || throw(
                    _FirstOrderGRAMISProposalError(
                        proposal_slot,
                        :factor_diagonal_invalid,
                        value,
                    ),
                )
                logabsdet += log(value)
            end
        end
        lognormalizer = _gaussian_lognormalizer(T, dimension, logabsdet)
        isfinite(lognormalizer) || throw(
            _FirstOrderGRAMISProposalError(
                proposal_slot,
                :lognormalizer_nonfinite,
                lognormalizer,
            ),
        )
        candidate.lognormalizers[proposal_slot] = lognormalizer
    end
    return nothing
end

function _throw_first_order_gramis_covariance_failure(
    ::MLDataDevices.AbstractCPUDevice,
    info,
    transfers,
    execution,
)
    @inbounds for proposal_slot in eachindex(info)
        iszero(info[proposal_slot]) || throw(
            _FirstOrderGRAMISCovarianceError(
                proposal_slot,
                info[proposal_slot],
            ),
        )
    end
    return nothing
end

function _minimum_first_order_gramis_whitened_distance(
    ::MLDataDevices.AbstractCPUDevice,
    whitened_means,
    transfers,
    execution,
)
    T = eltype(whitened_means)
    minimum_distance = T(Inf)
    @inbounds for right in 2:size(whitened_means, 2), left in 1:(right - 1)
        distance = zero(T)
        for row in axes(whitened_means, 1)
            distance = hypot(
                distance,
                whitened_means[row, right] - whitened_means[row, left],
            )
        end
        minimum_distance = min(minimum_distance, distance)
    end
    return minimum_distance
end

function _first_order_gramis_diagnostic_summary(
    ::MLDataDevices.AbstractCPUDevice,
    status,
    steps,
    trials,
    transfers,
    execution,
)
    all_zero = 0
    tempering = 0
    backtracking = 0
    target_trials = 0
    @inbounds for proposal_slot in eachindex(status, steps, trials)
        proposal_status = status[proposal_slot]
        all_zero += proposal_status == _GRAMIS_ALL_ZERO_LOCAL
        tempering += proposal_status == _GRAMIS_TEMPERING_FALLBACK
        backtracking += iszero(steps[proposal_slot])
        target_trials += trials[proposal_slot]
    end
    return (
        all_zero=all_zero,
        tempering=tempering,
        backtracking=backtracking,
        target_trials=target_trials,
    )
end

function _importance_sample_cpu!(
    sampler,
    method_state::_PreparedFirstOrderGRAMIS,
    threaded,
)
    execution = threaded ? _ThreadedCPUExecution() : _SerialCPUExecution()
    plan = method_state.plan
    workspace = method_state.workspace
    buffers = sampler.random_buffers
    rounds = length(plan.schedule)
    total_samples = last(plan.offsets) - 1
    dimension, proposal_count = size(method_state.committed.locations)
    T = eltype(method_state.committed.locations)
    L = eltype(workspace.round_logweights)
    transfers = _ResultTransferCounter(0, 0)

    storage = _capture_first_order_gramis_round(
        method_state,
        transfers,
        1,
        :result_construction,
        0,
    ) do
        allocated_active_rounds = findall(
            !iszero,
            method_state.repulsion_strength,
        )
        active_repulsion_count = length(allocated_active_rounds)
        (
            samples=similar(workspace.samples, T, dimension, total_samples),
            logweights=similar(workspace.round_logweights, L, total_samples),
            round_ids=similar(workspace.round_ids, Int, total_samples),
            proposal_ids=similar(
                workspace.round_proposal_ids,
                Int,
                total_samples,
            ),
            local_ess=similar(workspace.local_ess, T, proposal_count, rounds),
            tempering_powers=similar(
                workspace.tempering_powers,
                T,
                proposal_count,
                rounds,
            ),
            fallback_status=similar(
                workspace.factor_status,
                UInt8,
                proposal_count,
                rounds,
            ),
            accepted_steps=similar(
                workspace.steps,
                T,
                proposal_count,
                rounds,
            ),
            backtracking_trials=similar(
                workspace.backtracking_trials,
                Int,
                proposal_count,
                rounds,
            ),
            collision_counts=similar(
                workspace.collision_counts,
                Int,
                proposal_count,
                rounds,
            ),
            round_ess=Vector{L}(undef, rounds),
            round_lognormalizers=Vector{L}(undef, rounds),
            active_repulsion_rounds=allocated_active_rounds,
            minimum_whitened_distances=Vector{T}(
                undef,
                active_repulsion_count,
            ),
        )
    end
    samples = storage.samples
    logweights = storage.logweights
    round_ids = storage.round_ids
    proposal_ids = storage.proposal_ids
    local_ess = storage.local_ess
    tempering_powers = storage.tempering_powers
    fallback_status = storage.fallback_status
    accepted_steps = storage.accepted_steps
    backtracking_trials = storage.backtracking_trials
    collision_counts = storage.collision_counts
    round_ess = storage.round_ess
    round_lognormalizers = storage.round_lognormalizers
    active_repulsion_rounds = storage.active_repulsion_rounds::Vector{Int}
    minimum_whitened_distances = storage.minimum_whitened_distances::Vector{T}
    active_repulsion_position = Ref(0)
    fallback_all_zero = Ref(0)
    fallback_tempering = Ref(0)
    fallback_backtracking = Ref(0)
    target_trial_evaluations = Ref(0)

    _capture_first_order_gramis_round(
        method_state,
        transfers,
        1,
        :proposal,
        0,
    ) do
        _copy_first_order_gramis_population!(
            method_state.run,
            method_state.committed,
        )
    end
    target = _capture_first_order_gramis_round(
        method_state,
        transfers,
        1,
        :target,
        0,
    ) do
        _bind_resolved_target(
            sampler.target,
            view(method_state.run.locations, :, 1),
        )
    end
    target_evaluator, target_failures = _capture_first_order_gramis_round(
        method_state,
        transfers,
        1,
        :target,
        0,
    ) do
        _native_target_evaluator(
            KernelAbstractions.get_backend(buffers.normal),
            target,
            L,
            buffers.failure_scratch.target_failures,
        )
    end

    for round in eachindex(plan.schedule)
        views = _first_order_gramis_round_views(method_state, round)
        output_indices = plan.offsets[round]:(plan.offsets[round + 1] - 1)
        denominator = _RealizedMixtureDenominator(plan.logcoefficients, round)
        normal_buffer = view(
            buffers.normal,
            1:(dimension * views.round_size),
        )

        _capture_first_order_gramis_round(
            method_state,
            transfers,
            round,
            :sampling,
            round - 1,
        ) do
            Random.randn!(sampler.rng, normal_buffer)
            _first_order_gramis_sample_round!(
                views.samples,
                views.logweights,
                views.local_logweights,
                views.generating_logdensities,
                views.proposal_ids,
                views.round_ids,
                buffers.failure_scratch.record.storage,
                normal_buffer,
                target_evaluator,
                method_state.run,
                views.assignments,
                denominator,
                workspace.solve_scratch,
                round,
                execution,
                sampler.device,
                sampler.factor_execution,
            )
            snapshot = _device_failure_snapshot(buffers.failure_scratch.record)
            _record_reported_transfer!(
                transfers,
                snapshot.transfers.count,
                snapshot.transfers.bytes,
                Val(:failure_snapshot),
            )
            _throw_native_failures(
                snapshot.failure,
                snapshot.draw_failure,
                target_failures,
                _NoSampleTransform(),
            )
        end
        _capture_first_order_gramis_round(
            method_state,
            transfers,
            round,
            :covariance,
            round - 1,
        ) do
            _fit_local_covariances!(method_state, round, execution)
        end
        _capture_first_order_gramis_round(
            method_state,
            transfers,
            round,
            :derivative,
            round - 1,
        ) do
            _evaluate_frozen_gradients!(method_state, target, execution)
            _precondition_gradients!(method_state, execution)
        end
        _capture_first_order_gramis_round(
            method_state,
            transfers,
            round,
            :repulsion,
            round - 1,
        ) do
            strength = method_state.repulsion_strength[round]
            _repulsion!(
                workspace.repulsion,
                workspace.collision_counts,
                workspace.pooled_covariance,
                workspace.whitened_means,
                method_state.run.locations,
                method_state.run.factors,
                strength,
                method_state.repulsion_softening,
                execution,
            )
            if !iszero(strength)
                active_repulsion_position[] += 1
                minimum_whitened_distances[active_repulsion_position[]] =
                    _minimum_first_order_gramis_whitened_distance(
                        sampler.device,
                        workspace.whitened_means,
                        transfers,
                        execution,
                    )
            end
        end
        _capture_first_order_gramis_round(
            method_state,
            transfers,
            round,
            :derivative,
            round - 1,
        ) do
            _backtrack_means!(method_state, target, execution)
            _add_first_order_gramis_repulsion!(
                sampler.device,
                method_state.candidate.locations,
                workspace.repulsion,
                transfers,
                execution,
            )
        end
        _capture_first_order_gramis_round(
            method_state,
            transfers,
            round,
            :covariance,
            round - 1,
        ) do
            copyto!(
                method_state.candidate.lognormalizers,
                method_state.run.lognormalizers,
            )
            _update_local_covariances!(
                sampler.device,
                method_state,
                round,
                workspace.factor_info,
                execution,
            )
            _throw_first_order_gramis_covariance_failure(
                sampler.device,
                workspace.factor_info,
                transfers,
                execution,
            )
            _validate_first_order_gramis_candidate_factors!(
                sampler.device,
                method_state.candidate,
                workspace.factor_status,
                transfers,
                execution,
            )
        end
        _capture_first_order_gramis_round(
            method_state,
            transfers,
            round,
            :proposal,
            round - 1,
        ) do
            method_state.run, method_state.candidate =
                method_state.candidate, method_state.run
        end
        summary = _capture_first_order_gramis_round(
            method_state,
            transfers,
            round,
            :diagnostics,
            round,
        ) do
            copyto!(_sample_view(samples, output_indices), views.samples)
            copyto!(view(logweights, output_indices), views.logweights)
            copyto!(view(round_ids, output_indices), views.round_ids)
            copyto!(view(proposal_ids, output_indices), views.proposal_ids)
            copyto!(view(local_ess, :, round), workspace.local_ess)
            copyto!(
                view(tempering_powers, :, round),
                workspace.tempering_powers,
            )
            copyto!(view(fallback_status, :, round), workspace.factor_status)
            copyto!(view(accepted_steps, :, round), workspace.steps)
            copyto!(
                view(backtracking_trials, :, round),
                workspace.backtracking_trials,
            )
            copyto!(
                view(collision_counts, :, round),
                workspace.collision_counts,
            )
            (
                weights=_logweight_summary(views.logweights, transfers),
                bounded=_first_order_gramis_diagnostic_summary(
                    sampler.device,
                    workspace.factor_status,
                    workspace.steps,
                    workspace.backtracking_trials,
                    transfers,
                    execution,
                ),
            )
        end
        round_ess[round] = summary.weights.ess
        round_lognormalizers[round] = summary.weights.lognormalizer
        fallback_all_zero[] += summary.bounded.all_zero
        fallback_tempering[] += summary.bounded.tempering
        fallback_backtracking[] += summary.bounded.backtracking
        target_trial_evaluations[] += summary.bounded.target_trials
    end

    diagnostics = _capture_first_order_gramis_round(
        method_state,
        transfers,
        rounds,
        :result_construction,
        rounds,
    ) do
        (
            method=:first_order_gramis,
            execution=_execution_name(execution),
            threaded=sampler.threaded,
            factor_execution_policy=_factor_execution_name(
                sampler.factor_execution,
            ),
            rounds=rounds,
            round_sizes=collect(plan.schedule),
            round_ess=round_ess,
            round_lognormalizers=round_lognormalizers,
            local_ess=local_ess,
            tempering_powers=tempering_powers,
            fallback_status=fallback_status,
            accepted_steps=accepted_steps,
            backtracking_trials=backtracking_trials,
            collision_counts=collision_counts,
            minimum_whitened_pair_distance=(
                round=active_repulsion_rounds,
                value=minimum_whitened_distances,
            ),
            fallbacks=(
                all_zero_local_weights=fallback_all_zero[],
                tempering=fallback_tempering[],
                backtracking_exhaustion=fallback_backtracking[],
            ),
            target_evaluations=total_samples + rounds * proposal_count +
                               target_trial_evaluations[],
            gradient_evaluations=rounds * proposal_count,
            proposal_evaluations=proposal_count * total_samples,
            denominator_evaluations=total_samples,
            failures=0,
            transfers=transfers,
        )
    end
    result = _capture_first_order_gramis_round(
        method_state,
        transfers,
        rounds,
        :result_construction,
        rounds,
    ) do
        constructed = _adopt_validated_weighted_samples(
            samples,
            logweights;
            provenance=(round=round_ids, proposal_id=proposal_ids),
            diagnostics=diagnostics,
        )
        KernelAbstractions.synchronize(
            KernelAbstractions.get_backend(method_state.run.locations),
        )
        constructed
    end
    method_state.committed, method_state.run =
        method_state.run, method_state.committed
    return result
end
