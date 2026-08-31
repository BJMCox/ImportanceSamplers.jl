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

@inline function _first_order_gramis_local_weight_arguments(
    local_logweights, generating_logdensities, proposal_ids, round_ids,
    round, failure_storage,
)
    return (
        local_logweights,
        generating_logdensities,
        proposal_ids,
        round_ids,
        round,
        failure_storage,
    )
end

@inline _first_order_gramis_workgroupsize(execution, backend, ndrange) =
    _native_workgroupsize(execution, ndrange)

@inline function _first_order_gramis_workgroupsize(
    ::_ThreadedCPUExecution,
    ::KernelAbstractions.CPU,
    ndrange,
)
    return min(
        1_024,
        max(1, fld(ndrange, Threads.nthreads(:default))),
    )
end

@inline function _first_order_gramis_sample_slot!(
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
    sample_index,
)
    T = eltype(returned_logweights)
    @inbounds returned_logweights[sample_index] = T(-Inf)
    @inbounds local_logweights[sample_index] = T(-Inf)
    @inbounds generating_logdensities[sample_index] = T(-Inf)
    @inbounds proposal_ids[sample_index] = 0
    @inbounds round_ids[sample_index] = 0
    valid, target_log, logdenominator, generating_slot, generating_logdensity =
        _mis_round_values!(
            T,
            sample_index,
            samples,
            failure_storage,
            normal_buffer,
            target,
            bank,
            assignments,
            denominator,
            solve_scratch,
        )
    valid || return nothing
    returned_logweight, returned_reason =
        _subtract_logweight(target_log, logdenominator)
    if !iszero(returned_reason)
        _record_native_failure!(
            failure_storage,
            sample_index,
            0,
            returned_reason,
        )
        return nothing
    end
    @inbounds returned_logweights[sample_index] = returned_logweight
    @inbounds local_logweights[sample_index] = target_log
    @inbounds generating_logdensities[sample_index] = generating_logdensity
    @inbounds proposal_ids[sample_index] = bank.proposal_ids[generating_slot]
    local_logweight, local_reason =
        _subtract_logweight(target_log, generating_logdensity)
    if iszero(local_reason)
        @inbounds local_logweights[sample_index] = local_logweight
        @inbounds round_ids[sample_index] = round
    else
        _record_native_failure!(failure_storage, sample_index, 0, local_reason)
    end
    return nothing
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
    ::_SerialCPUExecution,
    ::MLDataDevices.AbstractCPUDevice,
    ::FusedFactorExecution,
)
    @inbounds for sample_index in eachindex(returned_logweights)
        _first_order_gramis_sample_slot!(
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
            sample_index,
        )
    end
    return nothing
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
        _first_order_gramis_local_weight_arguments(
            local_logweights,
            generating_logdensities,
            proposal_ids,
            round_ids,
            round,
            failure_storage,
        )...;
        ndrange=length(local_logweights),
        workgroupsize=_first_order_gramis_workgroupsize(
            execution,
            backend,
            length(local_logweights),
        ),
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

const _GRAMIS_FROZEN_VALUE_NONFINITE = UInt16(0x4001)
const _GRAMIS_GRADIENT_NONFINITE = UInt16(0x4002)
const _GRAMIS_MOVE_NONFINITE = UInt16(0x4003)
const _GRAMIS_CANDIDATE_VALUE_NONFINITE = UInt16(0x4004)
const _GRAMIS_POOLED_COVARIANCE_NONFINITE = UInt16(0x4005)
const _GRAMIS_FORCE_NONFINITE = UInt16(0x4006)
const _GRAMIS_LOCATION_NONFINITE = UInt16(0x4007)
const _GRAMIS_FACTOR_NONFINITE = UInt16(0x4008)
const _GRAMIS_FACTOR_DIAGONAL_INVALID = UInt16(0x4009)
const _GRAMIS_LOGNORMALIZER_NONFINITE = UInt16(0x400a)
const _GRAMIS_COVARIANCE_FACTORIZATION_FAILED = UInt16(0x400b)

function _first_order_gramis_failure_snapshot!(transfers, failure_record)
    snapshot = _device_failure_snapshot(failure_record)
    _record_reported_transfer!(
        transfers,
        snapshot.transfers.count,
        snapshot.transfers.bytes,
        Val(:failure_snapshot),
    )
    return snapshot.failure
end

function _first_order_gramis_failure_value(device, values, index, transfers)
    _is_host_storage(values) || throw(
        SamplerDeviceError(device, :kernel_argument_unsupported),
    )
    return @inbounds vec(values)[index]
end

function _first_order_gramis_derivative_reason(reason)
    reason == _GRAMIS_FROZEN_VALUE_NONFINITE && return :frozen_value_nonfinite
    reason == _GRAMIS_GRADIENT_NONFINITE && return :gradient_nonfinite
    reason == _GRAMIS_MOVE_NONFINITE && return :move_nonfinite
    reason == _GRAMIS_CANDIDATE_VALUE_NONFINITE &&
        return :candidate_value_nonfinite
    error("unknown FirstOrderGRAMIS derivative failure code")
end

function _throw_first_order_gramis_device_derivative_failure(
    device,
    failure_record,
    failure_values,
    transfers,
)
    failure = _first_order_gramis_failure_snapshot!(transfers, failure_record)
    iszero(failure.count) && return nothing
    slot = failure.first_logical_index
    value = _first_order_gramis_failure_value(
        device,
        failure_values,
        slot,
        transfers,
    )
    _throw_first_order_gramis_derivative_error(
        slot,
        _first_order_gramis_derivative_reason(failure.reason_bits),
        value,
    )
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
    ::_KernelExecution,
)
    dimension = size(factors, 1)
    proposal_count = size(factors, 3)
    packed_factors = reshape(
        factors,
        dimension,
        dimension * proposal_count,
    )
    T = eltype(pooled_covariance)
    LinearAlgebra.mul!(
        pooled_covariance,
        packed_factors,
        transpose(packed_factors),
        inv(T(proposal_count)),
        zero(T),
    )
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

function _throw_first_order_gramis_device_repulsion_failure(
    device,
    failure_record,
    failure_values,
    transfers,
)
    failure = _first_order_gramis_failure_snapshot!(transfers, failure_record)
    iszero(failure.count) && return nothing
    index = failure.first_logical_index
    value = _first_order_gramis_failure_value(
        device,
        failure_values,
        index,
        transfers,
    )
    if failure.reason_bits == _GRAMIS_POOLED_COVARIANCE_NONFINITE
        _throw_first_order_gramis_repulsion_error(
            0,
            :pooled_covariance_nonfinite,
            value,
        )
    elseif failure.reason_bits == _GRAMIS_FORCE_NONFINITE
        _throw_first_order_gramis_repulsion_error(
            index,
            :force_nonfinite,
            value,
        )
    end
    error("unknown FirstOrderGRAMIS repulsion failure code")
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

@kernel function _validate_pooled_covariance_kernel!(failure_storage, pooled)
    entry = @index(Global, Linear)
    value = @inbounds pooled[entry]
    isfinite(value) || _record_native_failure!(
        failure_storage,
        entry,
        0,
        _GRAMIS_POOLED_COVARIANCE_NONFINITE,
    )
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

@kernel function _repulsion_force_kernel!(
    repulsion,
    collision_counts,
    means,
    whitened_means,
    strength,
    round,
    softening,
)
    proposal_slot = @index(Global, Linear)
    _repulsion_slot!(
        repulsion,
        collision_counts,
        means,
        whitened_means,
        @inbounds(strength[round]),
        softening,
        proposal_slot,
    )
end

@inline _repulsion_force_arguments(
    repulsion,
    collision_counts,
    means,
    whitened_means,
    strength,
    round,
    softening,
) = (
    repulsion,
    collision_counts,
    means,
    whitened_means,
    strength,
    round,
    softening,
)

function _repulsion_force!(
    repulsion,
    collision_counts,
    means,
    whitened_means,
    strength,
    round,
    softening,
    execution::_KernelExecution,
)
    backend = KernelAbstractions.get_backend(repulsion)
    proposal_count = size(means, 2)
    kernel = _repulsion_force_kernel!(backend)
    kernel(
        _repulsion_force_arguments(
            repulsion,
            collision_counts,
            means,
            whitened_means,
            strength,
            round,
            softening,
        )...;
        ndrange=proposal_count,
        workgroupsize=_first_order_gramis_workgroupsize(
            execution,
            backend,
            proposal_count,
        ),
    )
    KernelAbstractions.synchronize(backend)
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

@inline function _repulsion_failure(repulsion, proposal_slot)
    @inbounds for row in axes(repulsion, 1)
        value = repulsion[row, proposal_slot]
        isfinite(value) || return (_GRAMIS_FORCE_NONFINITE, value)
    end
    return (UInt16(0), zero(eltype(repulsion)))
end

function _validate_repulsion!(repulsion)
    @inbounds for proposal_slot in axes(repulsion, 2)
        reason, value = _repulsion_failure(repulsion, proposal_slot)
        iszero(reason) || _throw_first_order_gramis_repulsion_error(
            proposal_slot,
            :force_nonfinite,
            value,
        )
    end
    return nothing
end

@kernel function _validate_repulsion_kernel!(
    failure_storage,
    failure_values,
    repulsion,
)
    proposal_slot = @index(Global, Linear)
    reason, value = _repulsion_failure(repulsion, proposal_slot)
    if !iszero(reason)
        @inbounds failure_values[proposal_slot] = value
        _record_native_failure!(failure_storage, proposal_slot, 0, reason)
    end
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

function _repulsion!(
    repulsion,
    collision_counts,
    pooled_covariance,
    whitened_means,
    means,
    factors,
    strengths::AbstractVector,
    round,
    softening,
    execution::Union{_SerialCPUExecution,_ThreadedCPUExecution},
)
    return _repulsion!(
        repulsion,
        collision_counts,
        pooled_covariance,
        whitened_means,
        means,
        factors,
        @inbounds(strengths[round]),
        softening,
        execution,
    )
end

function _repulsion!(
    repulsion,
    collision_counts,
    pooled_covariance,
    whitened_means,
    means,
    factors,
    strength,
    round,
    softening,
    execution::_KernelExecution;
    device=nothing,
    failure_record=nothing,
    failure_values=nothing,
    transfers=nothing,
)
    _pooled_covariance!(pooled_covariance, factors, execution)
    backend = KernelAbstractions.get_backend(pooled_covariance)
    if device !== nothing
        fill!(failure_record.storage, zero(eltype(failure_record.storage)))
        pooled_validation = _validate_pooled_covariance_kernel!(backend)
        pooled_validation(
            failure_record.storage,
            pooled_covariance;
            ndrange=length(pooled_covariance),
            workgroupsize=_first_order_gramis_workgroupsize(
                execution,
                backend,
                length(pooled_covariance),
            ),
        )
        KernelAbstractions.synchronize(backend)
        _throw_first_order_gramis_device_repulsion_failure(
            device,
            failure_record,
            pooled_covariance,
            transfers,
        )
    end
    _factor_pooled_covariance!(pooled_covariance)
    _whiten_means!(whitened_means, pooled_covariance, means)
    _repulsion_force!(
        repulsion,
        collision_counts,
        means,
        whitened_means,
        strength,
        round,
        softening,
        execution,
    )
    device === nothing && return nothing
    fill!(failure_record.storage, zero(eltype(failure_record.storage)))
    force_validation = _validate_repulsion_kernel!(backend)
    proposal_count = size(repulsion, 2)
    force_validation(
        failure_record.storage,
        failure_values,
        repulsion;
        ndrange=proposal_count,
        workgroupsize=_first_order_gramis_workgroupsize(
            execution,
            backend,
            proposal_count,
        ),
    )
    KernelAbstractions.synchronize(backend)
    return _throw_first_order_gramis_device_repulsion_failure(
        device,
        failure_record,
        failure_values,
        transfers,
    )
end

@inline function _frozen_derivative_failure(values, gradients, proposal_slot)
    value = @inbounds values[proposal_slot]
    isfinite(value) || return (_GRAMIS_FROZEN_VALUE_NONFINITE, value)
    @inbounds for row in axes(gradients, 1)
        value = gradients[row, proposal_slot]
        isfinite(value) || return (_GRAMIS_GRADIENT_NONFINITE, value)
    end
    return (UInt16(0), value)
end

function _validate_frozen_derivatives!(values, gradients)
    @inbounds for proposal_slot in eachindex(values)
        reason, value = _frozen_derivative_failure(
            values,
            gradients,
            proposal_slot,
        )
        iszero(reason) || _throw_first_order_gramis_derivative_error(
            proposal_slot,
            _first_order_gramis_derivative_reason(reason),
            value,
        )
    end
    return nothing
end

@inline function _preconditioned_move_failure(moves, proposal_slot)
    @inbounds for row in axes(moves, 1)
        value = moves[row, proposal_slot]
        isfinite(value) || return (_GRAMIS_MOVE_NONFINITE, value)
    end
    return (UInt16(0), zero(eltype(moves)))
end

function _validate_preconditioned_moves!(moves)
    @inbounds for proposal_slot in axes(moves, 2)
        reason, value = _preconditioned_move_failure(moves, proposal_slot)
        iszero(reason) || _throw_first_order_gramis_derivative_error(
            proposal_slot,
            _first_order_gramis_derivative_reason(reason),
            value,
        )
    end
    return nothing
end

@inline function _backtracking_candidate_failure(
    candidate_values,
    trials,
    trial,
    proposal_slot,
)
    @inbounds(trials[proposal_slot]) == trial ||
        return (false, UInt16(0), zero(eltype(candidate_values)))
    value = @inbounds candidate_values[proposal_slot]
    reason = isfinite(value) || value == -Inf ?
             UInt16(0) : _GRAMIS_CANDIDATE_VALUE_NONFINITE
    return (true, reason, value)
end

function _validate_backtracking_candidates!(
    candidate_values,
    active_mask,
    trials,
    trial,
)
    any_active = false
    @inbounds for proposal_slot in eachindex(candidate_values, active_mask, trials)
        evaluated, reason, value = _backtracking_candidate_failure(
            candidate_values,
            trials,
            trial,
            proposal_slot,
        )
        evaluated || continue
        iszero(reason) || _throw_first_order_gramis_derivative_error(
            proposal_slot,
            _first_order_gramis_derivative_reason(reason),
            value,
        )
        any_active |= active_mask[proposal_slot]
    end
    return any_active
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

@kernel function _precondition_gradients_kernel!(moves, gradients, factors)
    proposal_slot = @index(Global, Linear)
    _precondition_gradient_slot!(moves, gradients, factors, proposal_slot)
end

@kernel function _validate_preconditioned_moves_kernel!(
    failure_storage,
    failure_values,
    moves,
)
    proposal_slot = @index(Global, Linear)
    reason, value = _preconditioned_move_failure(moves, proposal_slot)
    if !iszero(reason)
        @inbounds failure_values[proposal_slot] = value
        _record_native_failure!(failure_storage, proposal_slot, 0, reason)
    end
end

function _precondition_gradients!(
    moves,
    gradients,
    factors,
    execution::_KernelExecution,
)
    backend = KernelAbstractions.get_backend(moves)
    kernel = _precondition_gradients_kernel!(backend)
    proposal_count = size(gradients, 2)
    kernel(
        moves,
        gradients,
        factors;
        ndrange=proposal_count,
        workgroupsize=_first_order_gramis_workgroupsize(
            execution,
            backend,
            proposal_count,
        ),
    )
    KernelAbstractions.synchronize(backend)
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

@kernel function _evaluate_frozen_gradients_kernel!(
    values,
    gradients,
    target,
    bound_gradient,
    locations,
)
    proposal_slot = @index(Global, Linear)
    _evaluate_frozen_gradient_slot!(
        values,
        gradients,
        target,
        bound_gradient,
        locations,
        proposal_slot,
    )
end

@kernel function _validate_frozen_derivatives_kernel!(
    failure_storage,
    failure_values,
    values,
    gradients,
)
    proposal_slot = @index(Global, Linear)
    reason, value = _frozen_derivative_failure(values, gradients, proposal_slot)
    if !iszero(reason)
        @inbounds failure_values[proposal_slot] = value
        _record_native_failure!(failure_storage, proposal_slot, 0, reason)
    end
end

function _evaluate_frozen_gradients!(
    values,
    gradients,
    target,
    bound_gradient,
    locations,
    execution::_KernelExecution,
)
    backend = KernelAbstractions.get_backend(values)
    kernel = _evaluate_frozen_gradients_kernel!(backend)
    proposal_count = size(locations, 2)
    kernel(
        values,
        gradients,
        target,
        bound_gradient,
        locations;
        ndrange=proposal_count,
        workgroupsize=_first_order_gramis_workgroupsize(
            execution,
            backend,
            proposal_count,
        ),
    )
    KernelAbstractions.synchronize(backend)
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

_first_order_gramis_bound_gradient(
    method_state::_PreparedFirstOrderGRAMIS,
    ::_KernelExecution,
) = method_state.serial_gradient

function _evaluate_frozen_gradients!(
    method_state::_PreparedFirstOrderGRAMIS,
    target,
    execution::Union{_SerialCPUExecution,_ThreadedCPUExecution,_KernelExecution},
    ;
    device=nothing,
    failure_record=nothing,
    transfers=nothing,
)
    workspace = method_state.workspace
    device === nothing || fill!(
        failure_record.storage,
        zero(eltype(failure_record.storage)),
    )
    _evaluate_frozen_gradients!(
        workspace.frozen_values,
        workspace.gradients,
        target,
        _first_order_gramis_bound_gradient(method_state, execution),
        method_state.run.locations,
        execution,
    )
    device === nothing && return nothing
    backend = KernelAbstractions.get_backend(workspace.frozen_values)
    proposal_count = length(workspace.frozen_values)
    _validate_frozen_derivatives_kernel!(backend)(
        failure_record.storage,
        workspace.candidate_values,
        workspace.frozen_values,
        workspace.gradients;
        ndrange=proposal_count,
        workgroupsize=_first_order_gramis_workgroupsize(
            execution,
            backend,
            proposal_count,
        ),
    )
    KernelAbstractions.synchronize(backend)
    return _throw_first_order_gramis_device_derivative_failure(
        device,
        failure_record,
        workspace.candidate_values,
        transfers,
    )
end

function _precondition_gradients!(
    method_state::_PreparedFirstOrderGRAMIS,
    execution::Union{_SerialCPUExecution,_ThreadedCPUExecution,_KernelExecution},
    ;
    device=nothing,
    failure_record=nothing,
    transfers=nothing,
)
    workspace = method_state.workspace
    device === nothing || fill!(
        failure_record.storage,
        zero(eltype(failure_record.storage)),
    )
    _precondition_gradients!(
        workspace.moves,
        workspace.gradients,
        method_state.run.factors,
        execution,
    )
    device === nothing && return nothing
    backend = KernelAbstractions.get_backend(workspace.moves)
    proposal_count = size(workspace.moves, 2)
    _validate_preconditioned_moves_kernel!(backend)(
        failure_record.storage,
        workspace.candidate_values,
        workspace.moves;
        ndrange=proposal_count,
        workgroupsize=_first_order_gramis_workgroupsize(
            execution,
            backend,
            proposal_count,
        ),
    )
    KernelAbstractions.synchronize(backend)
    return _throw_first_order_gramis_device_derivative_failure(
        device,
        failure_record,
        workspace.candidate_values,
        transfers,
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

@kernel function _initialize_backtracking_kernel!(
    candidate_locations,
    candidate_values,
    active_mask,
    steps,
    trials,
    frozen_values,
    locations,
)
    proposal_slot = @index(Global, Linear)
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

@kernel function _backtracking_trial_kernel!(
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
)
    proposal_slot = @index(Global, Linear)
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

@kernel function _finish_backtracking_kernel!(
    candidate_locations,
    candidate_values,
    active_mask,
    frozen_values,
    locations,
)
    proposal_slot = @index(Global, Linear)
    _finish_backtracking_slot!(
        candidate_locations,
        candidate_values,
        active_mask,
        frozen_values,
        locations,
        proposal_slot,
    )
end

@kernel function _validate_backtracking_candidates_kernel!(
    failure_storage,
    candidate_values,
    trials,
    trial,
)
    proposal_slot = @index(Global, Linear)
    _, reason, _ = _backtracking_candidate_failure(
        candidate_values,
        trials,
        trial,
        proposal_slot,
    )
    if !iszero(reason)
        _record_native_failure!(failure_storage, proposal_slot, 0, reason)
    end
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
    execution::_KernelExecution;
    device=nothing,
    failure_record=nothing,
    transfers=nothing,
)
    backend = KernelAbstractions.get_backend(candidate_locations)
    proposal_count = size(locations, 2)
    workgroupsize = _first_order_gramis_workgroupsize(
        execution,
        backend,
        proposal_count,
    )
    initialize_kernel = _initialize_backtracking_kernel!(backend)
    initialize_kernel(
        candidate_locations,
        candidate_values,
        active_mask,
        steps,
        trials,
        frozen_values,
        locations;
        ndrange=proposal_count,
        workgroupsize,
    )
    trial_kernel = _backtracking_trial_kernel!(backend)
    for trial in 1:max_trials
        step = ldexp(one(eltype(steps)), 1 - trial)
        trial_kernel(
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
            trial;
            ndrange=proposal_count,
            workgroupsize,
        )
        if device !== nothing
            KernelAbstractions.synchronize(backend)
            fill!(
                failure_record.storage,
                zero(eltype(failure_record.storage)),
            )
            validation_kernel =
                _validate_backtracking_candidates_kernel!(backend)
            validation_kernel(
                failure_record.storage,
                candidate_values,
                trials,
                trial;
                ndrange=proposal_count,
                workgroupsize,
            )
            KernelAbstractions.synchronize(backend)
            _throw_first_order_gramis_device_derivative_failure(
                device,
                failure_record,
                candidate_values,
                transfers,
            )
        end
    end
    finish_kernel = _finish_backtracking_kernel!(backend)
    finish_kernel(
        candidate_locations,
        candidate_values,
        active_mask,
        frozen_values,
        locations;
        ndrange=proposal_count,
        workgroupsize,
    )
    KernelAbstractions.synchronize(backend)
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
        _validate_backtracking_candidates!(
            candidate_values,
            active_mask,
            trials,
            trial,
        ) || break
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
    execution::Union{_SerialCPUExecution,_ThreadedCPUExecution,_KernelExecution},
    ;
    device=nothing,
    failure_record=nothing,
    transfers=nothing,
)
    workspace = method_state.workspace
    arguments = (
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
    )
    execution isa _KernelExecution && return _backtrack_means!(
        arguments...,
        execution;
        device,
        failure_record,
        transfers,
    )
    return _backtrack_means!(arguments..., execution)
end

function _execute_first_order_gramis_live_preflight!(
    device::MLDataDevices.AbstractAcceleratorDevice,
    method_state::_PreparedFirstOrderGRAMIS,
    target,
    random_buffers,
)
    execution = _KernelExecution(_SerialCPUExecution())
    transfers = _ResultTransferCounter(0, 0)
    failure_record = random_buffers.failure_scratch.record
    _evaluate_frozen_gradients!(
        method_state,
        target,
        execution,
        ; device, failure_record, transfers,
    )
    _precondition_gradients!(
        method_state,
        execution,
        ; device, failure_record, transfers,
    )
    try
        _backtrack_means!(
            method_state,
            target,
            execution,
            ; device, failure_record, transfers,
        )
    finally
        copyto!(
            method_state.candidate.locations,
            method_state.run.locations,
        )
    end
    return nothing
end

const _GRAMIS_COVARIANCE_READY = UInt8(0)
const _GRAMIS_ALL_ZERO_LOCAL = UInt8(1)
const _GRAMIS_TEMPERING_FALLBACK = UInt8(2)
const _GRAMIS_REDUCTION_WORKGROUP_SIZE = 256
const _GRAMIS_REDUCTION_OFFSETS = (128, 64, 32, 16, 8, 4, 2, 1)

function _first_order_gramis_local_weight_arguments(
    method_state::_PreparedFirstOrderGRAMIS,
    round,
)
    workspace = method_state.workspace
    return (
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
        method_state.tempering_max_iterations,
    )
end

function _first_order_gramis_covariance_kernel_arguments(
    method_state::_PreparedFirstOrderGRAMIS,
    round,
)
    workspace = method_state.workspace
    bank = method_state.run
    return (
        _first_order_gramis_local_weight_arguments(method_state, round),
        (
            workspace.covariances,
            workspace.normalized_weights,
            workspace.tempering_powers,
            workspace.factor_status,
            workspace.samples,
            bank.locations,
            bank.factors,
            workspace.local_starts,
            method_state.plan.counts,
            round,
        ),
        (
            workspace.covariances,
            bank.factors,
            workspace.factor_status,
            method_state.covariance_rate,
            round,
            method_state.covariance_regularization,
        ),
    )
end

function _first_order_gramis_covariance_centre_arguments(
    method_state::_PreparedFirstOrderGRAMIS,
    round,
)
    workspace = method_state.workspace
    return (
        workspace.covariance_centres,
        workspace.normalized_weights,
        workspace.tempering_powers,
        workspace.factor_status,
        workspace.samples,
        method_state.run.locations,
        workspace.local_starts,
        method_state.plan.counts,
        round,
    )
end

function _first_order_gramis_accelerator_covariance_arguments(
    method_state::_PreparedFirstOrderGRAMIS,
    round,
)
    workspace = method_state.workspace
    return (
        workspace.covariances,
        workspace.covariance_centres,
        workspace.normalized_weights,
        workspace.factor_status,
        workspace.samples,
        method_state.run.factors,
        workspace.local_starts,
        method_state.plan.counts,
        round,
    )
end

@inline _first_order_gramis_factor_arguments(
    method_state,
    info=method_state.workspace.factor_info,
) = (
    method_state.candidate.factors,
    method_state.workspace.covariances,
    info,
    method_state.workspace.factor_status,
)

@inline function _gramis_local_group(starts, counts, proposal_slot, round)
    return @inbounds(starts[proposal_slot, round]),
    @inbounds(counts[proposal_slot, round])
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
    execution::Union{_SerialCPUExecution,_ThreadedCPUExecution},
)
    workspace = method_state.workspace
    backend = KernelAbstractions.get_backend(workspace.normalized_weights)
    proposal_count = size(method_state.run.locations, 2)
    kernel = _local_weight_summary_kernel!(backend)
    arguments = _first_order_gramis_local_weight_arguments(method_state, round)
    kernel(
        arguments[1:7]...,
        round;
        ndrange=proposal_count,
        workgroupsize=_first_order_gramis_workgroupsize(
            execution,
            backend,
            proposal_count,
        ),
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
    execution::Union{_SerialCPUExecution,_ThreadedCPUExecution},
)
    workspace = method_state.workspace
    backend = KernelAbstractions.get_backend(workspace.normalized_weights)
    proposal_count = size(method_state.run.locations, 2)
    kernel = _tempering_power_kernel!(backend)
    kernel(
        _first_order_gramis_local_weight_arguments(method_state, round)...;
        ndrange=proposal_count,
        workgroupsize=_first_order_gramis_workgroupsize(
            execution,
            backend,
            proposal_count,
        ),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function _cooperative_local_weights_kernel!(
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
    proposal_slot = @index(Group, Linear)
    lane_index = @index(Local, Linear)
    lane = @private Int (1,)
    @inbounds lane[1] = lane_index
    @uniform lane_count = @groupsize()[1]
    @uniform T = eltype(normalized_weights)
    maxima = @localmem eltype(normalized_weights) (
        _GRAMIS_REDUCTION_WORKGROUP_SIZE,
    )
    raw_maxima = @localmem eltype(local_logweights) (
        _GRAMIS_REDUCTION_WORKGROUP_SIZE,
    )
    totals = @localmem eltype(normalized_weights) (
        _GRAMIS_REDUCTION_WORKGROUP_SIZE,
    )
    squared_totals = @localmem eltype(normalized_weights) (
        _GRAMIS_REDUCTION_WORKGROUP_SIZE,
    )
    tempering_state = @localmem eltype(normalized_weights) (6,)
    group_state = @localmem eltype(starts) (3,)
    if @inbounds(lane[1]) == 1
        @inbounds group_state[1] = proposal_slot
        @inbounds group_state[2] = starts[proposal_slot, round]
        @inbounds group_state[3] =
            group_state[2] + counts[proposal_slot, round] - 1
    end
    @synchronize()

    lane_maximum = eltype(local_logweights)(-Inf)
    for sample_index in (@inbounds(group_state[2]) + @inbounds(lane[1]) - 1):lane_count:(@inbounds(group_state[3]))
        lane_maximum = max(
            lane_maximum,
            @inbounds(local_logweights[sample_index]),
        )
    end
    @inbounds raw_maxima[lane[1]] = lane_maximum
    @synchronize()
    for reduction_offset in _GRAMIS_REDUCTION_OFFSETS
        if @inbounds(lane[1]) <= reduction_offset
            @inbounds raw_maxima[lane[1]] = max(
                raw_maxima[lane[1]],
                raw_maxima[lane[1] + reduction_offset],
            )
        end
        @synchronize()
    end

    if @inbounds(lane[1]) == 1 &&
       @inbounds(raw_maxima[1]) == eltype(local_logweights)(-Inf)
        @inbounds local_ess[group_state[1]] = zero(T)
        @inbounds tempering_powers[group_state[1]] = zero(T)
        @inbounds status[group_state[1]] = _GRAMIS_ALL_ZERO_LOCAL
    end
    @synchronize()
    if @inbounds(raw_maxima[1]) != eltype(local_logweights)(-Inf)
        lane_total = zero(T)
        lane_squared_total = zero(T)
        for sample_index in (@inbounds(group_state[2]) + @inbounds(lane[1]) - 1):lane_count:(@inbounds(group_state[3]))
            weight = T(exp(
                @inbounds(local_logweights[sample_index]) -
                @inbounds(raw_maxima[1]),
            ))
            @inbounds normalized_weights[sample_index] = weight
            lane_total += weight
            lane_squared_total += abs2(weight)
        end
        @inbounds totals[lane[1]] = lane_total
        @inbounds squared_totals[lane[1]] = lane_squared_total
        @synchronize()
        for reduction_offset in _GRAMIS_REDUCTION_OFFSETS
            if @inbounds(lane[1]) <= reduction_offset
                @inbounds totals[lane[1]] += totals[lane[1] + reduction_offset]
                @inbounds squared_totals[lane[1]] +=
                    squared_totals[lane[1] + reduction_offset]
            end
            @synchronize()
        end

        for sample_index in (@inbounds(group_state[2]) + @inbounds(lane[1]) - 1):lane_count:(@inbounds(group_state[3]))
            @inbounds normalized_weights[sample_index] *= inv(totals[1])
        end
        if @inbounds(lane[1]) == 1
            @inbounds local_ess[group_state[1]] =
                abs2(totals[1]) / squared_totals[1]
            @inbounds tempering_powers[group_state[1]] = one(T)
            @inbounds status[group_state[1]] = _GRAMIS_COVARIANCE_READY
        end
        @synchronize()

        if @inbounds(local_ess[group_state[1]]) <
           T(@inbounds(thresholds[group_state[1], round]))
            if @inbounds(lane[1]) == 1
                @inbounds tempering_state[1] = zero(T)
                @inbounds tempering_state[2] = one(T)
                @inbounds tempering_state[3] = zero(T)
            end
            @synchronize()

            for _ in 1:max_iterations
                if @inbounds(lane[1]) == 1
                    @inbounds tempering_state[4] =
                        (tempering_state[1] + tempering_state[2]) / T(2)
                end
                @synchronize()
                lane_maximum = T(-Inf)
                for sample_index in (@inbounds(group_state[2]) + @inbounds(lane[1]) - 1):lane_count:(@inbounds(group_state[3]))
                    scaled = @inbounds(tempering_state[4]) * T(
                        @inbounds local_logweights[sample_index]
                    )
                    lane_maximum = max(lane_maximum, scaled)
                end
                @inbounds maxima[lane[1]] = lane_maximum
                @synchronize()
                for reduction_offset in _GRAMIS_REDUCTION_OFFSETS
                    if @inbounds(lane[1]) <= reduction_offset
                        @inbounds maxima[lane[1]] = max(
                            maxima[lane[1]],
                            maxima[lane[1] + reduction_offset],
                        )
                    end
                    @synchronize()
                end

                lane_total = zero(T)
                lane_squared_total = zero(T)
                for sample_index in (@inbounds(group_state[2]) + @inbounds(lane[1]) - 1):lane_count:(@inbounds(group_state[3]))
                    scaled = @inbounds(tempering_state[4]) * T(
                        @inbounds local_logweights[sample_index]
                    )
                    shifted_weight = exp(scaled - @inbounds(maxima[1]))
                    lane_total += shifted_weight
                    lane_squared_total += abs2(shifted_weight)
                end
                @inbounds totals[lane[1]] = lane_total
                @inbounds squared_totals[lane[1]] = lane_squared_total
                @synchronize()
                for reduction_offset in _GRAMIS_REDUCTION_OFFSETS
                    if @inbounds(lane[1]) <= reduction_offset
                        @inbounds totals[lane[1]] +=
                            totals[lane[1] + reduction_offset]
                        @inbounds squared_totals[lane[1]] +=
                            squared_totals[lane[1] + reduction_offset]
                    end
                    @synchronize()
                end

                if @inbounds(lane[1]) == 1
                    ess = abs2(@inbounds(totals[1])) /
                          @inbounds(squared_totals[1])
                    if ess >= T(@inbounds(thresholds[group_state[1], round]))
                        @inbounds tempering_state[1] = tempering_state[4]
                        @inbounds tempering_state[3] = ess
                        @inbounds tempering_state[5] = maxima[1]
                        @inbounds tempering_state[6] = totals[1]
                    else
                        @inbounds tempering_state[2] = tempering_state[4]
                    end
                end
                @synchronize()
                @inbounds(tempering_state[2]) -
                @inbounds(tempering_state[1]) <= tolerance && break
            end

            if @inbounds(tempering_state[1]) > zero(T)
                for sample_index in (@inbounds(group_state[2]) + @inbounds(lane[1]) - 1):lane_count:(@inbounds(group_state[3]))
                    scaled = @inbounds(tempering_state[1]) * T(
                        @inbounds local_logweights[sample_index]
                    )
                    @inbounds normalized_weights[sample_index] = exp(
                        scaled - @inbounds(tempering_state[5]),
                    ) * inv(@inbounds(tempering_state[6]))
                end
                if @inbounds(lane[1]) == 1
                    @inbounds local_ess[group_state[1]] = tempering_state[3]
                    @inbounds tempering_powers[group_state[1]] =
                        tempering_state[1]
                end
            elseif @inbounds(lane[1]) == 1
                @inbounds tempering_powers[group_state[1]] = zero(T)
                @inbounds status[group_state[1]] = _GRAMIS_TEMPERING_FALLBACK
            end
            @synchronize()
        end
    end
end

function _cooperative_local_weights!(
    method_state::_PreparedFirstOrderGRAMIS,
    round,
    ::_KernelExecution,
)
    workspace = method_state.workspace
    backend = KernelAbstractions.get_backend(workspace.normalized_weights)
    proposal_count = size(method_state.run.locations, 2)
    kernel = _cooperative_local_weights_kernel!(
        backend,
        _GRAMIS_REDUCTION_WORKGROUP_SIZE,
    )
    kernel(
        _first_order_gramis_local_weight_arguments(method_state, round)...;
        ndrange=_GRAMIS_REDUCTION_WORKGROUP_SIZE * proposal_count,
        workgroupsize=_GRAMIS_REDUCTION_WORKGROUP_SIZE,
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

@kernel function _fit_accelerator_covariance_centres_kernel!(
    covariance_centres,
    normalized_weights,
    tempering_powers,
    status,
    samples,
    locations,
    starts,
    counts,
    round,
)
    group_index = @index(Group, Linear)
    lane_index = @index(Local, Linear)
    lane = @private Int (1,)
    @inbounds lane[1] = lane_index
    @uniform lane_count = @groupsize()[1]
    @uniform T = eltype(covariance_centres)
    partial_centres = @localmem eltype(covariance_centres) (
        _GRAMIS_REDUCTION_WORKGROUP_SIZE,
    )
    group_state = @localmem eltype(starts) (5,)
    if @inbounds(lane[1]) == 1
        dimension = size(samples, 1)
        proposal_slot = (group_index - 1) ÷ dimension + 1
        @inbounds group_state[1] = (group_index - 1) % dimension + 1
        @inbounds group_state[2] = proposal_slot
        @inbounds group_state[3] = starts[proposal_slot, round]
        @inbounds group_state[4] =
            group_state[3] + counts[proposal_slot, round] - 1
        @inbounds group_state[5] =
            status[proposal_slot] == _GRAMIS_COVARIANCE_READY &&
            tempering_powers[proposal_slot] < one(T)
    end
    @synchronize()

    partial = zero(T)
    if @inbounds(group_state[5]) == 1
        for sample_index in (@inbounds(group_state[3]) + @inbounds(lane[1]) - 1):lane_count:(@inbounds(group_state[4]))
            partial += @inbounds(normalized_weights[sample_index]) *
                       @inbounds(samples[group_state[1], sample_index])
        end
    end
    @inbounds partial_centres[lane[1]] = partial
    @synchronize()
    for reduction_offset in _GRAMIS_REDUCTION_OFFSETS
        if @inbounds(lane[1]) <= reduction_offset
            @inbounds partial_centres[lane[1]] +=
                partial_centres[lane[1] + reduction_offset]
        end
        @synchronize()
    end
    if @inbounds(lane[1]) == 1
        if @inbounds(group_state[5]) == 1
            @inbounds covariance_centres[group_state[1], group_state[2]] =
                partial_centres[1]
        else
            @inbounds covariance_centres[group_state[1], group_state[2]] =
                locations[group_state[1], group_state[2]]
        end
    end
end

@kernel function _fit_accelerator_covariances_kernel!(
    covariances,
    covariance_centres,
    normalized_weights,
    status,
    samples,
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
    if column <= row
        if @inbounds(status[proposal_slot]) == _GRAMIS_COVARIANCE_READY
            first_sample, sample_count = _gramis_local_group(
                starts,
                counts,
                proposal_slot,
                round,
            )
            last_sample = first_sample + sample_count - 1
            center_row = @inbounds covariance_centres[row, proposal_slot]
            center_column = @inbounds covariance_centres[column, proposal_slot]
            T = eltype(covariances)
            covariance = zero(T)
            for sample_index in first_sample:last_sample
                weight = @inbounds normalized_weights[sample_index]
                centered_row = @inbounds(samples[row, sample_index]) - center_row
                centered_column =
                    @inbounds(samples[column, sample_index]) - center_column
                covariance += weight * centered_row * centered_column
            end
        else
            covariance =
                _gramis_current_covariance(factors, row, column, proposal_slot)
        end
        @inbounds covariances[row, column, proposal_slot] = covariance
        @inbounds covariances[column, row, proposal_slot] = covariance
    end
end

function _fit_accelerator_covariance_centres!(
    method_state::_PreparedFirstOrderGRAMIS,
    round,
    ::_KernelExecution,
)
    workspace = method_state.workspace
    backend = KernelAbstractions.get_backend(workspace.covariance_centres)
    pair_count = length(workspace.covariance_centres)
    kernel = _fit_accelerator_covariance_centres_kernel!(
        backend,
        _GRAMIS_REDUCTION_WORKGROUP_SIZE,
    )
    kernel(
        _first_order_gramis_covariance_centre_arguments(method_state, round)...;
        ndrange=_GRAMIS_REDUCTION_WORKGROUP_SIZE * pair_count,
        workgroupsize=_GRAMIS_REDUCTION_WORKGROUP_SIZE,
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

function _fit_accelerator_covariances!(
    method_state::_PreparedFirstOrderGRAMIS,
    round,
    execution::_KernelExecution,
)
    workspace = method_state.workspace
    backend = KernelAbstractions.get_backend(workspace.covariances)
    covariance_entries = length(workspace.covariances)
    kernel = _fit_accelerator_covariances_kernel!(backend)
    kernel(
        _first_order_gramis_accelerator_covariance_arguments(
            method_state,
            round,
        )...;
        ndrange=covariance_entries,
        workgroupsize=_first_order_gramis_workgroupsize(
            execution,
            backend,
            covariance_entries,
        ),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

function _prepare_local_covariance_weights!(
    method_state::_PreparedFirstOrderGRAMIS,
    round,
    execution::Union{_SerialCPUExecution,_ThreadedCPUExecution},
)
    _local_weight_summary!(method_state, round, execution)
    _tempering_power!(method_state, round, execution)
    return nothing
end

function _prepare_local_covariance_weights!(
    method_state::_PreparedFirstOrderGRAMIS,
    round,
    execution::_KernelExecution,
)
    return _cooperative_local_weights!(method_state, round, execution)
end

function _fit_local_covariances!(
    method_state::_PreparedFirstOrderGRAMIS,
    round,
    execution::Union{_SerialCPUExecution,_ThreadedCPUExecution},
)
    _prepare_local_covariance_weights!(method_state, round, execution)

    workspace = method_state.workspace
    backend = KernelAbstractions.get_backend(workspace.covariances)
    covariance_entries = length(workspace.covariances)
    kernel = _fit_local_covariances_kernel!(backend)
    kernel(
        _first_order_gramis_covariance_kernel_arguments(
            method_state,
            round,
        )[2]...;
        ndrange=covariance_entries,
        workgroupsize=_first_order_gramis_workgroupsize(
            execution,
            backend,
            covariance_entries,
        ),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

function _fit_local_covariances!(
    method_state::_PreparedFirstOrderGRAMIS,
    round,
    execution::_KernelExecution,
)
    _prepare_local_covariance_weights!(method_state, round, execution)
    _fit_accelerator_covariance_centres!(method_state, round, execution)
    return _fit_accelerator_covariances!(method_state, round, execution)
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
        _first_order_gramis_covariance_kernel_arguments(
            method_state,
            round,
        )[3]...;
        ndrange=proposal_count,
        workgroupsize=_first_order_gramis_workgroupsize(
            execution,
            backend,
            proposal_count,
        ),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

const _GRAMIS_COVARIANCE_NONFINITE_INFO = Int32(-1)
const _GRAMIS_CHOLESKY_WORKGROUP_SIZE = 64

@kernel function _factor_population_kernel!(factors, covariances, info, status)
    proposal_slot = @index(Group, Linear)
    lane = @index(Local, Linear)
    @uniform lane_count = @groupsize()[1]
    @uniform dimension = size(factors, 1)
    lane_bad = @localmem Int32 (_GRAMIS_CHOLESKY_WORKGROUP_SIZE,)

    lane == 1 && (@inbounds info[proposal_slot] = Int32(0))
    bad = Int32(0)
    for entry in lane:lane_count:(dimension * dimension)
        row = (entry - 1) % dimension + 1
        column = (entry - 1) ÷ dimension + 1
        if @inbounds(status[proposal_slot]) == _GRAMIS_COVARIANCE_READY
            value = @inbounds covariances[row, column, proposal_slot]
            bad |= isfinite(value) ? Int32(0) : Int32(1)
            @inbounds factors[row, column, proposal_slot] =
                row < column ? zero(eltype(factors)) : value
        end
    end
    @inbounds lane_bad[lane] = bad
    @synchronize()
    if lane == 1
        group_bad = @inbounds lane_bad[1]
        for other_lane in 2:lane_count
            group_bad |= @inbounds lane_bad[other_lane]
        end
        iszero(group_bad) || (@inbounds info[proposal_slot] =
            _GRAMIS_COVARIANCE_NONFINITE_INFO)
    end
    @synchronize()

    for column in 1:dimension
        if lane == 1 &&
           @inbounds(status[proposal_slot]) == _GRAMIS_COVARIANCE_READY &&
           iszero(@inbounds(info[proposal_slot]))
            pivot = @inbounds factors[column, column, proposal_slot]
            for previous in 1:(column - 1)
                pivot -= abs2(
                    @inbounds factors[column, previous, proposal_slot]
                )
            end
            if isfinite(pivot) && pivot > zero(pivot)
                @inbounds factors[column, column, proposal_slot] = sqrt(pivot)
            else
                @inbounds info[proposal_slot] = Int32(column)
            end
        end
        @synchronize()

        bad = Int32(0)
        if @inbounds(status[proposal_slot]) == _GRAMIS_COVARIANCE_READY &&
           iszero(@inbounds(info[proposal_slot]))
            diagonal = @inbounds factors[column, column, proposal_slot]
            for row in (column + lane):lane_count:dimension
                value = @inbounds factors[row, column, proposal_slot]
                for previous in 1:(column - 1)
                    value -= @inbounds(
                        factors[row, previous, proposal_slot] *
                        factors[column, previous, proposal_slot]
                    )
                end
                value /= diagonal
                @inbounds factors[row, column, proposal_slot] = value
                bad |= isfinite(value) ? Int32(0) : Int32(1)
            end
        end
        @inbounds lane_bad[lane] = bad
        @synchronize()
        if lane == 1 &&
           @inbounds(status[proposal_slot]) == _GRAMIS_COVARIANCE_READY &&
           iszero(@inbounds(info[proposal_slot]))
            group_bad = @inbounds lane_bad[1]
            for other_lane in 2:lane_count
                group_bad |= @inbounds lane_bad[other_lane]
            end
            iszero(group_bad) ||
                @inbounds(info[proposal_slot] = Int32(column))
        end
        @synchronize()
    end
end

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
    info::StridedVector{I},
    ::_SerialCPUExecution,
) where {T<:Union{Float32,Float64},I<:Signed}
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
    info::StridedVector{I},
    ::_ThreadedCPUExecution,
) where {T<:Union{Float32,Float64},I<:Signed}
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
    info::StridedVector{I},
    execution,
) where {I<:Signed}
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

@kernel function _record_first_order_gramis_factor_failures!(
    failure_storage,
    info,
)
    proposal_slot = @index(Global, Linear)
    iszero(@inbounds(info[proposal_slot])) || _record_native_failure!(
        failure_storage,
        proposal_slot,
        0,
        _GRAMIS_COVARIANCE_FACTORIZATION_FAILED,
    )
end

function _update_local_covariances!(
    device::MLDataDevices.AbstractAcceleratorDevice,
    method_state::_PreparedFirstOrderGRAMIS,
    round,
    info,
    execution::_KernelExecution,
    failure_record,
    transfers,
)
    copyto!(method_state.candidate.factors, method_state.run.factors)
    _blend_local_covariances!(method_state, round, execution)
    workspace = method_state.workspace
    _factor_population!(
        device,
        _first_order_gramis_factor_arguments(method_state, info)...,
    )
    fill!(failure_record.storage, zero(eltype(failure_record.storage)))
    backend = KernelAbstractions.get_backend(info)
    kernel = _record_first_order_gramis_factor_failures!(backend)
    kernel(
        failure_record.storage,
        info;
        ndrange=length(info),
        workgroupsize=_first_order_gramis_workgroupsize(
            execution,
            backend,
            length(info),
        ),
    )
    KernelAbstractions.synchronize(backend)
    failure = _first_order_gramis_failure_snapshot!(transfers, failure_record)
    iszero(failure.count) || copyto!(
        method_state.candidate.factors,
        method_state.run.factors,
    )
    return failure
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

macro _capture_first_order_gramis_round(
    method_state,
    transfers,
    round,
    phase,
    completed_rounds,
    expression,
)
    return quote
        try
            $(esc(expression))
        catch cause
            cause isa FirstOrderGRAMISRoundError && rethrow()
            failure_phase = _first_order_gramis_round_phase(
                cause,
                $(esc(phase)),
            )
            details = _first_order_gramis_failure_details(cause)
            throw(
                FirstOrderGRAMISRoundError(
                    $(esc(round)),
                    failure_phase,
                    cause,
                    (
                        round_size=$(esc(method_state)).plan.schedule[$(esc(round))],
                        completed_rounds=$(esc(completed_rounds)),
                        covariance=details.covariance,
                        derivative=details.derivative,
                        transfers=$(esc(transfers)),
                        pre_call_state_preserved=true,
                    ),
                ),
            )
        end
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

@kernel function _add_first_order_gramis_repulsion_kernel!(
    failure_storage,
    failure_values,
    candidate,
    repulsion,
)
    proposal_slot = @index(Global, Linear)
    @inbounds for row in axes(candidate, 1)
        value = candidate[row, proposal_slot] + repulsion[row, proposal_slot]
        candidate[row, proposal_slot] = value
        if !isfinite(value)
            failure_values[proposal_slot] = value
            _record_native_failure!(
                failure_storage,
                proposal_slot,
                0,
                _GRAMIS_LOCATION_NONFINITE,
            )
            break
        end
    end
end

function _throw_first_order_gramis_device_proposal_failure(
    device,
    failure_record,
    failure_values,
    transfers,
)
    failure = _first_order_gramis_failure_snapshot!(transfers, failure_record)
    iszero(failure.count) && return nothing
    slot = failure.first_logical_index
    value = _first_order_gramis_failure_value(
        device,
        failure_values,
        slot,
        transfers,
    )
    throw(
        _FirstOrderGRAMISProposalError(
            slot,
            _first_order_gramis_proposal_reason(failure.reason_bits),
            value,
        ),
    )
end

function _first_order_gramis_proposal_reason(reason)
    reason == _GRAMIS_LOCATION_NONFINITE && return :location_nonfinite
    reason == _GRAMIS_FACTOR_NONFINITE && return :factor_nonfinite
    reason == _GRAMIS_FACTOR_DIAGONAL_INVALID &&
        return :factor_diagonal_invalid
    reason == _GRAMIS_LOGNORMALIZER_NONFINITE &&
        return :lognormalizer_nonfinite
    error("unknown FirstOrderGRAMIS proposal failure code")
end

function _add_first_order_gramis_repulsion!(
    device::MLDataDevices.AbstractAcceleratorDevice,
    candidate,
    repulsion,
    transfers,
    execution::_KernelExecution,
    failure_record,
    failure_values,
)
    fill!(failure_record.storage, zero(eltype(failure_record.storage)))
    backend = KernelAbstractions.get_backend(candidate)
    kernel = _add_first_order_gramis_repulsion_kernel!(backend)
    proposal_count = size(candidate, 2)
    kernel(
        failure_record.storage,
        failure_values,
        candidate,
        repulsion;
        ndrange=proposal_count,
        workgroupsize=_first_order_gramis_workgroupsize(
            execution,
            backend,
            proposal_count,
        ),
    )
    KernelAbstractions.synchronize(backend)
    return _throw_first_order_gramis_device_proposal_failure(
        device,
        failure_record,
        failure_values,
        transfers,
    )
end

@inline function _candidate_factor_validation(factors, status, proposal_slot)
    T = eltype(factors)
    @inbounds(status[proposal_slot]) == _GRAMIS_COVARIANCE_READY ||
        return (false, UInt16(0), zero(T), zero(T))
    dimension = size(factors, 1)
    logabsdet = zero(T)
    value = zero(T)
    @inbounds for column in axes(factors, 2), row in axes(factors, 1)
        value = factors[row, column, proposal_slot]
        isfinite(value) ||
            return (true, _GRAMIS_FACTOR_NONFINITE, value, zero(T))
        if row == column
            value > zero(T) || return (
                true,
                _GRAMIS_FACTOR_DIAGONAL_INVALID,
                value,
                zero(T),
            )
            logabsdet += log(value)
        end
    end
    lognormalizer = _gaussian_lognormalizer(T, dimension, logabsdet)
    reason = isfinite(lognormalizer) ?
             UInt16(0) : _GRAMIS_LOGNORMALIZER_NONFINITE
    return (true, reason, lognormalizer, lognormalizer)
end

function _validate_first_order_gramis_candidate_factors!(
    ::MLDataDevices.AbstractCPUDevice,
    candidate,
    status,
    transfers,
    execution,
)
    @inbounds for proposal_slot in axes(candidate.factors, 3)
        ready, reason, value, lognormalizer = _candidate_factor_validation(
            candidate.factors,
            status,
            proposal_slot,
        )
        ready || continue
        iszero(reason) || throw(
            _FirstOrderGRAMISProposalError(
                proposal_slot,
                _first_order_gramis_proposal_reason(reason),
                value,
            ),
        )
        candidate.lognormalizers[proposal_slot] = lognormalizer
    end
    return nothing
end

@kernel function _validate_first_order_gramis_candidate_factors_kernel!(
    failure_storage,
    failure_values,
    factors,
    lognormalizers,
    status,
)
    proposal_slot = @index(Global, Linear)
    ready, reason, value, lognormalizer = _candidate_factor_validation(
        factors,
        status,
        proposal_slot,
    )
    if ready
        if !iszero(reason)
            @inbounds failure_values[proposal_slot] = value
            _record_native_failure!(failure_storage, proposal_slot, 0, reason)
        else
            @inbounds lognormalizers[proposal_slot] = lognormalizer
        end
    end
end

function _validate_first_order_gramis_candidate_factors!(
    device::MLDataDevices.AbstractAcceleratorDevice,
    candidate,
    status,
    transfers,
    execution::_KernelExecution,
    failure_record,
    failure_values,
)
    fill!(failure_record.storage, zero(eltype(failure_record.storage)))
    backend = KernelAbstractions.get_backend(candidate.factors)
    kernel = _validate_first_order_gramis_candidate_factors_kernel!(backend)
    proposal_count = size(candidate.factors, 3)
    kernel(
        failure_record.storage,
        failure_values,
        candidate.factors,
        candidate.lognormalizers,
        status;
        ndrange=proposal_count,
        workgroupsize=_first_order_gramis_workgroupsize(
            execution,
            backend,
            proposal_count,
        ),
    )
    KernelAbstractions.synchronize(backend)
    return _throw_first_order_gramis_device_proposal_failure(
        device,
        failure_record,
        failure_values,
        transfers,
    )
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

function _throw_first_order_gramis_covariance_failure(
    device::MLDataDevices.AbstractAcceleratorDevice,
    info,
    failure,
    transfers,
    execution::_KernelExecution,
)
    iszero(failure.count) && return nothing
    slot = failure.first_logical_index
    factor_info = _first_order_gramis_failure_value(
        device,
        info,
        slot,
        transfers,
    )
    throw(_FirstOrderGRAMISCovarianceError(slot, Int(factor_info)))
end

@inline function _minimum_first_order_gramis_whitened_distance_value(
    whitened_means,
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

@kernel function _minimum_first_order_gramis_whitened_distance_kernel!(
    output,
    whitened_means,
)
    @inbounds output[1] =
        _minimum_first_order_gramis_whitened_distance_value(whitened_means)
end

function _minimum_first_order_gramis_whitened_distance(
    ::MLDataDevices.AbstractCPUDevice,
    whitened_means,
    transfers,
    execution,
)
    return _minimum_first_order_gramis_whitened_distance_value(whitened_means)
end

function _minimum_first_order_gramis_whitened_distance(
    device::MLDataDevices.AbstractAcceleratorDevice,
    whitened_means,
    transfers,
    execution::_KernelExecution,
    output,
)
    backend = KernelAbstractions.get_backend(whitened_means)
    kernel = _minimum_first_order_gramis_whitened_distance_kernel!(backend)
    kernel(output, whitened_means; ndrange=1, workgroupsize=1)
    KernelAbstractions.synchronize(backend)
    return _first_order_gramis_failure_value(device, output, 1, transfers)
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
    cpu_execution = threaded ? _ThreadedCPUExecution() : _SerialCPUExecution()
    execution = sampler.device isa MLDataDevices.AbstractAcceleratorDevice ?
                _KernelExecution(cpu_execution) : cpu_execution
    accelerator_device = execution isa _KernelExecution ? sampler.device : nothing
    plan = method_state.plan
    workspace = method_state.workspace
    buffers = sampler.random_buffers
    rounds = length(plan.schedule)
    total_samples = last(plan.offsets) - 1
    dimension, proposal_count = size(method_state.committed.locations)
    T = eltype(method_state.committed.locations)
    L = eltype(workspace.round_logweights)
    transfers = _ResultTransferCounter(0, 0)

    storage = @_capture_first_order_gramis_round(
        method_state,
        transfers,
        1,
        :result_construction,
        0,
        begin
            allocated_active_rounds = collect(
                Int,
                method_state.active_repulsion_rounds,
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
        end,
    )
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

    @_capture_first_order_gramis_round(
        method_state,
        transfers,
        1,
        :proposal,
        0,
        _copy_first_order_gramis_population!(
            method_state.run,
            method_state.committed,
        ),
    )
    target = @_capture_first_order_gramis_round(
        method_state,
        transfers,
        1,
        :target,
        0,
        begin
            _bind_resolved_target(
                sampler.target,
                view(method_state.run.locations, :, 1),
            )
        end,
    )
    target_evaluator, target_failures = @_capture_first_order_gramis_round(
        method_state,
        transfers,
        1,
        :target,
        0,
        begin
            _native_target_evaluator(
                KernelAbstractions.get_backend(buffers.normal),
                target,
                L,
                buffers.failure_scratch.target_failures,
            )
        end,
    )

    for round in eachindex(plan.schedule)
        views = _first_order_gramis_round_views(method_state, round)
        output_indices = plan.offsets[round]:(plan.offsets[round + 1] - 1)
        denominator = _RealizedMixtureDenominator(plan.logcoefficients, round)
        normal_buffer = view(
            buffers.normal,
            1:(dimension * views.round_size),
        )

        @_capture_first_order_gramis_round(
            method_state,
            transfers,
            round,
            :sampling,
            round - 1,
            begin
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
            end,
        )
        @_capture_first_order_gramis_round(
            method_state,
            transfers,
            round,
            :covariance,
            round - 1,
            _fit_local_covariances!(method_state, round, execution),
        )
        @_capture_first_order_gramis_round(
            method_state,
            transfers,
            round,
            :derivative,
            round - 1,
            begin
                _evaluate_frozen_gradients!(
                    method_state,
                    target,
                    execution,
                    ;
                    device=accelerator_device,
                    failure_record=buffers.failure_scratch.record,
                    transfers,
                )
                _precondition_gradients!(
                    method_state,
                    execution,
                    ;
                    device=accelerator_device,
                    failure_record=buffers.failure_scratch.record,
                    transfers,
                )
            end,
        )
        @_capture_first_order_gramis_round(
            method_state,
            transfers,
            round,
            :repulsion,
            round - 1,
            begin
                if round in method_state.active_repulsion_rounds
                    repulsion_arguments = (
                        workspace.repulsion,
                        workspace.collision_counts,
                        workspace.pooled_covariance,
                        workspace.whitened_means,
                        method_state.run.locations,
                        method_state.run.factors,
                        method_state.repulsion_strength,
                        round,
                        method_state.repulsion_softening,
                    )
                    if accelerator_device === nothing
                        _repulsion!(repulsion_arguments..., execution)
                    else
                        _repulsion!(
                            repulsion_arguments...,
                            execution;
                            device=accelerator_device,
                            failure_record=buffers.failure_scratch.record,
                            failure_values=workspace.candidate_values,
                            transfers,
                        )
                    end
                    active_repulsion_position[] += 1
                    distance_arguments = (
                        sampler.device,
                        workspace.whitened_means,
                        transfers,
                        execution,
                    )
                    minimum_whitened_distances[active_repulsion_position[]] =
                        accelerator_device === nothing ?
                        _minimum_first_order_gramis_whitened_distance(
                            distance_arguments...,
                        ) : _minimum_first_order_gramis_whitened_distance(
                            distance_arguments...,
                            workspace.candidate_values,
                        )
                else
                    fill!(workspace.repulsion, zero(T))
                    fill!(
                        workspace.collision_counts,
                        _GRAMIS_COLLISIONS_UNAVAILABLE,
                    )
                end
            end,
        )
        @_capture_first_order_gramis_round(
            method_state,
            transfers,
            round,
            :derivative,
            round - 1,
            begin
                _backtrack_means!(
                    method_state,
                    target,
                    execution,
                    ;
                    device=accelerator_device,
                    failure_record=buffers.failure_scratch.record,
                    transfers,
                )
                repulsion_add_arguments = (
                    sampler.device,
                    method_state.candidate.locations,
                    workspace.repulsion,
                    transfers,
                    execution,
                )
                accelerator_device === nothing ?
                _add_first_order_gramis_repulsion!(repulsion_add_arguments...) :
                _add_first_order_gramis_repulsion!(
                    repulsion_add_arguments...,
                    buffers.failure_scratch.record,
                    workspace.candidate_values,
                )
            end,
        )
        @_capture_first_order_gramis_round(
            method_state,
            transfers,
            round,
            :covariance,
            round - 1,
            begin
                copyto!(
                    method_state.candidate.lognormalizers,
                    method_state.run.lognormalizers,
                )
                covariance_arguments = (
                    sampler.device,
                    method_state,
                    round,
                    workspace.factor_info,
                    execution,
                )
                factor_failure = accelerator_device === nothing ?
                                 _update_local_covariances!(
                    covariance_arguments...,
                ) : _update_local_covariances!(
                    covariance_arguments...,
                    buffers.failure_scratch.record,
                    transfers,
                )
                covariance_failure_arguments = (
                    sampler.device,
                    workspace.factor_info,
                    transfers,
                    execution,
                )
                accelerator_device === nothing ?
                _throw_first_order_gramis_covariance_failure(
                    covariance_failure_arguments...,
                ) : _throw_first_order_gramis_covariance_failure(
                    sampler.device,
                    workspace.factor_info,
                    factor_failure,
                    transfers,
                    execution,
                )
                factor_validation_arguments = (
                    sampler.device,
                    method_state.candidate,
                    workspace.factor_status,
                    transfers,
                    execution,
                )
                accelerator_device === nothing ?
                _validate_first_order_gramis_candidate_factors!(
                    factor_validation_arguments...,
                ) : _validate_first_order_gramis_candidate_factors!(
                    factor_validation_arguments...,
                    buffers.failure_scratch.record,
                    workspace.candidate_values,
                )
            end,
        )
        @_capture_first_order_gramis_round(
            method_state,
            transfers,
            round,
            :proposal,
            round - 1,
            begin
                method_state.run, method_state.candidate =
                    method_state.candidate, method_state.run
            end,
        )
        summary = @_capture_first_order_gramis_round(
            method_state,
            transfers,
            round,
            :diagnostics,
            round,
            begin
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
            end,
        )
        round_ess[round] = summary.weights.ess
        round_lognormalizers[round] = summary.weights.lognormalizer
        fallback_all_zero[] += summary.bounded.all_zero
        fallback_tempering[] += summary.bounded.tempering
        fallback_backtracking[] += summary.bounded.backtracking
        target_trial_evaluations[] += summary.bounded.target_trials
    end

    diagnostics = @_capture_first_order_gramis_round(
        method_state,
        transfers,
        rounds,
        :result_construction,
        rounds,
        begin
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
        end,
    )
    result = @_capture_first_order_gramis_round(
        method_state,
        transfers,
        rounds,
        :result_construction,
        rounds,
        begin
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
        end,
    )
    method_state.committed, method_state.run =
        method_state.run, method_state.committed
    return result
end
