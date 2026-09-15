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
        workgroupsize=_population_workgroupsize(
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

function _throw_first_order_gramis_device_backtracking_failure(
    device,
    failure_record,
    failure_values,
    proposal_count,
    transfers,
)
    failure = _first_order_gramis_failure_snapshot!(transfers, failure_record)
    iszero(failure.count) && return nothing
    slot = mod1(failure.first_logical_index, proposal_count)
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
    family=GaussianFamily(),
)
    dimension = size(pooled_covariance, 1)
    row = (entry - 1) % dimension + 1
    column = (entry - 1) ÷ dimension + 1
    proposal_count = size(factors, 3)
    T = eltype(pooled_covariance)
    covariance = zero(T)
    for proposal_slot in axes(factors, 3)
        covariance += _population_factor_covariance(
            factors,
            row,
            column,
            proposal_slot,
        ) * _covariance_multiplier(_radial_family_at(family, proposal_slot), T)
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
    _threaded_foreach(1:length(pooled_covariance)) do entry
        _pooled_covariance_entry!(pooled_covariance, factors, entry)
    end
    return nothing
end

_pooled_covariance!(covariance, factors, execution, ::GaussianFamily) =
    _pooled_covariance!(covariance, factors, execution)
@kernel function _pooled_radial_covariance_kernel!(covariance, factors, family)
    entry = @index(Global, Linear)
    _pooled_covariance_entry!(covariance, factors, entry, family)
end
function _pooled_covariance!(covariance, factors, execution, family::_PackedStudentTFamily)
    backend = KernelAbstractions.get_backend(covariance)
    _pooled_radial_covariance_kernel!(backend)(covariance, factors, family;
        ndrange=length(covariance),
        workgroupsize=_native_workgroupsize(execution, length(covariance)))
    KernelAbstractions.synchronize(backend)
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
        workgroupsize=_population_workgroupsize(
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
    _threaded_foreach(axes(means, 2)) do proposal_slot
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
    execution::Union{_SerialCPUExecution,_ThreadedCPUExecution};
    family=GaussianFamily(),
) where {T}
    if iszero(strength)
        fill!(repulsion, zero(eltype(repulsion)))
        fill!(collision_counts, _GRAMIS_COLLISIONS_UNAVAILABLE)
        return nothing
    end

    _pooled_covariance!(pooled_covariance, factors, execution, family)
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
    execution::Union{_SerialCPUExecution,_ThreadedCPUExecution};
    family=GaussianFamily(),
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
        execution;
        family,
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
    family=GaussianFamily(),
    device=nothing,
    failure_record=nothing,
    failure_values=nothing,
    transfers=nothing,
    backend_execution=nothing,
)
    _launch_gramis_pool!(backend_execution, pooled_covariance, factors, execution, family, device, failure_record)
    device === nothing || _throw_first_order_gramis_device_repulsion_failure(
        device, failure_record, pooled_covariance, transfers)
    _factor_gramis_pool!(backend_execution, pooled_covariance)
    _launch_gramis_force!(backend_execution, repulsion, collision_counts, pooled_covariance,
        whitened_means, means, strength, round, softening, execution, device, failure_record, failure_values)
    device === nothing && return nothing
    return _throw_first_order_gramis_device_repulsion_failure(
        device, failure_record, failure_values, transfers)
end

_factor_gramis_pool!(::Nothing, pooled_covariance) = _factor_pooled_covariance!(pooled_covariance)

function _launch_gramis_pool!(::Nothing, pooled_covariance, factors, execution, family, device, failure_record)
    _pooled_covariance!(pooled_covariance, factors, execution, family)
    backend = KernelAbstractions.get_backend(pooled_covariance)
    if device !== nothing
        fill!(failure_record.storage, zero(eltype(failure_record.storage)))
        pooled_validation = _validate_pooled_covariance_kernel!(backend)
        pooled_validation(
            failure_record.storage,
            pooled_covariance;
            ndrange=length(pooled_covariance),
            workgroupsize=_population_workgroupsize(
                execution,
                backend,
                length(pooled_covariance),
            ),
        )
        KernelAbstractions.synchronize(backend)
    end
    return nothing
end

function _launch_gramis_force!(::Nothing, repulsion, collision_counts, pooled_covariance,
    whitened_means, means, strength, round, softening, execution, device, failure_record, failure_values)
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
    backend = KernelAbstractions.get_backend(pooled_covariance)
    fill!(failure_record.storage, zero(eltype(failure_record.storage)))
    force_validation = _validate_repulsion_kernel!(backend)
    proposal_count = size(repulsion, 2)
    force_validation(
        failure_record.storage,
        failure_values,
        repulsion;
        ndrange=proposal_count,
        workgroupsize=_population_workgroupsize(
            execution,
            backend,
            proposal_count,
        ),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
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
    family,
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
        @inbounds moves[row, proposal_slot] = move *
            _covariance_multiplier(_radial_family_at(family, proposal_slot), T)
    end
    return nothing
end

function _precondition_gradients!(
    moves,
    gradients,
    factors,
    ::_SerialCPUExecution,
    family=GaussianFamily(),
)
    @inbounds for proposal_slot in axes(gradients, 2)
        _precondition_gradient_slot!(moves, gradients, factors, proposal_slot, family)
    end
    _validate_preconditioned_moves!(moves)
    return nothing
end

@kernel function _precondition_gradients_kernel!(moves, gradients, factors, family)
    proposal_slot = @index(Global, Linear)
    _precondition_gradient_slot!(moves, gradients, factors, proposal_slot, family)
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
    family=GaussianFamily(),
)
    backend = KernelAbstractions.get_backend(moves)
    kernel = _precondition_gradients_kernel!(backend)
    proposal_count = size(gradients, 2)
    kernel(
        moves,
        gradients,
        factors,
        family;
        ndrange=proposal_count,
        workgroupsize=_population_workgroupsize(
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
    family=GaussianFamily(),
)
    _threaded_foreach(axes(gradients, 2)) do proposal_slot
        _precondition_gradient_slot!(moves, gradients, factors, proposal_slot, family)
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
    _gradient!(gradient, bound_gradient, location, proposal_slot)
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
        workgroupsize=_population_workgroupsize(
            execution,
            backend,
            proposal_count,
        ),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

function _evaluate_frozen_gradients!(
    values, gradients, target, bound_gradient::_BoundBatchGradient,
    locations, ::_KernelExecution,
)
    _batch_value_and_gradient!(values, gradients, bound_gradient, locations)
    KernelAbstractions.synchronize(KernelAbstractions.get_backend(values))
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
    backend_execution=nothing,
)
    workspace = method_state.workspace
    bound_gradient = _first_order_gramis_bound_gradient(method_state, execution)
    transfers === nothing || _record_gradient_transfers!(transfers, bound_gradient)
    _launch_gramis_gradients!(backend_execution, method_state, target, bound_gradient,
        execution, device, failure_record)
    device === nothing && return nothing
    return _throw_first_order_gramis_device_derivative_failure(
        device, failure_record, workspace.candidate_values, transfers)
end

function _launch_gramis_gradients!(::Nothing, method_state, target, bound_gradient, execution, device, failure_record)
    workspace = method_state.workspace
    device === nothing || fill!(failure_record.storage, zero(eltype(failure_record.storage)))
    _evaluate_frozen_gradients!(
        workspace.frozen_values,
        workspace.gradients,
        target,
        bound_gradient,
        method_state.run.locations,
        execution,
    )
    device === nothing && return nothing
    _validate_gramis_gradients!(workspace, failure_record, execution)
    return nothing
end

function _validate_gramis_gradients!(workspace, failure_record, execution)
    backend = KernelAbstractions.get_backend(workspace.frozen_values)
    proposal_count = length(workspace.frozen_values)
    _validate_frozen_derivatives_kernel!(backend)(
        failure_record.storage,
        workspace.candidate_values,
        workspace.frozen_values,
        workspace.gradients;
        ndrange=proposal_count,
        workgroupsize=_population_workgroupsize(
            execution,
            backend,
            proposal_count,
        ),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

function _precondition_gradients!(
    method_state::_PreparedFirstOrderGRAMIS,
    execution::Union{_SerialCPUExecution,_ThreadedCPUExecution,_KernelExecution},
    ;
    device=nothing,
    failure_record=nothing,
    transfers=nothing,
    backend_execution=nothing,
)
    _launch_gramis_precondition!(backend_execution, method_state, execution, device, failure_record)
    device === nothing && return nothing
    return _throw_first_order_gramis_device_derivative_failure(
        device, failure_record, method_state.workspace.candidate_values, transfers)
end

function _launch_gramis_precondition!(::Nothing, method_state, execution, device, failure_record)
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
        method_state.run.family,
    )
    device === nothing && return nothing
    backend = KernelAbstractions.get_backend(workspace.moves)
    proposal_count = size(workspace.moves, 2)
    _validate_preconditioned_moves_kernel!(backend)(
        failure_record.storage,
        workspace.candidate_values,
        workspace.moves;
        ndrange=proposal_count,
        workgroupsize=_population_workgroupsize(
            execution,
            backend,
            proposal_count,
        ),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

function _evaluate_frozen_gradients!(
    values,
    gradients,
    target,
    bound_gradient,
    locations,
    ::_ThreadedCPUExecution,
)
    _threaded_foreach(axes(locations, 2)) do proposal_slot
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
    _threaded_foreach(axes(locations, 2)) do proposal_slot
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
    _threaded_foreach(axes(locations, 2)) do proposal_slot
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

@kernel function _backtrack_means_kernel!(
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
    failure_storage,
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
    for trial in 1:max_trials
        @inbounds active_mask[proposal_slot] || break
        step = ldexp(one(eltype(steps)), 1 - trial)
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
        failure_storage === nothing && continue
        _, reason, _ = _backtracking_candidate_failure(
            candidate_values,
            trials,
            trial,
            proposal_slot,
        )
        if !iszero(reason)
            @inbounds active_mask[proposal_slot] = false
            logical_index = (trial - 1) * size(locations, 2) + proposal_slot
            _record_native_failure!(
                failure_storage,
                logical_index,
                0,
                reason,
            )
        end
    end
    _finish_backtracking_slot!(
        candidate_locations,
        candidate_values,
        active_mask,
        frozen_values,
        locations,
        proposal_slot,
    )
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
    backend_execution=nothing,
)
    batch = (; candidate_locations, candidate_values, active_mask, steps, trials,
        target, frozen_values, locations, moves, max_trials)
    _launch_gramis_backtracking!(backend_execution, batch, execution, device, failure_record)
    device === nothing || _throw_first_order_gramis_device_backtracking_failure(
        device, failure_record, candidate_values, size(locations, 2), transfers)
    return nothing
end

function _launch_gramis_backtracking!(::Nothing, batch, execution, device, failure_record)
    backend = KernelAbstractions.get_backend(batch.candidate_locations)
    proposal_count = size(batch.locations, 2)
    workgroupsize = _population_workgroupsize(
        execution,
        backend,
        proposal_count,
    )
    failure_storage = if device === nothing
        nothing
    else
        fill!(
            failure_record.storage,
            zero(eltype(failure_record.storage)),
        )
        failure_record.storage
    end
    kernel = _backtrack_means_kernel!(backend)
    kernel(
        values(batch)...,
        failure_storage;
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
    _threaded_foreach(axes(locations, 2)) do proposal_slot
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
    backend_execution=nothing,
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
        backend_execution,
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
            workspace.covariance_centres,
            workspace.normalized_weights,
            workspace.tempering_powers,
            workspace.factor_status,
            workspace.samples,
            bank.locations,
            bank,
            workspace.local_starts,
            method_state.plan.counts,
            round,
        ),
        (
            workspace.covariances,
            bank.factors,
            bank.family,
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
        method_state.run,
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

function _prepare_local_covariance_weights!(
    method_state::_PreparedFirstOrderGRAMIS,
    round,
    execution,
)
    return _population_local_weights!(
        _first_order_gramis_local_weight_arguments(method_state, round)...,
        execution,
    )
end

function _fit_local_covariances!(
    method_state::_PreparedFirstOrderGRAMIS,
    round,
    execution,
)
    _prepare_local_covariance_weights!(method_state, round, execution)
    workspace = method_state.workspace
    return _fit_population_covariances!(
        workspace.covariances,
        workspace.covariance_centres,
        workspace.normalized_weights,
        workspace.tempering_powers,
        workspace.factor_status,
        workspace.samples,
        method_state.run.locations,
        method_state.run,
        workspace.local_starts,
        method_state.plan.counts,
        round,
        execution,
    )
end

@kernel function _blend_local_covariances_kernel!(
    covariances,
    factors,
    family,
    status,
    covariance_rate,
    round,
    regularization,
)
    proposal_slot = @index(Global, Linear)
    if @inbounds(status[proposal_slot]) == _POPULATION_COVARIANCE_READY
        T = eltype(covariances)
        dimension = size(covariances, 1)
        previous_trace = zero(T)
        for column in 1:dimension, row in column:dimension
            previous_trace += abs2(@inbounds factors[row, column, proposal_slot])
        end
        multiplier = _covariance_multiplier(_radial_family_at(family, proposal_slot), T)
        ridge = _scale_aware_ridge(
            previous_trace * multiplier,
            dimension,
            regularization,
        )
        rate = T(@inbounds covariance_rate[round])
        for column in 1:dimension, row in column:dimension
            estimate = (
                @inbounds(covariances[row, column, proposal_slot]) +
                @inbounds(covariances[column, row, proposal_slot])
            ) / T(2)
            old = _population_factor_covariance(
                factors,
                row,
                column,
                proposal_slot,
            )
            blended = (one(T) - rate) * old * multiplier + rate * estimate
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
        workgroupsize=_population_workgroupsize(
            execution,
            backend,
            proposal_count,
        ),
    )
    KernelAbstractions.synchronize(backend)
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
    _factor_population_covariances!(
        method_state.candidate.factors,
        workspace.covariances,
        info,
        workspace.factor_status,
        execution,
    )
    _scale_population_factors!(method_state.candidate.factors,
        method_state.candidate.family, workspace.factor_status, execution)
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
    backend_execution=nothing,
)
    _launch_gramis_covariance!(backend_execution, method_state, round, info, execution, failure_record)
    failure = _first_order_gramis_failure_snapshot!(transfers, failure_record)
    iszero(failure.count) || copyto!(method_state.candidate.factors, method_state.run.factors)
    return failure
end

function _launch_gramis_covariance!(::Nothing, method_state, round, info, execution, failure_record)
    copyto!(method_state.candidate.factors, method_state.run.factors)
    _blend_local_covariances!(method_state, round, execution)
    workspace = method_state.workspace
    _factor_population_covariances!(
        method_state.candidate.factors,
        workspace.covariances,
        info,
        workspace.factor_status,
        execution,
    )
    _scale_population_factors!(method_state.candidate.factors,
        method_state.candidate.family, workspace.factor_status, execution)
    fill!(failure_record.storage, zero(eltype(failure_record.storage)))
    backend = KernelAbstractions.get_backend(info)
    kernel = _record_first_order_gramis_factor_failures!(backend)
    kernel(
        failure_record.storage,
        info;
        ndrange=length(info),
        workgroupsize=_population_workgroupsize(
            execution,
            backend,
            length(info),
        ),
    )
    KernelAbstractions.synchronize(backend)
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
    backend_execution=nothing,
)
    _launch_gramis_add_repulsion!(backend_execution, candidate, repulsion, execution, failure_record, failure_values)
    return _throw_first_order_gramis_device_proposal_failure(
        device, failure_record, failure_values, transfers)
end

function _launch_gramis_add_repulsion!(::Nothing, candidate, repulsion, execution, failure_record, failure_values)
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
        workgroupsize=_population_workgroupsize(
            execution,
            backend,
            proposal_count,
        ),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

@inline function _candidate_factor_validation(factors, status, proposal_slot, family)
    T = eltype(factors)
    @inbounds(status[proposal_slot]) == _POPULATION_COVARIANCE_READY ||
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
    lognormalizer = _radial_lognormalizer(_radial_family_at(family, proposal_slot), T, dimension, logabsdet)
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
            candidate.family,
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
    family,
)
    proposal_slot = @index(Global, Linear)
    ready, reason, value, lognormalizer = _candidate_factor_validation(
        factors,
        status,
        proposal_slot,
        family,
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
    backend_execution=nothing,
)
    _launch_gramis_factor_validation!(backend_execution, candidate, status, execution, failure_record, failure_values)
    return _throw_first_order_gramis_device_proposal_failure(
        device, failure_record, failure_values, transfers)
end

function _launch_gramis_factor_validation!(::Nothing, candidate, status, execution, failure_record, failure_values)
    fill!(failure_record.storage, zero(eltype(failure_record.storage)))
    backend = KernelAbstractions.get_backend(candidate.factors)
    kernel = _validate_first_order_gramis_candidate_factors_kernel!(backend)
    proposal_count = size(candidate.factors, 3)
    kernel(
        failure_record.storage,
        failure_values,
        candidate.factors,
        candidate.lognormalizers,
        status,
        candidate.family;
        ndrange=proposal_count,
        workgroupsize=_population_workgroupsize(
            execution,
            backend,
            proposal_count,
        ),
    )
    KernelAbstractions.synchronize(backend)
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

function _first_order_gramis_failure_value(
    ::MLDataDevices.AbstractAcceleratorDevice,
    values,
    index,
    transfers,
)
    snapshot = Array(view(vec(values), index:index))
    _record_device_scalar_transfer!(
        transfers,
        values,
        eltype(values),
    )
    return only(snapshot)
end

function _first_order_gramis_diagnostic_summary(
    ::MLDataDevices.AbstractAcceleratorDevice,
    status,
    steps,
    trials,
    transfers,
    ::_KernelExecution,
)
    all_zero = count(==(_POPULATION_ALL_ZERO_LOCAL), status)
    tempering = count(
        ==(_POPULATION_TEMPERING_FAILED),
        status,
    )
    backtracking = count(iszero, steps)
    target_trials = sum(trials)
    for values in (status, status, steps, trials)
        _record_device_scalar_transfer!(
            transfers,
            values,
            Int,
        )
    end
    return (; all_zero, tempering, backtracking, target_trials)
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
        all_zero += proposal_status == _POPULATION_ALL_ZERO_LOCAL
        tempering += proposal_status == _POPULATION_TEMPERING_FAILED
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

_reset_prepared_gramis!(::Nothing, state, record) =
    _copy_first_order_gramis_population!(state.run, state.committed)
_fit_prepared_gramis_covariances!(::Nothing, state, round, execution) =
    _fit_local_covariances!(state, round, execution)
_minimum_prepared_gramis_distance(::Nothing, arguments...) =
    _minimum_first_order_gramis_whitened_distance(arguments...)
_copy_prepared_gramis_lognormalizers!(::Nothing, state) =
    copyto!(state.candidate.lognormalizers, state.run.lognormalizers)
function _clear_prepared_gramis_repulsion!(::Nothing, workspace)
    fill!(workspace.repulsion, zero(eltype(workspace.repulsion)))
    fill!(workspace.collision_counts, _GRAMIS_COLLISIONS_UNAVAILABLE)
    return nothing
end
_gramis_result_samples(::Nothing, sampler, samples, transfers) =
    _map_result_samples(sampler.target, samples, sampler.random_buffers.failure_scratch,
        transfers, sampler.threaded)

_sample_prepared_gramis_round!(::Nothing, sampler, state, target, round, execution) =
    _launch_gramis_round!(sampler.rng, sampler.random_buffers, state, target, round,
        execution, sampler.device, sampler.factor_execution)

function _launch_gramis_round!(rng, buffers, state, target, round, execution, device, factor_execution)
    views = _first_order_gramis_round_views(state, round)
    normal = view(buffers.normal, 1:(size(state.run.locations, 1) * views.round_size))
    Random.randn!(rng, normal)
    _fill_radial_buffers!(rng, buffers.radial)
    _prepare_mis_normals!(normal, buffers.radial, state.run,
        views.assignments, buffers.failure_scratch.record.storage, execution)
    _first_order_gramis_sample_round!(views.samples, views.logweights, views.local_logweights,
        views.generating_logdensities, views.proposal_ids, views.round_ids,
        buffers.failure_scratch.record.storage, normal, target, state.run, views.assignments,
        _RealizedMixtureDenominator(state.plan.logcoefficients, round), state.workspace.solve_scratch,
        round, execution, device, _resolved_factor_execution(device, factor_execution))
    return nothing
end

function _allocate_gramis_output(state)
    workspace = state.workspace
    rounds, count = length(state.plan.schedule), last(state.plan.offsets) - 1
    dimension, proposals = size(state.committed.locations)
    T, L = eltype(state.committed.locations), eltype(workspace.round_logweights)
    return (
        samples=similar(workspace.samples, T, dimension, count),
        logweights=similar(workspace.round_logweights, L, count),
        round_ids=similar(workspace.round_ids, Int, count),
        proposal_ids=similar(workspace.round_proposal_ids, Int, count),
        local_ess=similar(workspace.local_ess, T, proposals, rounds),
        tempering_powers=similar(workspace.tempering_powers, T, proposals, rounds),
        fallback_status=similar(workspace.factor_status, UInt8, proposals, rounds),
        accepted_steps=similar(workspace.steps, T, proposals, rounds),
        backtracking_trials=similar(workspace.backtracking_trials, Int, proposals, rounds),
        collision_counts=similar(workspace.collision_counts, Int, proposals, rounds),
    )
end

function _publish_gramis_round!(state, round, output)
    workspace = state.workspace
    views = _first_order_gramis_round_views(state, round)
    indices = state.plan.offsets[round]:(state.plan.offsets[round + 1] - 1)
    copyto!(_sample_view(output.samples, indices), views.samples)
    copyto!(view(output.logweights, indices), views.logweights)
    copyto!(view(output.round_ids, indices), views.round_ids)
    copyto!(view(output.proposal_ids, indices), views.proposal_ids)
    copyto!(view(output.local_ess, :, round), workspace.local_ess)
    copyto!(view(output.tempering_powers, :, round), workspace.tempering_powers)
    copyto!(view(output.fallback_status, :, round), workspace.factor_status)
    copyto!(view(output.accepted_steps, :, round), workspace.steps)
    copyto!(view(output.backtracking_trials, :, round), workspace.backtracking_trials)
    copyto!(view(output.collision_counts, :, round), workspace.collision_counts)
    return nothing
end

function _publish_prepared_gramis!(::Nothing, sampler, state, round, output, transfers, execution)
    _publish_gramis_round!(state, round, output)
    workspace = state.workspace
    return (
        weights=_logweight_summary(_first_order_gramis_round_views(state, round).logweights, transfers),
        bounded=_first_order_gramis_diagnostic_summary(sampler.device, workspace.factor_status,
            workspace.steps, workspace.backtracking_trials, transfers, execution),
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
    proposal_count = size(method_state.committed.locations, 2)
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
                output=_allocate_gramis_output(method_state),
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
    output = storage.output
    samples = output.samples
    logweights = output.logweights
    round_ids = output.round_ids
    proposal_ids = output.proposal_ids
    local_ess = output.local_ess
    tempering_powers = output.tempering_powers
    fallback_status = output.fallback_status
    accepted_steps = output.accepted_steps
    backtracking_trials = output.backtracking_trials
    collision_counts = output.collision_counts
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
        _reset_prepared_gramis!(sampler.backend_execution, method_state, buffers.failure_scratch.record),
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
        @_capture_first_order_gramis_round(
            method_state,
            transfers,
            round,
            :sampling,
            round - 1,
            begin
                _sample_prepared_gramis_round!(sampler.backend_execution, sampler,
                    method_state, target_evaluator, round, execution)
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
            _fit_prepared_gramis_covariances!(sampler.backend_execution, method_state, round, execution),
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
                    backend_execution=sampler.backend_execution,
                )
                _precondition_gradients!(
                    method_state,
                    execution,
                    ;
                    device=accelerator_device,
                    failure_record=buffers.failure_scratch.record,
                    transfers,
                    backend_execution=sampler.backend_execution,
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
                        _repulsion!(repulsion_arguments..., execution; family=method_state.run.family)
                    else
                        _repulsion!(
                            repulsion_arguments...,
                            execution;
                            family=method_state.run.family,
                            device=accelerator_device,
                            failure_record=buffers.failure_scratch.record,
                            failure_values=workspace.candidate_values,
                            transfers,
                            backend_execution=sampler.backend_execution,
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
                        _minimum_prepared_gramis_distance(
                            sampler.backend_execution, distance_arguments...,
                        ) : _minimum_prepared_gramis_distance(
                            sampler.backend_execution,
                            distance_arguments...,
                            workspace.candidate_values,
                        )
                else
                    _clear_prepared_gramis_repulsion!(sampler.backend_execution, workspace)
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
                    backend_execution=sampler.backend_execution,
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
                    sampler.backend_execution,
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
                _copy_prepared_gramis_lognormalizers!(sampler.backend_execution, method_state)
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
                    sampler.backend_execution,
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
                    sampler.backend_execution,
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
            _publish_prepared_gramis!(sampler.backend_execution, sampler,
                method_state, round, output, transfers, execution),
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
                    sampler.device,
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
            result_samples = _gramis_result_samples(sampler.backend_execution, sampler, samples, transfers)
            constructed = _adopt_validated_weighted_samples(
                result_samples,
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
