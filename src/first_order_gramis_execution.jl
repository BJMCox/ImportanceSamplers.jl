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

const _GRAMIS_COVARIANCE_READY = UInt8(0)
const _GRAMIS_ALL_ZERO_LOCAL = UInt8(1)
const _GRAMIS_TEMPERING_FALLBACK = UInt8(2)

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
