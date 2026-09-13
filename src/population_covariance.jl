const _POPULATION_COVARIANCE_READY = UInt8(0)
const _POPULATION_ALL_ZERO_LOCAL = UInt8(1)
const _POPULATION_TEMPERING_FAILED = UInt8(2)
const _POPULATION_REDUCTION_WORKGROUP_SIZE = 256
const _POPULATION_REDUCTION_OFFSETS = (128, 64, 32, 16, 8, 4, 2, 1)
const _POPULATION_COVARIANCE_MAX_ROW_LANES = 32
const _POPULATION_CHOLESKY_WORKGROUP_SIZE = 64
const _POPULATION_COVARIANCE_NONFINITE_INFO = Int32(-1)

function _default_population_covariance_ess_threshold(sample_count)
    quotient, remainder = divrem(sample_count, 10)
    return 3 * quotient + cld(3 * remainder, 10)
end

@inline _population_workgroupsize(execution, backend, ndrange) =
    _native_workgroupsize(execution, ndrange)

@inline function _population_workgroupsize(
    ::_ThreadedCPUExecution,
    ::KernelAbstractions.CPU,
    ndrange,
)
    return min(1_024, max(1, fld(ndrange, Threads.nthreads(:default))))
end

@inline function _population_local_group(starts, counts, proposal_slot, round)
    return @inbounds(starts[proposal_slot, round]),
    @inbounds(counts[proposal_slot, round])
end

@inline function _population_covariance_row_lane_count(dimension)
    row_lane_count = 1
    while row_lane_count < dimension &&
          row_lane_count < _POPULATION_COVARIANCE_MAX_ROW_LANES
        row_lane_count *= 2
    end
    return row_lane_count
end

@inline function _population_covariance_tile_count(dimension, row_lane_count)
    quotient, remainder = divrem(dimension, row_lane_count)
    return row_lane_count * quotient * (quotient + 1) ÷ 2 +
           remainder * (quotient + 1)
end

@inline _population_covariance_sample(samples::AbstractVector, row, sample) =
    @inbounds samples[sample]
@inline _population_covariance_sample(samples::AbstractMatrix, row, sample) =
    @inbounds samples[row, sample]

@inline _store_population_raw_mean!(::Nothing, args...) = nothing

function _store_population_raw_mean!(
    raw_means,
    samples,
    normalized_weights,
    first_sample,
    last_sample,
    proposal_slot,
)
    T = eltype(raw_means)
    for coordinate in axes(raw_means, 1)
        mean = zero(T)
        for sample_index in first_sample:last_sample
            mean += T(@inbounds(normalized_weights[sample_index])) * T(
                _population_covariance_sample(samples, coordinate, sample_index),
            )
        end
        @inbounds raw_means[coordinate, proposal_slot] = mean
    end
    return nothing
end

@inline function _population_current_covariance(
    bank::_PackedFactorBank,
    row,
    column,
    proposal_slot,
)
    return _population_factor_covariance(
        bank.factors,
        row,
        column,
        proposal_slot,
    ) * _covariance_multiplier(_radial_family_at(bank.family, proposal_slot), eltype(bank.factors))
end

@inline function _population_factor_covariance(
    factors,
    row,
    column,
    proposal_slot,
)
    covariance = zero(eltype(factors))
    for factor_column in 1:min(row, column)
        covariance += @inbounds(
            factors[row, factor_column, proposal_slot] *
            factors[column, factor_column, proposal_slot]
        )
    end
    return covariance
end

@inline function _population_current_covariance(
    bank::_PackedDiagonalBank,
    row,
    column,
    proposal_slot,
)
    row == column || return zero(eltype(bank.scales))
    return abs2(@inbounds(bank.scales[row, proposal_slot])) *
        _covariance_multiplier(_radial_family_at(bank.family, proposal_slot), eltype(bank.scales))
end

@inline function _population_power_ess(
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

function _population_local_weights_slot!(
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
    proposal_slot,
    raw_means,
    samples,
)
    first_sample, sample_count = _population_local_group(
        starts,
        counts,
        proposal_slot,
        round,
    )
    last_sample = first_sample + sample_count - 1
    T = eltype(normalized_weights)
    maximum_logweight = maximum(view(local_logweights, first_sample:last_sample))
    if maximum_logweight == -Inf
        @inbounds local_ess[proposal_slot] = zero(T)
        @inbounds tempering_powers[proposal_slot] = zero(T)
        @inbounds status[proposal_slot] = _POPULATION_ALL_ZERO_LOCAL
        return nothing
    end

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
    _store_population_raw_mean!(
        raw_means,
        samples,
        normalized_weights,
        first_sample,
        last_sample,
        proposal_slot,
    )
    raw_ess = abs2(total) / square_total
    @inbounds local_ess[proposal_slot] = raw_ess
    @inbounds tempering_powers[proposal_slot] = one(T)
    @inbounds status[proposal_slot] = _POPULATION_COVARIANCE_READY
    raw_ess >= T(@inbounds(thresholds[proposal_slot, round])) && return nothing

    feasible_lower = zero(T)
    infeasible_upper = one(T)
    feasible_ess = zero(T)
    for _ in 1:max_iterations
        power = (feasible_lower + infeasible_upper) / T(2)
        ess = _population_power_ess(
            local_logweights,
            first_sample,
            last_sample,
            power,
            T,
        )
        if ess >= T(@inbounds(thresholds[proposal_slot, round]))
            feasible_lower = power
            feasible_ess = ess
        else
            infeasible_upper = power
        end
        infeasible_upper - feasible_lower <= tolerance && break
    end
    if iszero(feasible_lower)
        @inbounds tempering_powers[proposal_slot] = zero(T)
        @inbounds status[proposal_slot] = _POPULATION_TEMPERING_FAILED
        return nothing
    end

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
    return nothing
end

@kernel function _cooperative_population_local_weights_kernel!(
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
    raw_means,
    samples,
)
    proposal_slot = @index(Group, Linear)
    lane = @index(Local, Linear)
    @uniform lane_count = @groupsize()[1]
    @uniform T = eltype(normalized_weights)
    maxima = @localmem eltype(local_logweights) (
        _POPULATION_REDUCTION_WORKGROUP_SIZE,
    )
    totals = @localmem eltype(normalized_weights) (
        _POPULATION_REDUCTION_WORKGROUP_SIZE,
    )
    squared_totals = @localmem eltype(normalized_weights) (
        _POPULATION_REDUCTION_WORKGROUP_SIZE,
    )
    # accepted power, rejected power, accepted ESS, trial power, maximum, total
    tempering = @localmem eltype(normalized_weights) (6,)
    group = @localmem eltype(starts) (3,)
    if lane == 1
        @inbounds group[1] = proposal_slot
        @inbounds group[2] = starts[group[1], round]
        @inbounds group[3] = group[2] + counts[group[1], round] - 1
    end
    @synchronize()

    lane_maximum = eltype(local_logweights)(-Inf)
    for sample in (@inbounds(group[2]) + lane - 1):lane_count:(@inbounds(group[3]))
        lane_maximum = max(lane_maximum, @inbounds(local_logweights[sample]))
    end
    @inbounds maxima[lane] = lane_maximum
    @synchronize()
    for offset in _POPULATION_REDUCTION_OFFSETS
        if lane <= offset
            @inbounds maxima[lane] = max(maxima[lane], maxima[lane + offset])
        end
        @synchronize()
    end
    if lane == 1 && @inbounds(maxima[1]) == eltype(local_logweights)(-Inf)
        @inbounds local_ess[group[1]] = zero(T)
        @inbounds tempering_powers[group[1]] = zero(T)
        @inbounds status[group[1]] = _POPULATION_ALL_ZERO_LOCAL
    end
    @synchronize()
    if @inbounds(maxima[1]) != eltype(local_logweights)(-Inf)
    lane_total = zero(T)
    lane_square_total = zero(T)
    for sample in (@inbounds(group[2]) + lane - 1):lane_count:(@inbounds(group[3]))
        weight = T(exp(@inbounds(local_logweights[sample]) - @inbounds(maxima[1])))
        @inbounds normalized_weights[sample] = weight
        lane_total += weight
        lane_square_total += abs2(weight)
    end
    @inbounds totals[lane] = lane_total
    @inbounds squared_totals[lane] = lane_square_total
    @synchronize()
    for offset in _POPULATION_REDUCTION_OFFSETS
        if lane <= offset
            @inbounds totals[lane] += totals[lane + offset]
            @inbounds squared_totals[lane] += squared_totals[lane + offset]
        end
        @synchronize()
    end
    for sample in (@inbounds(group[2]) + lane - 1):lane_count:(@inbounds(group[3]))
        @inbounds normalized_weights[sample] *= inv(totals[1])
    end
    if lane == 1
        @inbounds local_ess[group[1]] = abs2(totals[1]) / squared_totals[1]
        @inbounds tempering_powers[group[1]] = one(T)
        @inbounds status[group[1]] = _POPULATION_COVARIANCE_READY
    end
    @synchronize()
    if !(raw_means isa Nothing)
    for coordinate in axes(raw_means, 1)
        lane_total = zero(T)
        for sample_index in (@inbounds(group[2]) + lane - 1):lane_count:(@inbounds(group[3]))
            lane_total += T(@inbounds(normalized_weights[sample_index])) * T(
                _population_covariance_sample(samples, coordinate, sample_index),
            )
        end
        @inbounds totals[lane] = lane_total
        @synchronize()
        for offset in _POPULATION_REDUCTION_OFFSETS
            if lane <= offset
                @inbounds totals[lane] += totals[lane + offset]
            end
            @synchronize()
        end
        lane == 1 && (@inbounds raw_means[coordinate, group[1]] = totals[1])
        @synchronize()
    end
    end
    if @inbounds(local_ess[group[1]]) <
       T(@inbounds(thresholds[group[1], round]))
    if lane == 1
        @inbounds tempering[1] = zero(T)
        @inbounds tempering[2] = one(T)
        @inbounds tempering[3] = zero(T)
    end
    @synchronize()
    for _ in 1:max_iterations
        lane == 1 && (@inbounds tempering[4] = (tempering[1] + tempering[2]) / T(2))
        @synchronize()
        lane_maximum = T(-Inf)
        for sample in (@inbounds(group[2]) + lane - 1):lane_count:(@inbounds(group[3]))
            scaled = @inbounds(tempering[4]) * T(@inbounds(local_logweights[sample]))
            lane_maximum = max(lane_maximum, scaled)
        end
        @inbounds maxima[lane] = lane_maximum
        @synchronize()
        for offset in _POPULATION_REDUCTION_OFFSETS
            if lane <= offset
                @inbounds maxima[lane] = max(maxima[lane], maxima[lane + offset])
            end
            @synchronize()
        end
        lane_total = zero(T)
        lane_square_total = zero(T)
        for sample in (@inbounds(group[2]) + lane - 1):lane_count:(@inbounds(group[3]))
            scaled = @inbounds(tempering[4]) * T(@inbounds(local_logweights[sample]))
            weight = exp(scaled - @inbounds(maxima[1]))
            lane_total += weight
            lane_square_total += abs2(weight)
        end
        @inbounds totals[lane] = lane_total
        @inbounds squared_totals[lane] = lane_square_total
        @synchronize()
        for offset in _POPULATION_REDUCTION_OFFSETS
            if lane <= offset
                @inbounds totals[lane] += totals[lane + offset]
                @inbounds squared_totals[lane] += squared_totals[lane + offset]
            end
            @synchronize()
        end
        if lane == 1
            ess = abs2(@inbounds(totals[1])) / @inbounds(squared_totals[1])
            if ess >= T(@inbounds(thresholds[group[1], round]))
                @inbounds tempering[1] = tempering[4]
                @inbounds tempering[3] = ess
                @inbounds tempering[5] = maxima[1]
                @inbounds tempering[6] = totals[1]
            else
                @inbounds tempering[2] = tempering[4]
            end
        end
        @synchronize()
        @inbounds(tempering[2]) - @inbounds(tempering[1]) <= tolerance && break
    end
    if @inbounds(tempering[1]) > zero(T)
        for sample in (@inbounds(group[2]) + lane - 1):lane_count:(@inbounds(group[3]))
            scaled = @inbounds(tempering[1]) * T(@inbounds(local_logweights[sample]))
            @inbounds normalized_weights[sample] =
                exp(scaled - @inbounds(tempering[5])) * inv(@inbounds(tempering[6]))
        end
        if lane == 1
            @inbounds local_ess[group[1]] = tempering[3]
            @inbounds tempering_powers[group[1]] = tempering[1]
        end
    elseif lane == 1
        @inbounds tempering_powers[group[1]] = zero(T)
        @inbounds status[group[1]] = _POPULATION_TEMPERING_FAILED
    end
    end
    end
end

function _population_local_weights!(
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
    execution::Union{_SerialCPUExecution,_ThreadedCPUExecution},
    raw_means=nothing,
    samples=nothing,
)
    proposal_slots = axes(counts, 1)
    apply_slot = proposal_slot -> _population_local_weights_slot!(
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
        proposal_slot,
        raw_means,
        samples,
    )
    if execution isa _ThreadedCPUExecution
        _threaded_foreach(apply_slot, proposal_slots)
    else
        foreach(apply_slot, proposal_slots)
    end
    return nothing
end

function _population_local_weights!(
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
    ::_KernelExecution,
    raw_means=nothing,
    samples=nothing,
)
    backend = KernelAbstractions.get_backend(normalized_weights)
    proposal_count = size(counts, 1)
    kernel = _cooperative_population_local_weights_kernel!(
        backend,
        _POPULATION_REDUCTION_WORKGROUP_SIZE,
    )
    kernel(
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
        raw_means,
        samples;
        ndrange=_POPULATION_REDUCTION_WORKGROUP_SIZE * proposal_count,
        workgroupsize=_POPULATION_REDUCTION_WORKGROUP_SIZE,
    )
    return nothing
end

function _fit_population_covariance_slot!(
    covariances,
    covariance_centres,
    normalized_weights,
    tempering_powers,
    status,
    samples,
    locations,
    current_bank,
    starts,
    counts,
    round,
    proposal_slot,
)
    dimension = size(covariances, 1)
    first_sample, sample_count = _population_local_group(
        starts,
        counts,
        proposal_slot,
        round,
    )
    last_sample = first_sample + sample_count - 1
    T = eltype(covariances)
    if @inbounds(status[proposal_slot]) != _POPULATION_COVARIANCE_READY
        for column in 1:dimension, row in column:dimension
            covariance = _population_current_covariance(
                current_bank,
                row,
                column,
                proposal_slot,
            )
            @inbounds covariances[row, column, proposal_slot] = covariance
            @inbounds covariances[column, row, proposal_slot] = covariance
        end
        return nothing
    end
    for row in 1:dimension
        center = @inbounds locations[row, proposal_slot]
        if @inbounds(tempering_powers[proposal_slot]) < one(T)
            center = zero(T)
            for sample in first_sample:last_sample
                center += @inbounds(normalized_weights[sample]) *
                          _population_covariance_sample(samples, row, sample)
            end
        end
        @inbounds covariance_centres[row, proposal_slot] = center
    end
    for column in 1:dimension, row in column:dimension
        covariance = zero(T)
        center_row = @inbounds covariance_centres[row, proposal_slot]
        center_column = @inbounds covariance_centres[column, proposal_slot]
        for sample in first_sample:last_sample
            weight = @inbounds normalized_weights[sample]
            centered_row =
                _population_covariance_sample(samples, row, sample) - center_row
            centered_column =
                _population_covariance_sample(samples, column, sample) - center_column
            covariance += weight * centered_row * centered_column
        end
        @inbounds covariances[row, column, proposal_slot] = covariance
        @inbounds covariances[column, row, proposal_slot] = covariance
    end
    return nothing
end

@kernel function _population_covariance_centres_kernel!(
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
        _POPULATION_REDUCTION_WORKGROUP_SIZE,
    )
    # coordinate, proposal, first sample, last sample, weighted-centre flag
    group_state = @localmem eltype(starts) (5,)
    if @inbounds(lane[1]) == 1
        dimension = size(covariance_centres, 1)
        proposal_slot = (group_index - 1) ÷ dimension + 1
        @inbounds group_state[1] = (group_index - 1) % dimension + 1
        @inbounds group_state[2] = proposal_slot
        @inbounds group_state[3] = starts[proposal_slot, round]
        @inbounds group_state[4] =
            group_state[3] + counts[proposal_slot, round] - 1
        @inbounds group_state[5] =
            status[proposal_slot] == _POPULATION_COVARIANCE_READY &&
            tempering_powers[proposal_slot] < one(T)
    end
    @synchronize()

    partial = zero(T)
    if @inbounds(group_state[5]) == 1
        for sample in (@inbounds(group_state[3]) + @inbounds(lane[1]) - 1):lane_count:(@inbounds(group_state[4]))
            partial += @inbounds(normalized_weights[sample]) *
                       _population_covariance_sample(
                samples,
                @inbounds(group_state[1]),
                sample,
            )
        end
    end
    @inbounds partial_centres[lane[1]] = partial
    @synchronize()
    for offset in _POPULATION_REDUCTION_OFFSETS
        if @inbounds(lane[1]) <= offset
            @inbounds partial_centres[lane[1]] +=
                partial_centres[lane[1] + offset]
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

@kernel function _population_covariances_kernel!(
    covariances,
    covariance_centres,
    normalized_weights,
    status,
    samples,
    current_bank,
    starts,
    counts,
    round,
)
    group_index = @index(Group, Linear)
    lane = @index(Local, Linear)
    @uniform lane_count = @groupsize()[1]
    @uniform dimension = size(covariances, 1)
    @uniform row_lane_count = _population_covariance_row_lane_count(dimension)
    @uniform sample_lane_count = lane_count ÷ row_lane_count
    @uniform T = eltype(covariances)
    partial_covariances = @localmem eltype(covariances) (
        _POPULATION_REDUCTION_WORKGROUP_SIZE,
    )
    # first row, column, proposal, first sample, last sample, covariance-ready flag
    group_state = @localmem eltype(starts) (6,)
    if lane == 1
        tiles_per_proposal =
            _population_covariance_tile_count(dimension, row_lane_count)
        proposal_slot = (group_index - 1) ÷ tiles_per_proposal + 1
        tile = (group_index - 1) % tiles_per_proposal + 1
        column = 1
        tiles_in_column = cld(dimension, row_lane_count)
        while tile > tiles_in_column
            tile -= tiles_in_column
            column += 1
            tiles_in_column = cld(dimension - column + 1, row_lane_count)
        end
        @inbounds group_state[1] = column + (tile - 1) * row_lane_count
        @inbounds group_state[2] = column
        @inbounds group_state[3] = proposal_slot
        @inbounds group_state[4] = starts[proposal_slot, round]
        @inbounds group_state[5] =
            group_state[4] + counts[proposal_slot, round] - 1
        @inbounds group_state[6] =
            status[proposal_slot] == _POPULATION_COVARIANCE_READY
    end
    @synchronize()

    row_lane = (lane - 1) % row_lane_count + 1
    sample_lane = (lane - 1) ÷ row_lane_count + 1
    row = @inbounds(group_state[1]) + row_lane - 1
    active_row = row <= dimension
    covariance = zero(T)
    if @inbounds(group_state[6]) == 1 && active_row
        center_row = @inbounds covariance_centres[row, group_state[3]]
        center_column =
            @inbounds covariance_centres[group_state[2], group_state[3]]
        for sample in (@inbounds(group_state[4]) + sample_lane - 1):sample_lane_count:(@inbounds(group_state[5]))
            weight = @inbounds normalized_weights[sample]
            centered_row = _population_covariance_sample(
                samples,
                row,
                sample,
            ) - center_row
            centered_column = _population_covariance_sample(
                samples,
                @inbounds(group_state[2]),
                sample,
            ) - center_column
            covariance += weight * centered_row * centered_column
        end
    end
    @inbounds partial_covariances[lane] = covariance
    @synchronize()
    for offset in _POPULATION_REDUCTION_OFFSETS
        if offset >= row_lane_count && lane <= offset
            @inbounds partial_covariances[lane] +=
                partial_covariances[lane + offset]
        end
        @synchronize()
    end
    if lane <= row_lane_count
        row = @inbounds(group_state[1]) + lane - 1
        if row <= dimension
            if @inbounds(group_state[6]) == 1
                covariance = @inbounds partial_covariances[lane]
            else
                covariance = _population_current_covariance(
                    current_bank,
                    row,
                    @inbounds(group_state[2]),
                    @inbounds(group_state[3]),
                )
            end
            @inbounds covariances[row, group_state[2], group_state[3]] = covariance
            @inbounds covariances[group_state[2], row, group_state[3]] = covariance
        end
    end
end

function _fit_population_covariances!(
    covariances,
    covariance_centres,
    normalized_weights,
    tempering_powers,
    status,
    samples,
    locations,
    current_bank,
    starts,
    counts,
    round,
    execution::Union{_SerialCPUExecution,_ThreadedCPUExecution},
)
    proposal_slots = axes(counts, 1)
    fit_slot = proposal_slot -> _fit_population_covariance_slot!(
        covariances,
        covariance_centres,
        normalized_weights,
        tempering_powers,
        status,
        samples,
        locations,
        current_bank,
        starts,
        counts,
        round,
        proposal_slot,
    )
    if execution isa _ThreadedCPUExecution
        _threaded_foreach(fit_slot, proposal_slots)
    else
        foreach(fit_slot, proposal_slots)
    end
    return nothing
end

function _fit_population_covariances!(
    covariances,
    covariance_centres,
    normalized_weights,
    tempering_powers,
    status,
    samples,
    locations,
    current_bank,
    starts,
    counts,
    round,
    execution::_KernelExecution,
)
    backend = KernelAbstractions.get_backend(covariances)
    proposal_count = size(counts, 1)
    dimension = size(covariances, 1)
    centre_kernel = _population_covariance_centres_kernel!(
        backend,
        _POPULATION_REDUCTION_WORKGROUP_SIZE,
    )
    centre_kernel(
        covariance_centres,
        normalized_weights,
        tempering_powers,
        status,
        samples,
        locations,
        starts,
        counts,
        round;
        ndrange=_POPULATION_REDUCTION_WORKGROUP_SIZE * dimension * proposal_count,
        workgroupsize=_POPULATION_REDUCTION_WORKGROUP_SIZE,
    )
    covariance_kernel = _population_covariances_kernel!(
        backend,
        _POPULATION_REDUCTION_WORKGROUP_SIZE,
    )
    row_lane_count = _population_covariance_row_lane_count(dimension)
    covariance_tile_count =
        _population_covariance_tile_count(dimension, row_lane_count) * proposal_count
    covariance_kernel(
        covariances,
        covariance_centres,
        normalized_weights,
        status,
        samples,
        current_bank,
        starts,
        counts,
        round;
        ndrange=_POPULATION_REDUCTION_WORKGROUP_SIZE * covariance_tile_count,
        workgroupsize=_POPULATION_REDUCTION_WORKGROUP_SIZE,
    )
    KernelAbstractions.synchronize(backend)
    return nothing
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
    if any(!isfinite, factor)
        @inbounds info[proposal_slot] = _POPULATION_COVARIANCE_NONFINITE_INFO
        return nothing
    end
    factor, factor_info = LinearAlgebra.LAPACK.potrf!('L', factor)
    @inbounds info[proposal_slot] = factor_info
    if iszero(factor_info)
        for column in axes(factor, 2), row in 1:(column - 1)
            @inbounds factor[row, column] = zero(eltype(factor))
        end
    end
    return nothing
end

@kernel function _factor_population_covariances_kernel!(
    factors,
    covariances,
    info,
    status,
)
    proposal_slot = @index(Group, Linear)
    lane = @index(Local, Linear)
    @uniform lane_count = @groupsize()[1]
    @uniform dimension = size(factors, 1)
    lane_bad = @localmem Int32 (_POPULATION_CHOLESKY_WORKGROUP_SIZE,)
    lane == 1 && (@inbounds info[proposal_slot] = Int32(0))
    bad = Int32(0)
    for entry in lane:lane_count:(dimension * dimension)
        row = (entry - 1) % dimension + 1
        column = (entry - 1) ÷ dimension + 1
        if @inbounds(status[proposal_slot]) == _POPULATION_COVARIANCE_READY
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
            _POPULATION_COVARIANCE_NONFINITE_INFO)
    end
    @synchronize()
    for column in 1:dimension
        if lane == 1 &&
           @inbounds(status[proposal_slot]) == _POPULATION_COVARIANCE_READY &&
           iszero(@inbounds(info[proposal_slot]))
            pivot = @inbounds factors[column, column, proposal_slot]
            for previous in 1:(column - 1)
                pivot -= abs2(@inbounds factors[column, previous, proposal_slot])
            end
            if isfinite(pivot) && pivot > zero(pivot)
                @inbounds factors[column, column, proposal_slot] = sqrt(pivot)
            else
                @inbounds info[proposal_slot] = Int32(column)
            end
        end
        @synchronize()
        bad = Int32(0)
        if @inbounds(status[proposal_slot]) == _POPULATION_COVARIANCE_READY &&
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
           @inbounds(status[proposal_slot]) == _POPULATION_COVARIANCE_READY &&
           iszero(@inbounds(info[proposal_slot]))
            group_bad = @inbounds lane_bad[1]
            for other_lane in 2:lane_count
                group_bad |= @inbounds lane_bad[other_lane]
            end
            iszero(group_bad) || (@inbounds info[proposal_slot] = Int32(column))
        end
        @synchronize()
    end
end

function _factor_population_covariances!(
    factors::StridedArray{T,3},
    covariances::StridedArray{T,3},
    info::StridedVector{I},
    status,
    execution::Union{_SerialCPUExecution,_ThreadedCPUExecution},
) where {T<:Union{Float32,Float64},I<:Signed}
    proposal_slots = axes(factors, 3)
    factor_slot = proposal_slot -> begin
        if @inbounds(status[proposal_slot]) == _POPULATION_COVARIANCE_READY
            _factor_population_slot!(factors, covariances, info, proposal_slot)
        else
            @inbounds info[proposal_slot] = 0
        end
    end
    if execution isa _ThreadedCPUExecution
        _threaded_foreach(factor_slot, proposal_slots)
    else
        foreach(factor_slot, proposal_slots)
    end
    return nothing
end

function _factor_population_covariances!(
    factors,
    covariances,
    info,
    status,
    ::_KernelExecution,
)
    backend = KernelAbstractions.get_backend(factors)
    proposal_count = size(factors, 3)
    kernel = _factor_population_covariances_kernel!(
        backend,
        _POPULATION_CHOLESKY_WORKGROUP_SIZE,
    )
    kernel(
        factors,
        covariances,
        info,
        status;
        ndrange=_POPULATION_CHOLESKY_WORKGROUP_SIZE * proposal_count,
        workgroupsize=_POPULATION_CHOLESKY_WORKGROUP_SIZE,
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

# Convert fitted covariance factors once, before publishing proposal scale factors.
_scale_population_factors!(factors, ::GaussianFamily, status, execution) = nothing
@kernel function _scale_population_factors_kernel!(factors, family, status)
    row, column, slot = @index(Global, NTuple)
    if status === nothing || status[slot] == _POPULATION_COVARIANCE_READY
        factors[row, column, slot] *= sqrt(inv(_covariance_multiplier(
            _radial_family_at(family, slot), eltype(factors))))
    end
end
function _scale_population_factors!(factors, family::_PackedStudentTFamily, status, execution)
    backend = KernelAbstractions.get_backend(factors)
    _scale_population_factors_kernel!(backend)(factors, family, status; ndrange=size(factors))
    KernelAbstractions.synchronize(backend)
    return nothing
end
