const _LOCAL_REDUCTION_WORKGROUP_SIZE = 256
const _LOCAL_REDUCTION_OFFSETS = (128, 64, 32, 16, 8, 4, 2, 1)

@inline _population_sample_coordinate(samples::AbstractVector, coordinate, sample) =
    @inbounds samples[sample]
@inline _population_sample_coordinate(samples::AbstractMatrix, coordinate, sample) =
    @inbounds samples[coordinate, sample]

@inline function _store_population_location!(
    locations::AbstractVector,
    coordinate,
    proposal,
    value,
)
    @inbounds locations[proposal] = value
    return nothing
end

@inline function _store_population_location!(
    locations::AbstractMatrix,
    coordinate,
    proposal,
    value,
)
    @inbounds locations[coordinate, proposal] = value
    return nothing
end

@kernel function _local_logweight_maxima_kernel!(
    proposal_maxima,
    logtargets,
    generating_logdensities,
    counts,
    round,
    proposal_offset,
)
    proposal_group = @index(Group, Linear)
    lane = @index(Local, Linear)
    @uniform lane_count = @groupsize()[1]
    maxima = @localmem eltype(logtargets) (_LOCAL_REDUCTION_WORKGROUP_SIZE,)
    group = @localmem Int (3,)

    if lane == 1
        @inbounds group[3] = proposal_group + proposal_offset - 1
        first_sample = 1
        for prior_proposal in 1:(@inbounds(group[3]) - 1)
            first_sample += @inbounds counts[prior_proposal, round]
        end
        @inbounds group[1] = first_sample
        @inbounds group[2] = first_sample + counts[group[3], round] - 1
    end
    @synchronize()

    lane_maximum = eltype(logtargets)(-Inf)
    for sample in (@inbounds(group[1]) + lane - 1):lane_count:(@inbounds(group[2]))
        local_logweight = @inbounds(logtargets[sample]) -
                          @inbounds(generating_logdensities[sample])
        lane_maximum = max(lane_maximum, local_logweight)
    end
    @inbounds maxima[lane] = lane_maximum
    @synchronize()
    for offset in _LOCAL_REDUCTION_OFFSETS
        if lane <= offset
            @inbounds maxima[lane] = max(maxima[lane], maxima[lane + offset])
        end
        @synchronize()
    end
    lane == 1 && (@inbounds proposal_maxima[group[3]] = maxima[1])
end

@kernel function _scaled_local_weights_kernel!(
    scaled_weights,
    logtargets,
    generating_logdensities,
    assignments,
    proposal_maxima,
)
    sample = @index(Global, Linear)
    proposal = @inbounds assignments[sample]
    maximum_logweight = @inbounds proposal_maxima[proposal]
    local_logweight = @inbounds(logtargets[sample]) -
                      @inbounds(generating_logdensities[sample])
    @inbounds scaled_weights[sample] = isfinite(maximum_logweight) ?
                                        exp(local_logweight - maximum_logweight) :
                                        zero(eltype(scaled_weights))
end

function _scale_local_weights!(
    scaled_weights,
    logtargets,
    generating_logdensities,
    assignments,
    proposal_maxima,
    ::KernelAbstractions.CPU,
    execution,
)
    @inbounds @simd for sample in eachindex(scaled_weights)
        proposal = assignments[sample]
        maximum_logweight = proposal_maxima[proposal]
        local_logweight = logtargets[sample] - generating_logdensities[sample]
        scaled_weights[sample] = isfinite(maximum_logweight) ?
                                 exp(local_logweight - maximum_logweight) :
                                 zero(eltype(scaled_weights))
    end
    return nothing
end

function _scale_local_weights!(
    scaled_weights,
    logtargets,
    generating_logdensities,
    assignments,
    proposal_maxima,
    backend,
    execution,
)
    kernel = _scaled_local_weights_kernel!(backend)
    kernel(
        scaled_weights,
        logtargets,
        generating_logdensities,
        assignments,
        proposal_maxima;
        ndrange=length(scaled_weights),
        workgroupsize=_native_workgroupsize(execution, length(scaled_weights)),
    )
    return nothing
end

@kernel function _local_weighted_means_kernel!(
    candidate_locations,
    proposal_maxima,
    samples,
    scaled_weights,
    counts,
    round,
    dimension,
    proposal_offset,
)
    proposal_group = @index(Group, Linear)
    lane = @index(Local, Linear)
    @uniform lane_count = @groupsize()[1]
    partials = @localmem eltype(scaled_weights) (_LOCAL_REDUCTION_WORKGROUP_SIZE,)
    weight_sum = @localmem eltype(scaled_weights) (1,)
    group = @localmem Int (4,)

    if lane == 1
        @inbounds group[4] = proposal_group + proposal_offset - 1
        first_sample = 1
        for prior_proposal in 1:(@inbounds(group[4]) - 1)
            first_sample += @inbounds counts[prior_proposal, round]
        end
        @inbounds group[1] = first_sample
        @inbounds group[2] = first_sample + counts[group[4], round] - 1
        @inbounds group[3] = 1
    end
    @synchronize()

    lane_total = zero(eltype(scaled_weights))
    for sample in (@inbounds(group[1]) + lane - 1):lane_count:(@inbounds(group[2]))
        lane_total += @inbounds scaled_weights[sample]
    end
    @inbounds partials[lane] = lane_total
    @synchronize()
    for offset in _LOCAL_REDUCTION_OFFSETS
        if lane <= offset
            @inbounds partials[lane] += partials[lane + offset]
        end
        @synchronize()
    end
    if lane == 1
        @inbounds weight_sum[1] = partials[1]
        @inbounds group[3] = isfinite(weight_sum[1]) &&
                             weight_sum[1] > zero(eltype(scaled_weights)) ? 1 : 0
    end
    @synchronize()
    for coordinate in 1:dimension
        lane_moment = zero(eltype(scaled_weights))
        for sample in (@inbounds(group[1]) + lane - 1):lane_count:(@inbounds(group[2]))
            lane_moment += @inbounds(scaled_weights[sample]) *
                           _population_sample_coordinate(samples, coordinate, sample)
        end
        @inbounds partials[lane] = lane_moment
        @synchronize()
        for offset in _LOCAL_REDUCTION_OFFSETS
            if lane <= offset
                @inbounds partials[lane] += partials[lane + offset]
            end
            @synchronize()
        end
        if lane == 1 && @inbounds(group[3]) == 1
            value = @inbounds(partials[1]) / @inbounds(weight_sum[1])
            if isfinite(value)
                _store_population_location!(
                    candidate_locations,
                    coordinate,
                    @inbounds(group[4]),
                    value,
                )
            else
                @inbounds group[3] = 0
            end
        end
        @synchronize()
    end
    if lane == 1 && @inbounds(group[3]) == 0
        @inbounds proposal_maxima[group[4]] = eltype(proposal_maxima)(NaN)
    end
end

function _local_weighted_means!(
    candidate_locations,
    proposal_maxima,
    samples,
    scaled_weights,
    logtargets,
    generating_logdensities,
    assignments,
    counts,
    round,
    execution,
    transfers,
)
    _launch_local_weighted_means!(candidate_locations, proposal_maxima, samples,
        scaled_weights, logtargets, generating_logdensities, assignments, counts,
        round, execution)
    valid = _local_means_valid(proposal_maxima)
    _record_device_scalar_transfer!(transfers, proposal_maxima, Bool, Val(:local_mean_validity))
    valid || throw(AllZeroWeightsError())
    return nothing
end

function _launch_local_weighted_means!(candidate_locations, proposal_maxima, samples,
    scaled_weights, logtargets, generating_logdensities, assignments, counts, round, execution)
    backend = KernelAbstractions.get_backend(logtargets)
    proposal_count = size(counts, 1)
    maxima_kernel = _local_logweight_maxima_kernel!(backend)
    if backend isa KernelAbstractions.CPU && execution isa _SerialCPUExecution
        for proposal in 1:proposal_count
            maxima_kernel(
                proposal_maxima,
                logtargets,
                generating_logdensities,
                counts,
                round,
                proposal;
                ndrange=_LOCAL_REDUCTION_WORKGROUP_SIZE,
                workgroupsize=_LOCAL_REDUCTION_WORKGROUP_SIZE,
            )
        end
    else
        maxima_kernel(
            proposal_maxima,
            logtargets,
            generating_logdensities,
            counts,
            round,
            1;
            ndrange=_LOCAL_REDUCTION_WORKGROUP_SIZE * proposal_count,
            workgroupsize=_LOCAL_REDUCTION_WORKGROUP_SIZE,
        )
    end
    _scale_local_weights!(
        scaled_weights,
        logtargets,
        generating_logdensities,
        assignments,
        proposal_maxima,
        KernelAbstractions.get_backend(scaled_weights),
        execution,
    )
    means_kernel = _local_weighted_means_kernel!(backend)
    dimension = samples isa AbstractVector ? 1 : size(samples, 1)
    if backend isa KernelAbstractions.CPU && execution isa _SerialCPUExecution
        for proposal in 1:proposal_count
            means_kernel(
                candidate_locations,
                proposal_maxima,
                samples,
                scaled_weights,
                counts,
                round,
                dimension,
                proposal;
                ndrange=_LOCAL_REDUCTION_WORKGROUP_SIZE,
                workgroupsize=_LOCAL_REDUCTION_WORKGROUP_SIZE,
            )
        end
    else
        means_kernel(
            candidate_locations,
            proposal_maxima,
            samples,
            scaled_weights,
            counts,
            round,
            dimension,
            1;
            ndrange=_LOCAL_REDUCTION_WORKGROUP_SIZE * proposal_count,
            workgroupsize=_LOCAL_REDUCTION_WORKGROUP_SIZE,
        )
    end
    KernelAbstractions.synchronize(backend)
    return nothing
end

_local_means_valid(proposal_maxima) = all(isfinite, proposal_maxima)

function _capture_population_round(
    f,
    algorithm,
    round,
    phase,
    round_size,
    completed_rounds,
)
    try
        return f()
    catch cause
        error_type = _population_round_error(algorithm)
        cause isa error_type && rethrow()
        throw(
            error_type(
                round,
                phase,
                cause,
                (round_size=round_size, completed_rounds=completed_rounds),
            ),
        )
    end
end

_population_execution(sampler, threaded, method_state) =
    threaded ? _ThreadedCPUExecution() : _SerialCPUExecution()

_population_denominator(method_state, round) =
    _RealizedMixtureDenominator(method_state.plan.logcoefficients, round)

function _reset_population_run!(method_state, run, committed)
    copyto!(run.locations, committed.locations)
    return nothing
end

function _commit_population_round!(method_state, run)
    copyto!(run.locations, method_state.workspace.candidate_locations)
    return nothing
end

_before_population_round!(sampler, method_state, target, round, execution, transfers) = nothing
_before_prepared_population_round!(plan, sampler, state, target, round, execution, transfers) =
    _before_population_round!(sampler, state, target, round, execution, transfers)

function _commit_population_run!(method_state)
    method_state.bank, method_state.run_bank = method_state.run_bank, method_state.bank
    return nothing
end

_reset_prepared_population!(::Nothing, state, record) =
    _reset_population_run!(state, state.run_bank, state.bank)
_fill_population_normals!(::Nothing, sampler) = _fill_population_normals!(sampler.rng, sampler.random_buffers)
function _fill_population_normals!(rng, buffers)
    Random.randn!(rng, buffers.normals)
    _fill_radial_buffers!(rng, buffers.radial)
    return nothing
end
_draw_population_round!(::Nothing, sampler, state, target, round, execution) =
    _launch_population_round!(state, sampler.random_buffers, target, round, execution,
        sampler.device, sampler.factor_execution)

function _launch_population_round!(state, buffers, target, round, execution, device, factor_execution)
    views = _population_round_views(state, round)
    denominator = _population_denominator(state, round)
    bank = state.run_bank
    _prepare_mis_normals!(buffers.normals, buffers.radial, bank,
        views.assignments, buffers.failure_scratch.record.storage, execution)
    launch = _use_factor_batch_mis_path(device, bank, denominator,
        eltype(views.logweights), factor_execution) ? _launch_factor_batch_mis_round! : _launch_mis_round!
    launch(views.samples, _MISRoundOutput(views.logweights, views.proposal_ids,
        _population_mis_adaptation(state, views)), buffers.failure_scratch.record.storage,
        buffers.normals, target, bank, views.assignments, denominator,
        state.workspace.solve_scratch, execution)
    return nothing
end

function _advance_prepared_population!(::Nothing, sampler, state, views, round, execution, transfers)
    _advance_population!(sampler, state, views, round, execution, transfers)
    return _logweight_summary(views.logweights, transfers)
end

function _publish_population_round!(::Nothing, state, round, samples, logweights, round_ids, proposal_ids)
    views = _population_round_views(state, round)
    indices = state.plan.offsets[round]:(state.plan.offsets[round + 1] - 1)
    copyto!(_sample_view(samples, indices), views.samples)
    copyto!(view(logweights, indices), views.logweights)
    fill!(view(round_ids, indices), round)
    copyto!(view(proposal_ids, indices), views.proposal_ids)
    return nothing
end

_commit_prepared_population!(::Nothing, state) = _commit_population_round!(state, state.run_bank)
_population_result_samples(::Nothing, sampler, samples, transfers) =
    _map_result_samples(sampler.target, samples, sampler.random_buffers.failure_scratch,
        transfers, sampler.threaded)

function _importance_sample_fixed_population!(sampler, method_state, threaded)
    algorithm = sampler.algorithm
    execution = _population_execution(sampler, threaded, method_state)
    bank = method_state.run_bank
    _reset_prepared_population!(sampler.backend_execution, method_state,
        sampler.random_buffers.failure_scratch.record)
    plan = method_state.plan
    workspace = method_state.workspace
    buffers = sampler.random_buffers
    total_samples = last(plan.offsets) - 1
    samples = _allocate_packed_static_mis_samples(
        buffers.normals,
        bank,
        total_samples,
    )
    logweights = similar(
        workspace.round_logweights,
        eltype(workspace.round_logweights),
        total_samples,
    )
    round_ids = similar(workspace.round_proposal_ids, Int, total_samples)
    proposal_ids = similar(workspace.round_proposal_ids, Int, total_samples)
    round_ess = Vector{eltype(logweights)}(undef, length(plan.schedule))
    round_lognormalizers = similar(round_ess)
    transfers = _ResultTransferCounter(0, 0)
    target = _capture_population_round(
        algorithm,
        1,
        :sample_and_weight,
        plan.schedule[1],
        0,
    ) do
        _bind_resolved_target(sampler.target, _population_binding_sample(bank))
    end
    target_evaluator, target_failures = _capture_population_round(
        algorithm,
        1,
        :sample_and_weight,
        plan.schedule[1],
        0,
    ) do
        _native_target_evaluator(
            KernelAbstractions.get_backend(buffers.normals),
            target,
            eltype(logweights),
            buffers.failure_scratch.target_failures,
        )
    end

    for round in eachindex(plan.schedule)
        round_views = _population_round_views(method_state, round)
        round_size = round_views.round_size
        _capture_population_round(algorithm, round, :transition, round_size, round - 1) do
            _before_prepared_population_round!(sampler.backend_execution, sampler,
                method_state, target, round, execution, transfers)
        end
        _capture_population_round(
            algorithm,
            round,
            :normal_buffer,
            round_size,
            round - 1,
        ) do
            _fill_population_normals!(sampler.backend_execution, sampler)
        end
        _capture_population_round(
            algorithm,
            round,
            :sample_and_weight,
            round_size,
            round - 1,
        ) do
            _draw_population_round!(sampler.backend_execution, sampler, method_state,
                target_evaluator, round, execution)
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
        summary = _capture_population_round(
            algorithm,
            round,
            _population_round_phase(algorithm),
            round_size,
            round - 1,
        ) do
            _advance_prepared_population!(sampler.backend_execution, sampler,
                method_state, round_views, round, execution, transfers)
        end
        _capture_population_round(
            algorithm,
            round,
            :commit_output,
            round_size,
            round - 1,
        ) do
            _publish_population_round!(sampler.backend_execution, method_state, round,
                samples, logweights, round_ids, proposal_ids)
        end
        _capture_population_round(
            algorithm,
            round,
            :advance_population,
            round_size,
            round - 1,
        ) do
            _commit_prepared_population!(sampler.backend_execution, method_state)
            KernelAbstractions.synchronize(
                KernelAbstractions.get_backend(bank.locations),
            )
        end
        _capture_population_round(
            algorithm,
            round,
            :diagnostics,
            round_size,
            round,
        ) do
            round_ess[round] = summary.ess
            round_lognormalizers[round] = summary.lognormalizer
        end
    end

    diagnostics = _population_diagnostics(
        sampler,
        method_state,
        execution,
        plan.schedule,
        round_ess,
        round_lognormalizers,
        transfers,
    )
    final_round = lastindex(plan.schedule)
    result = _capture_population_round(
        algorithm,
        final_round,
        :result_construction,
        plan.schedule[final_round],
        final_round,
    ) do
        result_samples = _population_result_samples(sampler.backend_execution, sampler, samples, transfers)
        _adopt_validated_weighted_samples(
            result_samples,
            logweights;
            provenance=(round=round_ids, proposal_id=proposal_ids),
            diagnostics=diagnostics,
        )
    end
    _commit_population_run!(method_state)
    return result
end
