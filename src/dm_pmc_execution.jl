"""
    DMPMCRoundError

Exception thrown when a DM-PMC round cannot complete. `round` and `phase`
locate the failure, `cause` stores the underlying exception, and `diagnostics`
reports the requested round size and number of completed rounds. The prepared
sampler retains its pre-call proposal population.
"""
struct DMPMCRoundError{E,D<:NamedTuple} <: Exception
    round::Int
    phase::Symbol
    cause::E
    diagnostics::D
end

function Base.showerror(io::IO, error::DMPMCRoundError)
    print(
        io,
        "DM-PMC failed in round ",
        error.round,
        " during ",
        error.phase,
        ": ",
    )
    showerror(io, error.cause)
end

struct _DMPMCWorkspace{S,W,I,Q,C,M,A,L}
    round_samples::S
    round_logweights::W
    round_proposal_ids::I
    solve_scratch::Q
    resampling_cdf::C
    resampling_maxima::M
    ancestors::A
    candidate_locations::L
end

_population_binding_sample(
    bank::_PackedDiagonalGaussianBank{L,S,N,M,C,I,<:_ScalarGaussianLayout},
) where {L,S,N,M,C,I} = zero(eltype(bank.locations))

_population_binding_sample(bank::_PackedDiagonalGaussianBank) =
    view(bank.locations, :, firstindex(bank.locations, 2))

_population_binding_sample(bank::_PackedFactorGaussianBank) =
    view(bank.locations, :, firstindex(bank.locations, 2))

function _allocate_dm_pmc_workspace(bank, plan, ::Type{T}) where {T}
    capacity = maximum(plan.schedule)
    prototype = bank.locations
    round_samples = _allocate_packed_static_mis_samples(
        prototype,
        bank,
        capacity,
    )
    round_logweights = similar(prototype, T, capacity)
    round_proposal_ids = similar(prototype, Int, capacity)
    solve_scratch = _allocate_mis_solve_scratch(prototype, bank, capacity)
    resampling_cdf = similar(prototype, T, capacity)
    resampling_maxima = similar(prototype, T, _active_proposal_count(bank))
    ancestors = similar(prototype, Int, _active_proposal_count(bank))
    candidate_locations = _allocate_packed_static_mis_samples(
        prototype,
        bank,
        _active_proposal_count(bank),
    )
    return _DMPMCWorkspace(
        round_samples,
        round_logweights,
        round_proposal_ids,
        solve_scratch,
        resampling_cdf,
        resampling_maxima,
        ancestors,
        candidate_locations,
    )
end

@kernel function _local_resample_locations_kernel!(
    destination,
    ancestors,
    source,
    logweights,
    uniforms,
    counts,
    round,
)
    slot = @index(Global, Linear)
    first_sample = 1
    for prior_slot in 1:(slot - 1)
        first_sample += @inbounds counts[prior_slot, round]
    end
    last_sample = first_sample + @inbounds(counts[slot, round]) - 1

    maximum_logweight = eltype(logweights)(-Inf)
    for sample_index in first_sample:last_sample
        maximum_logweight = max(maximum_logweight, @inbounds(logweights[sample_index]))
    end
    ancestor = 0
    if maximum_logweight != -Inf
        total = zero(eltype(logweights))
        for sample_index in first_sample:last_sample
            total += exp(@inbounds(logweights[sample_index]) - maximum_logweight)
        end
        if isfinite(total) && total > zero(total)
            threshold = @inbounds(uniforms[slot]) * total
            cumulative = zero(total)
            ancestor = last_sample
            for sample_index in first_sample:last_sample
                cumulative += exp(
                    @inbounds(logweights[sample_index]) - maximum_logweight,
                )
                if threshold < cumulative
                    ancestor = sample_index
                    break
                end
            end
        end
    end
    @inbounds ancestors[slot] = ancestor
    if ancestor > 0
        _gather_resampled_sample!(destination, source, slot, ancestor)
    end
end

@kernel function _local_logweight_maxima_kernel!(
    proposal_maxima,
    logweights,
    counts,
    round,
)
    proposal_slot = @index(Group, Linear)
    lane = @index(Local, Linear)
    @uniform lane_count = @groupsize()[1]
    maxima = @localmem eltype(logweights) (_LOCAL_REDUCTION_WORKGROUP_SIZE,)
    group = @localmem Int (2,)

    if lane == 1
        first_sample = 1
        for prior_slot in 1:(proposal_slot - 1)
            first_sample += @inbounds counts[prior_slot, round]
        end
        @inbounds group[1] = first_sample
        @inbounds group[2] = first_sample + counts[proposal_slot, round] - 1
    end
    @synchronize()

    lane_maximum = eltype(logweights)(-Inf)
    for sample_index in (@inbounds(group[1]) + lane - 1):lane_count:(@inbounds(group[2]))
        lane_maximum = max(lane_maximum, @inbounds(logweights[sample_index]))
    end
    @inbounds maxima[lane] = lane_maximum
    @synchronize()
    for offset in _LOCAL_REDUCTION_OFFSETS
        if lane <= offset
            @inbounds maxima[lane] = max(maxima[lane], maxima[lane + offset])
        end
        @synchronize()
    end
    if lane == 1
        @inbounds proposal_maxima[proposal_slot] = maxima[1]
    end
end

@kernel function _local_scaled_weights_kernel!(
    scaled_weights,
    logweights,
    assignments,
    proposal_maxima,
    round,
)
    sample_index = @index(Global, Linear)
    proposal_slot = @inbounds assignments[sample_index, round]
    maximum_logweight = @inbounds proposal_maxima[proposal_slot]
    @inbounds scaled_weights[sample_index] = maximum_logweight == eltype(logweights)(-Inf) ?
                                               zero(eltype(logweights)) :
                                               exp(logweights[sample_index] - maximum_logweight)
end

@kernel function _cooperative_local_resample_locations_kernel!(
    destination,
    ancestors,
    source,
    scaled_weights,
    uniforms,
    counts,
    round,
)
    proposal_slot = @index(Group, Linear)
    lane = @index(Local, Linear)
    @uniform lane_count = @groupsize()[1]
    chunk_totals = @localmem eltype(scaled_weights) (
        _LOCAL_REDUCTION_WORKGROUP_SIZE,
    )
    group = @localmem Int (4,)
    threshold = @localmem eltype(scaled_weights) (1,)

    if lane == 1
        first_sample = 1
        for prior_slot in 1:(proposal_slot - 1)
            first_sample += @inbounds counts[prior_slot, round]
        end
        @inbounds group[1] = first_sample
        @inbounds group[2] = first_sample + counts[proposal_slot, round] - 1
        @inbounds group[3] = 0
        @inbounds group[4] = 0
    end
    @synchronize()

    sample_count = @inbounds(group[2]) - @inbounds(group[1]) + 1
    chunk_size = (sample_count + lane_count - 1) ÷ lane_count
    chunk_first = @inbounds(group[1]) + (lane - 1) * chunk_size
    chunk_last = min(chunk_first + chunk_size - 1, @inbounds(group[2]))
    lane_total = zero(eltype(scaled_weights))
    for sample_index in chunk_first:chunk_last
        lane_total += @inbounds scaled_weights[sample_index]
    end
    @inbounds chunk_totals[lane] = lane_total
    @synchronize()

    if lane == 1
        total = zero(eltype(scaled_weights))
        for other_lane in 1:lane_count
            total += @inbounds chunk_totals[other_lane]
        end
        if isfinite(total) && total > zero(total)
            draw = @inbounds(uniforms[proposal_slot]) * total
            prefix = zero(total)
            chosen_lane = min(lane_count, sample_count)
            remainder = @inbounds chunk_totals[chosen_lane]
            for other_lane in 1:lane_count
                next_prefix = prefix + @inbounds(chunk_totals[other_lane])
                if draw < next_prefix
                    chosen_lane = other_lane
                    remainder = draw - prefix
                    break
                end
                prefix = next_prefix
            end
            @inbounds group[3] = chosen_lane
            @inbounds threshold[1] = remainder
        end
    end
    @synchronize()

    if lane == @inbounds(group[3])
        cumulative = zero(eltype(scaled_weights))
        ancestor = chunk_last
        for sample_index in chunk_first:chunk_last
            cumulative += @inbounds scaled_weights[sample_index]
            if @inbounds(threshold[1]) < cumulative
                ancestor = sample_index
                break
            end
        end
        @inbounds ancestors[proposal_slot] = ancestor
        _gather_resampled_sample!(destination, source, proposal_slot, ancestor)
        @inbounds group[4] = 1
    end
    @synchronize()
    if lane == 1 && @inbounds(group[4]) == 0
        @inbounds ancestors[proposal_slot] = 0
    end
end

function _resample_dm_pmc_population!(
    rng,
    cdf,
    uniforms,
    ancestors,
    source,
    destination,
    logweights,
    _assignments,
    _proposal_maxima,
    _counts,
    _round,
    execution,
    transfers,
    ::GlobalResampling,
)
    _resampling_cdf!(cdf, logweights, transfers)
    Random.rand!(rng, uniforms)
    _resample_and_gather!(
        cdf,
        uniforms,
        ancestors,
        source,
        destination,
        execution,
    )
    return nothing
end

function _resample_dm_pmc_population!(
    rng,
    scaled_weights,
    uniforms,
    ancestors,
    source,
    destination,
    logweights,
    assignments,
    proposal_maxima,
    counts,
    round,
    execution,
    transfers,
    ::LocalResampling,
)
    backend = KernelAbstractions.get_backend(logweights)
    if backend isa KernelAbstractions.CPU
        Random.rand!(rng, uniforms)
        kernel = _local_resample_locations_kernel!(backend)
        kernel(
            destination,
            ancestors,
            source,
            logweights,
            uniforms,
            counts,
            round;
            ndrange=length(ancestors),
            workgroupsize=_native_workgroupsize(execution, length(ancestors)),
        )
    else
        maxima_kernel = _local_logweight_maxima_kernel!(backend)
        maxima_kernel(
            proposal_maxima,
            logweights,
            counts,
            round;
            ndrange=_LOCAL_REDUCTION_WORKGROUP_SIZE * length(ancestors),
            workgroupsize=_LOCAL_REDUCTION_WORKGROUP_SIZE,
        )
        scale_kernel = _local_scaled_weights_kernel!(backend)
        scale_kernel(
            scaled_weights,
            logweights,
            assignments,
            proposal_maxima,
            round;
            ndrange=length(logweights),
            workgroupsize=_native_workgroupsize(execution, length(logweights)),
        )
        Random.rand!(rng, uniforms)
        select_kernel = _cooperative_local_resample_locations_kernel!(backend)
        select_kernel(
            destination,
            ancestors,
            source,
            scaled_weights,
            uniforms,
            counts,
            round;
            ndrange=_LOCAL_REDUCTION_WORKGROUP_SIZE * length(ancestors),
            workgroupsize=_LOCAL_REDUCTION_WORKGROUP_SIZE,
        )
    end
    KernelAbstractions.synchronize(backend)
    valid = minimum(ancestors) > 0
    _record_device_scalar_transfer!(
        transfers,
        ancestors,
        Int,
        Val(:local_resampling_validity),
    )
    valid || throw(AllZeroWeightsError())
    return nothing
end

_population_round_error(::DeterministicMixturePMC) = DMPMCRoundError
_population_round_phase(::DeterministicMixturePMC) = :resampling
_population_round_views(method_state::_PreparedDMPMC, round) =
    _dm_pmc_round_views(method_state, round)
_population_mis_adaptation(::_PreparedDMPMC, round_views) = nothing

function _advance_population!(
    sampler,
    method_state::_PreparedDMPMC,
    round_views,
    round,
    execution,
    transfers,
)
    workspace = method_state.workspace
    buffers = sampler.random_buffers
    _resample_dm_pmc_population!(
        sampler.rng,
        round_views.cdf,
        buffers.resampling_uniforms,
        workspace.ancestors,
        round_views.samples,
        workspace.candidate_locations,
        round_views.logweights,
        round_views.assignments,
        workspace.resampling_maxima,
        method_state.plan.counts,
        round,
        execution,
        transfers,
        sampler.algorithm.resampling,
    )
    return nothing
end

function _population_diagnostics(
    sampler,
    method_state::_PreparedDMPMC,
    execution,
    schedule,
    round_ess,
    round_lognormalizers,
    transfers,
)
    return (
        method=:deterministic_mixture_pmc,
        execution=_execution_name(execution),
        threaded=sampler.threaded,
        factor_execution_policy=_use_factor_batch_mis_path(
            sampler.device,
            method_state.run_bank,
            _population_denominator(method_state, firstindex(schedule)),
            eltype(method_state.workspace.round_logweights),
            sampler.factor_execution,
        ) ? :batched : :fused,
        rounds=sampler.algorithm.rounds,
        round_sizes=collect(schedule),
        round_ess=round_ess,
        round_lognormalizers=round_lognormalizers,
        resampling=_pmc_resampling_name(sampler.algorithm.resampling),
        failures=0,
        transfers=transfers,
    )
end

function _importance_sample_cpu!(sampler, method_state::_PreparedDMPMC, threaded)
    return _importance_sample_fixed_population!(sampler, method_state, threaded)
end
