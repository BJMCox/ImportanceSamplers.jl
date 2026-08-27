"""
    DMPMCRoundError

Exception thrown when a DM-PMC round cannot complete atomically. `round` and
`phase` locate the failure, `cause` stores the underlying exception, and
`diagnostics` reports the requested round size and number of previously
committed rounds. The prepared sampler retains its last committed proposal
population.
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

struct _DMPMCWorkspace{S,W,I,Q,C,A,L}
    round_samples::S
    round_logweights::W
    round_proposal_ids::I
    solve_scratch::Q
    resampling_cdf::C
    ancestors::A
    candidate_locations::L
end

struct _DMPMCRoundDenominator{L}
    logcoefficients::L
    round::Int
end
Adapt.@adapt_structure _DMPMCRoundDenominator

@inline _mis_term_bounds(bank, ::_DMPMCRoundDenominator, generating_slot) =
    (1, _active_proposal_count(bank))

@inline function _mis_denominator_term(
    ::Type{T},
    bank,
    denominator::_DMPMCRoundDenominator,
    term_index,
) where {T}
    return term_index, convert(
        T,
        @inbounds(denominator.logcoefficients[term_index, denominator.round]),
    )
end

_dm_pmc_binding_sample(
    bank::_PackedDiagonalGaussianBank{L,S,N,M,C,I,<:_ScalarGaussianLayout},
) where {L,S,N,M,C,I} = zero(eltype(bank.locations))

_dm_pmc_binding_sample(bank::_PackedDiagonalGaussianBank) =
    view(bank.locations, :, firstindex(bank.locations, 2))

_dm_pmc_binding_sample(bank::_PackedFactorGaussianBank) =
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
        ancestors,
        candidate_locations,
    )
end

@kernel function _dm_pmc_finalize_cdf_kernel!(cdf, last_index)
    @inbounds cdf[last_index] = one(eltype(cdf))
end

@kernel function _dm_pmc_select_ancestors_kernel!(ancestors, uniforms, cdf, last_index)
    proposal_slot = @index(Global, Linear)
    uniform = @inbounds uniforms[proposal_slot]
    first = 1
    last = last_index
    while first < last
        middle = first + ((last - first) >> 1)
        if uniform < @inbounds(cdf[middle])
            last = middle
        else
            first = middle + 1
        end
    end
    @inbounds ancestors[proposal_slot] = first
end

@inline function _dm_pmc_gather_sample!(
    candidates::AbstractVector,
    samples::AbstractVector,
    proposal_slot,
    ancestor,
)
    @inbounds candidates[proposal_slot] = samples[ancestor]
    return nothing
end

@inline function _dm_pmc_gather_sample!(
    candidates::AbstractMatrix,
    samples::AbstractMatrix,
    proposal_slot,
    ancestor,
)
    for coordinate in axes(candidates, 1)
        @inbounds candidates[coordinate, proposal_slot] = samples[coordinate, ancestor]
    end
    return nothing
end

@kernel function _dm_pmc_gather_ancestors_kernel!(candidates, samples, ancestors)
    proposal_slot = @index(Global, Linear)
    ancestor = @inbounds ancestors[proposal_slot]
    _dm_pmc_gather_sample!(candidates, samples, proposal_slot, ancestor)
end

function _launch_dm_pmc_resampling!(
    cdf,
    uniforms,
    ancestors,
    samples,
    candidate_locations,
    execution,
)
    backend = KernelAbstractions.get_backend(cdf)
    last_index = length(cdf)
    finalize = _dm_pmc_finalize_cdf_kernel!(backend)
    finalize(cdf, last_index; ndrange=1, workgroupsize=1)
    select = _dm_pmc_select_ancestors_kernel!(backend)
    select(
        ancestors,
        uniforms,
        cdf,
        last_index;
        ndrange=length(ancestors),
        workgroupsize=_native_workgroupsize(execution, length(ancestors)),
    )
    gather = _dm_pmc_gather_ancestors_kernel!(backend)
    gather(
        candidate_locations,
        samples,
        ancestors;
        ndrange=length(ancestors),
        workgroupsize=_native_workgroupsize(execution, length(ancestors)),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

function _dm_pmc_resampling_cdf!(
    cdf,
    logweights,
    transfers::_ResultTransferCounter=_ResultTransferCounter(0, 0),
)
    maximum_logweight = maximum(logweights)
    _record_device_scalar_transfer!(
        transfers,
        logweights,
        eltype(logweights),
        Val(:cdf_maximum),
    )
    maximum_logweight == -Inf && throw(AllZeroWeightsError())
    cdf .= exp.(logweights .- maximum_logweight)
    total = sum(cdf)
    _record_device_scalar_transfer!(
        transfers,
        cdf,
        eltype(cdf),
        Val(:cdf_sum),
    )
    isfinite(total) && total > zero(total) || throw(AllZeroWeightsError())
    cdf ./= total
    cumsum!(cdf, cdf)
    return cdf
end

function _dm_pmc_round_summary(
    logweights,
    transfers::_ResultTransferCounter=_ResultTransferCounter(0, 0),
)
    maximum_logweight = maximum(logweights)
    _record_device_scalar_transfer!(
        transfers,
        logweights,
        eltype(logweights),
        Val(:summary_maximum),
    )
    scaled_sum = mapreduce(
        value -> exp(value - maximum_logweight),
        +,
        logweights;
        init=zero(eltype(logweights)),
    )
    _record_device_scalar_transfer!(
        transfers,
        logweights,
        eltype(logweights),
        Val(:summary_scaled_sum),
    )
    scaled_square_sum = mapreduce(
        value -> abs2(exp(value - maximum_logweight)),
        +,
        logweights;
        init=zero(eltype(logweights)),
    )
    _record_device_scalar_transfer!(
        transfers,
        logweights,
        eltype(logweights),
        Val(:summary_scaled_square_sum),
    )
    T = eltype(logweights)
    return (
        ess=abs2(scaled_sum) / scaled_square_sum,
        lognormalizer=maximum_logweight + log(scaled_sum) -
                      log(T(length(logweights))),
    )
end

function _capture_dm_pmc_round(f, round, phase, round_size, committed_rounds)
    try
        return f()
    catch cause
        cause isa DMPMCRoundError && rethrow()
        throw(
            DMPMCRoundError(
                round,
                phase,
                cause,
                (round_size=round_size, committed_rounds=committed_rounds),
            ),
        )
    end
end

function _importance_sample_cpu!(
    sampler,
    method_state::_PreparedDMPMC,
    threaded,
)
    execution = threaded ? _ThreadedCPUExecution() : _SerialCPUExecution()
    bank = method_state.bank
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
    target = _capture_dm_pmc_round(1, :sample_and_weight, plan.schedule[1], 0) do
        _bind_resolved_target(sampler.target, _dm_pmc_binding_sample(bank))
    end
    target_evaluator, target_failures = _capture_dm_pmc_round(
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
        round_views = _dm_pmc_round_views(method_state, round)
        round_size = round_views.round_size
        round_samples = round_views.samples
        round_logweights = round_views.logweights
        round_proposal_ids = round_views.proposal_ids
        assignments = round_views.assignments
        cdf = round_views.cdf

        _capture_dm_pmc_round(round, :normal_buffer, round_size, round - 1) do
            Random.randn!(sampler.rng, buffers.normals)
        end
        _capture_dm_pmc_round(round, :sample_and_weight, round_size, round - 1) do
            _launch_mis_round!(
                round_samples,
                _MISRoundOutput(round_logweights, round_proposal_ids),
                buffers.failure_scratch.record.storage,
                buffers.normals,
                target_evaluator,
                bank,
                assignments,
                _DMPMCRoundDenominator(plan.logcoefficients, round),
                workspace.solve_scratch,
                execution,
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
        summary = _capture_dm_pmc_round(round, :resampling, round_size, round - 1) do
            _dm_pmc_resampling_cdf!(cdf, round_logweights, transfers)
            Random.rand!(sampler.rng, buffers.resampling_uniforms)
            _launch_dm_pmc_resampling!(
                cdf,
                buffers.resampling_uniforms,
                workspace.ancestors,
                round_samples,
                workspace.candidate_locations,
                execution,
            )
            _dm_pmc_round_summary(round_logweights, transfers)
        end
        output_indices = plan.offsets[round]:(plan.offsets[round + 1] - 1)
        _capture_dm_pmc_round(round, :commit_output, round_size, round - 1) do
            destination = _sample_view(samples, output_indices)
            copyto!(destination, round_samples)
            copyto!(view(logweights, output_indices), round_logweights)
            fill!(view(round_ids, output_indices), round)
            copyto!(view(proposal_ids, output_indices), round_proposal_ids)
        end
        _capture_dm_pmc_round(round, :commit_population, round_size, round - 1) do
            copyto!(bank.locations, workspace.candidate_locations)
            KernelAbstractions.synchronize(
                KernelAbstractions.get_backend(bank.locations),
            )
        end
        _capture_dm_pmc_round(round, :diagnostics, round_size, round) do
            round_ess[round] = summary.ess
            round_lognormalizers[round] = summary.lognormalizer
        end
    end

    diagnostics = (
        method=:deterministic_mixture_pmc,
        execution=_execution_name(execution),
        threaded=sampler.threaded,
        rounds=sampler.algorithm.rounds,
        round_sizes=collect(plan.schedule),
        round_ess=round_ess,
        round_lognormalizers=round_lognormalizers,
        failures=0,
        transfers=transfers,
    )
    return _adopt_validated_weighted_samples(
        samples,
        logweights;
        provenance=(round=round_ids, proposal_id=proposal_ids),
        diagnostics=diagnostics,
    )
end
