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

struct _DMPMCWorkspace{S,W,I,Q,C,A,L}
    round_samples::S
    round_logweights::W
    round_proposal_ids::I
    solve_scratch::Q
    resampling_cdf::C
    ancestors::A
    candidate_locations::L
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

function _capture_dm_pmc_round(f, round, phase, round_size, completed_rounds)
    try
        return f()
    catch cause
        cause isa DMPMCRoundError && rethrow()
        throw(
            DMPMCRoundError(
                round,
                phase,
                cause,
                (round_size=round_size, completed_rounds=completed_rounds),
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
    committed_bank = method_state.bank
    bank = method_state.run_bank
    copyto!(bank.locations, committed_bank.locations)
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
            denominator = _RealizedMixtureDenominator(plan.logcoefficients, round)
            launch = _use_factor_batch_mis_path(
                sampler.device,
                bank,
                denominator,
                eltype(round_logweights),
                sampler.factor_execution,
            ) ? _launch_factor_batch_mis_round! : _launch_mis_round!
            launch(
                round_samples,
                _MISRoundOutput(round_logweights, round_proposal_ids, nothing),
                buffers.failure_scratch.record.storage,
                buffers.normals,
                target_evaluator,
                bank,
                assignments,
                denominator,
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
            _resampling_cdf!(cdf, round_logweights, transfers)
            Random.rand!(sampler.rng, buffers.resampling_uniforms)
            _resample_and_gather!(
                cdf,
                buffers.resampling_uniforms,
                workspace.ancestors,
                round_samples,
                workspace.candidate_locations,
                execution,
            )
            _logweight_summary(round_logweights, transfers)
        end
        output_indices = plan.offsets[round]:(plan.offsets[round + 1] - 1)
        _capture_dm_pmc_round(round, :commit_output, round_size, round - 1) do
            destination = _sample_view(samples, output_indices)
            copyto!(destination, round_samples)
            copyto!(view(logweights, output_indices), round_logweights)
            fill!(view(round_ids, output_indices), round)
            copyto!(view(proposal_ids, output_indices), round_proposal_ids)
        end
        _capture_dm_pmc_round(round, :advance_population, round_size, round - 1) do
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
        factor_execution_policy=_factor_execution_name(
            sampler.device,
            sampler.factor_execution,
        ),
        rounds=sampler.algorithm.rounds,
        round_sizes=collect(plan.schedule),
        round_ess=round_ess,
        round_lognormalizers=round_lognormalizers,
        failures=0,
        transfers=transfers,
    )
    final_round = lastindex(plan.schedule)
    result = _capture_dm_pmc_round(
        final_round,
        :result_construction,
        plan.schedule[final_round],
        final_round,
    ) do
        _adopt_validated_weighted_samples(
            samples,
            logweights;
            provenance=(round=round_ids, proposal_id=proposal_ids),
            diagnostics=diagnostics,
        )
    end
    method_state.bank, method_state.run_bank = bank, committed_bank
    return result
end
