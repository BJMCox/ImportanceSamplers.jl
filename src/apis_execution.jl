"""
    APISRoundError

Exception thrown when an APIS epoch cannot complete. `round` and `phase` locate
the failure, `cause` stores the underlying exception, and `diagnostics` reports
the requested epoch size and number of completed epochs. The prepared sampler
retains its pre-call proposal population.
"""
struct APISRoundError{E,D<:NamedTuple} <: Exception
    round::Int
    phase::Symbol
    cause::E
    diagnostics::D
end

function Base.showerror(io::IO, error::APISRoundError)
    print(io, "APIS failed in epoch ", error.round, " during ", error.phase, ": ")
    showerror(io, error.cause)
end

struct _APISWorkspace{S,W,I,T,G,L,M,C,Q}
    round_samples::S
    round_logweights::W
    round_proposal_ids::I
    round_logtargets::T
    round_generating_logdensities::G
    scaled_local_weights::L
    proposal_maxima::M
    candidate_locations::C
    solve_scratch::Q
end

function _allocate_apis_workspace(bank, plan, ::Type{T}) where {T}
    capacity = maximum(plan.schedule)
    prototype = bank.locations
    round_samples = _allocate_packed_static_mis_samples(prototype, bank, capacity)
    round_logweights = similar(prototype, T, capacity)
    round_proposal_ids = similar(prototype, Int, capacity)
    round_logtargets = similar(prototype, T, capacity)
    round_generating_logdensities = similar(prototype, T, capacity)
    scaled_local_weights = similar(prototype, T, capacity)
    proposal_maxima = similar(prototype, T, _active_proposal_count(bank))
    candidate_locations = _allocate_packed_static_mis_samples(
        prototype,
        bank,
        _active_proposal_count(bank),
    )
    solve_scratch = _allocate_mis_solve_scratch(prototype, bank, capacity)
    return _APISWorkspace(
        round_samples,
        round_logweights,
        round_proposal_ids,
        round_logtargets,
        round_generating_logdensities,
        scaled_local_weights,
        proposal_maxima,
        candidate_locations,
        solve_scratch,
    )
end

_population_round_error(::APIS) = APISRoundError
_population_round_phase(::APIS) = :adaptation
_population_round_views(method_state::_PreparedAPIS, round) =
    _apis_round_views(method_state, round)
_population_mis_adaptation(::_PreparedAPIS, round_views) =
    _MISAdaptationOutput(
        round_views.logtargets,
        round_views.generating_logdensities,
    )

function _advance_population!(
    sampler,
    method_state::_PreparedAPIS,
    round_views,
    round,
    execution,
    transfers,
)
    workspace = method_state.workspace
    _local_weighted_means!(
        workspace.candidate_locations,
        workspace.proposal_maxima,
        round_views.samples,
        round_views.scaled_local_weights,
        round_views.logtargets,
        round_views.generating_logdensities,
        round_views.assignments,
        method_state.plan.counts,
        round,
        execution,
        transfers,
    )
    return nothing
end

function _population_diagnostics(
    sampler,
    ::_PreparedAPIS,
    execution,
    schedule,
    round_ess,
    round_lognormalizers,
    transfers,
)
    proposal_count = length(sampler.algorithm.bank.proposals)
    total_samples = sum(schedule)
    return (
        method=:apis,
        execution=_execution_name(execution),
        threaded=sampler.threaded,
        factor_execution_policy=_factor_execution_name(
            sampler.device,
            sampler.factor_execution,
        ),
        rounds=sampler.algorithm.rounds,
        round_sizes=collect(schedule),
        round_ess=round_ess,
        round_lognormalizers=round_lognormalizers,
        target_evaluations=total_samples,
        proposal_evaluations=proposal_count * total_samples,
        failures=0,
        transfers=transfers,
    )
end

function _importance_sample_cpu!(sampler, method_state::_PreparedAPIS, threaded)
    return _importance_sample_fixed_population!(sampler, method_state, threaded)
end

function _preflight_accelerator_method(
    device,
    target,
    ::APIS,
    method_state::_PreparedAPIS,
    buffers::_PopulationNormalBuffers,
    factor_execution,
)
    bank = method_state.bank
    plan = method_state.plan
    workspace = method_state.workspace
    binding_sample = _population_binding_sample(bank)
    bound_target = _bind_resolved_target(target, binding_sample)
    log_type = _resolve_packed_static_mis_logweight_type(
        bound_target,
        bank,
        typeof(binding_sample),
    )
    target_argument = _NativeDeviceTarget{log_type,typeof(bound_target)}(bound_target)
    backend = KernelAbstractions.get_backend(buffers.normals)
    representative_round = findmax(plan.schedule)[2]
    round_views = _apis_round_views(method_state, representative_round)
    adaptation = _MISAdaptationOutput(
        round_views.logtargets,
        round_views.generating_logdensities,
    )
    denominator =
        _RealizedMixtureDenominator(plan.logcoefficients, representative_round)
    round_kernel = _mis_round_launch_kernel!(backend)
    for argument in (
        round_views.samples,
        round_views.logweights,
        round_views.proposal_ids,
        buffers.failure_scratch.record.storage,
        buffers.normals,
        target_argument,
        bank,
        round_views.assignments,
        denominator,
        workspace.solve_scratch,
        adaptation,
    )
        _preflight_kernel_argument(device, round_kernel, argument)
    end
    if _use_factor_batch_mis_path(
        device,
        bank,
        denominator,
        log_type,
        factor_execution,
    )
        draw_kernel = _factor_batch_mis_draw_target_kernel!(backend)
        for argument in (target_argument, adaptation)
            _preflight_kernel_argument(device, draw_kernel, argument)
        end
    end

    maxima_kernel = _local_logweight_maxima_kernel!(backend)
    for argument in (
        workspace.proposal_maxima,
        round_views.logtargets,
        round_views.generating_logdensities,
        plan.counts,
        representative_round,
        1,
    )
        _preflight_kernel_argument(device, maxima_kernel, argument)
    end
    scale_kernel = _scaled_local_weights_kernel!(backend)
    for argument in (
        round_views.scaled_local_weights,
        round_views.logtargets,
        round_views.generating_logdensities,
        round_views.assignments,
        workspace.proposal_maxima,
    )
        _preflight_kernel_argument(device, scale_kernel, argument)
    end
    means_kernel = _local_weighted_means_kernel!(backend)
    for argument in (
        workspace.candidate_locations,
        workspace.proposal_maxima,
        round_views.samples,
        round_views.scaled_local_weights,
        plan.counts,
        representative_round,
        round_views.samples isa AbstractVector ? 1 : size(round_views.samples, 1),
        1,
    )
        _preflight_kernel_argument(device, means_kernel, argument)
    end
    return nothing
end
