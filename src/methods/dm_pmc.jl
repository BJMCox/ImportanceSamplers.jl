mutable struct _ValidatedDeterministicMixturePMCToken end
const _VALIDATED_DETERMINISTIC_MIXTURE_PMC_TOKEN =
    _ValidatedDeterministicMixturePMCToken()

"""
    AbstractPMCResamplingPolicy

Abstract supertype for DM-PMC ancestor-selection scopes.
"""
abstract type AbstractPMCResamplingPolicy end

"""
    GlobalResampling()

Select every next-round proposal location from the complete weighted round.
"""
struct GlobalResampling <: AbstractPMCResamplingPolicy end

"""
    LocalResampling()

Select each next-round proposal location from its own weighted sample group.
"""
struct LocalResampling <: AbstractPMCResamplingPolicy end

_pmc_resampling_name(::GlobalResampling) = :global
_pmc_resampling_name(::LocalResampling) = :local

"""
    DeterministicMixturePMC(
        bank;
        rounds,
        round_size,
        resampling=GlobalResampling(),
    )

Configure deterministic-mixture population Monte Carlo with a fixed round
schedule. `round_size` is either one positive `Int` repeated for every round or
a positive `Vector{Int}` with one entry per round. `resampling` selects global
or local ancestor selection while keeping the same deterministic-mixture
weights.

Native Gaussian and Student-t populations use one radial family. Adaptation
changes locations and preserves each scale and positive Student-t degrees of freedom.
"""
struct DeterministicMixturePMC{
    B<:ProposalBank,
    S,
    R<:AbstractPMCResamplingPolicy,
} <: _FixedPopulationSampler
    bank::B
    rounds::Int
    round_size::S
    resampling::R

    function DeterministicMixturePMC(
        bank::B,
        rounds::Int,
        round_size::S,
        resampling::R,
        token::_ValidatedDeterministicMixturePMCToken,
    ) where {B<:ProposalBank,S,R<:AbstractPMCResamplingPolicy}
        token === _VALIDATED_DETERMINISTIC_MIXTURE_PMC_TOKEN || throw(
            ArgumentError("invalid internal algorithm-construction token"),
        )
        return new{B,S,R}(bank, rounds, round_size, resampling)
    end
end

function DeterministicMixturePMC(
    bank::ProposalBank;
    rounds,
    round_size,
    resampling::AbstractPMCResamplingPolicy=GlobalResampling(),
)
    validated_round_size = _validate_adaptive_schedule(rounds, round_size)
    return DeterministicMixturePMC(
        bank,
        rounds,
        validated_round_size,
        resampling,
        _VALIDATED_DETERMINISTIC_MIXTURE_PMC_TOKEN,
    )
end

_algorithm_proposal(algorithm::DeterministicMixturePMC) = algorithm.bank
_algorithm_sample_budget(algorithm::DeterministicMixturePMC) =
    _adaptive_sample_budget(algorithm.rounds, algorithm.round_size)

function _retarget_algorithm(
    sampler::_PreparedImportanceSampler{R,B,T,A},
) where {R,B,T,A<:DeterministicMixturePMC}
    algorithm = sampler.algorithm
    return DeterministicMixturePMC(
        current_proposal(MLDataDevices.cpu_device(), sampler);
        rounds=algorithm.rounds,
        round_size=algorithm.round_size,
        resampling=algorithm.resampling,
    )
end

"""
    current_proposal(sampler)
    current_proposal(destination, sampler)

Return an independent snapshot of the proposal population currently owned by a
CPU-prepared [`DeterministicMixturePMC`](@ref) sampler. Active proposals use the
latest adapted locations; their configured scales or factors and masses remain
fixed. Inert zero-mass proposals retain their configured positions and stable
IDs.

The returned [`ProposalBank`](@ref) does not alias the sampler. The one-argument
form is CPU-only and never hides a device transfer. For an accelerator-prepared
sampler, pass an explicit preserving CPU destination:
`current_proposal(MLDataDevices.cpu_device(), sampler)`. That form copies only
the current packed locations and stable proposal IDs under the sampler's
physical-device scope; it does not migrate the prepared sampler, RNG,
workspaces, target, or result storage. Scalar-converting and non-CPU
destinations are rejected. [`APIS`](@ref) uses the same population snapshot
and transfer contract.
"""
function current_proposal(
    sampler::_PreparedImportanceSampler{R,B,T,A,M,D},
) where {
    R,
    B,
    T,
    A<:DeterministicMixturePMC,
    M,
    D,
}
    return _current_population_proposal(sampler)
end

function current_proposal(
    destination::MLDataDevices.AbstractCPUDevice,
    sampler::_PreparedImportanceSampler{R,B,T,A,M,D},
) where {
    R,
    B,
    T,
    A<:DeterministicMixturePMC,
    M,
    D,
}
    return _current_population_proposal(destination, sampler)
end

function current_proposal(
    destination::MLDataDevices.AbstractDevice,
    sampler::_PreparedImportanceSampler{R,B,T,A,M,D},
) where {
    R,
    B,
    T,
    A<:DeterministicMixturePMC,
    M,
    D,
}
    return _current_population_proposal(destination, sampler)
end

mutable struct _PreparedDMPMC{B,P,W}
    bank::B
    run_bank::B
    plan::P
    workspace::W
end

_accelerator_method_state_limit(
    ::_PreparedDMPMC{<:Union{
        _PackedDiagonalBank,
        _PackedFactorBank,
    }},
) = nothing

function _prepare_method_state(algorithm::DeterministicMixturePMC)
    bank, plan = _prepare_fixed_population_state(algorithm)
    workspace = _allocate_dm_pmc_workspace(
        bank,
        plan,
        eltype(bank.lognormalizers),
    )
    return _PreparedDMPMC(bank, _population_run_bank(bank), plan, workspace)
end

function _prepare_method_state(algorithm::DeterministicMixturePMC, prepared_target)
    bank, plan = _prepare_fixed_population_state(algorithm)
    binding_sample = _population_binding_sample(bank)
    target = _bind_resolved_target(prepared_target, binding_sample)
    log_type = _resolve_packed_static_mis_logweight_type(
        target,
        bank,
        typeof(binding_sample),
    )
    workspace = _allocate_dm_pmc_workspace(bank, plan, log_type)
    return _PreparedDMPMC(bank, _population_run_bank(bank), plan, workspace)
end

function _allocate_random_buffers(
    ::MLDataDevices.AbstractDevice,
    ::ProposalBank,
    method_state::_PreparedDMPMC,
    sample_budget,
)
    bank = method_state.bank
    maximum_round_size = maximum(method_state.plan.schedule)
    normals = similar(
        bank.locations,
        eltype(bank.locations),
        size(bank.locations, 1) * maximum_round_size,
    )
    resampling_uniforms = similar(
        bank.cdf,
        eltype(bank.cdf),
        _active_proposal_count(bank),
    )
    failure_scratch = _allocate_native_failure_scratch(
        normals,
        maximum_round_size; capacity=max(sample_budget, _active_proposal_count(bank)),
    )
    return _DMPMCRandomBuffers(
        normals,
        resampling_uniforms,
        failure_scratch,
        _allocate_radial_buffers(normals, bank.family, maximum_round_size),
    )
end

function _dm_pmc_round_views(method_state::_PreparedDMPMC, round)
    plan = method_state.plan
    workspace = method_state.workspace
    round_size = plan.schedule[round]
    return (
        round_size=round_size,
        samples=_sample_view(workspace.round_samples, 1:round_size),
        logweights=view(workspace.round_logweights, 1:round_size),
        proposal_ids=view(workspace.round_proposal_ids, 1:round_size),
        assignments=view(plan.assignments, 1:round_size, round),
        cdf=view(workspace.resampling_cdf, 1:round_size),
    )
end

function _copy_accelerator_algorithm(
    device,
    algorithm::DeterministicMixturePMC,
    ::_PreparedDMPMC,
)
    return deepcopy(algorithm)
end

function _prepare_transferred_method_state(
    device,
    algorithm::DeterministicMixturePMC,
    method_state::_PreparedDMPMC,
    _transferred_target,
)
    plan = method_state.plan
    transferred_plan = _transfer_population_plan(device, plan)
    workspace = method_state.workspace
    transferred_workspace = _DMPMCWorkspace(
        _copy_to_device(device, workspace.round_samples),
        _copy_to_device(device, workspace.round_logweights),
        _copy_to_device(device, workspace.round_proposal_ids),
        _copy_to_device(device, workspace.solve_scratch),
        _copy_to_device(device, workspace.resampling_cdf),
        _copy_to_device(device, workspace.resampling_maxima),
        _copy_to_device(device, workspace.ancestors),
        _copy_to_device(device, workspace.candidate_locations),
    )
    transferred_bank = _copy_packed_bank(device, method_state.bank)
    return _PreparedDMPMC(
        transferred_bank,
        _population_run_bank(transferred_bank),
        transferred_plan,
        transferred_workspace,
    )
end

_transferred_backend_state(
    algorithm,
    method_state::_PreparedDMPMC,
    target,
    random_buffers,
) = (method_state, target, random_buffers)

_prepared_backend_state(sampler, method_state::_PreparedDMPMC) = (
    method_state,
    sampler.target,
    sampler.random_buffers,
    sampler.rng,
)

function _preflight_accelerator_method(
    device,
    target,
    algorithm::DeterministicMixturePMC,
    method_state::_PreparedDMPMC,
    buffers::_DMPMCRandomBuffers,
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
    target_argument = _native_device_evaluator(bound_target, log_type)
    backend = KernelAbstractions.get_backend(buffers.normals)
    representative_round = findmax(plan.schedule)[2]
    round_views = _dm_pmc_round_views(method_state, representative_round)

    round_kernel = _mis_round_launch_kernel!(backend)
    denominator =
        _RealizedMixtureDenominator(plan.logcoefficients, representative_round)
    use_factor_batch = _use_factor_batch_mis_path(
        device,
        bank,
        denominator,
        log_type,
        factor_execution,
    )
    solve_scratch = use_factor_batch ? workspace.solve_scratch :
                    _fused_mis_solve_scratch(workspace.solve_scratch, backend)
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
        solve_scratch,
    )
        _preflight_kernel_argument(device, round_kernel, argument)
    end
    if use_factor_batch
        batch_kernel = _factor_batch_mis_draw_target_kernel!(backend)
        _preflight_kernel_argument(device, batch_kernel, target_argument)
    end

    _preflight_dm_pmc_resampling(
        device,
        backend,
        round_views,
        workspace,
        buffers,
        plan,
        representative_round,
        algorithm.resampling,
    )
    return nothing
end

function _preflight_dm_pmc_resampling(
    device,
    backend,
    round_views,
    workspace,
    buffers,
    _plan,
    _round,
    ::GlobalResampling,
)
    select_kernel = _select_resampling_ancestors_kernel!(backend)
    for argument in (
        workspace.ancestors,
        buffers.resampling_uniforms,
        round_views.cdf,
        round_views.logweights,
        round_views.round_size,
    )
        _preflight_kernel_argument(device, select_kernel, argument)
    end
    gather_kernel = _gather_resampled_samples_kernel!(backend)
    for argument in (
        workspace.candidate_locations,
        round_views.samples,
        workspace.ancestors,
    )
        _preflight_kernel_argument(device, gather_kernel, argument)
    end
    return nothing
end

function _preflight_dm_pmc_resampling(
    device,
    backend,
    round_views,
    workspace,
    buffers,
    plan,
    round,
    ::LocalResampling,
)
    maxima_kernel = _local_logweight_maxima_kernel!(backend)
    for argument in (
        workspace.resampling_maxima,
        round_views.logweights,
        plan.counts,
        round,
    )
        _preflight_kernel_argument(device, maxima_kernel, argument)
    end
    scale_kernel = _local_scaled_weights_kernel!(backend)
    for argument in (
        round_views.cdf,
        round_views.logweights,
        round_views.assignments,
        workspace.resampling_maxima,
        round,
    )
        _preflight_kernel_argument(device, scale_kernel, argument)
    end
    select_kernel = _cooperative_local_resample_locations_kernel!(backend)
    for argument in (
        workspace.candidate_locations,
        workspace.ancestors,
        round_views.samples,
        round_views.cdf,
        buffers.resampling_uniforms,
        plan.counts,
        round,
    )
        _preflight_kernel_argument(device, select_kernel, argument)
    end
    return nothing
end

function _copy_algorithm(device, algorithm::DeterministicMixturePMC)
    return DeterministicMixturePMC(
        _copy_population_bank(device, algorithm.bank);
        rounds=algorithm.rounds,
        round_size=algorithm.round_size,
        resampling=algorithm.resampling,
    )
end

_copy_algorithm(
    ::MLDataDevices.CPUDevice{Missing},
    algorithm::DeterministicMixturePMC,
) = deepcopy(algorithm)
