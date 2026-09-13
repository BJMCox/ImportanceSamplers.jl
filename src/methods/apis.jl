mutable struct _ValidatedAPISToken end
const _VALIDATED_APIS_TOKEN = _ValidatedAPISToken()

"""
    APIS(bank; rounds, round_size)

Configure adaptive population importance sampling with a fixed Gaussian or Student-t
proposal population. Student-t locations adapt, but each positive `nu` and
scale stays fixed. A bank uses one radial family. Each round is one APIS epoch. `round_size` is the total
number of samples in an epoch and may be a positive `Int` or a vector with one
entry per epoch.

APIS requires equal positive proposal masses, equal allocation within every
epoch, and at least two samples per proposal. It updates proposal means after
each epoch while retaining the configured scale factors.
"""
struct APIS{B<:ProposalBank,S} <: _FixedPopulationSampler
    bank::B
    rounds::Int
    round_size::S

    function APIS(
        bank::B,
        rounds::Int,
        round_size::S,
        token::_ValidatedAPISToken,
    ) where {B<:ProposalBank,S}
        token === _VALIDATED_APIS_TOKEN || throw(
            ArgumentError("invalid internal algorithm-construction token"),
        )
        return new{B,S}(bank, rounds, round_size)
    end
end

function APIS(bank::ProposalBank; rounds, round_size)
    validated_round_size = _validate_adaptive_schedule(rounds, round_size)
    masses = bank.masses
    all(>(zero(eltype(masses))), masses) || throw(
        ArgumentError("APIS requires every proposal mass to be positive"),
    )
    all(==(first(masses)), masses) || throw(
        ArgumentError("APIS requires equal proposal masses"),
    )
    schedule = _resolve_adaptive_schedule(rounds, validated_round_size)
    proposal_count = length(bank.proposals)
    for (round, count) in pairs(schedule)
        count % proposal_count == 0 || throw(
            ArgumentError(
                "APIS round $round of size $count must allocate equally " *
                "across $proposal_count proposals",
            ),
        )
        count >= 2 * proposal_count || throw(
            ArgumentError(
                "APIS round $round must allocate at least two samples per proposal",
            ),
        )
    end
    return APIS(
        bank,
        rounds,
        validated_round_size,
        _VALIDATED_APIS_TOKEN,
    )
end

_algorithm_proposal(algorithm::APIS) = algorithm.bank
_algorithm_sample_budget(algorithm::APIS) =
    _adaptive_sample_budget(algorithm.rounds, algorithm.round_size)

mutable struct _PreparedAPIS{B,P,W}
    bank::B
    run_bank::B
    plan::P
    workspace::W
end

_accelerator_method_state_limit(
    ::_PreparedAPIS{<:Union{
        _PackedDiagonalBank,
        _PackedFactorBank,
    }},
) = nothing

function _prepare_method_state(algorithm::APIS)
    bank, plan = _prepare_fixed_population_state(algorithm)
    workspace = _allocate_apis_workspace(
        bank,
        plan,
        eltype(bank.lognormalizers),
    )
    return _PreparedAPIS(bank, _population_run_bank(bank), plan, workspace)
end

function _prepare_method_state(algorithm::APIS, prepared_target)
    bank, plan = _prepare_fixed_population_state(algorithm)
    binding_sample = _population_binding_sample(bank)
    target = _bind_resolved_target(prepared_target, binding_sample)
    log_type = _resolve_packed_static_mis_logweight_type(
        target,
        bank,
        typeof(binding_sample),
    )
    workspace = _allocate_apis_workspace(bank, plan, log_type)
    return _PreparedAPIS(bank, _population_run_bank(bank), plan, workspace)
end

function _allocate_random_buffers(
    ::MLDataDevices.AbstractDevice,
    ::ProposalBank,
    method_state::_PreparedAPIS,
    sample_budget,
)
    bank = method_state.bank
    maximum_round_size = maximum(method_state.plan.schedule)
    normals = similar(
        bank.locations,
        eltype(bank.locations),
        size(bank.locations, 1) * maximum_round_size,
    )
    failure_scratch = _allocate_native_failure_scratch(
        normals,
        maximum_round_size,
    )
    return _PopulationNormalBuffers(normals, failure_scratch,
        _allocate_radial_buffers(normals, bank.family, maximum_round_size))
end

function _apis_round_views(method_state::_PreparedAPIS, round)
    plan = method_state.plan
    workspace = method_state.workspace
    round_size = plan.schedule[round]
    return (
        round_size=round_size,
        samples=_sample_view(workspace.round_samples, 1:round_size),
        logweights=view(workspace.round_logweights, 1:round_size),
        proposal_ids=view(workspace.round_proposal_ids, 1:round_size),
        assignments=view(plan.assignments, 1:round_size, round),
        logtargets=view(workspace.round_logtargets, 1:round_size),
        generating_logdensities=view(
            workspace.round_generating_logdensities,
            1:round_size,
        ),
        scaled_local_weights=view(workspace.scaled_local_weights, 1:round_size),
    )
end

function _copy_accelerator_algorithm(device, algorithm::APIS, ::_PreparedAPIS)
    return deepcopy(algorithm)
end

function _prepare_transferred_method_state(
    device,
    algorithm::APIS,
    method_state::_PreparedAPIS,
    _transferred_target,
)
    plan = method_state.plan
    transferred_plan = _transfer_population_plan(device, plan)
    workspace = method_state.workspace
    transferred_workspace = _APISWorkspace(
        _copy_to_device(device, workspace.round_samples),
        _copy_to_device(device, workspace.round_logweights),
        _copy_to_device(device, workspace.round_proposal_ids),
        _copy_to_device(device, workspace.round_logtargets),
        _copy_to_device(device, workspace.round_generating_logdensities),
        _copy_to_device(device, workspace.scaled_local_weights),
        _copy_to_device(device, workspace.proposal_maxima),
        _copy_to_device(device, workspace.candidate_locations),
        _copy_to_device(device, workspace.solve_scratch),
    )
    transferred_bank = _copy_packed_bank(device, method_state.bank)
    return _PreparedAPIS(
        transferred_bank,
        _population_run_bank(transferred_bank),
        transferred_plan,
        transferred_workspace,
    )
end

_transferred_backend_state(
    algorithm,
    method_state::_PreparedAPIS,
    target,
    random_buffers,
) = (method_state, target, random_buffers)

_prepared_backend_state(sampler, method_state::_PreparedAPIS) = (
    method_state,
    sampler.target,
    sampler.random_buffers,
    sampler.rng,
)

function _copy_algorithm(device, algorithm::APIS)
    return APIS(
        _copy_population_bank(device, algorithm.bank);
        rounds=algorithm.rounds,
        round_size=algorithm.round_size,
    )
end

_copy_algorithm(::MLDataDevices.CPUDevice{Missing}, algorithm::APIS) =
    deepcopy(algorithm)

function _retarget_algorithm(
    sampler::_PreparedImportanceSampler{R,B,T,A},
) where {R,B,T,A<:APIS}
    algorithm = sampler.algorithm
    return APIS(
        current_proposal(MLDataDevices.cpu_device(), sampler);
        rounds=algorithm.rounds,
        round_size=algorithm.round_size,
    )
end

function current_proposal(
    sampler::_PreparedImportanceSampler{R,B,T,A,M,D},
) where {R,B,T,A<:APIS,M,D}
    return _current_population_proposal(sampler)
end

function current_proposal(
    destination::MLDataDevices.AbstractCPUDevice,
    sampler::_PreparedImportanceSampler{R,B,T,A,M,D},
) where {R,B,T,A<:APIS,M,D}
    return _current_population_proposal(destination, sampler)
end

function current_proposal(
    destination::MLDataDevices.AbstractDevice,
    sampler::_PreparedImportanceSampler{R,B,T,A,M,D},
) where {R,B,T,A<:APIS,M,D}
    return _current_population_proposal(destination, sampler)
end
