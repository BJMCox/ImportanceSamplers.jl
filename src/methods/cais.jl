mutable struct _ValidatedCAISToken end
const _VALIDATED_CAIS_TOKEN = _ValidatedCAISToken()

"""
    CAIS(bank; rounds, round_size, covariance_ess_threshold=nothing)

Configure canonical covariance-adaptive importance sampling for an equally
weighted population of native Gaussian or Student-t proposals. Student-t fits
require `nu > 2`, retain each component's `nu`, and convert fitted covariance
to Student-t scale. Each bank must use one radial family. `round_size` is the total
sample count in each round and may be a positive `Int` or a `Vector{Int}` with
one entry per round.

Every proposal receives the same number of samples. If
`covariance_ess_threshold` is `nothing`, the package uses
`max(d + 1, ceil(Int, 0.3m))` for dimension `d` and local count `m`; this is a
package default, not a default specified by the CAIS paper. An explicit real
threshold must satisfy `d < threshold < m` in every round.
"""
struct CAIS{B<:ProposalBank,S,E} <: _FixedPopulationSampler
    bank::B
    rounds::Int
    round_size::S
    covariance_ess_threshold::E

    function CAIS(
        bank::B,
        rounds::Int,
        round_size::S,
        covariance_ess_threshold::E,
        token::_ValidatedCAISToken,
    ) where {B<:ProposalBank,S,E}
        token === _VALIDATED_CAIS_TOKEN || throw(
            ArgumentError("invalid internal algorithm-construction token"),
        )
        return new{B,S,E}(bank, rounds, round_size, covariance_ess_threshold)
    end
end

function CAIS(
    bank::ProposalBank;
    rounds,
    round_size,
    covariance_ess_threshold=nothing,
)
    validated_round_size = _validate_adaptive_schedule(rounds, round_size)
    masses = bank.masses
    all(>(zero(eltype(masses))), masses) || throw(
        ArgumentError("CAIS requires every proposal mass to be positive"),
    )
    all(==(first(masses)), masses) || throw(
        ArgumentError("CAIS requires equal proposal masses"),
    )
    proposals = bank.proposals
    first_proposal = first(proposals)
    _validate_cais_proposal(first_proposal)
    first_location = first_proposal.location
    scalar_layout = first_location isa _NativeGaussianFloat
    T = _gaussian_float_type(first_location)
    dimension = _gaussian_dimension(first_location)
    for proposal in Iterators.drop(proposals, 1)
        _validate_cais_proposal(proposal)
        location = proposal.location
        (location isa _NativeGaussianFloat) == scalar_layout || throw(
            ArgumentError("CAIS proposals must share scalar or vector layout"),
        )
        _gaussian_float_type(location) === T || throw(
            ArgumentError("CAIS proposals must use one floating type"),
        )
        _gaussian_dimension(location) == dimension || throw(
            DimensionMismatch("CAIS proposals must share one dimension"),
        )
    end

    _validate_radial_bank_family(proposals)
    threshold = covariance_ess_threshold
    threshold === nothing ||
        threshold isa Real && !(threshold isa Bool) && isfinite(threshold) || throw(
            ArgumentError("covariance_ess_threshold must be nothing or finite real"),
        )
    schedule = _resolve_adaptive_schedule(rounds, validated_round_size)
    proposal_count = length(proposals)
    for (round, count) in pairs(schedule)
        count % proposal_count == 0 || throw(
            ArgumentError(
                "CAIS round $round of size $count must allocate equally " *
                "across $proposal_count proposals",
            ),
        )
        local_count = count ÷ proposal_count
        local_count >= dimension + 2 || throw(
            ArgumentError(
                "CAIS round $round must assign at least d + 2 samples " *
                "to every proposal",
            ),
        )
        threshold === nothing || dimension < threshold < local_count || throw(
            ArgumentError(
                "covariance_ess_threshold must satisfy d < threshold < m " *
                "in CAIS round $round",
            ),
        )
    end
    return CAIS(
        bank,
        rounds,
        validated_round_size,
        covariance_ess_threshold,
        _VALIDATED_CAIS_TOKEN,
    )
end

function _validate_cais_proposal(proposal)
    proposal isa _NativeRadialProposal && _is_packable_native_radial(proposal) || throw(
        ArgumentError(
            "CAIS requires native Float32 or Float64 spherical, diagonal, " *
            "or factor Gaussian or Student-t proposals",
        ),
    )
    _validate_moment_family(proposal.family)
    return nothing
end

_algorithm_proposal(algorithm::CAIS) = algorithm.bank
_algorithm_sample_budget(algorithm::CAIS) =
    _adaptive_sample_budget(algorithm.rounds, algorithm.round_size)

const _CAIS_TEMPERING_TOLERANCE = 1.0e-4
const _CAIS_TEMPERING_MAX_ITERATIONS = 16

struct _CAISWorkspace{S,W,I,N,O,R,C,V,F,J,E,P,H,U,Q}
    round_samples::S
    round_logweights::W
    round_proposal_ids::I
    normalized_weights::N
    solve_scratch::O
    local_starts::R
    covariance_centres::C
    covariances::V
    fitted_factors::F
    factor_info::J
    local_ess::E
    tempering_powers::P
    factor_status::H
    local_ess_history::U
    tempering_power_history::Q
end

mutable struct _PreparedCAIS{B,P,E,T,W}
    bank::B
    run_bank::B
    candidate_bank::B
    plan::P
    covariance_ess_threshold::E
    tempering_tolerance::T
    tempering_max_iterations::Int
    workspace::W
end

function _prepare_cais_bank(bank::ProposalBank)
    proposal_ids, logmasses, cdf = _prepare_active_proposal_metadata(bank)
    proposals = view(bank.proposals, proposal_ids)
    layout = first(proposals).location isa _NativeGaussianFloat
    pack_kind = layout ? Val(:diagonal) : Val(:factor)
    packed = _pack_native_radial_bank(
        bank,
        proposal_ids,
        logmasses,
        cdf,
        pack_kind,
    )
    packed isa Union{_PackedDiagonalBank,_PackedFactorBank} ||
        error("validated CAIS bank did not pack as a native population")
    return packed
end

function _cais_group_starts(counts)
    starts = similar(counts)
    for round in axes(counts, 2)
        first_sample = 1
        for proposal in axes(counts, 1)
            starts[proposal, round] = first_sample
            first_sample += counts[proposal, round]
        end
    end
    return starts
end

function _resolve_cais_thresholds(algorithm, plan, dimension, ::Type{T}) where {T}
    proposal_count, rounds = size(plan.counts)
    thresholds = Matrix{T}(undef, proposal_count, rounds)
    for round in 1:rounds, proposal in 1:proposal_count
        sample_count = plan.counts[proposal, round]
        raw = algorithm.covariance_ess_threshold
        value = if raw === nothing
            max(
                dimension + 1,
                _default_population_covariance_ess_threshold(sample_count),
            )
        else
            raw
        end
        converted = try
            T(value)
        catch
            throw(ArgumentError("covariance_ess_threshold must be convertible to $T"))
        end
        isfinite(converted) && T(dimension) < converted < T(sample_count) || throw(
            ArgumentError(
                "covariance_ess_threshold must satisfy d < threshold < m " *
                "for round $round, proposal $proposal",
            ),
        )
        thresholds[proposal, round] = converted
    end
    return thresholds
end

function _allocate_cais_workspace(bank, plan, ::Type{L}) where {L}
    T = eltype(bank.locations)
    dimension, proposal_count = size(bank.locations)
    capacity = maximum(plan.schedule)
    prototype = bank.locations
    rounds = length(plan.schedule)
    return _CAISWorkspace(
        _allocate_packed_static_mis_samples(prototype, bank, capacity),
        similar(prototype, L, capacity),
        similar(prototype, Int, capacity),
        similar(prototype, T, capacity),
        _allocate_mis_solve_scratch(prototype, bank, capacity),
        _cais_group_starts(plan.counts),
        similar(prototype, T, dimension, proposal_count),
        similar(prototype, T, dimension, dimension, proposal_count),
        similar(prototype, T, dimension, dimension, proposal_count),
        similar(prototype, Int32, proposal_count),
        similar(prototype, T, proposal_count),
        similar(prototype, T, proposal_count),
        similar(prototype, UInt8, proposal_count),
        similar(prototype, T, proposal_count, rounds),
        similar(prototype, T, proposal_count, rounds),
    )
end

function _prepare_cais_state(algorithm::CAIS, prepared_target=nothing)
    bank = _prepare_cais_bank(algorithm.bank)
    schedule = _resolve_adaptive_schedule(algorithm.rounds, algorithm.round_size)
    active_masses = algorithm.bank.masses[bank.proposal_ids]
    plan = _deterministic_allocation_plan(bank, active_masses, schedule)
    binding_sample = _population_binding_sample(bank)
    log_type = if prepared_target === nothing
        eltype(bank.lognormalizers)
    else
        target = _bind_resolved_target(prepared_target, binding_sample)
        _resolve_packed_static_mis_logweight_type(
            target,
            bank,
            typeof(binding_sample),
        )
    end
    T = eltype(bank.locations)
    thresholds = _resolve_cais_thresholds(
        algorithm,
        plan,
        size(bank.locations, 1),
        T,
    )
    tolerance = T(_CAIS_TEMPERING_TOLERANCE)
    workspace = _allocate_cais_workspace(bank, plan, log_type)
    return _PreparedCAIS(
        bank,
        _population_state_bank(bank),
        _population_state_bank(bank),
        plan,
        thresholds,
        tolerance,
        _CAIS_TEMPERING_MAX_ITERATIONS,
        workspace,
    )
end

_prepare_method_state(algorithm::CAIS) = _prepare_cais_state(algorithm)
_prepare_method_state(algorithm::CAIS, prepared_target) =
    _prepare_cais_state(algorithm, prepared_target)

_accelerator_method_state_limit(
    ::_PreparedCAIS{<:Union{
        _PackedDiagonalBank,
        _PackedFactorBank,
    }},
) = nothing

function _allocate_random_buffers(
    ::MLDataDevices.AbstractDevice,
    ::ProposalBank,
    method_state::_PreparedCAIS,
    sample_budget,
)
    bank = method_state.bank
    capacity = maximum(method_state.plan.schedule)
    normals = similar(
        bank.locations,
        eltype(bank.locations),
        size(bank.locations, 1) * capacity,
    )
    failure_scratch = _allocate_native_failure_scratch(
        normals, capacity; capacity=max(sample_budget, _active_proposal_count(bank)),
    )
    return _PopulationNormalBuffers(normals, failure_scratch,
        _allocate_radial_buffers(normals, bank.family, capacity))
end

function _cais_round_views(method_state::_PreparedCAIS, round)
    workspace = method_state.workspace
    round_size = method_state.plan.schedule[round]
    return (
        round_size,
        samples=_sample_view(workspace.round_samples, 1:round_size),
        logweights=view(workspace.round_logweights, 1:round_size),
        proposal_ids=view(workspace.round_proposal_ids, 1:round_size),
        assignments=view(method_state.plan.assignments, 1:round_size, round),
    )
end

function _copy_cais_workspace(device, workspace)
    return _CAISWorkspace(
        _copy_to_device(device, workspace.round_samples),
        _copy_to_device(device, workspace.round_logweights),
        _copy_to_device(device, workspace.round_proposal_ids),
        _copy_to_device(device, workspace.normalized_weights),
        _copy_to_device(device, workspace.solve_scratch),
        _copy_to_device(device, workspace.local_starts),
        _copy_to_device(device, workspace.covariance_centres),
        _copy_to_device(device, workspace.covariances),
        _copy_to_device(device, workspace.fitted_factors),
        _copy_to_device(device, workspace.factor_info),
        _copy_to_device(device, workspace.local_ess),
        _copy_to_device(device, workspace.tempering_powers),
        _copy_to_device(device, workspace.factor_status),
        _copy_to_device(device, workspace.local_ess_history),
        _copy_to_device(device, workspace.tempering_power_history),
    )
end

function _prepare_transferred_method_state(
    device,
    algorithm::CAIS,
    method_state::_PreparedCAIS,
    _transferred_target,
)
    bank = _copy_packed_bank(device, method_state.bank)
    plan = _transfer_population_plan(device, method_state.plan)
    return _PreparedCAIS(
        bank,
        _population_state_bank(bank),
        _population_state_bank(bank),
        plan,
        _copy_to_device(device, method_state.covariance_ess_threshold),
        method_state.tempering_tolerance,
        method_state.tempering_max_iterations,
        _copy_cais_workspace(device, method_state.workspace),
    )
end

_transferred_backend_state(
    algorithm,
    method_state::_PreparedCAIS,
    target,
    random_buffers,
) = (method_state, target, random_buffers)

_prepared_backend_state(sampler, method_state::_PreparedCAIS) = (
    method_state,
    sampler.target,
    sampler.random_buffers,
    sampler.rng,
)

function _copy_algorithm(device, algorithm::CAIS)
    return CAIS(
        _copy_population_bank(device, algorithm.bank);
        rounds=algorithm.rounds,
        round_size=algorithm.round_size,
        covariance_ess_threshold=algorithm.covariance_ess_threshold,
    )
end

_copy_algorithm(::MLDataDevices.CPUDevice{Missing}, algorithm::CAIS) =
    deepcopy(algorithm)

_copy_accelerator_algorithm(device, algorithm::CAIS, ::_PreparedCAIS) =
    deepcopy(algorithm)

function _retarget_algorithm(
    sampler::_PreparedImportanceSampler{R,B,T,A},
) where {R,B,T,A<:CAIS}
    algorithm = sampler.algorithm
    return CAIS(
        current_proposal(MLDataDevices.cpu_device(), sampler);
        rounds=algorithm.rounds,
        round_size=algorithm.round_size,
        covariance_ess_threshold=algorithm.covariance_ess_threshold,
    )
end

function current_proposal(
    sampler::_PreparedImportanceSampler{R,B,T,A,M,D},
) where {R,B,T,A<:CAIS,M,D}
    sampler.device isa MLDataDevices.AbstractAcceleratorDevice && throw(
        ArgumentError(
            "current_proposal(sampler) does not copy accelerator state " *
            "implicitly; call current_proposal(cpu_device(), sampler)",
        ),
    )
    return current_proposal(MLDataDevices.cpu_device(), sampler)
end

function current_proposal(
    destination::MLDataDevices.AbstractCPUDevice,
    sampler::_PreparedImportanceSampler{R,B,T,A,M,D},
) where {R,B,T,A<:CAIS,M,D}
    _preserving_cpu_destination(destination)
    committed = sampler.method_state.bank
    parameters, family = if sampler.device isa MLDataDevices.AbstractCPUDevice
        _cais_snapshot_parameters(committed), committed.family
    else
        _with_backend_device(sampler.device) do
            (map(destination ∘ Array, _cais_snapshot_parameters(committed)),
                _copy_to_device(destination, committed.family))
        end
    end
    return _cais_snapshot(
        committed,
        parameters...,
        sampler.algorithm.bank.masses,
        family,
    )
end

function current_proposal(
    destination::MLDataDevices.AbstractDevice,
    sampler::_PreparedImportanceSampler{R,B,T,A,M,D},
) where {R,B,T,A<:CAIS,M,D}
    throw(
        ArgumentError(
            "current_proposal requires a CPU destination; got " *
            string(typeof(destination)),
        ),
    )
end

_cais_snapshot_parameters(bank::_PackedDiagonalBank) = (
    copy(bank.locations),
    copy(bank.scales),
    copy(bank.lognormalizers),
    copy(bank.proposal_ids),
)

_cais_snapshot_parameters(bank::_PackedFactorBank) = (
    copy(bank.locations),
    copy(bank.factors),
    copy(bank.lognormalizers),
    copy(bank.proposal_ids),
)

function _cais_snapshot(
    bank::_PackedDiagonalBank,
    locations,
    scales,
    lognormalizers,
    proposal_ids,
    configured_masses,
    family,
)
    proposals = Vector{typeof(_radial_proposal(_radial_family_at(family, 1),
        zero(eltype(locations)), _SphericalGaussianScale(one(eltype(scales))), zero(eltype(lognormalizers))))}(
        undef,
        length(configured_masses),
    )
    for (slot, proposal_id) in pairs(proposal_ids)
        proposals[proposal_id] = _radial_proposal(
            _radial_family_at(family, slot),
            locations[1, slot],
            _SphericalGaussianScale(scales[1, slot]),
            lognormalizers[slot],
        )
    end
    snapshot = ProposalBank(proposals, configured_masses)
    copyto!(snapshot.masses, configured_masses)
    return snapshot
end

function _cais_snapshot(
    bank::_PackedFactorBank,
    locations,
    factors,
    lognormalizers,
    proposal_ids,
    configured_masses,
    family,
)
    return _factor_population_snapshot(
        locations,
        factors,
        lognormalizers,
        proposal_ids,
        configured_masses,
        family,
    )
end
