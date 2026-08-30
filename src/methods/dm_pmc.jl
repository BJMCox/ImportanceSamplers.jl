mutable struct _ValidatedDeterministicMixturePMCToken end
const _VALIDATED_DETERMINISTIC_MIXTURE_PMC_TOKEN =
    _ValidatedDeterministicMixturePMCToken()

function _validate_adaptive_schedule(rounds, round_size)
    rounds isa Int && rounds > 0 || throw(ArgumentError("rounds must be a positive Int"))
    if round_size isa Int
        round_size > 0 || throw(ArgumentError("round_size must be positive"))
        return round_size
    end
    round_size isa Vector{Int} || throw(
        ArgumentError("round_size must be a positive Int or Vector{Int}"),
    )
    length(round_size) == rounds || throw(
        DimensionMismatch("round_size must contain one entry per round"),
    )
    all(>(0), round_size) || throw(
        ArgumentError("every round_size entry must be positive"),
    )
    return copy(round_size)
end

_resolve_adaptive_schedule(rounds, round_size::Int) = fill(round_size, rounds)
_resolve_adaptive_schedule(rounds, round_size::Vector{Int}) = copy(round_size)
_adaptive_sample_budget(rounds, round_size::Int) = rounds * round_size
_adaptive_sample_budget(rounds, round_size::Vector{Int}) = sum(round_size)

"""
    DeterministicMixturePMC(bank; rounds, round_size)

Configure deterministic-mixture population Monte Carlo with a fixed round
schedule. `round_size` is either one positive `Int` repeated for every round or
a positive `Vector{Int}` with one entry per round.
"""
struct DeterministicMixturePMC{B<:ProposalBank,S} <: AbstractImportanceSampler
    bank::B
    rounds::Int
    round_size::S

    function DeterministicMixturePMC(
        bank::B,
        rounds::Int,
        round_size::S,
        token::_ValidatedDeterministicMixturePMCToken,
    ) where {B<:ProposalBank,S}
        token === _VALIDATED_DETERMINISTIC_MIXTURE_PMC_TOKEN || throw(
            ArgumentError("invalid internal algorithm-construction token"),
        )
        return new{B,S}(bank, rounds, round_size)
    end
end

function DeterministicMixturePMC(bank::ProposalBank; rounds, round_size)
    validated_round_size = _validate_adaptive_schedule(rounds, round_size)
    return DeterministicMixturePMC(
        bank,
        rounds,
        validated_round_size,
        _VALIDATED_DETERMINISTIC_MIXTURE_PMC_TOKEN,
    )
end

_algorithm_proposal(algorithm::DeterministicMixturePMC) = algorithm.bank
_algorithm_sample_budget(algorithm::DeterministicMixturePMC) =
    _adaptive_sample_budget(algorithm.rounds, algorithm.round_size)

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
destinations are rejected. Other prepared algorithms do not currently
implement this accessor.
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
    sampler.device isa MLDataDevices.AbstractAcceleratorDevice && throw(
        ArgumentError(
            "current_proposal(sampler) does not copy accelerator state " *
            "implicitly; call current_proposal(cpu_device(), " *
            "sampler) to request an explicit CPU snapshot",
        ),
    )
    packed = sampler.method_state.bank
    return _dm_pmc_proposal_snapshot(
        copy(packed.locations),
        copy(packed.proposal_ids),
        sampler,
    )
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
    if applicable(eltype, destination)
        policy = eltype(destination)
        policy in (Missing, Nothing) || throw(
            ArgumentError(
                "current_proposal requires a preserving CPU destination; " *
                "use MLDataDevices.cpu_device() without a scalar conversion",
            ),
        )
    end
    packed = sampler.method_state.bank
    locations, proposal_ids = _with_backend_device(sampler.device) do
        (
            destination(Array(packed.locations)),
            destination(Array(packed.proposal_ids)),
        )
    end
    return _dm_pmc_proposal_snapshot(locations, proposal_ids, sampler)
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
    throw(
        ArgumentError(
            "current_proposal requires a CPU destination; got " *
            string(typeof(destination)),
        ),
    )
end

function _dm_pmc_proposal_snapshot(locations, proposal_ids, sampler)
    proposals = deepcopy(sampler.algorithm.bank.proposals)
    for (slot, proposal_id) in pairs(proposal_ids)
        proposal = proposals[proposal_id]
        location = _dm_pmc_snapshot_location(
            locations,
            slot,
            sampler.method_state.bank,
        )
        proposals[proposal_id] = _dm_pmc_with_location(proposal, location)
    end
    configured_masses = sampler.algorithm.bank.masses
    snapshot = ProposalBank(proposals, configured_masses)
    # The configured masses are already validated and normalized. Restore their
    # exact values after constructor validation instead of normalizing twice.
    copyto!(snapshot.masses, configured_masses)
    return snapshot
end

_dm_pmc_snapshot_location(locations, slot, bank::_PackedDiagonalGaussianBank) =
    _dm_pmc_snapshot_location(locations, slot, bank.layout)

_dm_pmc_snapshot_location(locations, slot, ::_ScalarGaussianLayout) =
    locations[1, slot]

_dm_pmc_snapshot_location(locations, slot, ::_VectorGaussianLayout) =
    copy(view(locations, :, slot))

_dm_pmc_snapshot_location(locations, slot, ::_PackedFactorGaussianBank) =
    copy(view(locations, :, slot))

function _dm_pmc_with_location(proposal::_GaussianProposal, location)
    return _GaussianProposal(
        proposal.family,
        location,
        deepcopy(proposal.scale),
        proposal.lognormalizer,
    )
end

struct _DeterministicAllocationPlan{S,C,A,L,O}
    schedule::S
    counts::C
    assignments::A
    logcoefficients::L
    offsets::O
end

mutable struct _PreparedDMPMC{B,P,W}
    bank::B
    run_bank::B
    plan::P
    workspace::W
end

function _dm_pmc_with_locations(bank::_PackedDiagonalGaussianBank, locations)
    return _PackedDiagonalGaussianBank(
        locations,
        bank.scales,
        bank.lognormalizers,
        bank.logmasses,
        bank.cdf,
        bank.proposal_ids,
        bank.layout,
    )
end

function _dm_pmc_with_locations(bank::_PackedFactorGaussianBank, locations)
    return _PackedFactorGaussianBank(
        locations,
        bank.factors,
        bank.lognormalizers,
        bank.logmasses,
        bank.cdf,
        bank.proposal_ids,
    )
end

function _dm_pmc_run_bank(bank)
    locations = similar(bank.locations)
    return _dm_pmc_with_locations(bank, locations)
end

_accelerator_method_state_limit(
    ::_PreparedDMPMC{<:Union{
        _PackedDiagonalGaussianBank,
        _PackedFactorGaussianBank,
    }},
) = nothing

function _prepare_dm_pmc_bank(bank::ProposalBank)
    destination_type = _dm_pmc_prepared_proposal_type(
        eltype(bank.masses),
        eltype(bank.proposals),
    )
    return _prepare_dm_pmc_bank(bank, destination_type)
end

function _prepare_dm_pmc_bank(bank::ProposalBank, ::Nothing)
    proposal_ids, logmasses, cdf = _prepare_active_proposal_metadata(bank)
    packed = _pack_native_gaussian_bank(
        bank,
        proposal_ids,
        logmasses,
        cdf,
    )
    packed isa Union{_PackedDiagonalGaussianBank,_PackedFactorGaussianBank} || throw(
        ArgumentError(
            "DM-PMC requires native Float32 or Float64 spherical, diagonal, or factor Gaussian proposals",
        ),
    )
    return packed
end

function _prepare_dm_pmc_bank(
    bank::ProposalBank{P,M},
    ::Type{D},
) where {
    T<:_NativeGaussianFloat,
    P,
    M<:AbstractVector{T},
    D<:_GaussianProposal,
}
    proposal_ids, logmasses, cdf = _prepare_active_proposal_metadata(bank)
    proposals = view(bank.proposals, proposal_ids)
    all(proposal -> proposal isa D, proposals) || throw(
        ArgumentError("active DM-PMC proposals do not match their destination type"),
    )

    first_proposal = first(proposals)::D
    dimension = _gaussian_dimension(first_proposal.location)
    locations = Matrix{T}(undef, dimension, length(proposals))
    lognormalizers = Vector{T}(undef, length(proposals))
    for (slot, untyped_proposal) in pairs(proposals)
        proposal = untyped_proposal::D
        _gaussian_dimension(proposal.location) == dimension || throw(
            DimensionMismatch(
                "positive-mass native Gaussian proposals must have one common dimension",
            ),
        )
        if proposal.location isa _NativeGaussianFloat
            locations[1, slot] = proposal.location
        else
            copyto!(view(locations, :, slot), proposal.location)
        end
        lognormalizers[slot] = proposal.lognormalizer
    end

    return _pack_native_gaussian_storage(
        locations,
        lognormalizers,
        logmasses,
        cdf,
        proposal_ids,
        proposals,
        _dm_pmc_gaussian_layout(D),
        _native_gaussian_pack_kind(D),
    )
end

function _dm_pmc_exact_mass_proportions(active_masses)
    active_count = length(active_masses)
    exact_masses = Vector{Rational{BigInt}}(undef, active_count)
    for slot in eachindex(active_masses)
        exact_masses[slot] = rationalize(
            BigInt,
            active_masses[slot];
            tol=0,
        )
    end
    total_mass = sum(exact_masses)
    for slot in eachindex(exact_masses)
        exact_masses[slot] /= total_mass
    end
    return exact_masses
end

function _dm_pmc_round_counts(
    exact_mass_proportions::Vector{Rational{BigInt}},
    round_size,
    round,
)
    active_count = length(exact_mass_proportions)
    counts = Vector{Int}(undef, active_count)
    remainders = similar(exact_mass_proportions)
    for slot in eachindex(exact_mass_proportions)
        quota = exact_mass_proportions[slot] * round_size
        counts[slot] = floor(Int, quota)
        remainders[slot] = quota - counts[slot]
    end

    remaining = round_size - sum(counts)
    0 <= remaining < active_count || error("invalid DM-PMC allocation remainder")
    first_tie_slot = mod1(round, active_count)
    tie_order = collect(1:active_count)
    sort!(
        tie_order;
        by=slot -> (
            -remainders[slot],
            mod(slot - first_tie_slot, active_count),
        ),
        alg=Base.Sort.MergeSort,
    )
    for index in 1:remaining
        counts[tie_order[index]] += 1
    end
    return counts
end

function _deterministic_allocation_plan(bank, active_masses, schedule)
    active_count = length(bank.proposal_ids)
    rounds = length(schedule)
    counts = Matrix{Int}(undef, active_count, rounds)
    assignments = zeros(Int, maximum(schedule), rounds)
    logcoefficients = Matrix{eltype(active_masses)}(undef, active_count, rounds)
    offsets = Vector{Int}(undef, rounds + 1)
    offsets[1] = 1
    exact_mass_proportions = _dm_pmc_exact_mass_proportions(active_masses)

    for round in eachindex(schedule)
        round_size = schedule[round]
        counts[:, round] .= _dm_pmc_round_counts(
            exact_mass_proportions,
            round_size,
            round,
        )

        all(>(0), view(counts, :, round)) || throw(
            ArgumentError(
                "round $round of size $round_size cannot allocate at least one sample to every positive-mass proposal",
            ),
        )

        assignment_index = 1
        for slot in 1:active_count
            count = counts[slot, round]
            last_assignment = assignment_index + count - 1
            fill!(view(assignments, assignment_index:last_assignment, round), slot)
            logcoefficients[slot, round] = log(
                eltype(active_masses)(count) /
                eltype(active_masses)(round_size),
            )
            assignment_index = last_assignment + 1
        end
        offsets[round + 1] = offsets[round] + round_size
    end

    return _DeterministicAllocationPlan(
        schedule,
        counts,
        assignments,
        logcoefficients,
        offsets,
    )
end

function _prepare_dm_pmc_state(algorithm::DeterministicMixturePMC)
    schedule = _resolve_adaptive_schedule(algorithm.rounds, algorithm.round_size)
    bank = _prepare_dm_pmc_bank(algorithm.bank)
    active_masses = algorithm.bank.masses[bank.proposal_ids]
    plan = _deterministic_allocation_plan(bank, active_masses, schedule)
    return bank, plan
end

function _prepare_method_state(algorithm::DeterministicMixturePMC)
    bank, plan = _prepare_dm_pmc_state(algorithm)
    workspace = _allocate_dm_pmc_workspace(
        bank,
        plan,
        eltype(bank.lognormalizers),
    )
    return _PreparedDMPMC(bank, _dm_pmc_run_bank(bank), plan, workspace)
end

function _prepare_method_state(algorithm::DeterministicMixturePMC, prepared_target)
    bank, plan = _prepare_dm_pmc_state(algorithm)
    binding_sample = _dm_pmc_binding_sample(bank)
    target = _bind_resolved_target(prepared_target, binding_sample)
    log_type = _resolve_packed_static_mis_logweight_type(
        target,
        bank,
        typeof(binding_sample),
    )
    workspace = _allocate_dm_pmc_workspace(bank, plan, log_type)
    return _PreparedDMPMC(bank, _dm_pmc_run_bank(bank), plan, workspace)
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
        maximum_round_size,
    )
    return _DMPMCRandomBuffers(
        normals,
        resampling_uniforms,
        failure_scratch,
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
)
    plan = method_state.plan
    transferred_plan = _DeterministicAllocationPlan(
        Tuple(plan.schedule),
        _copy_to_device(device, plan.counts),
        _copy_to_device(device, plan.assignments),
        _copy_to_device(device, plan.logcoefficients),
        Tuple(plan.offsets),
    )
    workspace = method_state.workspace
    transferred_workspace = _DMPMCWorkspace(
        _copy_to_device(device, workspace.round_samples),
        _copy_to_device(device, workspace.round_logweights),
        _copy_to_device(device, workspace.round_proposal_ids),
        _copy_to_device(device, workspace.solve_scratch),
        _copy_to_device(device, workspace.resampling_cdf),
        _copy_to_device(device, workspace.ancestors),
        _copy_to_device(device, workspace.candidate_locations),
    )
    transferred_bank = _copy_packed_gaussian_bank(device, method_state.bank)
    return _PreparedDMPMC(
        transferred_bank,
        _dm_pmc_run_bank(transferred_bank),
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
    binding_sample = _dm_pmc_binding_sample(bank)
    bound_target = _bind_resolved_target(target, binding_sample)
    log_type = _resolve_packed_static_mis_logweight_type(
        bound_target,
        bank,
        typeof(binding_sample),
    )
    target_argument = _NativeDeviceTarget{log_type,typeof(bound_target)}(bound_target)
    backend = KernelAbstractions.get_backend(buffers.normals)
    representative_round = findmax(plan.schedule)[2]
    round_views = _dm_pmc_round_views(method_state, representative_round)

    round_kernel = _mis_round_launch_kernel!(backend)
    denominator =
        _RealizedMixtureDenominator(plan.logcoefficients, representative_round)
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
        batch_kernel = _factor_batch_mis_draw_target_kernel!(backend)
        _preflight_kernel_argument(device, batch_kernel, target_argument)
    end

    finalize_kernel = _dm_pmc_finalize_cdf_kernel!(backend)
    for argument in (round_views.cdf, round_views.round_size)
        _preflight_kernel_argument(device, finalize_kernel, argument)
    end
    select_kernel = _dm_pmc_select_ancestors_kernel!(backend)
    for argument in (
        workspace.ancestors,
        buffers.resampling_uniforms,
        round_views.cdf,
        round_views.round_size,
    )
        _preflight_kernel_argument(device, select_kernel, argument)
    end
    gather_kernel = _dm_pmc_gather_ancestors_kernel!(backend)
    for argument in (
        workspace.candidate_locations,
        round_views.samples,
        workspace.ancestors,
    )
        _preflight_kernel_argument(device, gather_kernel, argument)
    end
    return nothing
end

function _copy_dm_pmc_active_proposal(device, proposal::_GaussianProposal)
    adapted = _copy_to_device(device, proposal)
    T = _gaussian_float_type(adapted.location)
    return _GaussianProposal(
        adapted.family,
        adapted.location,
        adapted.scale,
        convert(T, adapted.lognormalizer),
    )
end

_copy_dm_pmc_location(::Type{T}, location::_NativeGaussianFloat) where {T} =
    convert(T, location)

_copy_dm_pmc_location(::Type{T}, location::AbstractVector) where {T} =
    T.(location)

function _copy_dm_pmc_active_proposal(
    ::MLDataDevices.CPUDevice{T},
    proposal::_GaussianProposal{F,L,<:_SphericalGaussianScale},
) where {T<:_NativeGaussianFloat,F,L}
    return SphericalGaussian(
        _copy_dm_pmc_location(T, proposal.location),
        convert(T, proposal.scale.scale),
    )
end

function _copy_dm_pmc_active_proposal(
    ::MLDataDevices.CPUDevice{T},
    proposal::_GaussianProposal{F,L,<:_DiagonalGaussianScale},
) where {T<:_NativeGaussianFloat,F,L}
    return DiagonalGaussian(
        _copy_dm_pmc_location(T, proposal.location),
        T.(proposal.scale.scales),
    )
end

function _copy_dm_pmc_active_proposal(
    ::MLDataDevices.CPUDevice{T},
    proposal::_GaussianProposal{F,L,<:_FactorGaussianScale},
) where {T<:_NativeGaussianFloat,F,L}
    return FactorGaussian(
        _copy_dm_pmc_location(T, proposal.location),
        T.(proposal.scale.factor),
    )
end

function _copy_dm_pmc_generic_bank(device, bank::ProposalBank)
    proposals = map(eachindex(bank.proposals, bank.masses)) do proposal_id
        proposal = bank.proposals[proposal_id]
        iszero(bank.masses[proposal_id]) && return deepcopy(proposal)
        return _copy_dm_pmc_active_proposal(device, proposal)
    end
    masses = _copy_to_device(device, bank.masses)
    return ProposalBank(proposals, masses)
end

_copy_dm_pmc_bank(device, bank::ProposalBank{P,M}) where {P,M} =
    _copy_dm_pmc_generic_bank(device, bank)

function _copy_dm_pmc_homogeneous_bank(
    device,
    bank::ProposalBank{P},
    ::Type{D},
) where {P,D}
    S = eltype(P)
    proposals = Vector{Union{S,D}}(undef, length(bank.proposals))
    for proposal_id in eachindex(bank.proposals, bank.masses)
        proposal = bank.proposals[proposal_id]
        proposals[proposal_id] = iszero(bank.masses[proposal_id]) ?
                                 deepcopy(proposal) :
                                 _copy_dm_pmc_active_proposal(device, proposal)
    end
    masses = _copy_to_device(device, bank.masses)
    return ProposalBank(proposals, masses)
end

_copy_dm_pmc_resolved_bank(device, bank, ::Nothing) =
    _copy_dm_pmc_generic_bank(device, bank)

_copy_dm_pmc_resolved_bank(device, bank, ::Type{D}) where {D<:_GaussianProposal} =
    _copy_dm_pmc_homogeneous_bank(device, bank, D)

function _dm_pmc_destination_proposal_type(
    ::Type{T},
    ::Type{<:_GaussianProposal{F,L,<:_SphericalGaussianScale,N}},
) where {T<:_NativeGaussianFloat,F,L<:_NativeGaussianFloat,N}
    return _GaussianProposal{
        GaussianFamily,
        T,
        _SphericalGaussianScale{T},
        T,
    }
end

function _dm_pmc_destination_proposal_type(
    ::Type{T},
    ::Type{<:_GaussianProposal{F,L,<:_SphericalGaussianScale,N}},
) where {T<:_NativeGaussianFloat,F,L<:AbstractVector,N}
    return _GaussianProposal{
        GaussianFamily,
        Vector{T},
        _SphericalGaussianScale{T},
        T,
    }
end

function _dm_pmc_destination_proposal_type(
    ::Type{T},
    ::Type{<:_GaussianProposal{F,L,<:_DiagonalGaussianScale,N}},
) where {T<:_NativeGaussianFloat,F,L,N}
    return _GaussianProposal{
        GaussianFamily,
        Vector{T},
        _DiagonalGaussianScale{Vector{T}},
        T,
    }
end

function _dm_pmc_destination_proposal_type(
    ::Type{T},
    ::Type{<:_GaussianProposal{F,L,<:_FactorGaussianScale,N}},
) where {T<:_NativeGaussianFloat,F,L,N}
    return _GaussianProposal{
        GaussianFamily,
        Vector{T},
        _FactorGaussianScale{Matrix{T}},
        T,
    }
end

@generated function _dm_pmc_destination_proposal_type(
    ::Type{T},
    ::Type{G},
) where {T,G}
    proposal_types = Base.uniontypes(G)
    length(proposal_types) == 2 || return :(nothing)
    A, B = proposal_types
    return quote
        destination_a = _dm_pmc_destination_proposal_type(T, $A)
        destination_b = _dm_pmc_destination_proposal_type(T, $B)
        destination_a === destination_b ? destination_a : nothing
    end
end

@generated function _dm_pmc_prepared_proposal_type(
    ::Type{T},
    ::Type{G},
) where {T,G}
    if length(Base.uniontypes(G)) > 1
        return :(_dm_pmc_destination_proposal_type(T, G))
    end
    return quote
        destination_type = _dm_pmc_destination_proposal_type(T, G)
        destination_type === G ? destination_type : nothing
    end
end

_dm_pmc_gaussian_layout(
    ::Type{<:_GaussianProposal{F,L}},
) where {F,L<:_NativeGaussianFloat} = _ScalarGaussianLayout()

_dm_pmc_gaussian_layout(
    ::Type{<:_GaussianProposal{F,L}},
) where {F,L<:AbstractVector} = _VectorGaussianLayout()

function _copy_dm_pmc_bank(
    device::MLDataDevices.CPUDevice{T},
    bank::ProposalBank{P,M},
) where {
    T<:_NativeGaussianFloat,
    G<:_GaussianProposal,
    P<:AbstractVector{G},
    M,
}
    D = _dm_pmc_destination_proposal_type(T, G)
    return _copy_dm_pmc_resolved_bank(device, bank, D)
end

function _copy_algorithm(device, algorithm::DeterministicMixturePMC)
    return DeterministicMixturePMC(
        _copy_dm_pmc_bank(device, algorithm.bank);
        rounds=algorithm.rounds,
        round_size=algorithm.round_size,
    )
end

_copy_algorithm(
    ::MLDataDevices.CPUDevice{Missing},
    algorithm::DeterministicMixturePMC,
) = deepcopy(algorithm)
