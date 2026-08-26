mutable struct _ValidatedDeterministicMixturePMCToken end
const _VALIDATED_DETERMINISTIC_MIXTURE_PMC_TOKEN =
    _ValidatedDeterministicMixturePMCToken()

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
    rounds isa Int && rounds > 0 || throw(
        ArgumentError("rounds must be a positive Int"),
    )
    validated_round_size = _validate_dm_pmc_round_size(round_size, rounds)
    return DeterministicMixturePMC(
        bank,
        rounds,
        validated_round_size,
        _VALIDATED_DETERMINISTIC_MIXTURE_PMC_TOKEN,
    )
end

function _validate_dm_pmc_round_size(round_size::Int, rounds)
    round_size > 0 || throw(ArgumentError("round_size must be positive"))
    return round_size
end

function _validate_dm_pmc_round_size(round_size::Vector{Int}, rounds)
    length(round_size) == rounds || throw(
        DimensionMismatch("round_size must contain one entry per round"),
    )
    all(>(0), round_size) || throw(
        ArgumentError("every round_size entry must be positive"),
    )
    return copy(round_size)
end

_validate_dm_pmc_round_size(round_size, rounds) = throw(
    ArgumentError("round_size must be a positive Int or Vector{Int}"),
)

function _resolve_round_schedule(
    algorithm::DeterministicMixturePMC{B,Int},
) where {B}
    return fill(algorithm.round_size, algorithm.rounds)
end

function _resolve_round_schedule(
    algorithm::DeterministicMixturePMC{B,<:Vector{Int}},
) where {B}
    return copy(algorithm.round_size)
end

_algorithm_proposal(algorithm::DeterministicMixturePMC) = algorithm.bank
_algorithm_sample_budget(algorithm::DeterministicMixturePMC) =
    sum(_resolve_round_schedule(algorithm))

struct _DMPMCAllocationPlan{S,C,A,L,O}
    schedule::S
    counts::C
    assignments::A
    logcoefficients::L
    offsets::O
end

mutable struct _PreparedDMPMC{B,P}
    bank::B
    plan::P
end

_accelerator_method_state_limit(::_PreparedDMPMC) = :unsupported_device

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

function _dm_pmc_round_counts(active_masses, round_size, round)
    exact_mass_proportions = _dm_pmc_exact_mass_proportions(active_masses)
    return _dm_pmc_round_counts(exact_mass_proportions, round_size, round)
end

function _dm_pmc_allocation_plan(bank, active_masses, schedule)
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

    return _DMPMCAllocationPlan(
        schedule,
        counts,
        assignments,
        logcoefficients,
        offsets,
    )
end

function _prepare_method_state(algorithm::DeterministicMixturePMC)
    schedule = _resolve_round_schedule(algorithm)
    bank = _prepare_dm_pmc_bank(algorithm.bank)
    active_masses = algorithm.bank.masses[bank.proposal_ids]
    plan = _dm_pmc_allocation_plan(bank, active_masses, schedule)
    return _PreparedDMPMC(bank, plan)
end

function _allocate_random_buffers(
    ::MLDataDevices.AbstractCPUDevice,
    ::ProposalBank,
    ::_PreparedDMPMC,
    sample_budget,
)
    return _NoRandomBuffers()
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

function _copy_dm_pmc_bank(device, bank::ProposalBank{P,M}) where {P,M}
    proposals = map(eachindex(bank.proposals, bank.masses)) do proposal_id
        proposal = bank.proposals[proposal_id]
        iszero(bank.masses[proposal_id]) && return deepcopy(proposal)
        return _copy_dm_pmc_active_proposal(device, proposal)
    end
    masses = _copy_to_device(device, bank.masses)
    return ProposalBank(proposals, masses)
end

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
    return _copy_dm_pmc_homogeneous_bank(device, bank, D)
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
