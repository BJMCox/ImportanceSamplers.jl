# Shared adaptive schedules and population preparation, snapshots, and transfers.
# Method-specific configuration, state, and adaptation laws remain with each method.

abstract type _FixedPopulationSampler <: AbstractImportanceSampler end

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

function _adaptive_sample_budget(rounds, round_size::Int)
    return try
        Base.checked_mul(rounds, round_size)
    catch error
        error isa OverflowError || rethrow()
        throw(ArgumentError("total adaptive sample count exceeds Int"))
    end
end

function _adaptive_sample_budget(rounds, round_size::Vector{Int})
    total = 0
    for count in round_size
        total = try
            Base.checked_add(total, count)
        catch error
            error isa OverflowError || rethrow()
            throw(ArgumentError("total adaptive sample count exceeds Int"))
        end
    end
    return total
end

function _population_group_starts(counts)
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

function _current_population_proposal(sampler)
    sampler.device isa MLDataDevices.AbstractAcceleratorDevice && throw(
        ArgumentError(
            "current_proposal(sampler) does not copy accelerator state " *
            "implicitly; call current_proposal(cpu_device(), " *
            "sampler) to request an explicit CPU snapshot",
        ),
    )
    packed = sampler.method_state.bank
    return _population_proposal_snapshot(
        copy(packed.locations),
        copy(packed.proposal_ids),
        sampler,
    )
end

function _current_population_proposal(
    destination::MLDataDevices.AbstractCPUDevice,
    sampler,
)
    _preserving_cpu_destination(destination)
    packed = sampler.method_state.bank
    locations, proposal_ids = _with_backend_device(sampler.device) do
        (
            destination(Array(packed.locations)),
            destination(Array(packed.proposal_ids)),
        )
    end
    return _population_proposal_snapshot(locations, proposal_ids, sampler)
end

function _current_population_proposal(
    destination::MLDataDevices.AbstractDevice,
    sampler,
)
    throw(
        ArgumentError(
            "current_proposal requires a CPU destination; got " *
            string(typeof(destination)),
        ),
    )
end

function _population_proposal_snapshot(locations, proposal_ids, sampler)
    proposals = deepcopy(sampler.algorithm.bank.proposals)
    for (slot, proposal_id) in pairs(proposal_ids)
        proposal = proposals[proposal_id]
        location = _population_snapshot_location(
            locations,
            slot,
            sampler.method_state.bank,
        )
        proposals[proposal_id] = _radial_proposal(proposal.family, location,
            proposal.scale, proposal.lognormalizer)
    end
    configured_masses = sampler.algorithm.bank.masses
    snapshot = ProposalBank(proposals, configured_masses)
    # The configured masses are already validated and normalized. Restore their
    # exact values after constructor validation instead of normalizing twice.
    copyto!(snapshot.masses, configured_masses)
    return snapshot
end

_population_snapshot_location(locations, slot, bank::_PackedDiagonalBank) =
    _population_snapshot_location(locations, slot, bank.layout)

_population_snapshot_location(locations, slot, ::_ScalarGaussianLayout) =
    locations[1, slot]

_population_snapshot_location(locations, slot, ::Union{_VectorGaussianLayout,_PackedFactorBank}) =
    copy(view(locations, :, slot))

struct _DeterministicAllocationPlan{S,C,A,L,O}
    schedule::S
    counts::C
    assignments::A
    logcoefficients::L
    offsets::O
end

# Accelerator schedules stay on the host, but their runtime length must not
# enter the prepared sampler type.
struct _HostIntSequence
    values::Tuple{Vararg{Int}}
end

_HostIntSequence(values::AbstractVector{Int}) = _HostIntSequence(Tuple(values))
Base.IteratorSize(::Type{_HostIntSequence}) = Base.HasLength()
Base.IteratorEltype(::Type{_HostIntSequence}) = Base.HasEltype()
Base.eltype(::Type{_HostIntSequence}) = Int
Base.length(sequence::_HostIntSequence) = length(sequence.values)
Base.firstindex(sequence::_HostIntSequence) = firstindex(sequence.values)
Base.lastindex(sequence::_HostIntSequence) = lastindex(sequence.values)
Base.eachindex(sequence::_HostIntSequence) = eachindex(sequence.values)
Base.keys(sequence::_HostIntSequence) = keys(sequence.values)
Base.getindex(sequence::_HostIntSequence, index::Int) = sequence.values[index]
Base.iterate(sequence::_HostIntSequence, state...) = iterate(sequence.values, state...)
Base.last(sequence::_HostIntSequence) = last(sequence.values)

function _population_with_locations(bank::_PackedDiagonalBank, locations)
    return _PackedDiagonalBank(
        locations,
        bank.scales,
        bank.lognormalizers,
        bank.logmasses,
        bank.cdf,
        bank.proposal_ids,
        bank.layout,
        bank.family,
    )
end

function _population_with_locations(bank::_PackedFactorBank, locations)
    return _PackedFactorBank(
        locations,
        bank.factors,
        bank.lognormalizers,
        bank.logmasses,
        bank.cdf,
        bank.proposal_ids,
        bank.family,
        bank.exponent_bounds,
    )
end

function _population_run_bank(bank)
    locations = similar(bank.locations)
    return _population_with_locations(bank, locations)
end

function _prepare_population_bank(bank::ProposalBank)
    destination_type = _dm_pmc_prepared_proposal_type(
        eltype(bank.masses),
        eltype(bank.proposals),
    )
    return _prepare_population_bank(bank, destination_type)
end

function _prepare_population_bank(bank::ProposalBank, ::Nothing)
    proposal_ids, logmasses, cdf = _prepare_active_proposal_metadata(bank)
    packed = _pack_native_radial_bank(
        bank,
        proposal_ids,
        logmasses,
        cdf,
    )
    packed isa Union{_PackedDiagonalBank,_PackedFactorBank} || throw(
        ArgumentError(
            "fixed population adaptation requires native Float32 or " *
            "Float64 spherical, diagonal, or factor Gaussian or Student-t proposals",
        ),
    )
    return packed
end

function _prepare_population_bank(
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
        ArgumentError(
            "active population proposals do not match their destination type",
        ),
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

    return _pack_native_radial_storage(
        locations,
        lognormalizers,
        logmasses,
        cdf,
        proposal_ids,
        proposals,
        _dm_pmc_gaussian_layout(D),
        _native_radial_pack_kind(D),
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
    sample_budget = _adaptive_sample_budget(rounds, schedule)
    try
        Base.checked_add(1, sample_budget)
    catch error
        error isa OverflowError || rethrow()
        throw(ArgumentError("adaptive allocation offsets exceed Int"))
    end
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

function _prepare_fixed_population_state(
    algorithm::_FixedPopulationSampler,
)
    schedule = _resolve_adaptive_schedule(algorithm.rounds, algorithm.round_size)
    bank = _prepare_population_bank(algorithm.bank)
    active_masses = algorithm.bank.masses[bank.proposal_ids]
    plan = _deterministic_allocation_plan(bank, active_masses, schedule)
    return bank, plan
end

function _transfer_population_plan(device, plan::_DeterministicAllocationPlan)
    return _DeterministicAllocationPlan(
        _HostIntSequence(plan.schedule),
        _copy_to_device(device, plan.counts),
        _copy_to_device(device, plan.assignments),
        _copy_to_device(device, plan.logcoefficients),
        _HostIntSequence(plan.offsets),
    )
end

function _copy_dm_pmc_active_proposal(device, proposal::_NativeRadialProposal)
    adapted = _copy_to_device(device, proposal)
    T = _gaussian_float_type(adapted.location)
    return _radial_proposal(
        adapted.family,
        adapted.location,
        adapted.scale,
        convert(T, adapted.lognormalizer),
    )
end

function _copy_dm_pmc_active_proposal(
    ::MLDataDevices.CPUDevice{T},
    proposal::_StudentTProposal{F,L,<:_SphericalGaussianScale},
) where {T<:_NativeGaussianFloat,F,L}
    return SphericalStudentT(T(proposal.family.dof),
        _copy_dm_pmc_location(T, proposal.location), T(proposal.scale.scale))
end

function _copy_dm_pmc_active_proposal(
    ::MLDataDevices.CPUDevice{T},
    proposal::_StudentTProposal{F,L,<:_DiagonalGaussianScale},
) where {T<:_NativeGaussianFloat,F,L}
    return DiagonalStudentT(T(proposal.family.dof),
        _copy_dm_pmc_location(T, proposal.location), T.(proposal.scale.scales))
end

function _copy_dm_pmc_active_proposal(
    ::MLDataDevices.CPUDevice{T},
    proposal::_StudentTProposal{F,L,<:_FactorGaussianScale},
) where {T<:_NativeGaussianFloat,F,L}
    return FactorStudentT(T(proposal.family.dof),
        _copy_dm_pmc_location(T, proposal.location), T.(proposal.scale.factor))
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

_copy_population_bank(device, bank::ProposalBank{P,M}) where {P,M} =
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

function _copy_population_bank(
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
