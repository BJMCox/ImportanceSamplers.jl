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

function _dm_pmc_allocation_plan(bank, active_masses, schedule)
    active_count = length(bank.proposal_ids)
    rounds = length(schedule)
    counts = Matrix{Int}(undef, active_count, rounds)
    assignments = zeros(Int, maximum(schedule), rounds)
    logcoefficients = Matrix{eltype(active_masses)}(undef, active_count, rounds)
    offsets = Vector{Int}(undef, rounds + 1)
    remainders = similar(active_masses)
    tie_order = collect(1:active_count)
    offsets[1] = 1

    for round in eachindex(schedule)
        round_size = schedule[round]
        for slot in eachindex(active_masses)
            quota = active_masses[slot] * round_size
            count = floor(Int, quota)
            counts[slot, round] = count
            remainders[slot] = quota - count
        end

        remaining = round_size - sum(view(counts, :, round))
        0 <= remaining <= active_count || error("invalid DM-PMC allocation remainder")
        first_tie_slot = mod1(round, active_count)
        sort!(
            tie_order;
            by=slot -> (
                -remainders[slot],
                mod(slot - first_tie_slot, active_count),
            ),
            alg=Base.Sort.MergeSort,
        )
        for index in 1:remaining
            counts[tie_order[index], round] += 1
        end

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

function _copy_algorithm(device, algorithm::DeterministicMixturePMC)
    return DeterministicMixturePMC(
        _copy_to_device(device, algorithm.bank);
        rounds=algorithm.rounds,
        round_size=algorithm.round_size,
    )
end

_copy_algorithm(
    ::MLDataDevices.CPUDevice{Missing},
    algorithm::DeterministicMixturePMC,
) = deepcopy(algorithm)
