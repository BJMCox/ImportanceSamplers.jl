using ImportanceSamplers
using LinearAlgebra
using LogExpFunctions
using Pkg
using Random
using Test

const DM_PMC_GLOBAL_COMMAND =
    "julia --project=validation validation/reproducers/dm_pmc_global.jl"

mutable struct OraclePrefilledRNG{T<:AbstractFloat} <: Random.AbstractRNG
    normal_batches::Vector{Vector{T}}
    uniform_batches::Vector{Vector{T}}
    normal_index::Int
    uniform_index::Int
end

OraclePrefilledRNG(normal_batches::Vector{Vector{T}}, uniform_batches::Vector{Vector{T}}) where {T<:AbstractFloat} =
    OraclePrefilledRNG{T}(normal_batches, uniform_batches, 1, 1)

function Random.randn!(rng::OraclePrefilledRNG, destination::AbstractArray)
    source = rng.normal_batches[rng.normal_index]
    length(source) == length(destination) || throw(
        DimensionMismatch("oracle normal batch length does not match the package buffer"),
    )
    copyto!(destination, source)
    rng.normal_index += 1
    return destination
end

function Random.rand!(rng::OraclePrefilledRNG, destination::AbstractArray)
    source = rng.uniform_batches[rng.uniform_index]
    length(source) == length(destination) || throw(
        DimensionMismatch("oracle uniform batch length does not match the package buffer"),
    )
    copyto!(destination, source)
    rng.uniform_index += 1
    return destination
end

struct OracleGaussian{T<:AbstractFloat}
    location::Vector{T}
    factor::Matrix{T}
    lognormalizer::T
end

struct OracleQuadraticTarget{T<:AbstractFloat}
    dimension::Int
end

function (target::OracleQuadraticTarget{T})(sample) where {T}
    return -T(0.5) * T(target.dimension) * log(T(2pi)) -
           T(0.5) * sum(abs2, sample)
end

function oracle_gaussian_logdensity(proposal::OracleGaussian{T}, sample) where {T}
    delta = T.(sample) .- proposal.location
    standardized = LowerTriangular(proposal.factor) \ delta
    return proposal.lognormalizer - T(0.5) * sum(abs2, standardized)
end

function oracle_assignment(counts)
    return reduce(vcat, (fill(slot, count) for (slot, count) in pairs(counts)))
end

function oracle_strict_upper_bound(cdf, uniform)
    return min(searchsortedlast(cdf, uniform) + 1, lastindex(cdf))
end

function oracle_dm_pmc(
    configured_proposals,
    active_ids,
    schedule,
    expected_counts,
    normal_batches,
    uniform_batches,
    target,
)
    T = eltype(first(configured_proposals).location)
    proposals = configured_proposals[active_ids]
    length(expected_counts) == length(schedule) || throw(
        DimensionMismatch("one explicit allocation row is required per round"),
    )
    locations = reduce(hcat, (copy(proposal.location) for proposal in proposals))
    factors = [copy(proposal.factor) for proposal in proposals]
    all_samples = Matrix{T}(undef, size(locations, 1), 0)
    all_denominators = T[]
    all_logweights = T[]
    all_round_ids = Int[]
    all_proposal_ids = Int[]
    rounds = NamedTuple[]

    for round in eachindex(schedule)
        round_size = schedule[round]
        counts = copy(expected_counts[round])
        length(counts) == length(active_ids) || throw(
            DimensionMismatch("explicit allocation width does not match active proposals"),
        )
        sum(counts) == round_size || throw(
            DimensionMismatch("explicit allocation row does not sum to round size"),
        )
        assignments = oracle_assignment(counts)
        round_proposals = [
            OracleGaussian(
                copy(view(locations, :, slot)),
                factors[slot],
                proposals[slot].lognormalizer,
            ) for slot in eachindex(proposals)
        ]
        round_samples = Matrix{T}(undef, size(locations, 1), round_size)
        round_denominators = Vector{T}(undef, round_size)
        round_logweights = Vector{T}(undef, round_size)
        normals = normal_batches[round]

        for sample_index in 1:round_size
            slot = assignments[sample_index]
            offset = size(locations, 1) * (sample_index - 1)
            standardized = view(normals, offset .+ (1:size(locations, 1)))
            sample = locations[:, slot] + factors[slot] * standardized
            round_samples[:, sample_index] .= sample
            terms = [
                log(T(counts[term_slot]) / T(round_size)) +
                oracle_gaussian_logdensity(round_proposals[term_slot], sample) for
                term_slot in eachindex(counts) if counts[term_slot] > 0
            ]
            denominator = LogExpFunctions.logsumexp(terms)
            round_denominators[sample_index] = denominator
            round_logweights[sample_index] = target(sample) - denominator
        end

        normalized = exp.(
            round_logweights .- LogExpFunctions.logsumexp(round_logweights),
        )
        cdf = cumsum(normalized)
        cdf[end] = one(T)
        ancestors = [
            oracle_strict_upper_bound(cdf, uniform) for
            uniform in uniform_batches[round]
        ]
        next_locations = round_samples[:, ancestors]
        push!(
            rounds,
            (;
                counts,
                assignments,
                locations=copy(locations),
                samples=round_samples,
                denominators=round_denominators,
                logweights=round_logweights,
                ancestors,
                next_locations=copy(next_locations),
            ),
        )
        locations .= next_locations
        all_samples = hcat(all_samples, round_samples)
        append!(all_denominators, round_denominators)
        append!(all_logweights, round_logweights)
        append!(all_round_ids, fill(round, round_size))
        append!(all_proposal_ids, active_ids[assignments])
    end

    return (;
        active_ids,
        samples=all_samples,
        denominators=all_denominators,
        logweights=all_logweights,
        round_ids=all_round_ids,
        proposal_ids=all_proposal_ids,
        rounds,
        final_locations=copy(locations),
    )
end

function oracle_factor(::Type{T}, dimension, slot) where {T}
    factor = zeros(T, dimension, dimension)
    for coordinate in 1:dimension
        factor[coordinate, coordinate] = T(0.65 + 0.04 * slot + 0.02 * coordinate)
        for column in 1:(coordinate - 1)
            factor[coordinate, column] = T(0.025 * (slot + coordinate - column))
        end
    end
    return factor
end

function oracle_case(::Type{T}, kind) where {T}
    dimension = 4
    configured_count = 4
    configured_mass_ratios = T[1, 3, 2, 0]
    oracle_proposals = OracleGaussian{T}[]
    package_proposals = Any[]
    for slot in 1:configured_count
        location = T[
            1.1 * (slot - 2.5) + 0.07 * coordinate for coordinate in 1:dimension
        ]
        factor = oracle_factor(T, dimension, slot)
        if kind === :diagonal
            factor = Matrix(Diagonal(diag(factor)))
            push!(package_proposals, DiagonalGaussian(location, diag(factor)))
        else
            push!(package_proposals, FactorGaussian(location, factor))
        end
        lognormalizer = -T(0.5) * T(dimension) * log(T(2pi)) -
                        sum(log, diag(factor))
        push!(
            oracle_proposals,
            OracleGaussian(copy(location), copy(factor), lognormalizer),
        )
    end
    bank = ProposalBank(package_proposals, configured_mass_ratios)
    schedule = [7, 10, 13]
    active_ids = [1, 3, 2]
    expected_counts = (Int[1, 2, 4], Int[2, 3, 5], Int[2, 4, 7])
    expected_counts_by_id = (Int[1, 4, 2, 0], Int[2, 5, 3, 0], Int[2, 7, 4, 0])
    capacity = maximum(schedule)
    normal_batches = [
        T[
            0.7 * sin(0.31 * index + 0.4 * round) +
            0.15 * cos(0.17 * index - 0.2 * round) for
            index in 1:(dimension * capacity)
        ] for round in eachindex(schedule)
    ]
    active_count = length(active_ids)
    uniform_values = T[0.25, 0.75, 0.5]
    uniform_batches = [fill(uniform_values[round], active_count) for round in eachindex(schedule)]
    return (;
        bank,
        masses=copy(configured_mass_ratios),
        active_ids,
        expected_counts,
        expected_counts_by_id,
        oracle_proposals,
        schedule,
        normal_batches,
        uniform_batches,
        target=OracleQuadraticTarget{T}(dimension),
    )
end

function validate_oracle_tie_rotation(::Type{T}) where {T}
    configured_mass_ratios = T[1, 1, 1, 0]
    proposals = [DiagonalGaussian(T[0.15 * (slot - 2.5)], T[1]) for slot in 1:4]
    schedule = [4, 4, 4]
    expected_counts_by_id = (Int[2, 1, 1, 0], Int[1, 2, 1, 0], Int[1, 1, 2, 0])
    normal_batches = [
        T[0.2 * sin(0.7 * index + round) for index in 1:maximum(schedule)] for
        round in eachindex(schedule)
    ]
    uniform_batches = [fill(T(0.5), 3) for _ in eachindex(schedule)]
    target = OracleQuadraticTarget{T}(1)
    sampler = prepare_sampler(
        OraclePrefilledRNG(normal_batches, uniform_batches),
        target,
        DeterministicMixturePMC(
            ProposalBank(proposals, configured_mass_ratios);
            rounds=length(schedule),
            round_size=schedule,
        );
        threaded=false,
    )
    result = importance_sample!(sampler)
    observed = Tuple(
        [
            count(
                index -> result.provenance.round[index] == round &&
                         result.provenance.proposal_id[index] == proposal_id,
                eachindex(result.logweights),
            ) for proposal_id in eachindex(configured_mass_ratios)
        ] for round in eachindex(schedule)
    )
    @test observed == expected_counts_by_id
    return (; configured_mass_ratios, expected_counts_by_id, observed)
end

function validate_oracle_cdf_boundaries(::Type{T}) where {T}
    cdf = T[0.25, 0.75, 1]
    @test oracle_strict_upper_bound(cdf, T(0.25)) == 2
    @test oracle_strict_upper_bound(cdf, T(0.75)) == 3
    return (cdf=Tuple(cdf), selected=(2, 3))
end

function infer_round_locations(round_samples, assignments, factors, normals)
    dimension = size(round_samples, 1)
    proposal_count = length(factors)
    locations = similar(round_samples, dimension, proposal_count)
    for slot in 1:proposal_count
        sample_index = findfirst(==(slot), assignments)
        offset = dimension * (sample_index - 1)
        standardized = view(normals, offset .+ (1:dimension))
        locations[:, slot] .= round_samples[:, sample_index] -
                              factors[slot] * standardized
    end
    return locations
end

function infer_ancestor_ids(previous_samples, locations, tolerance)
    ancestors = Int[]
    for slot in axes(locations, 2)
        distances = [
            maximum(abs, locations[:, slot] .- previous_samples[:, sample_index]) for
            sample_index in axes(previous_samples, 2)
        ]
        ancestor = argmin(distances)
        @test distances[ancestor] <= tolerance
        push!(ancestors, ancestor)
    end
    return ancestors
end

function validate_oracle_case(::Type{T}, kind) where {T}
    case = oracle_case(T, kind)
    oracle = oracle_dm_pmc(
        case.oracle_proposals,
        case.active_ids,
        case.schedule,
        case.expected_counts,
        case.normal_batches,
        case.uniform_batches,
        case.target,
    )
    sampler = prepare_sampler(
        OraclePrefilledRNG(case.normal_batches, case.uniform_batches),
        case.target,
        DeterministicMixturePMC(
            case.bank;
            rounds=length(case.schedule),
            round_size=case.schedule,
        );
        threaded=false,
    )
    result = importance_sample!(sampler)
    tolerance = T(4096) * eps(T) * T(size(oracle.samples, 1))
    observed_denominators = [
        case.target(view(result.samples, :, index)) - result.logweights[index] for
        index in eachindex(result.logweights)
    ]

    @test length(result) == sum(case.schedule)
    @test size(result.samples) == size(oracle.samples)
    @test result.samples ≈ oracle.samples atol = tolerance rtol = zero(T)
    @test observed_denominators ≈ oracle.denominators atol = tolerance rtol = zero(T)
    @test result.logweights ≈ oracle.logweights atol = tolerance rtol = zero(T)
    @test result.provenance.round == oracle.round_ids
    @test result.provenance.proposal_id == oracle.proposal_ids
    @test 4 ∉ result.provenance.proposal_id
    @test all(round -> length(unique(round.ancestors)) < length(round.ancestors), oracle.rounds)

    factors = [proposal.factor for proposal in case.oracle_proposals[case.active_ids]]
    inferred_ancestors = Vector{Vector{Int}}()
    for round in 2:length(case.schedule)
        observed_range = (sum(case.schedule[1:(round - 1)]) + 1):sum(case.schedule[1:round])
        observed_samples = result.samples[:, observed_range]
        inferred_locations = infer_round_locations(
            observed_samples,
            oracle.rounds[round].assignments,
            factors,
            case.normal_batches[round],
        )
        @test inferred_locations ≈ oracle.rounds[round].locations atol = tolerance rtol = zero(T)
        push!(
            inferred_ancestors,
            infer_ancestor_ids(
                oracle.rounds[round - 1].samples,
                inferred_locations,
                tolerance,
            ),
        )
        @test last(inferred_ancestors) == oracle.rounds[round - 1].ancestors
    end

    method_state = getfield(sampler, :method_state)
    persistent_locations = getfield(getfield(method_state, :bank), :locations)
    final_ancestors = collect(getfield(getfield(method_state, :workspace), :ancestors))
    @test persistent_locations ≈ oracle.final_locations atol = tolerance rtol = zero(T)
    @test final_ancestors == last(oracle.rounds).ancestors
    @test length(unique(final_ancestors)) < length(final_ancestors)
    expected_lognormalizer = LogExpFunctions.logsumexp(oracle.logweights) -
                             log(T(length(oracle.logweights)))
    @test lognormalizer(result) ≈ expected_lognormalizer atol = tolerance rtol = zero(T)

    round_counts = [
        [
            count(
                index -> result.provenance.round[index] == round &&
                         result.provenance.proposal_id[index] == proposal_id,
                eachindex(result.logweights),
            ) for proposal_id in 1:length(case.bank.proposals)
        ] for round in eachindex(case.schedule)
    ]
    @test Tuple(round_counts) == case.expected_counts_by_id

    return (;
        scalar_type=T,
        bank=kind,
        schedule=Tuple(case.schedule),
        total_count=length(result),
        maximum_sample_error=maximum(abs, result.samples .- oracle.samples),
        maximum_denominator_error=maximum(abs, observed_denominators .- oracle.denominators),
        maximum_logweight_error=maximum(abs, result.logweights .- oracle.logweights),
        lognormalizer_error=abs(lognormalizer(result) - expected_lognormalizer),
        proposal_ids=true,
        ancestor_ids=true,
        duplicates=true,
        persistent_population=true,
    )
end

function package_versions()
    wanted = Set(("ImportanceSamplers", "LogExpFunctions"))
    return sort!(
        [
            (dependency.name, something(dependency.version, "unversioned")) for
            dependency in values(Pkg.dependencies()) if dependency.name in wanted
        ];
        by=first,
    )
end

function run_dm_pmc_global_reproducer()
    rows = [
        validate_oracle_case(T, kind) for
        (T, kind) in ((Float64, :diagonal), (Float32, :factor))
    ]
    tie_rotation = validate_oracle_tie_rotation(Float64)
    cdf_boundaries = (
        Float32=validate_oracle_cdf_boundaries(Float32),
        Float64=validate_oracle_cdf_boundaries(Float64),
    )
    return (;
        status=:passed,
        classification=:paper_equation_and_mechanism_check,
        paper_equations=(12, 14, 18, 19, 20),
        command=DM_PMC_GLOBAL_COMMAND,
        deterministic_buffer_identity=:trigonometric_normals_and_fixed_uniforms,
        julia=VERSION,
        packages=package_versions(),
        rows,
        tie_rotation,
        cdf_boundaries,
    )
end

@testset "DM-PMC independent global-resampling reproducer" begin
    fixture = oracle_case(Float64, :diagonal)
    @test fixture.masses == Float64[1, 3, 2, 0]
    @test run_dm_pmc_global_reproducer().status === :passed
end
