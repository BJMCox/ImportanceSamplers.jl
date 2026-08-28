using Test
using ImportanceSamplers
import MLDataDevices
import Random

include("support/dm_pmc.jl")

const DMPMCIS = ImportanceSamplers

@testset "DM-PMC constructor and schedule" begin
    bank = ProposalBank([
        SphericalGaussian(-1.0, 1.0),
        SphericalGaussian(1.0, 1.0),
    ])
    input_schedule = [50, 100, 200]
    fixed = @inferred DeterministicMixturePMC(bank; rounds=3, round_size=100)
    varied = @inferred DeterministicMixturePMC(
        bank;
        rounds=3,
        round_size=input_schedule,
    )

    @test fixed.bank === bank
    @test fixed.rounds === 3
    @test fixed.round_size === 100
    @test varied.round_size == [50, 100, 200]
    @test varied.round_size !== input_schedule
    @test fixed isa AbstractImportanceSampler
    @test fieldnames(typeof(fixed)) == (:bank, :rounds, :round_size)

    input_schedule[1] = 1
    @test varied.round_size == [50, 100, 200]
    resolved = @inferred DMPMCIS._resolve_adaptive_schedule(
        varied.rounds,
        varied.round_size,
    )
    @test resolved == [50, 100, 200]
    @test resolved !== varied.round_size
    resolved[1] = 2
    @test varied.round_size == [50, 100, 200]
    @test @inferred(
        DMPMCIS._resolve_adaptive_schedule(fixed.rounds, fixed.round_size)
    ) == [100, 100, 100]

    @test_throws ArgumentError DeterministicMixturePMC(bank; rounds=0, round_size=100)
    @test_throws ArgumentError DeterministicMixturePMC(bank; rounds=-1, round_size=100)
    @test_throws ArgumentError DeterministicMixturePMC(bank; rounds=true, round_size=100)
    @test_throws ArgumentError DeterministicMixturePMC(bank; rounds=3.0, round_size=100)
    @test_throws ArgumentError DeterministicMixturePMC(bank; rounds=3, round_size=0)
    @test_throws ArgumentError DeterministicMixturePMC(bank; rounds=3, round_size=-1)
    @test_throws ArgumentError DeterministicMixturePMC(
        bank;
        rounds=3,
        round_size=[1, 0, 2],
    )
    @test_throws DimensionMismatch DeterministicMixturePMC(
        bank;
        rounds=3,
        round_size=[1, 2],
    )
    @test_throws ArgumentError DeterministicMixturePMC(
        bank;
        rounds=3,
        round_size=[1.0, 2.0, 3.0],
    )
    @test_throws ArgumentError DeterministicMixturePMC(
        bank;
        rounds=3,
        round_size=() -> 8,
    )
end

@testset "DM-PMC allocation oracle" begin
    configured_bank = ProposalBank(
        [SphericalGaussian(Float64(id), 1.0) for id in 1:4],
        [1, 3, 2, 0],
    )
    algorithm = DeterministicMixturePMC(
        configured_bank;
        rounds=3,
        round_size=[13, 14, 13],
    )
    sampler = @inferred prepare_sampler(
        Random.Xoshiro(0x5600),
        DMPMCTarget{Float64}(),
        algorithm;
        threaded=false,
    )
    state = sampler.method_state
    plan = state.plan
    counts_by_id = dm_pmc_counts_by_proposal_id(state.bank, plan, 4)
    expected_counts = [
        2 2 2
        7 7 7
        4 5 4
        0 0 0
    ]

    @test state isa DMPMCIS._PreparedDMPMC
    @test plan isa DMPMCIS._DMPMCAllocationPlan
    @test plan.schedule == [13, 14, 13]
    @test plan.schedule !== algorithm.round_size
    @test state.bank.proposal_ids == [1, 3, 2]
    @test counts_by_id == expected_counts
    @test vec(sum(plan.counts; dims=1)) == plan.schedule
    @test all(>(0), plan.counts)
    @test plan.offsets == [1, 14, 28, 41]
    @test size(plan.assignments) == (14, 3)
    @test exp.(plan.logcoefficients) ≈
          plan.counts ./ reshape(plan.schedule, 1, :)

    for round in eachindex(plan.schedule)
        used_assignments = view(plan.assignments, 1:plan.schedule[round], round)
        @test all(slot -> 1 <= slot <= length(state.bank.proposal_ids), used_assignments)
        for slot in eachindex(state.bank.proposal_ids)
            @test count(==(slot), used_assignments) == plan.counts[slot, round]
        end
    end

    @test DMPMCIS._algorithm_proposal(algorithm) === configured_bank
    @test DMPMCIS._algorithm_sample_budget(algorithm) == 40
end

@testset "DM-PMC rotating largest-remainder ties" begin
    bank = ProposalBank(
        [SphericalGaussian(Float64(id), 1.0) for id in 1:3],
        [1, 1, 1],
    )
    sampler = @inferred prepare_sampler(
        Random.Xoshiro(0x5601),
        DMPMCTarget{Float64}(),
        DeterministicMixturePMC(bank; rounds=3, round_size=[4, 4, 4]);
        threaded=false,
    )
    plan = sampler.method_state.plan

    @test sampler.method_state.bank.proposal_ids == [1, 2, 3]
    @test plan.counts == [
        2 1 1
        1 2 1
        1 1 2
    ]
end

@testset "DM-PMC large Float32 round counts stay exact" begin
    round_size = 2^24 + 3
    masses = Float32[0.5, 0.5]
    exact_masses = DMPMCIS._dm_pmc_exact_mass_proportions(masses)
    first_round = @inferred DMPMCIS._dm_pmc_round_counts(exact_masses, round_size, 1)
    second_round = @inferred DMPMCIS._dm_pmc_round_counts(exact_masses, round_size, 2)

    @test first_round == [8_388_610, 8_388_609]
    @test second_round == [8_388_609, 8_388_610]
    @test sum(first_round) == round_size
    @test sum(second_round) == round_size
    @test @allocated(DMPMCIS._dm_pmc_round_counts(exact_masses, round_size, 1)) < 1_000_000
end

@testset "DM-PMC allocation converts exact masses once per plan" begin
    bank = ProposalBank(
        [SphericalGaussian(-1.0, 1.0), SphericalGaussian(1.0, 1.0)],
        Float32[1, 3],
    )
    packed = DMPMCIS._prepare_dm_pmc_bank(bank)
    masses = DMPMCCountingMasses(bank.masses[packed.proposal_ids])
    schedule = [13, 14, 13, 14]
    plan = DMPMCIS._dm_pmc_allocation_plan(packed, masses, schedule)

    @test masses.reads[] == length(masses)
    @test vec(sum(plan.counts; dims=1)) == schedule
end

@testset "DM-PMC allocation rejects uncovered active proposals before RNG use" begin
    bank = ProposalBank(
        [SphericalGaussian(Float64(id), 1.0) for id in 1:3],
        [100, 1, 1],
    )
    rng = Random.Xoshiro(0x5602)
    expected_rng = copy(rng)

    @test_throws ArgumentError prepare_sampler(
        rng,
        DMPMCTarget{Float64}(),
        DeterministicMixturePMC(bank; rounds=1, round_size=3);
        threaded=false,
    )
    @test rand(rng, UInt64) == rand(expected_rng, UInt64)
end

@testset "DM-PMC native preparation and scalar inference" begin
    spherical32 = ProposalBank(
        [
            SphericalGaussian(Float32[-1, 0], 1.0f0),
            SphericalGaussian(Float32[1, 0], 1.0f0),
        ],
        Float32[1, 3],
    )
    diagonal64 = ProposalBank([
        DiagonalGaussian([-1.0, 0.0], [1.0, 2.0]),
        DiagonalGaussian([1.0, 0.0], [2.0, 1.0]),
    ])
    factor64 = ProposalBank([
        FactorGaussian([-1.0, 0.0], [1.0 0.0; 0.25 2.0]),
        FactorGaussian([1.0, 0.0], [2.0 0.0; -0.25 1.0]),
    ])

    sampler32 = @inferred prepare_sampler(
        Random.Xoshiro(0x5603),
        DMPMCTarget{Float32}(),
        DeterministicMixturePMC(spherical32; rounds=2, round_size=8);
        threaded=false,
    )
    diagonal_sampler = @inferred prepare_sampler(
        Random.Xoshiro(0x5604),
        DMPMCTarget{Float64}(),
        DeterministicMixturePMC(diagonal64; rounds=2, round_size=[4, 5]);
        threaded=false,
    )
    factor_sampler = @inferred prepare_sampler(
        Random.Xoshiro(0x5605),
        DMPMCTarget{Float64}(),
        DeterministicMixturePMC(factor64; rounds=2, round_size=4);
        threaded=false,
    )

    @test sampler32.method_state.bank isa DMPMCIS._PackedDiagonalGaussianBank
    @test eltype(sampler32.method_state.bank.locations) === Float32
    @test eltype(sampler32.method_state.plan.logcoefficients) === Float32
    @test diagonal_sampler.method_state.bank isa DMPMCIS._PackedDiagonalGaussianBank
    @test eltype(diagonal_sampler.method_state.plan.logcoefficients) === Float64
    @test factor_sampler.method_state.bank isa DMPMCIS._PackedFactorGaussianBank
    @test eltype(factor_sampler.method_state.plan.logcoefficients) === Float64
    @test sampler32.random_buffers isa DMPMCIS._DMPMCRandomBuffers
    @test factor_sampler.random_buffers isa DMPMCIS._DMPMCRandomBuffers

    copied_factor_sampler = @inferred MLDataDevices.cpu_device()(factor_sampler)
    @test copied_factor_sampler !== factor_sampler
    @test copied_factor_sampler.algorithm !== factor_sampler.algorithm
    @test copied_factor_sampler.algorithm.round_size == 4
    @test copied_factor_sampler.method_state.plan.counts ==
          factor_sampler.method_state.plan.counts

    inert_generic = DMPMCGenericProposal()
    configured_bank = ProposalBank(
        Any[SphericalGaussian(0.0, 1.0), inert_generic],
        [1, 0],
    )
    original_proposals = copy(configured_bank.proposals)
    original_masses = copy(configured_bank.masses)
    inert_sampler = prepare_sampler(
        Random.Xoshiro(0x5606),
        DMPMCTarget{Float64}(),
        DeterministicMixturePMC(configured_bank; rounds=1, round_size=4);
        threaded=false,
    )
    @test inert_sampler.method_state.bank isa DMPMCIS._PackedDiagonalGaussianBank
    @test inert_sampler.method_state.bank.proposal_ids == [1]
    @test configured_bank.proposals == original_proposals
    @test configured_bank.masses == original_masses
    @test inert_generic.draw_count[] == 0
end

@testset "DM-PMC Float32 CPU transfer converts spherical proposals" begin
    scalar_bank = ProposalBank(
        [SphericalGaussian(-1.0, 2.0), SphericalGaussian(1.0, 0.5)],
        [1.0, 1.0],
    )
    vector_bank = ProposalBank(
        [
            SphericalGaussian([-1.0, 0.0], 2.0),
            SphericalGaussian([1.0, 0.0], 0.5),
        ],
        [1.0, 1.0],
    )
    scalar_algorithm = DeterministicMixturePMC(
        scalar_bank;
        rounds=1,
        round_size=2,
    )
    vector_algorithm = DeterministicMixturePMC(
        vector_bank;
        rounds=1,
        round_size=2,
    )
    cpu32 = MLDataDevices.cpu_device(Float32)
    copied_scalar_algorithm = @inferred DMPMCIS._copy_algorithm(
        cpu32,
        scalar_algorithm,
    )
    copied_vector_algorithm = @inferred DMPMCIS._copy_algorithm(
        cpu32,
        vector_algorithm,
    )
    scalar_source = prepare_sampler(
        Random.Xoshiro(0x5607),
        DMPMCTarget{Float64}(),
        scalar_algorithm;
        threaded=false,
    )
    vector_source = prepare_sampler(
        Random.Xoshiro(0x5608),
        DMPMCTarget{Float64}(),
        vector_algorithm;
        threaded=false,
    )
    scalar = @inferred cpu32(scalar_source)
    vector = @inferred cpu32(vector_source)

    @test eltype(copied_scalar_algorithm.bank.masses) === Float32
    @test eltype(copied_vector_algorithm.bank.masses) === Float32
    @test all(
        proposal -> proposal.location isa Float32,
        scalar.algorithm.bank.proposals,
    )
    @test all(
        proposal -> proposal.scale.scale isa Float32,
        scalar.algorithm.bank.proposals,
    )
    @test eltype(scalar.method_state.bank.locations) === Float32
    @test eltype(scalar.method_state.bank.scales) === Float32
    @test scalar.method_state.bank.layout isa DMPMCIS._ScalarGaussianLayout
    @test all(
        proposal -> eltype(proposal.location) === Float32,
        vector.algorithm.bank.proposals,
    )
    @test all(
        proposal -> proposal.scale.scale isa Float32,
        vector.algorithm.bank.proposals,
    )
    @test eltype(vector.method_state.bank.locations) === Float32
    @test eltype(vector.method_state.bank.scales) === Float32
    @test vector.method_state.bank.layout isa DMPMCIS._VectorGaussianLayout
    @test scalar_source.algorithm.bank.proposals[1].location isa Float64
    @test vector_source.algorithm.bank.proposals[1].scale.scale isa Float64
end

@testset "DM-PMC Float32 CPU transfer preserves mixed native banks" begin
    spherical = SphericalGaussian([-1.0, 0.0], 2.0)
    diagonal = DiagonalGaussian([1.0, 0.0], [2.0, 1.0])
    factor = FactorGaussian([1.0, 0.0], [2.0 0.0; 0.25 1.0])
    erased_bank = ProposalBank(
        DMPMCIS._GaussianProposal[spherical, diagonal],
        [1.0, 1.0],
    )
    spherical_diagonal_bank = ProposalBank(
        Union{typeof(spherical),typeof(diagonal)}[spherical, diagonal],
        [1.0, 1.0],
    )
    diagonal_factor_bank = ProposalBank(
        Union{typeof(diagonal),typeof(factor)}[diagonal, factor],
        [1.0, 1.0],
    )
    inert_factor = FactorGaussian([2.0, 0.0], [1.0 0.0; 0.5 1.0])
    inert_bank = ProposalBank(
        Union{typeof(diagonal),typeof(inert_factor)}[diagonal, inert_factor],
        [1.0, 0.0],
    )
    source_banks = (
        erased_bank,
        spherical_diagonal_bank,
        diagonal_factor_bank,
        inert_bank,
    )
    sources = map(enumerate(source_banks)) do (index, bank)
        prepare_sampler(
            Random.Xoshiro(0x5608 + index),
            DMPMCTarget{Float64}(),
            DeterministicMixturePMC(bank; rounds=1, round_size=2);
            threaded=false,
        )
    end
    cpu32 = MLDataDevices.cpu_device(Float32)
    erased, spherical_diagonal, diagonal_factor, inert = map(cpu32, sources)

    @test sources[1].method_state.bank isa DMPMCIS._PackedDiagonalGaussianBank
    @test sources[2].method_state.bank isa DMPMCIS._PackedDiagonalGaussianBank
    @test sources[3].method_state.bank isa DMPMCIS._PackedFactorGaussianBank
    @test sources[4].method_state.bank isa DMPMCIS._PackedDiagonalGaussianBank
    @test sources[4].method_state.bank.proposal_ids == [1]

    for copied in (erased, spherical_diagonal, diagonal_factor, inert)
        @test eltype(copied.algorithm.bank.masses) === Float32
        @test eltype(copied.method_state.bank.lognormalizers) === Float32
        @test eltype(copied.method_state.plan.logcoefficients) === Float32
    end
    @test erased.method_state.bank.proposal_ids == [1, 2]
    @test spherical_diagonal.method_state.bank.proposal_ids == [1, 2]
    @test diagonal_factor.method_state.bank.proposal_ids == [1, 2]
    @test all(
        proposal -> eltype(proposal.location) === Float32,
        erased.algorithm.bank.proposals,
    )
    @test all(
        proposal -> eltype(proposal.location) === Float32,
        spherical_diagonal.algorithm.bank.proposals,
    )
    @test diagonal_factor.method_state.bank isa DMPMCIS._PackedFactorGaussianBank
    @test eltype(diagonal_factor.method_state.bank.locations) === Float32
    @test eltype(diagonal_factor.method_state.bank.factors) === Float32
    @test inert.method_state.bank isa DMPMCIS._PackedDiagonalGaussianBank
    @test inert.method_state.bank.proposal_ids == [1]
    @test eltype(inert.algorithm.bank.proposals[1].location) === Float32
    @test inert.algorithm.bank.proposals[2] !== inert_factor
    @test inert.algorithm.bank.proposals[2].location !== inert_factor.location
    @test inert.algorithm.bank.proposals[2].scale.factor !== inert_factor.scale.factor
    @test eltype(inert.algorithm.bank.proposals[2].location) === Float64
end

@testset "DM-PMC Float32 CPU transfer recomputes Gaussian caches" begin
    source_scale = 4.414264425841938e-5
    diagonal_bank = ProposalBank(
        [DiagonalGaussian([0.0], [source_scale])],
        [1.0],
    )
    factor_bank = ProposalBank(
        [FactorGaussian([0.0], reshape([source_scale], 1, 1))],
        [1.0],
    )
    cpu32 = MLDataDevices.cpu_device(Float32)
    diagonal = cpu32(
        prepare_sampler(
            Random.Xoshiro(0x5609),
            DMPMCTarget{Float64}(),
            DeterministicMixturePMC(diagonal_bank; rounds=1, round_size=1);
            threaded=false,
        ),
    )
    factor = cpu32(
        prepare_sampler(
            Random.Xoshiro(0x560a),
            DMPMCTarget{Float64}(),
            DeterministicMixturePMC(factor_bank; rounds=1, round_size=1);
            threaded=false,
        ),
    )

    @test diagonal.algorithm.bank.proposals[1].lognormalizer === 9.109145f0
    @test factor.algorithm.bank.proposals[1].lognormalizer === 9.109145f0
    @test diagonal.method_state.bank.lognormalizers == Float32[9.109145]
    @test factor.method_state.bank.lognormalizers == Float32[9.109145]
end

@testset "DM-PMC Float32 CPU transfer rejects invalid narrowing" begin
    invalid_proposals = (
        DiagonalGaussian([0.0], [1e-50]),
        FactorGaussian([0.0], reshape([1e-50], 1, 1)),
        DiagonalGaussian([1e100], [1.0]),
    )
    cpu32 = MLDataDevices.cpu_device(Float32)

    for (index, proposal) in pairs(invalid_proposals)
        source = prepare_sampler(
            Random.Xoshiro(0x560a + index),
            DMPMCTarget{Float64}(),
            DeterministicMixturePMC(
                ProposalBank([proposal]);
                rounds=1,
                round_size=1,
            );
            threaded=false,
        )
        @test_throws ArgumentError cpu32(source)
    end

    inert = DiagonalGaussian([1e100], [1.0])
    inert_source = prepare_sampler(
        Random.Xoshiro(0x560e),
        DMPMCTarget{Float64}(),
        DeterministicMixturePMC(
            ProposalBank(
                [DiagonalGaussian([0.0], [1.0]), inert],
                [1.0, 0.0],
            );
            rounds=1,
            round_size=1,
        );
        threaded=false,
    )
    copied_inert = cpu32(inert_source).algorithm.bank.proposals[2]
    @test copied_inert !== inert
    @test copied_inert.location == inert.location
    @test copied_inert.location !== inert.location
    @test copied_inert.scale.scales == inert.scale.scales
    @test copied_inert.scale.scales !== inert.scale.scales
    @test copied_inert.lognormalizer === inert.lognormalizer
    @test eltype(copied_inert.location) === Float64
end

@testset "DM-PMC Float32 CPU transfer converts complete Gaussian state" begin
    diagonal_bank = ProposalBank(
        [
            DiagonalGaussian([-1.0, 0.0], [1.0, 2.0]),
            DiagonalGaussian([1.0, 0.0], [2.0, 1.0]),
        ],
        [3.0, 1.0],
    )
    factor_bank = ProposalBank(
        [
            FactorGaussian([-1.0, 0.0], [1.0 0.0; 0.25 2.0]),
            FactorGaussian([1.0, 0.0], [2.0 0.0; -0.25 1.0]),
        ],
        [1.0, 3.0],
    )
    diagonal_algorithm = DeterministicMixturePMC(
        diagonal_bank;
        rounds=2,
        round_size=[4, 5],
    )
    factor_algorithm = DeterministicMixturePMC(
        factor_bank;
        rounds=2,
        round_size=4,
    )
    cpu32 = MLDataDevices.cpu_device(Float32)
    copied_diagonal_algorithm = @inferred DMPMCIS._copy_algorithm(
        cpu32,
        diagonal_algorithm,
    )
    copied_factor_algorithm = @inferred DMPMCIS._copy_algorithm(
        cpu32,
        factor_algorithm,
    )
    diagonal_source = prepare_sampler(
        Random.Xoshiro(0x5607),
        DMPMCTarget{Float64}(),
        diagonal_algorithm;
        threaded=false,
    )
    factor_source = prepare_sampler(
        Random.Xoshiro(0x5608),
        DMPMCTarget{Float64}(),
        factor_algorithm;
        threaded=false,
    )
    diagonal = @inferred cpu32(diagonal_source)
    factor = @inferred cpu32(factor_source)

    @test eltype(copied_diagonal_algorithm.bank.masses) === Float32
    @test eltype(copied_factor_algorithm.bank.masses) === Float32
    @test eltype(diagonal.algorithm.bank.masses) === Float32
    @test all(
        proposal -> eltype(proposal.location) === Float32,
        diagonal.algorithm.bank.proposals,
    )
    @test all(
        proposal -> eltype(proposal.scale.scales) === Float32,
        diagonal.algorithm.bank.proposals,
    )
    @test all(
        proposal -> proposal.lognormalizer isa Float32,
        diagonal.algorithm.bank.proposals,
    )
    @test eltype(diagonal.method_state.bank.locations) === Float32
    @test diagonal.method_state.bank isa DMPMCIS._PackedDiagonalGaussianBank
    @test eltype(diagonal.method_state.bank.scales) === Float32
    @test eltype(diagonal.method_state.bank.lognormalizers) === Float32
    @test eltype(diagonal.method_state.bank.logmasses) === Float32
    @test eltype(diagonal.method_state.plan.logcoefficients) === Float32
    @test diagonal.method_state.bank.proposal_ids ==
          diagonal_source.method_state.bank.proposal_ids == [2, 1]

    @test eltype(factor.algorithm.bank.masses) === Float32
    @test all(
        proposal -> eltype(proposal.location) === Float32,
        factor.algorithm.bank.proposals,
    )
    @test all(
        proposal -> eltype(proposal.scale.factor) === Float32,
        factor.algorithm.bank.proposals,
    )
    @test all(
        proposal -> proposal.lognormalizer isa Float32,
        factor.algorithm.bank.proposals,
    )
    @test eltype(factor.method_state.bank.locations) === Float32
    @test factor.method_state.bank isa DMPMCIS._PackedFactorGaussianBank
    @test eltype(factor.method_state.bank.factors) === Float32
    @test eltype(factor.method_state.bank.lognormalizers) === Float32
    @test eltype(factor.method_state.bank.logmasses) === Float32
    @test eltype(factor.method_state.plan.logcoefficients) === Float32
    @test factor.method_state.bank.proposal_ids ==
          factor_source.method_state.bank.proposal_ids == [1, 2]

    @test eltype(diagonal_source.method_state.bank.locations) === Float64
    @test eltype(factor_source.method_state.bank.locations) === Float64
end

@testset "DM-PMC type seams are ambiguity-free" begin
    ambiguities = Test.detect_ambiguities(DMPMCIS; recursive=false)
    dm_pmc_ambiguities = filter(ambiguities) do ambiguity
        any(
            method -> occursin("_dm_pmc", string(method.name)),
            ambiguity,
        )
    end
    unbound_methods = Test.detect_unbound_args(DMPMCIS)
    dm_pmc_unbound_methods = filter(unbound_methods) do method
        occursin("_dm_pmc", string(method.name))
    end

    @test isempty(dm_pmc_ambiguities)
    @test isempty(dm_pmc_unbound_methods)
end

@testset "DM-PMC rejects unsupported banks before RNG use" begin
    transformed = ProposalBank([
        TransformedProposal(SphericalGaussian(-1.0, 1.0), IdentityTransform()),
        TransformedProposal(SphericalGaussian(1.0, 1.0), IdentityTransform()),
    ])
    product = ProposalBank([
        ProductProposal((x=SphericalGaussian(-1.0, 1.0),)),
        ProductProposal((x=SphericalGaussian(1.0, 1.0),)),
    ])
    generic = ProposalBank([DMPMCGenericProposal(), DMPMCGenericProposal()])
    bigfloat_masses = ProposalBank(
        [SphericalGaussian(-1.0, 1.0), SphericalGaussian(1.0, 1.0)],
        BigFloat[1, 1],
    )

    for (index, bank) in pairs((transformed, product, generic, bigfloat_masses))
        rng = Random.Xoshiro(0x5610 + index)
        expected_rng = copy(rng)
        @test_throws ArgumentError prepare_sampler(
            rng,
            _ -> 0.0,
            DeterministicMixturePMC(bank; rounds=1, round_size=4);
            threaded=false,
        )
        @test rand(rng, UInt64) == rand(expected_rng, UInt64)
    end
end

@testset "ImportanceSampling preparation and execution remain unchanged" begin
    proposal = SphericalGaussian(0.0, 1.0)
    algorithm = ImportanceSampling(proposal; nsamples=8)
    sampler = @inferred prepare_sampler(
        Random.Xoshiro(0x5620),
        DMPMCTarget{Float64}(),
        algorithm;
        threaded=false,
    )
    result = @inferred importance_sample!(sampler)

    @test DMPMCIS._algorithm_proposal(algorithm) === proposal
    @test DMPMCIS._algorithm_sample_budget(algorithm) === 8
    @test sampler.method_state isa DMPMCIS._SingleProposalMethodState
    @test length(result) == 8
    @test result.diagnostics.method === :importance_sampling
end

@testset "DM-PMC execution is available" begin
    bank = ProposalBank([
        SphericalGaussian(-1.0, 1.0),
        SphericalGaussian(1.0, 1.0),
    ])
    sampler = prepare_sampler(
        Random.Xoshiro(0x5630),
        DMPMCTarget{Float64}(),
        DeterministicMixturePMC(bank; rounds=2, round_size=4);
        threaded=false,
    )

    result = importance_sample!(sampler)

    @test result isa WeightedSamples
    @test length(result) == 8
end

dm_pmc_context_target(sample, context) = context.shift - abs2(sample) / 2

@testset "DM-PMC one-shot execution" begin
    algorithm = DeterministicMixturePMC(
        ProposalBank([SphericalGaussian(-1.0, 1.0), SphericalGaussian(1.0, 1.0)]);
        rounds=2,
        round_size=4,
    )

    context_free = importance_sample(
        Random.Xoshiro(0x5631),
        DMPMCTarget{Float64}(),
        algorithm;
        threaded=false,
    )
    contextual = importance_sample(
        Random.Xoshiro(0x5632),
        dm_pmc_context_target,
        (shift=0.25,),
        algorithm;
        threaded=false,
    )

    @test length(context_free) == 8
    @test length(contextual) == 8
    @test contextual.diagnostics.method === :deterministic_mixture_pmc

    importance_algorithm = ImportanceSampling(
        SphericalGaussian(0.0, 1.0);
        nsamples=4,
    )
    context_free_method = which(
        importance_sample,
        (Random.Xoshiro, DMPMCTarget{Float64}, typeof(algorithm)),
    )
    importance_method = which(
        importance_sample,
        (Random.Xoshiro, DMPMCTarget{Float64}, typeof(importance_algorithm)),
    )
    contextual_method = which(
        importance_sample,
        (
            Random.Xoshiro,
            typeof(dm_pmc_context_target),
            @NamedTuple{shift::Float64},
            typeof(algorithm),
        ),
    )
    contextual_importance_method = which(
        importance_sample,
        (
            Random.Xoshiro,
            typeof(dm_pmc_context_target),
            @NamedTuple{shift::Float64},
            typeof(importance_algorithm),
        ),
    )

    @test context_free_method === importance_method
    @test contextual_method === contextual_importance_method
end

function caught_dm_pmc_error(f)
    try
        f()
    catch error
        return error
    end
    return nothing
end

function make_dm_pmc_oracle_case(::Type{T}, repeats=1) where {T}
    schedule = [11, 13, 17]
    maximum_round_size = maximum(schedule)
    normal_batches = [
        T.([mod(index + 3round, 11) - 5 for index in 1:maximum_round_size]) ./ T(7)
        for round in 1:(3repeats)
    ]
    uniform_batches = [
        T[mod(T(0.17) * round, one(T)), mod(T(0.73) * round, one(T))]
        for round in 1:(3repeats)
    ]
    bank = ProposalBank(
        [
            SphericalGaussian(T(-2), T(0.75)),
            SphericalGaussian(T(99), one(T)),
            SphericalGaussian(T(2), T(1.25)),
        ],
        T[1, 0, 1],
    )
    algorithm = DeterministicMixturePMC(bank; rounds=3, round_size=schedule)
    return (; schedule, normal_batches, uniform_batches, bank, algorithm)
end

@testset "DM-PMC all-round result, oracle, and inference" begin
    for T in (Float32, Float64)
        case = make_dm_pmc_oracle_case(T)
        rng = DMPMCPrefilledRNG(case.normal_batches, case.uniform_batches)
        sampler = @inferred prepare_sampler(
            rng,
            DMPMCTarget{T}(),
            case.algorithm;
            threaded=false,
        )
        initial_locations = vec(copy(sampler.method_state.bank.locations))
        scales = vec(copy(sampler.method_state.bank.scales))
        oracle = dm_pmc_scalar_oracle(
            initial_locations,
            scales,
            sampler.method_state.bank.proposal_ids,
            sampler.method_state.plan,
            case.normal_batches,
            case.uniform_batches,
            DMPMCTarget{T}(),
        )

        result = @inferred importance_sample!(sampler)

        @test length(result) == 41
        @test result.samples ≈ oracle.samples rtol = 32eps(T)
        @test result.logweights ≈ oracle.logweights rtol = 64eps(T)
        @test result.provenance.round == oracle.rounds
        @test result.provenance.proposal_id == oracle.proposal_ids
        @test count(==(1), result.provenance.round) == 11
        @test count(==(2), result.provenance.round) == 13
        @test count(==(3), result.provenance.round) == 17
        @test unique(result.provenance.proposal_id) == [1, 3]
        @test vec(sampler.method_state.bank.locations) ≈ oracle.locations rtol = 32eps(T)
        expected_logz = LogExpFunctions.logsumexp(oracle.logweights) - log(T(41))
        @test lognormalizer(result) ≈ expected_logz rtol = 64eps(T)
        @test eltype(result.samples) === T
        @test eltype(result.logweights) === T
        @test sampler.method_state.workspace isa DMPMCIS._DMPMCWorkspace
        @test result.diagnostics.method === :deterministic_mixture_pmc
        @test result.diagnostics.execution === :serial
        @test result.diagnostics.round_sizes == case.schedule
        @test length(result.diagnostics.round_ess) == 3
        @test length(result.diagnostics.round_lognormalizers) == 3
        @test result.diagnostics.failures == 0
        @test result.diagnostics.transfers.count == 0
        @test result.diagnostics.transfers.bytes == 0
    end
end

@testset "DM-PMC prepared state persists without result aliasing" begin
    T = Float64
    case = make_dm_pmc_oracle_case(T, 2)
    sampler = prepare_sampler(
        DMPMCPrefilledRNG(case.normal_batches, case.uniform_batches),
        DMPMCTarget{T}(),
        case.algorithm;
        threaded=false,
    )
    initial_locations = vec(copy(sampler.method_state.bank.locations))
    scales = vec(copy(sampler.method_state.bank.scales))
    first_oracle = dm_pmc_scalar_oracle(
        initial_locations,
        scales,
        sampler.method_state.bank.proposal_ids,
        sampler.method_state.plan,
        case.normal_batches[1:3],
        case.uniform_batches[1:3],
        DMPMCTarget{T}(),
    )
    second_oracle = dm_pmc_scalar_oracle(
        first_oracle.locations,
        scales,
        sampler.method_state.bank.proposal_ids,
        sampler.method_state.plan,
        case.normal_batches[4:6],
        case.uniform_batches[4:6],
        DMPMCTarget{T}(),
    )

    first_result = importance_sample!(sampler)
    first_snapshot = (
        samples=copy(first_result.samples),
        logweights=copy(first_result.logweights),
        round=copy(first_result.provenance.round),
        proposal_id=copy(first_result.provenance.proposal_id),
    )
    second_result = importance_sample!(sampler)

    @test first_result.samples == first_snapshot.samples
    @test first_result.logweights == first_snapshot.logweights
    @test first_result.provenance.round == first_snapshot.round
    @test first_result.provenance.proposal_id == first_snapshot.proposal_id
    @test first_result.samples !== second_result.samples
    @test first_result.logweights !== second_result.logweights
    @test second_result.samples ≈ second_oracle.samples rtol = 32eps(T)
    @test vec(sampler.method_state.bank.locations) ≈ second_oracle.locations rtol = 32eps(T)
    @test second_result.samples[1] ≈
          first_oracle.locations[sampler.method_state.plan.assignments[1, 1]] +
          scales[sampler.method_state.plan.assignments[1, 1]] * case.normal_batches[4][1]
end

@testset "DM-PMC current proposal snapshots are independent" begin
    inert = FactorGaussian([9.0, -9.0], [2.0 0.0; 0.5 1.5])
    bank = ProposalBank(
        Union{
            typeof(DiagonalGaussian([-1.0, 0.0], [0.8, 1.2])),
            typeof(inert),
        }[
            DiagonalGaussian([-1.0, 0.0], [0.8, 1.2]),
            inert,
            DiagonalGaussian([1.0, 0.0], [1.1, 0.7]),
        ],
        [1.0, 0.0, 2.0],
    )
    sampler = prepare_sampler(
        Random.Xoshiro(0x43555252454e54),
        DMPMCTarget{Float64}(),
        DeterministicMixturePMC(bank; rounds=2, round_size=[5, 7]);
        threaded=false,
    )

    initial = @inferred current_proposal(sampler)
    explicit_initial = @inferred current_proposal(
        MLDataDevices.cpu_device(),
        sampler,
    )
    @test initial isa ProposalBank
    @test explicit_initial isa ProposalBank
    @test explicit_initial !== initial
    @test explicit_initial.proposals !== initial.proposals
    @test explicit_initial.masses !== initial.masses
    @test explicit_initial.proposals[1].location == initial.proposals[1].location

    converting_destination_error = try
        current_proposal(MLDataDevices.cpu_device(Float32), sampler)
        nothing
    catch error
        error
    end
    @test converting_destination_error isa ArgumentError
    @test occursin("preserving CPU destination", converting_destination_error.msg)
    @test initial.masses == bank.masses
    @test length(initial.proposals) == 3
    @test initial.proposals[2].location == inert.location
    @test initial.proposals[2] !== sampler.algorithm.bank.proposals[2]
    @test initial.proposals[2].location !== sampler.algorithm.bank.proposals[2].location
    @test initial.proposals[2].scale.factor !==
          sampler.algorithm.bank.proposals[2].scale.factor

    importance_sample!(sampler)
    adapted = @inferred current_proposal(sampler)
    @test adapted.masses == bank.masses
    @test adapted.proposals[1].location == sampler.method_state.bank.locations[:, 1]
    @test adapted.proposals[3].location == sampler.method_state.bank.locations[:, 2]
    @test adapted.proposals[2].location == inert.location

    retained_locations = copy(sampler.method_state.bank.locations)
    adapted.proposals[1].location[1] = 1.0e6
    adapted.proposals[2].location[1] = -1.0e6
    adapted.proposals[2].scale.factor[1, 1] = 1.0e6
    adapted.masses[1] = 0.0
    @test sampler.method_state.bank.locations == retained_locations
    @test sampler.algorithm.bank.proposals[2].location == inert.location
    @test sampler.algorithm.bank.proposals[2].scale.factor == inert.scale.factor
    @test sampler.algorithm.bank.masses == bank.masses

    plain = prepare_sampler(
        Random.Xoshiro(0x43555252454e55),
        DMPMCTarget{Float64}(),
        ImportanceSampling(SphericalGaussian(0.0, 1.0); nsamples=2);
        threaded=false,
    )
    @test_throws MethodError current_proposal(plain)

    nonidempotent_masses = ProposalBank(
        [SphericalGaussian(Float64(index), 1.0) for index in 1:4],
        [0.1, 0.2, 0.3, 0.4],
    )
    nonidempotent_masses.masses .= [0.1, 0.2, 0.3, 0.4]
    mass_sampler = prepare_sampler(
        Random.Xoshiro(0x43555252454e56),
        DMPMCTarget{Float64}(),
        DeterministicMixturePMC(nonidempotent_masses; rounds=1, round_size=10);
        threaded=false,
    )
    @test current_proposal(mass_sampler).masses == nonidempotent_masses.masses
    @test current_proposal(
        MLDataDevices.cpu_device(),
        mass_sampler,
    ).masses == nonidempotent_masses.masses
end

@testset "DM-PMC vector diagonal and factor CPU execution" begin
    for (index, bank) in pairs((
        ProposalBank([
            DiagonalGaussian([-1.0, 0.0], [1.0, 2.0]),
            DiagonalGaussian([1.0, 0.0], [2.0, 1.0]),
        ]),
        ProposalBank([
            FactorGaussian([-1.0, 0.0], [1.0 0.0; 0.25 2.0]),
            FactorGaussian([1.0, 0.0], [2.0 0.0; -0.25 1.0]),
        ]),
    ))
        algorithm = DeterministicMixturePMC(bank; rounds=2, round_size=[5, 7])
        result = @inferred importance_sample!(
            prepare_sampler(
                Random.Xoshiro(0x5640 + index),
                DMPMCTarget{Float64}(),
                algorithm;
                threaded=false,
            ),
        )

        @test size(result.samples) == (2, 12)
        @test length(result.logweights) == 12
        @test count(==(1), result.provenance.round) == 5
        @test count(==(2), result.provenance.round) == 7
    end
end

@testset "DM-PMC warmed allocations are result-sized" begin
    round_size = 4_096
    rounds = 3
    sampler = prepare_sampler(
        Random.Xoshiro(0x5650),
        DMPMCTarget{Float64}(),
        DeterministicMixturePMC(
            ProposalBank([
                SphericalGaussian(-1.0, 0.75),
                SphericalGaussian(1.0, 1.25),
            ]);
            rounds,
            round_size,
        );
        threaded=false,
    )
    importance_sample!(sampler)
    allocation = @allocated importance_sample!(sampler)
    result_storage = rounds * round_size * (
        2sizeof(Float64) + 2sizeof(Int)
    )

    @test allocation <= result_storage + 128_000
end

@testset "DM-PMC serial and threaded consume identical prefilled buffers" begin
    T = Float64
    case = make_dm_pmc_oracle_case(T)
    serial = prepare_sampler(
        DMPMCPrefilledRNG(deepcopy(case.normal_batches), deepcopy(case.uniform_batches)),
        DMPMCTarget{T}(),
        case.algorithm;
        threaded=false,
    )
    threaded = prepare_sampler(
        DMPMCPrefilledRNG(deepcopy(case.normal_batches), deepcopy(case.uniform_batches)),
        DMPMCTarget{T}(),
        case.algorithm;
        threaded=true,
    )

    serial_result = importance_sample!(serial)
    threaded_result = importance_sample!(threaded)

    @test threaded_result.samples == serial_result.samples
    @test threaded_result.logweights == serial_result.logweights
    @test threaded_result.provenance == serial_result.provenance
    @test threaded.method_state.bank.locations == serial.method_state.bank.locations
end

@testset "DM-PMC exact mixture weights and duplicate ancestors" begin
    for T in (Float32, Float64)
        bank = ProposalBank(
            [SphericalGaussian(T(-1), one(T)), SphericalGaussian(T(1), one(T))],
            T[1, 1],
        )
        target = DMPMCMixtureTarget(T[-1, 1], T[1, 1], T[log(T(0.5)), log(T(0.5))])
        sampler = prepare_sampler(
            DMPMCPrefilledRNG([T[-1, 0, 1, 0.5]], [T[0.1, 0.1]]),
            target,
            DeterministicMixturePMC(bank; rounds=1, round_size=4);
            threaded=false,
        )

        result = @inferred importance_sample!(sampler)

        @test all(iszero, result.logweights)
        @test sampler.method_state.workspace.ancestors == [1, 1]
        @test vec(sampler.method_state.bank.locations) == fill(result.samples[1], 2)
    end
end

@testset "DM-PMC round failures preserve the last committed population" begin
    T = Float64
    schedule = [4, 4, 4]
    normals = [fill(T(round) / 10, 4) for round in 1:3]
    uniforms = [T[0.1, 0.9] for _ in 1:3]
    bank = ProposalBank([SphericalGaussian(T(-1), one(T)), SphericalGaussian(T(1), one(T))])
    algorithm = DeterministicMixturePMC(bank; rounds=3, round_size=schedule)

    target_failure_target = DMPMCFailAfterTarget(T, 5)
    target_failure_rng = DMPMCPrefilledRNG(deepcopy(normals), deepcopy(uniforms))
    target_failure_sampler = prepare_sampler(
        target_failure_rng,
        target_failure_target,
        algorithm;
        threaded=false,
    )
    target_oracle = dm_pmc_scalar_oracle(
        T[-1, 1],
        T[1, 1],
        [1, 2],
        target_failure_sampler.method_state.plan,
        normals[1:1],
        uniforms[1:1],
        DMPMCTarget{T}(),
    )
    target_failure = caught_dm_pmc_error() do
        importance_sample!(target_failure_sampler)
    end

    @test target_failure isa DMPMCRoundError
    @test target_failure.round == 2
    @test target_failure.phase == :sample_and_weight
    @test target_failure.cause isa SamplerExecutionError
    @test vec(target_failure_sampler.method_state.bank.locations) == target_oracle.locations
    @test target_failure_rng.normal_index == 3
    @test target_failure_rng.uniform_index == 2
    @test occursin("round 2", sprint(showerror, target_failure))
    @test occursin("sample_and_weight", sprint(showerror, target_failure))
    @test occursin("intentional DM-PMC target failure", sprint(showerror, target_failure))

    zero_target = DMPMCNegativeInfinityAfterTarget(T, 4)
    zero_rng = DMPMCPrefilledRNG(deepcopy(normals), deepcopy(uniforms))
    zero_sampler = prepare_sampler(
        zero_rng,
        zero_target,
        algorithm;
        threaded=false,
    )
    zero_failure = caught_dm_pmc_error() do
        importance_sample!(zero_sampler)
    end

    @test zero_failure isa DMPMCRoundError
    @test zero_failure.round == 2
    @test zero_failure.phase == :resampling
    @test zero_failure.cause isa AllZeroWeightsError
    @test vec(zero_sampler.method_state.bank.locations) == target_oracle.locations
    @test zero_rng.normal_index == 3
    @test zero_rng.uniform_index == 2
end
