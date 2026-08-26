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
    resolved = @inferred DMPMCIS._resolve_round_schedule(varied)
    @test resolved == [50, 100, 200]
    @test resolved !== varied.round_size
    resolved[1] = 2
    @test varied.round_size == [50, 100, 200]
    @test @inferred(DMPMCIS._resolve_round_schedule(fixed)) == [100, 100, 100]

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
    first_round = @inferred DMPMCIS._dm_pmc_round_counts(masses, round_size, 1)
    second_round = @inferred DMPMCIS._dm_pmc_round_counts(masses, round_size, 2)

    @test first_round == [8_388_610, 8_388_609]
    @test second_round == [8_388_609, 8_388_610]
    @test sum(first_round) == round_size
    @test sum(second_round) == round_size
    @test @allocated(DMPMCIS._dm_pmc_round_counts(masses, round_size, 1)) < 1_000_000
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
    @test sampler32.random_buffers isa DMPMCIS._NoRandomBuffers
    @test factor_sampler.random_buffers isa DMPMCIS._NoRandomBuffers

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
