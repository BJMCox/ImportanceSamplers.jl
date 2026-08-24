using Test
using ImportanceSamplers

@testset "proposal-bank configuration" begin
    proposals = [
        SphericalGaussian(-1.0, 1.0),
        SphericalGaussian(1.0, 1.0),
    ]

    equal_bank = @inferred ProposalBank(proposals)
    weighted_bank = @inferred ProposalBank(proposals, [1, 3])

    @test ProposalBank <: AbstractProposalPopulation
    @test equal_bank.proposals !== proposals
    @test equal_bank.proposals == proposals
    @test equal_bank.masses == [0.5, 0.5]
    @test weighted_bank.masses == [0.25, 0.75]

    largest = floatmax(Float64)
    near_overflow = ProposalBank(proposals, [largest, largest / 2])
    @test near_overflow.masses ≈ [2 / 3, 1 / 3]

    smallest = nextfloat(0.0)
    subnormal = ProposalBank(proposals, [smallest, 2smallest])
    @test subnormal.masses ≈ [1 / 3, 2 / 3]
    @test all(>(0), subnormal.masses)

    integer_overflow = ProposalBank(proposals, fill(typemax(Int), 2))
    @test integer_overflow.masses == [0.5, 0.5]

    low_precision_count = 70_000
    low_precision = ProposalBank(
        fill(first(proposals), low_precision_count),
        fill(Float16(1), low_precision_count),
    )
    @test eltype(low_precision.masses) === Float32
    @test all(>(0), low_precision.masses)
    @test isapprox(sum(low_precision.masses), 1; rtol=4eps(Float32))
    @test eltype(ProposalBank(proposals, Float64[1, 1]).masses) === Float64
    @test eltype(ProposalBank(proposals, BigFloat[1, 1]).masses) === BigFloat

    @test_throws ArgumentError ProposalBank(typeof(proposals)())
    @test_throws ArgumentError ProposalBank(proposals, [1])
    @test_throws ArgumentError ProposalBank(proposals, Bool[true, false])
    @test_throws ArgumentError ProposalBank(proposals, [1, -1])
    @test_throws ArgumentError ProposalBank(proposals, [1.0, NaN])
    @test_throws ArgumentError ProposalBank(proposals, [1.0, Inf])
    @test_throws ArgumentError ProposalBank(proposals, [1.0, -Inf])
    @test_throws ArgumentError ProposalBank(proposals, [0, 0])
    @test_throws ArgumentError ProposalBank(proposals, Real[1, 3])
    @test_throws ArgumentError ProposalBank(proposals, Any[1, 3])
    @test_throws ArgumentError ProposalBank(proposals, [1 + 0im, 3 + 0im])
    @test_throws ArgumentError ProposalBank(proposals, ["1", "3"])
    @test_throws MethodError ProposalBank(Tuple(proposals))
    @test_throws MethodError ProposalBank(proposals, (1, 3))
end

@testset "static-MIS scheme and algorithm configuration" begin
    proposals = [
        SphericalGaussian(-1.0, 1.0),
        SphericalGaussian(1.0, 1.0),
    ]
    bank = ProposalBank(proposals, [1, 3])

    stratified = @inferred StratifiedMixture()
    random = @inferred RandomMixture()
    standard = @inferred StandardMIS()
    groups = ((1, 2),)
    partial = @inferred PartialDeterministicMixture(groups)

    @test stratified isa AbstractMISScheme
    @test random isa AbstractMISScheme
    @test standard isa AbstractMISScheme
    @test partial isa AbstractMISScheme
    @test partial.groups === groups

    default_algorithm = @inferred ImportanceSampling(bank; nsamples=9)
    standard_algorithm = @inferred ImportanceSampling(
        bank;
        nsamples=9,
        mis_scheme=standard,
    )

    @test default_algorithm.proposal === bank
    @test default_algorithm.nsamples === 9
    @test default_algorithm.mis_scheme isa StratifiedMixture
    @test standard_algorithm.mis_scheme === standard
    @test_throws UndefKeywordError ImportanceSampling(bank)
    @test_throws ArgumentError ImportanceSampling(bank; nsamples=true)
    @test_throws ArgumentError ImportanceSampling(bank; nsamples=9.0)
    @test_throws MethodError ImportanceSampling(
        first(proposals);
        nsamples=9,
        mis_scheme=standard,
    )
end
