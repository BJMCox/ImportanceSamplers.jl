using Test
using ImportanceSamplers
using LinearAlgebra: I
import Random

const AMISIS = ImportanceSamplers

struct AMISTarget{T} end

function (::AMISTarget{T})(sample)::T where {T}
    radius = sample isa Number ? abs2(sample) : sum(abs2, sample)
    return -T(0.5) * radius
end

function assert_concrete_amis_fields(value)
    @test isconcretetype(typeof(value))
    @test all(isconcretetype, fieldtypes(typeof(value)))
    return nothing
end

function assert_amis_constructor(proposal)
    input_schedule = [4, 5, 6]
    algorithm = @inferred AMIS(
        proposal;
        rounds=3,
        round_size=input_schedule,
    )

    @test algorithm.proposal == proposal
    @test algorithm.proposal === proposal
    @test algorithm.rounds == 3
    @test algorithm.round_size == [4, 5, 6]
    @test algorithm.round_size !== input_schedule
    @test AMISIS._algorithm_proposal(algorithm) === algorithm.proposal
    @test AMISIS._algorithm_sample_budget(algorithm) == 15
    @test algorithm isa AbstractImportanceSampler
    @test fieldnames(typeof(algorithm)) == (:proposal, :rounds, :round_size)

    input_schedule[1] = 100
    @test algorithm.round_size == [4, 5, 6]
    return algorithm
end

@testset "AMIS constructor and proposal validation" begin
    proposal = FactorGaussian(zeros(2), Matrix{Float64}(I, 2, 2))
    assert_amis_constructor(proposal)

    for T in (Float32, Float64)
        assert_amis_constructor(SphericalGaussian(T(0), T(1)))
        assert_amis_constructor(SphericalGaussian(T[0, 1], T(2)))
        assert_amis_constructor(DiagonalGaussian(T[0, 1], T[2, 3]))
        assert_amis_constructor(
            FactorGaussian(T[0, 1], T[2 0; -1 3]),
        )
    end

    @test_throws UndefKeywordError AMIS(proposal)
    @test_throws ArgumentError AMIS(proposal; rounds=0, round_size=1)
    @test_throws DimensionMismatch AMIS(proposal; rounds=2, round_size=[1])
    @test_throws ArgumentError AMIS(
        TransformedProposal(
            SphericalGaussian(0.0, 1.0),
            IdentityTransform(),
        );
        rounds=1,
        round_size=1,
    )
    @test_throws ArgumentError AMIS(TestScalarProposal(0.0); rounds=1, round_size=1)

    malformed = (
        AMISIS._GaussianProposal(
            AMISIS.GaussianFamily(),
            0,
            AMISIS._SphericalGaussianScale(1),
            0.0,
        ),
        AMISIS._GaussianProposal(
            AMISIS.GaussianFamily(),
            Float64[],
            AMISIS._SphericalGaussianScale(1.0),
            0.0,
        ),
        AMISIS._GaussianProposal(
            AMISIS.GaussianFamily(),
            Float32[0],
            AMISIS._SphericalGaussianScale(1.0),
            0.0f0,
        ),
    )
    for proposal in malformed
        @test_throws ArgumentError AMIS(proposal; rounds=1, round_size=1)
    end
end

function assert_factor_amis_storage(proposal, expected_factor, ::Type{L}) where {L}
    rng = Random.Xoshiro(0x6100)
    expected_rng = copy(rng)
    sampler = @inferred prepare_sampler(
        rng,
        AMISTarget{L}(),
        AMIS(proposal; rounds=3, round_size=[4, 5, 6]);
        threaded=false,
    )
    state = sampler.method_state
    history = state.history
    workspace = state.workspace
    T = eltype(proposal.location)
    d = length(proposal.location)

    @test state isa AMISIS._PreparedAMIS
    @test history isa AMISIS._AMISFactorHistory
    @test state.schedule == [4, 5, 6]
    @test state.offsets == [1, 5, 10, 16]
    @test state.logcounts ≈ log.(T[4, 5, 6])
    @test size(history.means) == (d, 3)
    @test size(history.factors) == (d, d, 3)
    @test size(history.lognormalizers) == (3,)
    @test history.means[:, 1] == proposal.location
    @test history.factors[:, :, 1] == expected_factor
    @test history.lognormalizers[1] == proposal.lognormalizer
    @test history.means !== proposal.location
    if proposal.scale isa AMISIS._DiagonalGaussianScale
        @test history.factors !== proposal.scale.scales
    elseif proposal.scale isa AMISIS._FactorGaussianScale
        @test history.factors !== proposal.scale.factor
    end
    @test size(workspace.samples) == (d, 15)
    @test length(workspace.logtargets) == 15
    @test length(workspace.lognumerators) == 15
    @test length(workspace.logweights) == 15
    @test length(workspace.round_ids) == 15
    @test length(workspace.normalized_weights) == 15
    @test size(workspace.centered_scaled) == (d, 15)
    @test size(workspace.covariance) == (d, d)
    @test eltype(history.means) === T
    @test eltype(history.factors) === T
    @test eltype(workspace.logtargets) === L
    @test eltype(workspace.normalized_weights) === T
    @test length(sampler.random_buffers.normal) == d * 6

    for value in (state, history, workspace, sampler.random_buffers)
        assert_concrete_amis_fields(value)
    end
    @test rand(rng, UInt64) == rand(expected_rng, UInt64)
    return sampler
end

@testset "AMIS fixed factor storage" begin
    for T in (Float32, Float64)
        spherical = SphericalGaussian(T[1, -2], T(3))
        diagonal = DiagonalGaussian(T[1, -2], T[2, 4])
        factor = FactorGaussian(T[1, -2], T[2 0; -1 4])

        assert_factor_amis_storage(
            spherical,
            T[3 0; 0 3],
            T,
        )
        assert_factor_amis_storage(
            diagonal,
            T[2 0; 0 4],
            Float64,
        )
        assert_factor_amis_storage(
            factor,
            T[2 0; -1 4],
            T,
        )
    end
end

@testset "AMIS fixed scalar storage" begin
    for T in (Float32, Float64)
        proposal = SphericalGaussian(T(1), T(2))
        rng = Random.Xoshiro(0x6101)
        expected_rng = copy(rng)
        sampler = @inferred prepare_sampler(
            rng,
            AMISTarget{T}(),
            AMIS(proposal; rounds=3, round_size=[4, 5, 6]);
            threaded=false,
        )
        state = sampler.method_state
        history = state.history
        workspace = state.workspace

        @test history isa AMISIS._AMISScalarHistory
        @test history.means isa Vector{T}
        @test history.scales isa Vector{T}
        @test history.lognormalizers isa Vector{T}
        @test (history.means[1], history.scales[1]) == (T(1), T(2))
        @test history.lognormalizers[1] == proposal.lognormalizer
        @test size(workspace.samples) == (15,)
        @test size(workspace.centered_scaled) == (15,)
        @test size(workspace.covariance) == (1,)
        @test !hasproperty(history, :factors)
        @test sampler.random_buffers.normal isa Vector{T}
        @test length(sampler.random_buffers.normal) == 6
        for value in (state, history, workspace, sampler.random_buffers)
            assert_concrete_amis_fields(value)
        end
        @test rand(rng, UInt64) == rand(expected_rng, UInt64)
    end
end
