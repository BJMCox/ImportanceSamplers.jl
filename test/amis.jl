using Test
using ImportanceSamplers
using LinearAlgebra: I
import LinearAlgebra
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
    @test !hasproperty(workspace, :round_ids)
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

@testset "AMIS positional construction remains private" begin
    @test_throws MethodError AMIS(TestScalarProposal(0.0), 0, [0])
end

function literal_amis_moments(samples, logweights, previous_covariance)
    T = eltype(samples)
    shifted = exp.(T.(logweights .- maximum(logweights)))
    weights = shifted ./ sum(shifted)
    if samples isa AbstractVector
        mean = sum(weights[index] * samples[index] for index in eachindex(weights))
        variance = sum(
            weights[index] * abs2(samples[index] - mean) for
            index in eachindex(weights)
        )
        ridge = sqrt(eps(T)) * previous_covariance
        return mean, variance + ridge, weights
    end

    dimension = size(samples, 1)
    mean = zeros(T, dimension)
    for sample_index in axes(samples, 2), coordinate in axes(samples, 1)
        mean[coordinate] +=
            weights[sample_index] * samples[coordinate, sample_index]
    end
    covariance = zeros(T, dimension, dimension)
    for sample_index in axes(samples, 2), column in 1:dimension, row in 1:dimension
        covariance[row, column] += weights[sample_index] *
                                   (samples[row, sample_index] - mean[row]) *
                                   (samples[column, sample_index] - mean[column])
    end
    ridge = sqrt(eps(T)) * LinearAlgebra.tr(previous_covariance) / T(dimension)
    covariance += ridge * I
    return mean, covariance, weights
end

function amis_factor_fit_allocated!(workspace, history, sample_count)
    return @allocated AMISIS._fit_amis_proposal!(
        workspace,
        history,
        1,
        sample_count,
    )
end

function amis_factor_candidate_valid_allocated(mean, factor, lognormalizer)
    return @allocated AMISIS._amis_factor_candidate_valid(
        mean,
        factor,
        lognormalizer,
    )
end

@testset "AMIS factor candidate publication contract" begin
    for T in (Float32, Float64)
        mean = T[1, -2]
        factor = T[2 0; -1 3]
        lognormalizer = T(-1)

        @test AMISIS._amis_factor_candidate_valid(mean, factor, lognormalizer)
        @test amis_factor_candidate_valid_allocated(mean, factor, lognormalizer) == 0

        invalid_mean = copy(mean)
        invalid_mean[1] = T(Inf)
        @test !AMISIS._amis_factor_candidate_valid(
            invalid_mean,
            factor,
            lognormalizer,
        )

        invalid_lower = copy(factor)
        invalid_lower[2, 1] = T(NaN)
        @test !AMISIS._amis_factor_candidate_valid(
            mean,
            invalid_lower,
            lognormalizer,
        )

        for diagonal in (zero(T), -one(T), T(Inf))
            invalid_diagonal = copy(factor)
            invalid_diagonal[1, 1] = diagonal
            @test !AMISIS._amis_factor_candidate_valid(
                mean,
                invalid_diagonal,
                lognormalizer,
            )
        end

        @test !AMISIS._amis_factor_candidate_valid(mean, factor, T(Inf))
    end
end

@testset "AMIS fitting reproduces literal two-pass weighted moments" begin
    for T in (Float32, Float64)
        scalar_proposal = SphericalGaussian(T(-0.5), T(1.75))
        scalar_state = AMISIS._prepare_method_state(
            AMIS(scalar_proposal; rounds=1, round_size=4),
        )
        scalar_samples = T[-3, -0.25, 1.5, 4]
        scalar_logs = T[-1000, -3, -1, -2]
        copyto!(scalar_state.workspace.samples, scalar_samples)
        copyto!(scalar_state.workspace.logweights, scalar_logs)
        scalar_mean, scalar_variance, scalar_weights = literal_amis_moments(
            scalar_samples,
            scalar_logs,
            abs2(scalar_proposal.scale.scale),
        )

        scalar_fitted = @inferred AMISIS._fit_amis_proposal!(
            scalar_state.workspace,
            scalar_state.history,
            1,
            4,
        )

        @test scalar_fitted.location ≈ scalar_mean rtol = 8eps(T)
        @test abs2(scalar_fitted.scale.scale) ≈ scalar_variance rtol = 16eps(T)
        @test scalar_state.workspace.normalized_weights ≈ scalar_weights rtol = 8eps(T)

        vector_proposal = FactorGaussian(
            T[-1, 2],
            T[2 0; -0.5 1.25],
        )
        vector_state = AMISIS._prepare_method_state(
            AMIS(vector_proposal; rounds=1, round_size=4),
        )
        vector_samples = T[-2 0 3 5; 4 -1 2 7]
        vector_logs = T[-900, -2, -0.5, -4]
        copyto!(vector_state.workspace.samples, vector_samples)
        copyto!(vector_state.workspace.logweights, vector_logs)
        previous_covariance = vector_proposal.scale.factor *
                              vector_proposal.scale.factor'
        vector_mean, vector_covariance, vector_weights = literal_amis_moments(
            vector_samples,
            vector_logs,
            previous_covariance,
        )

        candidate_mean = vector_state.workspace.candidate_mean
        candidate_factor = vector_state.workspace.candidate_scale
        vector_fitted = @inferred AMISIS._fit_amis_proposal!(
            vector_state.workspace,
            vector_state.history,
            1,
            4,
        )

        @test vector_fitted === nothing
        @test candidate_mean ≈ vector_mean rtol = 16eps(T)
        @test candidate_factor * candidate_factor' ≈
              vector_covariance rtol = 32eps(T)
        @test vector_state.workspace.covariance ≈ vector_covariance rtol = 32eps(T)
        @test vector_state.workspace.normalized_weights ≈ vector_weights rtol = 8eps(T)
        @test amis_factor_fit_allocated!(
            vector_state.workspace,
            vector_state.history,
            4,
        ) == 0
    end
end
