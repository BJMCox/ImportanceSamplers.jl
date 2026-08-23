import DensityInterface
import LinearAlgebra
import Random
import Random: rand

const IS = ImportanceSamplers

@testset "native Gaussian constructor validation" begin
    @testset "locations" begin
        for location in (
            0,
            big"0.0",
            Float32[],
            Float64[],
            [0, 1],
            Real[0.0, 1.0],
            Float32[0, NaN32],
            Float64[0, Inf],
            NaN32,
            -Inf,
        )
            @test_throws ArgumentError SphericalGaussian(location, 1.0f0)
        end
    end

    @testset "spherical scales" begin
        for scale in (0, big"1.0", 0.0f0, -1.0f0, Inf32, NaN32)
            @test_throws ArgumentError SphericalGaussian(0.0f0, scale)
        end
        @test_throws ArgumentError SphericalGaussian(0.0f0, 1.0)
        @test_throws ArgumentError SphericalGaussian(zeros(Float32, 2), 1.0)
    end

    @testset "diagonal scales" begin
        @test_throws ArgumentError DiagonalGaussian(Float32[], Float32[])
        @test_throws ArgumentError DiagonalGaussian(zeros(Float32, 2), Float32[1])
        @test_throws ArgumentError DiagonalGaussian(zeros(Float32, 2), Float32[1, 2, 3])
        @test_throws ArgumentError DiagonalGaussian(zeros(Float32, 2), [1, 2])
        @test_throws ArgumentError DiagonalGaussian(zeros(Float32, 2), ones(Float64, 2))
        for scales in (
            Float32[1, 0],
            Float32[1, -1],
            Float32[1, Inf],
            Float32[1, NaN],
        )
            @test_throws ArgumentError DiagonalGaussian(zeros(Float32, 2), scales)
        end
    end

    @testset "dense factors" begin
        location = zeros(Float32, 2)
        @test_throws ArgumentError FactorGaussian(Float32[], zeros(Float32, 0, 0))
        @test_throws ArgumentError FactorGaussian(location, ones(Float32, 2, 1))
        @test_throws ArgumentError FactorGaussian(location, ones(Float32, 3, 3))
        upper_entry = Matrix{Float32}(LinearAlgebra.I, 2, 2) .+
                      Float32[0 1; 0 0]
        @test_throws ArgumentError FactorGaussian(location, upper_entry)
        @test_throws ArgumentError FactorGaussian(location, Matrix{Int}(LinearAlgebra.I, 2, 2))
        @test_throws ArgumentError FactorGaussian(location, Matrix{Float64}(LinearAlgebra.I, 2, 2))
        for factor in (
            Float32[0 0; 0 1],
            Float32[-1 0; 0 1],
            Float32[1 0; 0 Inf],
            Float32[1 0; 0 NaN],
            Float32[1 0; Inf 1],
            Float32[1 0; NaN 1],
        )
            @test_throws ArgumentError FactorGaussian(location, factor)
        end
    end
end

@testset "native Gaussian public families and storage" begin
    @test AbstractRadialProposalFamily <: AbstractProposalFamily
    @test IS.GaussianFamily <: AbstractRadialProposalFamily
    @test :GaussianFamily ∉ names(ImportanceSamplers)
    for internal_name in (:_SphericalGaussianScale, :_DiagonalGaussianScale, :_FactorGaussianScale)
        @test internal_name ∉ names(ImportanceSamplers)
    end

    location_source = Float32[1, -2, 3]
    scale_source = Float32[0.5, 1.5, 2.5]
    diagonal = @inferred DiagonalGaussian(view(location_source, :), view(scale_source, :))
    fill!(location_source, 100)
    fill!(scale_source, 100)
    @test IS._proposal_dimension(diagonal) == 3
    @test DensityInterface.logdensityof(diagonal, Float32[1, -2, 3]) ≈
          -Float32(1.5 * log(2pi)) - log(0.5f0 * 1.5f0 * 2.5f0)

    raw_factor = Float64[2 0; 0.5 1.5]
    factor = @inferred FactorGaussian(Float64[0, 0], raw_factor)
    raw_factor .= 100
    @test DensityInterface.logdensityof(factor, Float64[2, 2]) ≈
          -log(2pi) - log(3.0) - 1.0

    covariance = Float64[4 1; 1 2]
    chol_factor = @inferred FactorGaussian(
        Float64[0, 0],
        LinearAlgebra.cholesky(LinearAlgebra.Symmetric(covariance)),
    )
    lower_factor = @inferred FactorGaussian(
        Float64[0, 0],
        LinearAlgebra.LowerTriangular(Float64[2 0; 0.5 1.5]),
    )
    @test IS._proposal_dimension(chol_factor) == 2
    @test IS._proposal_dimension(lower_factor) == 2
end

@testset "native Gaussian analytic densities" begin
    scalar = SphericalGaussian(1.25, 2.5)
    scalar_sample = -0.75
    scalar_expected = -0.5 * abs2((scalar_sample - 1.25) / 2.5) -
                      log(2.5) - 0.5 * log(2pi)
    @test DensityInterface.logdensityof(scalar, scalar_sample) ≈ scalar_expected

    spherical = SphericalGaussian(Float64[1, -1, 2], 2.0)
    spherical_expected = -0.5 * (1 + 4 + 9) - 3log(2.0) - 1.5log(2pi)
    @test DensityInterface.logdensityof(spherical, Float64[3, -5, 8]) ≈
          spherical_expected

    diagonal = DiagonalGaussian(Float64[1, -1, 2], Float64[2, 4, 0.5])
    diagonal_expected = -0.5 * (1 + 1 + 4) - log(2 * 4 * 0.5) - 1.5log(2pi)
    @test DensityInterface.logdensityof(diagonal, Float64[3, 3, 3]) ≈
          diagonal_expected

    factor = FactorGaussian(Float64[0, 0], Float64[2 0; 0.5 1.5])
    @test DensityInterface.logdensityof(factor, Float64[2, 2]) ≈
          -log(2pi) - log(3.0) - 1.0

    @test_throws DimensionMismatch DensityInterface.logdensityof(spherical, zeros(2))
    @test_throws DimensionMismatch DensityInterface.logdensityof(diagonal, zeros(2))
    @test_throws DimensionMismatch DensityInterface.logdensityof(factor, zeros(3))
end

@testset "native Gaussian subnormal scales" begin
    for T in (Float32, Float64)
        scale = nextfloat(zero(T))
        logtwopi = log(T(2) * T(pi))

        scalar = SphericalGaussian(zero(T), scale)
        scalar_peak = -log(scale) - T(0.5) * logtwopi
        scalar_peak_observed = DensityInterface.logdensityof(scalar, zero(T))
        @test isfinite(scalar_peak_observed)
        @test scalar_peak_observed ≈ scalar_peak rtol = 4eps(T)
        @test DensityInterface.logdensityof(scalar, scale) ≈
              scalar_peak - T(0.5) rtol = 4eps(T)

        spherical = SphericalGaussian(zeros(T, 2), scale)
        vector_peak = -T(2) * log(scale) - logtwopi
        spherical_peak_observed = DensityInterface.logdensityof(
            spherical,
            zeros(T, 2),
        )
        @test isfinite(spherical_peak_observed)
        @test spherical_peak_observed ≈ vector_peak rtol = 4eps(T)
        @test DensityInterface.logdensityof(spherical, T[scale, 0]) ≈
              vector_peak - T(0.5) rtol = 4eps(T)

        diagonal = DiagonalGaussian(zeros(T, 2), fill(scale, 2))
        diagonal_peak_observed = DensityInterface.logdensityof(
            diagonal,
            zeros(T, 2),
        )
        @test isfinite(diagonal_peak_observed)
        @test diagonal_peak_observed ≈ vector_peak rtol = 4eps(T)
        @test DensityInterface.logdensityof(diagonal, T[scale, 0]) ≈
              vector_peak - T(0.5) rtol = 4eps(T)
    end
end

@testset "native Gaussian precision, inference, and draw shapes" begin
    for T in (Float32, Float64)
        scalar = @inferred SphericalGaussian(T(1), T(2))
        spherical = @inferred SphericalGaussian(T[1, 2], T(2))
        diagonal = @inferred DiagonalGaussian(T[1, 2], T[0.5, 2])
        factor = @inferred FactorGaussian(T[1, 2], T[1 0; 0.25 2])

        scalar_draw = @inferred rand(Random.Xoshiro(11), scalar)
        @test scalar_draw isa T
        @test @inferred(DensityInterface.logdensityof(scalar, scalar_draw)) isa T

        for proposal in (spherical, diagonal, factor)
            draw = @inferred rand(Random.Xoshiro(12), proposal)
            @test draw isa Vector{T}
            @test length(draw) == 2
            @test @inferred(DensityInterface.logdensityof(proposal, draw)) isa T
        end
    end
end

function _draw_matrix(rng, proposal, nsamples)
    first_draw = rand(rng, proposal)
    draws = Matrix{eltype(first_draw)}(undef, length(first_draw), nsamples)
    draws[:, 1] = first_draw
    for sample_index in 2:nsamples
        draws[:, sample_index] = rand(rng, proposal)
    end
    return draws
end

function _empirical_mean_covariance(draws)
    nsamples = size(draws, 2)
    mean = vec(sum(draws; dims=2)) / nsamples
    centered = draws .- mean
    covariance = centered * transpose(centered) / (nsamples - 1)
    return mean, covariance
end

@testset "native Gaussian seeded moments" begin
    nsamples = 30_000

    scalar = SphericalGaussian(1.5, 2.0)
    scalar_rng = Random.Xoshiro(10_000)
    scalar_draws = [rand(scalar_rng, scalar) for _ in 1:nsamples]
    @test sum(scalar_draws) / nsamples ≈ 1.5 atol = 0.04
    scalar_mean = sum(scalar_draws) / nsamples
    scalar_variance = sum(x -> abs2(x - scalar_mean), scalar_draws) / (nsamples - 1)
    @test scalar_variance ≈ 4.0 atol = 0.10

    cases = (
        (
            SphericalGaussian(Float64[1, -2], 1.5),
            Float64[1, -2],
            Float64[2.25 0; 0 2.25],
            101,
        ),
        (
            DiagonalGaussian(Float64[1, -2], Float64[0.5, 2]),
            Float64[1, -2],
            Float64[0.25 0; 0 4],
            202,
        ),
        (
            FactorGaussian(Float64[1, -2], Float64[1.25 0; 0.4 0.8]),
            Float64[1, -2],
            Float64[1.5625 0.5; 0.5 0.8],
            303,
        ),
    )
    for (proposal, expected_mean, expected_covariance, seed) in cases
        draws = _draw_matrix(Random.Xoshiro(seed), proposal, nsamples)
        observed_mean, observed_covariance = _empirical_mean_covariance(draws)
        @test observed_mean ≈ expected_mean atol = 0.04
        @test observed_covariance ≈ expected_covariance atol = 0.07
    end
end

@testset "native Gaussian plain IS normalizer identity" begin
    proposals = (
        SphericalGaussian(1.0, 2.0),
        SphericalGaussian(Float64[1, -2], 1.5),
        DiagonalGaussian(Float64[1, -2], Float64[0.5, 2]),
        FactorGaussian(Float64[1, -2], Float64[1.25 0; 0.4 0.8]),
    )
    for (index, proposal) in enumerate(proposals)
        target = let proposal = proposal
            sample -> DensityInterface.logdensityof(proposal, sample)
        end
        result = importance_sample(
            Random.Xoshiro(index),
            target,
            ImportanceSampling(proposal; nsamples=32);
            threaded=false,
        )
        @test maximum(abs, result.logweights) <= 64eps()
        @test abs(lognormalizer(result)) <= 64eps()
    end

    factor = FactorGaussian(Float64[0, 0], Float64[2 0; 0.5 1.5])
    unnormalized_standard_kernel = function (sample)
        solved_first = sample[1] / 2
        solved_second = (sample[2] - 0.5 * solved_first) / 1.5
        return -0.5 * (abs2(solved_first) + abs2(solved_second))
    end
    normalizer_result = importance_sample(
        Random.Xoshiro(99),
        unnormalized_standard_kernel,
        ImportanceSampling(factor; nsamples=32);
        threaded=false,
    )
    expected_log_normalizer = log(2pi) + log(3.0)
    @test all(
        logweight -> isapprox(logweight, expected_log_normalizer; atol=8eps()),
        normalizer_result.logweights,
    )
    @test lognormalizer(normalizer_result) ≈ expected_log_normalizer atol = 8eps()
end
