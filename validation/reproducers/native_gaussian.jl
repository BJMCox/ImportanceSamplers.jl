using DensityInterface
using ImportanceSamplers
using Random

function empirical_moments(rng, proposal, nsamples)
    first_draw = rand(rng, proposal)
    dimension = length(first_draw)
    coordinate_sums = zeros(eltype(first_draw), dimension)
    product_sums = zeros(eltype(first_draw), dimension, dimension)
    for sample_index in 1:nsamples
        draw = isone(sample_index) ? first_draw : rand(rng, proposal)
        for row in 1:dimension
            coordinate_sums[row] += draw[row]
            for column in 1:dimension
                product_sums[row, column] += draw[row] * draw[column]
            end
        end
    end
    mean = coordinate_sums / nsamples
    covariance = (product_sums - nsamples * (mean * transpose(mean))) / (nsamples - 1)
    return mean, covariance
end

function validate_native_gaussians()
    scalar32 = SphericalGaussian(1.0f0, 2.0f0)
    spherical = SphericalGaussian(Float64[1, -2], 1.5)
    diagonal = DiagonalGaussian(Float64[1, -2], Float64[0.5, 2])
    factor = FactorGaussian(Float64[1, -2], Float64[1.25 0; 0.4 0.8])

    scalar_x = -0.5f0
    scalar_expected = -0.5f0 * abs2((scalar_x - 1.0f0) / 2.0f0) -
                      log(2.0f0) - 0.5f0 * log(2.0f0 * Float32(pi))
    @assert DensityInterface.logdensityof(scalar32, scalar_x) ≈ scalar_expected

    factor_x = Float64[2.25, -0.8]
    # x - location = [1.25, 1.2], so the lower-factor solve is exactly [1, 1].
    factor_expected = -log(2pi) - log(1.25 * 0.8) - 1.0
    @assert DensityInterface.logdensityof(factor, factor_x) ≈ factor_expected

    nsamples = 30_000
    spherical_mean, spherical_covariance = empirical_moments(
        Xoshiro(101), spherical, nsamples
    )
    diagonal_mean, diagonal_covariance = empirical_moments(
        Xoshiro(202), diagonal, nsamples
    )
    factor_mean, factor_covariance = empirical_moments(Xoshiro(303), factor, nsamples)
    @assert isapprox(spherical_mean, Float64[1, -2]; atol=0.04)
    @assert isapprox(
        spherical_covariance,
        Float64[2.25 0; 0 2.25];
        atol=0.07,
    )
    @assert isapprox(diagonal_mean, Float64[1, -2]; atol=0.04)
    @assert isapprox(
        diagonal_covariance,
        Float64[0.25 0; 0 4];
        atol=0.07,
    )
    @assert isapprox(factor_mean, Float64[1, -2]; atol=0.04)
    @assert isapprox(
        factor_covariance,
        Float64[1.5625 0.5; 0.5 0.8];
        atol=0.07,
    )

    for (index, proposal) in enumerate((scalar32, spherical, diagonal, factor))
        target = let proposal = proposal
            sample -> DensityInterface.logdensityof(proposal, sample)
        end
        result = importance_sample(
            Xoshiro(index),
            target,
            ImportanceSampling(proposal; nsamples=64);
            threaded=false,
        )
        tolerance = 8eps(eltype(result.logweights))
        @assert all(logweight -> abs(logweight) <= tolerance, result.logweights)
        @assert abs(lognormalizer(result)) <= tolerance
    end

    normalization_factor = FactorGaussian(
        Float64[0, 0],
        Float64[2 0; 0.5 1.5],
    )
    unnormalized_standard_kernel = function (sample)
        solved_first = sample[1] / 2
        solved_second = (sample[2] - 0.5 * solved_first) / 1.5
        return -0.5 * (abs2(solved_first) + abs2(solved_second))
    end
    normalizer_result = importance_sample(
        Xoshiro(99),
        unnormalized_standard_kernel,
        ImportanceSampling(normalization_factor; nsamples=64);
        threaded=false,
    )
    expected_log_normalizer = log(2pi) + log(3.0)
    @assert all(
        logweight -> isapprox(logweight, expected_log_normalizer; atol=8eps()),
        normalizer_result.logweights,
    )
    @assert isapprox(
        lognormalizer(normalizer_result),
        expected_log_normalizer;
        atol=8eps(),
    )

    rng = Xoshiro(404)
    scalar_draw = rand(rng, scalar32)
    spherical_draw = rand(rng, spherical)
    diagonal_draw = rand(rng, diagonal)
    factor_draw = rand(rng, factor)
    @assert only(Base.return_types(SphericalGaussian, Tuple{Float32,Float32})) ===
            typeof(scalar32)
    @assert only(Base.return_types(rand, Tuple{typeof(rng),typeof(scalar32)})) === Float32
    @assert only(
        Base.return_types(
            DensityInterface.logdensityof,
            Tuple{typeof(scalar32),Float32},
        ),
    ) === Float32
    @assert only(Base.return_types(rand, Tuple{typeof(rng),typeof(factor)})) ===
            Vector{Float64}
    @assert only(
        Base.return_types(
            DensityInterface.logdensityof,
            Tuple{typeof(factor),Vector{Float64}},
        ),
    ) === Float64

    rand(rng, scalar32)
    rand(rng, spherical)
    rand(rng, diagonal)
    rand(rng, factor)
    DensityInterface.logdensityof(scalar32, scalar_draw)
    DensityInterface.logdensityof(spherical, spherical_draw)
    DensityInterface.logdensityof(diagonal, diagonal_draw)
    DensityInterface.logdensityof(factor, factor_draw)
    allocations = (
        scalar_constructor=Base.@allocated(SphericalGaussian(1.0f0, 2.0f0)),
        spherical_constructor=Base.@allocated(
            SphericalGaussian(Float64[1, -2], 1.5)
        ),
        diagonal_constructor=Base.@allocated(
            DiagonalGaussian(Float64[1, -2], Float64[0.5, 2])
        ),
        factor_constructor=Base.@allocated(
            FactorGaussian(Float64[1, -2], Float64[1.25 0; 0.4 0.8])
        ),
        scalar_draw=Base.@allocated(rand(rng, scalar32)),
        spherical_draw=Base.@allocated(rand(rng, spherical)),
        diagonal_draw=Base.@allocated(rand(rng, diagonal)),
        factor_draw=Base.@allocated(rand(rng, factor)),
        scalar_logdensity=Base.@allocated(
            DensityInterface.logdensityof(scalar32, scalar_draw)
        ),
        spherical_logdensity=Base.@allocated(
            DensityInterface.logdensityof(spherical, spherical_draw)
        ),
        diagonal_logdensity=Base.@allocated(
            DensityInterface.logdensityof(diagonal, diagonal_draw)
        ),
        factor_logdensity=Base.@allocated(
            DensityInterface.logdensityof(factor, factor_draw)
        ),
    )

    return (
        formulas=:passed,
        moments=:passed,
        plain_is=:passed,
        inference=:passed,
        allocations=allocations,
    )
end

validate_native_gaussians()
