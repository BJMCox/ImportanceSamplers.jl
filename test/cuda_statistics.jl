using Test
using CUDA
using ImportanceSamplers
using MLDataDevices
using Random
using Statistics

CUDA.allowscalar(false)

@testset "CUDA weighted statistics" begin
    shifted = WeightedSamples(CuArray(Float32[1, 3]), CuArray(fill(1f20, 2)))
    @test Array(normalized_weights(shifted)) == Float32[0.5, 0.5]
    @test (mean(shifted), var(shifted)) == (2f0, 1f0)

    logweights = CuArray(100f0 .+ log.(Float32[1, 2, 1]))
    samples = CuArray(Float32[1 3 10; 2 4 8])
    result = ImportanceSamplers._adopt_weighted_samples(samples, logweights)

    @test (@inferred mean(result)) isa CuArray
    @test (@inferred var(result)) isa CuArray
    @test (@inferred std(result)) isa CuArray
    @test (@inferred cov(result)) isa CuArray
    @test (@inferred mean(x -> x[1] + 2x[2], result)) ≈ 13.25f0
    @test (@inferred var(x -> x[1] + 2x[2], result)) ≈ 60.1875f0
    @test_throws ArgumentError quantile(result, 0.5)
end

@testset "CUDA weighted variance range" begin
    samples = Float32[0, 1f20]
    for logmass in (log(1f-20), -Inf32)
        mass = exp(BigFloat(logmass))
        expected = Float32(mass * BigFloat(samples[2])^2 / (1 + mass)^2)
        logweights = CuArray(Float32[0, logmass])
        scalar = WeightedSamples(CuArray(samples), logweights)
        matrix = WeightedSamples(CuArray(reshape(samples, 1, :)), logweights)
        @test var(scalar) ≈ expected rtol=8eps(Float32)
        @test only(Array(var(matrix))) ≈ expected rtol=8eps(Float32)
    end
end

@testset "CUDA task-local RNG transfer" begin
    device = MLDataDevices.CUDADevice{typeof(CUDA.device()),Nothing}(CUDA.device())
    fetch(@async begin
        Random.seed!(0x9114)
        task_rng = Random.default_rng()
        reference_rng = copy(task_rng)
        expected_rng = copy(task_rng)
        rand(expected_rng, UInt64)
        algorithm = ImportanceSampling(SphericalGaussian(0f0, 1f0); nsamples=8)
        target(x) = -abs2(x) / 2f0
        source = prepare_sampler(task_rng, target, algorithm)
        destination = device(source)
        reference = device(prepare_sampler(reference_rng, target, algorithm))

        @test Array(importance_sample!(destination).samples) ==
              Array(importance_sample!(reference).samples)
        @test rand(task_rng, UInt64) == rand(expected_rng, UInt64)
    end)
end
