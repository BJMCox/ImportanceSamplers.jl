using Test
using CUDA
using ImportanceSamplers
using MLDataDevices
using Random
using Statistics

CUDA.allowscalar(false)

@testset "CUDA multinomial resampling" begin
    samples = CuArray(Float32[1 2 3; 4 5 6])
    logweights = CuArray(Float32[-Inf, 0, -Inf])
    result = ImportanceSamplers._adopt_weighted_samples(samples, logweights)

    draws = @inferred resample(Random.Xoshiro(21), result, 5)
    @test draws.samples isa CuArray
    @test Array(draws.samples) == Float32[2 2 2 2 2; 5 5 5 5 5]
    @test (@inferred mean(draws)) isa CuArray
    @test Array(mean(draws)) == Float32[2, 5]
    @test Array(var(draws)) == zeros(Float32, 2)
    @test Array(std(draws)) == zeros(Float32, 2)
    @test Array(cov(draws)) == zeros(Float32, 2, 2)
    @test (@inferred mean(x -> x[1] + x[2], draws)) == 7.0f0
    @test_throws ArgumentError median(draws)
    @test_throws ArgumentError draws[1]

    cpu_draws = @inferred MLDataDevices.cpu_device()(draws)
    @test cpu_draws[1] == Float32[2, 5]
end

@testset "CUDA structured resampling" begin
    samples = (
        location=CuArray(Float32[1, 2, 3]),
        state=(position=CuArray(Float32[1 2 3; 4 5 6]),),
    )
    result = ImportanceSamplers._adopt_weighted_samples(
        samples,
        CuArray(Float32[-Inf, 0, -Inf]),
    )

    draws = resample(Random.Xoshiro(22), result, 2)
    @test Array(draws.samples.location) == Float32[2, 2]
    @test Array(draws.samples.state.position) == Float32[2 2; 5 5]
    @test mean(draws).location == 2.0f0
end
