using Test
using CUDA
using ImportanceSamplers
using Statistics

CUDA.allowscalar(false)

@testset "CUDA weighted statistics" begin
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
