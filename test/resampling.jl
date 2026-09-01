using Test
using ImportanceSamplers
using MLDataDevices
using Random
using Statistics

@testset "public multinomial resampling" begin
    result = WeightedSamples([10.0, 20.0, 30.0], [-Inf, 0.0, -Inf])

    draws = resample(Random.Xoshiro(11), result, 5)
    @test draws isa UnweightedSamples
    @test length(draws) == 5
    @test draws.samples == fill(20.0, 5)
    @test draws[3] == 20.0
    @test collect(draws) == fill(20.0, 5)

    default_count = resample(
        Random.Xoshiro(12),
        result;
        method=MultinomialResampling(),
    )
    @test length(default_count) == length(result)

    structured = WeightedSamples(
        (location=[1.0, 2.0, 3.0], state=(position=[1.0 2.0 3.0; 4.0 5.0 6.0],)),
        [-Inf, 0.0, -Inf],
    )
    structured_draws = resample(Random.Xoshiro(13), structured, 2)
    @test structured_draws.samples.location == [2.0, 2.0]
    @test structured_draws.samples.state.position == [2.0 2.0; 5.0 5.0]
    @test structured_draws[1] == (location=2.0, state=(position=[2.0, 5.0],))

    @test_throws AllZeroWeightsError resample(
        Random.Xoshiro(14),
        WeightedSamples([1.0, 2.0], fill(-Inf, 2)),
    )
end

@testset "unweighted sample statistics and transfer" begin
    source = [1.0, 3.0, 10.0]
    samples = UnweightedSamples(source)
    source[1] = -1.0

    @test samples.samples == [1.0, 3.0, 10.0]
    @test (@inferred mean(samples)) ≈ 14 / 3
    @test (@inferred var(samples)) ≈ 67 / 3
    @test (@inferred std(samples)) ≈ sqrt(67 / 3)
    @test (@inferred mean(abs2, samples)) ≈ 110 / 3
    @test (@inferred quantile(samples, 0.25)) ≈ 2.0
    @test (@inferred median(samples)) ≈ 3.0

    matrix = UnweightedSamples([1.0 3.0 10.0; 2.0 4.0 8.0])
    @test mean(matrix) ≈ [14 / 3, 14 / 3]
    @test var(matrix) ≈ [67 / 3, 28 / 3]
    @test cov(matrix) ≈ [67 / 3 43 / 3; 43 / 3 28 / 3]
    @test median(matrix) ≈ [3.0, 4.0]

    structured = UnweightedSamples((location=[1.0, 3.0, 10.0],))
    @test mean(structured).location ≈ 14 / 3

    transferred = @inferred MLDataDevices.cpu_device()(samples)
    @test transferred.samples == samples.samples
    @test transferred.samples !== samples.samples
end
