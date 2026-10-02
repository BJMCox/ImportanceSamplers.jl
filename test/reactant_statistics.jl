using ImportanceSamplers, Reactant, Statistics, Test

@testset "Reactant resident statistics" begin
    device = ImportanceSamplers.MLDataDevices.with_eltype(
        ImportanceSamplers.MLDataDevices.ReactantDevice(), nothing)
    for offset in (0f0, 1f20)
        shifted = device(WeightedSamples(Float32[1, 3], fill(offset, 2)))
        @test Array(normalized_weights(shifted)) == Float32[0.5, 0.5]
        @test Float32(mean(shifted)) == 2f0
    end
    points = Float32[1 2 4 8; 3 -1 2 0]
    logs = Float32[-Inf, 0, log(2), 0]
    result = device(WeightedSamples(points, logs))
    variance = Float32[4.75, 1.6875]
    for (f, expected) in zip((mean, var, std, cov),
        (Float32[4.5, 0.75], variance, sqrt.(variance), Float32[4.75 0.125; 0.125 1.6875]))
        value = f(result)
        @test value isa Reactant.AnyConcreteRArray && eltype(value) === Float32 &&
            Array(value) ≈ expected
    end
    @test Float32(mean(sum, result)) ≈ 5.25f0
    @test Float32(var(sum, result)) ≈ 6.6875f0
    @test Float32(std(sum, result)) ≈ sqrt(6.6875f0)

    named = device(WeightedSamples((a=points[1, :], b=points[2:2, :]), logs))
    named_variance = var(named)
    @test Float32(named_variance.a) ≈ variance[1]
    @test Array(named_variance.b) ≈ variance[2:2]
    # Equal shapes must not freeze either live weights or view indices.
    point_mass = device(WeightedSamples(points, Float32[0, -Inf, -Inf, -Inf]))
    @test Array(mean(point_mass)) == points[:, 1]
    @test Array(mean(result[1:3])) ≈ Float32[10/3, 1]
    @test Array(mean(result[2:4])) ≈ Float32[4.5, 0.75]

    unweighted = device(UnweightedSamples(points))
    @test Array(mean(unweighted)) ≈ vec(mean(points; dims=2))
    for corrected in (false, true)
        @test Array(var(unweighted; corrected)) ≈ vec(var(points; dims=2, corrected))
        @test Array(cov(unweighted; corrected)) ≈ cov(points; dims=2, corrected)
    end
    @test Array(std(unweighted)) ≈ vec(std(points; dims=2))
    totals = vec(sum(points; dims=1))
    @test Float32(mean(sum, unweighted)) ≈ mean(totals)
    @test Float32(std(sum, unweighted; corrected=false)) ≈ std(totals; corrected=false)
end
