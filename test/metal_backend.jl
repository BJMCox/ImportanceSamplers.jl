using ImportanceSamplers, Metal, Random, Test

@testset "Metal sampling, reuse and target failure" begin
    device = ImportanceSamplers.MLDataDevices.MetalDevice{Nothing}()
    proposal = SphericalGaussian(0f0, 1f0)
    algorithm = ImportanceSampling(proposal; nsamples=512)
    prepared = device(prepare_sampler(Xoshiro(2), x -> -x^2/2, algorithm))
    first_result = importance_sample!(prepared)
    first_samples = Array(first_result.samples)
    @test all(isapprox(log(Float32(2pi))/2; atol=2f-6), Array(first_result.logweights))
    second_result = importance_sample!(prepared)
    @test Array(first_result.samples) == first_samples
    @test Array(second_result.samples) != first_samples

    invalid = device(prepare_sampler(Xoshiro(2),
        x -> x > 0 ? Float32(NaN) : -x^2/2, algorithm))
    error = try
        importance_sample!(invalid)
        nothing
    catch caught
        caught
    end
    @test error isa SamplerExecutionError
    if error isa SamplerExecutionError
        @test error.phase == :target
        @test error.sample_index == findfirst(>(0), first_samples)
    end
end
