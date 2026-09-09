module CUDAMixedPrecisionValidation
using CUDA, ImportanceSamplers, LinearAlgebra, MLDataDevices, Random, Test

wide_target(x) = 1e8 + 0.1

function main()
    CUDA.allowscalar(false)
    physical = CUDA.device()
    device = CUDADevice{typeof(physical),Nothing}(physical)
    proposal = FactorGaussian(zeros(Float32,2), Matrix{Float32}(I,2,2))
    bank = ProposalBank([proposal, proposal])
    @testset "Explicit mixed-precision factor batching" begin
        sampler = device(prepare_sampler(Xoshiro(41), wide_target,
            ImportanceSampling(bank; nsamples=32);
            factor_execution=BatchedFactorExecution()))
        result = cpu_device()(importance_sample!(sampler))
        expected = 1e8 .+ 0.1 .+ log(2pi) .+
            vec(sum(abs2, Float64.(result.samples); dims=1)) ./ 2
        @test result.logweights ≈ expected atol=1e-5 rtol=0
        @test result.diagnostics.factor_execution_policy === :batched
        default = device(prepare_sampler(Xoshiro(41), wide_target,
            ImportanceSampling(bank; nsamples=32)))
        @test importance_sample!(default).diagnostics.factor_execution_policy === :fused
    end
end
end
