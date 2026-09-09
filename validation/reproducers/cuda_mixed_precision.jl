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
    @testset "Adaptive diagnostics report actual factor execution" begin
        adaptive_bank = ProposalBank([
            proposal,
            FactorGaussian(Float32[1,0], Matrix{Float32}(I,2,2)),
        ])
        apis = APIS(adaptive_bank; rounds=1, round_size=8)
        default_apis = device(prepare_sampler(Xoshiro(42), wide_target, apis))
        @test importance_sample!(default_apis).diagnostics.factor_execution_policy === :fused

        batched_apis = device(prepare_sampler(Xoshiro(42), wide_target, apis;
            factor_execution=BatchedFactorExecution()))
        @test importance_sample!(batched_apis).diagnostics.factor_execution_policy === :batched

        dm_pmc = DeterministicMixturePMC(adaptive_bank; rounds=1, round_size=8)
        default_dm_pmc = device(prepare_sampler(Xoshiro(43), wide_target, dm_pmc))
        @test importance_sample!(default_dm_pmc).diagnostics.factor_execution_policy === :fused

        lais = LAIS(adaptive_bank;
            transition=RandomWalkMetropolis(Matrix{Float32}(I,2,2)),
            rounds=1, round_size=8)
        default_lais = device(prepare_sampler(Xoshiro(44), wide_target, lais))
        @test importance_sample!(default_lais).diagnostics.factor_execution_policy === :fused

        amis = AMIS(SphericalGaussian(0.0f0, 1.0f0); rounds=1, round_size=8)
        default_amis = device(prepare_sampler(Xoshiro(45), wide_target, amis))
        @test importance_sample!(default_amis).diagnostics.factor_execution_policy === :fused
    end
end
end
