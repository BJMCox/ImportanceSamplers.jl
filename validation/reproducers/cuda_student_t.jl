using CUDA
using ImportanceSamplers
using DensityInterface
using LinearAlgebra
using MLDataDevices
using Random
using Test

CUDA.functional() || error("CUDA is not functional")
CUDA.allowscalar(false)

physical = CUDA.device()
device = MLDataDevices.CUDADevice{typeof(physical),Nothing}(physical)

@inline function scalar_target(sample::T)::T where {T<:Union{Float32,Float64}}
    return -abs2(sample)
end

function student_gradient!(destination, sample)
    for index in eachindex(sample)
        destination[index] = -2f0 * sample[index]
    end
    return destination
end

@inline function vector_target(sample)::Float32
    total = 0.0f0
    for index in eachindex(sample)
        total += abs2(sample[index])
    end
    return -total
end

function run_case(proposal, target, seed)
    sampler = prepare_sampler(
        Random.Xoshiro(seed),
        target,
        ImportanceSampling(proposal; nsamples=65_536),
    )
    result = importance_sample!(device(sampler))
    samples = Array(result.samples)
    logweights = Array(result.logweights)
    @test result.samples isa CUDA.AnyCuArray
    @test all(isfinite, samples)
    @test all(isfinite, logweights)
    return samples
end

@testset "native Student-t CUDA" begin
    run_case(SphericalStudentT(0.5f0, 0.0f0, 1.0f0), scalar_target, 11)
    run_case(SphericalStudentT(6.0, 1.0, 2.0), scalar_target, 15)
    run_case(
        DiagonalStudentT(1.0f0, Float32[0, 0], Float32[1, 2]),
        vector_target,
        12,
    )
    samples = run_case(
        FactorStudentT(8.0f0, Float32[1, -2], Float32[1.25 0; 0.4 0.8]),
        vector_target,
        13,
    )
    @test vec(sum(samples; dims=2)) ./ size(samples, 2) ≈ Float32[1, -2] atol=0.04

    transformed = TransformedProposal(
        SphericalStudentT(4.0f0, 0.0f0, 1.0f0),
        PositiveTransform(),
    )
    @test all(>(0), run_case(transformed, scalar_target, 14))
end

@testset "shared Student-t bank and adaptive CUDA" begin
    bank = ProposalBank([
        FactorStudentT(5f0, Float32[-1, 0], Float32[1 0; 0.3 1]),
        FactorStudentT(7f0, Float32[1, 0], Float32[1 0; -0.2 1]),
    ])
    target = LogTarget(vector_target; grad=student_gradient!)
    algorithms = (
        ImportanceSampling(bank; nsamples=12_288),
        DeterministicMixturePMC(bank; rounds=3, round_size=4096),
        APIS(bank; rounds=3, round_size=4096),
        LAIS(bank; transition=RandomWalkMetropolis(0.5f0), rounds=3, round_size=4096),
        AMIS(first(bank.proposals); rounds=3, round_size=4096),
        NPMC(first(bank.proposals); rounds=3, round_size=4096),
        CAIS(bank; rounds=3, round_size=4096),
        FirstOrderGRAMIS(bank; rounds=3, round_size=4096, repulsion_strength=0.1f0),
    )
    for algorithm in algorithms
        prepared = device(prepare_sampler(Xoshiro(17), target, algorithm))
        result = importance_sample!(prepared)
        @test result.samples isa CUDA.AnyCuArray
        @test all(isfinite, result.logweights)
        learned = algorithm isa ImportanceSampling ? bank : current_proposal(cpu_device(), prepared)
        proposals = learned isa ProposalBank ? learned.proposals : (learned,)
        @test all(enumerate(proposals)) do (index, proposal)
            factor = proposal.scale.factor
            offset = Float32[1, -1]
            dof = index == 1 ? 5f0 : 7f0
            expected = -log(2f0 * Float32(pi)) - sum(log, diag(factor)) -
                (dof + 2f0) / 2f0 * log1p(sum(abs2, LowerTriangular(factor) \ offset) / dof)
            logdensityof(proposal, proposal.location + offset) ≈ expected
        end
    end
end
