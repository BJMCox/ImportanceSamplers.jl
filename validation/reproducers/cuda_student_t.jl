using CUDA
using ImportanceSamplers
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
