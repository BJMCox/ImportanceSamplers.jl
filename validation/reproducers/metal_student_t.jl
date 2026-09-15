using ImportanceSamplers
using LinearAlgebra
using Metal
using Random
using Test

Metal.functional() || error("Metal is not functional")
Metal.allowscalar(false)

function student_target(x, p)
    a = x[1] - p[1]
    b = x[2] - p[2] - 0.4f0 * a
    return -(a * a + b * b / 0.64f0) / 2f0
end

function student_gradient!(g, x, p)
    a = x[1] - p[1]
    b = (x[2] - p[2] - 0.4f0 * a) / 0.64f0
    g[1] = -a + 0.4f0 * b
    g[2] = -b
    return nothing
end

@testset "Student-t covariance adaptation on Metal" begin
    device = ImportanceSamplers.MLDataDevices.MetalDevice{Float32}()
    factor = Float32[1.4 0; 0.3 1.2]
    proposal = FactorStudentT(5f0, zeros(Float32, 2), factor)
    bank = ProposalBank([proposal, FactorStudentT(7f0, Float32[0.2, -0.1], factor)])
    options = (; rounds=3, round_size=4096)
    algorithms = (
        AMIS(proposal; options...),
        # Explicit device precision must also convert the history's scalar family data.
        NPMC(FactorStudentT(5.0, zeros(2), Float64.(factor)); options...),
        CAIS(bank; options...),
        FirstOrderGRAMIS(bank; repulsion_strength=0f0, options...),
    )
    for algorithm in algorithms
        @testset "$(nameof(typeof(algorithm)))" begin
            p = Float32[0.25, -0.15]
            sampler = device(prepare_sampler(
                Xoshiro(62), LogTarget(student_target; grad=student_gradient!), p, algorithm))
            first_result = importance_sample!(sampler)
            saved = Array(first_result.samples)
            for result in (first_result, importance_sample!(sampler))
                weights = Array(normalized_weights(result))
                @test Array(result.samples) * weights ≈ p atol=0.1
                # The target covariance has determinant 0.64, so Z = 2π * 0.8.
                @test lognormalizer(result) ≈ log(2pi * 0.8) atol=0.08
            end
            @test Array(first_result.samples) == saved
        end
    end
end
