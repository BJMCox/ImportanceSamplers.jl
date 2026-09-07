module CUDANPMCValidation

using CUDA, ImportanceSamplers, MLDataDevices, Random, Test, LinearAlgebra

struct QuadraticTarget{T} end
struct TruncatedTarget end
(::TruncatedTarget)(x) = x > -5 ? -abs2(x) / 2 : -Inf

mutable struct RoundNormals{A} <: Random.AbstractRNG
    batches::A
    index::Int
end
function Random.randn!(rng::RoundNormals, destination::CUDA.AnyCuArray)
    copyto!(destination, view(rng.batches, :, rng.index))
    rng.index += 1
    return destination
end

function check_sparse_rollback(device)
    initial = SphericalGaussian(0.0, 1.0)
    base = device(prepare_sampler(Xoshiro(71), TruncatedTarget(),
        NPMC(initial; rounds=2, round_size=9)))
    normal = [-2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0]
    batches = CUDA.CuArray(hcat(normal, normal, normal, vcat(fill(-100.0, 8), 0.0)))
    # Supply deterministic resident random buffers through a test-only RNG seam.
    sampler = ImportanceSamplers._PreparedImportanceSampler(
        RoundNormals(batches, 1), base.random_buffers, base.target, base.algorithm,
        base.method_state, base.device, base.factor_execution, base.threaded, false, false)
    importance_sample!(sampler)
    previous = current_proposal(cpu_device(), sampler)
    failure = try importance_sample!(sampler); nothing catch error; error end
    @test failure isa NPMCRoundError
    @test failure.round == 2
    @test failure.cause isa AllZeroWeightsError
    @test current_proposal(cpu_device(), sampler).location == previous.location
    @test current_proposal(cpu_device(), sampler).scale.scale == previous.scale.scale
end

(::QuadraticTarget{T})(x::Real) where {T} = -abs2(x - T(0.75)) / T(2)
function (::QuadraticTarget{T})(x::AbstractVector) where {T}
    return -(abs2(x[1] - T(0.75)) + abs2(x[2] - T(0.6) * x[1])) / T(2)
end

# Infer the generating proposals from public samples and an independent moment fit.
function check_recurrence(result, initial, learned, schedule, target, ::Type{T}) where {T}
    scalar = initial.location isa Real
    mu = scalar ? [Float64(initial.location)] : Float64.(initial.location)
    factor = scalar ? fill(Float64(initial.scale.scale), 1, 1) : Float64.(initial.scale.factor)
    d = length(mu)
    offset = 0
    tolerance = T === Float32 ? 2e-3 : 1e-10
    for (round, n) in enumerate(schedule)
        indices = (offset + 1):(offset + n)
        x = scalar ? reshape(Float64.(result.samples[indices]), 1, :) : Float64.(result.samples[:, indices])
        logs = [Float64(target(scalar ? T(x[1, i]) : T.(x[:, i]))) +
                d * log(2pi) / 2 + sum(log, diag(factor)) +
                sum(abs2, factor \ (x[:, i] - mu)) / 2 for i in 1:n]
        @test result.logweights[indices] ≈ logs rtol=tolerance atol=tolerance
        @test all(==(round), result.provenance.round[indices])
        cap = sort(logs; rev=true)[isqrt(n)]
        clipped = min.(logs, cap)
        w = exp.(clipped .- maximum(clipped))
        w ./= sum(w)
        previous_trace = sum(abs2, factor)
        mu = x * w
        centered = x .- mu
        covariance = (centered .* w') * centered' + sqrt(eps(T)) * previous_trace / d * I
        factor = Matrix(cholesky(Symmetric(covariance)).L)
        offset += n
    end
    learned_mu = scalar ? [learned.location] : learned.location
    learned_factor = scalar ? fill(learned.scale.scale, 1, 1) : learned.scale.factor
    @test learned_mu ≈ mu rtol=tolerance atol=tolerance
    @test learned_factor * learned_factor' ≈ factor * factor' rtol=tolerance atol=tolerance
end

function main()
    CUDA.allowscalar(false)
    CUDA.functional() || error("CUDA is required for this reproducer")
    physical = CUDA.device()
    device = MLDataDevices.CUDADevice{typeof(physical),Nothing}(physical)
    records = []
    @testset "CUDA NPMC independent recurrence and residency" begin
        check_sparse_rollback(device)
        for (T, scalar, policy) in ((Float32, true, nothing),
                                  (Float64, true, nothing),
                                  (Float32, false, BatchedFactorExecution()),
                                  (Float64, false, BatchedFactorExecution()),
                                  (Float64, false, FusedFactorExecution()))
            proposal = scalar ? SphericalGaussian(T(-1), T(2)) :
                FactorGaussian(T[-1, 1], T[2 0; 0.3 1.5])
            target = QuadraticTarget{T}()
            schedule = [4096, 2048, 4096]
            options = isnothing(policy) ? (;) : (; factor_execution=policy)
            prepared = device(prepare_sampler(Xoshiro(42), target,
                NPMC(proposal; rounds=3, round_size=schedule); options...))
            result = importance_sample!(prepared)
            @test result.samples isa CUDA.AnyCuArray
            @test result.logweights isa CUDA.AnyCuArray
            @test length(result) == sum(schedule)
            host = cpu_device()(result)
            learned = current_proposal(cpu_device(), prepared)
            check_recurrence(host, proposal, learned, schedule, target, T)
            next_result = importance_sample!(prepared)
            check_recurrence(cpu_device()(next_result), learned,
                current_proposal(cpu_device(), prepared), schedule, target, T)
            @test Array(result.logweights) == host.logweights
            push!(records, (; type=T, scalar, policy=typeof(policy),
                transfers=result.diagnostics.transfers.count))
        end
    end
    return (; hardware=CUDA.name(CUDA.device()), records)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    CUDANPMCValidation.main()
end
