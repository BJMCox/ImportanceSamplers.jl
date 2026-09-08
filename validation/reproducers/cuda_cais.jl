module CUDACAISValidation

using CUDA
using ImportanceSamplers
using LinearAlgebra
using MLDataDevices
using Random
using Test

const IS = ImportanceSamplers

struct QuadraticTarget{T} end
function (::QuadraticTarget{T})(x::AbstractVector) where {T}
    return -(abs2(x[1] - T(0.4)) + abs2(x[2] + T(0.3) * x[1])) / T(2)
end

struct StandardNormalTarget{T} end
function (::StandardNormalTarget{T})(x::Real) where {T}
    return -log(T(2pi)) / T(2) - abs2(x) / T(2)
end

struct TiltedStandardNormalTarget{T} end
function (::TiltedStandardNormalTarget{T})(x::AbstractVector) where {T}
    return -log(T(2pi)) - sum(abs2, x) / T(2) + T(6) * x[1]
end

struct StandardVectorTarget{T} end
function (::StandardVectorTarget{T})(x::AbstractVector) where {T}
    return -length(x) * log(T(2pi)) / T(2) - sum(abs2, x) / T(2)
end

mutable struct DeviceNormals{A} <: Random.AbstractRNG
    batches::A
    index::Int
end

function Random.randn!(rng::DeviceNormals, destination::CUDA.AnyCuArray)
    copyto!(destination, view(rng.batches, :, rng.index))
    rng.index += 1
    return destination
end

function cuda_device()
    physical = CUDA.device()
    return MLDataDevices.CUDADevice{typeof(physical),Nothing}(physical)
end

function with_device_normals(source, device, batches)
    base = device(source)
    return IS._PreparedImportanceSampler(
        DeviceNormals(CuArray(hcat(batches...)), 1),
        base.random_buffers,
        base.target,
        base.algorithm,
        base.method_state,
        base.device,
        base.factor_execution,
        base.threaded,
        false,
        false,
    )
end

function gaussian_logdensity(sample, location, factor)
    standardized = factor \ (sample - location)
    T = eltype(sample)
    return -length(location) * log(T(2pi)) / T(2) -
           sum(log, diag(factor)) - sum(abs2, standardized) / T(2)
end

function normalized_weights(logweights)
    weights = exp.(logweights .- maximum(logweights))
    return weights ./ sum(weights)
end

function tempered_weights(logweights, threshold)
    logs = BigFloat.(logweights)
    lower = big"0"
    upper = big"1"
    for _ in 1:200
        power = (lower + upper) / 2
        weights = normalized_weights(power .* logs)
        if inv(sum(abs2, weights)) >= threshold
            lower = power
        else
            upper = power
        end
    end
    return Float64.(normalized_weights(lower .* logs))
end

function check_vector_recurrence(
    result,
    learned,
    bank,
    schedule,
    target,
    threshold,
    ::Type{T},
) where {T}
    means = [Float64.(proposal.location) for proposal in bank.proposals]
    factors = [Float64.(proposal.scale.factor) for proposal in bank.proposals]
    proposal_count = length(means)
    offset = 0
    saw_high = false
    saw_low = false
    weight_tolerance = T === Float32 ? 4e-4 : 3e-5
    mean_tolerance = T === Float32 ? 4e-3 : 8e-6
    factor_tolerance = T === Float32 ? 6e-3 : 2e-4
    for (round, count) in pairs(schedule)
        local_count = count ÷ proposal_count
        indices = (offset + 1):(offset + count)
        samples = Float64.(result.samples[:, indices])
        next_means = similar(means)
        next_factors = similar(factors)
        for proposal in 1:proposal_count
            group_indices =
                ((proposal - 1) * local_count + 1):(proposal * local_count)
            logs = [
                Float64(target(T.(samples[:, sample]))) - gaussian_logdensity(
                    samples[:, sample],
                    means[proposal],
                    factors[proposal],
                ) for sample in group_indices
            ]
            @test all(isapprox.(
                Float64.(result.logweights[indices[group_indices]]),
                logs;
                rtol=weight_tolerance,
                atol=weight_tolerance,
            ))
            raw = normalized_weights(logs)
            next_means[proposal] = samples[:, group_indices] * raw
            raw_ess = inv(sum(abs2, raw))
            covariance_weights = raw
            center = means[proposal]
            if raw_ess < threshold
                saw_low = true
                covariance_weights = tempered_weights(logs, threshold)
                center = samples[:, group_indices] * covariance_weights
            else
                saw_high = true
            end
            centered = samples[:, group_indices] .- center
            covariance = centered * Diagonal(covariance_weights) * centered'
            next_factors[proposal] = Matrix(cholesky(Symmetric(covariance)).L)
        end
        @test result.provenance.round[indices] == fill(round, count)
        @test result.provenance.proposal_id[indices] ==
              repeat(1:proposal_count; inner=local_count)
        means = next_means
        factors = next_factors
        offset += count
    end
    @test [Float64.(proposal.location) for proposal in learned.proposals] ≈
          means rtol=mean_tolerance atol=mean_tolerance
    @test [Float64.(proposal.scale.factor) for proposal in learned.proposals] ≈
          factors rtol=factor_tolerance atol=factor_tolerance
    return (; saw_high, saw_low)
end

function public_device_result(result)
    @test result.samples isa CUDA.AnyCuArray
    @test result.logweights isa CUDA.AnyCuArray
    @test result.provenance.round isa CUDA.AnyCuArray
    @test result.provenance.proposal_id isa CUDA.AnyCuArray
    host = cpu_device()(result)
    @test Array(result.samples) == host.samples
    @test Array(result.logweights) == host.logweights
    @test Array(result.provenance.round) == host.provenance.round
    @test Array(result.provenance.proposal_id) == host.provenance.proposal_id
    return host
end

function check_vector_paths(device, ::Type{T}) where {T}
    bank = ProposalBank([
        FactorGaussian(T[-1, 0.5], T[1.1 0; 0.25 0.8]),
        FactorGaussian(T[1.25, -0.75], T[0.7 0; -0.2 1.3]),
    ], T[1, 1])
    schedule = [16, 20]
    records = []
    for policy in (BatchedFactorExecution(), FusedFactorExecution())
        prepared = device(prepare_sampler(
            Xoshiro(0xCA15),
            QuadraticTarget{T}(),
            CAIS(
                bank;
                rounds=2,
                round_size=schedule,
                covariance_ess_threshold=5,
            );
            factor_execution=policy,
        ))
        result = importance_sample!(prepared)
        host = public_device_result(result)
        learned = current_proposal(cpu_device(), prepared)
        branches = check_vector_recurrence(
            host,
            learned,
            bank,
            schedule,
            QuadraticTarget{T}(),
            5,
            T,
        )
        @test branches.saw_high
        @test branches.saw_low
        push!(records, (; host, learned, branches))
    end
    tolerance = T === Float32 ? 6e-3 : 3e-8
    @test records[1].host.samples ≈ records[2].host.samples rtol=tolerance atol=tolerance
    @test records[1].host.logweights ≈
          records[2].host.logweights rtol=tolerance atol=tolerance
    @test records[1].host.provenance.round == records[2].host.provenance.round
    @test records[1].host.provenance.proposal_id ==
          records[2].host.provenance.proposal_id
    @test [p.location for p in records[1].learned.proposals] ≈
          [p.location for p in records[2].learned.proposals] rtol=tolerance atol=tolerance
    @test [p.scale.factor for p in records[1].learned.proposals] ≈
          [p.scale.factor for p in records[2].learned.proposals] rtol=tolerance atol=tolerance
    return nothing
end

function check_rollback(device, ::Type{T}) where {T}
    regular = T[-1, 0, 1]
    batches = [regular, reverse(regular), regular, zeros(T, 3)]
    bank = ProposalBank([SphericalGaussian(zero(T), one(T))], T[1])
    source = prepare_sampler(
        Xoshiro(2),
        StandardNormalTarget{T}(),
        CAIS(bank; rounds=2, round_size=3, covariance_ess_threshold=2),
    )
    prepared = with_device_normals(source, device, batches)
    importance_sample!(prepared)
    committed = only(current_proposal(cpu_device(), prepared).proposals)
    failure = try
        importance_sample!(prepared)
        nothing
    catch error
        error
    end
    retained = only(current_proposal(cpu_device(), prepared).proposals)
    @test failure isa CAISRoundError
    @test failure.round == 2
    @test retained.location == committed.location
    @test retained.scale.scale == committed.scale.scale
    @test prepared.rng.index == 5
    return nothing
end

function check_workgroup_boundary_factor(device)
    T = Float64
    dimension = 40
    sample_count = 48
    bank = ProposalBank([
        FactorGaussian(zeros(T, dimension), Matrix{T}(I, dimension, dimension)),
    ], T[1])
    prepared = device(prepare_sampler(
        Xoshiro(0xCA1540),
        StandardVectorTarget{T}(),
        CAIS(
            bank;
            rounds=1,
            round_size=sample_count,
            covariance_ess_threshold=41,
        );
        factor_execution=FusedFactorExecution(),
    ))
    result = public_device_result(importance_sample!(prepared))
    learned = only(current_proposal(cpu_device(), prepared).proposals)
    expected_covariance = result.samples * result.samples' / T(sample_count)

    @test learned.scale.factor * learned.scale.factor' ≈
          expected_covariance rtol=2e-10 atol=2e-10
    return nothing
end

function main()
    CUDA.allowscalar(false)
    CUDA.functional() || error("CUDA is required for this reproducer")
    device = cuda_device()
    @testset "CUDA CAIS recurrence, residency, factor paths, and rollback" begin
        check_vector_paths(device, Float64)
        check_rollback(device, Float64)
        check_workgroup_boundary_factor(device)
    end
    return nothing
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    CUDACAISValidation.main()
end
