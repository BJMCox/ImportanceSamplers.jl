module CUDAAPISValidation

using CUDA, ImportanceSamplers, LinearAlgebra, MLDataDevices, Random, Test

const IS = ImportanceSamplers

struct QuadraticTarget{T} end
(::QuadraticTarget{T})(x::Real) where {T} = -abs2(x - T(0.5)) / T(2)
function (::QuadraticTarget{T})(x::AbstractVector) where {T}
    return -(abs2(x[1] - T(0.5)) + abs2(x[2] + T(0.4) * x[1])) / T(2)
end

struct TruncatedTarget end
(::TruncatedTarget)(x) = x > -50 ? -abs2(x) / 2 : -Inf

mutable struct RoundNormals{A} <: Random.AbstractRNG
    batches::A
    index::Int
end

function Random.randn!(rng::RoundNormals, destination::CUDA.AnyCuArray)
    copyto!(destination, view(rng.batches, :, rng.index))
    rng.index += 1
    return destination
end

location_vector(proposal) = proposal.location isa Real ?
                            [Float64(proposal.location)] :
                            Float64.(proposal.location)

factor_matrix(proposal) = proposal.location isa Real ?
                          fill(Float64(proposal.scale.scale), 1, 1) :
                          Float64.(proposal.scale.factor)

function gaussian_logdensity(sample, location, factor)
    standardized = factor \ (sample - location)
    return -length(location) * log(2pi) / 2 - sum(log, diag(factor)) -
           sum(abs2, standardized) / 2
end

function logadd(left, right)
    largest = max(left, right)
    return largest + log(exp(left - largest) + exp(right - largest))
end

# One independent oracle checks both estimator and adaptation weights from the
# public samples. It never calls package-private density or moment helpers.
function check_recurrence(
    result,
    initial_bank,
    learned,
    schedule,
    target,
    ::Type{T},
) where {T}
    means = location_vector.(initial_bank.proposals)
    factors = factor_matrix.(initial_bank.proposals)
    proposal_count = length(means)
    scalar = initial_bank.proposals[1].location isa Real
    tolerance = T === Float32 ? 3e-3 : 2e-10
    offset = 0
    for (round, count) in pairs(schedule)
        draws_per_proposal = count ÷ proposal_count
        indices = (offset + 1):(offset + count)
        samples = scalar ?
                  reshape(Float64.(result.samples[indices]), 1, :) :
                  Float64.(result.samples[:, indices])
        expected_logweights = Float64[]
        next_means = similar(means)
        for sample in eachcol(samples)
            mixture = -Inf
            for proposal in eachindex(means)
                term = gaussian_logdensity(sample, means[proposal], factors[proposal]) -
                       log(proposal_count)
                mixture = isfinite(mixture) ? logadd(mixture, term) : term
            end
            value = scalar ? T(sample[1]) : T.(sample)
            push!(expected_logweights, Float64(target(value)) - mixture)
        end
        @test Float64.(result.logweights[indices]) ≈
              expected_logweights rtol=tolerance atol=tolerance
        @test all(==(round), result.provenance.round[indices])
        @test result.provenance.proposal_id[indices] ==
              repeat(1:proposal_count; inner=draws_per_proposal)
        for proposal in eachindex(means)
            first_sample = (proposal - 1) * draws_per_proposal + 1
            group_indices = first_sample:(proposal * draws_per_proposal)
            local_logs = [
                Float64(target(scalar ? T(samples[1, sample]) : T.(samples[:, sample]))) -
                gaussian_logdensity(samples[:, sample], means[proposal], factors[proposal])
                for sample in group_indices
            ]
            weights = exp.(local_logs .- maximum(local_logs))
            weights ./= sum(weights)
            next_means[proposal] = samples[:, group_indices] * weights
        end
        means = next_means
        offset += count
    end
    learned_means = location_vector.(learned.proposals)
    @test learned_means ≈ means rtol=tolerance atol=tolerance
    @test factor_matrix.(learned.proposals) == factors
    return nothing
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

function validation_bank(::Type{T}, scalar) where {T}
    return scalar ? ProposalBank([
        SphericalGaussian(T(-2), T(1)),
        SphericalGaussian(T(2), T(1.5)),
    ], T[1, 1]) : ProposalBank([
        FactorGaussian(T[-2, 0.5], T[1.2 0; 0.3 0.8]),
        FactorGaussian(T[2, -0.5], T[0.7 0; -0.2 1.4]),
    ], T[1, 1])
end

function check_local_zero_rollback(device)
    normal = [-0.5, 0.0, 0.5, -0.5, 0.0, 0.5]
    batches = CuArray(hcat(normal, reverse(normal), normal, fill(-100.0, 6)))
    source = prepare_sampler(
        Xoshiro(101),
        TruncatedTarget(),
        APIS(validation_bank(Float64, true); rounds=2, round_size=6),
    )
    base = device(source)
    sampler = IS._PreparedImportanceSampler(
        RoundNormals(batches, 1),
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
    importance_sample!(sampler)
    committed = current_proposal(cpu_device(), sampler)
    failure = try
        importance_sample!(sampler)
        nothing
    catch error
        error
    end
    @test failure isa APISRoundError
    @test failure.round == 2
    @test location_vector.(current_proposal(cpu_device(), sampler).proposals) ==
          location_vector.(committed.proposals)
    @test sampler.rng.index == 5
    return nothing
end

function main()
    CUDA.allowscalar(false)
    CUDA.functional() || error("CUDA is required for this reproducer")
    physical = CUDA.device()
    device = MLDataDevices.CUDADevice{typeof(physical),Nothing}(physical)
    schedule = [8, 12]
    records = []
    @testset "CUDA APIS recurrence, residency, reuse, and rollback" begin
        check_local_zero_rollback(device)
        scalar_bank = validation_bank(Float32, true)
        scalar = device(prepare_sampler(
            Xoshiro(102),
            QuadraticTarget{Float32}(),
            APIS(scalar_bank; rounds=2, round_size=schedule),
        ))
        scalar_result = importance_sample!(scalar)
        scalar_host = public_device_result(scalar_result)
        scalar_learned = current_proposal(cpu_device(), scalar)
        check_recurrence(
            scalar_host,
            scalar_bank,
            scalar_learned,
            schedule,
            QuadraticTarget{Float32}(),
            Float32,
        )
        scalar_next = importance_sample!(scalar)
        scalar_next_host = public_device_result(scalar_next)
        check_recurrence(
            scalar_next_host,
            scalar_learned,
            current_proposal(cpu_device(), scalar),
            schedule,
            QuadraticTarget{Float32}(),
            Float32,
        )
        push!(records, (scalar_type=Float32, scalar=true))

        factor_bank = validation_bank(Float64, false)
        factor = device(prepare_sampler(
            Xoshiro(103),
            QuadraticTarget{Float64}(),
            APIS(factor_bank; rounds=2, round_size=schedule);
            factor_execution=BatchedFactorExecution(),
        ))
        factor_result = importance_sample!(factor)
        factor_host = public_device_result(factor_result)
        check_recurrence(
            factor_host,
            factor_bank,
            current_proposal(cpu_device(), factor),
            schedule,
            QuadraticTarget{Float64}(),
            Float64,
        )
        push!(records, (scalar_type=Float64, scalar=false))
    end
    return (; hardware=CUDA.name(CUDA.device()), records)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    CUDAAPISValidation.main()
end
