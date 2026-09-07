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

function assert_resident(prepared, result)
    state = prepared.method_state
    bank = state.bank
    workspace = state.workspace
    arrays = Any[
        bank.locations,
        bank.lognormalizers,
        bank.logmasses,
        bank.cdf,
        bank.proposal_ids,
        state.plan.counts,
        state.plan.assignments,
        state.plan.logcoefficients,
        workspace.round_samples,
        workspace.round_logweights,
        workspace.round_proposal_ids,
        workspace.round_logtargets,
        workspace.round_generating_logdensities,
        workspace.scaled_local_weights,
        workspace.proposal_maxima,
        workspace.candidate_locations,
        prepared.random_buffers.normals,
        prepared.random_buffers.failure_scratch.record.storage,
        result.samples,
        result.logweights,
        result.provenance.round,
        result.provenance.proposal_id,
    ]
    push!(arrays, bank isa IS._PackedFactorGaussianBank ? bank.factors : bank.scales)
    workspace.solve_scratch isa IS._NoMISSolveScratch ||
        push!(arrays, workspace.solve_scratch)
    @test all(array -> array isa CUDA.AnyCuArray, arrays)
    return nothing
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
    @test failure.phase == :adaptation
    @test failure.cause isa AllZeroWeightsError
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
    schedule = [4096, 2048, 4096]
    records = []
    @testset "CUDA APIS recurrence, residency, reuse, and rollback" begin
        check_local_zero_rollback(device)
        for (T, scalar, policy) in (
            (Float32, true, nothing),
            (Float64, true, nothing),
            (Float32, false, nothing),
            (Float64, false, BatchedFactorExecution()),
            (Float64, false, FusedFactorExecution()),
        )
            bank = validation_bank(T, scalar)
            options = isnothing(policy) ? (;) : (; factor_execution=policy)
            prepared = device(prepare_sampler(
                Xoshiro(102),
                QuadraticTarget{T}(),
                APIS(bank; rounds=3, round_size=schedule);
                options...,
            ))
            result = importance_sample!(prepared)
            assert_resident(prepared, result)
            host = cpu_device()(result)
            learned = current_proposal(cpu_device(), prepared)
            check_recurrence(host, bank, learned, schedule, QuadraticTarget{T}(), T)

            next_result = importance_sample!(prepared)
            next_learned = current_proposal(cpu_device(), prepared)
            check_recurrence(
                cpu_device()(next_result),
                learned,
                next_learned,
                schedule,
                QuadraticTarget{T}(),
                T,
            )
            @test Array(result.logweights) == host.logweights
            @test result.diagnostics.transfers.reasons.local_mean_validity.count == 3
            push!(records, (
                scalar_type=T,
                scalar,
                factor_execution=typeof(policy),
                transfers=result.diagnostics.transfers.count,
            ))
        end
    end
    return (; hardware=CUDA.name(CUDA.device()), records)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    CUDAAPISValidation.main()
end
