using Test
using ImportanceSamplers
import LinearAlgebra, Random

mutable struct APISPrefilledNormals{T} <: Random.AbstractRNG
    batches::Vector{Vector{T}}
    index::Int
end

function Random.randn!(rng::APISPrefilledNormals, destination::AbstractArray)
    copyto!(destination, rng.batches[rng.index])
    rng.index += 1
    return destination
end

function apis_normal_logdensity(sample, location, scale)
    T = typeof(sample)
    return -log(scale) - log(T(2pi)) / T(2) -
           abs2((sample - location) / scale) / T(2)
end

function apis_logadd(left, right)
    maximum = max(left, right)
    return maximum + log(exp(left - maximum) + exp(right - maximum))
end

function apis_scalar_oracle(initial_means, scales, batches, logtarget)
    means = copy(initial_means)
    samples = eltype(initial_means)[]
    logweights = eltype(initial_means)[]
    for batch in batches
        epoch_samples = [
            means[proposal] + scales[proposal] * batch[sample]
            for proposal in eachindex(means)
            for sample in ((proposal - 1) * 3 + 1):(proposal * 3)
        ]
        next_means = similar(means)
        for proposal in eachindex(means)
            indices = ((proposal - 1) * 3 + 1):(proposal * 3)
            local_logs = [
                logtarget(epoch_samples[sample]) - apis_normal_logdensity(
                    epoch_samples[sample],
                    means[proposal],
                    scales[proposal],
                ) for sample in indices
            ]
            local_weights = exp.(local_logs .- maximum(local_logs))
            next_means[proposal] = sum(
                local_weights .* epoch_samples[indices],
            ) / sum(local_weights)
        end
        for sample in epoch_samples
            left = apis_normal_logdensity(sample, means[1], scales[1])
            right = apis_normal_logdensity(sample, means[2], scales[2])
            push!(
                logweights,
                logtarget(sample) - apis_logadd(left, right) + log(typeof(sample)(2)),
            )
        end
        append!(samples, epoch_samples)
        means .= next_means
    end
    return (; samples, logweights, means)
end

function apis_factor_logdensity(sample, location, factor)
    T = eltype(sample)
    standardized = factor \ (sample - location)
    return -length(location) * log(T(2pi)) / T(2) -
           sum(log, LinearAlgebra.diag(factor)) - sum(abs2, standardized) / T(2)
end

function apis_factor_oracle(initial_means, factors, batches, schedule, logtarget)
    T = eltype(first(initial_means))
    means = deepcopy(initial_means)
    all_samples = Matrix{T}(undef, length(first(means)), 0)
    all_logweights = T[]
    for (epoch, count) in pairs(schedule)
        draws_per_proposal = count ÷ length(means)
        normals = reshape(
            batches[epoch][1:(length(first(means)) * count)],
            length(first(means)),
            count,
        )
        epoch_samples = similar(normals)
        for proposal in eachindex(means)
            first_sample = (proposal - 1) * draws_per_proposal + 1
            indices = first_sample:(proposal * draws_per_proposal)
            epoch_samples[:, indices] .= means[proposal] .+
                                         factors[proposal] * normals[:, indices]
        end
        next_means = similar(means)
        for proposal in eachindex(means)
            first_sample = (proposal - 1) * draws_per_proposal + 1
            indices = first_sample:(proposal * draws_per_proposal)
            local_logs = [
                logtarget(epoch_samples[:, sample]) - apis_factor_logdensity(
                    epoch_samples[:, sample],
                    means[proposal],
                    factors[proposal],
                ) for sample in indices
            ]
            local_weights = exp.(local_logs .- maximum(local_logs))
            local_weights ./= sum(local_weights)
            next_means[proposal] = epoch_samples[:, indices] * local_weights
        end
        for sample in eachcol(epoch_samples)
            left = apis_factor_logdensity(sample, means[1], factors[1])
            right = apis_factor_logdensity(sample, means[2], factors[2])
            push!(
                all_logweights,
                logtarget(sample) - apis_logadd(left, right) + log(T(2)),
            )
        end
        all_samples = hcat(all_samples, epoch_samples)
        means .= next_means
    end
    return (; samples=all_samples, logweights=all_logweights, means)
end

@testset "APIS correlated batched recurrence retains factors" begin
    initial_means = [[-1.0, 0.5], [1.5, -0.5]]
    factors = [[1.2 0.0; 0.4 0.8], [0.7 0.0; -0.3 1.4]]
    schedule = [8, 12]
    batches = [randn(Random.Xoshiro(seed), 24) for seed in (81, 82)]
    logtarget(sample) = -(
        abs2(sample[1] - 0.25) + abs2(sample[2] + 0.4sample[1])
    ) / 2
    bank = ProposalBank([
        FactorGaussian(initial_means[1], factors[1]),
        FactorGaussian(initial_means[2], factors[2]),
    ])
    sampler = prepare_sampler(
        APISPrefilledNormals(batches, 1),
        logtarget,
        APIS(bank; rounds=2, round_size=schedule);
        factor_execution=BatchedFactorExecution(),
    )

    result = importance_sample!(sampler)
    expected = apis_factor_oracle(initial_means, factors, batches, schedule, logtarget)
    learned = current_proposal(sampler)

    @test result.samples ≈ expected.samples
    @test result.logweights ≈ expected.logweights
    @test result.provenance.round == vcat(fill(1, 8), fill(2, 12))
    @test result.provenance.proposal_id == vcat(
        fill(1, 4),
        fill(2, 4),
        fill(1, 6),
        fill(2, 6),
    )
    @test [proposal.location for proposal in learned.proposals] ≈ expected.means
    @test [proposal.scale.factor for proposal in learned.proposals] == factors
end

@testset "APIS execution, variable epochs, and target scale invariance" begin
    initial_means = [-1.0, 1.5]
    scales = [1.25, 0.75]
    schedule = [8, 12]
    batches = [randn(Random.Xoshiro(seed), 12) for seed in (91, 92)]
    logtarget(sample) = -abs2(sample - 0.3) / 2
    bank = ProposalBank([
        SphericalGaussian(initial_means[1], scales[1]),
        SphericalGaussian(initial_means[2], scales[2]),
    ])
    algorithm = APIS(bank; rounds=2, round_size=schedule)
    serial = prepare_sampler(
        APISPrefilledNormals(deepcopy(batches), 1),
        logtarget,
        algorithm;
        threaded=false,
    )
    threaded = prepare_sampler(
        APISPrefilledNormals(deepcopy(batches), 1),
        logtarget,
        algorithm;
        threaded=true,
    )
    shifted = prepare_sampler(
        APISPrefilledNormals(deepcopy(batches), 1),
        sample -> logtarget(sample) + 1000,
        algorithm;
        threaded=true,
    )

    serial_result = importance_sample!(serial)
    threaded_result = importance_sample!(threaded)
    shifted_result = importance_sample!(shifted)

    @test threaded_result.samples == serial_result.samples
    @test threaded_result.logweights == serial_result.logweights
    @test shifted_result.logweights ≈ serial_result.logweights .+ 1000
    @test [p.location for p in current_proposal(shifted).proposals] ≈
          [p.location for p in current_proposal(serial).proposals]
    @test lognormalizer(shifted_result) ≈ lognormalizer(serial_result) + 1000
end

@testset "APIS repeated calls own results and retarget learned state" begin
    logtarget(sample) = -abs2(sample) / 2
    retarget_logtarget(sample) = -abs2(sample - 1) / 2
    initial_means = [-2.0, 2.0]
    scales = [1.0, 1.5]
    batches = [
        [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0],
        [0.5, -0.5, 1.0, -1.0, 0.5, 0.0],
        [-0.25, 0.5, 1.25, -1.25, -0.5, 0.25],
        [1.0, 0.0, -1.0, 0.75, -0.25, -0.75],
    ]
    bank = ProposalBank([
        SphericalGaussian(initial_means[1], scales[1]),
        SphericalGaussian(initial_means[2], scales[2]),
    ])
    sampler = prepare_sampler(
        APISPrefilledNormals(deepcopy(batches), 1),
        logtarget,
        APIS(bank; rounds=2, round_size=6),
    )
    first_result = importance_sample!(sampler)
    first_samples = copy(first_result.samples)
    first_logweights = copy(first_result.logweights)
    first_learned = current_proposal(sampler)
    second_result = importance_sample!(sampler)
    expected_second = apis_scalar_oracle(
        [proposal.location for proposal in first_learned.proposals],
        scales,
        batches[3:4],
        logtarget,
    )

    @test first_result.samples == first_samples
    @test first_result.logweights == first_logweights
    @test second_result.samples ≈ expected_second.samples
    @test second_result.logweights ≈ expected_second.logweights
    @test [p.location for p in current_proposal(sampler).proposals] ≈
          expected_second.means

    source_proposal = current_proposal(sampler)
    source_means = [p.location for p in source_proposal.proposals]
    retarget_batches = deepcopy(batches[1:2])
    retargeted = retarget(
        APISPrefilledNormals(retarget_batches, 1),
        sampler,
        retarget_logtarget,
    )
    retargeted_result = importance_sample!(retargeted)
    expected_retargeted = apis_scalar_oracle(
        source_means,
        scales,
        retarget_batches,
        retarget_logtarget,
    )
    @test retargeted_result.logweights ≈ expected_retargeted.logweights
    @test [p.location for p in current_proposal(retargeted).proposals] ≈
          expected_retargeted.means
    @test [p.location for p in current_proposal(sampler).proposals] == source_means
end

@testset "APIS later-epoch failure preserves the committed bank" begin
    normal = [-0.5, 0.0, 0.5, -0.5, 0.0, 0.5]
    batches = [normal, reverse(normal), normal, fill(-100.0, 6)]
    logtarget(sample) = sample > -50 ? -abs2(sample) / 2 : -Inf
    bank = ProposalBank([
        SphericalGaussian(-1.0, 1.0),
        SphericalGaussian(1.0, 1.0),
    ])
    rng = APISPrefilledNormals(batches, 1)
    sampler = prepare_sampler(
        rng,
        logtarget,
        APIS(bank; rounds=2, round_size=6),
    )
    importance_sample!(sampler)
    committed = current_proposal(sampler)

    failure = try
        importance_sample!(sampler)
        nothing
    catch error
        error
    end

    @test failure isa APISRoundError
    @test failure.round == 2
    @test [p.location for p in current_proposal(sampler).proposals] ==
          [p.location for p in committed.proposals]
    @test rng.index == 5
end

@testset "APIS public scalar epoch recurrence" begin
    T = Float32
    initial_means = T[-2, 2]
    scales = T[1, 2]
    normal_batches = [
        T[-1, 0.5, 1.5, -0.5, 0.25, 1],
        T[0.2, -1, 1.2, 0, 1, -1],
    ]
    logtarget(sample) = -abs2(sample - T(0.75)) / T(3)
    bank = ProposalBank([
        SphericalGaussian(initial_means[1], scales[1]),
        SphericalGaussian(initial_means[2], scales[2]),
    ])
    sampler = prepare_sampler(
        APISPrefilledNormals(normal_batches, 1),
        logtarget,
        APIS(bank; rounds=2, round_size=6);
        threaded=false,
    )

    result = importance_sample!(sampler)
    expected = apis_scalar_oracle(initial_means, scales, normal_batches, logtarget)
    learned = current_proposal(sampler)

    @test result.samples ≈ expected.samples rtol=5f-6
    @test result.logweights ≈ expected.logweights rtol=5f-6
    @test result.provenance.round == [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    @test result.provenance.proposal_id == [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2]
    @test [proposal.location for proposal in learned.proposals] ≈
          expected.means rtol=5f-6
    @test [proposal.scale.scale for proposal in learned.proposals] == scales
end
