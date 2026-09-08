using Test
using ImportanceSamplers
import LinearAlgebra, Random

mutable struct CAISPrefilledNormals{T} <: Random.AbstractRNG
    batches::Vector{Vector{T}}
    index::Int
end

function Random.randn!(rng::CAISPrefilledNormals, destination::AbstractArray)
    copyto!(destination, rng.batches[rng.index])
    rng.index += 1
    return destination
end

function cais_factor_logdensity(sample, location, factor)
    T = eltype(sample)
    standardized = factor \ (sample - location)
    return -length(location) * log(T(2pi)) / T(2) -
           sum(log, LinearAlgebra.diag(factor)) - sum(abs2, standardized) / T(2)
end

function cais_softmax(logweights)
    weights = exp.(logweights .- maximum(logweights))
    return weights ./ sum(weights)
end

function cais_tempered_weights(logweights, threshold)
    logs = BigFloat.(logweights)
    low = big"0"
    high = big"1"
    for _ in 1:200
        power = (low + high) / 2
        weights = cais_softmax(power .* logs)
        if inv(sum(abs2, weights)) >= threshold
            low = power
        else
            high = power
        end
    end
    return Float64.(cais_softmax(low .* logs))
end

@testset "CAIS covariance kernel preserves lane state across CPU barriers" begin
    T = Float64
    dimension = 4
    sample_count = 7
    bank = ProposalBank(
        [
            FactorGaussian(
                zeros(T, dimension),
                Matrix{T}(LinearAlgebra.I, dimension, dimension),
            ),
        ],
        T[1],
    )
    algorithm = CAIS(
        bank;
        rounds=1,
        round_size=sample_count,
        covariance_ess_threshold=5,
    )
    sampler = prepare_sampler(
        Random.Xoshiro(1),
        x -> -sum(abs2, x) / T(2),
        algorithm;
        threaded=false,
    )
    method_state = sampler.method_state
    workspace = method_state.workspace
    samples = reshape(T.(1:(dimension * sample_count)), dimension, sample_count) ./
              T(10)
    copyto!(workspace.round_samples, samples)
    fill!(workspace.normalized_weights, inv(T(sample_count)))
    fill!(workspace.tempering_powers, one(T))
    fill!(
        workspace.factor_status,
        ImportanceSamplers._POPULATION_COVARIANCE_READY,
    )

    ImportanceSamplers._fit_population_covariances!(
        workspace.covariances,
        workspace.covariance_centres,
        workspace.normalized_weights,
        workspace.tempering_powers,
        workspace.factor_status,
        workspace.round_samples,
        method_state.run_bank.locations,
        method_state.run_bank,
        workspace.local_starts,
        method_state.plan.counts,
        1,
        ImportanceSamplers._KernelExecution(
            ImportanceSamplers._SerialCPUExecution(),
        ),
    )

    @test view(workspace.covariances, :, :, 1) ≈
          samples * samples' / T(sample_count)
end

@testset "CAIS public vector low-ESS recurrence" begin
    T = Float32
    samples = T[-1 0 1 0; 0 1 0 -2]
    initial_mean = zeros(T, 2)
    initial_factor = Matrix{T}(LinearAlgebra.I, 2, 2)
    logtarget(x) = cais_factor_logdensity(x, initial_mean, initial_factor) + T(6)x[1]
    algorithm = CAIS(
        ProposalBank([FactorGaussian(initial_mean, initial_factor)]);
        rounds=1,
        round_size=4,
        covariance_ess_threshold=3,
    )
    sampler = prepare_sampler(
        CAISPrefilledNormals([vec(samples)], 1),
        logtarget,
        algorithm;
        threaded=false,
    )

    result = importance_sample!(sampler)
    learned = only(current_proposal(sampler).proposals)
    raw_logweights = T(6) .* vec(samples[1, :])
    raw_weights = cais_softmax(raw_logweights)
    covariance_weights = cais_tempered_weights(raw_logweights, 3)
    raw_mean = samples * raw_weights
    transformed_mean = samples * covariance_weights
    centered = samples .- transformed_mean
    expected_covariance = centered * LinearAlgebra.Diagonal(covariance_weights) * centered'

    @test result.samples ≈ samples rtol=5f-6
    @test result.logweights ≈ raw_logweights atol=5f-6
    @test learned.location ≈ raw_mean atol=5f-5
    @test learned.location ≉ transformed_mean atol=1e-2
    @test learned.scale.factor * learned.scale.factor' ≈
          expected_covariance atol=8f-4
    @test result.diagnostics.tempering_powers[1, 1] < 1
end

@testset "CAIS public scalar m - 1 threshold recurrence" begin
    bank = ProposalBank([SphericalGaussian(0.0, 1.0)])
    algorithm = CAIS(
        bank;
        rounds=1,
        round_size=4,
        covariance_ess_threshold=3,
    )

    logtarget(x) = x == 3 ? -Inf : -log(2pi) / 2 - abs2(x) / 2
    sampler = prepare_sampler(
        CAISPrefilledNormals([[0.0, 1.0, 2.0, 3.0]], 1),
        logtarget,
        algorithm;
        threaded=false,
    )
    result = importance_sample!(sampler)
    learned = only(current_proposal(sampler).proposals)

    @test result.samples == [0.0, 1.0, 2.0, 3.0]
    @test result.logweights[1:3] ≈ zeros(3) atol=1e-14
    @test result.logweights[4] == -Inf
    @test learned.location ≈ 1.0
    @test learned.scale.scale ≈ sqrt(5 / 3)
    @test result.diagnostics.local_ess[1, 1] ≈ 3.0
    @test result.diagnostics.tempering_powers[1, 1] == 1.0
end

function cais_population_oracle(
    initial_means,
    initial_factors,
    batches,
    schedule,
    logtarget,
    threshold,
)
    means = deepcopy(initial_means)
    factors = deepcopy(initial_factors)
    dimension = length(first(means))
    proposal_count = length(means)
    samples = Matrix{eltype(first(means))}(undef, dimension, 0)
    logweights = eltype(first(means))[]
    round_ids = Int[]
    proposal_ids = Int[]
    for (round, count) in pairs(schedule)
        local_count = count ÷ proposal_count
        normals = reshape(batches[round][1:(dimension * count)], dimension, count)
        round_samples = similar(normals)
        next_means = similar(means)
        next_factors = similar(factors)
        for proposal in eachindex(means)
            indices = ((proposal - 1) * local_count + 1):(proposal * local_count)
            round_samples[:, indices] .= means[proposal] .+
                                          factors[proposal] * normals[:, indices]
            local_logs = [
                logtarget(round_samples[:, sample]) - cais_factor_logdensity(
                    round_samples[:, sample],
                    means[proposal],
                    factors[proposal],
                ) for sample in indices
            ]
            raw_weights = cais_softmax(local_logs)
            raw_ess = inv(sum(abs2, raw_weights))
            next_means[proposal] = round_samples[:, indices] * raw_weights
            covariance_weights = raw_weights
            center = means[proposal]
            if raw_ess < threshold
                covariance_weights = cais_tempered_weights(local_logs, threshold)
                center = round_samples[:, indices] * covariance_weights
            end
            centered = round_samples[:, indices] .- center
            covariance = centered *
                         LinearAlgebra.Diagonal(covariance_weights) * centered'
            next_factors[proposal] = Matrix(
                LinearAlgebra.cholesky(LinearAlgebra.Symmetric(covariance)).L,
            )
            append!(logweights, local_logs)
            append!(proposal_ids, fill(proposal, local_count))
        end
        samples = hcat(samples, round_samples)
        append!(round_ids, fill(round, count))
        means = next_means
        factors = next_factors
    end
    return (; samples, logweights, round_ids, proposal_ids, means, factors)
end

@testset "CAIS two-round correlated population recurrence and policies" begin
    initial_means = [[-1.0, 0.5], [1.25, -0.75]]
    initial_factors = [[1.1 0.0; 0.25 0.8], [0.7 0.0; -0.2 1.3]]
    schedule = [8, 10]
    batches = [randn(Random.Xoshiro(seed), 20) for seed in (101, 102)]
    logtarget(x) = -(abs2(x[1] - 0.2) + abs2(x[2] + 0.3x[1])) / 2
    bank = ProposalBank([
        FactorGaussian(initial_means[1], initial_factors[1]),
        FactorGaussian(initial_means[2], initial_factors[2]),
    ])
    algorithm = CAIS(
        bank;
        rounds=2,
        round_size=schedule,
        covariance_ess_threshold=3,
    )
    oracle = cais_population_oracle(
        initial_means,
        initial_factors,
        batches,
        schedule,
        logtarget,
        3,
    )
    serial = prepare_sampler(
        CAISPrefilledNormals(deepcopy(batches), 1),
        logtarget,
        algorithm;
        threaded=false,
        factor_execution=FusedFactorExecution(),
    )
    batched = prepare_sampler(
        CAISPrefilledNormals(deepcopy(batches), 1),
        logtarget,
        algorithm;
        threaded=false,
        factor_execution=BatchedFactorExecution(),
    )
    serial_result = importance_sample!(serial)
    batched_result = importance_sample!(batched)

    @test serial_result.samples ≈ oracle.samples atol=1e-3
    @test serial_result.logweights ≈ oracle.logweights atol=1e-3
    @test serial_result.provenance.round == oracle.round_ids
    @test serial_result.provenance.proposal_id == oracle.proposal_ids
    @test batched_result.samples ≈ serial_result.samples
    @test batched_result.logweights ≈ serial_result.logweights
    @test batched_result.provenance.round == oracle.round_ids
    @test batched_result.provenance.proposal_id == oracle.proposal_ids
    @test [p.location for p in current_proposal(serial).proposals] ≈
          oracle.means atol=1e-3
    @test [p.scale.factor for p in current_proposal(serial).proposals] ≈
          oracle.factors atol=8e-4
    @test [p.location for p in current_proposal(batched).proposals] ≈
          [p.location for p in current_proposal(serial).proposals]
    @test [p.scale.factor for p in current_proposal(batched).proposals] ≈
          [p.scale.factor for p in current_proposal(serial).proposals]
end

@testset "CAIS target scale invariance and Float32 recurrence" begin
    T = Float32
    bank = ProposalBank([SphericalGaussian(T(-1), T(1.5))], T[1])
    algorithm = CAIS(bank; rounds=2, round_size=[4, 5])
    batches = [T[-1, -0.25, 0.5, 1, 0], T[-0.75, -0.1, 0.25, 0.8, 1.2]]
    logtarget(x) = -abs2(x - T(0.25)) / T(2)
    base = prepare_sampler(
        CAISPrefilledNormals(deepcopy(batches), 1),
        logtarget,
        algorithm,
    )
    shifted = prepare_sampler(
        CAISPrefilledNormals(deepcopy(batches), 1),
        x -> logtarget(x) + T(100),
        algorithm,
    )
    base_result = importance_sample!(base)
    shifted_result = importance_sample!(shifted)
    base_proposal = only(current_proposal(base).proposals)
    shifted_proposal = only(current_proposal(shifted).proposals)

    @test shifted_result.samples ≈ base_result.samples rtol=2f-5
    @test shifted_result.logweights ≈ base_result.logweights .+ T(100) rtol=2f-5
    @test shifted_proposal.location ≈ base_proposal.location rtol=2f-5
    @test shifted_proposal.scale.scale ≈ base_proposal.scale.scale rtol=2f-5
    @test lognormalizer(shifted_result) ≈ lognormalizer(base_result) + T(100) rtol=2f-5
end

@testset "CAIS repeated calls own results and retarget final factors" begin
    schedule = [4, 4]
    batches = [randn(Random.Xoshiro(seed), 8) for seed in 103:108]
    bank = ProposalBank([FactorGaussian([0.0, 0.0], [1.0 0.0; 0.2 0.8])])
    algorithm = CAIS(bank; rounds=2, round_size=schedule, covariance_ess_threshold=3)
    logtarget(x) = -(abs2(x[1] - 0.4) + abs2(x[2] + 0.2)) / 2
    retarget_logtarget(x) = -sum(abs2, x) / 2
    sampler = prepare_sampler(CAISPrefilledNormals(deepcopy(batches), 1), logtarget, algorithm)
    first_result = importance_sample!(sampler)
    first_samples = copy(first_result.samples)
    first_logweights = copy(first_result.logweights)
    first_proposal = current_proposal(sampler)
    second_result = importance_sample!(sampler)
    second_control = prepare_sampler(
        CAISPrefilledNormals(deepcopy(batches[3:4]), 1),
        logtarget,
        CAIS(
            first_proposal;
            rounds=2,
            round_size=schedule,
            covariance_ess_threshold=3,
        ),
    )
    control_result = importance_sample!(second_control)

    @test first_result.samples == first_samples
    @test first_result.logweights == first_logweights
    @test second_result.samples == control_result.samples
    @test second_result.logweights == control_result.logweights
    @test only(current_proposal(sampler).proposals).location ==
          only(current_proposal(second_control).proposals).location
    @test only(current_proposal(sampler).proposals).scale.factor ==
          only(current_proposal(second_control).proposals).scale.factor

    source = only(current_proposal(sampler).proposals)
    source_location = copy(source.location)
    source_factor = copy(source.scale.factor)
    retarget_batches = deepcopy(batches[5:6])
    retargeted = retarget(
        CAISPrefilledNormals(retarget_batches, 1),
        sampler,
        retarget_logtarget,
    )
    retargeted_result = importance_sample!(retargeted)
    retargeted_control = prepare_sampler(
        CAISPrefilledNormals(deepcopy(retarget_batches), 1),
        retarget_logtarget,
        CAIS(
            current_proposal(sampler);
            rounds=2,
            round_size=schedule,
            covariance_ess_threshold=3,
        ),
    )
    retargeted_control_result = importance_sample!(retargeted_control)
    @test retargeted_result.samples == retargeted_control_result.samples
    @test retargeted_result.logweights == retargeted_control_result.logweights
    @test only(current_proposal(retargeted).proposals).scale.factor ==
          only(current_proposal(retargeted_control).proposals).scale.factor
    @test only(current_proposal(sampler).proposals).location == source_location
    @test only(current_proposal(sampler).proposals).scale.factor == source_factor
end

@testset "CAIS later-round singular fit preserves the committed bank" begin
    regular = [-1.0, 0.0, 1.0]
    batches = [
        regular,
        reverse(regular),
        regular,
        zeros(3),
        reverse(regular),
        regular,
    ]
    rng = CAISPrefilledNormals(batches, 1)
    algorithm = CAIS(
        ProposalBank([SphericalGaussian(0.0, 1.0)]);
        rounds=2,
        round_size=3,
        covariance_ess_threshold=2,
    )
    sampler = prepare_sampler(rng, x -> -abs2(x) / 2, algorithm; threaded=false)
    importance_sample!(sampler)
    committed = only(current_proposal(sampler).proposals)

    failure = try
        importance_sample!(sampler)
        nothing
    catch error
        error
    end
    retained = only(current_proposal(sampler).proposals)

    @test failure isa CAISRoundError
    @test failure.round == 2
    @test retained.location == committed.location
    @test retained.scale.scale == committed.scale.scale
    @test rng.index == 5

    control = prepare_sampler(
        CAISPrefilledNormals(deepcopy(batches[5:6]), 1),
        x -> -abs2(x) / 2,
        CAIS(
            ProposalBank([committed]);
            rounds=2,
            round_size=3,
            covariance_ess_threshold=2,
        );
        threaded=false,
    )
    recovered_result = importance_sample!(sampler)
    control_result = importance_sample!(control)
    @test recovered_result.samples ≈ control_result.samples
    @test recovered_result.logweights ≈ control_result.logweights
    @test only(current_proposal(sampler).proposals).location ≈
          only(current_proposal(control).proposals).location
    @test only(current_proposal(sampler).proposals).scale.scale ≈
          only(current_proposal(control).proposals).scale.scale
end

@testset "CAIS public factor recurrence is nestable with threaded execution" begin
    if Threads.nthreads(:default) > 1
        initial_mean = zeros(2)
        initial_factor = Matrix{Float64}(LinearAlgebra.I, 2, 2)
        normals = [-1.0 1.0 0.0 0.0; 0.0 0.0 -1.0 1.0]
        worker_count = min(Threads.nthreads(:default), 4)
        learned = Vector{Any}(undef, worker_count)
        failures = Vector{Any}(undef, worker_count)
        fill!(failures, nothing)

        Threads.@threads :static for worker in 1:worker_count
            try
                target(x) = cais_factor_logdensity(x, initial_mean, initial_factor)
                sampler = prepare_sampler(
                    CAISPrefilledNormals([vec(normals)], 1),
                    target,
                    CAIS(
                        ProposalBank([FactorGaussian(initial_mean, initial_factor)]);
                        rounds=1,
                        round_size=4,
                        covariance_ess_threshold=3,
                    );
                    threaded=true,
                )
                importance_sample!(sampler)
                learned[worker] = only(current_proposal(sampler).proposals)
            catch error
                failures[worker] = error
            end
        end

        @test all(isnothing, failures)
        if all(isnothing, failures)
            @test all(proposal -> proposal.location ≈ zeros(2), learned)
            @test all(
                proposal -> proposal.scale.factor * proposal.scale.factor' ≈
                            0.5 .* Matrix{Float64}(LinearAlgebra.I, 2, 2),
                learned,
            )
        end
    else
        @test_skip "requires multiple default-pool threads"
    end
end
