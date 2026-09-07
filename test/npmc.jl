using Test, ImportanceSamplers
import Random, LinearAlgebra, DensityInterface, Statistics

# The RNG seam belongs to the test. Both executions consume the same round buffers.
mutable struct NPMCNormals{T} <: Random.AbstractRNG
    batches::Vector{Vector{T}}
    index::Int
end
function Random.randn!(rng::NPMCNormals, destination::AbstractArray)
    copyto!(destination, rng.batches[rng.index])
    rng.index += 1
    return destination
end

function npmc_oracle(result, initial, batches, schedule, logtarget)
    scalar = initial.location isa Number
    T = scalar ? typeof(initial.location) : eltype(initial.location)
    mean = scalar ? [initial.location] : copy(initial.location)
    factor = scalar ? fill(initial.scale.scale, 1, 1) : copy(initial.scale.factor)
    offset = 0
    for (round, n) in enumerate(schedule)
        indices = (offset + 1):(offset + n)
        x = scalar ? reshape(result.samples[indices], 1, :) : result.samples[:, indices]
        z = reshape(batches[round][1:(length(mean) * n)], length(mean), n)
        @test x ≈ mean .+ factor * z
        logq = [-length(mean) * log(T(2pi)) / 2 - sum(log, LinearAlgebra.diag(factor)) -
                sum(abs2, factor \ (x[:, i] - mean)) / 2 for i in 1:n]
        logs = [logtarget(scalar ? x[1, i] : x[:, i]) - logq[i] for i in 1:n]
        @test result.logweights[indices] ≈ logs
        @test all(==(round), result.provenance.round[indices])
        cap = sort(logs; rev=true)[isqrt(n)]
        clipped = min.(logs, cap)
        weights = exp.(clipped .- maximum(clipped))
        weights ./= sum(weights)
        previous_covariance = factor * factor'
        mean = x * weights
        centered = x .- mean
        covariance = (centered .* weights') * centered'
        ridge = sqrt(eps(T)) * LinearAlgebra.tr(previous_covariance) / length(mean)
        factor = Matrix(LinearAlgebra.cholesky(LinearAlgebra.Symmetric(
            covariance + ridge * LinearAlgebra.I,
        )).L)
        offset += n
    end
    return scalar ? SphericalGaussian(only(mean), only(factor)) : FactorGaussian(mean, factor)
end

@testset "NPMC raw weights and clipped current-round recurrence" begin
    for T in (Float32, Float64)
        initial = SphericalGaussian(T(-1), T(2))
        batches = [T[-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3],
                   T[3, 2, 1, 0.5, 0, -0.5, -1, -2, -3]]
        logtarget(x) = -abs2(x - T(0.75)) / T(3)
        sampler = prepare_sampler(NPMCNormals(batches, 1), logtarget,
            NPMC(initial; rounds=2, round_size=[9, 4]); threaded=false)
        result = importance_sample!(sampler)
        expected = npmc_oracle(result, initial, batches, [9, 4], logtarget)
        learned = current_proposal(sampler)
        @test learned.location ≈ expected.location
        @test learned.scale.scale ≈ expected.scale.scale
        @test length(result) == 13
        @test result.diagnostics.method == :npmc
        @test result.diagnostics.target_evaluations == 13
        @test result.diagnostics.proposal_evaluations == 13
    end
end

@testset "NPMC learns correlation and preserves log-scale invariance" begin
    initial = FactorGaussian([-1.0, 1.0], [2.0 0.0; 0.3 1.5])
    batches = [randn(Random.Xoshiro(seed), 32) for seed in (71, 72)]
    logtarget(x) = -0.5 * (abs2(x[1]) + abs2(x[2] - 0.8x[1]))
    algorithm = NPMC(initial; rounds=2, round_size=16)
    serial = prepare_sampler(NPMCNormals(batches, 1), logtarget, algorithm; threaded=false)
    result = importance_sample!(serial)
    expected = npmc_oracle(result, initial, batches, [16, 16], logtarget)
    learned = current_proposal(serial)
    @test learned.location ≈ expected.location
    @test learned.scale.factor * learned.scale.factor' ≈ expected.scale.factor * expected.scale.factor'
    threaded = importance_sample(NPMCNormals(batches, 1), logtarget, algorithm; threaded=true)
    @test threaded.samples == result.samples
    @test threaded.logweights == result.logweights
    batched = prepare_sampler(NPMCNormals(batches, 1), logtarget, algorithm;
        factor_execution=BatchedFactorExecution())
    batch_result = importance_sample!(batched)
    batch_expected = npmc_oracle(batch_result, initial, batches, [16, 16], logtarget)
    @test current_proposal(batched).location ≈ batch_expected.location
    @test batch_result.logweights ≈ result.logweights
    shifted = prepare_sampler(NPMCNormals(batches, 1), x -> logtarget(x) + 1000, algorithm)
    shifted_result = importance_sample!(shifted)
    @test shifted_result.samples ≈ result.samples
    @test shifted_result.logweights ≈ result.logweights .+ 1000
    @test current_proposal(shifted).location ≈ learned.location
    @test lognormalizer(shifted_result) ≈ lognormalizer(result) + 1000
end

@testset "NPMC clipping ties and sparse-support failure" begin
    initial = SphericalGaussian(0.0, 1.0)
    batches = [[-2.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0]]
    logtarget(x) = DensityInterface.logdensityof(initial, x)
    sampler = prepare_sampler(NPMCNormals(batches, 1), logtarget,
        NPMC(initial; rounds=1, round_size=9))
    result = importance_sample!(sampler)
    @test maximum(abs, result.logweights) < 1e-14
    @test current_proposal(sampler).location ≈ 0 atol=1e-14
    sparse(x) = x == 2 ? 0.0 : -Inf
    failed = prepare_sampler(NPMCNormals(batches, 1), sparse,
        NPMC(initial; rounds=1, round_size=9))
    @test_throws NPMCRoundError importance_sample!(failed)
    @test current_proposal(failed).location == initial.location
    @test current_proposal(failed).scale.scale == initial.scale.scale
end

@testset "NPMC repeated calls retain proposals and own results" begin
    logtarget(x) = -abs2(x) / 2
    sampler = prepare_sampler(Random.Xoshiro(71), logtarget,
        NPMC(SphericalGaussian(-1.0, 2.0); rounds=2, round_size=64))
    result = importance_sample!(sampler)
    saved_samples, saved_logs = copy(result.samples), copy(result.logweights)
    proposal = current_proposal(sampler)
    next_result = importance_sample!(sampler)
    @test next_result.logweights[1:64] ≈ [logtarget(x) -
        DensityInterface.logdensityof(proposal, x) for x in next_result.samples[1:64]]
    @test result.samples == saved_samples
    @test result.logweights == saved_logs
    retargeted = retarget(Random.Xoshiro(72), sampler, logtarget)
    @test current_proposal(retargeted).location == current_proposal(sampler).location
end

@testset "NPMC later-round failure retains the previous successful proposal" begin
    normal = [-2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0]
    batches = [normal, normal, normal, fill(-100.0, 9)]
    logtarget(x) = x > -5 ? -abs2(x) / 2 : -Inf
    sampler = prepare_sampler(NPMCNormals(batches, 1), logtarget,
        NPMC(SphericalGaussian(0.0, 1.0); rounds=2, round_size=9))
    importance_sample!(sampler)
    previous = current_proposal(sampler)
    failure = try
        importance_sample!(sampler)
        nothing
    catch error
        error
    end
    @test failure isa NPMCRoundError
    @test failure.round == 2
    @test failure.cause isa AllZeroWeightsError
    @test current_proposal(sampler).location == previous.location
    @test current_proposal(sampler).scale.scale == previous.scale.scale
end
