using Test, ImportanceSamplers
import Random, LinearAlgebra, Statistics

include("support/lais.jl")

@testset "LAIS correlated upper moves allow one lower draw per proposal" begin
    factor = [1.0 0.0; 0.2 0.8]
    bank = ProposalBank([
        FactorGaussian([-1.0, 0.0], factor),
        FactorGaussian([1.0, 0.0], factor),
    ])
    rng = LAISScriptedRNG([[1.0, 2.0, -1.0, 1.0], zeros(4)], [[0.5, 0.5]], 1, 1)
    sampler = prepare_sampler(rng, x -> 0.0,
        LAIS(bank; transition=RandomWalkMetropolis([1.0 0.5; 0.5 1.25]),
            rounds=1, round_size=2))
    result = importance_sample!(sampler)
    @test result.samples ≈ [0.0 0.0; 2.5 0.5]
    delta = factor \ [0.0, 2.0]
    density = (1 + exp(-sum(abs2, delta) / 2)) / (4pi * prod(LinearAlgebra.diag(factor)))
    @test result.logweights ≈ fill(-log(density), 2)
end

@testset "LAIS moves before sampling and uses the current mixture" begin
    bank = ProposalBank([SphericalGaussian(-1.0, 1.0), SphericalGaussian(1.0, 2.0)])
    expected = [-0.5, 0.5, 0.0, 2.0, -1.5, -0.5, -1.0, 1.0]
    centres = ([0.0, 1.0], [-1.0, 0.0])
    expected_logweights = map(eachindex(expected)) do i
        x = expected[i]
        means = centres[(i - 1) ÷ 4 + 1]
        density = (exp(-abs2(x - means[1]) / 2) +
                   exp(-abs2((x - means[2]) / 2) / 2) / 2) / (2sqrt(2pi))
        -abs2(x) / 2 - log(density)
    end
    for threaded in (false, true)
        rng = LAISScriptedRNG(
            [[1.0, 1.0], [-0.5, 0.5, -0.5, 0.5],
             [-1.0, -1.0], [-0.5, 0.5, -0.5, 0.5]],
            [[0.5, 0.9], [0.5, 0.9]], 1, 1,
        )
        sampler = prepare_sampler(
            rng, x -> -abs2(x) / 2,
            LAIS(bank; transition=RandomWalkMetropolis(1.0), rounds=2, round_size=4);
            threaded,
        )
        result = importance_sample!(sampler)
        @test result.samples ≈ expected
        @test result.logweights ≈ expected_logweights
        @test result.provenance.round == [1, 1, 1, 1, 2, 2, 2, 2]
        @test result.provenance.proposal_id == [1, 1, 2, 2, 1, 1, 2, 2]
        @test [p.location for p in current_proposal(sampler).proposals] == [-1.0, 0.0]
    end
end

@testset "RAM follows probability-based updates and upper-only warmup" begin
    # Both precisions cover the factor recurrence; warmup needs only one case.
    for (T, warmup) in ((Float32, false), (Float64, false), (Float64, true))
        oracle = lais_ram_oracle(T)
        bank = ProposalBank([FactorGaussian(zeros(T, 2), Matrix{T}(LinearAlgebra.I, 2, 2))])
        target = x -> -sum(abs2, x) / T(20)
        normals = warmup ? [oracle.normals; [zeros(T, 2)]] :
            reduce(vcat, [[u, zeros(T, 2)] for u in oracle.normals])
        rng = LAISScriptedRNG(normals, [[v] for v in oracle.uniforms], 1, 1)
        tuning = warmup ? WarmupTuning(2) : ContinuousTuning()
        split_calls = T === Float64 && !warmup
        sampler = prepare_sampler(rng, target,
            LAIS(bank; transition=RAM(T[1 1; 1 2]; tuning),
                rounds=warmup || split_calls ? 1 : 3, round_size=1))
        results = [importance_sample!(sampler) for _ in 1:(split_calls ? 3 : 1)]
        expected = warmup ? oracle.centres[:, 3:3] : oracle.centres
        @test reduce(hcat, (result.samples for result in results)) ≈ expected
        @test reduce(vcat, (result.logweights for result in results)) ≈
            [target(x) + log(T(2pi)) for x in eachcol(expected)]
    end
end

@testset "LAIS reuse freezes warmup and retarget owns fresh state" begin
    scale = sqrt(1 + (1 - 0.234))
    rng = LAISScriptedRNG([fill(u, 2) for u in (1.0, 2.0, 0.0, 3.0, 0.0, -1.0, 0.0)],
        [fill(u, 2) for u in (0.1, 0.2, 0.3, 0.4)], 1, 1)
    algorithm = LAIS(ProposalBank([SphericalGaussian(0.0, 1.0), SphericalGaussian(0.0, 1.0)]);
        transition=RAM(1.0; tuning=WarmupTuning(1)), rounds=1, round_size=2)
    source = prepare_sampler(rng, x -> 0.0, algorithm)
    first = importance_sample!(source)
    second = importance_sample!(source)
    @test first.samples ≈ fill(1 + 2scale, 2)
    @test second.samples ≈ fill(1 + 5scale, 2)
    @test (first.diagnostics.target_evaluations, second.diagnostics.target_evaluations) == (8, 4)
    @test first.diagnostics.transition == (
        initial_target_evaluations=2, warmup_target_evaluations=2,
        production_target_evaluations=2, warmup_proposals=2, production_proposals=2, accepted=4)
    @test second.diagnostics.transition == (
        initial_target_evaluations=0, warmup_target_evaluations=0,
        production_target_evaluations=2, warmup_proposals=0, production_proposals=2, accepted=2)

    target = x -> -abs2(x) / 2
    newrng = LAISScriptedRNG([fill(u, 2) for u in (-1.0, -1.0, 0.5)],
        [fill(u, 2) for u in (0.1, 0.2)], 1, 1)
    changed = retarget(newrng, source, target)
    result = importance_sample!(changed)
    expected = 1 + 4scale - scale^2 + 0.5
    @test result.samples ≈ fill(expected, 2)
    @test result.logweights ≈ fill(target(expected) + 0.5^2 / 2 + log(2pi) / 2, 2)
    @test result.diagnostics.target_evaluations == 8
    @test importance_sample!(source).samples ≈ fill(1 + 4scale, 2)
end

@testset "LAIS retries from committed state after a late target failure" begin
    armed = Ref(true)
    target = x -> armed[] && x > 3 ? error("scripted target failure") : 0.0
    rng = LAISScriptedRNG([[1.0], [0.0], [2.0], [1.0], [0.0], [2.0], [0.0]],
        [[0.1], [0.2], [0.1], [0.2]], 1, 1)
    sampler = prepare_sampler(rng, target,
        LAIS(ProposalBank([SphericalGaussian(0.0, 1.0)]);
            transition=RAM(1.0; tuning=ContinuousTuning()), rounds=2, round_size=1))
    try
        importance_sample!(sampler)
    catch error
        error isa LAISRoundError || rethrow()
    end
    armed[] = false
    result = importance_sample!(sampler)
    @test result.samples ≈ [1.0, 1 + 2sqrt(1.766)]
    @test result.logweights ≈ fill(log(2pi) / 2, 2)
    @test result.diagnostics.target_evaluations == 5
end

@testset "LAIS rejects outside-support candidates without poisoning its cache" begin
    rng = LAISScriptedRNG([[1.0], [-0.5], [-1.0], [0.5]], [[0.0], [0.1]], 1, 1)
    sampler = prepare_sampler(rng, x -> x <= 0 ? 0.0 : -Inf,
        LAIS(ProposalBank([SphericalGaussian(0.0, 1.0)]);
            transition=RandomWalkMetropolis(1.0), rounds=2, round_size=1))
    result = importance_sample!(sampler)
    @test result.samples ≈ [-0.5, -0.5]
    @test result.logweights ≈ fill(0.5^2 / 2 + log(2pi) / 2, 2)
end

@testset "LAIS recovers from invalid initialization and RAM direction" begin
    ready = Ref(true)
    rng = LAISScriptedRNG([[0.0], [1.0], [0.0]], [[0.1], [0.1]], 1, 1)
    sampler = prepare_sampler(rng, x -> ready[] ? 0.0 : -Inf,
        LAIS(ProposalBank([SphericalGaussian(0.0, 1.0)]);
            transition=RAM(1.0; tuning=ContinuousTuning()), rounds=1, round_size=1))
    ready[] = false
    try
        importance_sample!(sampler)
    catch error
        error isa LAISRoundError || rethrow()
    end
    ready[] = true
    try
        importance_sample!(sampler)
    catch error
        error isa LAISRoundError || rethrow()
    end
    result = importance_sample!(sampler)
    @test result.samples ≈ [1.0]
    @test result.diagnostics.target_evaluations == 3
end

@testset "LAIS all-round raw integrals match an unequal Gaussian mixture" begin
    logtarget(x) = begin
        left = log(0.3) - log(2pi * 0.5) / 2 - abs2(x + 2) / (2 * 0.5)
        right = log(0.7) - log(2pi * 1.5) / 2 - abs2(x - 1) / (2 * 1.5)
        max(left, right) + log1p(exp(-abs(left - right)))
    end
    bank = ProposalBank([SphericalGaussian(-2.0, 1.5), SphericalGaussian(1.0, 1.5)])
    result = importance_sample!(prepare_sampler(Random.Xoshiro(302), logtarget,
        LAIS(bank; transition=RAM(0.5; tuning=ContinuousTuning()), rounds=3, round_size=4096)))
    weights = exp.(result.logweights)
    for (power, exact) in ((0, 1.0), (1, 0.1), (2, 3.1))
        values = weights .* result.samples .^ power
        uncertainty = Statistics.std(values) / sqrt(length(values))
        @test abs(Statistics.mean(values) - exact) <= 6uncertainty
    end
end
