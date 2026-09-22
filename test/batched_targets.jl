using Test, Random, Statistics

struct BatchOverflowNormals <: Random.AbstractRNG end
Random.randn!(::BatchOverflowNormals, destination::AbstractArray) = fill!(destination, 2)

@testset "failed draws never reach a batch callback" begin
    calls = Ref(0)
    scalar(x) = 0.0
    batch!(out, xs) = (calls[] += 1; fill!(out, 0.0))
    algorithm = ImportanceSampling(SphericalGaussian(0.0, floatmax(Float64)); nsamples=4)
    @test_throws SamplerExecutionError importance_sample(
        BatchOverflowNormals(), LogTarget(scalar; batch=batch!), algorithm; threaded=false)
    @test calls[] == 0
end

@testset "invalid backtracking values end the batch sequence" begin
    small_calls = Ref(0)
    scalar(x) = -sum(abs2, x) / 2
    grad!(g, x) = (g .= .-x)
    function batch!(out, xs)
        out .= vec(-sum(abs2, xs; dims=1)) ./ 2
        if length(out) <= 2
            small_calls[] += 1
            if small_calls[] == 2
                out .= [NaN, -Inf]
            elseif small_calls[] > 2
                error("a later callback must not mask the invalid batch")
            end
        end
    end
    bank = ProposalBank([FactorGaussian([x, -x], [2.0 0.0; 0.2 1.7]) for x in (-0.7, 0.7)])
    algorithm = FirstOrderGRAMIS(bank; rounds=1, round_size=64, repulsion_strength=0.0)
    @test_throws FirstOrderGRAMISRoundError importance_sample(Xoshiro(31),
        LogTarget(scalar; grad=grad!, batch=batch!), algorithm; threaded=false)
    @test small_calls[] == 2
end

@testset "batch target values preserve the sampling law" begin
    scalar(x, p) = -sum(abs2, x .- p) / 2
    widths = Int[]
    function batch!(out, xs, p)
        push!(widths, size(xs, 2))
        out .= vec(-sum(abs2, xs .- p; dims=1)) ./ 2
    end
    algorithm = ImportanceSampling(TestVectorProposal(zeros(2)); nsamples=32)
    context = [0.2, -0.3]
    expected = importance_sample(Xoshiro(12), scalar, context, algorithm; threaded=false)
    target = LogTarget(scalar; batch=batch!)
    actual = importance_sample(Xoshiro(12), target, context, algorithm; threaded=true)
    @test actual.samples == expected.samples
    @test actual.logweights ≈ expected.logweights
    @test widths == [32]
end

@testset "failed batch evaluation preserves the adapted proposal" begin
    scalar(x, p) = -sum(abs2, x .- p.centre) / 2
    function batch!(out, xs, p)
        out .= vec(-sum(abs2, xs .- p.centre; dims=1)) ./ 2
        p.fail[] && (out[end] = NaN)
    end
    p = (centre=[0.1, -0.2], fail=Ref(false))
    algorithm = AMIS(FactorGaussian(zeros(2), [1.5 0.0; 0.2 1.2]); rounds=2, round_size=64)
    sampler = prepare_sampler(Xoshiro(22), LogTarget(scalar; batch=batch!), p, algorithm; threaded=false)
    previous = importance_sample!(sampler)
    saved_samples, saved_weights = copy(previous.samples), copy(previous.logweights)
    proposal = current_proposal(sampler)
    p.fail[] = true
    @test_throws AMISRoundError importance_sample!(sampler)
    after = current_proposal(sampler)
    @test after.location == proposal.location && after.scale.factor == proposal.scale.factor
    p.fail[] = false
    @test all(isfinite, importance_sample!(sampler).logweights)
    @test previous.samples == saved_samples && previous.logweights == saved_weights
end

@testset "GRAMIS batches values independently of explicit derivatives" begin
    scalar_calls = Ref(0)
    function scalar(x)
        scalar_calls[] += 1
        return -sum(abs2, x) / 2
    end
    grad!(g, x) = (g .= .-x)
    widths = Int[]
    function batch!(out, xs)
        push!(widths, size(xs, 2))
        out .= vec(-sum(abs2, xs; dims=1)) ./ 2
    end
    bank = ProposalBank([FactorGaussian([x, -x], [2.0 0.0; 0.2 1.7]) for x in (-0.7, 0.7)])
    algorithm = FirstOrderGRAMIS(bank; rounds=2, round_size=64, repulsion_strength=0.0)
    expected = importance_sample(Xoshiro(31), LogTarget(scalar; grad=grad!), algorithm; threaded=false)
    scalar_calls[] = 0
    actual = importance_sample(Xoshiro(31), LogTarget(scalar; grad=grad!, batch=batch!), algorithm; threaded=false)
    @test actual.samples ≈ expected.samples
    @test actual.logweights ≈ expected.logweights
    @test scalar_calls[] == 0
    @test count(<=(2), widths) >= 4
end

@testset "LAIS batches dependent moves without changing their order" begin
    scalar(x) = -sum(abs2, x) / 2
    widths = Int[]
    function batch!(out, xs)
        push!(widths, size(xs, 2))
        out .= vec(-sum(abs2, xs; dims=1)) ./ 2
    end
    bank = ProposalBank([SphericalGaussian([x, -x], 1.0) for x in (-0.5, 0.5)])
    for transition in (RandomWalkMetropolis(0.3), RAM(0.3; tuning=WarmupTuning(3)),
        SampleMetropolisHastings(SphericalGaussian(zeros(2), 1.5); moves=3))
        algorithm = LAIS(bank; transition, rounds=2, round_size=32)
        a = prepare_sampler(Xoshiro(24), scalar, algorithm; threaded=false)
        b = prepare_sampler(Xoshiro(24), LogTarget(scalar; batch=batch!), algorithm; threaded=false)
        for _ in 1:2
            empty!(widths)
            expected, actual = importance_sample!(a), importance_sample!(b)
            @test actual.samples ≈ expected.samples
            @test actual.logweights ≈ expected.logweights
            @test actual.diagnostics.transition == expected.diagnostics.transition
            @test count(!=(32), widths) > 0
        end
    end
end

@testset "batch callbacks receive logical named leaves" begin
    scalar(x, p) = sum(log, x.mixture) - x.scale / p - abs2(x.offset) / 2
    function batch!(out, xs, p)
        out .= vec(sum(log, xs.mixture; dims=1)) .- xs.scale ./ p .- abs2.(xs.offset) ./ 2
    end
    transform = (mixture=1:2 => SimplexTransform(3), scale=3 => PositiveTransform(), offset=4 => IdentityTransform())
    algorithm = ImportanceSampling(SphericalGaussian(zeros(4), 0.5); nsamples=32)
    expected = importance_sample(Xoshiro(18), scalar, 2.0, algorithm; transform, threaded=false)
    actual = importance_sample(Xoshiro(18), LogTarget(scalar; batch=batch!), 2.0,
        algorithm; transform, threaded=false)
    @test actual.samples == expected.samples
    @test actual.logweights ≈ expected.logweights
end

@testset "native batch values preserve shared round laws" begin
    scalar(x) = -sum(abs2, x) / 2
    widths = Int[]
    function batch!(out, xs)
        push!(widths, size(xs, 2))
        out .= vec(-sum(abs2, xs; dims=1)) ./ 2
    end
    proposal = FactorGaussian([0.2, -0.3], [1.3 0.0; 0.2 1.1])
    bank = ProposalBank([proposal, FactorGaussian([-0.2, 0.3], [1.1 0.0; -0.2 1.3])])
    algorithms = (
        (ImportanceSampling(proposal; nsamples=64), [64]),
        (ImportanceSampling(bank; nsamples=64), [64]),
        (APIS(bank; rounds=2, round_size=64), [64, 64]),
        (AMIS(proposal; rounds=2, round_size=[64, 96]), [64, 96]),
        (NPMC(proposal; rounds=2, round_size=64), [64, 64]),
    )
    for (algorithm, expected_widths) in algorithms
        empty!(widths)
        expected = importance_sample(Xoshiro(14), scalar, algorithm; threaded=false)
        actual = importance_sample(Xoshiro(14), LogTarget(scalar; batch=batch!),
            algorithm; threaded=false)
        @test actual.samples ≈ expected.samples
        @test actual.logweights ≈ expected.logweights
        @test widths == expected_widths
    end
end
