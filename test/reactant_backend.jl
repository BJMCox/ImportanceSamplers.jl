using ImportanceSamplers, Reactant, CUDA, Random, Test, LinearAlgebra, ADTypes

@testset "Reactant sampling and owned results" begin
    device = ImportanceSamplers.MLDataDevices.with_eltype(
        ImportanceSamplers.MLDataDevices.ReactantDevice(), nothing)
    target(theta) = -(theta.a^2 + theta.b^2)/2
    prepared = device(prepare_sampler(Xoshiro(3), target,
        ImportanceSampling(SphericalGaussian(zeros(Float32, 2), 1f0); nsamples=128);
        transform=(a=(1=>IdentityTransform()), b=(2=>IdentityTransform()))))
    first_result = importance_sample!(prepared)
    saved = Array(first_result.samples.a)
    second_result = importance_sample!(prepared)
    @test all(isapprox(log(Float32(2pi)); atol=2f-6), Array(first_result.logweights))
    @test all(isapprox(log(Float32(2pi)); atol=2f-6), Array(second_result.logweights))
    @test Array(first_result.samples.a) == saved
    @test Array(second_result.samples.a) != saved
end

@testset "Reactant stable result weights" begin
    device = ImportanceSamplers.MLDataDevices.with_eltype(
        ImportanceSamplers.MLDataDevices.ReactantDevice(), nothing)
    for shift in (-1000f0, 1000f0)
        result = device(WeightedSamples(Float32[1, 2, 3],
            Float32[-Inf, 0, log(3f0)] .+ shift))
        @test Array(normalized_weights(result)) ≈ Float32[0, 0.25, 0.75] atol=5f-5
        @test lognormalizer(result) ≈ shift + log(4f0/3f0) atol=1f-4
    end
    zero_mass = device(WeightedSamples(Float32[1, 2], fill(-Inf32, 2)))
    @test lognormalizer(zero_mass) == -Inf32
    @test_throws AllZeroWeightsError resample(Xoshiro(7), zero_mass, 16)
    point_mass = device(WeightedSamples(Float32[1, 2, 3], Float32[-Inf, 0, -Inf]))
    @test Array(normalized_weights(point_mass[2:3])) == Float32[1, 0]
    @test Array(normalized_weights(point_mass[1:2])) == Float32[0, 1]
    draws = resample(Xoshiro(7), point_mass[2:3], 16)
    @test draws.samples isa Reactant.AnyConcreteRArray && Array(draws.samples) == fill(2f0, 16)
    for index in 2:3
        mass = device(WeightedSamples(Float32[1, 2, 3], ifelse.(1:3 .== index, 0f0, -Inf32)))
        @test Array(resample(Xoshiro(7), mass, 16).samples) == fill(Float32(index), 16)
    end
end

@testset "Reactant moment and population adaptation" begin
    device = ImportanceSamplers.MLDataDevices.with_eltype(
        ImportanceSamplers.MLDataDevices.ReactantDevice(), nothing)
    proposal = FactorGaussian(zeros(Float32, 2), Float32[1.4 0; 0.3 1.2])
    bank = ProposalBank([proposal, proposal])
    target(x) = -(x[1]^2 + ((x[2] - 0.4f0*x[1])/0.8f0)^2)/2
    if Reactant.XLA.device_kind(Reactant.XLA.device(device(zeros(Float32, 1)))) != "cpu"
        for algorithm in (AMIS(proposal; rounds=3, round_size=1024),
            AMIS(proposal; rounds=2, round_size=1024),
            NPMC(proposal; rounds=3, round_size=1024),
            APIS(bank; rounds=3, round_size=1024), CAIS(bank; rounds=3, round_size=1024),
            DeterministicMixturePMC(bank; rounds=3, round_size=1024,
                resampling=LocalResampling()))
            sampler = device(prepare_sampler(Xoshiro(51), target, algorithm))
            for _ in 1:2
                result = importance_sample!(sampler)
                @test lognormalizer(result) ≈ log(2f0*Float32(pi)*0.8f0) atol=0.15
            end
        end
    else
        for algorithm in (AMIS(proposal; rounds=3, round_size=1024),
            APIS(bank; rounds=3, round_size=1024), CAIS(bank; rounds=3, round_size=1024),
            DeterministicMixturePMC(bank; rounds=3, round_size=1024,
                resampling=LocalResampling()))
            source = prepare_sampler(Xoshiro(51), target, algorithm)
            @test_throws SamplerDeviceError device(source)
            # The rejected transfer must leave the source RNG and sampler usable.
            @test importance_sample!(source).logweights == importance_sample!(
                prepare_sampler(Xoshiro(51), target, algorithm)).logweights
        end
        # Global resampling has no workgroup kernels and stays enabled.
        sampler = device(prepare_sampler(Xoshiro(51), target,
            DeterministicMixturePMC(bank; rounds=3, round_size=1024)))
        @test lognormalizer(importance_sample!(sampler)) ≈
            log(2f0*Float32(pi)*0.8f0) atol=0.15
    end
end

function reactant_named_target(theta, p)
    value = -(theta.scale^2 + theta.offset^2)/2
    for i in eachindex(theta.weights)
        value += p.alpha[i] * log(theta.weights[i])
    end
    return value
end

function reactant_named_gradient!(g, theta, p)
    for i in eachindex(theta.weights)
        g.weights[i] = p.alpha[i] / theta.weights[i]
    end
    g.scale[] = -theta.scale
    g.offset[] = -theta.offset
    return nothing
end

function reactant_named_batch!(values, samples, p)
    values .= .-(abs2.(samples.scale) .+ abs2.(samples.offset)) ./ 2 .+
              vec(sum(p.alpha .* log.(samples.weights); dims=1))
    return nothing
end

# The box is small enough that no drawn sample falls inside it.
function reactant_boxed_batch!(values, samples, p)
    reactant_named_batch!(values, samples, p)
    hit = (abs.(samples.offset .- p.offset) .< 1f-3) .& (abs.(samples.scale .- p.scale) .< 1f-3)
    values .= ifelse.(hit, NaN32, values)
    return nothing
end

@testset "Reactant GRAMIS gradients and backend safety" begin
    device = ImportanceSamplers.MLDataDevices.with_eltype(
        ImportanceSamplers.MLDataDevices.ReactantDevice(), nothing)
    bank = ProposalBank([
        FactorGaussian(zeros(Float32, 4), 0.5f0 * Matrix{Float32}(I, 4, 4)),
        FactorGaussian(fill(0.1f0, 4), 0.5f0 * Matrix{Float32}(I, 4, 4)),
    ])
    algorithm = FirstOrderGRAMIS(bank; rounds=2, round_size=64, repulsion_strength=0f0)
    layout = (weights=(1:2=>SimplexTransform(3)), scale=(3=>PositiveTransform()),
        offset=(4=>IdentityTransform()))
    automatic = LogTarget(reactant_named_target, AutoEnzyme())
    explicit = LogTarget(reactant_named_target; grad=reactant_named_gradient!)
    p = (; alpha=Float32[0.7, 1.4, 2.1])
    if Reactant.XLA.device_kind(Reactant.XLA.device(device(zeros(Float32, 1)))) == "cpu"
        @test_throws SamplerDeviceError device(prepare_sampler(
            Xoshiro(91), automatic, p, algorithm; transform=layout))
        batched = LogTarget(reactant_named_target; grad=reactant_named_gradient!,
            batch=reactant_named_batch!)
        rejection = try
            device(prepare_sampler(Xoshiro(91), batched, p, algorithm; transform=layout))
            nothing
        catch error
            error
        end
        @test rejection isa SamplerDeviceError &&
            rejection.reason === :reactant_cpu_cooperative_kernels
    else
        generated = device(prepare_sampler(Xoshiro(91), automatic, p, algorithm; transform=layout))
        reference = device(prepare_sampler(Xoshiro(91), explicit, p, algorithm; transform=layout))
        for pass in 1:3
            if pass == 3
                p2 = (; alpha=Float32[1.2, 0.7, 1.4])
                generated = retarget(Xoshiro(12), generated, automatic, p2)
                reference = retarget(Xoshiro(12), reference, explicit, p2)
            end
            actual, expected = importance_sample!(generated), importance_sample!(reference)
            @test Array(actual.logweights) ≈ Array(expected.logweights) rtol=8f-4 atol=8f-5
            @test all(keys(actual.samples)) do field
                isapprox(Array(actual.samples[field]), Array(expected.samples[field]); rtol=8f-4, atol=8f-5)
            end
        end
        # Reactant evaluates the full candidate width on every backtracking trial.
        # `algorithm` accepts every slot on trial 1, so each padded loop runs once.
        # In `wide` the wide proposal overshoots. Its slot exhausts four trials in
        # round 1 and accepts late in round 2, while the narrow slot accepts on trial 1.
        batched = LogTarget(reactant_named_target; grad=reactant_named_gradient!,
            batch=reactant_named_batch!)
        wide = FirstOrderGRAMIS(ProposalBank([
            FactorGaussian(zeros(Float32, 4), 0.5f0 * Matrix{Float32}(I, 4, 4)),
            FactorGaussian(fill(0.1f0, 4), 3f0 * Matrix{Float32}(I, 4, 4)),
        ]); rounds=2, round_size=64, repulsion_strength=0f0, max_backtracking_trials=4)
        for bank_algorithm in (algorithm, wide)
            padded = device(prepare_sampler(Xoshiro(91), batched, p, bank_algorithm; transform=layout))
            scalar = device(prepare_sampler(Xoshiro(91), explicit, p, bank_algorithm; transform=layout))
            for _ in 1:2
                actual, expected = importance_sample!(padded), importance_sample!(scalar)
                @test Array(actual.logweights) ≈ Array(expected.logweights) rtol=8f-4 atol=8f-5
                @test all(keys(actual.samples)) do field
                    isapprox(Array(actual.samples[field]), Array(expected.samples[field]); rtol=8f-4, atol=8f-5)
                end
                @test Array(actual.diagnostics.accepted_steps) == Array(expected.diagnostics.accepted_steps)
                padded_trials = Array(actual.diagnostics.backtracking_trials)
                scalar_trials = Array(expected.diagnostics.backtracking_trials)
                @test padded_trials == repeat(maximum(scalar_trials; dims=1), size(scalar_trials, 1))
                @test actual.diagnostics.target_evaluations - expected.diagnostics.target_evaluations ==
                    sum(padded_trials) - sum(scalar_trials)
            end
        end
        # A clean run gives the round-2 trial-2 candidate of the wide slot. The
        # narrow slot is inactive on that trial, so only the wide slot may fail.
        cuda = ImportanceSamplers.MLDataDevices.with_eltype(
            ImportanceSamplers.MLDataDevices.CUDADevice(), nothing)
        failures = map((device, cuda)) do backend
            clean = backend(prepare_sampler(Xoshiro(91), batched, p, wide; transform=layout))
            steps = Array(importance_sample!(clean).diagnostics.accepted_steps)
            @test steps[1, 2] == 1 && steps[2, 1] == 0 && steps[2, 2] <= 0.25f0
            # Slot 2 exhausts round 1, so round 2 starts from its initial location.
            start = fill(0.1f0, 4)
            accepted = current_proposal(ImportanceSamplers.MLDataDevices.cpu_device(), clean).proposals[2].location
            candidate = start .+ (accepted .- start) ./ 2steps[2, 2]
            boxed = LogTarget(reactant_named_target; grad=reactant_named_gradient!,
                batch=reactant_boxed_batch!)
            context = (; p.alpha, offset=candidate[4], scale=exp(candidate[3]))
            sampler = backend(prepare_sampler(Xoshiro(91), boxed, context, wide; transform=layout))
            try
                importance_sample!(sampler)
                nothing
            catch error
                error
            end
        end
        @test all(failures) do failure
            failure isa FirstOrderGRAMISRoundError && failure.round == 2 && failure.phase === :derivative
        end
        @test isequal(failures[1].diagnostics.derivative, failures[2].diagnostics.derivative)
        @test failures[1].diagnostics.derivative.proposal_slot == 2
        @test failures[1].diagnostics.derivative.reason === :candidate_value_nonfinite
        @test isnan(failures[1].diagnostics.derivative.value)
    end
end
