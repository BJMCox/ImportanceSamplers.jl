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
    end
end
