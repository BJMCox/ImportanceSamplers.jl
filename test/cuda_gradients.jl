using Test, CUDA, Enzyme, ImportanceSamplers, Random, LinearAlgebra
import ADTypes, MLDataDevices

CUDA.allowscalar(false)

function cuda_gradient_logtarget(x, p)
    value = zero(eltype(x))
    for i in eachindex(x)
        z = x[i] - p.center[i]
        value -= z^2 / 2 + z^4 / 4
    end
    return value
end

function cuda_gradient!(g, x, p)
    for i in eachindex(x)
        z = x[i] - p.center[i]
        g[i] = -z - z^3
    end
    return g
end

cuda_gradient_logtarget(x) = cuda_gradient_logtarget(x, (; center=(0f0, 0f0)))
cuda_gradient!(g, x) = cuda_gradient!(g, x, (; center=(0f0, 0f0)))

function cuda_named_logtarget(theta, p)
    value = -p.rate * theta.scale - theta.offset^2 / 2
    for i in eachindex(theta.weights)
        value += p.alpha[i] * log(theta.weights[i])
    end
    return value
end

function cuda_named_gradient!(g, theta, p)
    for i in eachindex(theta.weights)
        g.weights[i] = p.alpha[i] / theta.weights[i]
    end
    g.scale[] = -p.rate
    g.offset[] = -theta.offset
    return nothing
end

@testset "CUDA named gradients retain the transform through reuse and retarget" begin
    device = MLDataDevices.CUDADevice{typeof(CUDA.device()),Nothing}(CUDA.device())
    bank = ProposalBank([
        FactorGaussian(zeros(Float32, 4), 0.5f0 * Matrix{Float32}(I, 4, 4)),
        FactorGaussian(fill(0.1f0, 4), 0.5f0 * Matrix{Float32}(I, 4, 4)),
    ])
    algorithm = FirstOrderGRAMIS(bank; rounds=2, round_size=192, repulsion_strength=0f0)
    layout = (weights=(1:2=>SimplexTransform(3)), scale=(3=>PositiveTransform()),
        offset=(4=>IdentityTransform()))
    ad = ADTypes.AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
    automatic = LogTarget(cuda_named_logtarget, ad)
    explicit = LogTarget(cuda_named_logtarget; grad=cuda_named_gradient!)
    p = (; alpha=Float32[0.7, 1.4, 2.1], rate=0.8f0)
    generated = device(prepare_sampler(Xoshiro(91), automatic, p, algorithm; transform=layout))
    reference = device(prepare_sampler(Xoshiro(91), explicit, p, algorithm; transform=layout))
    for pass in 1:3
        if pass == 3
            p2 = (; alpha=Float32[1.2, 0.7, 1.4], rate=1.3f0)
            generated = retarget(Xoshiro(12), generated, automatic, p2)
            reference = retarget(Xoshiro(12), reference, explicit, p2)
        end
        actual, expected = importance_sample!(generated), importance_sample!(reference)
        for field in keys(actual.samples)
            @test Array(actual.samples[field]) ≈ Array(expected.samples[field]) rtol=8f-4 atol=8f-5
        end
        @test Array(actual.logweights) ≈ Array(expected.logweights) rtol=8f-4 atol=8f-5
    end
end

@testset "CUDA automatic gradients preserve adaptation and retargeting ($T)" for T in (Float32, Float64)
    device = MLDataDevices.CUDADevice{typeof(CUDA.device()),Nothing}(CUDA.device())
    bank = ProposalBank([
        FactorGaussian(T[-1, 0], T[1 0; 0.2 1]),
        FactorGaussian(T[1, 0], T[1 0; -0.2 1]),
    ])
    algorithm = FirstOrderGRAMIS(bank; rounds=3, round_size=192, repulsion_strength=0f0)
    ad = ADTypes.AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
    automatic = LogTarget(cuda_gradient_logtarget, ad)
    explicit = LogTarget(cuda_gradient_logtarget, ad; grad=cuda_gradient!)
    context = (; center=T[0.25, -0.5])
    rtol, atol = T === Float32 ? (5f-4, 5f-5) : (5e-11, 5e-12)
    generated = device(prepare_sampler(Xoshiro(123), automatic, context, algorithm))
    reference = device(prepare_sampler(Xoshiro(123), explicit, context, algorithm))
    for _ in 1:2
        actual, expected = importance_sample!(generated), importance_sample!(reference)
        @test Array(actual.samples) ≈ Array(expected.samples) rtol=rtol atol=atol
        @test Array(actual.logweights) ≈ Array(expected.logweights) rtol=rtol atol=atol
    end
    shifted = (; center=T[-0.5, 0.75])
    new_generated = retarget(Xoshiro(456), generated, automatic, shifted)
    new_reference = retarget(Xoshiro(456), reference, explicit, shifted)
    actual, expected = importance_sample!(new_generated), importance_sample!(new_reference)
    @test Array(actual.samples) ≈ Array(expected.samples) rtol=rtol atol=atol
    @test Array(actual.logweights) ≈ Array(expected.logweights) rtol=rtol atol=atol
    @test context.center == T[0.25, -0.5]
    @test shifted.center == T[-0.5, 0.75]
    no_context = retarget(Xoshiro(789), generated, automatic)
    no_context_reference = retarget(Xoshiro(789), reference, explicit)
    actual, expected = importance_sample!(no_context), importance_sample!(no_context_reference)
    @test Array(actual.samples) ≈ Array(expected.samples) rtol=rtol atol=atol
    @test Array(actual.logweights) ≈ Array(expected.logweights) rtol=rtol atol=atol
    forward = LogTarget(cuda_gradient_logtarget, ADTypes.AutoEnzyme(; mode=Enzyme.Forward))
    @test_throws SamplerDeviceError device(prepare_sampler(Xoshiro(42), forward, context, algorithm))
end
