using ImportanceSamplers, LinearAlgebra, Random, Statistics

# A curved density: X₁ = 3Z₁ and X₂ = Z₂ + 0.2(X₁² - 9), with independent
# standard normal Z₁,Z₂. The Jacobian is 3, so this log density is normalized.
function banana_logtarget(x, p)
    z1 = x[1] / p.width
    z2 = x[2] - p.bend * (abs2(x[1]) - abs2(p.width))
    return -(abs2(z1) + abs2(z2)) / 2 - log(2pi) - log(p.width)
end

function prepare_example(; seed=42, rounds=30, round_size=16_384)
    rng = Xoshiro(seed)
    data = (width=3.0, bend=0.2)
    centres = 0.5 .* randn(rng, 2, 16)

    # For ν > 2, covariance C = ν/(ν-2) * S, where S is Student-t scale.
    # If L*L' is the desired covariance, sqrt((ν-2)/ν)*L is its scale factor.
    # All coordinates share one radial draw: this is not independent univariate t's.
    nu = 5.0
    # Broad lower proposals cover the curved tails before the centres explore them.
    covariance_factor = [3.0 0.0; 0.2 2.7]
    scale_factor = sqrt((nu - 2) / nu) .* covariance_factor
    bank = ProposalBank([
        FactorStudentT(nu, copy(centre), scale_factor) for centre in eachcol(centres)
    ])

    # LAIS moves the lower Student-t centres with upper MCMC chains. Its lower
    # scale factors and ν stay fixed. Upper Gaussian move covariance is separate.
    # These illustrative settings are not a tuned optimum for this target.
    transition = RandomWalkMetropolis(Diagonal([9.0, 7.48]))
    algorithm = LAIS(bank; transition, rounds, round_size)
    return prepare_sampler(rng, banana_logtarget, data, algorithm)
end

function main()
    prepared = prepare_example()
    samples = importance_sample!(prepared)

    # Use all rounds and their importance weights, not mean(samples.samples).
    # Exact references: E[X]=[0,0], E[X₂²]=1+2*0.2²*3⁴=7.48, log(Z)=0.
    # A single run is an estimate, not an accuracy guarantee or a superiority test.
    @show length(samples) mean(samples)
    @show mean(x -> abs2(x[2]), samples) lognormalizer(samples)
    learned = current_proposal(prepared) # Independent snapshot with learned centres.
    return (; prepared, samples, learned)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

# Optional CUDA run after loading this file. Move a fresh prepared sampler, not
# only the proposal. Its target data, RNG and workspaces move together.
# using CUDA, MLDataDevices
# CUDA.allowscalar(false)
# physical = CUDA.device()
# device = MLDataDevices.CUDADevice{typeof(physical),Nothing}(physical)
# gpu_prepared = prepare_example() |> device
# gpu_samples = importance_sample!(gpu_prepared)
# mean(x -> abs2(x[2]), gpu_samples) # Scalar estimate; sample arrays stay on GPU.
# host_samples = gpu_samples |> cpu_device() # Explicit transfer, only when needed.
