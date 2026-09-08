using ImportanceSamplers, Random, Statistics

# A normalized two-dimensional mixture: unequal masses, correlated coordinates.
# Each component has covariance [1 0.6; 0.6 1.36]. Lower proposals are broader
# to cover both modes even when some upper chains drift into the same mode.
function logtarget(x)
    left = log(0.3) - ((x[1] + 2)^2 + (x[2] - 0.6(x[1] + 2))^2) / 2
    right = log(0.7) - ((x[1] - 2)^2 + (x[2] - 0.6(x[1] - 2))^2) / 2
    largest = max(left, right)
    return largest + log1p(exp(min(left, right) - largest)) - log(2pi)
end

bank = ProposalBank([
    FactorGaussian([location, offset], [1.5 0.0; 0.9 1.5])
    for location in (-2.0, 2.0) for offset in (-1.0, -0.3, 0.3, 1.0)
])

# Upper MCMC chains move the lower proposal centres. RAM tunes the upper
# covariance only. The lower factors above stay fixed throughout the run.
# These RAM settings, including acceptance/decay defaults, illustrate the API.
# They are not benchmark-selected defaults.
algorithm = LAIS(bank;
    transition=RAM(1.0; tuning=WarmupTuning(100)),
    rounds=20, round_size=10_000,
)
prepared = prepare_sampler(Xoshiro(42), logtarget, algorithm)
samples = importance_sample!(prepared)

# All 200,000 lower samples contribute. Upper-chain states are not returned as
# extra samples. Self-normalized expectations use the raw importance weights.
@show mean(samples) # Exact mean: [0.8, 0.0].
@show mean(x -> x[1]^2, samples) # Exact second moment: 5.
@show lognormalizer(samples) # Exact value: 0.
learned = current_proposal(prepared)
@show [proposal.location for proposal in learned.proposals]

# Reuse keeps learned centres/factors and does not repeat completed warmup.
# next_samples = importance_sample!(prepared)

# Optional CUDA execution: transfer a fresh complete prepared sampler explicitly.
# using CUDA, MLDataDevices
# physical = CUDA.device()
# device = MLDataDevices.CUDADevice{typeof(physical),Nothing}(physical)
# gpu_prepared = device(prepare_sampler(Xoshiro(43), logtarget, algorithm))
# gpu_samples = importance_sample!(gpu_prepared)
# host_samples = cpu_device()(gpu_samples)
# Pass target data as prepare_sampler(rng, logtarget, data, algorithm), so data
# transfers with the prepared sampler. Avoid captured CPU arrays in GPU targets.
