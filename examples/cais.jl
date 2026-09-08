using ImportanceSamplers
using LinearAlgebra
using Random
using Statistics

# A correlated, unnormalized two-dimensional target. CAIS consumes log
# densities directly and does not apply `log` to this return value.
function logtarget(x)
    centered = x .- [0.75, -0.5]
    factor = [1.0 0.0; 0.65 0.8]
    standardized = factor \ centered
    return -sum(abs2, standardized) / 2
end

# Equal masses and allocation are part of the canonical CAIS contract.
# Vector proposals are represented by full factors after covariance fitting.
proposals = [
    SphericalGaussian([-3.0, 1.0], 1.8),
    DiagonalGaussian([3.0, -1.0], [1.2, 2.0]),
]
bank = ProposalBank(proposals, [1.0, 1.0])
algorithm = CAIS(
    bank;
    rounds=4,
    round_size=4_000,
    covariance_ess_threshold=500,
)
prepared = prepare_sampler(Xoshiro(42), logtarget, algorithm)
samples = importance_sample!(prepared)
learned = current_proposal(prepared)

@show mean(samples)
@show cov(samples)
@show lognormalizer(samples)
@show [proposal.location for proposal in learned.proposals]
@show [proposal.scale.factor for proposal in learned.proposals]
@show samples.diagnostics.tempering_powers

# Returned log weights always use the proposal that generated each sample:
# logtarget(x) - log(q_generating(x)). Power tempering is restricted to the
# covariance fit. In a low-ESS group its transformed weighted mean centers the
# covariance, while the retained next proposal mean still uses raw weights.
# In a high-ESS group the covariance is centered at the old generating mean.
# The successful final-round fit is committed for the next prepared call.
next_samples = importance_sample!(prepared)
@show mean(next_samples)
