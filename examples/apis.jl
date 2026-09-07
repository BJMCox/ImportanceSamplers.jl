using ImportanceSamplers
using Random
using Statistics

# A two-mode unnormalized target. APIS receives log densities directly.
function logtarget(x)
    left = -abs2((x + 3) / 0.7) / 2
    right = -abs2((x - 2) / 1.2) / 2
    largest = max(left, right)
    return largest + log(exp(left - largest) + exp(right - largest))
end

# Equal masses give the paper's equal population mixture. Proposal scales stay
# fixed; only their means are learned from proposal-local epoch samples.
bank = ProposalBank([
    SphericalGaussian(-5.0, 1.5),
    SphericalGaussian( 5.0, 1.5),
])
algorithm = APIS(bank; rounds=4, round_size=4_000)
prepared = prepare_sampler(Xoshiro(42), logtarget, algorithm)
samples = importance_sample!(prepared)
learned = current_proposal(prepared)

@show mean(samples)
@show lognormalizer(samples)
@show [proposal.location for proposal in learned.proposals]

# `samples.logweights` retain logtarget-log(epoch mixture) for estimation.
# The learned means instead used logtarget-log(generating proposal), normalized
# separately inside each proposal group. The final fit is retained for reuse.
next_samples = importance_sample!(prepared)
@show mean(next_samples)
