# [DM-PMC, GR-PMC, and LR-PMC](@id dmpmc-method)

DM-PMC moves proposal centres by resampling completed rounds.
It preserves each proposal's scale and uses a current-population mixture denominator.

## Example

```@example dmpmc
using ImportanceSamplers, Random, Statistics

logtarget(x) = -sum(abs2, x) / 2
bank = ProposalBank([
    FactorGaussian([-2.0, 0.0], [1.0 0.0; 0.4 0.8]),
    FactorGaussian([ 2.0, 0.0], [1.0 0.0; 0.4 0.8]),
])
algorithm = DeterministicMixturePMC(
    bank; rounds=4, round_size=2_000,
    resampling=LocalResampling(),
)
prepared = prepare_sampler(Xoshiro(42), logtarget, algorithm)
samples = importance_sample!(prepared)

(mean=mean(samples), count=length(samples))
```

This equal-mass, equal-allocation configuration uses local resampling, the LR-PMC case.
Replace `LocalResampling()` with `GlobalResampling()` for the GR-PMC case.
Global resampling is the default.

## Choose the resampling scope

| Policy | Source of the next centre in each slot |
|:--|:--|
| `GlobalResampling()` | The complete weighted round |
| `LocalResampling()` | That proposal's own weighted sample group |

Both policies use the same complete-mixture weights.
Local resampling preserves one descendant per proposal.
Global resampling can select duplicate ancestors and reduce population diversity.

The selected sample becomes the next proposal location.
The original scale or factor remains unchanged.
Resampling does **not** tune proposal widths.

## Allocation and weights

Equal masses and divisible round sizes give equal allocation.
For unequal masses, the package uses largest-remainder allocation with rotated ties.
Every active proposal needs at least one draw.

For realized counts ``n_{t,j}`` and total ``N_t``, the denominator is

```math
\psi_t(x)=\sum_j\frac{n_{t,j}}{N_t}q_{t,j}(x),
\qquad
\ell_{t,i}=\log\pi(x_{t,i})-\log\psi_t(x_{t,i}).
```

The coefficients are realized count fractions, not nominal masses.
Unequal allocation is a package extension to the equal-allocation paper cases.

Each sample keeps its own round's weight.
All rounds contribute to the returned result.

## Reuse and limitations

The last resampling step runs after the final returned round.
Its centres become the starting population for the next prepared call.

Gaussian and Student-t proposals retain their original scales and Student-t degrees of freedom.
Good centres cannot compensate for unsuitable scales.

An all-zero round fails.
Local resampling also fails when one proposal's sample group has no nonzero weight.
A failure preserves the pre-call population, but not the old RNG position.

The linear normalizer estimate is unbiased for a fixed schedule when each conditional
mixture covers the integrable target and adaptation uses only completed rounds.
This does not make the logarithm or normalized expectations unbiased.

## Sources

- Elvira et al., [*Improving Population Monte Carlo: Alternative Weighting and Resampling Schemes*](https://victorelvira.github.io/assets/papers/elvira2017improving_pre.pdf).
- [Equation reproducer](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/validation/reproducers/dm_pmc_global.jl).
- [Standalone example](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/examples/dm_pmc.jl).

See [Adaptation and reuse](@ref reuse-guide) and [Devices](@ref devices-guide).
