# [Adaptive multiple importance sampling](@id amis-method)

AMIS fits one proposal using all retained samples.
Each new generating proposal changes the weights of both new and earlier samples.

## Example

```@example amis
using ImportanceSamplers, Random, Statistics

function logtarget(x)
    a = x[1] - 1
    b = (x[2] + 1 - 0.8a) / 0.6
    return -(a^2 + b^2) / 2
end

algorithm = AMIS(
    SphericalStudentT(5.0, zeros(2), 2.0);
    rounds=4, round_size=2_000,
)
prepared = prepare_sampler(Xoshiro(42), logtarget, algorithm)
samples = importance_sample!(prepared)

(count=length(samples), mean=mean(samples), covariance=cov(samples))
```

The reference mean is `[1, -1]`.
The initial proposal is spherical, but adaptation learns a full covariance.

`round_size` is a total count per round.
Use `round_size=[1_000, 2_000, 3_000, 4_000]` for unequal counts.

## Understand one round

1. Draw new samples from the current proposal.
2. Evaluate their target values.
3. Update every retained sample's temporal-mixture denominator.
4. Fit a weighted mean and covariance using all retained samples.
5. Retain the fitted proposal for the next round or prepared call.

Target values are not reevaluated for earlier samples.
Student-t adaptation keeps the degrees of freedom fixed and requires `nu > 2`.
The fitted covariance is converted to Student-t scale.

## Retrospective denominator

After round ``t``, with counts ``N_1,\ldots,N_t``, the denominator is

```math
\psi_t(x)=\frac{\sum_{s=1}^tN_sq_s(x)}{\sum_{s=1}^tN_s},
\qquad
\ell_i^{(t)}=\log\pi(x_i)-\log\psi_t(x_i).
```

Counts determine the mixture coefficients.
The final fitted proposal generated no returned draw and does not enter that denominator.

The result includes every round under the final retrospective weights.
`diagnostics.round_ess` and `round_lognormalizers` summarize cumulative history at each round,
not each round in isolation.

## Cost and interpretation

AMIS retains samples and proposal history.
For `T` rounds and `N` total draws, it performs `N` target evaluations and `T * N` proposal-density evaluations.
It accumulates denominators without storing a dense sample-by-proposal density matrix.

The package makes no general finite-sample unbiasedness or consistency claim for adaptive AMIS.
The related MAMIS consistency results concern a different learning rule.
Weight ESS still measures concentration, not proof of tail coverage.

A successful call retains its final fitted proposal.
A failed call preserves the previously committed proposal and leaves the RNG advanced.
See [Adaptation and reuse](@ref reuse-guide).

## Sources

- Cornuet et al., [*Adaptive Multiple Importance Sampling*](https://arxiv.org/abs/0907.1254).
- Marin et al., [modified AMIS consistency analysis](https://arxiv.org/abs/1211.2548).
- [Equation reproducer](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/validation/reproducers/amis.jl).
- [Standalone example](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/examples/amis.jl).
