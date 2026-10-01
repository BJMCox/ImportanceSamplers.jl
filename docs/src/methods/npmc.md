# [Nonlinear population Monte Carlo](@id npmc-method)

N-PMC fits one proposal from each round using clipped weights.
The returned samples keep their original importance weights.

## Example

```@example npmc
using ImportanceSamplers, Random, Statistics

logtarget(x) = -(abs2(x[1] - 1) + abs2((x[2] + 1 - 0.8(x[1] - 1)) / 0.6)) / 2
algorithm = NPMC(
    SphericalGaussian(zeros(2), 2.0);
    rounds=4, round_size=2_000,
)
prepared = prepare_sampler(Xoshiro(42), logtarget, algorithm)
samples = importance_sample!(prepared)

(mean=mean(samples), covariance=cov(samples))
```

The reference mean is `[1, -1]`.
Vector proposals learn full covariance even when their initial scale is spherical or diagonal.

## Separate fitting weights from estimator weights

For each round:

1. Draw `n` samples from the current proposal.
2. Compute raw log weights `logtarget - logproposal`.
3. Clip fitting weights at the `isqrt(n)`-th largest weight.
4. Fit a mean and covariance from that round's clipped weights.
5. Add a scale-relative covariance ridge and retain the fitted proposal.

The estimator uses the original raw weights from every round.
There is no temporal-mixture denominator or resampling step.

`diagnostics.adaptation_ess` describes the clipped fitting weights.
`diagnostics.round_ess` describes each round's raw estimator weights.
They answer different questions.

## Scope and failures

This is an adaptation variant of N-PMC.
The original transformed-weight estimator is not the estimator returned here.

Student-t fits require `nu > 2` and retain `nu`.
A round with too few nonzero weights can give a zero clipping threshold and fail adaptation.
A failed fit preserves the proposal committed before the call.

For a fixed schedule, normalized proposals, support coverage, and integrability,
the raw linear normalizer estimate avoids clipping bias.
Its logarithm and self-normalized expectations still have finite-sample bias.

See [Adaptation and reuse](@ref reuse-guide) and [Devices](@ref devices-guide).

## Sources

- Koblents and Míguez, [*A Population Monte Carlo Scheme with Transformed Weights and its Application to Stochastic Kinetic Models*](https://arxiv.org/abs/1208.5600).
- [Standalone example](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/examples/npmc.jl).
- [CUDA reproducer](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/validation/reproducers/cuda_npmc.jl).
