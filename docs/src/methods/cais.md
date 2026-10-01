# [Covariance-adaptive importance sampling](@id cais-method)

CAIS fits proposal means and covariances from proposal-local samples.
It tempers fitting weights when their concentration would make covariance estimation unreliable.

## Example

```@example cais
using ImportanceSamplers, Random, Statistics

logtarget(x) = -(abs2(x[1] - 1) + abs2((x[2] + 1 - 0.8(x[1] - 1)) / 0.6)) / 2
bank = ProposalBank([
    SphericalGaussian([-1.0, 0.0], 2.0),
    SphericalGaussian([ 1.0, 0.0], 2.0),
])
prepared = prepare_sampler(
    Xoshiro(42), logtarget,
    CAIS(bank; rounds=4, round_size=2_000),
)
samples = importance_sample!(prepared)

(mean=mean(samples), covariance=cov(samples))
```

The reference mean is `[1, -1]`.
Vector proposals learn a full covariance.

Use equal positive bank masses and a divisible round size.
Every proposal needs at least `d + 2` samples per round in dimension `d`.

## Estimator and adaptation

Returned weights use the **generating proposal**, not a full mixture:

```math
\ell_{t,j,i}=\log\pi(x_{t,j,i})-\log q_{t,j}(x_{t,j,i}).
```

The next mean always uses the normalized raw local weights.
The covariance rule depends on their local ESS:

| Local weight ESS | Covariance weights | Covariance centre |
|:--|:--|:--|
| At least the threshold | Raw normalized weights | Old proposal mean |
| Below the threshold | Power-tempered weights | Tempered weighted mean |

The second branch changes the covariance fit, not the returned estimator weights or next mean.

## Set the covariance threshold

```julia
algorithm = CAIS(
    bank; rounds=4, round_size=2_000,
    covariance_ess_threshold=300,
)
```

For local count `m`, an explicit threshold must satisfy `d < threshold < m`.
The default is `max(d + 1, ceil(Int, 0.3m))`.
That fraction is a package default, not a universal paper prescription.

`diagnostics.local_ess` records the weights used for the covariance fit.
`diagnostics.tempering_powers` distinguishes raw from tempered fitting.

## Exact replacement and failures

CAIS replaces the covariance rather than blending it.
It adds no ridge, eigenvalue floor, or silent old-factor fallback.

Enough samples and sufficient ESS do not guarantee a positive-definite covariance.
All-zero local weights, failed tempering, or a failed factorization abort the call.
The pre-call population remains committed.

Student-t fits require `nu > 2` and convert fitted covariance to Student-t scale.
The degrees of freedom stay fixed.

## Sources

- El-Laham, Elvira, and Bugallo, [*Robust Covariance Adaptation in Adaptive Importance Sampling*](https://arxiv.org/abs/1806.00093), Section III-B.
- [Standalone example](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/examples/cais.jl).
- [CUDA reproducer](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/validation/reproducers/cuda_cais.jl).

See [Adaptation and reuse](@ref reuse-guide) and [Devices](@ref devices-guide).
