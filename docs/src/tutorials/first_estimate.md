# [Your first weighted estimate](@id first-estimate)

This tutorial estimates the moments of a three-dimensional Gaussian.
It introduces the same workflow used by every sampler.

## 1. Define the log target

```@example first
using ImportanceSamplers, Random, Statistics

logtarget(x) = -sum(abs2, x) / 2
nothing # hide
```

Here `x` is the vector being sampled. The function returns
``\log\pi(x)=-\lVert x\rVert^2/2``, not ``\pi(x)``.
The corresponding normalized target is a standard Gaussian in three dimensions.

The omitted normalizing constant does not affect weighted means or variances.

## 2. Choose a proposal and draw samples

```@example first
proposal = SphericalGaussian(zeros(3), 1.5)
algorithm = ImportanceSampling(proposal; nsamples=20_000)
samples = importance_sample(Xoshiro(42), logtarget, algorithm)

(size(samples.samples), length(samples.logweights))
```

The proposal determines the sample dimension. Its second argument is a standard
deviation, not a variance. This proposal is wider than the target.

`Xoshiro(42)` fixes the random stream for this execution environment.
`nsamples` belongs to the algorithm and specifies the exact output count.

The result stores one sample per column: `samples.samples[:, i]` is the
three-dimensional draw at index `i`.

## 3. Compute weighted summaries

```@example first
(
    mean=round.(mean(samples); digits=3),
    variance=round.(var(samples); digits=3),
    covariance=round.(cov(samples); digits=3),
)
```

The reference mean is `zeros(3)`. The reference covariance is the identity matrix.
The estimates differ because the run has finite size.

These methods use normalized importance weights. In contrast,
`mean(samples.samples; dims=2)` ignores the weights and describes the proposal draws.

Functions of the sampled parameters use the same interface:

```@example first
(
    squared_radius=mean(x -> sum(abs2, x), samples),
    first_coordinate_second_moment=mean(x -> x[1]^2, samples),
)
```

The reference values are three and one, respectively.

## 4. Inspect the weights

```@example first
weights = normalized_weights(samples)
weight_ess = inv(sum(abs2, weights))
(weight_sum=sum(weights), weight_ess=weight_ess, draws=length(samples))
```

The raw log weight is

```math
\ell_i=\log\pi(x_i)-\log q(x_i).
```

`samples.logweights` stores these values. `normalized_weights` returns a new array
``\bar w_i=\exp(\ell_i)/\sum_j\exp(\ell_j)`` without changing the raw log weights.

Weight ESS measures concentration. It lies between one and the draw count when
the weights are valid. It does not test whether important regions were missed.

## 5. Estimate the normalizing constant

```@example first
(
    estimate=lognormalizer(samples),
    reference=3log(2pi) / 2,
)
```

`lognormalizer` estimates the logarithm of ``Z=\int\pi(x)\,dx``.
Here ``Z=(2\pi)^{3/2}``, since the target omitted the Gaussian normalization.

Adding a constant to the log target adds that constant to this result.
For Bayesian evidence, retain all prior and likelihood normalization constants.

## Next step

[Logistic regression](@ref logistic-tutorial) adds observed data and an adaptive proposal.
[Working with results](@ref results-guide) covers indexing, intervals, resampling, and diagnostics.
