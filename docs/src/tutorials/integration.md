# [Numerical integration](@id integration-tutorial)

Importance sampling also estimates ordinary integrals.
No Bayesian model or observed data is required.

Consider

```math
I=\int_{\mathbb R^3}\exp(-\lVert x\rVert^2)\,dx=\pi^{3/2}.
```

## Integrate a nonnegative function

```@example integration
using ImportanceSamplers, Random, Statistics

logintegrand(x) = -sum(abs2, x)
proposal = SphericalGaussian(zeros(3), 1.0)
samples = importance_sample(
    Xoshiro(8), logintegrand,
    ImportanceSampling(proposal; nsamples=20_000),
)

(estimate=exp(lognormalizer(samples)), reference=pi^(3/2))
```

The log target is the logarithm of the integrand.
The normalized proposal supplies draws over the integration domain.

For plain IS, the estimate is the average of ``f(x_i)/q(x_i)``.
The package stores this ratio in log form.

## Integrate a signed function

A signed integrand has no real logarithm everywhere.
Choose a positive envelope ``g`` and write the integrand as ``g(x)h(x)`` instead.

The same samples estimate

```math
J=\int_{\mathbb R^3}e^{-\lVert x\rVert^2}\cos(x_1)\,dx
=\pi^{3/2}e^{-1/4}.
```

```@example integration
estimate = exp(lognormalizer(samples)) * mean(x -> cos(x[1]), samples)
(estimate=estimate, reference=pi^(3/2) * exp(-1/4))
```

Here `logintegrand` defines ``\log g`` and the function passed to `mean` defines ``h``.
The product equals the ordinary weighted average
``n^{-1}\sum_i g(x_i)h(x_i)/q(x_i)``.

This estimate can have either sign. It needs proposal support wherever ``g h`` contributes
and finite variance for conventional Monte Carlo error estimates.

## Integrate over a constrained domain

A proposal may live directly on the domain, or a transform may map unconstrained draws into it.
The target must use the same reference measure as the proposal.

[Constraints and named parameters](@ref transforms-guide) explains Jacobian ownership.
The standalone [integration example](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/examples/numerical_integration.jl)
also treats a one-dimensional integral.
