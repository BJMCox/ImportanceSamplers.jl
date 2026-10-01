# [Targets and data](@id targets-guide)

A target returns the log of the density or nonnegative integrand you want to study.
The proposal determines which parameters are sampled and their shape.

## Pass data separately

```@example targets
using ImportanceSamplers, Random, Statistics

function logtarget(theta, p)
    return -sum(abs2, theta .- p.centre) / (2p.scale^2)
end

p = (; centre=[1.0, -1.0, 0.5], scale=0.8)
algorithm = ImportanceSampling(SphericalGaussian(zeros(3), 2.0); nsamples=5_000)
samples = importance_sample(Xoshiro(42), logtarget, p, algorithm)

mean(samples)
```

Each `theta` is a three-element vector. The sampler does not estimate `p`.
The same context also works with `prepare_sampler(rng, logtarget, p, algorithm)`.

A target without data can use `logtarget(theta)` instead.
A named target can use `theta.scale` or `theta.weights` through a
[transform layout](@ref transforms-guide).

## Adapt an existing keyword function

Julia keyword arguments need an ordinary callable wrapper:

```@example targets
model(theta; centre, scale) = -sum(abs2, theta .- centre) / (2scale^2)
wrapped(theta, p) = model(theta; p...)

other = importance_sample(Xoshiro(42), wrapped, p, algorithm)
mean(other)
```

There is no reserved keyword name or fixed number of keywords.
The wrapper translates your function's signature into the package's `(theta, p)` convention.

Pass numerical arrays through `p` for device execution.
A closure over host arrays does not transfer those arrays to a GPU.

## Use an existing density interface

Objects implementing `DensityInterface.logdensityof` can serve as targets.
For example, with Distributions.jl installed:

```julia
using Distributions, ImportanceSamplers, Random

target = MvNormal(zeros(3), ones(3))
proposal = MvNormal(zeros(3), fill(1.5, 3))
samples = importance_sample(
    Xoshiro(42), target,
    ImportanceSampling(proposal; nsamples=5_000),
)
```

The unknown is still the three-dimensional vector drawn from `proposal`.
The target distribution supplies its density, not a separate parameter list.

LogDensityProblems objects can supply `logdensity`, `dimension`, and `capabilities`.
Preparation checks their declared dimension against the proposal.
A first-order interface can also supply [gradients](@ref gradients-guide).

Use `LogTarget(callable)` to select callable semantics explicitly when an object
also advertises a density interface. The selected interface does not change after an execution error.

## Observe the numerical contract

A log target must return `Float32` or `Float64` with an inferable return type.
Finite values and `-Inf` are valid. `-Inf` means zero target density.
`NaN`, `+Inf`, and exceptions fail the run.

Match constants and arrays to the intended precision.
For example, use `0.5f0` and `Float32` data with a single-precision model.

Scalar targets may run concurrently. Treat the input sample and context as read-only.
Avoid hidden mutable scratch storage. Use an explicit [batch callback](@ref batch-guide)
when evaluation should share work across samples.

## Keep density measures consistent

The target and proposal must describe densities against the same reference measure.
A transformed target needs exactly one Jacobian adjustment.

Omitting a target constant preserves normalized expectations.
It changes `lognormalizer`, so it is not valid when you need absolute evidence.
