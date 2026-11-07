# ImportanceSamplers.jl

```@meta
CurrentModule = ImportanceSamplers
```

Importance sampling estimates integrals and expectations using weighted draws.
ImportanceSamplers provides fixed and adaptive proposals, multiple-proposal methods,
and explicit CPU or accelerator execution.

You supply a **log target**, a normalized proposal, and a random-number generator.
The result contains samples and their raw log weights.

## Install

Install the package in your project:

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaBayes/ImportanceSamplers.jl.git")
```

Accelerator and automatic-differentiation packages are optional.

## Draw weighted samples

Estimate the mean of a three-dimensional standard Gaussian. Its log density is
`-sum(abs2, x) / 2`, up to an additive constant. Each `x` is a vector of the three
parameters being sampled.

```@example overview
using ImportanceSamplers, Random, Statistics

logtarget(x) = -sum(abs2, x) / 2
proposal = SphericalGaussian(zeros(3), 1.5)
algorithm = ImportanceSampling(proposal; nsamples=10_000)
samples = importance_sample(Xoshiro(42), logtarget, algorithm)

size(samples.samples)
```

The proposal is a Gaussian with mean `zeros(3)` and standard deviation `1.5` in
each coordinate. The result stores one draw per column, with a matching raw log
weight in `samples.logweights`.

## Estimate an expectation

```@example overview
(
    mean=round.(mean(samples); digits=3),
    variance=round.(var(samples); digits=3),
    squared_radius=round(mean(x -> sum(abs2, x), samples); digits=3),
)
```

The reference mean is `[0, 0, 0]`, the marginal variances are `[1, 1, 1]`, and
the mean squared radius is `3`. These estimates use the importance weights.
`mean(samples.samples; dims=2)` ignores them and describes the proposal instead.

For plain IS, each log weight is `logtarget(x) - logdensityof(proposal, x)`.
The weights correct for drawing from the proposal rather than the target.
The proposal must cover every region that contributes to the expectation.
Weight ESS measures concentration, not whether a run found every mode or tail.

## Continue the example

[Your first weighted estimate](@ref first-estimate) samples a three-dimensional
target and computes its mean, covariance, and normalizing constant.

Then follow either worked application:

- [Logistic regression](@ref logistic-tutorial): define a posterior, adapt a proposal, and predict a probability.
- [Numerical integration](@ref integration-tutorial): estimate ordinary integrals, including signed integrands.

Every tutorial defines its data and runs from top to bottom.

## Find a specific task

| Task | Page |
|:--|:--|
| Pass data or use an existing density | [Targets and data](@ref targets-guide) |
| Choose Gaussian or Student-t proposals | [Proposals](@ref proposals-guide) |
| Use positive, bounded, or simplex parameters | [Constraints and named parameters](@ref transforms-guide) |
| Compute expectations, intervals, or resampled draws | [Working with results](@ref results-guide) |
| Reuse a fitted proposal | [Adaptation and reuse](@ref reuse-guide) |
| Choose an importance-sampling method | [Choosing a method](@ref choosing-method) |
| Evaluate many samples with matrix operations | [Batch targets](@ref batch-guide) |
| Supply gradients or use automatic differentiation | [Gradients](@ref gradients-guide) |
| Execute on CUDA, Metal, or Reactant | [Devices](@ref devices-guide) |
| Look up a signature | [API reference](@ref api-reference) |

## Documentation versions

`dev` documents the current `main` branch. Tagged releases retain their own
documentation. `stable` points to the latest release. Use the deployed site's
version selector to choose the documentation for your installed package.
