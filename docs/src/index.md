# ImportanceSamplers.jl

Importance sampling estimates integrals and expectations using weighted draws.
ImportanceSamplers provides fixed and adaptive proposals, multiple-proposal methods,
and explicit CPU or accelerator execution.

You supply a **log target**, a normalized proposal, and a random-number generator.
The result contains samples and their raw log weights.

## Install

The package requires Julia 1.10 or later. Install the repository in your project:

```julia
using Pkg
Pkg.add(url="https://github.com/BJMCox/ImportanceSamplers.jl.git")
```

Accelerator and automatic-differentiation packages are optional.

## Start with an example

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

## What the weights mean

The target may omit its normalizing constant. The proposal must be normalized
and cover the regions that contribute to the integral.

Weighted estimates need not resemble unweighted sample averages.
A large weight ESS does not prove that a proposal found every mode or tail.
The [results guide](@ref results-guide) explains these distinctions before introducing diagnostics.
