# [Batch targets](@id batch-guide)

A scalar target evaluates one draw.
An optional batch callback evaluates many draws with shared matrix operations or device kernels.

## Evaluate a regression model in batches

```@example batch
using ImportanceSamplers, Random, Statistics, LinearAlgebra

function logtarget(beta, p)
    value = -sum(abs2, beta) / 8
    for row in axes(p.X, 1)
        residual = -p.y[row]
        for column in eachindex(beta)
            residual += p.X[row, column] * beta[column]
        end
        value -= residual^2 / 2
    end
    return value
end

function batch!(values, samples, p)
    residuals = p.X * samples .- p.y
    values .= .-vec(sum(abs2, residuals; dims=1)) ./ 2 .-
              vec(sum(abs2, samples; dims=1)) ./ 8
    return nothing
end

rng = Xoshiro(42)
X = randn(rng, 80, 3)
y = X * [-0.4, 1.1, -0.8] + randn(rng, 80)
p = (; X, y)
proposal = SphericalGaussian(X \ y, 0.3)
target = LogTarget(logtarget; batch=batch!)
samples = importance_sample(
    rng, target, p,
    ImportanceSampling(proposal; nsamples=5_000),
)

mean(samples)
```

The model has unit observation noise and independent `Normal(0, 2)` coefficient priors.
The scalar and batch functions compute the same unnormalized log posterior.

Each sample is a column.
The matrix multiplication `p.X * samples` evaluates all proposed coefficient vectors together.

## Match the callback shape

| Scalar target input | Batch input | Output buffer |
|:--|:--|:--|
| Scalar | Length-`n` vector | Length-`n` vector |
| Length-`d` vector | `d × n` matrix | Length-`n` vector |
| Named parameters | Named tuple of batched leaves | Length-`n` vector |

With no context, use `batch!(values, samples)`.
With context, use `batch!(values, samples, p)`.
The scalar callable remains required.

Fill every output entry. Accept array views rather than requiring concrete `Matrix` types.
Treat inputs as read-only and retain none of the borrowed buffers.
The return value is ignored.

Each output must equal the scalar target for that sample.
It must not depend on other samples or the batch width.
Widths may vary. Empty batches are skipped.

## Control work and memory

The callback owns CPU parallelism.
`threaded=true` does not divide one callback into concurrent calls.
Proposal and weighting work retain their normal execution policy.

The example allocates an observations-by-samples residual matrix.
For large datasets, use bounded internal blocks or reusable model-specific scratch where appropriate.
Scratch must remain valid for the callback's actual width and device.

Without `batch=`, the package already evaluates independent scalar targets in parallel.
A batch callback changes how the model computes, not whether the sampler can run in parallel.

## Use constraints, gradients, and devices

With `transform=`, the callback receives logical named leaves.
The package adds the Jacobian. Do not add it in the callback.

`grad=` and AD selectors remain independent of `batch=`.
Automatic differentiation uses the scalar target, not this callback.
There is no separate batch-gradient callback.

A device callback receives resident inputs, outputs, and context.
Use the current task's stream or complete private-stream work before returning.
Failure checks may synchronize at batch boundaries, not once per sample.

## Understand adaptive calls

Adaptive methods batch the current round's samples.
AMIS caches earlier target values.

LAIS cannot combine dependent future MCMC steps into one target evaluation.
GRAMIS similarly cannot evaluate future backtracking trials in advance.

Native GRAMIS batches only active backtracking candidates.
Reactant uses fixed-width padded batches so prepared executables can be reused.
Inactive candidates retain their accepted locations and their new values are discarded.
Their extra evaluations appear in the diagnostics.

## Compare both paths

Batching can help data-heavy matrix models.
Cheap scalar targets can be faster with fused scalar execution.
Measure both on the intended model and device.

The [standalone example](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/examples/batched_linear_regression.jl)
also shows CUDA transfer.
The [benchmark guide](@ref benchmarks-guide) links paired scalar/batch measurements.
