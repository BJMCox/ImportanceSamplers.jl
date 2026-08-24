# ImportanceSamplers.jl

`ImportanceSamplers.jl` provides explicit plain importance sampling for Julia.
It supports ordinary normalized proposals on CPU, and the documented native
Gaussian and transform subset on CPU and CUDA. The package keeps estimator
inputs, device placement, and raw log weights visible.

## Minimal CPU example

```julia
using ImportanceSamplers
using Random

proposal = SphericalGaussian(0.0, 1.0)
logtarget(x)::Float64 = -0.5 * abs2(x) # unnormalized log density

result = importance_sample(
    Xoshiro(42),
    logtarget,
    ImportanceSampling(proposal; nsamples=1_000);
    threaded=false,
)

result.logweights        # raw log target/proposal ratios
normalized_weights(result)
lognormalizer(result)
```

The proposal must be normalized and cover every target region that contributes
mass. Target and proposal densities must use the same reference measure. The
target returns a `Float32` or `Float64` log density; the sampler never applies
`log` for you.

## Guides

- [Plain importance sampling](docs/src/methods/importance_sampling.md) explains
  the estimator, generic proposals, prepared reuse, and result semantics.
- [Native proposals](docs/src/guide/native_proposals.md) documents spherical,
  diagonal, and dense-factor Gaussians.
- [Transforms](docs/src/guide/transforms.md) documents positive, interval,
  simplex, and structured parameters.
- [Accelerators](docs/src/guide/accelerators.md) contains the complete CUDA
  example, exact support limits, device transfer, and real-hardware reproducer.
- [Public API](docs/src/reference.md) lists all exports.

Runnable checks include
[`native_gaussian.jl`](validation/reproducers/native_gaussian.jl),
[`simplex_transform.jl`](validation/reproducers/simplex_transform.jl), and the
real-CUDA matrix
[`cuda_plain_is.jl`](validation/reproducers/cuda_plain_is.jl).
The runnable
[`logistic_regression.jl`](examples/logistic_regression.jl) example performs
end-to-end Bayesian inference for an intercept and two regression slopes. It
uses a prior pilot to fit an inflated Gaussian proposal, then draws an
independent final importance sample. Its
[`cuda_logistic_regression.jl`](examples/cuda_logistic_regression.jl) variant
runs both sampling rounds on CUDA and transfers results to CPU only between
adaptation rounds and for the final summary.
For background, see the open-access survey
[“Advances in Importance Sampling”](https://arxiv.org/abs/2102.05407).
