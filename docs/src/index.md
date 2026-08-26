# ImportanceSamplers.jl

ImportanceSamplers provides explicit plain importance sampling on CPU and a
validated native CUDA path. Supply a normalized proposal, an RNG, and a target
that returns an unnormalized **log density**. Results retain samples and the
canonical raw log weights.

## Minimal CPU example

```jldoctest quickstart
using ImportanceSamplers
using Random

proposal = SphericalGaussian(0.0, 1.0)
logtarget(x)::Float64 = -0.5 * abs2(x)

result = importance_sample(
    Xoshiro(42),
    logtarget,
    ImportanceSampling(proposal; nsamples=32);
    threaded=false,
)

expected = 0.5 * log(2pi)
(
    length(result),
    all(w -> isapprox(w, expected; atol=8eps()), result.logweights),
    isapprox(lognormalizer(result), expected; atol=8eps()),
)

# output

(32, true, true)
```

The target omits the Gaussian normalizing constant, so every raw log weight and
the estimated log normalizer equal that omitted constant. The sampler does not
apply `log` to the target. Call [`normalized_weights`](@ref) only when you need
weights that sum to one.

For data or constants, use `logtarget(sample, p)` and pass `p` between the
target and algorithm arguments. The proposal alone determines sample shape.

## Where next

- [Plain importance sampling](@ref) covers estimator semantics, generic CPU
  proposals, prepared reuse, threading, results, and failures.
- [Static multiple importance sampling](@ref) covers proposal banks, all four
  complete assignment/denominator schemes, provenance, and CPU/CUDA limits.
- [Deterministic-mixture population Monte Carlo](@ref) covers adaptive spatial
  mixtures, global resampling, retained proposal state, and CPU/CUDA limits.
- [Native proposals](@ref) explains the Gaussian scale and factor contracts.
- [Transforms](@ref) covers constrained and structured parameters, including
  the simplex reference measure.
- [Accelerators](@ref) gives the complete CUDA example, transfer boundary,
  exact support matrix, and runnable validation.
- [Public API](@ref) lists every exported binding.

Runnable workflows include the public
[DM-PMC example](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/examples/dm_pmc.jl)
and a plain numerical-integration example for
[`integral(exp(-x^2)) = sqrt(pi)`](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/examples/numerical_integration.jl).

For mathematical background, see Elvira and Martino's open-access
[“Advances in Importance Sampling”](https://arxiv.org/abs/2102.05407) and
Agapiou et al.'s
[“Importance Sampling: Intrinsic Dimension and Computational Cost”](https://arxiv.org/abs/1511.06196).
