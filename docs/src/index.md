# ImportanceSamplers.jl

ImportanceSamplers provides plain, multiple, and adaptive importance sampling
on CPU and documented native CUDA paths. Supply a normalized proposal, an RNG, and a target
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

## Choose a method

| Method | Proposal input | What adapts | Retained-weight denominator | Count | Execution |
|:--|:--|:--|:--|:--|:--|
| Plain IS | one normalized proposal | nothing | generating proposal | `nsamples` | CPU; documented native subset on CUDA |
| Static MIS | fixed proposal bank | nothing | selected spatial or generating-proposal scheme | `nsamples` | CPU; documented native subset on CUDA |
| AMIS | one native Gaussian | mean and covariance | all-history temporal mixture | `round_size` | CPU and native Gaussian CUDA |
| APIS | native Gaussian bank | means | current population mixture | `round_size` | CPU and native Gaussian CUDA |
| LAIS | equal-mass native Gaussian bank | centres by independent or interacting upper MCMC; optional upper covariance tuning | equal current population mixture | divisible `round_size` | CPU and native Gaussian CUDA |
| CAIS | native Gaussian bank | means and covariances | generating proposal | `round_size` | CPU and native Gaussian CUDA |
| N-PMC | one native Gaussian | mean and covariance from clipped adaptation weights | generating proposal | `round_size` | CPU and native Gaussian CUDA |
| DM-PMC | proposal bank | locations by resampling | realized current population mixture | `round_size` | CPU; documented native subset on CUDA |
| GR-PMC | equal-mass proposal bank | locations by global resampling | equal current population mixture | divisible `round_size` | CPU; documented native subset on CUDA |
| LR-PMC | equal-mass proposal bank | locations by local resampling | equal current population mixture | divisible `round_size` | CPU; documented native subset on CUDA |
| First-order GRAMIS-CAIS | native Gaussian bank | means by gradient/repulsion; local covariances | realized current population mixture | `round_size` | CPU and native Gaussian CUDA |

Use the linked method guide for its support, allocation, adaptation, and failure
contract; the table is only a starting point.

## Where next

- [Plain importance sampling](@ref) covers estimator semantics, generic CPU
  proposals, prepared reuse, threading, results, and failures.
- [Static multiple importance sampling](@ref) covers proposal banks, all four
  complete assignment/denominator schemes, provenance, and CPU/CUDA limits.
- [Adaptive multiple importance sampling](@ref) covers retrospective temporal
  mixtures and learned Gaussian state.
- [Adaptive population importance sampling](@ref) covers epoch-local spatial
  mixtures, proposal-local mean fits, and retained fixed-covariance state.
- [Layered importance sampling](@ref) covers upper RWM/RAM chains, interacting
  Sample Metropolis-Hastings, fixed lower covariances, and all-round
  deterministic-mixture weights.
- [Canonical covariance-adaptive importance sampling](@ref) covers standard
  generating-proposal weights, raw mean fits, and robust covariance replacement.
- [Nonlinear population Monte Carlo](@ref) covers clipped adaptation and raw
  importance-weight estimation.
- [DM-PMC, GR-PMC, and LR-PMC](@ref) covers adaptive spatial
  mixtures, global and local resampling, retained proposal state, and CPU/CUDA
  limits.
- [First-order GRAMIS-CAIS](@ref) covers gradient moves, robust local covariance
  fitting, repulsion, causal rounds, and CPU/CUDA limits.
- [Native proposals](@ref) explains the Gaussian scale and factor contracts.
- [Transforms](@ref) covers constrained and structured parameters, including
  the simplex reference measure.
- [Accelerators](@ref) gives the complete CUDA example, transfer boundary,
  exact support matrix, and runnable validation.
- [Public API](@ref) lists every exported binding.
- [Validation and support](@ref) lists reproducible checks and their evidence limits.

Runnable workflows include the public
[DM-PMC example](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/examples/dm_pmc.jl),
[APIS example](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/examples/apis.jl),
and [CAIS example](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/examples/cais.jl)
and a plain numerical-integration example for
[`integral(exp(-x^2)) = sqrt(pi)`](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/examples/numerical_integration.jl).

For mathematical background, see Elvira and Martino's open-access
[“Advances in Importance Sampling”](https://arxiv.org/abs/2102.05407) and
Agapiou et al.'s
[“Importance Sampling: Intrinsic Dimension and Computational Cost”](https://arxiv.org/abs/1511.06196).
