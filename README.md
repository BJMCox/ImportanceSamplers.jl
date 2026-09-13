# ImportanceSamplers.jl

`ImportanceSamplers.jl` provides plain, multiple, and adaptive importance sampling
for Julia. It supports ordinary normalized proposals on CPU, and documented
native Gaussian, Student-t, and transform paths on CPU and CUDA. The package keeps estimator
inputs, device placement, and raw log weights visible.

## Installation

ImportanceSamplers requires Julia 1.12. Access to the private repository and
working GitHub authentication are required. Install directly from the repository
in the active Julia environment:

```julia
using Pkg
Pkg.add(url="https://github.com/BJMCox/ImportanceSamplers.jl.git")
```

For development from an existing clone, use its local path instead:

```julia
using Pkg
Pkg.develop(path="/path/to/ImportanceSamplers.jl")
```

CUDA is optional. Add and load CUDA.jl separately when using the documented
accelerator path.

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
- [Static multiple importance sampling](docs/src/methods/static_mis.md) explains
  proposal banks, the four complete MIS schemes, provenance, and support and
  device contracts.
- [Adaptive multiple importance sampling](docs/src/methods/amis.md) explains
  retrospective temporal-mixture weights and learned Gaussian or Student-t state.
- [Adaptive population importance sampling](docs/src/methods/apis.md) explains
  epoch-local deterministic-mixture weights and proposal-local mean updates.
- [Layered importance sampling](docs/src/methods/lais.md) explains upper RWM/RAM
  chains, interacting Sample Metropolis-Hastings, and lower importance samples
  with fixed scale factors.
- [Canonical covariance-adaptive importance sampling](docs/src/methods/cais.md)
  explains raw generating-proposal weights, ESS-tempered covariance fitting,
  and retained full-covariance state.
- [Nonlinear population Monte Carlo](docs/src/methods/npmc.md) explains clipped
  proposal adaptation with ordinary importance weights for estimation.
- [Deterministic-mixture population Monte Carlo](docs/src/methods/dm_pmc.md)
  explains adaptive spatial mixtures, global and local resampling, and retained
  state.
- [First-order GRAMIS-CAIS](docs/src/methods/first_order_gramis.md) explains
  the gradient move, robust local covariance fit, repulsion, and causal rounds.
- [Native proposals](docs/src/guide/native_proposals.md) documents spherical,
  diagonal, and dense-factor Gaussian and Student-t proposals.
- [Transforms](docs/src/guide/transforms.md) documents positive, interval,
  simplex, and structured parameters.
- [Accelerators](docs/src/guide/accelerators.md) contains the complete CUDA
  example, exact support limits, device transfer, and real-hardware reproducer.
- [Public API](docs/src/reference.md) lists all exports.

To build a persistent local copy of the rendered manual from the repository
root, prepare the documentation environment and run Documenter:

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Then open `docs/build/index.html`. The build is strict: doctest, link, and
export-documentation failures stop it.

Runnable checks include
[`native_gaussian.jl`](validation/reproducers/native_gaussian.jl),
[`simplex_transform.jl`](validation/reproducers/simplex_transform.jl), and the
real-CUDA matrix
[`cuda_plain_is.jl`](validation/reproducers/cuda_plain_is.jl).
Static MIS has an analytic CPU
[`static_mis.jl`](validation/reproducers/static_mis.jl) reproducer and an A100
[`cuda_static_mis.jl`](validation/reproducers/cuda_static_mis.jl) reproducer.
The concise end-to-end workflow is
[`examples/static_mis.jl`](examples/static_mis.jl).
The teaching [`examples/apis.jl`](examples/apis.jl) demonstrates fixed-scale
population adaptation and retained learned state.
[`examples/adaptive_student_t.jl`](examples/adaptive_student_t.jl) uses a
Student-t population on a curved target. It explains covariance versus scale,
weighted functional estimates, and optional explicit CUDA transfer.
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
