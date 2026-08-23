# ImportanceSamplers.jl

`ImportanceSamplers.jl` is a small CPU implementation of plain importance
sampling for Julia. It accepts ordinary normalized proposal types, ordinary
scalar log-target callables, and an explicit RNG. It returns samples plus the
raw log weights needed to inspect the estimator.

This first slice is intentionally narrow: plain nonadaptive importance
sampling, CPU execution, and `Float32` or `Float64` log densities. It does not
depend on Distributions, StatsBase, automatic differentiation, transform
packages, vendor GPU packages, BAT, or Turing.

## Five-minute quick start

The proposal below is a complete normalized Gaussian implemented with Julia's
standard RNG and `DensityInterface`:

```julia
using DensityInterface
using ImportanceSamplers
using Random

struct GaussianProposal{T<:AbstractFloat}
    mean::T
    scale::T
end

function Random.rand(rng::Random.AbstractRNG, proposal::GaussianProposal{T}) where {T}
    proposal.mean + proposal.scale * randn(rng, T)
end

function DensityInterface.logdensityof(proposal::GaussianProposal, x::Real)
    z = (x - proposal.mean) / proposal.scale
    -0.5 * abs2(z) - log(proposal.scale) - 0.5 * log(2pi)
end

proposal = GaussianProposal(0.0, 1.0)
logtarget(x)::Float64 = DensityInterface.logdensityof(proposal, x)
algorithm = ImportanceSampling(proposal; nsamples=1_000)

result = importance_sample(
    Xoshiro(42),
    logtarget,
    algorithm;
    threaded=false,
)

length(result)                  # 1000
result[1]                       # aligned sample, log weight, and provenance
normalized_weights(result)      # weights that sum to one
lognormalizer(result)           # 0.0 for this normalized identity
```

Nothing required by the example is hidden in test setup. The proposal's
`rand` and `logdensityof` methods must describe the same normalized measure.
The target returns a log density; the sampler never applies `log` for you.

`result.logweights` contains the raw values
`logtarget(x) - logdensityof(proposal, x)`. They are deliberately not normalized
probabilities. Raw log weights preserve the normalizer estimator and remain
stable for extreme ratios.

## Prior importance sampling

A normalized Bayesian prior can be used directly as the proposal. If the target
is `loglikelihood(theta, data) + logdensityof(prior, theta)`, the prior term
cancels the proposal log density exactly in the weight formula, leaving the log
likelihood. In that case `lognormalizer(result)` estimates Bayesian log evidence
only when the likelihood and prior retain every required normalizing constant.

Prior sampling is a useful baseline, but its weights often collapse when the
data are informative or the parameter dimension is high. See the
[prior importance sampling guide](docs/src/methods/importance_sampling.md#prior-importance-sampling)
for the cancellation, evidence semantics, warning signs, and a complete
structured zero-inflated Poisson regression example.

## Targets with data or constants

Use a two-argument target when a concrete context should stay separate from the
callable:

```julia
context = (mean=0.5, scale=0.8)
function logtarget(x, p)::Float64
    z = (x - p.mean) / p.scale
    -0.5 * abs2(z) - log(p.scale) - 0.5 * log(2pi)
end

result = importance_sample(
    Xoshiro(43),
    logtarget,
    context,
    ImportanceSampling(proposal; nsamples=2_000);
    threaded=false,
)
```

The proposal determines the sampled shape. A draw may be a scalar, a dense
vector, or a nonempty named-tuple tree of numeric scalar and vector leaves.
Results add one final sample axis to each logical leaf.

## Essential contracts

- The proposal is normalized and covers every target region that contributes
  mass.
- Target and proposal densities use the same reference measure, including any
  required Jacobians.
- Target calls return `Float32` or `Float64` log densities. Finite values and
  `-Inf` are accepted; `NaN` and `+Inf` are rejected.
- Every proposal draw has stable structure, leaf types, and vector lengths.
- `lognormalizer(result)` is Bayesian log evidence only when the target keeps
  every required constant.
- If all raw log weights are `-Inf`, samples are retained and the log normalizer
  is `-Inf`; `normalized_weights` throws rather than inventing uniform weights.

## Prepared reuse and threading

Prepare once when the target, proposal, sample count, device, and threading
policy stay fixed:

```julia
sampler = prepare_sampler(
    Xoshiro(44),
    logtarget,
    context,
    ImportanceSampling(proposal; nsamples=2_000);
    threaded=true,
)

first_result = importance_sample!(sampler)
second_result = importance_sample!(sampler)
```

The prepared sampler retains and advances the supplied RNG. Treat that RNG as
transferred into a single-owner, non-reentrant handle. Repeated results are
independent estimator runs, are not cumulative, and own separate arrays.

Threading is requested by default. Use `threaded=false` for an explicit serial
run. A Julia process with one default thread falls back to serial execution even
when threading was requested. All proposal draws finish before worker tasks
start; target and proposal-density callables must therefore be pure,
deterministic, and thread-safe.

Only CPU execution is implemented. Unsupported device requests fail during
preparation rather than falling back to the host. GPU work is deliberately
deferred; see the
[current and future device boundary](docs/src/methods/importance_sampling.md#future-accelerator-work).

## Results at a glance

`WeightedSamples` exposes `samples`, `logweights`, `provenance`, and
`diagnostics`. Indexing or iteration returns aligned records. Ranges, integer
index vectors, and masks return `WeightedSampleView`; views have descriptive
normalized weights but no estimator-valid `lognormalizer`.

Successful CPU diagnostics include `failures=0` and
`transfers=(count=0, bytes=0)`. These are facts for that run, not cumulative
sampler state.

See the [plain importance sampling guide](docs/src/methods/importance_sampling.md)
for support, reference-measure, ownership, threading, failure, shape, and
troubleshooting details. The
[package manual](docs/src/index.md) contains the tested quick start and public
API reference.

## Validation and benchmarks

- [`validation/reproducers/plain_is.jl`](validation/reproducers/plain_is.jl)
  checks labeled analytic identities with fixed seed, recorded versions,
  budgets, tolerances, and a failing exit on disagreement.
- [`benchmark/plain_is.jl`](benchmark/plain_is.jl) uses BenchmarkTools for
  preparation, warm prepared execution, throughput, allocations, and allocated
  memory. Pass `--smoke` for a fast wiring check. No machine-specific timing
  threshold is asserted.

For background, see the open-access overview by Elvira and Martino,
[“Advances in Importance Sampling”](https://arxiv.org/abs/2102.05407).
