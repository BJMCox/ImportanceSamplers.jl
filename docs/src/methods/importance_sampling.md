# Plain importance sampling

Plain importance sampling draws independent samples from one normalized
proposal and corrects expectations with density ratios. For a broader
open-access treatment, see Elvira and Martino,
[“Advances in Importance Sampling”](https://arxiv.org/abs/2102.05407).

## The estimator

Let ``q`` be the normalized proposal and ``\pi`` the target supplied to the
package. For ``x_i \sim q``, the stored log weight is

```math
\log w_i = \log \pi(x_i) - \log q(x_i).
```

The package keeps these raw, unnormalized log weights in
`result.logweights`. It estimates the target normalizer with

```math
\log \widehat Z = \operatorname{logsumexp}(\log w) - \log n.
```

[`lognormalizer(result)`](@ref lognormalizer) returns this quantity. The linear
estimate ``\widehat Z`` is unbiased under the usual support and integrability
conditions; its logarithm generally is not.

The estimate is Bayesian log evidence only when `logtarget` includes every
normalizing constant required by the model. Dropping a target constant shifts
`lognormalizer` by the same amount. It does not change normalized weighted
summaries.

## Prior importance sampling

Prior importance sampling uses a normalized Bayesian prior as the proposal.
For a target that retains the same prior density,

```math
\log \pi(\theta) = \log p(y \mid \theta) + \log p(\theta),
\qquad
\log q(\theta) = \log p(\theta),
```

the canonical raw weight cancels exactly to

```math
\log w(\theta) = \log p(y \mid \theta).
```

The package intentionally performs the ordinary subtraction and does not try
to detect this algebra inside an opaque target. Small roundoff can therefore
remain in computed log weights.

When the prior is normalized and `logtarget` contains the complete normalized
likelihood, the target normalizer is the marginal likelihood ``p(y)`` and
[`lognormalizer(result)`](@ref lognormalizer) estimates Bayesian log evidence.
If likelihood constants are omitted, the estimate has the same omitted
constant and is not the evidence of the stated probabilistic model.

The following copyable example uses a named-tuple prior draw for a
zero-inflated Poisson regression. It defines all proposal and model math
locally; no distribution package is required.

```jldoctest prior_is
using DensityInterface
using ImportanceSamplers
using Random

struct ZIPRegressionPrior end

function Random.rand(rng::Random.AbstractRNG, ::ZIPRegressionPrior)
    return (
        intercept=randn(rng),
        slope=randn(rng),
        zero_logit=randn(rng),
    )
end

function DensityInterface.logdensityof(::ZIPRegressionPrior, theta::NamedTuple)
    square_sum = abs2(theta.intercept) + abs2(theta.slope) +
                 abs2(theta.zero_logit)
    return -0.5 * square_sum - 1.5 * log(2pi)
end

function local_logaddexp(a::Float64, b::Float64)
    largest = max(a, b)
    return largest + log(exp(a - largest) + exp(b - largest))
end

function log_sigmoid(x::Float64)
    return x >= 0 ? -log1p(exp(-x)) : x - log1p(exp(x))
end

log_factorial(y::Int) = sum(log, 2:y; init=0.0)

function zip_loglikelihood(theta, p)::Float64
    log_zero_probability = log_sigmoid(theta.zero_logit)
    log_poisson_probability = log_sigmoid(-theta.zero_logit)
    value = 0.0
    for observation in eachindex(p.y, p.x)
        y = p.y[observation]
        log_rate = theta.intercept + theta.slope * p.x[observation]
        rate = exp(log_rate)
        poisson_logdensity = y * log_rate - rate - log_factorial(y)
        value += if iszero(y)
            local_logaddexp(
                log_zero_probability,
                log_poisson_probability + poisson_logdensity,
            )
        else
            log_poisson_probability + poisson_logdensity
        end
    end
    return value
end

function zip_logtarget(theta, p)::Float64
    return zip_loglikelihood(theta, p) +
           DensityInterface.logdensityof(p.prior, theta)
end

prior = ZIPRegressionPrior()
context = (
    x=[-1.0, 0.0, 1.0, 2.0],
    y=[0, 1, 0, 3],
    prior=prior,
)
result = importance_sample(
    Xoshiro(2026),
    zip_logtarget,
    context,
    ImportanceSampling(prior; nsamples=128);
    threaded=false,
)
likelihood_logs = [zip_loglikelihood(record.sample, context) for record in result]

(
    result[1].sample isa NamedTuple,
    all(isapprox.(result.logweights, likelihood_logs; rtol=8eps(), atol=8eps())),
    isfinite(lognormalizer(result)),
)

# output

(true, true, true)
```

Prior IS can collapse even when the implementation is correct: informative
data or high dimension may concentrate nearly all normalized weight on a tiny
fraction of prior draws. Treat it as a baseline and inspect weight
concentration before trusting posterior summaries. More samples do not repair
a proposal that almost never reaches the posterior; use a proposal adapted to
the posterior geometry when collapse is material.

## Transformed and product proposals

[`TransformedProposal`](@ref) applies a normalized change of variables, while
[`ProductProposal`](@ref) combines independent named blocks on CPU. See
[Transforms](@ref) for the scalar, structured, and simplex contracts and
[Accelerators](@ref) for the narrower CUDA-supported layout.

## Target contract

A context-free target has one argument:

```julia
logtarget(sample)
```

A contextual target follows the common SciML-style callable shape:

```julia
logtarget(sample, p)
```

`p` may be any concrete Julia value and is supplied to `prepare_sampler` or
`importance_sample` between the target and the algorithm. In either form the
return value is already a `Float32` or `Float64` log density. Finite values and
`-Inf` are valid. `NaN` and `+Inf` are rejected.

Targets may also advertise a density through `LogDensityProblems` or
`DensityInterface`; [`LogTarget`](@ref) explicitly chooses package-callable
semantics. Known interfaces and known LogDensityProblems dimensions are
resolved once during preparation, before any proposal draw. Callable
applicability and scalar return types are validated on the first retained
execution when the logical sample type is available. An execution exception is
never caught and retried through a lower-priority interface.

The target does not determine dimension or sample structure. That information
comes entirely from the proposal's draws.

## Proposal, support, and reference measure

A generic CPU proposal must implement both:

```julia
Random.rand(rng, proposal)
DensityInterface.logdensityof(proposal, sample)
```

Those operations must describe the same **normalized** probability measure.
Constructing `ImportanceSampling(proposal; nsamples=n)` is your assertion that
they do. The package cannot prove normalization for an opaque user type.

The proposal must cover every region that contributes target mass: formally,
the target measure must be absolutely continuous with respect to the proposal.
If important target support is absent from the proposal, no finite sample can
repair the estimator.

Target and proposal densities must also use the same reference measure. For
example, do not subtract a density with respect to Lebesgue measure from a
density with respect to a transformed coordinate measure unless the required
Jacobian is already included.

At every generated sample, proposal log density may be finite or `+Inf`, but
not `NaN` or `-Inf`. A generating proposal assigning itself zero density is an
invalid proposal contract.

## Sample shapes

One proposal draw may be a scalar, a dense vector, or a nonempty named-tuple
tree whose leaves are scalars or dense vectors. The returned storage mirrors
that logical shape and adds one sample axis:

| One draw | `result.samples` for `n` draws | `result[i].sample` |
|:--|:--|:--|
| `Float64` | `Vector{Float64}` of length `n` | scalar |
| `Vector{Float64}` of length `d` | `d × n` matrix | vector view |
| `(a=scalar, b=vector)` | `(a=vector, b=matrix)` | matching named tuple |

Every draw must retain the first draw's structure, leaf types, and vector
lengths. Unstable proposal return types fail the complete run instead of
producing heterogeneous storage. The first draw is a real retained sample;
preparation never consumes a hidden probe draw.

## Results

[`WeightedSamples`](@ref) is a complete estimator result. Its stable fields are
`samples`, `logweights`, `provenance`, and `diagnostics`. Plain IS has empty
per-sample provenance. Run-level diagnostics record the method, execution mode,
requested threading policy, exact sample count, successful-run `failures=0`,
and CPU `transfers=(count=0, bytes=0)`. These facts describe one completed run;
they are not cumulative sampler state.

Scalar indexing and iteration yield aligned records:

```julia
record = result[1]
record.sample
record.logweight
record.provenance

for record in result
    # use the aligned sample and raw log weight
end
```

Ranges, integer index vectors, and Boolean masks return a
[`WeightedSampleView`](@ref). A view supports aligned iteration and
[`normalized_weights`](@ref), but not [`lognormalizer`](@ref): an arbitrary
subset is not the complete estimator that produced the original normalizer.
Apply an MLDataDevices device directly to a view to create an independent
aligned copy, including an explicit CPU copy for scalar iteration.

`normalized_weights(result)` derives a new same-device array that sums to one.
The source of truth remains `result.logweights`. Results own their arrays, so a
later prepared run cannot change an earlier result. User mutation of result
arrays is possible on CPU but unsupported because it can invalidate estimator
semantics.

### Weighted summaries

Use Julia's standard `Statistics` functions directly:

```julia
using Statistics

estimate = mean(result)
variance = var(result)
deviation = std(result)

# Expectation and variance of a scalar function of each sample.
functional_mean = mean(sample -> abs2(sample), result)
functional_variance = var(sample -> abs2(sample), result)
```

These methods use self-normalized importance weights. `var`, `std`, and `cov`
return uncorrected moments of the represented target approximation;
`corrected=true` is unsupported. `mean`, `var`, `std`, and `cov` keep array
results on the input device. A device functional must compile there and return
one concrete scalar per sample.

`quantile(result, p)` and `median(result)` use component-wise weighted
quantiles. Exact ordering remains CPU-only. Transfer a device result to CPU
explicitly before calling either function.

### Unweighted resampling

Use [`resample`](@ref) when an API needs ordinary draws instead of a weighted
estimator:

```jldoctest resampling
using ImportanceSamplers
using Random
using Statistics

weighted = WeightedSamples([10.0, 20.0, 30.0], [-Inf, 0.0, -Inf])
draws = resample(Xoshiro(7), weighted, 4)

(draws.samples, draws[1], mean(draws))

# output

([20.0, 20.0, 20.0, 20.0], 20.0, 20.0)
```

The default [`MultinomialResampling`](@ref) method makes independent weighted
draws with replacement. Omit the count to request exactly `length(weighted)`
draws. The output [`UnweightedSamples`](@ref) contains samples only. It does
not copy the source importance weights or invent uniform importance weights,
so keep the original weighted result for evidence, ESS, and weighted
summaries.

`mean`, `var`, `std`, `cov`, `quantile`, and `median` use ordinary unweighted
definitions on `UnweightedSamples`. Resampling preserves scalar, vector, and
named-tuple sample structure and keeps output on the input device. Exact
quantiles and scalar indexing remain CPU-only.

## All-zero weights

If every target evaluation is `-Inf`, the run still returns its samples and raw
`-Inf` log weights. `lognormalizer(result)` is `-Inf`. Normalized weights do not
exist in this case, so `normalized_weights(result)` throws
[`AllZeroWeightsError`](@ref) rather than inventing uniform weights.

## Prepared reuse and RNG ownership

One-shot execution is preparation followed by one prepared run:

```julia
result = importance_sample(rng, logtarget, algorithm)
```

For repeated independent estimator runs with the same bound configuration,
prepare once:

```julia
sampler = prepare_sampler(rng, logtarget, algorithm; threaded=true)
first_result = importance_sample!(sampler)
second_result = importance_sample!(sampler)
```

The prepared sampler retains the exact RNG object supplied at preparation and
advances its stream across calls. Treat that RNG as transferred into the
single-owner sampler: do not draw from it elsewhere while relying on replay.
Repeated results are separate, noncumulative estimators and own separate
arrays.

### Retarget an adapted sampler

Use `retarget` to reuse a committed adaptive proposal with a new log target and
fresh RNG:

```julia
adapted = prepare_sampler(rng1, old_logtarget, p1, algorithm)
importance_sample!(adapted)

warm = retarget(rng2, adapted, new_logtarget, p2)
result = importance_sample!(warm)
```

This operation supports `DeterministicMixturePMC`, `APIS`, `AMIS`, and
`FirstOrderGRAMIS`. It preserves the committed proposal, algorithm controls,
execution policy, and device. It resets every target-specific history and
workspace. The old sampler remains usable and keeps its RNG stream.

On CPU, the new sampler owns `rng2`. Accelerator setup consumes one `UInt64`
from `rng2` after preflight to seed an owned backend RNG. A later backend RNG
construction failure can therefore consume that value. `rng2` must not be the
RNG owned by `adapted`.

For an accelerator sampler, retargeting stages the committed proposal in CPU
memory, allocates fresh CPU workspaces, and transfers the complete new sampler
to the same device. This is a setup boundary, not an execution fallback. No old
samples or weights cross targets. The caller must ensure that the reused
proposal covers the new target's support. Normal preparation still checks known
target dimensions, callable contracts, and method constraints.

Preparation always returns CPU state. Before its first execution, apply an
explicit MLDataDevices device to transfer the complete prepared sampler:

```julia
device = MLDataDevices.cpu_device()
transferred = device(sampler)
# Equivalent: transferred = sampler |> device
```

Transfer returns a distinct sampler. Its RNG, explicit context `p`, callable
target structs, proposal state, and numerical arrays are independent of the
source. Ordinary functions remain the same callable object; pass device data
through `p` instead of capturing host arrays in a closure. The source remains
valid. A named callable struct that subtypes `Function` is transferred only
when it supplies an explicit standard Adapt rule; the generic compiler-closure
reconstruction rule is never used. RNG state is cloned with the RNG's standard
`copy` operation. An RNG
whose copy is unavailable or aliases the source is rejected with
[`SamplerDeviceError`](@ref). Callable target structs that contain reachable
opaque closures are also rejected before transfer rather than allowing the
standard traversal to inspect or reconstruct their captures. Once an execution
has begun, later transfer throws [`SamplerAlreadyExecutedError`](@ref),
including after a failed run.

An accelerator destination is deliberately one-hop state. Only a CPU-origin
prepared sampler can create an accelerator destination; applying any device to
that destination fails as `:prepared_migration_unsupported`, even before its
first execution. The original CPU source remains valid and may create another
independent destination. Custom context or callable structs with array fields
must register standard Adapt support for the accelerator kernel argument
conversion; named tuples already do so. Unsupported representations fail at
transfer as `:kernel_argument_unsupported` without advancing the source RNG.

A prepared sampler is mutable and non-reentrant. Do not call
`importance_sample!` concurrently on the same handle; prepare separate
samplers with separate RNGs. Re-entry throws [`SamplerBusyError`](@ref). A
failed run clears the busy state, but never returns a partial result.

## CPU threading and purity

CPU outer threading is requested by default. Set `threaded=false` in the
one-shot call or at preparation for an explicitly serial run:

```julia
sampler = prepare_sampler(rng, logtarget, algorithm; threaded=false)
```

The choice is bound into a prepared sampler and cannot be changed per run. On
CPU, if Julia has only one default thread, `threaded=true` falls back to serial
execution. Accelerator execution uses its backend-parallel launch policy
regardless of the host thread count. The result diagnostic distinguishes the
requested policy (`diagnostics.threaded`) from the actual mode
(`diagnostics.execution`).

All proposal draws happen on the coordinator before worker tasks start.
Threaded phases only evaluate the scalar target and proposal density over
already-drawn samples. Therefore those callables must be pure, deterministic,
thread-safe, and free of hidden mutable scratch state. Worker tasks never draw
from the prepared RNG.

## Devices

Preparation has no `device=` keyword and the one-shot form starts on CPU.
Before first execution, apply an MLDataDevices device to the complete prepared
sampler. CPU accepts generic and native proposals. CUDA accepts the documented
native subset with a device-compatible target and `threaded=true`; AMDGPU and
Metal are unclaimed. See [Accelerators](@ref) for the complete transfer example,
public preserving-device construction, resident-result rules, and generated
capability matrix. An accelerator whose public scalar policy is `Missing` is
rejected as `:scalar_policy_unspecified` before the source RNG advances.

## Troubleshooting

**`NaN` or target `+Inf`.** The target is outside the accepted log-density
domain. Trace the first failing logical sample reported by
[`SamplerExecutionError`](@ref). Preserve valid `-Inf` for zero target density,
but prevent undefined arithmetic before returning.

**Proposal `-Inf` at a generated sample.** `rand` produced a value that
`logdensityof` says has zero proposal density. Make the two proposal operations
describe the same support and normalized measure. Proposal `NaN` is likewise
invalid; proposal `+Inf` is accepted and gives zero importance weight when the
target is finite.

**All weights are `-Inf`.** Check proposal support and whether the target is
zero at every draw. The raw result remains inspectable, but normalized
summaries are undefined.

**Unsupported device.** Apply `cpu_device()` to the CPU-origin prepared sampler
before first execution when an independent CPU copy is wanted. Accelerator
destinations cannot be transferred again. CUDA requires the native proposal
path, `threaded=true`, a functional backend, and a device-compatible target.
There is no host fallback. See [Accelerators](@ref).

**Unstable return types or shapes.** Make every proposal draw return the same
scalar type, vector length, named-tuple keys, and numeric leaf types. Make every
target and proposal log-density return bound provably contained in
`Union{Float32,Float64}`. `Float32`, `Float64`, and
`Union{Float32,Float64}` are accepted; a bound containing `Float64` selects
`Float64` storage. Empty or unprovable inference bounds, broader bounds or
bounds containing another type, runtime values outside exact `Float32` or
`Float64`, and conversions that would narrow the selected storage are rejected.

**Prepared-sampler re-entry.** Wait for the current call to finish or use a
separate prepared sampler with a separate RNG. The handle intentionally has no
lock or concurrent top-level execution mode.

## Analytic validation and local performance

The runnable `validation/reproducers/plain_is.jl` labels and checks two
analytic identities:
proposal equal to normalized target, and a Gaussian weighted-mean and
normalizer identity. It records the fixed seed, Julia and dependency versions,
scalar type, budgets, tolerances, rationale, and command, and exits with an
error on failure.

`benchmark/plain_is.jl` uses BenchmarkTools to measure preparation, warm
prepared execution, throughput, allocations, and allocated bytes. Its
`--smoke` mode checks wiring quickly. It prints measurements without asserting
machine-specific timing thresholds.
