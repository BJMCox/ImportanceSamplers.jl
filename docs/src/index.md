# ImportanceSamplers.jl

`ImportanceSamplers.jl` provides a small, explicit CPU implementation of plain
importance sampling. You supply a normalized proposal, an unnormalized target
**log density**, an RNG, and an exact sample count. The result keeps the samples
and the canonical raw log weights so that the estimator remains inspectable.

## Five-minute quick start

This complete example defines a normalized Gaussian proposal without
`Distributions.jl`. The two required proposal operations are visible:
`Random.rand` draws from the proposal, and
`DensityInterface.logdensityof` evaluates the density of that same normalized
measure.

```jldoctest quickstart
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

(length(result), all(iszero, result.logweights), lognormalizer(result))

# output

(1000, true, 0.0)
```

The target above is normalized only to make the first result easy to check. A
real target may be unnormalized. It must still return the **log** density; the
sampler never applies `log` for you. `result.logweights` stores
`logtarget(x) - logdensityof(proposal, x)`, not normalized probabilities. Call
[`normalized_weights`](@ref) only when you need weights that sum to one.

## A target with context

Use `logtarget(x, p)` when data or constants belong in a separate concrete
context. Pass that context between the target and algorithm arguments:

```jldoctest quickstart
context = (mean=0.5, scale=0.8)
function contextual_target(x, p)::Float64
    z = (x - p.mean) / p.scale
    -0.5 * abs2(z) - log(p.scale) - 0.5 * log(2pi)
end

contextual_result = importance_sample(
    Xoshiro(43),
    contextual_target,
    context,
    ImportanceSampling(proposal; nsamples=256);
    threaded=false,
)

(length(contextual_result), contextual_result.diagnostics.execution)

# output

(256, :serial)
```

The context-free and contextual forms have identical estimator semantics. The
proposal—not the context or the data—defines the sampled shape.

## Prepared device transfer

Preparation always starts on the CPU. Apply an explicit MLDataDevices device
to the complete prepared sampler before its first execution:

```julia
using MLDataDevices

prepared = prepare_sampler(
    Xoshiro(44),
    contextual_target,
    context,
    ImportanceSampling(proposal; nsamples=32);
    threaded=false,
)
transferred = cpu_device()(prepared)
transferred_result = importance_sample!(transferred)

(transferred !== prepared, length(transferred_result)) # (true, 32)
```

The pipe form `prepared |> device` is equivalent. Transfer returns a distinct
prepared sampler with independent RNG and numerical state, leaving the source
valid. RNGs use their standard `copy` operation; transfer raises a typed error
when that operation cannot return independent RNG state. Transfer is rejected
when callable target state contains a reachable opaque closure that the
standard device traversal would reconstruct. It is also rejected after the
source's first execution begins. Accelerator devices currently fail during
transfer because accelerator RNG buffers and execution are not implemented
yet; no accelerator request falls back to CPU.

## Choose the next page

- Read [Plain importance sampling](@ref) for the mathematical contract,
  prepared reuse, threading rules, result shapes, failure behavior, and
  troubleshooting.
- Run `validation/reproducers/plain_is.jl` for deterministic analytic
  identities with recorded provenance.
- Run `benchmark/plain_is.jl --smoke` for a fast benchmark wiring check, or
  omit `--smoke` for the normal local run budget.

## Checked capability

This row is produced during the documentation build. The build constructs and
runs both public preparation forms; it does not rely on a hand-maintained
registry.

```@eval
Main.PLAIN_IS_CAPABILITY_TABLE
```

CPU transfer is implemented today. Accelerator transfer has a checked API but
remains unavailable until the required RNG buffers and execution path exist.
See [Future accelerator work](@ref) for that boundary.

## Runnable analytic validation

From the package root, run the checked reproducer with:

```sh
julia --project=validation validation/reproducers/plain_is.jl
```

The source is `validation/reproducers/plain_is.jl`. It uses only local Gaussian
math and fails loudly if either labeled analytic identity misses its recorded
deterministic tolerance.

## Public API

```@docs
AbstractImportanceSampler
AbstractProposalFamily
AbstractRadialProposalFamily
AbstractSampleTransform
DiagonalGaussian
FactorGaussian
IdentityTransform
ImportanceSampling
IntervalTransform
InvalidTransformError
LogTarget
PositiveTransform
ProductProposal
SimplexTransform
SoftplusTransform
SphericalGaussian
TransformedProposal
prepare_sampler
importance_sample
importance_sample!
WeightedSamples
WeightedSampleView
normalized_weights
lognormalizer
AllZeroWeightsError
SamplerBusyError
SamplerAlreadyExecutedError
SamplerDeviceError
SamplerExecutionError
```
