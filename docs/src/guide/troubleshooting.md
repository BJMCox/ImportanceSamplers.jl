# [Troubleshooting](@id troubleshooting-guide)

Inspect the exception and its cause before increasing the sampling budget.
Many failures indicate a model, support, or execution mismatch.

## The weights concentrate on a few samples

Compute `inv(sum(abs2, normalized_weights(samples)))` and inspect the largest normalized weights.

Check proposal location, covariance, tails, and mode coverage.
Use an independent pilot or a broader family when appropriate.
A larger count does not repair missing support.

Location-only methods do not tune scales.
DM-PMC, APIS, and LAIS keep their lower proposal scales fixed.
LAIS RAM tunes upper MCMC increments, not those lower scales.

## All weights are zero

A valid `-Inf` target value means zero density.
For plain IS, an all-zero run retains its samples and reports `lognormalizer == -Inf`.
Normalized summaries throw `AllZeroWeightsError`.

Adaptive methods may fail earlier because a round or local fit has no usable mass.
Check constraints, proposal support, and target underflow.

## A target returns NaN or +Inf

`SamplerExecutionError` reports a phase and sample index.
A batch-wide failure uses index zero because no single sample is identified.
Its captured exception retains the original error and backtrace.

Use a stable log-density expression.
Do not replace invalid arithmetic with a finite number merely to continue sampling.

## Covariance adaptation fails

Inspect the method-specific round error, its `phase`, and its `cause`.
Check local sample counts, weight concentration, and repeated or degenerate samples.

CAIS uses exact covariance replacement and no ridge.
Its ESS threshold does not guarantee a positive-definite covariance.
AMIS, N-PMC, and first-order GRAMIS use their documented stabilization rules.

A failed adaptive call preserves the pre-call proposal state.
Its RNG still advances. Repeating the call is not an exact replay.

## A device transfer fails

`SamplerDeviceError.reason` identifies the rejected capability.
Common causes include:

- A generic or product proposal on an accelerator.
- A host array captured by a closure rather than passed through `p`.
- An unspecified device precision policy.
- `threaded=false` for accelerator execution.
- An unsupported gradient or Reactant CPU kernel.

Consult [Devices](@ref devices-guide).
An unsupported contract has no silent CPU fallback.

## A result cannot be indexed

Transfer device results to CPU before scalar indexing, iteration, quantiles, or medians.
Array slicing and resident weighted moments use separate supported paths.

## A prepared sampler cannot be moved or entered

Transfer it before the first execution begins.
An accelerator-prepared sampler cannot itself migrate to another device.
Request an explicit result or proposal copy instead.

A prepared sampler is mutable and non-reentrant.
Use one sampler and RNG per concurrent task.

## Report a reproducible failure

Include a small target, proposal, seed, count schedule, and device selection.
Record Julia and dependency versions and the package revision.
For a numerical error, include the expected mathematical quantity and observed result.
Do not include private data or credentials.
