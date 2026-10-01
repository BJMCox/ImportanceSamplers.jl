# [Custom proposals and transitions](@id extension-guide)

Use the existing interfaces when the built-in proposals or LAIS transitions do not fit your model.
These extension points do not change the public sampling workflow.

## Define a normalized CPU proposal

This example samples a uniform distribution on `(-1, 1)`:

```@example extension
using ImportanceSamplers, Random, DensityInterface, Statistics

struct UniformInterval end

Random.rand(rng::Random.AbstractRNG, ::UniformInterval) = 2rand(rng) - 1
DensityInterface.logdensityof(::UniformInterval, x) =
    -1 <= x <= 1 ? -log(2.0) : -Inf

samples = importance_sample(
    Xoshiro(42), x -> -1 <= x <= 1 ? -x^2 : -Inf,
    ImportanceSampling(UniformInterval(); nsamples=5_000),
)

mean(x -> x^2, samples)
```

This target is restricted to the proposal's support.
It does not describe a Gaussian over the entire real line.

The draw and density operations must describe the same normalized measure.
The package cannot prove that property for a user-defined type.

Generic proposals support CPU execution.
Adaptive methods require their supported native proposal forms.
A generic proposal does not become GPU-compatible through an Adapt rule alone.

## Preserve the sample structure

A draw may be a scalar, a dense vector, or a named tuple of scalar/vector leaves.
Every draw must retain its structure, dimensions, and leaf types.

Preparation does not consume a probe draw.
The first draw is retained in the result.
Target and density functions must be pure and thread-safe for threaded execution.

A generic proposal bank also needs one concrete proposal element type.
Use a concrete wrapper type when several instances differ only in parameters.

## Extend the LAIS upper transition

Subtype `AbstractMCMCTransition` and implement the existing namespaced interface:

| Method | Responsibility |
|:--|:--|
| `ImportanceSamplers.prepare_transition(config, centres, target, L)` | Build state without RNG use and retain the supplied centre storage |
| `ImportanceSamplers.transition_centres(state)` | Return the centre batch without copying |
| `ImportanceSamplers.transition!(state, target, rng, execution, transfers)` | Perform one upper round and update cumulative counters |
| `Base.copyto!(destination, source)` | Copy persistent state for rollback |
| `ImportanceSamplers.transition_diagnostics(state)` | Return cumulative work counts |
| `ImportanceSamplers.retarget_transition(config, state, destination)` | Build a configuration from learned parameters |

Here `L` is the log-density scalar type.
Diagnostics contain `initial_target_evaluations`, `warmup_target_evaluations`,
`production_target_evaluations`, `warmup_proposals`, `production_proposals`, and `accepted`.

Preparation uses `deepcopy` to separate committed and working state.
Custom device execution also needs Adapt support, preserved centre aliases,
and the package's device preflight hook.
It must not silently transfer each chain or sample to CPU.

These are implementation extension points, not a compatibility promise for arbitrary external MCMC packages.
The [built-in transitions](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/src/mcmc_transitions.jl)
provide the current concrete contract.

## Integrate a model package

An adapter must define the sampled coordinates, the log target, the context, and the proposal.
Exactly one side owns each constraint transform and Jacobian.

No BAT, Wren, or Turing adapter ships with this package.
Objects exposing supported density interfaces can already use those interfaces directly.
