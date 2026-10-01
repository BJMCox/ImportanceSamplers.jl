# [Gradients](@id gradients-guide)

Only `FirstOrderGRAMIS` needs target gradients among the implemented samplers.
Value-only operations still use the log target without requesting a gradient.

## Supply a gradient

```@example gradients
using ImportanceSamplers, Random, Statistics

logtarget(x, p) = -sum(abs2, x .- p.centre) / 2

function gradient!(g, x, p)
    g .= p.centre .- x
    return nothing
end

target = LogTarget(logtarget; grad=gradient!)
p = (; centre=[0.5, -0.5])
bank = ProposalBank([
    SphericalGaussian([-1.0, 0.0], 1.5),
    SphericalGaussian([ 1.0, 0.0], 1.5),
])
algorithm = FirstOrderGRAMIS(
    bank; rounds=3, round_size=2_000, repulsion_strength=0.0,
)
samples = importance_sample(Xoshiro(42), target, p, algorithm)

mean(samples)
```

The callback fills the gradient of the log target with respect to `x`.
The context `p` stays constant.

Without a context, use `gradient!(g, x)`.
On CPU, an out-of-place `gradient(x)` or `gradient(x, p)` is also accepted.
An applicable in-place form has priority.

A device gradient must support the supplied device arrays.
Do not retain its borrowed input or output buffers.

## Select automatic differentiation

Install an AD backend in your project, then pass its ADTypes selector through `LogTarget`.
For example, with ForwardDiff installed:

```julia
using ADTypes, ForwardDiff

target = LogTarget(logtarget, AutoForwardDiff())
samples = importance_sample(Xoshiro(42), target, p, algorithm)
```

The package uses DifferentiationInterface on CPU.
It prepares derivative state for the chosen target and input.
It does not choose a backend automatically.

An explicit `grad=` takes priority over `adtype`.
A bare first-order LogDensityProblems target can supply its own value/gradient interface.
Failure in the selected source does not trigger a different source.

## Differentiate named parameters

For a target taking `theta = (; rate, offset)`, a logical gradient has matching fields:

```@example named_gradients
using ImportanceSamplers, Random, Statistics

layout = (
    rate=(1 => PositiveTransform()),
    offset=(2 => IdentityTransform()),
)
named_target(theta) = -theta.rate - theta.offset^2 / 2

function named_gradient!(g, theta)
    g.rate[] = -one(theta.rate)
    g.offset[] = -theta.offset
    return nothing
end

bank = ProposalBank([
    SphericalGaussian([-0.5, -0.5], 1.0),
    SphericalGaussian([ 0.5,  0.5], 1.0),
])
samples = importance_sample(
    Xoshiro(7), LogTarget(named_target; grad=named_gradient!),
    FirstOrderGRAMIS(bank; rounds=3, round_size=2_000, repulsion_strength=0.0);
    transform=layout,
)

mean(samples)
```

Scalar gradient fields are zero-dimensional writable views, so use `g.rate[]`.
Vector fields use ordinary indexed writes.
For a simplex of length `K`, supply `K` logical derivatives.

Differentiate only the logical log target.
The package adds the transform pullback and the log-Jacobian derivative.
CPU AD instead differentiates the composed flat-coordinate target.

## Use gradients on accelerators

| Execution | Gradient route |
|:--|:--|
| Native CUDA | Device-compatible in-place callback or supported reverse-mode Enzyme |
| Metal | Device-compatible in-place callback |
| Reactant GPU | Supported compiled gradient, including `AutoEnzyme` |
| Ordinary CPU | Explicit callbacks, LogDensityProblems, or a supported DI backend |

A scalar target that runs on GPU need not be differentiable by the GPU AD backend.

For native CUDA with Enzyme, structured contexts can require explicit runtime activity:

```julia
using ADTypes, Enzyme

ad = AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
target = LogTarget(logtarget, ad)
```

The supplied mode is preserved. The package batches scalar target gradients internally.
It does not copy each sample or gradient to CPU.

`batch=` controls value evaluation, not a separate batch-gradient interface.
Automatic differentiation still uses the scalar target and may evaluate its primal.

See [Devices](@ref devices-guide) for setup and support limits.
The [DI documentation](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/stable/#Compatibility)
describes backend-specific restrictions.
