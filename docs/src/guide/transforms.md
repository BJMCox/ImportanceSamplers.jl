# Transforms

Use `transform=` on a prepared sampler to keep adaptation in flat coordinates
while exposing logical parameters to the target. Alternatively,
[`TransformedProposal`](@ref) owns the map at the proposal boundary. Its
normalized density is

```math
\log q_x(x) = \log q_z(z) - \log |J(z)|.
```

BAT or Wren integration may keep coordinates and Jacobian ownership in the host
package, or construct one [`TransformedProposal`](@ref) at the integration
boundary. Whichever side owns the transform applies its Jacobian; the other side
must not transform the same parameters or apply that Jacobian again.

## Scalar constraints

The built-in scalar transforms are:

| Logical support | Transform |
|:--|:--|
| unconstrained | [`IdentityTransform`](@ref) |
| positive | [`PositiveTransform`](@ref) or [`SoftplusTransform`](@ref) |
| lower bounded | [`IntervalTransform`](@ref) with `(lower, nothing)` |
| upper bounded | [`IntervalTransform`](@ref) with `(nothing, upper)` |
| bounded | [`IntervalTransform`](@ref) with `(lower, upper)` |

With a Gaussian base, `PositiveTransform` gives a lognormal upper tail, while
`SoftplusTransform` gives a Gaussian-like upper tail. Proposal tails must cover
target tails for stable importance weights, so choose between them with the
target's upper tail in mind. Neither transform alone guarantees finite weight
variance. Both have the same first-order behavior near zero and can still be
too light there for targets with substantial boundary mass.

Endpoints are excluded. Transform inputs, outputs, and log Jacobians must remain
finite. Invalid values from `TransformedProposal` fail the complete estimator
with a located [`InvalidTransformError`](@ref).

## Named targets with adaptive samplers

Use `transform=` on [`prepare_sampler`](@ref) or [`importance_sample`](@ref) to
give the target named parameters while the algorithm keeps flat coordinates.
This works with Base IS, static MIS, AMIS, DM-PMC, APIS, CAIS, NPMC, LAIS, and
FirstOrderGRAMIS. Names do not imply independent proposals. A full factor can
capture correlations between fields when the method adapts covariance.
Location-only methods keep their original covariance rule.

For numerical coordinates `z` and logical parameters `theta = T(z)`, the sampler
evaluates `logtarget(theta, p) + logabsjac(T, z)`. Its raw log weights subtract
the numerical proposal denominator from that value. The returned samples contain
`theta`; mapping results does not change weights or round/proposal provenance.

This example has four sampling coordinates and five logical scalar values:
three simplex weights, a positive scale, and an unconstrained offset.

```jldoctest named_adaptive
using ImportanceSamplers, Random, LinearAlgebra

layout = (weights=(1:2=>SimplexTransform(3)),
          scale=(3=>PositiveTransform()), offset=(4=>IdentityTransform()))

function named_logtarget(theta, p)
    value = zero(theta.scale)
    for i in eachindex(p.alpha)
        value += (p.alpha[i] - 1) * log(theta.weights[i])
    end
    return value - theta.scale / p.scale - (theta.offset - p.offset)^2 / 2 +
        p.coupling * theta.offset * theta.weights[1]
end

p = (alpha=[2.0, 3.0, 4.0], scale=1.0, offset=0.5, coupling=0.2)
factor = Matrix{Float64}(I, 4, 4)
algorithm = AMIS(FactorGaussian(zeros(4), factor); rounds=3, round_size=256)
prepared = prepare_sampler(Xoshiro(7), named_logtarget, p, algorithm; transform=layout)
samples = importance_sample!(prepared)
(size(samples.samples.weights), length(samples.samples.scale), length(samples.logweights))

# output

((3, 768), 768, 768)
```

An integer selector gives a scalar field. A range with `IdentityTransform()`
gives a vector field. `SimplexTransform(K)` consumes `K-1` coordinates and
returns a length-`K` vector. Selectors must cover the numerical dimension exactly,
without overlap. This explicit flat layout has no inferred or omitted fields.
Without `transform=`, existing sampling behavior stays unchanged.
Interval endpoints must match the coordinate precision, for example
`IntervalTransform(0f0, 1f0)` for `Float32` proposals.

`current_proposal(prepared)` returns the learned proposal in numerical coordinates.
Use `retarget(rng, prepared, new_logtarget, new_p)` to retain the layout and
learned proposal while rebuilding target-dependent state. Retargeting applies to
the existing adaptive sampler types, not plain/static IS. Each run owns its
returned arrays. Changing a later run does not alter earlier samples.

On CUDA, transfer the whole prepared sampler **before its first run**:

```julia
using CUDA, MLDataDevices

CUDA.allowscalar(false)
physical = CUDA.device()
device = MLDataDevices.CUDADevice{typeof(physical),Nothing}(physical)
prepared = prepare_sampler(Xoshiro(7), named_logtarget, p, algorithm; transform=layout)
prepared = device(prepared)  # Also transfers p.alpha and the proposal workspaces.
samples = importance_sample!(prepared)
host_samples = cpu_device()(samples)  # Explicit transfer, only when needed.
```

The named leaf arrays, log weights, and provenance stay on the selected device.
Target calls use borrowed views or computed simplex values, not a host allocation
or transfer for each sample. Treat those inputs as read-only `AbstractVector`s.
Use `current_proposal(cpu_device(), prepared)` for an explicit CPU proposal snapshot.
On CUDA, an invalid named transform during target evaluation uses the target-failure
diagnostic. CPU evaluation and result mapping retain located transform errors.
A failed call does not commit new adaptation state.

### Named gradients

For an explicit gradient, differentiate only the user's logical log target.
The package applies the transform pullback and the log-Jacobian derivative.
Vector fields use indexed writes. Scalar gradient fields are writable
zero-dimensional views and use `[]`:

```julia
function named_gradient!(g, theta, p)
    for i in eachindex(p.alpha)
        g.weights[i] = (p.alpha[i] - 1) / theta.weights[i]
    end
    g.weights[1] += p.coupling * theta.offset
    g.scale[] = -1 / p.scale
    g.offset[] = -(theta.offset - p.offset) + p.coupling * theta.weights[1]
    return nothing
end

target = LogTarget(named_logtarget; grad=named_gradient!)
bank = ProposalBank([
    FactorGaussian([-0.2, 0.0, 0.0, 0.0], factor),
    FactorGaussian([ 0.2, 0.0, 0.0, 0.0], factor),
])
algorithm = FirstOrderGRAMIS(bank; rounds=3, round_size=256, repulsion_strength=0.0)
prepared = prepare_sampler(Xoshiro(7), target, p, algorithm; transform=layout)
```

The simplex gradient has `K` logical entries, despite its `K-1` sampling
coordinates. CPU workers and GPU proposal slots own separate gradient scratch.
Do not retain the borrowed parameters or gradient buffers after the callback.

For CPU automatic gradients, use `LogTarget(named_logtarget, adtype)` as usual.
The package differentiates the composed flat-coordinate target through
DifferentiationInterface. Its AD path materializes ordinary arrays for broad
backend compatibility; ordinary sample evaluation keeps the borrowed path.
ForwardDiff, Zygote, ReverseDiff, and explicit runtime-activity Enzyme have been
checked independently. The package preserves the supplied backend and mode.
CPU results do not imply support for Reactant or another GPU AD backend.

FirstOrderGRAMIS also supports named layouts with reverse-mode `AutoEnzyme` on
CUDA. It differentiates the composed target in a device batch, without moving
parameters or gradients to CPU. The target's operations must support Enzyme's
device differentiation. A target that runs on GPU does not necessarily meet that
AD contract. Explicit named gradients do not depend on automatic differentiation.

Bare LogDensityProblems targets retain their flat-vector convention. A named
layout with a bare LDP target is ambiguous and is rejected. Use an explicit
`LogTarget` when a callable accepts the logical named parameters.

## Structured proposals

A named product base can omit identity fields. Here `offset` is unconstrained,
so its omitted transform is filled in as [`IdentityTransform`](@ref):

```jldoctest structured_transform
using ImportanceSamplers
using Random

proposal = TransformedProposal(
    ProductProposal((
        weights=SphericalGaussian(zeros(2), 1.0),
        rate=SphericalGaussian(0.0, 1.0),
        offset=SphericalGaussian(0.0, 1.0),
    )),
    (
        weights=SimplexTransform(3),
        rate=PositiveTransform(),
    ),
)

sample = rand(Xoshiro(7), proposal)
(
    keys=keys(sample),
    simplex_length=length(sample.weights),
    simplex_sum=sum(sample.weights),
    positive=sample.rate > 0,
    unconstrained=sample.offset isa Float64,
)

# output

(keys = (:weights, :rate, :offset), simplex_length = 3, simplex_sum = 1.0, positive = true, unconstrained = true)
```

Named product layouts are CPU-only. For the CUDA path, use one native vector
Gaussian and a complete flat selector layout. Flat selectors must be disjoint,
in bounds, and collectively cover every coordinate; identity blocks are
explicit:

```julia
using LinearAlgebra

flat = TransformedProposal(
    FactorGaussian(zeros(4), Matrix{Float64}(I, 4, 4)),
    (
        weights=(1:2 => SimplexTransform(3)),
        rate=(3 => PositiveTransform()),
        offset=(4 => IdentityTransform()),
    ),
)
```

## Simplex reference measure

`SimplexTransform(K)` maps `K - 1` orthonormal coordinates into the sum-zero
logit subspace and applies softmax. The logical density is measured against the
ordinary first-coordinate measure

```math
dx_1\,\cdots\,dx_{K-1}, \qquad x_K = 1 - \sum_{k=1}^{K-1}x_k.
```

Against that measure, the full forward log Jacobian is

```math
\log |J(z)| = \tfrac{1}{2}\log K + \sum_{k=1}^{K}\log x_k.
```

The dimension-dependent `sqrt(K)` factor is part of the density. For the
three-component example it is `sqrt(3)`; dropping it shifts every raw log
weight and the estimated log normalizer. The runnable
`validation/reproducers/simplex_transform.jl`
checks a normalized Dirichlet identity against this exact measure.
