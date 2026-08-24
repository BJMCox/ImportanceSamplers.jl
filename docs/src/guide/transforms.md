# Transforms

[`TransformedProposal`](@ref) owns the map from unconstrained proposal
coordinates to the logical value seen by the target. Its normalized density is

```math
\log q_x(x) = \log q_z(z) - \log |J(z)|.
```

This ownership is important for future BAT and Turing adapters: an adapter must
translate parameter declarations into one ImportanceSamplers transform at its
boundary. The target must then return a density against the logical reference
measure and must not apply the same Jacobian again.

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
finite; invalid generated values fail the complete estimator with a located
[`InvalidTransformError`](@ref).

## Structured parameters

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
