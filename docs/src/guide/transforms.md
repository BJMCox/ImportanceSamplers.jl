# [Constraints and named parameters](@id transforms-guide)

Use `transform=` when the target has constrained or named parameters.
The sampler works in flat numerical coordinates and presents logical parameters to the target.

## A simplex, a positive value, and an unconstrained value

This target has three positive weights that sum to one, a positive rate, and a real offset.

```@example transforms
using ImportanceSamplers, Random, Statistics

layout = (
    weights=(1:2 => SimplexTransform(3)),
    rate=(3 => PositiveTransform()),
    offset=(4 => IdentityTransform()),
)

function logtarget(theta, p)
    value = -theta.rate - theta.offset^2 / 2
    for i in eachindex(p.alpha)
        value += (p.alpha[i] - 1) * log(theta.weights[i])
    end
    return value
end

p = (; alpha=[2.0, 3.0, 4.0])
algorithm = AMIS(
    SphericalGaussian(zeros(4), 1.0);
    rounds=3, round_size=2_000,
)
prepared = prepare_sampler(
    Xoshiro(42), logtarget, p, algorithm;
    transform=layout,
)
samples = importance_sample!(prepared)

mean(samples)
```

The target receives a named tuple with `weights`, `rate`, and `offset`.
The reference means are `[2, 3, 4] / 9`, one, and zero.

The proposal has four coordinates, although the logical sample contains five scalar values:

| Numerical coordinates | Logical field | Transform |
|:--|:--|:--|
| `1:2` | Three simplex weights | `SimplexTransform(3)` |
| `3` | Positive rate | `PositiveTransform()` |
| `4` | Real offset | `IdentityTransform()` |

An integer selector gives a scalar field. A range gives a vector field.
A `K`-component simplex consumes `K - 1` coordinates.

## Inspect the returned shape

```@example transforms
(
    weights=size(samples.samples.weights),
    rates=length(samples.samples.rate),
    first=samples[1].sample,
)
```

Returned samples use logical coordinates. Weights and provenance remain aligned.
A full Gaussian or Student-t factor can capture correlations between named fields.
Names do not impose independence.

The flat selectors must cover every numerical coordinate exactly once.
Use explicit identity fields for unconstrained coordinates.
Omitting `transform` leaves the whole sample unchanged.

## Choose a scalar constraint

| Support | Constructor | Map |
|:--|:--|:--|
| Real line | `IdentityTransform()` | ``z`` |
| Positive | `PositiveTransform()` | ``e^z`` |
| Positive | `SoftplusTransform()` | ``\log(1+e^z)`` |
| Above `a` | `IntervalTransform(a, nothing)` | ``a+e^z`` |
| Below `b` | `IntervalTransform(nothing, b)` | ``b-e^z`` |
| Between `a` and `b` | `IntervalTransform(a, b)` | Scaled logistic |

Endpoints are excluded. For example, `IntervalTransform(0.0, 10.0)` maps into `(0, 10)`.

Use bounds with the coordinate precision, such as `IntervalTransform(0f0, 1f0)`.
Nonfinite or rounded boundary values fail instead of being silently clipped.

For a Gaussian base, the exponential map gives a lognormal upper tail.
Softplus gives a lighter, Gaussian-like upper tail.
Neither choice guarantees finite importance-weight variance.

## Apply the Jacobian once

For a map ``\theta=T(z)``, the sampler evaluates

```math
\log\pi(T(z))+\log|\det J_T(z)|.
```

Your target returns the log density in logical coordinates.
Do not add the same Jacobian yourself.

`current_proposal(prepared)` returns the fitted proposal in numerical coordinates.
Keep the layout when using that proposal in another sampler.

If another package already supplies an unconstrained target with its Jacobian,
do not apply the transform again.

## Transform the proposal instead

`TransformedProposal` owns the change of variables at the proposal boundary.
This is useful for fixed IS with a constrained proposal:

```@example transforms
proposal = TransformedProposal(
    SphericalGaussian(0.0, 1.0),
    PositiveTransform(),
)
positive = importance_sample(
    Xoshiro(9), x -> -x,
    ImportanceSampling(proposal; nsamples=5_000),
)

mean(positive)
```

The target here is an unnormalized exponential density on the positive line.
The proposal evaluates its normalized density with the inverse Jacobian.
Do not also pass the same transformation through `transform=`.

For CPU-only independent named blocks, combine `ProductProposal` and `TransformedProposal`:

```@example transforms
structured = TransformedProposal(
    ProductProposal((
        scale=SphericalGaussian(0.0, 1.0),
        offset=SphericalGaussian(0.0, 1.0),
    )),
    (scale=PositiveTransform(),),
)

rand(Xoshiro(9), structured)
```

An omitted known product field uses identity. Here `offset` stays unconstrained.
This omission rule applies to named product fields, not to flat selector layouts.

## Gradients, devices, and the simplex measure

Native flat layouts work with adaptive methods and supported devices.
`ProductProposal` remains CPU-only.
Explicit named gradients differentiate the logical target.
The package supplies the transform pullback and Jacobian derivative.
See [Gradients](@ref gradients-guide).

A simplex density uses the first-coordinate measure
``dx_1\cdots dx_{K-1}`` with ``x_K=1-\sum_{i<K}x_i``.
The orthonormal-logit construction has forward log Jacobian

```math
\log|J|=\tfrac12\log K+\sum_{i=1}^K\log x_i.
```

The constant matters for normalizers. It does not disappear from the proposal density.
The [simplex reproducer](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/validation/reproducers/simplex_transform.jl)
checks this measure against a normalized Dirichlet density.
