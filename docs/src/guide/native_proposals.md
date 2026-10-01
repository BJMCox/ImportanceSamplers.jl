# [Proposals](@id proposals-guide)

A proposal supplies normalized draws and their log density.
Its support must cover the regions that contribute to the target integral.

## Start with a Gaussian

```@example proposals
using ImportanceSamplers, Random, LinearAlgebra

scalar = SphericalGaussian(0.0, 2.0)
spherical = SphericalGaussian(zeros(3), 2.0)
diagonal = DiagonalGaussian(zeros(3), [0.5, 1.0, 2.0])

C = [1.0 0.6 0.0; 0.6 2.0 0.2; 0.0 0.2 1.0]
correlated = FactorGaussian(zeros(3), cholesky(Symmetric(C)))

rand(Xoshiro(7), correlated)
```

| Constructor | Scale argument | Gaussian covariance |
|:--|:--|:--|
| `SphericalGaussian(mu, sigma)` | Standard deviation | ``\sigma^2 I`` |
| `DiagonalGaussian(mu, scales)` | Marginal standard deviations | `Diagonal(scales .^ 2)` |
| `FactorGaussian(mu, L)` | Lower factor or `Cholesky` | `L * L'` |

The spherical form accepts a scalar or a vector location.
The diagonal and factor forms take vector locations.

Factor a covariance once. Do not pass the covariance itself as `L`.
The implementation uses multiplication and triangular solves, not a covariance inverse.

## Use heavier tails

```@example proposals
proposal = FactorStudentT(
    5.0, zeros(3),
    cholesky(Symmetric(C)),
)
rand(Xoshiro(7), proposal)
```

Student-t constructors follow the same layouts:

```julia
SphericalStudentT(nu, mu, sigma)
DiagonalStudentT(nu, mu, scales)
FactorStudentT(nu, mu, L)
```

The first argument is the positive degrees of freedom `nu`.
The factors define a scale matrix ``S``, not the covariance:

```math
\operatorname{Cov}(X)=\frac{\nu}{\nu-2}S,\qquad \nu>2.
```

To obtain a desired covariance factor `Lcov`, use
`sqrt((nu - 2) / nu) * Lcov` as the Student-t factor.

A multivariate Student-t draw uses one shared random radial scale.
A diagonal scale therefore does not make its coordinates independent.

The adaptive methods keep `nu` fixed.
AMIS, NPMC, CAIS, and FirstOrderGRAMIS require `nu > 2` because they fit covariance.
Location-only methods accept any finite `nu > 0`.
None fits `nu` by maximum likelihood.

## Form a proposal population

```@example proposals
bank = ProposalBank([
    SphericalGaussian([-2.0, 0.0, 0.0], 1.0),
    SphericalGaussian([ 2.0, 0.0, 0.0], 1.5),
], [1.0, 2.0])

bank.masses
```

`ProposalBank` copies the inputs and normalizes the masses.
Omitting masses gives equal masses. Some adaptive methods require equal positive masses.

A bank is sampler configuration, not a distribution.
It has no standalone `rand` or `logdensityof` method.
The [MIS scheme](@ref static-mis) defines assignment and weighting.

Native banks use one scalar/vector layout, dimension, floating type, and radial family.
A Gaussian/Student-t mixed bank is not supported.
Student-t components may have different degrees of freedom.

## Preserve precision

Native proposals support `Float32` and `Float64` and copy their input arrays.
Use matching types for locations, scales, factors, and Student-t degrees of freedom.

For a wholly single-precision bank, supply single-precision masses too:

```@example proposals
bank32 = ProposalBank([
    SphericalGaussian(Float32[-1, 0], 1f0),
    SphericalGaussian(Float32[ 1, 0], 1f0),
], Float32[1, 1])

eltype(bank32.masses)
```

Masses participate in numerical promotion.
A device changes precision only when you request that conversion explicitly.

## Use Distributions.jl

Loading Distributions.jl activates native conversion for `Normal`, conventional
`MvNormal` forms, `TDist`, `Cauchy`, and supported multivariate Student-t forms.

```julia
using Distributions, ImportanceSamplers

proposal = MvNormal(zeros(3), [1.0 0.6 0.0; 0.6 2.0 0.2; 0.0 0.2 1.0])
algorithm = AMIS(proposal; rounds=4, round_size=2_000)
```

Other normalized distributions remain generic CPU proposals.
They do not gain adaptive or GPU support merely by loading the extension.

A mixture distribution remains one atomic proposal.
To expose its components to MIS, construct
`ProposalBank(components(mixture), probs(mixture))` explicitly.

## Define a custom CPU proposal

Implement these two operations for the same normalized measure:

```julia
Random.rand(rng, proposal)
DensityInterface.logdensityof(proposal, sample)
```

A draw may be a scalar, vector, or named tuple of scalar/vector leaves.
Every draw must keep the same structure, numeric types, and dimensions.

See [Custom proposals and transitions](@ref extension-guide) for a complete example.
Use [transforms](@ref transforms-guide) for constrained support.
