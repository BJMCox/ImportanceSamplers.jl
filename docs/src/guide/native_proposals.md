# Native proposals

ImportanceSamplers includes normalized Gaussian proposals for the portable CPU
and CUDA execution path. They accept only `Float32` or `Float64`, copy their
input arrays, and reject nonfinite locations or invalid scales when constructed.

## Gaussian constructors

All scale arguments are standard-deviation factors, not variances or
precisions:

```jldoctest native_gaussians
using DensityInterface
using ImportanceSamplers
using LinearAlgebra

scalar = SphericalGaussian(0.0, 2.0)
spherical = SphericalGaussian([0.0, 1.0], 2.0)
diagonal = DiagonalGaussian([0.0, 1.0], [0.5, 2.0])

covariance = [4.0 1.0; 1.0 2.0]
factor = FactorGaussian(zeros(2), cholesky(Symmetric(covariance)))

(
    rand_dimension=length(rand(spherical)),
    normalized_density=DensityInterface.logdensityof(factor, zeros(2)) < 0,
)

# output

(rand_dimension = 2, normalized_density = true)
```

- `SphericalGaussian(mu, sigma)` uses covariance `sigma^2` for a scalar and
  `sigma^2 I` for a vector.
- `DiagonalGaussian(mu, scales)` uses covariance
  `Diagonal(scales .^ 2)`.
- `FactorGaussian(mu, L)` uses `x = mu + L*z`, `z ~ N(0, I)`, and covariance
  `L * L'`. Pass a lower-triangular matrix with positive diagonal or a
  `Cholesky` factorization.

If you start from a covariance matrix, factor it once with
`cholesky(Symmetric(covariance))`. Do not invert the covariance: construction,
drawing, and density evaluation use the factor directly, including triangular
solves and `sum(log, diag(L))` for the normalization constant.

The runnable
`validation/reproducers/native_gaussian.jl`
checks formulas, moments, inference, and allocation probes without a distribution
dependency. See [Transforms](@ref) to map these unconstrained proposals into
logical parameter spaces.

## Distributions.jl proposals

Loading Distributions.jl activates an optional extension. Its normal proposals
work directly in sampler constructors:

```julia
using Distributions
using ImportanceSamplers

proposal = MvNormal([0.0, 0.0], [1.0 0.4; 0.4 2.0])
algorithm = AMIS(proposal; rounds=4, round_size=1_000)
```

The extension converts `Normal` to `SphericalGaussian`. It converts an
`MvNormal` with spherical, diagonal, or full covariance to
`SphericalGaussian`, `DiagonalGaussian`, or `FactorGaussian`, respectively.
The conversion copies the parameters and factors a full covariance once. It
never forms a covariance inverse.

The native proposal rules still apply. Parameters must use `Float32` or
`Float64`, and scales must be finite and positive. Canonical normal forms and
other Distributions.jl families remain generic CPU proposals. Adaptive methods
that require native Gaussian storage reject those other forms.
