# Native proposals

ImportanceSamplers includes normalized Gaussian and Student-t proposals for the
portable CPU and CUDA execution path. They accept only `Float32` or `Float64`,
copy their input arrays, and reject nonfinite locations or invalid scales.

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

## Student-t constructors

Use Student-t proposals when the target needs heavier tails:

```jldoctest native_student_t
using ImportanceSamplers
using LinearAlgebra

spherical = SphericalStudentT(5.0, zeros(2), 2.0)
diagonal = DiagonalStudentT(5.0, zeros(2), [1.0, 2.0])
factor = FactorStudentT(5.0, zeros(2), cholesky(Symmetric([1.0 0.4; 0.4 2.0])))

(length(rand(spherical)), length(rand(diagonal)), length(rand(factor)))

# output

(2, 2, 2)
```

The first argument is the positive degrees of freedom `nu`. The scale arguments
define the elliptical scale matrix `Sigma`, not the covariance. For `nu > 2`,
the covariance is `nu / (nu - 2) * Sigma`. Every multivariate draw uses one
shared radial scale, including diagonal proposals. Setting `nu = 1` gives a
Cauchy proposal.

Proposal construction, storage and arithmetic preserve the supplied `Float32`
or `Float64` type. Construction computes the fixed dimension/degrees-of-freedom
normalization term once, without wider intermediates. A gamma recurrence avoids
subtracting large log-gamma values. Covariance adaptation then updates only the
scale determinant in the working array type. Precision changes require an
explicit conversion, such as a device configured with another element type.

CUDA uses the same constructors and keeps draws and density evaluation on the
device. General degrees of freedom use eight independent gamma candidates per
sample. The first accepted candidate has the exact gamma law. If all candidates
fail, the run throws `SamplerExecutionError` instead of returning a biased draw.
This path stores `d + 8` normals and eight uniforms per sample, plus one extra
uniform when `nu < 2`. The `nu = 1` path stores only `d + 1` normals.

Static MIS and adaptive samplers accept Student-t proposals on CPU and CUDA.
Choose the family in the proposal constructor, not in a sampler-specific option:

```jldoctest adaptive_student_t
using ImportanceSamplers, Random

proposal = FactorStudentT(5.0, [-1.0, 1.0], [2.0 0.0; 0.3 1.5])
algorithm = AMIS(proposal; rounds=3, round_size=256)
prepared = prepare_sampler(Xoshiro(42), x -> -sum(abs2, x) / 2, algorithm)
result = importance_sample!(prepared)
length(result)

# output

768
```

For a population method, pass a `ProposalBank` of the same constructors.
Each packed bank uses one radial family and floating type. Student-t components
may have different degrees of freedom, scales, locations, and permitted masses.
Mixed Gaussian/Student-t packed banks are not supported. Transfer the complete
prepared sampler with `prepared |> device`; targets, data, random buffers,
proposal parameters, and adaptation workspaces follow the existing device contract.

| Methods | Student-t adaptation |
|:--|:--|
| DM-PMC, GR-PMC, LR-PMC, APIS, LAIS | Update locations. Preserve each supplied scale and degrees of freedom. Any finite `nu > 0` is allowed. |
| AMIS, NPMC, CAIS, FirstOrderGRAMIS | Require `nu > 2`. Preserve degrees of freedom and fit location/covariance using the method's existing update. |

Covariance-fitting methods compute in covariance units, including regularization,
preconditioning, and repulsion where applicable. They store the fitted covariance
`C` as Student-t scale `(nu - 2) / nu * C`. The corresponding Cholesky factor
multiplier is `sqrt((nu - 2) / nu)`. Vector proposals learn full covariance even
when initialized with spherical or diagonal scales.

These are fixed-degrees-of-freedom family extensions. They do not fit `nu`,
perform Student-t maximum-likelihood fitting, or change a method's weighting rule.
`current_proposal`, repeated calls, and `retarget` retain the learned family.
Device execution has no per-sample host reads. Student-t banks use separate
resident radial buffers; Gaussian banks allocate none.

The runnable [adaptive Student-t example](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/examples/adaptive_student_t.jl)
uses LAIS on a curved target, converts a desired covariance factor to Student-t
scale, and computes weighted expectations with analytic reference values.
It also shows explicit prepared-sampler CUDA transfer. Broader tails can protect
against extreme importance ratios, but do not guarantee better accuracy per second.

The [CUDA reproducer](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/validation/reproducers/cuda_student_t.jl)
checks fractional, Cauchy, factor, and transformed proposals on an A100.

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

It also converts `TDist`, `Cauchy`, `IsoTDist`, `DiagTDist`, and full
multivariate Student-t distributions to the matching native Student-t form.
The Distributions.jl scale matrix remains an elliptical scale matrix.

The native proposal rules still apply. Parameters must use `Float32` or
`Float64`, and scales must be finite and positive. Canonical normal forms and
other Distributions.jl families remain generic CPU proposals. Adaptive methods accept the supported native Gaussian and Student-t forms.
Other families remain subject to each method's proposal requirements.
