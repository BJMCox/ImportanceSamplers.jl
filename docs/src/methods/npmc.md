# Nonlinear population Monte Carlo

[`NPMC`](@ref) learns a Gaussian proposal from clipped importance weights.
Clipping limits the influence of a few large weights on proposal adaptation.
Returned samples keep their original importance weights for estimation.

This is an adaptation variant of Koblents and Míguez's
[N-PMC](https://arxiv.org/abs/1208.5600). The original method also uses transformed
weights in its reported estimates. This package deliberately uses ordinary
importance weights instead, and retains samples from every round.

## Example

```julia
using ImportanceSamplers, Random, Statistics

# Infer a correlated pair. This function returns an unnormalized log density.
logtarget(x) = -0.5 * (abs2(x[1] - 1) + abs2((x[2] + 1 - 0.8(x[1] - 1)) / 0.6))
proposal = SphericalGaussian([-2.0, 2.0], 3.0)
algorithm = NPMC(proposal; rounds=4, round_size=10_000)
prepared = prepare_sampler(Xoshiro(42), logtarget, algorithm)
samples = importance_sample!(prepared)
estimate = mean(samples)
learned = current_proposal(prepared)
```

Scalar Gaussian proposals retain scalar samples. Vector spherical, diagonal,
and factor Gaussian proposals learn a full covariance, including correlations.
Native `Float32` and `Float64` are supported. Loading Distributions.jl also
allows `Normal` and conventional `MvNormal` inputs through native conversion.

## One round

For a round with ``n`` samples from the current proposal ``q_t``, compute

```math
\ell_i = \log\pi(x_i)-\log q_t(x_i),\qquad
k=\lfloor\sqrt n\rfloor,\qquad
c=\text{the }k\text{-th largest }\ell_i.
```

Normalize ``\exp(\min(\ell_i,c))`` with a stable shift. Use these clipped weights
to fit the next Gaussian's mean and covariance from **this round only**.
Apply a scale-relative covariance ridge and a Cholesky factorization.
The returned `samples.logweights` contain the unchanged ``\ell_i``.
No resampling or temporal-mixture denominator is used.

With ``k=1``, clipping leaves all weights unchanged. Ties at the cap remain tied.
If fewer than ``k`` weights are finite, the cap is ``-\infty`` and adaptation
fails with [`NPMCRoundError`](@ref), since no normalized clipped weights exist.
An all-zero round also fails. The sampler does not invent replacement weights.

Clipping runs every round. The default count follows the original clipping
theory's ``k\le\sqrt n`` regime. This does not establish a universal convergence
rate for adaptive proposals or arbitrary targets.
[Mean clipping](https://victorelvira.github.io/assets/papers/martino2018comparison_pre.pdf),
tempering, and ESS gating are separate policies outside this implementation.

## Results and reuse

`round_size` accepts a positive integer or a vector with one positive entry per
round. The result contains exactly the sum of these counts.
`samples.provenance.round` identifies each sample's generating round.
`normalized_weights`, `mean`, `var`, and `lognormalizer` use raw importance weights.

For a fixed schedule and proposals with adequate support, averaging the raw
weights estimates the target normalizing constant without clipping bias.
The logarithm of that estimate and self-normalized expectations remain biased
at finite sample sizes. Concentration ESS does not guarantee estimation accuracy.

A successful call retains its final fitted proposal for the next call.
Each call returns a new result, without accumulating earlier calls.
`current_proposal` returns an independent snapshot. `retarget(rng, prepared,
new_logtarget)` reuses the learned proposal with a new target.
A failed call retains the last committed proposal and leaves the RNG advanced.

## Devices

Transfer the prepared sampler explicitly, as with the other native methods:

```julia
using CUDA, MLDataDevices
physical = CUDA.device()
device = MLDataDevices.CUDADevice{typeof(physical),Nothing}(physical)
prepared = device(prepare_sampler(Xoshiro(43), logtarget, algorithm))
samples = importance_sample!(prepared)
host_samples = cpu_device()(samples)
host_proposal = current_proposal(cpu_device(), prepared)
```

The target must compile for the selected device. Samples, weights, clipping
scratch, and Gaussian fitting buffers remain resident during each round.
Small round summaries return to the host. Clipping needs an order statistic,
which can cost more than target evaluation for cheap targets.
CPU execution uses Julia's default thread pool unless `threaded=false`.
CUDA validation lives in `validation/reproducers/cuda_npmc.jl`.
AMDGPU and Metal support are not claimed.

The runnable correlated-Gaussian example is `examples/npmc.jl`.
