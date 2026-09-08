# Adaptive population importance sampling

[`APIS`](@ref) implements the epoch-based adaptive population importance
sampler of Martino, Elvira, Luengo, and Corander,
[“An Adaptive Population Importance Sampler: Learning From Uncertainty”](https://doi.org/10.1109/TSP.2015.2440215).
It adapts only proposal means. Covariances, scale factors, proposal masses, and
stable proposal IDs remain fixed.

## Epochs and allocation

```julia
using ImportanceSamplers, Random

bank = ProposalBank([
    SphericalGaussian(-2.0, 2.0),
    SphericalGaussian( 2.0, 3.0),
])
algorithm = APIS(bank; rounds=3, round_size=2048)
prepared = prepare_sampler(Xoshiro(42), x -> -abs2(x) / 2, algorithm)
samples = importance_sample!(prepared)
learned = current_proposal(prepared)
```

One library `round` is one paper epoch. With ``N`` proposals and total epoch
size `round_size`, each proposal contributes
``T_a = \mathtt{round\_size}/N`` draws. APIS requires equal positive bank
masses, exact divisibility, and ``T_a \ge 2``. A vector `round_size` supplies
one total count per epoch; this is the paper's discussed variable-epoch-length
extension. The returned sample count is the sum of those epoch sizes.

## Two weights with different jobs

Within epoch ``m``, all proposal parameters are fixed and the equal spatial
mixture is

```math
\psi^{(m)}(x) = \frac{1}{N}\sum_{j=1}^N q_j^{(m)}(x).
```

A sample ``x_{i,t}`` drawn by proposal ``i`` receives the retained estimator
weight

```math
\log w_{i,t} = \log\pi(x_{i,t}) - \log\psi^{(m)}(x_{i,t}).
```

These raw log weights are never replaced or recomputed under later proposal
populations. All epochs therefore contribute to `normalized_weights`, weighted
statistics, and `lognormalizer` using their generating epoch's mixture.

Mean adaptation deliberately uses a different, proposal-local weight:

```math
\log\rho_{i,t} = \log\pi(x_{i,t}) - \log q_i^{(m)}(x_{i,t}),
\qquad
\mu_i^{(m+1)} =
\frac{\sum_t \rho_{i,t}x_{i,t}}{\sum_t \rho_{i,t}}.
```

The implementation normalizes these local weights with a stable maximum shift.
It reuses the target and generating-density values computed while weighting;
adaptation performs no additional density evaluations. Samples from another
proposal never enter proposal ``i``'s mean fit.

Adaptation also runs after the final epoch, matching Table I and the prepared
sampler reuse contract. For `rounds=1`, the returned sample is the same static
deterministic-mixture law; the final fit affects only `current_proposal` and a
later prepared call, not samples already returned.

## Assumptions, failure, and reuse

Every proposal must be a normalized native `Float32` or `Float64` spherical,
diagonal, or lower-triangular-factor Gaussian with the same scalar/vector
layout and dimension. The target is an unnormalized **log density** with the
same reference measure. As in the paper, proposals must cover the target mass;
tail adequacy cannot be proved at construction.

If one proposal has no finite positive local adaptation mass, its mean is
undefined. [`APISRoundError`](@ref) reports the epoch and `:adaptation` phase
instead of silently freezing it. A failed call leaves the pre-call committed
bank unchanged while its owned RNG remains advanced. A successful call commits
the final learned bank. Repeated calls start from that bank and return separate,
noncumulative results. `retarget(rng, prepared, new_logtarget)` preserves the
learned means and fixed covariance factors in a fresh sampler.

## CPU, CUDA, and transfers

CPU supports serial and threaded execution. The same deterministic normal
buffer gives the same recurrence in either mode. Transfer the complete
prepared sampler explicitly before its first execution for CUDA:

```julia
using CUDA, MLDataDevices
physical = CUDA.device()
device = MLDataDevices.CUDADevice{typeof(physical),Nothing}(physical)
prepared = device(prepare_sampler(Xoshiro(43), x -> -abs2(x) / 2, algorithm))
samples = importance_sample!(prepared)
host_learned = current_proposal(cpu_device(), prepared)
```

Samples, raw weights, target/generating log densities, local moment scratch,
and proposal means remain on the selected accelerator. Each epoch reports only
small failure, local-fit validity, and log-weight summary scalars to the host.
There are no per-sample host reads. `current_proposal(prepared)` rejects an
accelerator sampler; the explicit `cpu_device()` form copies the packed current
locations and stable proposal IDs, then reconstructs the independent bank with
its fixed masses, scales, or factors. Moving a result to CPU is also explicit.

See the [teaching example](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/examples/apis.jl),
[real-CUDA reproducer](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/validation/reproducers/cuda_apis.jl),
and [benchmark harness](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/benchmark/apis.jl).
