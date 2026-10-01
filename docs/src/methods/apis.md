# [Adaptive population importance sampling](@id apis-method)

APIS fits each proposal's mean from its own samples.
Proposal scales stay fixed. The output weights use the full current mixture.

## Example

```@example apis
using ImportanceSamplers, Random, Statistics

logtarget(x) = -sum(abs2, x) / 2
bank = ProposalBank([
    SphericalStudentT(5.0, [-2.0, 0.0], 1.0),
    SphericalStudentT(5.0, [ 2.0, 0.0], 1.0),
])
prepared = prepare_sampler(
    Xoshiro(42), logtarget,
    APIS(bank; rounds=4, round_size=2_000),
)
samples = importance_sample!(prepared)

(mean=mean(samples), count=length(samples))
```

One package round is one APIS epoch.
Each epoch keeps its proposals fixed while drawing the assigned samples.

The bank needs equal positive masses.
Each round size must be divisible by the proposal count and give at least two draws per proposal.

## Distinguish the two weight roles

For a sample from proposal ``j``, the retained estimator weight uses

```math
\psi_t(x)=\frac1M\sum_{k=1}^M q_{t,k}(x),
\qquad
w(x)=\frac{\pi(x)}{\psi_t(x)}.
```

The next mean instead uses proposal-local ratios
``\rho_i=\pi(x_i)/q_{t,j}(x_i)``:

```math
\mu_{t+1,j}=\frac{\sum_{i\in j}\rho_i x_i}{\sum_{i\in j}\rho_i}.
```

Only proposal `j`'s draws enter its mean fit.
The fit reuses target and generating-density values already computed for weighting.

Output weights never change under a later population.
All epochs remain in the result.

## Reuse and limitations

Means adapt after the final epoch too.
A successful call retains them for later use.

Covariances, masses, and Student-t degrees of freedom remain fixed.
Broader or narrower proposals therefore require a different initial bank or a separate pilot.

A proposal group with no positive local weight has an undefined mean and fails the call.
The previous committed population remains available.
See [Adaptation and reuse](@ref reuse-guide).

## Sources

- Martino et al., [*An Adaptive Population Importance Sampler: Learning From Uncertainty*](https://vixra.org/pdf/1405.0280v4.pdf).
- [Standalone example](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/examples/apis.jl).
- [CUDA reproducer](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/validation/reproducers/cuda_apis.jl).

[Devices](@ref devices-guide) lists current backend limits.
