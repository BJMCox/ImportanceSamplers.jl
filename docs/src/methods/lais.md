# [Layered importance sampling](@id lais-method)

LAIS uses an upper MCMC process to move proposal centres.
It then draws lower importance samples around those centres.
Only the lower samples enter the returned weighted result.

## Start with random-walk Metropolis

```@example lais
using ImportanceSamplers, Random, Statistics

logtarget(x) = -sum(abs2, x) / 2
bank = ProposalBank([
    SphericalGaussian([-2.0, 0.0], 1.0),
    SphericalGaussian([ 2.0, 0.0], 1.0),
])
algorithm = LAIS(
    bank; transition=RandomWalkMetropolis(0.5),
    rounds=10, round_size=1_000,
)
prepared = prepare_sampler(Xoshiro(42), logtarget, algorithm)
samples = importance_sample!(prepared)

(mean=mean(samples), draws=length(samples))
```

Each round first advances the upper centres, then draws 1,000 lower samples.
There is no extra upper move after the final lower round.

The upper covariance `0.5` is a variance.
The lower `SphericalGaussian(..., 1.0)` argument is a standard deviation.
These control different distributions.

## Tune the upper transition with RAM

```@example lais
tuned_algorithm = LAIS(
    bank; transition=RAM(0.5; tuning=WarmupTuning(100)),
    rounds=10, round_size=1_000,
)
tuned = prepare_sampler(Xoshiro(43), logtarget, tuned_algorithm)
tuned_samples = importance_sample!(tuned)

mean(tuned_samples)
```

`WarmupTuning(100)` makes 100 upper-only moves per chain, then freezes its learned factors.
These moves add cost but return no importance samples.
Later calls do not repeat completed warmup.

`ContinuousTuning()` instead updates the upper factor during production with a diminishing gain.
Its tuning index continues across successful calls.

RAM uses acceptance probabilities, including rejected moves.
It does **not** fit the lower proposal scales.
Choose those scales separately, or fit them in an independent pilot.

Upper covariance inputs may be an isotropic variance, SPD matrix, `Cholesky` factorization,
or a vector with one covariance specification per chain.

## Use an interacting upper population

```@example lais
upper = SampleMetropolisHastings(
    SphericalStudentT(5.0, zeros(2), 2.0);
    moves=4,
)
interacting_algorithm = LAIS(
    bank; transition=upper,
    rounds=10, round_size=1_000,
)
interacting = importance_sample(Xoshiro(44), logtarget, interacting_algorithm)

mean(interacting)
```

Sample Metropolis-Hastings proposes from one fixed independent upper distribution.
Each event selects a population slot for possible replacement.
`moves=4` means four ordered replacement attempts, not one move for each of four chains.

The upper proposal must match the centre dimension, scalar/vector layout, and precision.
It must cover the target support for an irreducible independence transition.

RWM/RAM give independent upper chains.
SMH gives the interacting I²-MAIS construction.
RAM-driven LAIS is a package variant, not a transition specified by the original LAIS paper.

## Lower allocation and weights

The bank needs equal positive masses.
Every `round_size` must be divisible by the proposal count and allocate at least one draw per proposal.

Lower samples use the equal current-population mixture:

```math
\psi_t(x)=\frac1M\sum_{j=1}^M q_{t,j}(x),
\qquad
\ell_{t,i}=\log\pi(x_{t,i})-\log\psi_t(x_{t,i}).
```

All rounds keep these weights. There is no all-history reweighting.
Upper states are not extra output draws.

Lower Gaussian or Student-t scales stay fixed.
Student-t degrees of freedom also stay fixed and may be any finite positive value.

## Reuse, diagnostics, and failures

Successful calls retain the upper centres, caches, and tuning state.
`retarget` retains learned centres and factors but resets target caches and tuning progress.

Initial centres need finite target values.
An upper candidate with target `-Inf` is rejected.
Invalid target values or failed factors abort the call without committing its new state.

`samples.diagnostics.transition` separates initial, warmup, and production work.
Include upper work when comparing total sampling cost.

Supported device paths keep upper and lower state resident.
Warmup and SMH batch dependent moves without a host transfer per move.
See [Devices](@ref devices-guide) and the [custom transition interface](@ref extension-guide).

## Sources

- Martino et al., [*Layered Adaptive Importance Sampling*](https://arxiv.org/abs/1505.04732).
- Vihola, [*Robust Adaptive Metropolis Algorithm with Coerced Acceptance Rate*](https://users.jyu.fi/~mvihola/vihola_-_ram.pdf).
- [Standalone example](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/examples/lais.jl).
- [CUDA reproducer](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/validation/reproducers/cuda_lais.jl).
