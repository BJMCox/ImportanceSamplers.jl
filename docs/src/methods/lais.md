# Layered importance sampling

[`LAIS`](@ref) moves a bank of native proposal centres using independent or
interacting upper MCMC transitions. Each round then draws lower importance samples
from those proposals. The lower scale factors stay fixed. Upper states are
not extra output samples.

The fixed [`RandomWalkMetropolis`](@ref) transition follows the parallel-chain
construction in [Martino et al., *Layered Adaptive Importance Sampling*,
Section 5.1 and Table 8](https://arxiv.org/abs/1505.04732). [`RAM`](@ref) provides
an adaptive upper transition using [Vihola, *Robust adaptive Metropolis algorithm
with coerced acceptance rate*, Section 2](https://users.jyu.fi/~mvihola/vihola_-_ram.pdf).
RAM-driven LAIS is an adaptive variant, not a claim that the LAIS paper specifies RAM.
[`SampleMetropolisHastings`](@ref) supplies the interacting upper transition from
[Section 5.3.2 of the open LAIS paper](https://arxiv.org/html/1505.04732v4#S5.SS3.SSS2).
LAIS with this transition is I²-MAIS; it is not a second results or weighting API.

## Start with a proposal bank

```julia
using ImportanceSamplers, Random, Statistics

bank = ProposalBank([
    FactorGaussian([-2.0, 0.0], [1.0 0.0; 0.3 1.0]),
    FactorGaussian([ 2.0, 0.0], [1.0 0.0; 0.3 1.0]),
])
algorithm = LAIS(bank;
    transition=RAM(1.0; tuning=WarmupTuning(100)),
    rounds=20, round_size=10_000,
)
prepared = prepare_sampler(Xoshiro(42), x -> -sum(abs2, x) / 2, algorithm)
samples = importance_sample!(prepared)
mean(samples)
mean(x -> x[1]^2, samples)
learned = current_proposal(prepared)
```

For an interacting upper population on CPU, pass one fixed independent proposal
covering the target support:

```julia
using ImportanceSamplers, Random

T = Float32
factor = T[1 0; 0 1]
bank = ProposalBank([
    FactorGaussian(T[-2, 0], factor),
    FactorGaussian(T[ 2, 0], factor),
])
upper = SampleMetropolisHastings(
    DiagonalGaussian(zeros(T, 2), T[5, 2]); moves=32)
algorithm = LAIS(bank; transition=upper, rounds=40, round_size=65_536)
prepared = prepare_sampler(Xoshiro(42), x -> -sum(abs2, x) / 2, algorithm)
samples = importance_sample!(prepared)
```

For CUDA, transfer a fresh prepared sampler as a unit before its first call:

```julia
using CUDA, MLDataDevices

physical = CUDA.device()
device = MLDataDevices.CUDADevice{typeof(physical),Nothing}(physical)
gpu_prepared = device(prepare_sampler(
    Xoshiro(42), x -> -sum(abs2, x) / 2, algorithm))
gpu_samples = importance_sample!(gpu_prepared)
host_samples = cpu_device()(gpu_samples)
```

The upper proposal, centres, candidate batches, random buffers, target context,
and cached ratios remain resident. As with all documented CUDA paths, the target
must compile for the device and scalar access is unnecessary.

`round_size` counts all lower samples in one round. It accepts an integer or
one integer per round. Every count must divide equally among the proposals,
with at least one draw per proposal. All proposal masses must be equal and
positive. Target mixture masses need not be equal.

Native scalar, spherical-vector, diagonal, and factor Gaussian or Student-t
proposals support `Float32` and `Float64`. All lower proposals share a layout,
dimension, and radial family. Student-t lower proposals retain their supplied
positive degrees of freedom and scale as their centres move. Upper transition
settings do not change these lower parameters. See [Native proposals](@ref).
External MCMC packages are not supported by this method.

## Upper tuning and lower weights

`RandomWalkMetropolis(covariance)` fixes the upper covariance. `RAM(covariance;
tuning=...)` updates each chain's factor using its acceptance probability,
including rejected moves. Neither transition reads lower samples or weights.

- `WarmupTuning(steps)` makes upper-only moves before production, then freezes
  the learned factors. These moves cost target evaluations but return no samples.
- `ContinuousTuning()` updates on every production move with diminishing gain
  `k^(-decay)`. The index continues across successful calls.

Tuning is explicit. The RAM configuration above, including its variance,
warmup count, and default acceptance/decay values, illustrates the API.
It is not a benchmark-selected default or an optimal choice.
Initial upper covariance accepts an isotropic variance, SPD matrix,
`Cholesky`, or one such input per chain. It is independent of lower covariance.

`SampleMetropolisHastings(proposal; moves=1)` instead fixes one independent upper
proposal ``\varphi``. Native spherical, diagonal, and factor Gaussian and Student-t
families are supported. Its scalar/vector layout, dimension, and `Float32` or
`Float64` precision must exactly match the centre bank. The proposal does not
adapt during a run. For an irreducible independent transition, its support must
cover the target support; the native families have full real support. Initial
centres still require finite target and ``\varphi`` log densities.

For each replacement event, SMH selects an old slot with probability proportional
to ``r_i=\varphi(x_i)/\pi(x_i)`` and either replaces only that slot or leaves the entire
population unchanged. `moves=N` requests `N` ordered events; it does not update
each of the `N` slots exactly once. Later events see accepted earlier replacements.
A candidate with target log density `-Inf` is rejected without changing cached
old ratios.

For an independent candidate ``y``, let ``S=\sum_i r_i`` over the old population.
The replacement acceptance probability is

```math
\alpha=\frac{S}{S+r_y-\min(r_y,r_1,\ldots,r_N)}.
```

The minimum therefore includes the candidate and every old centre. At `N=1`
this is the usual independence-MH acceptance probability.

With the paper-default `moves=1`, SMH costs one upper target evaluation and one
upper ``\varphi`` evaluation per round, after the first call evaluates target and
``\varphi`` once at each of the `N` initial centres. Setting `moves=N` costs `N` of
each per round. These upper costs are separate from the `round_size` lower target
evaluations and the `N * round_size` lower mixture-density component evaluations.

Each production round makes one upper transition before drawing the lower samples.
RWM and RAM make one move per chain; SMH makes its configured number of ordered
replacement attempts. There is no extra upper transition after the last lower
round. For equal allocation, the retained denominator is the current spatial mixture:

```math
\phi_t(x)=\frac{1}{N}\sum_{j=1}^{N}q_{j,t}(x),\qquad
\log w_{i,t}=\log\pi(x_{i,t})-\log\phi_t(x_{i,t}).
```

All rounds retain their original raw log weights. `samples.provenance` preserves
proposal and round IDs. No history-wide reweighting occurs. Conditional on the
upper history, averaging each complete equal-allocation block estimates
`integral(pi*f)` without bias, provided support, integrability and independent
upper/lower randomness hold. Self-normalized expectations are generally biased
at finite sample size. This identity does not establish upper-chain convergence.

## Reuse, retarget, and failures

Successful calls retain final centres and transition state: RWM/RAM keep cached
target values and learned factors/tuning progress, while SMH keeps cached
``\log(\phi/\pi)`` ratios. Repeated calls return separate, noncumulative results.
`retarget(new_rng, prepared, new_logtarget)` creates independent state with the
learned centres/factors and fixed SMH proposal but fresh target caches, counters,
and tuning progress.
Pass target context in the usual positional `p` argument when needed.

Initial upper centres need finite target log density. A candidate with `-Inf`
log density is rejected. `NaN`, `+Inf`, target exceptions, or invalid RAM factors
fail the call. [`LAISRoundError`](@ref) reports the round. Failed calls preserve
the pre-call committed sampler state, but do not restore the RNG.

`samples.diagnostics.transition` reports initial, warmup and production target
evaluations, warmup and production proposals, and accepted moves for this call.
Accepted counts include warmup when present. Total target evaluations include
one evaluation per lower sample as well. Current mixture density work scales
with the number of proposals times the lower sample count.

## CPU and accelerators

CPU uses the default Julia thread pool unless `threaded=false`. Bulk RNG buffers
are filled before chain work. Serial and threaded execution share the recurrence.
For CUDA, transfer the complete prepared sampler as shown in [Accelerators](@ref).
Target context transfers with it. Samples, factors, weights and provenance remain
on device. Warmup runs sequential moves within each chain in a device kernel,
with chains in parallel. The host launches batches, not individual moves, and
checks failures before the first lower round. Failed chains latch their errors.
SMH generates each fixed-proposal candidate batch and evaluates its target and
proposal densities concurrently, then performs the population replacements in
order without copying candidates or ratios to the host. SMH and warmup batches
have at most 256 moves and use at most approximately 1 MiB of temporary random
and candidate buffers, unless one move alone requires more. The buffers are
reused between batches. Completed warmup needs no new warmup buffers on later calls.

Warmup and SMH add no per-move host transfers. Each production round still reports
bounded failure/summary scalars, not per-sample or per-chain host reads. Device
random generation uses bulk normal and uniform buffers. Batching changes exact
seeded trajectories from earlier implementations, not the transition law;
cross-version or CPU/CUDA bitwise identity is not promised.

Use `current_proposal(cpu_device(), prepared)` for an explicit host copy of the
current bank. Result transfers also remain explicit. Native transitions use
KernelAbstractions; actual accelerator validation currently covers CUDA only.

## Custom upper transitions

Subtype [`AbstractMCMCTransition`](@ref). Implement these namespaced methods;
they are not exported as a second user API:

| Method | Responsibility |
|:--|:--|
| `ImportanceSamplers.prepare_transition(config, centres, target, L)` | Build state without consuming randomness; retain the supplied centre storage. |
| `ImportanceSamplers.transition_centres(state)` | Return the centre batch without copying. |
| `ImportanceSamplers.transition!(state, target, rng, execution, transfers)` | Perform one round's upper work, updating centres and cumulative counters. |
| `Base.copyto!(destination, source)` | Copy persistent state for rollback-safe execution. |
| `ImportanceSamplers.transition_diagnostics(state)` | Return the six cumulative count fields listed above; `accepted` is the sixth field. |
| `ImportanceSamplers.retarget_transition(config, state, destination)` | Construct a fresh configuration from learned parameters. |

Preparation also needs `deepcopy` to create independent committed/run state.
Custom device support needs `Adapt.adapt_structure`, preserved centre aliases,
and `ImportanceSamplers._preflight_transition(device, state, target)` validation.
Do not silently stage chain batches through CPU. A custom transition must preserve
the upper/lower independence contract and provide valid device target evaluation.

See the [correlated-mixture example](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/examples/lais.jl),
[CPU recurrence checks](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/test/lais.jl),
and [CUDA reproducer](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/validation/reproducers/cuda_lais.jl).
