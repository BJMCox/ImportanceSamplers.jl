# Adaptive multiple importance sampling

[`AMIS`](@ref) implements the all-history Gaussian method of Cornuet et al.,
[“Adaptive Multiple Importance Sampling”](https://arxiv.org/abs/0907.1254).
Every round retains its samples and generating proposal, and every new proposal
retrospectively changes the weights of all earlier samples.

## Minimal prepared execution

```julia
using ImportanceSamplers
using Random

proposal = SphericalGaussian([-3.0, 3.0], 3.5)
logtarget(x)::Float64 = -0.5 * sum(abs2, x)
algorithm = AMIS(proposal; rounds=4, round_size=1_000)
sampler = prepare_sampler(Xoshiro(42), logtarget, algorithm; threaded=false)
result = importance_sample!(sampler)
learned = current_proposal(sampler)
```

The proposal must use native `Float32` or `Float64` storage. A scalar spherical
Gaussian keeps its scalar layout. Vector spherical, diagonal, and factor
Gaussians use the vector factor path and learn a full covariance.

`round_size` is either one positive `Int`, repeated for all `rounds`, or a
positive `Vector{Int}` with one entry per round. Preparation copies and resolves
the schedule. If its entries are ``N_1,\ldots,N_T``, the result contains exactly
``N=\sum_t N_t`` samples; `result.provenance.round` records their one-based
generating rounds.

## Retrospective weights and round order

After round ``t``, every retained sample uses the realized temporal mixture

```math
\psi_t(x) = \frac{\sum_{\ell=1}^t N_\ell q_\ell(x)}
                  {\sum_{\ell=1}^t N_\ell}, \qquad
\log w_i^{(t)} = \log \pi(x_i) - \log \psi_t(x_i).
```

The schedule counts, not equal-per-round coefficients, therefore set the
mixture masses. The implementation stores one online log numerator per sample,
updates it with stable log-mixture arithmetic, and never constructs a dense
sample-by-proposal density matrix.

Round ``t`` has this fixed order:

1. freeze the staged proposal as generating proposal ``q_t``;
2. copy ``q_t`` into proposal-history slot ``t``;
3. fill the round's random buffer;
4. draw exactly ``N_t`` new samples from ``q_t``;
5. evaluate the target once for each new sample;
6. append ``\log N_t + \log q_t(x)`` to every old log numerator;
7. initialize every new log numerator from ``q_1,\ldots,q_t``;
8. form all current retrospective log weights;
9. validate the complete current weight state;
10. record the all-sample concentration ESS and numerical log normalizer;
11. normalize all current weights on the execution device;
12. fit their weighted mean and covariance;
13. add the scale-dependent ridge and factor the covariance once on-device; and
14. stage the fitted proposal as ``q_{t+1}``.

The final fitted ``q_{T+1}`` generated no sample, so it never enters the returned
denominator. It is promoted only after result construction and backend
synchronization succeed.

## Learned state, results, and failures

Prepared execution owns and advances its RNG. A successful call retains its
final fitted proposal, and the next call begins a new, noncumulative estimator
run from that proposal. [`current_proposal`](@ref) returns an independent
snapshot on CPU. For an accelerator-prepared sampler, request the transfer
explicitly:

```julia
host_proposal = current_proposal(MLDataDevices.cpu_device(), sampler)
```

Returned results own their samples, weights, and provenance; later calls do not
change them.

A call is one proposal transaction. If any round, fit, or result-construction
step fails, no partial result or staged proposal is committed. The RNG remains
advanced, while the proposal committed before the call remains authoritative.
[`AMISRoundError`](@ref) records the failing round and phase, its cause, completed
round count, cumulative sample count, and any failure-only diagnostics and
reported transfers.

## Cost and storage

For ``T`` rounds, total count ``N``, and vector dimension ``d``, diagnostics
report exactly ``N`` target evaluations and ``TN`` proposal-density evaluations.
The retained factor history has ``T(d^2+d+1)`` scalar elements. The main
workspaces hold samples and centered adaptation scratch of ``dN`` elements each,
four length-``N`` scalar accumulators, plus ``O(d^2)`` covariance and candidate
storage. The repository benchmark measures end-to-end execution under this
storage contract.

## CPU, CUDA, and recorded transfers

The table is generated from `validation/amis_capabilities.jl`. Every CPU cell
runs both equal and unequal public schedules during the strict documentation
build. Each CUDA claim is the matching row exercised by the A100 reproducer.

```@eval
Main.AMIS_CAPABILITY_TABLE
```

Only native `Float32` and `Float64` CPU and CUDA execution are claimed. AMDGPU
and Metal remain unclaimed.

CUDA keeps samples, proposal history, retrospective denominators, weights,
adaptation state, and workspaces resident. A successful round reports six small
host transfers: one three-`UInt64` failure snapshot, two summary maxima, two
scaled sums, and one scaled-square sum. This is ``24 + 5\,\mathrm{sizeof}(T)``
bytes per round. The accounting is source-level and cannot observe transfers
inside CUDA or vendor libraries; CUDA's solver-status scalar is outside it.
`current_proposal(cpu_device(), sampler)` is a separate explicit ``O(d^2)``
snapshot, not part of execution.

## Normalizer claim limits

`result.diagnostics.round_ess[t]` and
`result.diagnostics.round_lognormalizers[t]` summarize every sample retained
through round ``t`` under its current retrospective weights. Concentration ESS
is not a variance-equivalent sample count or a stopping rule. Each normalizer is
a numerical estimate: this implementation claims neither finite-sample
unbiasedness nor generic consistency for adaptive AMIS, and applying `log` adds
the usual nonlinear bias. The consistency paper for the modified learning
scheme, [MAMIS](https://arxiv.org/abs/1211.2548), describes a different method;
this implementation is not MAMIS.

## Reproducers, benchmark, and example

- [Independent CPU equation reproducer](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/validation/reproducers/amis.jl)
- [A100 CUDA reproducer](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/validation/reproducers/cuda_amis.jl)
- [CPU/CUDA benchmark harness](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/benchmark/amis.jl)
- [Correlated Gaussian example](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/examples/amis.jl)
