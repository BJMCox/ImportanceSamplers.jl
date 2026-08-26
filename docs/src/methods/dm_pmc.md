# Deterministic-mixture population Monte Carlo

[`DeterministicMixturePMC`](@ref) implements fixed-population DM-PMC with global
multinomial resampling. The first-release method follows Elvira et al.,
[“Improving Population Monte Carlo”](https://victorelvira.github.io/assets/papers/elvira2017improving_pre.pdf),
and does not implement the paper's local or glocal variants.

## Minimal prepared execution

```julia
using ImportanceSamplers
using Random

bank = ProposalBank([
    FactorGaussian([-2.0, 0.0], [1.0 0.0; 0.4 0.8]),
    FactorGaussian([ 2.0, 0.0], [1.0 0.0; -0.4 0.8]),
])
algorithm = DeterministicMixturePMC(bank; rounds=6, round_size=10_000)
sampler = prepare_sampler(Xoshiro(42), logtarget, algorithm)
samples = importance_sample!(sampler)
```

`round_size` is the number of samples per round. A positive integer repeats
that size for all `rounds`; a positive `Vector{Int}` supplies one size per
round. Preparation copies and resolves the complete schedule. The returned
count is exactly its sum.

Prepared execution owns and advances its RNG and adaptive population. Repeated
calls start new estimator runs from the last learned population and return
independent, noncumulative results. [`current_proposal`](@ref) returns an
independent snapshot from a CPU-prepared sampler. The one-argument form rejects
accelerator-prepared samplers before reading device storage, so there is no
hidden transfer. Request the host transfer explicitly with
`current_proposal(MLDataDevices.cpu_device(), sampler)`. This copies only the
current packed locations and stable proposal IDs under the sampler's selected
physical-device scope, then reconstructs an independent CPU `ProposalBank` with
the configured fixed masses, scales or factors, and inert zero-mass proposals.
It does not migrate the prepared sampler, RNG, target, workspaces, or results.
Scalar-converting CPU destinations and non-CPU destinations are rejected.

## Round allocation and weighting

Only positive-mass proposals are active. Each fixed round count is allocated
among them by largest-remainder rounding, with tied remainders rotated across
rounds. Every active proposal must receive at least one draw. Zero-mass
proposals remain in configuration and retain their stable IDs, but contribute
neither samples nor denominator terms.

If round ``t`` realizes counts ``n_{t,j}`` and total ``N_t``, its spatial
mixture is fixed before any current sample is drawn:

```math
\psi_t(x) = \sum_j \frac{n_{t,j}}{N_t}q_{t,j}(x), \qquad
\log w_{t,i} = \log \pi(x_{t,i}) - \log \psi_t(x_{t,i}).
```

The denominator coefficients are therefore the realized count fractions, not
the nominal masses directly. Equal masses and divisible round sizes recover
the paper's equal spatial mixture; unequal masses give the documented
realized-count extension.

After a complete round has valid weights, global multinomial resampling draws
one ancestor for each proposal from the whole round. Duplicate ancestors are
allowed and can reduce population diversity. The selected values become the
next locations in slot order. Spherical scales, diagonal scales, and lower
triangular factors remain fixed. Resampling also occurs after the final round,
and that final population is retained for the next prepared call.

## Complete results, provenance, and failure

The result contains every sample from every round. Each canonical raw log
weight keeps its own round's current-mixture denominator. The complete linear
normalizer assigns every flattened sample the same ``1/N`` estimator
coefficient:

```math
\log \widehat Z = \operatorname{logsumexp}(\log w) - \log N,
\qquad N=\sum_t N_t.
```

`samples.provenance.round` and `samples.provenance.proposal_id` identify the
generating round and stable configured proposal ID. Diagnostics provide the
resolved round sizes, per-round log normalizers, and per-round normalized-weight
concentration ESS. This ESS describes weight concentration; it is not a
variance-equivalent sample count or an automatic stopping rule.

A failed round is not partially committed or resampled. [`DMPMCRoundError`](@ref)
records its round, phase, cause, and committed-round count. The sampler retains
the last committed population and its RNG remains advanced. Earlier returned
results remain unchanged after both later successes and failures.

## Linear-normalizer guarantee

The linear estimator is conditionally unbiased when every proposal is
normalized, every conditional spatial mixture covers the target's integrable
mass, the finite round schedule and allocation are fixed before current draws,
and each adapted population depends only on completed earlier rounds. Under
those conditions each round estimates the same integral conditional on its
past, so the sample-count-weighted flattened average does too.

This statement does not make `lognormalizer` itself unbiased after applying
`log`, does not imply finite variance without square-integrability, and does
not give generic finite-sample unbiasedness for nonlinear summaries or for
adaptive populations outside these conditions.

## CPU, CUDA, and transfer boundary

CPU execution supports serial evaluation and `threaded=true` preparation;
threaded workers consume prefilled random buffers. Accelerator use is explicit:
prepare on CPU, then apply a concrete MLDataDevices device to the complete
prepared sampler before its first execution. Samples, weights, proposal state,
resampling state, and workspaces remain on that device.

Inspecting the retained accelerator population is a separate, explicit
operation:

```julia
host_bank = current_proposal(MLDataDevices.cpu_device(), sampler)
```

The accessor restores the caller's physical-device selection after copying the
two packed arrays. The returned bank owns its locations, masses, scales, and
factors; mutating it cannot change the prepared sampler.

The table below is generated during every strict documentation build from the
executable rows in `validation/dm_pmc_capabilities.jl`. Each CPU cell performs
a public run while building these docs; each CUDA claim corresponds to the
same row exercised by the A100 reproducer.

```@eval
Main.DM_PMC_CAPABILITY_TABLE
```

AMDGPU and Metal are unclaimed. CUDA reports six explicit small scalar transfer
reasons per round: failure snapshot, CDF maximum and sum, and summary maximum,
scaled sum, and scaled-square sum. This is ``O(\text{rounds})`` source-level
accounting; it does not instrument hidden runtime or library transfers.

## Reproducers, benchmark, and examples

- [Independent CPU equation reproducer](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/validation/reproducers/dm_pmc_global.jl)
- [A100 CUDA reproducer](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/validation/reproducers/cuda_dm_pmc.jl)
- [CPU/CUDA benchmark harness](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/benchmark/dm_pmc.jl)
- [Public bimodal DM-PMC example](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/examples/dm_pmc.jl)

The benchmark reports throughput, allocations, device bytes, transfer
accounting, flattened concentration ESS, and separate per-round diagnostic ESS.
