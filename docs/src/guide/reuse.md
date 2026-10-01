# [Adaptation and reuse](@id reuse-guide)

An algorithm describes a sampling method.
A prepared sampler owns the target, random stream, workspaces, and learned proposal.

## Prepare once, run again

```@example reuse
using ImportanceSamplers, Random, Statistics

logtarget(x, p) = -sum(abs2, x .- p.centre) / 2
p = (; centre=[1.0, -1.0])
algorithm = AMIS(
    SphericalGaussian(zeros(2), 2.0);
    rounds=3, round_size=1_000,
)
prepared = prepare_sampler(Xoshiro(42), logtarget, p, algorithm)

first_run = importance_sample!(prepared)
second_run = importance_sample!(prepared)
(mean(first_run), mean(second_run))
```

The second call starts from the proposal learned by the first call.
Its result contains only its own samples, not samples from earlier calls.
Earlier result arrays remain unchanged.

A successful adaptive call retains its final fitted proposal even when that proposal
generated no sample in the completed call.
AMIS also starts a new retrospective history on the next call.

## Choose the count

Plain and static IS take `nsamples`.
Adaptive methods take `rounds` and `round_size`:

```julia
AMIS(proposal; rounds=3, round_size=1_000)          # 3,000 returned draws
AMIS(proposal; rounds=3, round_size=[500, 1_000, 2_000]) # 3,500 draws
```

`round_size` counts all draws in a round, not draws per proposal.
Every adaptive result retains all rounds.

Schedules are fixed during preparation. Round-size callbacks and changing the budget
of an existing prepared sampler are not supported.
Construct another algorithm to change its budget.

Population methods have additional [allocation requirements](@ref choosing-method).
A prepared call restarts round-indexed schedules at round one.
LAIS transition state follows its separate tuning rules.

## Reuse only the learned proposal

```@example reuse
proposal = current_proposal(prepared)
final_samples = importance_sample(
    Xoshiro(43), logtarget, p,
    ImportanceSampling(proposal; nsamples=5_000),
)

mean(final_samples)
```

This separates a pilot from production.
The new algorithm controls its own count and method.
`current_proposal` returns an independent snapshot, not a mutable view into the sampler.

For a named transform layout, the snapshot uses numerical sampling coordinates.
Pass the same layout to the new prepared sampler.

For an accelerator sampler, request an explicit CPU snapshot with
`current_proposal(MLDataDevices.CPUDevice(), prepared)`.
This transfers the proposal, not the complete result or sampler.

## Change the target

```@example reuse
new_p = (; centre=[1.5, -0.5])
new_sampler = retarget(Xoshiro(44), prepared, logtarget, new_p)
new_samples = importance_sample!(new_sampler)

mean(new_samples)
```

`retarget` keeps the learned proposal, algorithm controls, count schedule, transform,
execution policy, and device. It creates fresh target-dependent state and a new random stream.
The original sampler remains usable.

This operation supports every implemented adaptive method, including LAIS.
For plain or static IS, call `prepare_sampler` again with the existing algorithm.

The reused proposal must cover the new target.
Retargeting does not prove that condition.

On an accelerator, retargeting stages the learned proposal through CPU memory before
constructing fresh state on the original device. This is a setup cost, not a per-sample transfer.

## Respect ownership and failures

The sampler owns the RNG supplied to preparation.
Do not also consume it elsewhere when reproducibility matters.
Use separate samplers and RNGs for concurrent runs.

A prepared sampler is mutable and non-reentrant.
An adaptive failure preserves the proposal committed before that call.
It returns no partial result and does not restore consumed random numbers.

Device placement must occur before the first execution.
See [Devices](@ref devices-guide) for transfer and migration rules.
