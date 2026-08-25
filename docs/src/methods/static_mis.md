# Static multiple importance sampling

Static multiple importance sampling (MIS) draws from a fixed population of
normalized proposals and chooses a complete weighting scheme before execution.
The terminology and denominator families follow Elvira et al.,
[“Generalized Multiple Importance Sampling”](https://arxiv.org/abs/1511.03095).

## Bank and result contract

For proposals ``q_1,\ldots,q_J`` and normalized nominal masses
``\alpha_1,\ldots,\alpha_J``, construct an explicit bank:

```julia
bank = ProposalBank(proposals, masses)
algorithm = ImportanceSampling(
    bank;
    nsamples=10_003,
    mis_scheme=StratifiedMixture(),
)
```

The bank copies both vectors and normalizes finite nonnegative masses. A
zero-mass proposal remains visible at its original one-based index, but is never
assigned and never enters a denominator. Every run returns exactly `nsamples`
samples. The aligned generating IDs are in
`result.provenance.proposal_id`.

`ProposalBank` is deliberately not a mixture distribution: it defines neither
`rand` nor `DensityInterface.logdensityof`. Assignment and denominator choice
are separate parts of an MIS scheme. A distribution package's mixture object
therefore remains one atomic proposal unless its components are explicitly
expanded into a bank.

## Assignment and denominators

Let ``A_i`` be the generating proposal ID and ``d_i`` the scheme denominator.
The stored canonical weight is always raw and unnormalized:

```math
\log w_i = \log \pi(x_i) - \log d_i(x_i).
```

The complete assignment vector is fixed before any proposal draw. Stratified
assignment maps one uniform from each of ``N`` equal strata through the nominal
mass CDF. Random assignment instead draws ``A_i`` independently from that CDF.

| Scheme | Assignment | Denominator ``d_i(x)`` | Proposal-density cost |
|:--|:--|:--|:--|
| [`StratifiedMixture`](@ref) | stratified nominal masses | ``\sum_j \alpha_j q_j(x)`` | ``J`` per sample |
| [`RandomMixture`](@ref) | iid nominal masses | ``\sum_j \alpha_j q_j(x)`` | ``J`` per sample |
| [`StandardMIS`](@ref) | stratified nominal masses | ``q_{A_i}(x)`` | one per sample |
| [`PartialDeterministicMixture`](@ref) | stratified nominal masses | nominal mixture within the group containing ``A_i`` | group size per sample |

For a partial group ``G``, the denominator coefficients are normalized within
the group:

```math
d_G(x) = \sum_{j\in G}
\frac{\alpha_j}{\sum_{k\in G}\alpha_k}q_j(x).
```

`groups` must partition every original proposal ID exactly once. A group that
contains only zero-mass proposals is valid but inert.

## Support and unbiasedness

The linear estimate

```math
\widehat Z = \frac{1}{N}\sum_{i=1}^N w_i
```

is unbiased when the target integral exists and the chosen complete scheme has
the required support:

- full-mixture schemes require aggregate support from
  ``\sum_j\alpha_jq_j``;
- `StandardMIS` requires every positive-mass generating proposal to cover the
  target support;
- partial deterministic mixtures require each active group mixture to cover
  the target support.

Finite variance additionally requires the corresponding squared density ratios
to be integrable. `lognormalizer(result)` returns ``\log\widehat Z``; the
logarithm itself is generally biased even when ``\widehat Z`` is unbiased.

Full-mixture denominators use more density evaluations but typically reduce
weight variance by sharing information across proposals. Singleton denominators
are cheapest. Partial groups trade between those endpoints. Stratification
removes most proposal-count variation relative to iid mixture assignment, but
no strict variance ordering holds for every target and proposal bank.

## Preparation and execution capabilities

Positive-mass proposals must have one logical sample dimension. Packed native
Gaussian banks additionally require one scalar/vector layout and one floating
type. Mixed dimensions, mixed `Float32`/`Float64`, and scalar mixed with a
length-one vector are rejected during preparation, before RNG use.

The following matrix is generated during every strict documentation build from
the metadata also consumed by the CUDA reproducer. Its CPU examples execute all
four schemes, and its accelerator rejection rows check the typed reason.

```@eval
Main.STATIC_MIS_CAPABILITY_TABLE
```

CUDA execution keeps packed bank state, assignments, samples, raw log weights,
proposal IDs, target context, random buffers, and partial-group state on the
device. Transfer to CPU is explicit. AMDGPU and Metal remain unclaimed.

## Reproducers, benchmark, and example

The repository provides a deterministic analytic
[CPU reproducer](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/validation/reproducers/static_mis.jl)
and the real-hardware
[CUDA reproducer](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/validation/reproducers/cuda_static_mis.jl).
Run them from the package root with their isolated validation environment:

```text
julia --project=validation validation/reproducers/static_mis.jl
julia --project=validation validation/reproducers/cuda_static_mis.jl --correctness-only
```

`benchmark/static_mis.jl` measures preparation, warmed execution, allocations,
throughput, proposal-density evaluation counts, and CUDA result transfer. The
short runnable workflow is in `examples/static_mis.jl`.
