# [Static multiple importance sampling](@id static-mis)

Static MIS samples a fixed bank of proposals.
A complete scheme selects both the generating proposal and the weight denominator.

## Cover two modes

```@example static
using ImportanceSamplers, Random, Statistics

function logtarget(x)
    a = -abs2(x + 2) / 2 + log(0.3)
    b = -abs2(x - 2) / 2 + log(0.7)
    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m)) - log(2pi) / 2
end

bank = ProposalBank([
    SphericalGaussian(-2.0, 1.2),
    SphericalGaussian( 2.0, 1.2),
], [0.3, 0.7])
algorithm = ImportanceSampling(
    bank; nsamples=10_000, mis_scheme=StratifiedMixture(),
)
samples = importance_sample(Xoshiro(42), logtarget, algorithm)

(mean=mean(samples), reference_mean=0.8, lognormalizer=lognormalizer(samples))
```

The target is a normalized two-mode Gaussian mixture.
The population's masses match its two mixture masses, while the proposals are wider.

`samples.provenance.proposal_id` identifies the generating bank entry.
The bank keeps its input order and stable one-based identifiers.

## Choose the scheme

For masses ``\alpha_j``, define
``\psi(x)=\sum_j\alpha_jq_j(x)``.
Every raw weight has the form ``\log\pi(x)-\log d_i(x)``.

| Scheme | Proposal assignment | Denominator | Density cost per draw |
|:--|:--|:--|:--|
| `StratifiedMixture()` | Stratified masses | Full mixture ``\psi`` | All active proposals |
| `RandomMixture()` | Independent categorical draws | Full mixture ``\psi`` | All active proposals |
| `StandardMIS()` | Stratified masses | Generating proposal | One proposal |
| `PartialDeterministicMixture(groups)` | Stratified masses | Mixture within the generating proposal's group | Group size |

Stratification spreads assignment uniforms across equal strata before applying the mass CDF.
It does not promise identical component counts in every run.

Partial groups must partition every bank identifier exactly once:

```julia
scheme = PartialDeterministicMixture([[1, 2], [3, 4]])
algorithm = ImportanceSampling(four_proposal_bank; nsamples=10_000, mis_scheme=scheme)
```

Within a group, nominal masses are renormalized to sum to one.
Zero-mass proposals keep their identifiers but generate no samples.

## Weigh cost against coverage

A full-mixture denominator shares density information across proposals.
Its cost grows with the number of active proposals.
A generating-proposal denominator is cheaper but has a stricter support condition.

For an unbiased linear normalizer estimate:

- Full-mixture schemes need the aggregate mixture to cover the target.
- `StandardMIS` needs each active proposal to cover the target.
- Partial mixtures need each active group mixture to cover the target.

These statements also require integrability.
They do not imply finite variance or unbiased logarithms.

## Sources

- Elvira et al., [*Generalized Multiple Importance Sampling*](https://arxiv.org/abs/1511.03095).
- [CPU reproducer](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/validation/reproducers/static_mis.jl).
- [Standalone example](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/examples/static_mis.jl).

Use [Devices](@ref devices-guide) for backend support and [Proposals](@ref proposals-guide) for bank construction.
