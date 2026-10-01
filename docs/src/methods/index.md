# [Choosing a method](@id choosing-method)

Begin with a proposal that covers the target.
Adaptation can improve a useful initial proposal, but it cannot recover regions that the run never visits.

## Match the method to the task

| Method | What changes | Weight denominator |
|:--|:--|:--|
| [Plain IS](@ref plain-is) | Nothing | Single proposal |
| [Static MIS](@ref static-mis) | Nothing | Chosen fixed-bank scheme |
| [AMIS](@ref amis-method) | One mean and covariance | Mixture of all generating proposals |
| [N-PMC](@ref npmc-method) | One mean and covariance, using clipped fit weights | Generating proposal |
| [DM-PMC](@ref dmpmc-method) | Population centres, by resampling | Current realized population mixture |
| [GR-PMC](@ref dmpmc-method) | Centres, by global resampling with equal allocation | Current equal population mixture |
| [LR-PMC](@ref dmpmc-method) | Centres, by local resampling with equal allocation | Current equal population mixture |
| [APIS](@ref apis-method) | Population means | Current equal population mixture |
| [CAIS](@ref cais-method) | Population means and covariances | Generating proposal |
| [LAIS](@ref lais-method) | Population centres, using upper MCMC | Current equal population mixture |
| [First-order GRAMIS-CAIS](@ref gramis-method) | Gradient-driven means and local covariances | Current realized population mixture |

GR-PMC and LR-PMC use `DeterministicMixturePMC` with different resampling policies.
FirstOrderGRAMIS is a first-order hybrid, not the Hessian-based GRAMIS paper algorithm.

Plain IS suits a fixed proposal, including a prior or an independently fitted proposal.
Static MIS retains several fixed proposals.
AMIS and N-PMC fit one elliptical proposal.
Population methods retain several proposals, with different rules for moving or fitting them.

No method dominates for every target, dimension, budget, and estimand.
Compare accuracy as well as weight ESS and elapsed time.

## Understand the common workflow

```julia
prepared = prepare_sampler(rng, logtarget, p, algorithm)
samples = importance_sample!(prepared)
```

For a single run, `importance_sample(rng, logtarget, p, algorithm)` combines both steps.
Omit `p` for a context-free target.

Adaptive algorithms require `rounds` and `round_size`.
An integer repeats the total round size.
A vector gives one total count per round.
All rounds contribute to the returned result.

Only AMIS retrospectively changes earlier weights.
The other adaptive methods retain each sample's original round weight.

## Allocate enough samples

| Method | Per-round allocation requirement |
|:--|:--|
| AMIS, N-PMC | One proposal with a positive round size and a viable moment fit |
| DM-PMC | At least one draw for each active proposal |
| APIS | Equal positive masses, divisible round size, at least two draws per proposal |
| LAIS | Equal positive masses, divisible round size, at least one draw per proposal |
| CAIS | Equal positive masses, divisible round size, at least `d + 2` draws per proposal |
| FirstOrderGRAMIS | At least two distinct initial means, equal positive masses, at least `d + 2` draws per proposal |

These are admissibility conditions, not recommended statistical budgets.
Covariance fitting and tail coverage often need substantially more samples.

Adaptive methods use native Gaussian or Student-t proposals.
[Proposals](@ref proposals-guide) explains the supported layouts and Student-t covariance convention.
[Devices](@ref devices-guide) is the single reference for backend support.
