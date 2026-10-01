# [First-order GRAMIS-CAIS](@id gramis-method)

`FirstOrderGRAMIS` moves proposal means using target gradients and optional repulsion.
It fits local covariances from completed samples.
It is a first-order hybrid, not the Hessian-based GRAMIS algorithm.

## Example

```@example gramis
using ImportanceSamplers, Random, Statistics

logtarget(x) = -sum(abs2, x) / 2
gradient!(g, x) = (g .= .-x; nothing)

bank = ProposalBank([
    SphericalGaussian([-1.0, 0.0], 1.5),
    SphericalGaussian([ 1.0, 0.0], 1.5),
])
algorithm = FirstOrderGRAMIS(
    bank; rounds=4, round_size=2_000,
    repulsion_strength=0.0,
)
prepared = prepare_sampler(
    Xoshiro(42), LogTarget(logtarget; grad=gradient!), algorithm,
)
samples = importance_sample!(prepared)

(mean=mean(samples), covariance=cov(samples))
```

The example disables repulsion to isolate the gradient and covariance updates.
`repulsion_strength` is an explicit choice, not an inferred target scale.

The bank needs at least two distinct initial means, equal positive masses,
a shared dimension, and one native radial family.
Every proposal needs at least `d + 2` draws per round.

## Follow one round

1. Draw and weight samples from the current, fixed population.
2. Fit proposal-local covariances using raw or tempered fitting weights.
3. Evaluate gradients at the current proposal means.
4. Move means with covariance-preconditioned gradients and backtracking.
5. Add repulsion computed from the original population, then retain the next population.

Returned weights use the current realized-count mixture:

```math
\psi_t(x)=\sum_j\frac{n_{t,j}}{N_t}q_{t,j}(x),
\qquad
\ell_{t,i}=\log\pi(x_{t,i})-\log\psi_t(x_{t,i}).
```

Later adaptation does not change those weights.
The final fitted population remains available for the next prepared call.

## Control the mean update

Backtracking tests candidates of the form

```math
\mu'_{t,j}=\mu_{t,j}+\theta\Sigma_{t,j}\nabla\log\pi(\mu_{t,j}),
\qquad \theta=1,\tfrac12,\tfrac14,\ldots.
```

The preconditioner is the current proposal covariance, not the fit for the next round.
Backtracking accepts the first finite candidate that does not decrease the log target.
If none passes within `max_backtracking_trials`, the gradient move is zero.

Repulsion is added after that test.
It uses softened distances in pooled-whitened coordinates and a peer average.
The final mean, including repulsion, therefore need not increase the log target.

`repulsion_strength` accepts a scalar, one value per round, or a round-indexed function.
Schedules restart at round one on each prepared call.

## Control covariance fitting

Local fitting ratios use the generating proposal, not the returned mixture denominator.
Above the ESS threshold, the covariance uses raw weights centred at the old proposal mean.
Below it, tempered weights use their own weighted mean.

`covariance_rate` blends the accepted fit with the old covariance.
`covariance_regularization` adds a scale-relative ridge.
An all-zero local group or failed tempering search retains its old covariance.
An invalid fitted factor fails the call and leaves the committed population unchanged.

Student-t proposals require `nu > 2`.
Preconditioning and fitting use actual covariance before conversion to Student-t scale.

## Distinguish the published methods

| Aspect | This implementation |
|:--|:--|
| Derivatives | First-order gradients, no target Hessian |
| Round order | Sample the old population, then adapt |
| Covariance | CAIS-style local fitting, with blending and ridge |
| Repulsion | Pooled-whitened, softened distance rather than the paper's raw Euclidean expression |
| Estimation | All rounds retained under their original current-mixture weights |

Use the [gradient guide](@ref gradients-guide) for explicit derivatives and AD.
Use [Devices](@ref devices-guide) for native CUDA, Metal, and Reactant differences.
Constructor controls appear in the [sampler reference](@ref sampler-reference).

## Sources

- Elvira et al., [*Gradient-based Adaptive Importance Samplers*](https://arxiv.org/abs/2210.10785).
- El-Laham et al., [*Robust Covariance Adaptation in Adaptive Importance Sampling*](https://arxiv.org/abs/1806.00093).
- [Equation reproducer](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/validation/reproducers/first_order_gramis.jl).
- [Standalone example](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/examples/first_order_gramis.jl).
