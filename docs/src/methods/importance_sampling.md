# [Plain importance sampling](@id plain-is)

`ImportanceSampling` draws from one fixed normalized proposal.
It corrects the proposal draws with target-to-proposal density ratios.

## Example

```@example plain
using ImportanceSamplers, Random, Statistics

logtarget(x) = -sum(abs2, x) / 2
proposal = SphericalStudentT(5.0, zeros(3), 1.0)
samples = importance_sample(
    Xoshiro(42), logtarget,
    ImportanceSampling(proposal; nsamples=10_000),
)

(mean=mean(samples), lognormalizer=lognormalizer(samples))
```

The proposal controls the sample dimension.
`nsamples` is the exact number of returned draws.
The [first tutorial](@ref first-estimate) explains each argument and result.

## Weights and estimates

For ``x_i\sim q``, the raw log weight is

```math
\ell_i=\log\pi(x_i)-\log q(x_i).
```

The normalized proposal must cover every region contributing to the integral.
The target and proposal must use the same reference measure.

For integrable ``\pi f``, the linear estimate
``n^{-1}\sum_i e^{\ell_i}f(x_i)`` is unbiased under this support condition.
Finite variance needs the corresponding squared ratio to be integrable.
Self-normalized expectations and logarithms generally have finite-sample bias.

## Use the prior as proposal

For a Bayesian target
``\log\pi(\theta)=\log p(y\mid\theta)+\log p(\theta)``,
a normalized prior proposal gives raw log weights equal to the log likelihood.

```julia
logposterior(theta, p) = loglikelihood(theta, p.data) + logprior(theta)
samples = importance_sample(
    rng, logposterior, p,
    ImportanceSampling(prior; nsamples=10_000),
)
```

Here `prior` must implement both drawing and normalized density evaluation.
This is ordinary IS, not a separate algorithm or special weight path.

A concentrated posterior may leave most prior draws with negligible weight.
The [logistic tutorial](@ref logistic-tutorial) separates proposal fitting from final sampling.

## Reuse and failures

A prepared plain-IS sampler advances its RNG without changing the proposal.
Repeated calls return separate complete results.

If all target values are `-Inf`, the result retains zero weights.
Normalized summaries then fail, while `lognormalizer` returns `-Inf`.
Invalid target values or inconsistent proposal densities fail the run.

See [Results](@ref results-guide), [Targets](@ref targets-guide), and
[custom CPU proposals](@ref extension-guide) for the full input and output contracts.

## Sources

- Elvira and Martino, [*Advances in Importance Sampling*](https://arxiv.org/abs/2102.05407).
- [Analytic reproducer](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/validation/reproducers/plain_is.jl).
