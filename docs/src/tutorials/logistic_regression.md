# [Logistic regression](@id logistic-tutorial)

This example infers an intercept and two slopes from binary observations.
It uses a short adaptive run to fit a proposal, then draws from that fixed proposal.

## 1. Create a small dataset

```@example logistic
using ImportanceSamplers, Random, Statistics, LinearAlgebra

rng = Xoshiro(12)
truth = [-0.4, 1.1, -0.8]
X = hcat(ones(80), randn(rng, 80, 2))
probability = 1 ./ (1 .+ exp.(-(X * truth)))
y = rand(rng, 80) .< probability
p = (; X, y, prior_scale=2.0)

(size(X), count(y))
```

Each row of `X` describes one observation. The first column supplies the intercept.
The context `p` holds data and constants. Only the coefficient vector `beta` is sampled.

## 2. Define the posterior log density

Assume independent normal priors with standard deviation two for the coefficients.

```@example logistic
function logposterior(beta, p)
    value = -sum(abs2, beta) / (2p.prior_scale^2)
    for row in axes(p.X, 1)
        eta = zero(eltype(beta))
        for column in eachindex(beta)
            eta += p.X[row, column] * beta[column]
        end
        # Stable log(1 + exp(eta)).
        logpartition = max(eta, zero(eta)) + log1p(exp(-abs(eta)))
        value += p.y[row] * eta - logpartition
    end
    return value
end
nothing # hide
```

The log target adds the log prior and log likelihood.
It omits parameter-independent constants, so its normalizer is not the model evidence.

The two-argument signature separates the unknown `beta` from the fixed context `p`.
No wrapper is required.

## 3. Fit a proposal

```@example logistic
initial = SphericalStudentT(5.0, zeros(3), 2.0)
pilot = prepare_sampler(
    Xoshiro(31), logposterior, p,
    AMIS(initial; rounds=4, round_size=2_000),
)
pilot_samples = importance_sample!(pilot)
learned = current_proposal(pilot)

round.(mean(pilot_samples); digits=3)
```

AMIS adapts a location and full covariance. The Student-t degrees of freedom stay fixed.
`current_proposal` returns an independent snapshot in the original sampling coordinates.

The pilot starts broadly. It may still miss a remote mode.
A fitted proposal does not remove the need to inspect the final weights.

## 4. Draw the final sample

```@example logistic
samples = importance_sample(
    Xoshiro(32), logposterior, p,
    ImportanceSampling(learned; nsamples=20_000),
)

(
    mean=round.(mean(samples); digits=3),
    sd=round.(std(samples); digits=3),
    weight_ess=inv(sum(abs2, normalized_weights(samples))),
)
```

The final run uses a fresh random stream and a fixed proposal.
Its budget is independent of the pilot budget.
Only `samples` supplies the following estimates.

## 5. Summarize uncertainty and predict

```@example logistic
intervals = quantile(samples, [0.025, 0.975])
new_predictors = [1.0, 0.5, -0.2]
prediction = mean(
    beta -> inv(1 + exp(-dot(new_predictors, beta))),
    samples,
)

(intervals=intervals, predicted_probability=prediction)
```

The intervals are marginal posterior intervals for each coefficient.
The prediction averages the success probability over posterior uncertainty.

More observations can make prior importance sampling inefficient.
Adapting the proposal helps only when the pilot reaches the posterior mass.

## Extend the example

- Use [batch targets](@ref batch-guide) to evaluate a design matrix against many coefficient vectors.
- Transfer the complete prepared sampler for [GPU execution](@ref devices-guide).
- See the standalone [logistic regression script](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/examples/logistic_regression.jl) for a prior-based pilot.
