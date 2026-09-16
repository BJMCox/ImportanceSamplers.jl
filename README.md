# ImportanceSamplers.jl

[![CI](https://github.com/BJMCox/ImportanceSamplers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/BJMCox/ImportanceSamplers.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/BJMCox/ImportanceSamplers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/BJMCox/ImportanceSamplers.jl)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://bjmcox.github.io/ImportanceSamplers.jl/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Plain, multiple, and adaptive importance sampling in Julia: AMIS, APIS, LAIS,
CAIS, N-PMC, DM-PMC, and first-order GRAMIS-CAIS. Supports threaded CPU execution,
native Gaussian and Student-t proposals, constrained parameters, and optional
CUDA, Metal, and Reactant paths with [documented support limits](https://bjmcox.github.io/ImportanceSamplers.jl/guide/accelerators/).

Requires Julia 1.10 or later. Install from GitHub until the package is registered:

```julia
using Pkg
Pkg.add(url="https://github.com/BJMCox/ImportanceSamplers.jl.git")
```

```julia
using ImportanceSamplers, Random, Statistics

proposal = SphericalGaussian(0.0, 1.0)
logtarget(x) = -abs2(x) / 2  # unnormalized log density

samples = importance_sample(
    Xoshiro(42), logtarget, ImportanceSampling(proposal; nsamples=10_000),
)

mean(samples)                # weighted estimate of E[x]
var(samples)                 # weighted estimate of Var[x]
samples.logweights           # raw log target/proposal ratios
lognormalizer(samples)       # estimate of log integral(exp(logtarget(x)))
```

The proposal must cover the target's support. Results retain samples and log
weights. Transfer a prepared sampler explicitly with `device(sampler)` for
accelerator execution.

[Documentation](https://bjmcox.github.io/ImportanceSamplers.jl/) ·
[Logistic regression example](examples/logistic_regression.jl) ·
[Apache 2.0 license](LICENSE)
