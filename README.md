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

## ESS per second

Five seeds, Float64, eight CPU threads, and one A100. Timings include preparation,
warmup/adaptation, and posterior means, but exclude compilation.

| Sampler / device | Linear (32) | Logistic (12) | Poisson (12) | Robust (13) | Eight schools (10) |
|:--|--:|--:|--:|--:|--:|
| IS / CPU | 2.7e+04 | 4.3e+04 | 9.4e+04 | 1e+05 | 1.9e+04 |
| AMIS / CPU | 2.8e+04 | 4.9e+04 | 1.2e+05 | 1.1e+05 | 5.8e+05 |
| AdvancedHMC NUTS / CPU | 2.4e+03 | 3.1e+03 | 9.2e+03 | 1.1e+04 | divergences |
| AdvancedMH RWMH / CPU | 1.8e+02 | 5.9e+02 | 1.3e+03 | 1.2e+03 | 5.8e+03 |
| SliceSampling / CPU | 2.1e+02 | 5.8e+02 | 1.6e+03 | 1.3e+03 | 6.5e+04 |
| IS / CUDA | 2.9e+05 | 1.6e+06 | 2e+06 | 2.4e+06 | 1.8e+05 |
| AMIS / CUDA | 1.6e+05 | 6.8e+05 | 8.9e+05 | 1.1e+06 | 2e+06 |

IS/AMIS use weight ESS. MCMC uses minimum bulk ESS across parameters.
**These are different diagnostics, not an equal-accuracy speedup comparison.**
MH has R-hat above 1.01 at this budget. Plain IS varies widely on eight schools.

[Reproducer and pinned versions](benchmark/comparison/README.md) ·
[Timings, ranges, and accuracy checks](benchmark/comparison/results-2026-09-16.md) ·
[Models and methodology](https://bjmcox.github.io/ImportanceSamplers.jl/guide/benchmarks/)

[Documentation](https://bjmcox.github.io/ImportanceSamplers.jl/) ·
[Logistic regression example](examples/logistic_regression.jl) ·
[Apache 2.0 license](LICENSE)
