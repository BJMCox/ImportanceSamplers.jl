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

## Sampling efficiency

Effective sample size per second (ESS/s)\* for five posterior models. Parentheses
give the number of parameters. Measurements use Float64 and include initialization,
warmup or adaptation, sampling, and posterior-mean estimation.

| Sampler / device | Linear (32) | Logistic (12) | Poisson (12) | Robust (13) | Eight schools (10) |
|:--|--:|--:|--:|--:|--:|
| IS / CPU | 5.1e+04 | 7.4e+04 | 1.8e+05 | 1.6e+05 | 2.9e+04§ |
| AMIS / CPU | 4.9e+04 | 8.6e+04 | 1.8e+05 | 1.9e+05 | 7.6e+05 |
| DM-PMC / CPU | 9.7 | 5.9e+03 | 1.4e+04 | 1.1e+04 | 4.8e+04§ |
| CAIS / CPU | 2e+04 | 6.7e+04 | 1.5e+05 | 1.4e+05 | 1.8e+05§ |
| LAIS-RAM / CPU | 15 | 4e+03 | 1.1e+04 | 7.3e+03 | 6.2e+04 |
| AdvancedHMC NUTS / CPU | 1.5e+04 | 9e+03 | 2.1e+04 | 5e+04 | 1.3e+05† |
| AdvancedMH RWMH / CPU | 8.5e+02‡ | 2.6e+03 | 5.2e+03 | 4.5e+03 | 1.8e+04 |
| SliceSampling / CPU | 7.7e+02 | 1.9e+03 | 5.6e+03 | 4.4e+03 | 1.6e+05 |
| IS / CUDA | **5.4e+05** | **2.6e+06** | **3.3e+06** | **3.7e+06** | 4.8e+04 |
| AMIS / CUDA | 3.5e+05 | 9.4e+05 | 1.4e+06 | 1.3e+06 | **1.6e+06** |
| DM-PMC / CUDA | 33 | 1.1e+05 | 1.3e+05 | 1.1e+05 | 2.2e+04 |
| CAIS / CUDA | 1.9e+05 | 2.1e+06 | 2.4e+06 | 2.9e+06 | 3.5e+05 |
| LAIS-RAM / CUDA | 28 | 5.4e+03 | 1e+04 | 1.1e+04 | 1.8e+05 |

Bold denotes the highest measured ESS/s per model.

\* Importance sampling uses weight ESS. MCMC uses minimum bulk ESS across
parameters. These diagnostics do not define an equal-accuracy comparison.

† At least one retained NUTS transition diverged.

‡ At least one run had maximum R-hat above 1.01.

§ Weight ESS varied by more than a factor of ten across seeds.

These measurements use three seeds, 16 CPU threads, and one A100. MCMC retains
16,384 draws in each of 16 chains. Importance samplers retain 262,144 draws.
Compilation and post-run diagnostics are excluded.

[Reproducer and pinned versions](benchmark/comparison/README.md) ·
[Timings, ranges, and accuracy checks](benchmark/comparison/results-2026-09-16-long.md) ·
[Models and methodology](https://bjmcox.github.io/ImportanceSamplers.jl/guide/benchmarks/)

[Documentation](https://bjmcox.github.io/ImportanceSamplers.jl/) ·
[Logistic regression example](examples/logistic_regression.jl) ·
[Apache 2.0 license](LICENSE)
