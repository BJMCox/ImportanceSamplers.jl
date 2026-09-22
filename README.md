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

Sample a three-dimensional standard Gaussian using a broader Gaussian proposal:

```julia
using ImportanceSamplers, Random, Statistics

proposal = SphericalGaussian(zeros(3), 1.5)
logtarget(x) = -sum(abs2, x) / 2  # unnormalized log density, x is a 3-vector

samples = importance_sample(
    Xoshiro(42), logtarget, ImportanceSampling(proposal; nsamples=10_000),
)

mean(samples)           # coordinate means, approximately [0, 0, 0]
var(samples)            # coordinate variances, approximately [1, 1, 1]
samples.logweights      # raw log target/proposal ratios
lognormalizer(samples)  # estimate of log integral(exp(logtarget(x)))
```

The proposal must cover the target's support. Results retain samples and log
weights. Transfer a prepared sampler explicitly with `device(sampler)` for
accelerator execution.

## Sampling efficiency

Effective sample size per second (ESS/s)\* for six posterior models. Parentheses
give the number of parameters. Measurements use Float64 and include initialization,
warmup or adaptation, sampling, and posterior-mean estimation.
Each table lists importance samplers first, followed by other measured methods.

### CPU

| Sampler | Linear (32) | Logistic (12) | Poisson (12) | Robust (13) | Eight schools (10) | Signal-background (9) |
|:--|--:|--:|--:|--:|--:|--:|
| IS | **5.1e+04** | 7.4e+04 | 1.8e+05 | 1.6e+05 | 2.9e+04§ | 7.3e+04 |
| AMIS | 4.9e+04 | **8.6e+04** | 1.8e+05 | **1.9e+05** | **7.6e+05** | **2.3e+05** |
| DM-PMC | 49¶ | 7.7e+03 | 1.6e+04 | 1.4e+04 | 4.8e+04 | 6e+04 |
| CAIS | 2e+04 | 6.7e+04 | 1.5e+05 | 1.4e+05 | 1.8e+05§ | 1.8e+04§¶ |
| LAIS-RAM | 12¶ | 4.9e+03 | 1.2e+04 | 8.3e+03 | 7e+04 | 6.8e+04 |
| First-order GRAMIS-CAIS | 4.4e+04 | 8.3e+04 | **1.9e+05** | 1.6e+05 | 1.9e+04 | 9.7e+04 |
| AdvancedHMC NUTS | 1.5e+04 | 9e+03 | 2.1e+04 | 5e+04 | 1.3e+05† | 5.2e+04 |
| AdvancedMH RWMH | 8.5e+02‡ | 2.6e+03 | 5.2e+03 | 4.5e+03 | 1.8e+04 | 1.4e+04 |
| SliceSampling | 7.7e+02 | 1.9e+03 | 5.6e+03 | 4.4e+03 | 1.6e+05 | 1.5e+04 |
| EnsembleMCMC (selected) | 3e+04 | 2.4e+04 | 4.9e+04 | 5e+04 | 1e+04 | 3.5e+03‡ |

### GPU (CUDA)

Only importance samplers have GPU measurements in this comparison.

| Sampler | Linear (32) | Logistic (12) | Poisson (12) | Robust (13) | Eight schools (10) | Signal-background (9) |
|:--|--:|--:|--:|--:|--:|--:|
| IS | 5.4e+05 | 2.6e+06 | 3.3e+06 | 3.7e+06 | 4.8e+04 | 7.3e+05 |
| AMIS | 3.5e+05 | 9.4e+05 | 1.4e+06 | 1.3e+06 | 1.6e+06 | 2.1e+06 |
| DM-PMC | 1.8e+02¶ | 9.2e+04 | 2.2e+05 | 1.8e+05 | 3.3e+05 | 3.4e+05 |
| CAIS | 1.9e+05 | 2.1e+06 | 2.4e+06 | 2.9e+06 | 3.5e+05 | 1.7e+06 |
| LAIS-RAM | 4.8¶ | 7.5e+03 | 1.2e+04 | 1.4e+04 | 2.3e+05 | 6.2e+04¶ |
| First-order GRAMIS-CAIS | 3.6e+05 | 1.4e+06 | 2e+06 | 2.1e+06 | 1.5e+05 | 4.1e+05 |
| IS, AMIS-fitted‖ | 3.5e+05 | — | — | 1.6e+06 | — | — |
| LAIS-RWM, AMIS-fitted‖ | 2.7e+05 | — | — | 1.1e+06 | — | — |

Bold denotes the highest measured CPU ESS/s per model. GPU results appear separately.

EnsembleMCMC 0.0.2 uses the highest screening ESS/s among accuracy-eligible
candidates: DE for eight schools and Gaussian replacement with shrinkage 1 for
the other models. The table reports held-out results. All four moves remain
in the detailed report.

\* Importance sampling uses weight ESS. Independent-chain MCMC uses minimum
bulk ESS. EnsembleMCMC uses minimum mean ESS from the sweep-average process.
These diagnostics do not define an equal-accuracy comparison.

† At least one retained NUTS transition diverged.

‡ At least one run had maximum R-hat above 1.01. Ensemble R-hat splits the
sweep-average time series, not the walkers.

§ Weight ESS varied by more than a factor of ten across seeds.

¶ At least one run exceeded a mean error of 0.2 posterior standard deviations
or a marginal variance error of 30%. Archived rows lack variance checks.

‖ Independent 262,144-draw AMIS pilot, then 262,144 production draws, with both
costs included. Gaussian LAIS-RWM uses 16 proposals, four rounds, and no extra
MCMC warmup. Two seeds per model, 128 host threads, and the shared A100.
Dashes denote unmeasured cases. [Protocol, results, and reproducer](benchmark/comparison/lais.md).

The main comparison uses three seeds, 16 threads on a 128-thread EPYC CPU, and one A100.
Independent-chain MCMC retains 16,384 draws per chain in 16 chains. Importance
samplers retain 262,144 draws. Ensemble methods use 4d walkers and retain
16,384 sweeps per walker. Model-specific move widths come from a separate
three-seed screen, then remain fixed for three held-out seeds. Linear starts
at stationarity without warmup; other models use 1,024 warmup sweeps.
DM-PMC and LAIS include an independent 4,096-draw width pilot.
Compilation and post-run diagnostics are excluded. Unchanged rows reuse archived
measurements. The report records each row's source and timing protocol.
The September 22 EnsembleMCMC refresh used a shared host. Other rows retain
their original measurement windows; these are not matched cross-version timings.

Signal-background uses the nine-parameter BAT paper model and its original
37-event dataset. Its CAIS CPU and LAIS-RAM CUDA runs include failed moment
checks, marked above. The detailed report also retains ensemble snooker's
and DE's failed checks. Offline settings search is not timed.

The linear posterior is Gaussian, so a fitted Gaussian proposal leaves little
room for adaptation to improve weight ESS. Follow-up CUDA runs used an independent
AMIS fit and short Gaussian LAIS-RWM runs without extra MCMC warmup. Throughput
improved, but matched static IS remained faster on linear and robust regression.
Longer LAIS runs degraded the fitted proposal. These findings concern a different
configuration from the LAIS-RAM rows above, which remain for comparison.

[Reproducer and pinned versions](benchmark/comparison/README.md) ·
[Timings, ranges, and accuracy checks](benchmark/comparison/results-2026-09-22-refresh-comparison.md) ·
[Scalar/batch measurements](benchmark/comparison/README.md#scalar-and-batch-regression-targets) ·
[Models and methodology](https://bjmcox.github.io/ImportanceSamplers.jl/guide/benchmarks/)

[Documentation](https://bjmcox.github.io/ImportanceSamplers.jl/) ·
[Logistic regression example](examples/logistic_regression.jl) ·
[Apache 2.0 license](LICENSE)
