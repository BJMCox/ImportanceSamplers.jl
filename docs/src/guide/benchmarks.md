# [Benchmarks](@id benchmarks-guide)

The repository contains model comparisons, analytic accuracy checks, and per-method performance scripts.
Benchmark dependencies remain separate from package and documentation dependencies.

## Reproduce a model comparison

From the repository root, with Julia 1.13 for the pinned comparison environment:

```sh
julia --project=benchmark/comparison -e 'using Pkg; Pkg.instantiate()'
julia --threads=16 --project=benchmark/comparison benchmark/comparison/compare.jl --cuda
```

The runner prints tables and records versions, source hashes, hardware, timings, and diagnostics.
It refuses to overwrite an existing result.
See the [comparison guide](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/benchmark/comparison/README.md)
for CPU-only runs, selected methods, pilot settings, and resumption.

To print an existing table without sampling:

```sh
julia --project=benchmark/comparison benchmark/comparison/compare.jl --report benchmark/comparison/results-2026-09-16-comparison.toml
```

## Models and competing samplers

| Model | Parameters | Data |
|:--|--:|:--|
| Linear regression | 32 | 1,024 observations |
| Logistic regression | 12 | 1,024 observations |
| Poisson regression | 12 | 1,024 observations |
| Robust regression | 13 | 512 observations |
| Eight schools | 10 | Eight groups |
| Signal-background | 9 | 37 events across five detectors |

The comparison includes fixed and adaptive IS, AdvancedHMC NUTS, AdvancedMH,
SliceSampling, and EnsembleMCMC.
The [model source](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/benchmark/comparison/models.jl)
defines data, coordinates, priors, and target functions.

The [combined report](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/benchmark/comparison/results-2026-09-22-refresh-comparison.md)
retains timing ranges and accuracy warnings.
Some rows reuse archived measurements. They are not simultaneous measurements or a cross-version speedup test.

## Interpret ESS/s carefully

| Sampler class | Reported ESS |
|:--|:--|
| Importance sampling | Weight concentration, ``1/\sum_i\bar w_i^2`` |
| Independent MCMC chains | Minimum rank-normalized bulk ESS across coordinates |
| EnsembleMCMC | Minimum coordinate-mean ESS from within-sweep walker averages |

These are different diagnostics, not a common accuracy score.
Weight ESS does not detect missed modes or tails.
Ensemble walkers are not treated as independent chains.

CPU and GPU tables remain separate.
Pilot and adaptation costs belong in end-to-end timing.
Accuracy checks and ESS calculations occur outside the timed interval.

Use the full reports to inspect moment errors, divergences, R-hat, and per-seed variation.
A high ESS/s cell does not override a failed accuracy check.

## Compare scalar and batch targets

The paired [CPU report](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/benchmark/comparison/batch-cpu-2026-09-28-paired.md)
and [CUDA report](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/benchmark/comparison/batch-cuda-2026-09-28-paired.md)
compare scalar and batch evaluation.

Shared machines require interleaved comparisons.
Separate collection windows cannot establish a regression or speedup by themselves.

The [batch API guide](@ref batch-guide) explains the callback contract.
The comparison README gives the paired-run commands.

## Examine signal-background coverage

The model-specific conditional proposal has a separate runner:

```sh
julia --threads=16 --project=benchmark/comparison benchmark/comparison/signal_background_conditional.jl --cuda conditional-results.toml
julia --project=benchmark/comparison benchmark/comparison/signal_background_conditional.jl --report conditional-results.toml
```

It fits a conditional proposal from a CPU pilot, then compares independent production IS with LAIS.
It is benchmark-local code, not a new public sampler.

Its weight ESS is not tail-functional ESS.
The archived moment reference and the independent coordinate-specific tail checks answer different questions.
Passing those moment checks does not establish coverage of every tail functional.

## Measure other costs

- [`benchmark/time_to_accuracy.jl`](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/benchmark/time_to_accuracy.jl)
  compares normalizer, mean, and second-moment errors against independent references.
- [`benchmark/plain_is.jl`](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/benchmark/plain_is.jl)
  measures preparation, warmed execution, and host allocations.
- [`benchmark/logistic_gramis.jl`](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/benchmark/logistic_gramis.jl)
  separates preparation, first-call cost, warmed timing, and profiles.

Record the exact source and environment rather than copying a published timing to another machine.
