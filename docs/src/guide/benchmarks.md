# Sampler benchmarks

The [reproducer](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/benchmark/comparison/README.md)
compares plain IS, AMIS, DM-PMC, CAIS, and LAIS-RAM with AdvancedHMC NUTS,
AdvancedMH random-walk MH, SliceSampling, and EnsembleMCMC differential evolution.
It measures all five importance samplers on CPU and CUDA. The environment is
separate from the package and docs dependencies. Its committed manifest pins package versions, including the
EnsembleMCMC source revision.

The [recorded report](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/benchmark/comparison/results-2026-09-16-long.md)
contains the ESS/s table, per-seed ranges, timings, R-hat, and mean errors.
Symbol footnotes identify divergences, R-hat warnings, and large variation in
weight ESS. The report retains all measured configurations.

## Models

All models use `Float64`. The same scalar log target serves CPU and CUDA.
NUTS uses analytic gradients checked against ForwardDiff before measurement.
Means use the sampling coordinates, including log scales and standardised
school effects.

| Model | Parameters | Observations | Likelihood and prior |
|:--|--:|--:|:--|
| Linear regression | 32 | 1,024 | Gaussian noise with unit standard deviation |
| Logistic regression | 12 | 1,024 | Bernoulli with a logistic link |
| Poisson regression | 12 | 1,024 | Poisson with a log link |
| Robust regression | 13 | 512 | Student-t noise with four degrees of freedom and an inferred log scale |
| Eight schools | 10 | 8 | Non-centred normal hierarchy with an inferred mean and log scale |

Regression coefficients have independent `Normal(0, 2)` priors. The intercept
counts as a coefficient. Non-intercept predictors have correlation
`0.8^abs(i-j)`. Robust regression adds a `Normal(0, 1)` prior on log scale.
The fixed data seeds and generating coefficients appear in
[models.jl](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/benchmark/comparison/models.jl).

Eight schools uses the standard eight observations and standard errors.
Its latent standardised effects have `Normal(0, 1)` priors, its population mean
has a `Normal(0, 5)` prior, and its population scale has a half-normal prior
with scale 5. The log target includes the log-scale Jacobian. This is not the
half-Cauchy-prior variant of that model.

## What ESS/s means here

The README reports conventional ESS divided by elapsed time:

- Importance samplers use weight ESS, ``1/\sum_i \bar w_i^2``, from final normalized weights.
- NUTS, MH, and slice sampling use the smallest rank-normalized bulk ESS across
  parameters, computed by MCMCDiagnosticTools from independent CPU chains.

Each cell is mean ESS divided by mean elapsed time across three seeds. The full
report includes the per-seed ESS/s range and maximum R-hat. R-hat above 1.01
warns of incomplete mixing. Daggers mark NUTS divergences and double daggers mark
R-hat above 1.01. Section signs mark a weight-ESS range exceeding a factor of ten.
ESS and R-hat calculations run outside the timing interval.
Bold denotes the highest measured rate per model, selected before rounding.

These ESS definitions are **different diagnostics, not a common accuracy score**.
Weight ESS measures weight concentration. It does not detect missed modes or
estimator bias, and it ignores dependence from proposal adaptation. Bulk ESS
measures chain mixing for each parameter. Neither guarantees accuracy for an arbitrary
functional. EnsembleMCMC's coupled walkers are not independent chains, so it
appears in the accuracy comparison rather than the chain-ESS table.
Bulk ESS can exceed the retained draw count when correlations are antithetic.

## Accuracy checks

The default run uses the analytic linear-regression posterior and independent
NUTS reference runs for the other models. Each reference has 8,192 retained draws
in each of eight CPU chains, 2,048 warmup steps, and target acceptance 0.95.
References must have no divergences and maximum rank-normalized R-hat below 1.01. The report
saves reference ESS and mean standard errors. Its detailed table averages
posterior-mean estimates across timed seeds, then reports the largest absolute
error across parameters, in posterior standard deviations.
This checks central location, not tails, modes, variances, or evidence.

An earlier 20-seed accuracy run is also retained with the reproducer. Its
references used 131,072 draws per chain. The published ESS runs reuse these
already-computed moments. New runs use the smaller reference budget above.
The earlier run reports a separate, common posterior-mean accuracy score:

For independent runs ``r=1,\ldots,R``, parameter means ``\hat\mu_{rj}``, reference
means ``\mu_j``, posterior variances ``v_j``, and elapsed times ``t_r``, define

```math
E = \frac{1}{Rd}\sum_{r=1}^R\sum_{j=1}^d
    \frac{(\hat\mu_{rj}-\mu_j)^2}{v_j},
\qquad
\text{accuracy-based ESS/s} = \frac{1}{E\,\overline{t}}.
```

An average of ``n`` independent posterior draws has expected scaled squared
error ``1/n``. The reported rate uses this reference scale for every sampler.
It includes estimator bias and permits weighted and unweighted results in
one table. It does not establish tail, mode, variance, or evidence accuracy.
The score averages across variance-scaled coordinates, not the worst coordinate.
The accuracy-based ESS can exceed the draw count, for example with antithetic
MCMC correlations.

The accuracy report gives 95% bootstrap intervals from 1,000 resamples of whole
benchmark runs. It also resamples reference-chain means to account for
reference uncertainty. The estimates remain noisy, especially with only
20 runs. A close ranking is not evidence of a speed difference. Timed NUTS
runs with divergences retain their measured rates and receive a dagger.

## Cost and configuration

Each method runs with three independent seeds. Each timed call includes a fresh
Laplace fit, sampler preparation, warmup or adaptation, sampling, and its
posterior mean calculation. Data generation, compilation, and reference
calculations stay outside timing. BenchmarkTools measures the calls.

All methods receive the same Laplace location and full-covariance information.
IS and AMIS start from a Student-t proposal with eight degrees of freedom
and a Cholesky scale factor 1.2 times the Laplace factor. MCMC runs in the
corresponding whitened coordinates. This is a comparison with informed
initialisation, not prior proposals or default PPL initialisation.

DM-PMC, CAIS, and LAIS-RAM use 16 equal-mass Student-t proposals with the same
degrees of freedom and scale factors. Their centres have independent Gaussian
offsets with scale 0.25 in Laplace-whitened coordinates. DM-PMC uses global
resampling. CAIS uses its default covariance-ESS threshold. LAIS uses RAM upper
transitions, initialized with covariance ``2.38^2\Sigma/d`` and 1,024 upper-only
warmup moves. Its lower proposal scales remain fixed.

This fixed-scale configuration gives low ESS for DM-PMC and LAIS-RAM on the
32-parameter linear model. Their CPU runs average approximately 27 and 45
effective samples from 262,144 weighted draws. Their pooled-mean errors reach
0.19 and 0.18 posterior standard deviations. The report retains these results.

| Method | Retained sample budget | Warmup or adaptation |
|:--|--:|:--|
| Plain IS | 262,144 | No adaptation |
| AMIS, DM-PMC, CAIS, LAIS-RAM | 262,144 | Four rounds of 65,536 samples, all retained |
| NUTS | 16,384 per chain, 16 chains | 1,024 warmup steps per chain, target acceptance 0.8 |
| Random-walk MH | 16,384 per chain, 16 chains | 1,024 discarded steps per chain, proposal covariance ``2.38^2 I/d`` |
| Slice sampling | 16,384 per chain, 16 chains | 1,024 discarded steps per chain, random-permutation Gibbs with stepping-out width 2 |
| Ensemble differential evolution | At least 262,144 | ``4d`` walkers, 1,024 warmup sweeps, whole retained sweeps |

The retained sample budgets match. IS uses batches while MCMC uses correlated
transitions and warmup. These settings are not a search for each
method's best possible configuration. Do not interpret a large ESS/s ratio
between these different diagnostics as an equal-accuracy speedup.

CPU chains use Julia's default thread pool. BLAS and FFTW each use one thread.
EnsembleMCMC uses its threaded executor. Its dependent walkers are never
treated as independent chains for diagnostics or uncertainty estimates.

CUDA timing includes target-data and sampler transfers, device reductions,
and the final posterior-mean transfer to the CPU. That final transfer also
synchronises the measured work. GPU compilation is excluded. These are
end-to-end timings, not isolated kernel throughput.

## Reproduce

```sh
julia --project=benchmark/comparison -e 'using Pkg; Pkg.instantiate()'
julia --threads=16 --project=benchmark/comparison benchmark/comparison/compare.jl --cuda
```

Use Julia 1.13 for this pinned benchmark environment. Omit `--cuda` for CPU
only. The script prints the tables directly and saves per-run estimates,
times, host allocations, diagnostics, versions, and hardware metadata to
`benchmark/comparison/results.toml`. It refuses to overwrite existing results.
Add `--resume` to continue a saved run with the same environment and budgets.
Do not run timing comparisons on a busy host or accelerator.

To print a saved table without rerunning the samplers:

```sh
julia --project=benchmark/comparison benchmark/comparison/compare.jl --report benchmark/comparison/results-2026-09-16-long.toml
```

Use `--accuracy-report benchmark/comparison/accuracy-2026-09-16.toml` for the
earlier common-accuracy comparison. That artifact includes all 800 measured
runs, including ten NUTS divergences on eight schools. The reference chains
have no divergences. Their minimum bulk ESS exceeds 677,000 and their maximum
R-hat is below 1.00003.

Measurements use an AMD EPYC 7702P with 16 Julia threads and one A100-PCIE-40GB.
The recorded sampler source is `ec5aeec3c59b48a2e6237a312b5bd9c909b4c107`.
The later Julia 1.10 compatibility commit does not change these vector-model
sampling paths. Raw files retain the exact package versions and manifest hash.
The measured benchmark source is retained at commit `11dbb99`. Later changes
to table formatting do not alter the sampling code.
