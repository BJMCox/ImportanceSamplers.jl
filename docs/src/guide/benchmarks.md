# Sampler benchmarks

The [reproducer](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/benchmark/comparison/README.md)
compares plain IS and AMIS with AdvancedHMC NUTS, AdvancedMH random-walk MH,
SliceSampling, and EnsembleMCMC differential evolution. It also measures IS
and AMIS on CUDA. The environment is separate from the package and docs
dependencies. Its committed manifest pins package versions, including the
EnsembleMCMC source revision.

The [recorded report](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/benchmark/comparison/results-2026-09-16.md)
contains the ESS/s table, per-seed ranges, timings, R-hat, and mean errors.
MH has R-hat above 1.01 at this budget. Plain IS has unstable weights on eight
schools. Neither result is filtered out of the comparison.

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

- IS and AMIS use weight ESS, ``1/\sum_i \bar w_i^2``, from final normalized weights.
- NUTS, MH, and slice sampling use the smallest rank-normalized bulk ESS across
  parameters, computed by MCMCDiagnosticTools from independent CPU chains.

Each cell is mean ESS divided by mean elapsed time across five seeds. The full
report includes the per-seed ESS/s range and maximum R-hat. R-hat above 1.01
warns of incomplete mixing. NUTS runs with divergences are labelled in the
headline table. ESS and R-hat calculations run outside the timing interval.

These ESS definitions are **different diagnostics, not a common accuracy score**.
Weight ESS measures weight concentration. It does not detect missed modes or
estimator bias, and it ignores dependence from AMIS adaptation. Bulk ESS measures
chain mixing for each parameter. Neither guarantees accuracy for an arbitrary
functional. EnsembleMCMC's coupled walkers are not independent chains, so it
appears in the accuracy comparison rather than the chain-ESS table.

## Accuracy checks

The default run uses the analytic linear-regression posterior and independent
NUTS reference runs for the other models. Each reference has 8,192 retained draws
per CPU chain, 2,048 warmup steps, and target acceptance 0.95. References must
have no divergences and maximum rank-normalized R-hat below 1.01. The report
saves reference ESS and mean standard errors. Its detailed table gives the
largest posterior-mean error, in posterior standard deviations, across parameters.
This checks central location, not tails, modes, variances, or evidence.

An earlier 20-seed accuracy run is also retained with the reproducer. Its
references used 131,072 draws per chain. The five-seed ESS run reuses these
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
runs with divergences are labelled instead of receiving a headline rate.

## Cost and configuration

Each method runs with five independent seeds. Each timed call includes a fresh
Laplace fit, sampler preparation, warmup or adaptation, sampling, and its
posterior mean calculation. Data generation, compilation, and reference
calculations stay outside timing. BenchmarkTools measures the calls.

All methods receive the same Laplace location and full-covariance information.
IS and AMIS start from a Student-t proposal with eight degrees of freedom
and a Cholesky scale factor 1.2 times the Laplace factor. MCMC runs in the
corresponding whitened coordinates. This is a comparison with informed
initialisation, not prior proposals or default PPL initialisation.

| Method | Retained sample budget | Warmup or adaptation |
|:--|--:|:--|
| Plain IS | 65,536 | No adaptation |
| AMIS | 65,536 | Four rounds of 16,384 samples, all retained |
| NUTS | 8,192 across CPU chains | 1,024 warmup steps per chain, target acceptance 0.8 |
| Random-walk MH | 8,192 across CPU chains | 1,024 discarded steps per chain, proposal covariance ``2.38^2 I/d`` |
| Slice sampling | 8,192 across CPU chains | 1,024 discarded steps per chain, random-permutation Gibbs with stepping-out width 2 |
| Ensemble differential evolution | At least 8,192 | ``4d`` walkers, 256 warmup sweeps, whole retained sweeps |

The budgets differ deliberately. IS uses a large batch while MCMC pays for
correlated transitions and warmup. These settings are not a search for each
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
julia --threads=8 --project=benchmark/comparison benchmark/comparison/compare.jl --cuda
```

Use Julia 1.13 for this pinned benchmark environment. Omit `--cuda` for CPU
only. The script prints the tables directly and saves per-run estimates,
times, host allocations, diagnostics, versions, and hardware metadata to
`benchmark/comparison/results.toml`. It refuses to overwrite existing results.
Add `--resume` to continue a saved run with the same environment and budgets.
Do not run timing comparisons on a busy host or accelerator.

To print a saved table without rerunning the samplers:

```sh
julia --project=benchmark/comparison benchmark/comparison/compare.jl --report benchmark/comparison/results-2026-09-16.toml
```

Use `--accuracy-report benchmark/comparison/accuracy-2026-09-16.toml` for the
earlier common-accuracy comparison. That artifact includes all 800 measured
runs, including ten NUTS divergences on eight schools. The reference chains
have no divergences. Their minimum bulk ESS exceeds 677,000 and their maximum
R-hat is below 1.00003.

Measurements use an AMD EPYC 7702P with eight Julia threads and one A100-PCIE-40GB.
The recorded sampler source is `ec5aeec3c59b48a2e6237a312b5bd9c909b4c107`.
The later Julia 1.10 compatibility commit does not change these vector-model
sampling paths. Raw files retain the exact package versions and manifest hash.
