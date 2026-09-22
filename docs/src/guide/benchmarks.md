# Sampler benchmarks

The [reproducer](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/benchmark/comparison/README.md)
compares plain IS, AMIS, DM-PMC, CAIS, LAIS-RAM, and first-order GRAMIS-CAIS with AdvancedHMC NUTS,
AdvancedMH random-walk MH, SliceSampling, and EnsembleMCMC. The current harness
includes DE, Stretch, snooker and Gaussian replacement candidates. The archived
measurements contain only the first three. The headline shows one selected move
per model while the detailed report retains all held-out move rows. Width-screen
trials remain in the raw screening file.
It measures all six importance samplers on CPU and CUDA. The environment is
separate from the package and docs dependencies. Its committed manifest pins package versions, including the
EnsembleMCMC source revision.

The [recorded report](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/benchmark/comparison/results-2026-09-22-comparison.md)
contains the ESS/s table, raw timing ranges, R-hat, and moment errors.
Symbol footnotes identify divergences, R-hat warnings, large weight-ESS variation,
and failed moment checks. The report retains every measured run.
It combines archived unchanged rows with new measurements. Each row links to
its raw source, including the original seeds, source hashes, and timing protocol.
First-order GRAMIS-CAIS uses repulsion strength 0.1, the same Student-t bank and
round budget as CAIS, and the models' analytic gradients. This is not the
original second-order GRAMIS algorithm.

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
| Signal-background | 9 | 37 | BAT paper example: Poisson event counts, exponential background and a fixed Gaussian signal across five detectors |

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

Signal-background copies the data and model from
[BAT's paper example](https://github.com/bat/BAT.jl/blob/3ab3baf1d1666fcd22c08e6f4db3089feec57c39/examples/paper-example/paper_example.jl).
The signal rate has a uniform prior on `[0,10]`. The hierarchical background
parameters have uniform priors on `sigma_B ∈ [0.1,1]`, `m_B ∈ [1e-10,20]`, and
the exponential scale `lambda ∈ [1e-10,100]`. Detector rates satisfy
`log(B_j) = log(m_B) - sigma_B^2/2 + sigma_B*z_j`, with independent standard
normal `z_j`. The signal mean and standard deviation stay fixed at 100 and 2.
The bounded parent parameters use logits. All samplers use these same
unconstrained coordinates and full Jacobians. The copied data include 37 events
with detector counts `(13,10,10,2,2)`.

The target has checked analytic gradients and uses the same scalar code on CPU
and CUDA. Parameter-independent constants are omitted, so the benchmark does not
compare absolute evidence with BAT. Source-data hashes appear in new reports.
The reference has maximum R-hat 1.0004 and minimum bulk ESS 23,801. CAIS CPU,
LAIS-RAM CUDA, and ensemble snooker failed moment checks in at least one run;
the detailed report retains those measurements with warning symbols.

## What ESS/s means here

The README reports conventional ESS divided by elapsed time:

- Importance samplers use weight ESS, ``1/\sum_i \bar w_i^2``, from final normalized weights.
- NUTS, MH, and slice sampling use the smallest rank-normalized bulk ESS across
  parameters, computed by MCMCDiagnosticTools from independent CPU chains.
- EnsembleMCMC uses the smallest coordinate mean ESS. Its MCSE comes from the
  time series of within-sweep walker averages, not independent walker chains.

Each cell is mean ESS divided by mean elapsed time across three seeds. The full
report includes the per-seed ESS/s range and maximum R-hat. R-hat above 1.01
warns of incomplete mixing. Daggers mark NUTS divergences and double daggers mark
R-hat above 1.01. Section signs mark a weight-ESS range exceeding a factor of ten.
ESS and R-hat calculations run outside the timing interval.
CPU and GPU results appear in separate tables. Bold denotes the highest CPU
rate per model, selected before rounding. GPU cells are not bolded.

The EnsembleMCMC headline selects the highest screening mean ESS / mean seconds
among accuracy-eligible candidates at their independently selected settings.
Reporting seeds never choose or replace the selected move. Current archived
data select DE on five models and Stretch on signal-background. If no candidate
passes the screening checks, its headline cell shows a dash. The report names each
selected move and retains all held-out diagnostics. Gaussian replacement
requires a new screen and held-out measurements before it can appear as selected.

These ESS definitions are **different diagnostics, not a common accuracy score**.
Weight ESS measures weight concentration. It does not detect missed modes or
estimator bias, and it ignores dependence from proposal adaptation. Bulk ESS
measures chain mixing for each parameter. Neither guarantees accuracy for an arbitrary
functional. Ensemble mean ESS measures the precision of coordinate means, not
rank-normalized bulk mixing.
Bulk ESS can exceed the retained draw count when correlations are antithetic.

For ensemble coordinate ``j``, let ``F_{tj}=L^{-1}\sum_{k=1}^{L}x_{tkj}`` be
the mean across ``L`` walkers at sweep ``t``. The report computes

```math
\widehat{\mathrm{ESS}}_j =
\frac{\widehat{\operatorname{Var}}_\pi(x_j)}
     {\widehat{\operatorname{MCSE}}(\overline{F}_j)^2}.
```

The construction follows [Goodman and Weare, section 3](https://cims.nyu.edu/~weare/papers/d13.pdf).
MCMCDiagnosticTools estimates the MCSE from the autocovariance of ``F_{tj}``.
Cross-walker lag covariance therefore enters the estimate. R-hat splits this
sweep-mean series, not the walkers. Short histories can give noisy estimates.

## Accuracy checks

The default run uses the analytic linear-regression posterior and independent
NUTS reference runs for the other models. Each reference has 8,192 retained draws
in each of eight CPU chains, 2,048 warmup steps, and target acceptance 0.95.
References must have no divergences and maximum rank-normalized R-hat below 1.01. The report
saves reference ESS and mean standard errors. Its detailed table averages
posterior-mean estimates across timed seeds, then reports the largest absolute
error across parameters, in posterior standard deviations.
Pilcrows mark any run with a coordinate mean error above 0.2 posterior standard
deviations or a marginal variance error above 30%. The report retains those runs.
These checks cover central moments, not tails, modes, or evidence.
Archived rows retain their mean checks but lack per-run variance estimates.
A dash in the detailed variance column denotes an unavailable check, not a pass.

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
Refreshed rows use the median of up to three timed executions within a five-second
BenchmarkTools budget. A full execution may exceed that budget. All raw elapsed
and GC times remain in the saved report. CPU and CUDA calls for a given IS
method run consecutively. Archived rows retain one timed execution per seed.
The combined report does not imply simultaneous or exclusive-host measurements.

All methods receive the same Laplace location and full-covariance information.
IS and AMIS start from a Student-t proposal with eight degrees of freedom
and a Cholesky scale factor 1.2 times the Laplace factor. MCMC runs in the
corresponding whitened coordinates. This is a comparison with informed
initialisation, not prior proposals or default PPL initialisation.

DM-PMC, CAIS, LAIS-RAM, and first-order GRAMIS-CAIS use 16 equal-mass Student-t proposals with the same
degrees of freedom and scale factors. Their centres have independent Gaussian
offsets with scale 0.25 in Laplace-whitened coordinates. DM-PMC uses global
resampling. CAIS uses its default covariance-ESS threshold. LAIS uses RAM upper
transitions, initialized with covariance ``2.38^2\Sigma/d`` and 1,024 upper-only
warmup moves. Its lower proposal scales remain fixed.

The refresh adds an independent 4,096-draw width pilot to DM-PMC and LAIS-RAM.
It fits one common scale-factor multiplier between 0.25 and 2.0, without changing
centres, proposal count, masses, degrees of freedom, or main round sizes. Its
draws are discarded and its entire cost is timed. Later centre adaptation can
make the fitted initial width unsuitable. This pilot does not guarantee good
linear-model ESS or accuracy. Failed checks remain visible in the report.

| Method | Retained sample budget | Warmup or adaptation |
|:--|--:|:--|
| Plain IS | 262,144 | No adaptation |
| AMIS, DM-PMC, CAIS, LAIS-RAM, first-order GRAMIS-CAIS | 262,144 | Four rounds of 65,536 samples, all retained |
| NUTS | 16,384 per chain, 16 chains | 1,024 warmup steps per chain, target acceptance 0.8 |
| Random-walk MH | 16,384 per chain, 16 chains | 1,024 discarded steps per chain, proposal covariance ``2.38^2 I/d`` |
| Slice sampling | 16,384 per chain, 16 chains | 1,024 discarded steps per chain, random-permutation Gibbs with stepping-out width 2 |
| Ensemble DE, Stretch, snooker, Gaussian replacement | 16,384 sweeps per walker | ``4d`` walkers initially, model-specific widths/shrinkage and warmup |

Archived ensemble runs pooled at least 262,144 positions, giving linear only
2,048 retained sweeps. Fresh runs specify the time-axis budget directly.
Linear's stationary Gaussian start needs no warmup; other models initially use
1,024 warmup sweeps. The separate ensemble screen selects move widths per model
before the held-out seeds. Every row records the resolved configuration.
The screen uses seeds 8301–8303 and 4,096 sweeps. It selects by standardized
mean squared error times elapsed time, subject to the moment checks. The
held-out comparison uses seeds 8401–8403 and 16,384 sweeps. Search cost is
separate from per-run fitting and warmup. A failed screen keeps its baseline
and is disclosed separately from held-out diagnostics. Signal-background DE
was the only such case. It is excluded from the selected headline, not from the
raw archive. The recorded runs used EnsembleMCMC `2942c10`. The current
environment pins 0.0.2 at `5fcb74c`, including Gaussian replacement with
shrinkage candidates 0.5, 0 and 1. No new timings are implied by that update.
Other retained counts
match exactly. IS uses batches while MCMC uses correlated
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

Use Julia 1.13 for this pinned benchmark environment. The script prints the
tables and saves per-run estimates, raw times, host allocations, diagnostics,
versions, and hardware metadata. It refuses to overwrite existing results.
Add `--resume` to continue a saved run with the same environment and budgets.
For an independent pilot, pass `pilot=(nsamples=4096, scale_limits=(0.25,2.0))`
to `SamplerComparison.compare`. Use `only_methods` to select a subset.

The published combined table reuses the unchanged IS, AMIS, CAIS, NUTS, MH,
and slice rows with seeds 1001–1003. `refresh.jl` measures only the pilot paths,
first-order GRAMIS, and three ensemble moves with seeds 7101–7103. It also
reuses completed changed IS rows from the interrupted partial report.
Its `--report` option rebuilds the combined table without sampling.
Each source remains archived. Do not reuse its timings as measurements of a
different machine.
Use a quiet accelerator. On a shared CPU host, use a fixed thread budget,
record host load and CPU affinity, and inspect the raw timing spread.
For CPU-only runs or other settings, use `SamplerComparison.compare` as described
in the reproducer README.

To print a saved table without rerunning the samplers:

```sh
julia --project=benchmark/comparison benchmark/comparison/compare.jl --report benchmark/comparison/results-2026-09-16-comparison.toml
```

Use `--accuracy-report benchmark/comparison/accuracy-2026-09-16.toml` for the
earlier common-accuracy comparison. That artifact includes all 800 measured
runs, including ten NUTS divergences on eight schools. The reference chains
have no divergences. Their minimum bulk ESS exceeds 677,000 and their maximum
R-hat is below 1.00003.

Measurements use an AMD EPYC 7702P with 64 physical cores and 128 hardware
threads. Each benchmark process uses 16 Julia threads. CUDA runs use one
A100-PCIE-40GB. The refresh selected 16 distinct physical cores but did not
reserve them. Other users' work remained untouched.
Raw reports retain each sampler revision, benchmark-source hash, manifest hash,
and package versions. The original benchmark source is retained at commit
`11dbb99`. The refresh source accompanies this report.
