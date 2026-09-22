# Cross-package sampler comparison

From the repository root, use Julia 1.13 and the committed manifest:

```sh
julia --project=benchmark/comparison -e 'using Pkg; Pkg.instantiate()'
julia --threads=16 --project=benchmark/comparison benchmark/comparison/compare.jl --cuda
```

Omit `--cuda` for CPU only. CUDA is a benchmark dependency, but CPU runs do not
require CUDA hardware. The script prints Markdown tables, package versions, hardware,
thread counts, the package revision, and the manifest SHA-256. It saves each
completed seed to `benchmark/comparison/results.toml` and refuses to overwrite
an existing result unless `--resume` is given. Resume requires the same package
revision, manifest, hardware, thread count, and sample budgets:

```sh
julia --threads=16 --project=benchmark/comparison benchmark/comparison/compare.jl --cuda --resume
```

Print a saved result without sampling again:

```sh
julia --project=benchmark/comparison benchmark/comparison/compare.jl --report benchmark/comparison/results.toml
```

The default uses three timed seeds per method and model. Independent-chain MCMC retains
16,384 draws from each of 16 independent chains. Each importance-sampling run
returns 262,144 weighted draws. Accuracy checks use the analytic linear-regression
posterior and eight reference chains of 8,192 draws for the other models.
Reference runs are untimed and use different seeds.
Keep the accelerator otherwise idle. On a shared CPU host, record load and
inspect timing ranges. BLAS and FFTW use one thread. Julia uses the thread count
given at launch. The 16 CPU MCMC chains use Julia's default thread pool.

Each seed uses up to three timed executions within a five-second BenchmarkTools
budget. A full execution may exceed that budget. The median supplies the seed's
elapsed time. Raw elapsed and GC times remain in the TOML file. CPU and CUDA
calls for the same importance sampler run consecutively, not concurrently.

The headline uses weight ESS for importance samplers, minimum bulk ESS for
NUTS, MH, and slice sampling, and minimum mean ESS for EnsembleMCMC.
These are different diagnostics, each divided by elapsed time.
The report also prints R-hat, divergences, posterior-mean error, and timing ranges.
EnsembleMCMC includes DE, Stretch, snooker, and Gaussian replacement candidates
with model-specific settings. The headline shows only the selected move per
model. All held-out move rows remain in the detailed report. Width-screen trials
remain in the raw screening file.
The September 17 measurements used remote `main` at
`2942c10d5de675863ec5c216e9137ff4e28b81be`, including its workload-aware CPU
scheduling. The current environment pins `5fcb74c1a7c19bedf95795644fb6296901d25d0e`
(0.0.2). The September 22 refresh measures that revision on fresh held-out seeds.
It does not relabel archived measurements or establish a cross-version speedup.
Each report records its measured revision in addition to the package version.

## Scalar and batch regression targets

`batch.jl` compares explicit batch callbacks with the scalar targets on linear,
logistic, Poisson and robust regression. It uses all six importance samplers,
three seeds and 262,144 retained draws per run. It preserves the existing
proposal settings, adaptation budgets and independent DM-PMC/LAIS pilot.
Both the pilot and main run use the selected target mode.

```sh
julia --threads=16 --project=benchmark/comparison benchmark/comparison/batch.jl
julia --project=benchmark/comparison benchmark/comparison/batch.jl --report benchmark/comparison/batch-results.toml
```

The default runs CUDA only. Add `--cpu` for both backends or `--cpu-only` for CPU.
Use `--resume` to continue a saved run with matching settings and provenance.
The script saves raw measurements after each seed and writes a Markdown table.
CPU scalar and CUDA runs use one BLAS thread. Batch CPU uses sixteen BLAS
threads throughout the timed run, including fitting and setup, because the
callback owns parallelism. Each row records this setting.
Compare the full execution strategies, not only callback dispatch overhead.

The CPU callback reuses an observation-by-8192 scratch matrix and threads the
likelihood sums across samples. The GPU callback allocates scratch directly on
the device, once per call, capped at 8192 columns or the actual batch width.
At 1024 observations in Float64, this is at most 64 MiB. Both paths evaluate
likelihood values without computing unused gradients. Scratch allocation and
required data transfers remain inside the timed run. The reported bytes and
allocation counts are **host** measurements, not device memory measurements. CUDA memory
pools and compilation are warm. Each seed interleaves four complete executions
per mode in ABBA and BAAB blocks. Starting order changes across seeds, models
and methods. Both full workloads compile before timing. Full GC runs before
each measured execution, outside timing. CUDA synchronizes inside the timed
operation after the final mean transfer.

Times include fitting, pilot tuning, scratch setup, transfers, sampling and
posterior-mean estimation. Post-run ESS and accuracy diagnostics are excluded.
Each seed uses median timing and the last same-seed result. Diagnostics run only
after both balanced blocks. The report retains execution order, per-call times,
GC time, host allocations and load. Its paired ratio divides mean batch time by
mean scalar time within each block, then reports the median and range across
blocks. A ratio above one means batching took longer.

Interleaving reduces drift but does not remove rapid contention. CPU load also
affects GPU setup. Compare package revisions with the same driver, dependency
versions and settings in interleaved executions. Scalar/batch comparisons do not
test whether the original scalar path regressed. Older blocked-timing reports
remain readable, but cannot resume under this paired protocol.

The September 22 baseline reports cover AMIS and LAIS-RAM on all four regression
models. They use package revision `29c8fc5`, which fixes CPU LAIS closure boxing.
The [CPU report](batch-cpu-2026-09-22-paired.md) and
[CUDA report](batch-cuda-2026-09-22-paired.md) retain every measured time and
accuracy warning. Their raw TOML files include the execution chronology and
source hashes. These are separate scalar/batch comparisons, not replacements
for the cross-package table below. They predate the threaded/value-only callback
and native device-scratch changes described above. Their source hashes identify
the measured implementation; rerunning the current script measures the new callbacks.

The paired medians favour batching for linear regression on CPU and CUDA, and
for LAIS-RAM on CUDA. Nonlinear CPU callbacks and nonlinear CUDA AMIS are slower
in this configuration. Scalar/batch posterior means agree within `2.0e-14` on
CPU and `1.1e-12` on CUDA. Linear LAIS-RAM fails moment checks in both modes on
both backends; all other measured cases pass. The reports retain those warnings.

Run this subset with the current callbacks from the repository root in a session launched with
`--threads=16 --project=benchmark/comparison`:

```julia
include("benchmark/comparison/batch.jl")
BatchComparison.compare(methods=(:amis, :lais), cpu=true, cuda=false,
    output="batch-cpu-new.toml")
BatchComparison.compare(methods=(:amis, :lais), cpu=false, cuda=true,
    output="batch-cuda-new.toml")
```

## Published results

The [current combined report](results-2026-09-22-refresh-comparison.md) shows one
selected EnsembleMCMC move per model. It refreshes all four moves at 0.0.2 and
retains every non-Ensemble row unchanged.
Selection uses the highest mean ESS / mean seconds among the accuracy-eligible,
width-selected candidates in the independent screen. It never selects on the
reporting seeds. This selects DE for eight schools and Gaussian replacement
with shrinkage 1 for the other five models. All 18 selected-move runs pass the
moment checks. Signal-background retains an R-hat warning. Measurements use a
shared host and a different window from the archived non-Ensemble rows.
Raw files retain their original source hashes, versions, and timings.
Print the report without sampling:

```sh
julia --project=benchmark/comparison benchmark/comparison/compare.jl --report benchmark/comparison/results-2026-09-22-refresh-comparison.toml
```

Rebuild the combined TOML and Markdown from the saved raw artifacts:

```julia
include("benchmark/comparison/ensemble.jl")
EnsembleComparison.report(
    ensemblefile="benchmark/comparison/ensemble-results-2026-09-22-refresh.toml",
    screenfile="benchmark/comparison/ensemble-screen-2026-09-22-refresh.toml",
    output="benchmark/comparison/rebuilt-comparison.toml",
)
```

Run this in the comparison environment. It reuses the archived
`signal-background-results-2026-09-17.toml` and writes TOML and Markdown without
sampling or changing the raw files. The fresh screen uses seeds 8301–8303 and
held-out validation uses 9401–9403. The frozen selection and screen hash are in
`ensemble-selection-2026-09-22-refresh.toml`. The combined report records the
screening hash and retains a baseline when no screening candidate passes.
This occurred only for signal-background DE, which is ineligible for the
headline. Both DE and snooker fail held-out variance checks on that model.
The archived CAIS CPU and LAIS-RAM CUDA warnings remain visible.

The [September 17 selection report](results-2026-09-22-comparison.md) remains
unchanged. The default `ensemble.jl --report` command rebuilds that archive,
not the fresh report above.

The [previous combined report](results-2026-09-16-comparison.md) remains archived.
Print it without sampling:

```sh
julia --project=benchmark/comparison benchmark/comparison/compare.jl --report benchmark/comparison/results-2026-09-16-comparison.toml
```

That previous report reuses the published IS, AMIS, CAIS, NUTS, MH, and slice rows.
Only DM-PMC and LAIS with the independent 4,096-draw pilot, first-order GRAMIS,
and the three ensemble moves receive new measurements. Main populations remain
at 16 proposals and four rounds. Existing reference moments are reused.
Every row names its raw source. Archived rows use seeds 1001–1003 and one
timed execution per seed. Refreshed rows use seeds 7101–7103 and the bounded
repeat policy above. Archived rows lack per-run variance checks.

`refresh.jl` reproduces that earlier selective refresh. It reuses
completed importance-sampler rows from `results-2026-09-16-partial.toml`.
Use `--resume` to continue its saved run or `--report` to rebuild the combined
table without sampling. For a fresh run on another machine, use `compare` with
a new output path and the same explicit settings. Do not reuse recorded timings
as measurements of that machine.

The refreshed measurements record host load and CPU affinity. On Linux,
`taskset --cpu-list` can select a fixed set of cores. No cores are reserved.

CPU and GPU tables are separate. Only the highest CPU rate per model is bold.
The separate [fitted-LAIS study](lais.md) includes full AMIS pilot costs,
matched static controls, and the longer-run degradation case. Its named README
rows use two fresh seeds and do not replace the main LAIS-RAM measurements.
First-order GRAMIS-CAIS uses the same 16-proposal Student-t bank and four-round
budget, repulsion strength 0.1, and the models' checked analytic gradients.
The [earlier report](results-2026-09-16-long.md) remains archived. Its benchmark
source is retained at commit `11dbb99`.
The earlier short-chain results remain in `results-2026-09-16.toml`.
The earlier, larger accuracy run is retained in `accuracy-2026-09-16.toml`. Print its separate
accuracy-based comparison with:

```sh
julia --project=benchmark/comparison benchmark/comparison/compare.jl --accuracy-report benchmark/comparison/accuracy-2026-09-16.toml
```

To reuse those completed reference moments without sampling new references:

```julia
include("benchmark/comparison/compare.jl")
references = SamplerComparison.TOML.parsefile("benchmark/comparison/accuracy-2026-09-16.toml")
SamplerComparison.compare(; cuda=true, references)
```

See the [benchmark guide](../../docs/src/guide/benchmarks.md) for the models,
cost basis, ESS definitions, and limits of the comparison.

## Ensemble ESS

Fresh comparisons use `4d` walkers and retain **16,384 sweeps per walker**.
This is a time-axis budget, not 262,144 pooled walker positions. The old linear
configuration had only 2,048 sweeps. Longer histories improve the stability of
the diagnostic, but do not remove autocorrelation or guarantee good mixing.

DE keeps `gamma0=2.38/sqrt(2d)` and `sigma=1e-5`. Linear starts exactly at
stationarity under the fitted Gaussian, so its warmup is zero. Its Stretch width
is `1+2.151/sqrt(d)`, supported by the independent Gaussian study. Other models
start with 1,024 warmup sweeps and their move's default width. These choices are
starting points, not a requirement to use one setting across different targets.
All moves use `ThreadedExecutor` and pay for the common timed Laplace fit.

`ensemble.jl` screens Stretch and snooker widths separately for each model.
It also screens Gaussian replacement shrinkage at 0.5, 0 and 1 using the same
4d walkers. It retains the reviewed DE baseline. It uses three screening seeds, keeps every
trial, and selects by standardized posterior-mean squared error times run time,
subject to the usual mean and variance checks. The subsequent run freezes those
settings and uses three different seeds with 16,384 retained sweeps. Offline
configuration search is separate from each run's timed fit and warmup.

```sh
julia --threads=16 --project=benchmark/comparison benchmark/comparison/ensemble.jl --screen
julia --threads=16 --project=benchmark/comparison benchmark/comparison/ensemble.jl
```

The first command writes `ensemble-screen.toml`; the second writes
`ensemble-results.toml` and prints the table. Both refuse to overwrite results.
The driver requires a completed screen for every requested model and move.
To reuse an older screen without Gaussian replacement, restrict `only_methods`
to `(:ensemble, :stretch, :snooker)`. Gaussian replacement needs a fresh screen.
All six models reuse the references in `results-2026-09-17-comparison.toml`.
Fresh references are generated only for models missing from that archive.
The September 22 refresh uses held-out seeds 9401–9403. Reproduce its protocol
in a Julia session launched with `--threads=16 --project=benchmark/comparison`:

```julia
include("benchmark/comparison/ensemble.jl")
EnsembleComparison.screen(output="screen-new.toml")
EnsembleComparison.compare(screenfile="screen-new.toml", seed_start=9401,
    output="ensemble-new.toml")
```

To select settings directly, pass `ensemble_settings` to `compare`:

```julia
settings = Dict("linear" => Dict("EnsembleMCMC Stretch / CPU" =>
    Dict("scale" => 1.38, "warmup" => 0, "walkers" => 128)))
SamplerComparison.compare(; only_methods=(:stretch,), ensemble_settings=settings)
```

The accepted override keys are `walkers`, `sweeps`, `warmup`, and the move's
`gamma0`/`sigma`, `scale`, or Gaussian `shrinkage`. Results store the resolved values for every seed.
Resume checks the settings, budget, and configuration-search provenance. The
ensemble driver records its source hash, screening file hash, seeds, criterion,
and cases with no eligible candidate. Archived tables keep their original
shorter histories and are not relabelled as measurements of the new settings.

For each coordinate, average the walkers within each sweep. Estimate the mean's
MCSE from this time series with MCMCDiagnosticTools, then divide the marginal
variance by the squared MCSE. The reported ESS is the smallest coordinate value.
This follows the ensemble-average construction in
[Goodman and Weare, section 3](https://cims.nyu.edu/~weare/papers/d13.pdf).
The [MCSE implementation](https://julia.arviz.org/MCMCDiagnosticTools/#Monte-Carlo-standard-error)
accounts for serial dependence. Cross-walker lag covariance enters through the
sweep means. R-hat splits that time series, not the walkers. This mean ESS is
not rank-normalized bulk ESS. Short ensemble histories can give noisy estimates.

## Signal-background

The sixth model is the signal-plus-background posterior from the
[BAT.jl paper example](https://github.com/bat/BAT.jl/blob/3ab3baf1d1666fcd22c08e6f4db3089feec57c39/examples/paper-example/paper_example.jl).
Its unchanged CSV data and license are in [data/signal-background](data/signal-background/README.md).
The target has nine free parameters and 37 events from five detectors. It uses
the original Poisson counts, exponential background, fixed Gaussian signal,
and hierarchical background priors.

The four bounded parent parameters use logistic coordinates. Detector rates use
a noncentred lognormal representation. The target includes the corresponding
Jacobian and prior, and supplies an analytic gradient for NUTS and GRAMIS.
The scalar target serves CPU and CUDA. Diagnostics use these common sampling
coordinates for every method. The benchmark omits parameter-independent density
constants, so its log normalizer is not BAT's absolute evidence.

To repeat the new model's non-ensemble measurements after the ensemble screen:

```julia
include("benchmark/comparison/compare.jl")
references = SamplerComparison.TOML.parsefile("benchmark/comparison/ensemble-screen.toml")
SamplerComparison.compare(; selected=[6], cuda=true, references, seed_start=8501,
    only_methods=(:is, :amis, :dmpmc, :cais, :lais, :gramis, :nuts, :mh, :slice),
    pilot=(nsamples=4096, scale_limits=(0.25, 2.0)),
    output="benchmark/comparison/signal-background-results.toml")
```

The report records both source CSV hashes. It does not infer from truth columns
or event source labels. The reference used eight independent NUTS chains with
8,192 retained draws each. Its maximum R-hat was 1.0004 and minimum bulk ESS
was 23,801. The same reference checks every new-model sampler.

## Independent width pilot

The optional pilot tunes one common multiplier for the Gaussian or Student-t bank's existing
scale factors. It preserves centres, correlation shapes, degrees of freedom,
proposal count, and masses. Its own draw budget does not change the main
sampler's budget or round schedule. A separate random stream supplies pilot
draws, which are discarded from the main result.

```julia
include("benchmark/comparison/compare.jl")
SamplerComparison.compare(;
    pilot=(nsamples=4096, scale_limits=(0.25, 2.0)),
    output="width-pilot-results.toml",
)
```

This pilot applies to DM-PMC and LAIS-RAM. The full pilot cost counts in ESS/s,
and the report saves its duration, draw count, and selected width multiplier.
It uses a bounded scalar search for the estimated importance-weight second
moment. Pilot samples and distance buffers stay on the selected device. The
optimizer reads one scalar objective between evaluations.

The pilot improves the initial fixed linear bank in a controlled check, but
does not resolve its later DM-PMC or LAIS centre adaptation. It remains opt-in.
This is benchmark code, not a new package tuning API. The objective follows the
weight-variance criterion discussed by
[Akyildiz and Miguez](https://arxiv.org/abs/1903.12044).
Their exponential-family convergence results do not establish a guarantee for
this finite Student-t mixture pilot.

## Fixed Gaussian covariances

[Elvira et al., section 5.3](https://victorelvira.github.io/assets/papers/elvira2017improving_pre.pdf)
use Gaussian proposals with fixed covariances `sigma^2 * I`. Resampling changes
their centres, not their covariances. The paper tests several widths. It does
not prescribe an automatic covariance update or a universal bandwidth.

Select this family explicitly in the comparison:

```julia
include("benchmark/comparison/compare.jl")
settings = Dict("linear" => Dict("DM-PMC / CPU" => Dict(
    "family" => "gaussian", "scale" => 1.0, "count" => 256, "rounds" => 4,
)))
SamplerComparison.compare(; selected=[1], only_methods=[:dmpmc], settings,
    output="gaussian-dmpmc.toml")
```

Here the covariance is `scale^2 * L * L'`, where `L` is the timed Laplace factor.
This preserves correlations and applies the isotropic bandwidth in whitened
coordinates. It adapts the paper's setup to regression, rather than copying
its absolute widths into coefficient units. The value above is an example,
not a recommended bandwidth. The optional width pilot also supports this family.

Omitting `family` retains Student-t proposals with eight degrees of freedom.
Their covariance is `8/6 * scale^2 * L * L'`. Archived settings and result
tables retain that interpretation. Family and width changes require new runs.
The pilot fits the initial bank only. It does not prevent later centre
adaptation from degrading the mixture approximation.

## Archived offline configuration search

`tune.jl` searches DM-PMC and LAIS proposal width, population size, and round
count at the full 262,144-draw budget. It preserves every measured trial.
Seed 2101 screens the grid. Seeds 2102 and 2103 check the three best eligible
settings. Eligibility requires mean errors below 0.2 posterior standard
deviations and marginal variance errors below 30% on every pilot seed.
Selection uses the geometric mean of pilot ESS/s. These checks do not establish
tail or mode accuracy.

```julia
include("benchmark/comparison/tune.jl")
PopulationTuning.tune(; cuda=true, output="population-pilots-cuda.toml")
# In a separate process, with the same CPU thread count as the final run:
PopulationTuning.tune(; output="population-pilots-cpu.toml")
```

The archived search used separate seeds 5001–5003 for final comparisons and froze
settings before those runs. Each final run still paid for its own Laplace fit, proposal
preparation, warmup or adaptation, sampling, and mean calculation. The offline
configuration search is separate and must be disclosed with its raw trials.
Any additional per-run proposal-tuning step belongs inside `run_method` and
the timing interval.

The raw pilot artifacts remain separate from the headline table:

| Artifact | Scope |
|:--|:--|
| `population-pilots-2026-09-16-cuda.toml` | Full offline configuration search |
| `results-2026-09-16-tuned-cuda.toml` | Held-out results with the selected population counts and rounds |
| `width-pilot-20260916.toml` | Earlier CPU prototype with stratified pilot draws, not the final random-mixture pilot |
| `rejected-cais-pilot-20260916.toml` | Rejected covariance-pilot experiment |
| `gramis-cuda-20260916.toml` | Initial CUDA GRAMIS validation, superseded by the refresh |
