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
EnsembleMCMC includes DE, Stretch, and snooker moves with their default parameters.

The [combined report](results-2026-09-16-comparison.md) comes from
`results-2026-09-16-comparison.toml`. Print it without sampling:

```sh
julia --project=benchmark/comparison benchmark/comparison/compare.jl --report benchmark/comparison/results-2026-09-16-comparison.toml
```

The report reuses the published IS, AMIS, CAIS, NUTS, MH, and slice rows.
Only DM-PMC and LAIS with the independent 4,096-draw pilot, first-order GRAMIS,
and the three ensemble moves receive new measurements. Main populations remain
at 16 proposals and four rounds. Existing reference moments are reused.
Every row names its raw source. Archived rows use seeds 1001–1003 and one
timed execution per seed. Refreshed rows use seeds 7101–7103 and the bounded
repeat policy above. Archived rows lack per-run variance checks.

`refresh.jl` performs this selective refresh and combines the reports. It reuses
completed importance-sampler rows from `results-2026-09-16-partial.toml`.
Use `--resume` to continue its saved run or `--report` to rebuild the combined
table without sampling. For a fresh run on another machine, use `compare` with
a new output path and the same explicit settings. Do not reuse recorded timings
as measurements of that machine.

The refreshed measurements record host load and CPU affinity. On Linux,
`taskset --cpu-list` can select a fixed set of cores. No cores are reserved.

CPU and GPU tables are separate. Only the highest CPU rate per model is bold.
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

An ensemble has `4d` walkers. It discards 1,024 warmup sweeps, then retains
enough whole sweeps for at least 262,144 draws. The excess is less than one
sweep. All three moves use `ThreadedExecutor` and the common timed Laplace fit.

For each coordinate, average the walkers within each sweep. Estimate the mean's
MCSE from this time series with MCMCDiagnosticTools, then divide the marginal
variance by the squared MCSE. The reported ESS is the smallest coordinate value.
This follows the ensemble-average construction in
[Goodman and Weare, section 3](https://cims.nyu.edu/~weare/papers/d13.pdf).
The [MCSE implementation](https://julia.arviz.org/MCMCDiagnosticTools/#Monte-Carlo-standard-error)
accounts for serial dependence. Cross-walker lag covariance enters through the
sweep means. R-hat splits that time series, not the walkers. This mean ESS is
not rank-normalized bulk ESS. Short ensemble histories can give noisy estimates.

## Independent width pilot

The optional pilot tunes one common multiplier for the Student-t bank's existing
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
