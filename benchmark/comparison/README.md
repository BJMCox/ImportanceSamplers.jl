# Cross-package sampler comparison

From the repository root, use Julia 1.13 and the committed manifest:

```sh
julia --project=benchmark/comparison -e 'using Pkg; Pkg.instantiate()'
julia --threads=16 --project=benchmark/comparison benchmark/comparison/compare.jl --cuda
```

Omit `--cuda` for CPU only. CUDA is a benchmark dependency, not a requirement
for the CPU run. The script prints Markdown tables, package versions, hardware,
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

The default uses three timed seeds per method and model. Each MCMC run retains
16,384 draws from each of 16 independent chains. Each importance-sampling run
returns 262,144 weighted draws. Accuracy checks use the analytic linear-regression
posterior and eight reference chains of 8,192 draws for the other models.
Reference runs are untimed and use different seeds.
Use a quiet machine. BLAS and FFTW use one thread. Julia uses the thread count
given at launch. The 16 CPU MCMC chains use Julia's default thread pool.

The headline is weight ESS/s for IS, AMIS, DM-PMC, CAIS, and LAIS-RAM, and
minimum bulk ESS/s across parameters for NUTS, MH, and slice sampling.
These are different diagnostics.
The report also prints R-hat, divergences, posterior-mean error, and timing ranges.
EnsembleMCMC's coupled walkers appear in accuracy checks, not the chain-ESS table.

The [published report](results-2026-09-16-long.md) comes from `results-2026-09-16-long.toml`.
Bold denotes the highest measured rate per model. The measured benchmark source
is retained at commit `11dbb99`. Later table-formatting changes do not alter sampling.
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
