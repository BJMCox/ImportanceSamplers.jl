# Cross-package sampler comparison

From the repository root, use Julia 1.13 and the committed manifest:

```sh
julia --project=benchmark/comparison -e 'using Pkg; Pkg.instantiate()'
julia --threads=8 --project=benchmark/comparison benchmark/comparison/compare.jl --cuda
```

Omit `--cuda` for CPU only. CUDA is a benchmark dependency, not a requirement
for the CPU run. The script prints Markdown tables, package versions, hardware,
thread counts, the package revision, and the manifest SHA-256. It saves each
completed seed to `benchmark/comparison/results.toml` and refuses to overwrite
an existing result unless `--resume` is given. Resume requires the same package
revision, manifest, hardware, thread count, and sample budgets:

```sh
julia --threads=8 --project=benchmark/comparison benchmark/comparison/compare.jl --cuda --resume
```

Print a saved result without sampling again:

```sh
julia --project=benchmark/comparison benchmark/comparison/compare.jl --report benchmark/comparison/results.toml
```

The default uses five timed seeds per method and model. Accuracy checks use
the analytic linear-regression posterior and 8,192 reference draws per CPU
chain for the other models. Reference runs are untimed and use different seeds.
Use a quiet machine. BLAS and FFTW use one thread. Julia uses the thread count
given at launch. CPU MCMC runs one independent chain per Julia thread.

The headline is weight ESS/s for IS and AMIS, and minimum bulk ESS/s across
parameters for NUTS, MH, and slice sampling. These are different diagnostics.
The report also prints R-hat, divergences, posterior-mean error, and timing ranges.
EnsembleMCMC's coupled walkers appear in accuracy checks, not the chain-ESS table.

The [published report](results-2026-09-16.md) comes from `results-2026-09-16.toml`. The earlier,
larger accuracy run is retained in `accuracy-2026-09-16.toml`. Print its separate
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
