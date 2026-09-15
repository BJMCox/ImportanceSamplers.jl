# Validation and support

Support claims have three distinct sources:

- **Docs-build CPU checks** execute public preparation and sampling calls when
  the manual builds. They check representative inputs, not every possible target.
- **Reproducer contracts** describe the cases a checked-in script exercises.
  A script's presence does not prove that it passed on the current revision.
- **Hardware results** apply to the recorded source, package versions, device,
  scalar types, and cases. They do not establish AMDGPU or Metal support.

The strict build checks doctests, exported docstrings, and links. Private GitHub
source links listed in `linkcheck_ignore` are exempt from HTTP checks because
they require authentication. This exemption is not evidence of link availability.

## Current method coverage

The generated matrices live with their method contracts:

| Method | Capability documentation | Runnable validation under `validation/reproducers/` |
|:--|:--|:--|
| Plain IS | [Accelerators](@ref) | `plain_is.jl`, `cuda_plain_is.jl` |
| Static MIS | [Static multiple importance sampling](@ref) | `static_mis.jl`, `cuda_static_mis.jl` |
| AMIS | [Adaptive multiple importance sampling](@ref) | `amis.jl`, `cuda_amis.jl` |
| DM-PMC | [DM-PMC, GR-PMC, and LR-PMC](@ref) | `dm_pmc_global.jl`, `cuda_dm_pmc.jl` |
| GR-PMC | [DM-PMC, GR-PMC, and LR-PMC](@ref) | `dm_pmc_global.jl`, `cuda_dm_pmc.jl` |
| LR-PMC | [DM-PMC, GR-PMC, and LR-PMC](@ref) | `cuda_dm_pmc.jl` |
| APIS | [Adaptive population importance sampling](@ref), table below | `cuda_apis.jl` |
| LAIS | [Layered importance sampling](@ref), table below | `cuda_lais.jl`; CPU recurrence checks in `test/lais.jl` |
| CAIS | [Canonical covariance-adaptive importance sampling](@ref), table below | `cuda_cais.jl` |
| NPMC | [Nonlinear population Monte Carlo](@ref), table below | `cuda_npmc.jl` |
| First-order GRAMIS-CAIS | [First-order GRAMIS-CAIS](@ref) | `first_order_gramis.jl`, `cuda_first_order_gramis.jl` |

DM-PMC, GR-PMC, and LR-PMC share a constructor with explicit allocation,
weighting, and resampling controls. Their method guide defines the combinations.

The following rows execute two 64-sample rounds during each docs build.
APIS, LAIS and CAIS use a one-proposal bank. NPMC uses one proposal. Each target is
the initial proposal's log density. The checks verify the returned count.
Independent recurrence and failure checks belong to the test suite and CUDA
reproducers, not this documentation table.

```@eval
Main.POPULATION_CAPABILITY_TABLE
```

These are serial CPU checks. They do not validate CUDA compilation, threaded
execution, gradients, arbitrary user targets, or every combination of controls.

## Native Student-t proposals

The [Accelerators](@ref) table executes scalar, spherical-vector, diagonal, and
factor Student-t proposals on CPU. These rows are separate from the Gaussian
A100 metadata. `cuda_student_t.jl` covers selected scalar, diagonal, factor,
fractional-degree, Cauchy, and transformed cases. It does not validate the full
Cartesian product of shapes, degrees of freedom, types, and transforms.

Student-t proposals also use packed static MIS and adaptive CPU/CUDA execution.
The shared-family validation covers correlated Student-t banks and fixed-degree
adaptation. The older Gaussian capability matrices below remain Gaussian
reference checks; they do not imply every Student-t shape/type combination was
hardware-tested. See [Native proposals](@ref) for the adaptation requirements.

`validation/reproducers/metal_student_t.jl` checks AMIS, NPMC, CAIS and
FirstOrderGRAMIS covariance adaptation with correlated factor Student-t proposals.
It checks known Gaussian means and normalizers, repeated sampling and owned
results. The NPMC case also checks explicit Float64-to-Float32 device conversion.
Run it in an environment containing ImportanceSamplers and Metal, with Metal
scalar indexing disabled as in the script.

## Run and record validation

The release audit on 2026-09-08 checked the production source incorporated in
`298933116f9d345b2733f36b219d79c47ffb8643`. The merged CPU suite passed 6,872
checks on Julia 1.12.7. The preceding uncommitted CUDA candidate passed APIS
52/52, CAIS 60/60, and FirstOrderGRAMIS 90/90 on an A100 PCIe 40GB with
CUDA.jl 6.3.1, KernelAbstractions 0.9.42, and MLDataDevices 1.17.10.
The final candidate diff matched the committed diff. CUDA scalar indexing was
disabled. These results cover those validators, not every method or future
revision. The CUDA run recorded a dirty candidate, not a then-existing commit.

Run CPU tests and the manual from the repository root:

```sh
julia --project -e 'using Pkg; Pkg.test()'
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

On a CUDA host with the validation environment installed, run the selected
reproducer. For example:

```sh
julia --project=validation validation/reproducers/cuda_cais.jl
```

Record the source commit, tracked diff when dirty, Julia and package versions,
GPU model, scalar types, seed, command, and complete result. Use `versioninfo()`
and `Pkg.status()` for Julia/package context and `CUDA.versioninfo()` for CUDA.
Do not label an uncommitted checkout with only its base commit.

The GRAMIS wrapper additionally requires `IMPORTANCE_SAMPLERS_SOURCE_COMMIT`.
For a clean, inspected checkout:

```sh
git status --short
git rev-parse HEAD
```

Copy the full commit printed above into the environment variable:

```sh
IMPORTANCE_SAMPLERS_SOURCE_COMMIT=PASTE_FULL_COMMIT_HERE julia --project=validation validation/reproducers/cuda_first_order_gramis.jl
```

The wrapper records the supplied string. It does not independently prove that
the loaded package matches that commit. Verify the loaded package path and
checkout before claiming exact-revision validation.

Performance measurements need separate provenance: thread count, BLAS thread
count, device, precision, sample/round counts, warmup, timed boundary, and server
load. Use BenchmarkTools and synchronize GPU work inside the timed boundary.
Report repeated ranges. A single-seed concentration ESS is not a general claim
about estimator accuracy or cross-device ESS per second.

## Compare complete-run cost and accuracy

`benchmark/time_to_accuracy.jl` compares all shipped samplers against independent
normalizer, mean and second-moment references. Its four two-dimensional targets
are a correlated Gaussian, a Student-t, an unequal separated Gaussian mixture,
and two independent logistic/Bernoulli groups. The first three use analytic
references. The logistic reference uses one-dimensional quadrature per group.

The harness includes plain IS, four static MIS pairs, AMIS, NPMC, global/local
DM-PMC, APIS, CAIS, first-order GRAMIS, and LAIS with random-walk, RAM and sample-MH
transitions. It matches total lower-sample budgets and uses four adaptive rounds.
Initial single and bank proposals match mean and covariance, not density shape.
The heavy-tailed target uses Student-t proposals with heavier tails.

Start a Julia process with the desired thread count and benchmark environment:

```sh
julia --threads=32 --project=benchmark
```

Then measure one case:

```julia
using ImportanceSamplers, MLDataDevices, LinearAlgebra
include("benchmark/time_to_accuracy.jl")
BLAS.set_num_threads(1) # Avoid nested BLAS threading in this benchmark process.

device = CPUDevice()
# For CUDA, load CUDA and select a precision-preserving device explicitly:
# using CUDA
# CUDA.allowscalar(false)
# device = MLDataDevices.with_eltype(CUDADevice(), nothing)

T = Float64 # An explicit benchmark choice, not a package precision requirement.
case = TimeToAccuracy.cases(T).mixture
methods = TimeToAccuracy.algorithms(case, 1_048_576; T)
rows = TimeToAccuracy.measure(device, case, methods.cais; seeds=1:20, threaded=true)
TimeToAccuracy.summarize(rows)
```

Iterate over `TimeToAccuracy.cases(T)`, both total budgets `262_144` and
`1_048_576`, and `pairs(TimeToAccuracy.algorithms(case, total; T))` for the full
comparison. Run each backend separately. Record Julia/BLAS threads and host load.
Use `threaded=false` for an explicit serial control.

Each measured seed creates a fresh prepared sampler, transfers it to the device,
samples, and synchronizes completion. CPU runs retain the already-CPU preparation
instead of rebuilding it through a redundant device transfer.
BenchmarkTools performs an unmeasured fresh
warmup first. The timed boundary includes preparation and adaptation, but excludes
initial JIT compilation, oracle evaluation and result postprocessing. It never
inherits an earlier run's adapted proposal. Host bytes and allocation counts are
cumulative allocations, not peak memory or device allocations.
Forced per-seed garbage-collection sweeps are disabled. Natural collection stays
inside timing and each row records its `gc_seconds`. The full audit collects
between cells, not before every seed.

The reported errors are relative normalizer error, coordinate-standardized mean
error, and full raw second-moment error scaled by marginal standard deviations.
The latter includes cross moments. Seed-bootstrap intervals summarize RMSE
uncertainty. They do not establish correctness or tail-error bounds.
`mse_seconds` multiplies empirical MSE by mean runtime, including natural GC costs.
Both mean and median latency remain available. Smaller scores indicate
better error-cost trade-offs in that measured cell, not a predicted runtime to
arbitrary accuracy. Compare the observed error/time pairs at both budgets.

ESS measures weight concentration. It does not replace estimator error, and
equal lower-sample counts do not imply equal target/gradient work. Each row records
those logical sampling evaluation counts, not preparation probes or proposal
density evaluations. Defaults are held fixed rather than tuned per target, so
this small workload set cannot establish a universal sampler ranking.

For example, the million-sample logistic case on an A100 gave AMIS normalizer
RMSE `3.0e-4`, versus `2.9e-3` for partial deterministic-mixture IS. Their mean
fresh-run times were 211 ms and 4.7 ms. AMIS had the lower normalizer error-cost
score, but partial MIS had lower mean and second-moment error-cost scores.
Choose the estimand before choosing a sampler. These are 20-seed observations
from this benchmark, not general speed or accuracy guarantees.

BAT/Wren adapters are deferred until registration and will live in those host packages.
No BAT, Wren, Enzyme, or Reactant integration claim follows from these checks.
