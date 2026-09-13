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

BAT/Wren extensions are deferred until the core feature set is complete.
No BAT, Wren, Enzyme, or Reactant integration claim follows from these checks.
