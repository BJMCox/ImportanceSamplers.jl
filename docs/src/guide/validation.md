# [Validation and support](@id validation-guide)

The manual executes its CPU teaching examples during the build.
The package test suite and standalone reproducers check mathematical and device-specific contracts.

A successful example is not proof of accuracy for every model.
A checked-in GPU script is not evidence that it passed on the current machine.

## Run the package tests

From a checkout:

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

CPU CI runs Julia 1.10 and the latest stable Julia.
The latest-stable matrix covers Linux, macOS, and Windows.

Accelerator checks are separate, explicit hardware runs.
They do not run on the default hosted CPU CI jobs.

## Run a focused reproducer

Use the repository's validation environment with its supported Julia version:

```sh
julia --project=validation -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=validation validation/reproducers/plain_is.jl
```

The package supports Julia 1.10, but the validation environment currently requires Julia 1.12 or later.

| Subject | Script under `validation/reproducers/` |
|:--|:--|
| Plain IS identities | `plain_is.jl` |
| Native Gaussian formulas | `native_gaussian.jl` |
| Simplex measure and Jacobian | `simplex_transform.jl` |
| Static MIS schemes | `static_mis.jl` |
| AMIS recurrence | `amis.jl` |
| DM-PMC recurrence | `dm_pmc_global.jl` |
| First-order GRAMIS equations | `first_order_gramis.jl` |
| CUDA samplers | `cuda_plain_is.jl`, `cuda_static_mis.jl`, `cuda_amis.jl`, `cuda_dm_pmc.jl` |
| CUDA population methods | `cuda_apis.jl`, `cuda_cais.jl`, `cuda_npmc.jl`, `cuda_lais.jl` |
| CUDA GRAMIS | `cuda_first_order_gramis.jl` |
| Student-t devices | `cuda_student_t.jl`, `metal_student_t.jl` |
| Batch callbacks | `batch_targets.jl` |
| Reactant execution | `reactant_execution.jl` |

Read each script's setup before running optional-backend cases.
Metal and Reactant checks need their respective packages and hardware.
CPU LAIS recurrence checks are in
[`test/lais.jl`](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/test/lais.jl).

The CUDA GRAMIS wrapper additionally records a supplied source revision:

```sh
git status --short
git rev-parse HEAD
IMPORTANCE_SAMPLERS_SOURCE_COMMIT=PASTE_FULL_COMMIT_HERE julia --project=validation validation/reproducers/cuda_first_order_gramis.jl
```

Inspect the checkout and loaded package path first.
The environment variable records a claim. It does not verify the loaded source.

## Build the manual

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Open `docs/build/index.html`.
The build executes CPU examples and doctests, checks exported docstrings and internal links,
and checks external links.

Repository source links are checked against local files to avoid repeated GitHub rate limits.
That check does not verify the remote branch.
CI deploys the built manual from `main`.

## Record useful evidence

Record the commit and tracked diff, Julia version, package versions, command, seed,
precision, counts, threads, device, and complete result.

For GPU checks, disable scalar indexing and verify array residency.
Use a profiler to establish transfer or kernel claims.
Source-level transfer counters cannot observe every runtime or vendor-library transfer.

For performance, also record compilation boundaries and machine load.
[Benchmarks](@ref benchmarks-guide) separates accuracy, timing, and ESS diagnostics.
