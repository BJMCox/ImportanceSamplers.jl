# Accelerators

ImportanceSamplers uses the same value-transfer pattern as
[Lux device management](https://lux.csail.mit.edu/stable/manual/gpu_management):
prepare ordinary Julia state first, then apply an MLDataDevices device to the
complete prepared sampler. There is no `device=` preparation keyword and no
silent accelerator-to-CPU fallback.

## Minimal CUDA example

This complete example uses a top-level callable and puts both numerical arrays
in the explicit target context `p`:

```julia
using CUDA
using cuDNN
using ImportanceSamplers
using MLDataDevices
using Random

function gaussian_target(x, p)::Float64
    total = 0.0
    for i in eachindex(x)
        z = (x[i] - p.location[i]) / p.scale[i]
        total += -0.5 * abs2(z) - log(p.scale[i]) - 0.5 * log(2pi)
    end
    return total
end

p = (
    location=[0.25, -0.5],
    scale=[0.75, 1.25],
)
proposal = DiagonalGaussian(p.location, p.scale)
prepared = prepare_sampler(
    Xoshiro(42),
    gaussian_target,
    p,
    ImportanceSampling(proposal; nsamples=65_536);
    threaded=true,
)

device = MLDataDevices.gpu_device(nothing; force=true)
prepared = device(prepared)
# Equivalent replacement for the preceding line:
# prepared = prepared |> device

result = importance_sample!(prepared)
host_result = result |> cpu_device()
```

Loading CUDA and cuDNN before calling `gpu_device` activates CUDA selection.
Passing `nothing` as the public scalar policy preserves both `Float32` and
`Float64` state. To select a specific one-indexed device, use
`MLDataDevices.gpu_device(device_id, nothing; force=true)`. The default
`gpu_device()` policy has `eltype(device) === Missing` and is rejected as
`:scalar_policy_unspecified` before the source RNG advances; ImportanceSamplers
does not rewrite that policy through MLDataDevices internals. Explicit
non-`Missing` conversion policies remain explicit user choices and must satisfy
the proposal and target type checks below. Both
`p.location` and `p.scale` are recursively transferred as the complete target
context, together with the proposal state and random buffers; the target
indexes both arrays on the device. `prepared = device(prepared)` and
`prepared |> device` are equivalent. Transfer consumes one seed from the source
RNG to initialize an independently owned device stream. It returns a distinct
prepared handle. The CPU source remains usable and may create other independent
destinations. An accelerator destination is CPU-origin, one-hop state: it can
execute, but cannot itself be transferred to CPU or another accelerator, even
before execution. Such a request fails as `:prepared_migration_unsupported`.

The target returns an unnormalized log density. `result.logweights` contains
the raw values `logtarget(x, p) - logdensityof(proposal, x)`, not normalized
weights. The result's sample, log-weight, and provenance arrays remain resident
on CUDA until the explicit `result |> cpu_device()` transfer above.

## Targets and ownership

Device transfer recursively moves ordinary numerical fields. A custom context
or callable struct with array fields must also register standard Adapt support
so KernelAbstractions can form its isbits kernel representation. Otherwise
transfer fails as `:kernel_argument_unsupported`, before source RNG
consumption. Named tuples such as `p` in the example already satisfy this
conversion contract.

Julia does not expose a reliable general operation for inspecting and
reconstructing an opaque closure's captured environment. A reachable closure
that captures a host array is rejected with [`SamplerDeviceError`](@ref) before
accelerator execution. Global host arrays are likewise not transferred with a
callable. Use a top-level function or an isbits callable, and pass every
numerical array through `p` as in the example.

The CUDA extension constructs each prepared sampler's device RNG with public
`CUDA.RNG(seed)`; it neither retains nor seeds `CUDA.default_rng()`. The
prepared sampler owns its RNG stream and advances it on each run. Treat it
as a mutable, single-owner, non-reentrant handle. Concurrent use of one handle
is unsupported; prepare independent samplers with independent RNGs. An entry
that observes the handle already busy, including recursive re-entry, throws
[`SamplerBusyError`](@ref). This check does not synchronize simultaneous
callers. Placement is fixed once execution begins.

Device-resident `normalized_weights(result)`, `lognormalizer(result)`, and
array slicing are supported. Scalar indexing, iteration, quantiles, and medians
are deliberately unavailable because they imply scalar host access. Transfer
the complete result to CPU first.

## Checked support matrix

The table is generated during every strict documentation build. Its CPU rows
execute public sampling calls, and its rejection rows check the typed device
errors. The A100 cells come from one small metadata file that the real Task 12
reproducer also consumes when constructing and recording its matrix.

```@eval
Main.NATIVE_PLAIN_IS_CAPABILITY_TABLE
```

CUDA execution requires `threaded=true`, a supported native Gaussian and
transform layout, and a target that compiles for the device. Only the table row
labeled A100 execution has real-hardware evidence from Task 12; the other
native row was not A100-validated. Generic proposals and `ProductProposal` are
rejected. AMDGPU and Metal remain unclaimed; a KernelAbstractions backend
alone is not a package support guarantee.

## Backend documentation and reproducer

- [KernelAbstractions quickstart](https://juliagpu.github.io/KernelAbstractions.jl/stable/quickstart/)
  explains portable kernels, array-selected backends, and synchronization.
- [CUDA.jl array programming](https://cuda.juliagpu.org/stable/usage/array/)
  documents `CuArray` residency and explicit host/device copies.
- [MLDataDevices transfer API](https://lux.csail.mit.edu/stable/api/Accelerator_Support/MLDataDevices)
  documents callable devices and recursive data movement.
- [Lux GPU management](https://lux.csail.mit.edu/stable/manual/gpu_management)
  demonstrates the `value |> device` pattern used here.
- `validation/reproducers/cuda_plain_is.jl`
  is the runnable A100 CPU/CUDA scientific, residency, diagnostic, and rejection
  matrix. Run it from the package root with
  `julia --project=validation validation/reproducers/cuda_plain_is.jl` on a
  CUDA host.
