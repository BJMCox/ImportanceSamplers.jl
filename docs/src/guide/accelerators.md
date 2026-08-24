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

device = MLDataDevices.CUDADevice()
prepared = device(prepared)
# Equivalent replacement for the preceding line:
# prepared = prepared |> device

result = importance_sample!(prepared)
host_result = result |> cpu_device()
```

Loading CUDA before constructing `CUDADevice()` activates the backend. Both
`p.location` and `p.scale` are recursively transferred as the complete target
context, together with the proposal state and random buffers; the target
indexes both arrays on the device. `prepared = device(prepared)` and
`prepared |> device` are equivalent. Transfer consumes one seed from the source
RNG to initialize an independently owned device stream. It returns a distinct
prepared handle; source and destination remain usable, and each handle's
placement is fixed only when that handle begins execution.

The target returns an unnormalized log density. `result.logweights` contains
the raw values `logtarget(x, p) - logdensityof(proposal, x)`, not normalized
weights. The result's sample, log-weight, and provenance arrays remain resident
on CUDA until the explicit `result |> cpu_device()` transfer above.

## Targets and ownership

Device transfer can recursively move ordinary numerical fields, but Julia does
not expose a reliable general operation for inspecting and reconstructing an
opaque closure's captured environment. A reachable closure that captures a
host array is rejected with [`SamplerDeviceError`](@ref) before accelerator
execution. Global host arrays are likewise not transferred with a callable.
Use a top-level function or an isbits callable, and pass every numerical array
through `p` as in the example.

The prepared sampler owns its RNG stream and advances it on each run. Treat it
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
native row was not A100-validated. Generic proposal RNGs and `ProductProposal`
are rejected. AMDGPU and Metal remain unclaimed; a KernelAbstractions backend
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
