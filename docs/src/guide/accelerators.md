# [Devices](@id devices-guide)

Prepare on CPU, then apply an explicit MLDataDevices device to the complete sampler.
The target data, proposal, random buffers, and adaptation state move together.

## Prepare device-compatible state

```@example devices
using ImportanceSamplers, Random, Statistics
import MLDataDevices

function logtarget(x, p)
    value = zero(eltype(x))
    for i in eachindex(x)
        value -= abs2(x[i] - p.centre[i]) / 2
    end
    return value
end

p = (; centre=Float32[0.25, -0.5, 0.75])
algorithm = ImportanceSampling(
    SphericalGaussian(zeros(Float32, 3), 1.5f0);
    nsamples=10_000,
)
prepared = prepare_sampler(Xoshiro(42), logtarget, p, algorithm)
nothing # hide
```

This example chooses `Float32` so its numerical setup also works on Metal.
CPU and CUDA also support `Float64`.
The package does not require every model to use one precision.

The top-level function holds no hidden data.
Its numerical array lives in `p`, so device transfer can move it explicitly.

## Run on CUDA

Install CUDA.jl in your project, then continue:

```julia
using CUDA

CUDA.allowscalar(false)
device = MLDataDevices.with_eltype(MLDataDevices.CUDADevice(), nothing)
gpu_sampler = prepared |> device
gpu_samples = importance_sample!(gpu_sampler)

gpu_mean = mean(gpu_samples)
host_samples = MLDataDevices.CPUDevice()(gpu_samples)
```

`device(prepared)` and `prepared |> device` are equivalent.
The `nothing` element-type policy preserves the supplied precision.
A device with unspecified scalar policy is rejected.

To select another physical CUDA device explicitly:

```julia
physical = collect(CUDA.devices())[2]
device = MLDataDevices.CUDADevice{typeof(physical),Nothing}(physical)
```

Array summaries remain on the GPU.
`host_samples` is an explicit, independent CPU copy.

## Run on Metal

Use the same numerical setup, with Metal installed:

```julia
using Metal

Metal.allowscalar(false)
device = MLDataDevices.MetalDevice{Float32}()
metal_sampler = device(prepared)
metal_samples = importance_sample!(metal_sampler)
host_samples = MLDataDevices.CPUDevice()(metal_samples)
```

Metal uses `Float32`.
The explicit device policy may convert supplied floating-point state.
Constructing the whole model in `Float32` makes that choice visible from the start.

## Run through Reactant

For the checked NVIDIA GPU path, install Reactant and CUDA:

```julia
using CUDA, Reactant

Reactant.set_default_backend("gpu")
device = MLDataDevices.with_eltype(MLDataDevices.ReactantDevice(), nothing)
compiled_sampler = device(prepared)
compiled_samples = importance_sample!(compiled_sampler)
host_samples = MLDataDevices.CPUDevice()(compiled_samples)
```

CUDA is needed for Reactant's KernelAbstractions integration, including its CPU path.
Native CPU sampling itself has no CUDA dependency.

Device preparation compiles the built-in numerical phases.
Later calls reuse those executables with the current RNG and adapted proposals.
Preparation can therefore cost much more than one warmed run.

## Current backend scope

These entries describe implemented paths with representative hardware checks.
They do not imply that every user target compiles on every backend.

| Backend | Native sampler scope | Main limits |
|:--|:--|:--|
| Ordinary CPU | All methods, plus generic proposals where the method permits them | `Float32`/`Float64` log densities |
| Native CUDA | All methods with supported native Gaussian/Student-t proposals | Device-compatible target and gradient operations |
| Metal | Native Gaussian/Student-t paths for all methods | `Float32`, explicit GRAMIS gradient |
| Reactant GPU | Built-in compiled paths, including batch targets | Traceable operations and fixed prepared schedules |
| Reactant CPU | Plain/static IS, scalar AMIS/NPMC, global DM-PMC, and built-in LAIS cases | Cooperative-kernel paths listed below are rejected |
| AMDGPU | Not validated or claimed | A KernelAbstractions backend alone does not supply full package support |

Reactant CPU rejects GRAMIS, APIS, CAIS, locally resampled DM-PMC,
and factor-proposal AMIS/NPMC during preflight.
Their cooperative kernels hit an upstream CPU-lowering limit.
Use `MLDataDevices.CPUDevice()` for ordinary CPU execution.

Reactant GPU has checked factor Student-t paths for static MIS, AMIS, NPMC, DM-PMC,
APIS, CAIS, and all three LAIS transitions.
Those checks do not establish Student-t GRAMIS support on Reactant.

Native CUDA supports supplied gradients and supported Enzyme reverse-mode gradients.
Metal requires supplied gradients.
See [Gradients](@ref gradients-guide) for the exact interface.

Generic proposals and `ProductProposal` remain CPU-only.
Native `transform=` layouts can expose named, bounded, positive, and simplex parameters.
Do not infer bank support from a standalone transformed proposal.

## Preserve residency and ownership

Samples, log weights, proposal state, and numerical workspaces remain on the device.
Bounded failure and round summaries can synchronize with the host.
A batch callback can also require synchronization.

Scalar indexing, iteration, quantiles, and medians require an explicit CPU result transfer.
Device moments, scalar functionals, normalization, slicing, and resampling have resident paths.
Reactant scalar summaries remain Reactant numbers until explicitly converted.

Apply device transfer before the source sampler's first execution.
Transfer creates a distinct handle and leaves the CPU source usable.
An accelerator-prepared sampler cannot itself migrate to another device or back to CPU.

Use `current_proposal(MLDataDevices.CPUDevice(), sampler)` for a proposal snapshot.
Use [`retarget`](@ref reuse-guide) to rebuild target-dependent state on the same device.

Accelerator setup consumes a seed from the source RNG for an owned backend stream.
Reproducibility does not promise identical streams across devices or package versions.

## Pass state through supported containers

Named tuples already support recursive device conversion.
Custom context or callable structs with arrays need a standard `Adapt.adapt_structure` rule.

Do not capture host arrays in opaque closures or rely on global arrays.
Those arrays are not sampler-owned data and cannot be transferred reliably.

There is no silent CPU fallback for an unsupported device contract.
The exception's `reason` identifies the rejected capability.

## Further details

- [Performance](@ref performance-guide) explains factor batching and compilation costs.
- [Validation](@ref validation-guide) gives runnable backend checks.
- [MLDataDevices](https://lux.csail.mit.edu/stable/api/Accelerator_Support/MLDataDevices) documents callable devices.
- [Reactant CPU lowering issue](https://github.com/EnzymeAD/Reactant.jl/issues/3284) tracks the cooperative-kernel limitation.
