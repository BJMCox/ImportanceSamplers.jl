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
using ImportanceSamplers
using LinearAlgebra
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

physical = CUDA.device()
device = MLDataDevices.CUDADevice{typeof(physical),Nothing}(physical)
prepared = device(prepared)
# Equivalent replacement for the preceding line:
# prepared = prepared |> device

result = importance_sample!(prepared)
host_result = result |> cpu_device()
```

## Factor execution policy

Supported factor paths use fused sample kernels by default on CPU. Accelerators
use batched matrix multiplication and triangular solves when their extension
supports that path. This includes native Gaussian factors and homogeneous
Gaussian or Student-t banks and adaptive histories. Plain single-proposal
Student-t IS remains fused. This backend rule uses no hardware or sample-count cutoff.

Override the default explicitly when benchmarking another path:

```julia
dimension = 32
factor_context = (location=zeros(dimension), scale=ones(dimension))
factor_proposal = FactorGaussian(factor_context.location, Matrix(I, dimension, dimension))
prepared = prepare_sampler(
    Xoshiro(42),
    gaussian_target,
    factor_context,
    ImportanceSampling(factor_proposal; nsamples=5_000_000);
    factor_execution=BatchedFactorExecution(),
    threaded=true,
)
prepared = prepared |> device
```

The same policy applies to Base IS, static MIS, DM-PMC, AMIS, APIS, LAIS, CAIS, NPMC, and GRAMIS. It
remains part of the prepared sampler during device transfer. Use
`FusedFactorExecution()` to force fusion or `BatchedFactorExecution()` to force
batching where supported.

The default batch path requires matching native factor/log-weight precision and
CPU or CUDA support. Explicit `BatchedFactorExecution()` also permits supported
native MIS paths with wider log weights, using separate denominator storage.
Base IS requires an untransformed Gaussian factor proposal. A named target layout
passed through `transform=` still keeps that numerical factor path. Packed static MIS
and adaptive paths also support Student-t factors. Static MIS requires a
full-mixture denominator, as used by stratified and random-mixture MIS.
Unsupported cases use the fused path without changing the estimator.

Benchmark both policies on the target device because the crossover depends on
the factor dimension, backend, scalar type, and hardware. The result records
the resolved default or explicit policy as `:fused` or `:batched` in
`diagnostics.factor_execution_policy`.

The explicit `CUDADevice` construction avoids MLDataDevices backend
auto-selection and needs no cuDNN dependency. The `Nothing` scalar policy
preserves both `Float32` and `Float64` state. To select a specific one-indexed
device, use:

```julia
device_id = 2
physical = collect(CUDA.devices())[device_id]
device = MLDataDevices.CUDADevice{typeof(physical),Nothing}(physical)
```

The default `MLDataDevices.CUDADevice()` policy has
`eltype(device) === Missing` and is rejected as `:scalar_policy_unspecified`
before the source RNG advances; ImportanceSamplers does not rewrite that policy
through MLDataDevices internals. Explicit
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
conversion contract. This includes a named callable struct that subtypes
`Function`: an explicit Adapt rule transfers its numerical state, while the
generic compiler-closure reconstruction rule is not used.

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

Device-resident `normalized_weights(result)`, `lognormalizer(result)`,
`mean(result)`, `var(result)`, `std(result)`, `cov(result)`, and array slicing
are supported. Array summaries remain on the input device. Scalar summaries
return a scalar. Function summaries compile the function for that device and
require one concrete scalar output per sample.

`resample(rng, result, count)` also stays resident. On CUDA it consumes one
`UInt64` seed from the supplied RNG, fills device random buffers with an
owned CUDA RNG, and returns device-resident `UnweightedSamples`. Transfer those
samples explicitly with `cpu_device()` before scalar indexing.

Scalar indexing, iteration, quantiles, and medians are deliberately unavailable
because they imply scalar host access or global ordering. Transfer the complete
result or a sliced `WeightedSampleView` directly to CPU first.

## Checked support matrix

The table is generated during every strict documentation build. Its CPU rows
execute public sampling calls, and its rejection rows check the typed device
errors. The A100 cells come from metadata that the CUDA reproducer also
consumes when constructing and recording its matrix.

```@eval
Main.NATIVE_PLAIN_IS_CAPABILITY_TABLE
```

CUDA execution requires `threaded=true`, a supported native proposal and
transform layout, and a target that compiles for the device. Gaussian A100
metadata does not describe Student-t coverage. The separate Student-t reproducer
and its limits appear in [Validation and support](@ref). Generic proposals and `ProductProposal` are
rejected. AMDGPU remains unclaimed; a KernelAbstractions backend alone is not
a package support guarantee.

Static MIS uses the same transfer and residency contract. Its generated bank
matrix and scheme-specific limits are maintained on the
[Static multiple importance sampling](@ref) page.

## Metal

Load `Metal` and transfer a prepared sampler with
`MLDataDevices.MetalDevice{Float32}()`. This explicitly converts numerical state
to the precision supported by Metal. Sampling, owned-result reuse and resident
resampling have been checked with native Gaussian proposals for Base IS, static
MIS, AMIS, NPMC, DM-PMC, APIS, CAIS, LAIS and first-order GRAMIS.
GRAMIS requires a supplied gradient on Metal. Automatic Enzyme gradients are
not supported by the tested Metal backend.

Float32 factor Student-t proposals have also passed plain IS, static MIS,
DM-PMC, APIS, LAIS with random-walk Metropolis, AMIS, NPMC, CAIS and GRAMIS,
including owned-result reuse and resident resampling. Covariance adaptation
uses the Student-t family's fixed normalization term prepared on the CPU,
then updates only the scale determinant on device.

Metal uses 32-bit atomic failure records. This fallback needs eight bytes per
logical sample slot plus eight bytes for its count. Successful runs read only
the count. An error also copies the diagnostic payload to report the first
failing sample and transform block. CPU and CUDA retain their compact record.

## Reactant

Load `Reactant` and `CUDA`, then apply
`MLDataDevices.with_eltype(MLDataDevices.ReactantDevice(), nothing)` to the
prepared sampler. CUDA must be loaded for Reactant's KernelAbstractions
integration even when Reactant runs on CPU.

Native Base IS, static MIS, scalar/vector AMIS/NPMC, DM-PMC, APIS, CAIS, LAIS
and first-order GRAMIS retain compiled numerical phases in the prepared sampler.
Device preparation compiles the required phases once. Later runs reuse those
executables with live RNG state, resident context arrays and adapted proposal
arrays. Adaptive methods prepare phases for their fixed round schedules.
Preparation can therefore take seconds or minutes even when warmed sampling is fast.
LAIS retains execution for `RandomWalkMetropolis`, `RAM` and
`SampleMetropolisHastings`. Custom LAIS transitions and standalone result
operations still use eager compile-and-run calls. Their compatibility checks
do not establish competitive Reactant throughput.
On the tested NVIDIA A100, GRAMIS also supports
`LogTarget(logtarget, AutoEnzyme())` with resident array context,
simplex/positive/identity fields, adaptation, reuse and `retarget`.

The A100 checks also cover static MIS, AMIS, NPMC, DM-PMC, APIS, CAIS and LAIS
with `Float32` `FactorGaussian` proposals. LAIS covers RandomWalkMetropolis, RAM and
SampleMetropolisHastings transitions. These checks use a shifted correlated
Gaussian target, named fields and resident array context, including prepared
sampler reuse, adaptive retargeting and resident resampling. They establish
correctness for these cases, not throughput or other proposal families.

Float32 factor Student-t checks also cover static MIS, AMIS, NPMC, DM-PMC,
APIS, CAIS and all three LAIS transitions, with reuse, adaptive retargeting and
resident resampling. This does not establish Student-t GRAMIS support on Reactant.

`normalized_weights`, `lognormalizer` and `resample` support Reactant results
and their applicable view operations. Result normalization transfers two
scalars. CDF construction compiles normalization and cumulative summation
together, then transfers two summary scalars once. Adaptive weight normalization
transfers three summary scalars. Moment fitting compiles the weighted mean and
covariance together; Cholesky reads one success flag. The fitted factor and
proposal history remain on-device.

The extension compiles the batch gradient with supported Reactant options that
preserve nonlinear transform derivatives. It does not modify Reactant. A
gradient batch reads one status scalar on the host. Round summaries copy only
bounded diagnostics; samples and gradient arrays remain on-device.

Reactant's CPU backend supports plain IS and the compiled gradient calculation,
but cannot compile GRAMIS's cooperative kernels. Device transfer rejects that
combination before the compiler can abort Julia. Use `CPUDevice()` for CPU
GRAMIS. Factor-proposal AMIS and NPMC also remain rejected during Reactant CPU
preflight. Their new factorization support applies to GPU execution; it does
not enable the unvalidated cooperative CPU paths. Other Reactant CPU adaptive
methods and non-NVIDIA accelerators remain unvalidated. Metal GRAMIS still
requires an explicit gradient.

## Profile data-heavy GRAMIS targets

`benchmark/logistic_gramis.jl` profiles the logistic-regression example with
Float64 proposals, resident data and a supplied gradient. It reports preparation,
the first sampling call, warmed BenchmarkTools timings, CUDA/CUPTI device
activities and a separate Julia host profile. Warm runs retain adaptation;
they are not fresh-run accuracy comparisons.

Instantiate the `benchmark` environment, then run this from the repository root:

```julia
using CUDA, MLDataDevices, Reactant
include("benchmark/logistic_gramis.jl")

device = MLDataDevices.with_eltype(MLDataDevices.CUDADevice(), nothing)

# For Reactant, use these two lines instead in a fresh Julia process:
# Reactant.set_default_backend("gpu")
# device = MLDataDevices.with_eltype(MLDataDevices.ReactantDevice(), nothing)
report = LogisticGRAMISBenchmark.measure(device; observations=1000, round_size=65536)
```

Repeat with 50 observations and 262144 samples per round. The profile uses eight
proposals and four rounds. It disables scalar CUDA indexing. No target or gradient
evaluation copies the observations to the host. Keep instrumented device times
separate from uninstrumented wall times when interpreting the results.

An A100 PCIe 40GB run on Julia 1.13.0, CUDA 6.2.2 and Reactant 0.2.285 gave
these warm medians, using 20 trials and each backend's default factor policy:

| Observations | Samples/round | Native CUDA (ms) | Reactant (ms) |
| --- | --- | --- | --- |
| 50 | 65536 | 12.98 | 29.98 |
| 50 | 262144 | 21.88 | 34.79 |
| 1000 | 65536 | 75.00 | 67.50 |
| 1000 | 262144 | 95.27 | 78.06 |

The shared host was under load. Reactant preparation took 39–251 seconds across
these cases, including first-use compilation. No Reactant compiler frames were
sampled during the warm host profiles. This is not a universal backend ranking.

For 1000 observations, the four backtracking kernels took about 55.4 ms on
native CUDA and 28.3–29.8 ms through Reactant. Each proposal's GPU thread still
evaluates the scalar target's observation loop and backtracking trials serially.
Increasing sample count amortizes that work but does not parallelize it.

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
- `validation/reproducers/cuda_static_mis.jl` is the corresponding packed-bank
  static-MIS correctness and benchmark matrix.
