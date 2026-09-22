# Batch target evaluation

Use [`LogTarget`](@ref) with `batch=batch!` when a whole batch can share matrix
operations or launch efficient device kernels. The scalar callback remains
required. Omitting `batch` preserves the existing fused scalar path.

```julia
using ImportanceSamplers, Random

logtarget(x, p) = -sum(abs2, x .- p.centre) / 2
function logtarget_batch!(values, samples, p)
    values .= .-vec(sum(abs2, samples .- p.centre; dims=1)) ./ 2
    return nothing
end

target = LogTarget(logtarget; batch=logtarget_batch!)
p = (; centre=[0.1, -0.2, 0.3])
sampler = prepare_sampler(Xoshiro(42), target, p,
    ImportanceSampling(SphericalGaussian(zeros(3), 1.0); nsamples=10_000))
samples = importance_sample!(sampler)
```

Without an explicit context, use `logtarget(x)` and
`logtarget_batch!(values, samples)`.

The runnable [linear regression example](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/examples/batched_linear_regression.jl)
uses one matrix multiplication for all proposed coefficient vectors:

```julia
include("examples/batched_linear_regression.jl")
samples = batched_regression_example()

# Optional CUDA execution, after installing CUDA:
using CUDA
device = MLDataDevices.with_eltype(MLDataDevices.CUDADevice(), nothing)
gpu_samples = batched_regression_example(device; T=Float32)
cpu_samples = MLDataDevices.CPUDevice()(gpu_samples) # Explicit result transfer.
```

## Shapes and ownership

| One scalar-call input | Batch input | Output |
|:--|:--|:--|
| Scalar | Length-`n` vector | Length-`n` vector |
| Length-`d` vector | `d × n` matrix, one sample per column | Length-`n` vector |
| Named parameters | Named tuple of vector/matrix batch leaves | Length-`n` vector |

Accept array views rather than requiring concrete `Matrix` or `Vector` types.
Fill every output entry. Do not change input arrays or retain borrowed buffers.
The return value is ignored. Batch widths can vary, and empty batches are skipped.
Each output must equal the scalar log target for that sample, independently of
the other samples and the batch width. Missing values, NaN, and +Inf fail the
run. `-Inf` remains a valid outside-support value.

With a named `transform=`, the callback receives logical named batch leaves.
The package adds the log Jacobian once. Do not add it in the callback.

## Execution and derivatives

The callback owns its CPU parallelism. `threaded=true` does not divide one
callback into concurrent calls. Proposal and weighting work retain their usual
threading policy.

Transfer the complete prepared sampler with an explicit MLDataDevices device,
as described in [Accelerators](@ref). The callback receives resident arrays and
context. Launch work on the current task's stream, or finish private-stream work
before returning. No event object or separate executor is required. Validation
can synchronize at batch boundaries. Samples do not move to the host for evaluation.

Adaptive methods evaluate the current round's samples together. AMIS retains
previous target values. LAIS batches each dependent MCMC step separately.
GRAMIS batches frozen-centre values and each backtracking trial's active
candidates. It does not evaluate future dependent trials in advance.

`grad=` and `adtype` remain independent of `batch`. Automatic differentiation
uses the scalar callable, including any primal evaluations needed by the AD
backend. An explicit gradient need not evaluate the scalar log target. This API
does not define a batch-gradient callback.

## Backend limits

Native batch paths have been checked on CPU, CUDA, and Metal, including named
simplex and positive parameters. The callback's operations must support the
selected backend. Metal examples use Float32 throughout, including bank masses.

Reactant callbacks must be traceable. Its fixed-width batch phases retain their
compiled executables. Device transfer rejects explicit-batch GRAMIS because its
active backtracking width changes between trials. It does not pad callbacks or
compile new widths during sampling. Scalar GRAMIS remains available on Reactant
GPU backends, subject to the limits in [Accelerators](@ref).

Reactant 0.2.285 also fails when filling a one-element vector view during
two-round AMIS/NPMC preparation. This affects scalar and batch targets. The
batch comparisons used three rounds, which pass. See the
[backend reproducer](https://github.com/BJMCox/ImportanceSamplers.jl/blob/main/validation/reproducers/batch_targets.jl)
for the checked cases.

Small, cheap scalar targets can be faster without batching because fusion
avoids scratch storage and phase boundaries. Benchmark both paths for the
actual target and device. Batch matrix operations offer more opportunity when
many observations share the same design matrix.
