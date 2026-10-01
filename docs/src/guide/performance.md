# [Performance](@id performance-guide)

Measure the complete workflow that matters to your application.
Separate preparation, compilation, sampling, summaries, and explicit transfers.

## Use CPU threads deliberately

Launch Julia with a chosen thread budget:

```sh
julia --threads=8 --project
```

Sampling requests Julia's default thread pool unless you pass `threaded=false`.
With only one default thread, the request executes serially.
Prepared samplers keep their threading policy for later calls.

The coordinator owns RNG consumption.
Native workers use prefilled random buffers.
Generic proposal draws occur before threaded target and density evaluation.

Targets and proposal densities must be thread-safe.
Explicit [batch callbacks](@ref batch-guide) own their own parallelism.
Avoid oversubscribing the machine with both outer threads and multithreaded BLAS.

The largest thread count need not be fastest.
Compare several budgets on the same workload.

## Compare factor execution policies

```@example performance
using ImportanceSamplers, Random, LinearAlgebra

proposal = FactorGaussian(zeros(3), Matrix{Float64}(I, 3, 3))
algorithm = ImportanceSampling(proposal; nsamples=5_000)
prepared = prepare_sampler(
    Xoshiro(42), x -> -sum(abs2, x) / 2, algorithm;
    factor_execution=BatchedFactorExecution(),
)
samples = importance_sample!(prepared)
samples.diagnostics.factor_execution_policy
```

CPU defaults to fused sample kernels.
Supported accelerator factor paths default to batched multiplication and triangular solves.

`FusedFactorExecution()` and `BatchedFactorExecution()` request a policy explicitly.
There is no machine-specific sample-count threshold.
Unsupported batched cases use the fused path without changing the estimator.

Plain factor IS batching requires a Gaussian numerical proposal.
Packed bank/history paths also support Student-t factors.
Static MIS batching requires a full-mixture denominator.
Mixed sample/log-weight precision can require an explicit batched request.

The diagnostic records the resolved policy.
It does not prove that every phase used a matrix operation.

## Time warmed sampling

Use BenchmarkTools in a benchmark environment:

```julia
using BenchmarkTools, ImportanceSamplers, Random

algorithm = ImportanceSampling(SphericalGaussian(zeros(3), 1.5); nsamples=100_000)
prepared = prepare_sampler(Xoshiro(42), x -> -sum(abs2, x) / 2, algorithm)
importance_sample!(prepared) # Compile the path before timing.
@benchmark importance_sample!($prepared)
```

For CUDA, synchronize within the timed expression:

```julia
@benchmark CUDA.@sync importance_sample!($gpu_sampler)
```

A warmed adaptive benchmark retains its learned proposal across calls.
It is not a fresh-initialization accuracy experiment.
Use independent prepared samplers when comparing fresh-run accuracy.

## Account for memory and compilation

Every adaptive result retains all configured rounds.
AMIS additionally keeps proposal history and revisits denominators.
Data-heavy batch targets can allocate much more scratch than a scalar target.

Reported Julia allocations measure host allocation traffic, not peak device memory.
A cached GPU memory pool is not the same as live sample storage.

Reactant prepares built-in sampling phases once for the fixed schedule.
Result operations cache up to 64 executable signatures per process.
Further signatures and view operations may compile eagerly.
Functionals with captured state also compile eagerly so state changes remain visible.
Custom LAIS transitions do not receive the built-in compilation guarantees.

## Compare fairly

Interleave baseline and candidate timings on shared hardware.
Record thread counts, BLAS threads, device, precision, budgets, and host load.
Include pilots and upper-chain work in end-to-end comparisons.

Measure the error of the actual estimand as well as weight ESS.
See [Benchmarks](@ref benchmarks-guide) for reproducible model comparisons and accuracy reports.
