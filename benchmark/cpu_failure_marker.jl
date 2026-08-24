using BenchmarkTools
using ImportanceSamplers
using Random

const SMOKE_MODE = "--smoke" in ARGS
const BENCHMARK_SEED = 0x4350554641494c
const RESET_SAMPLE_COUNTS = (100_000, 1_000_000)
const COMPLETE_SAMPLE_COUNT = 1_000_000
const TRIAL_SAMPLES = SMOKE_MODE ? 2 : 10
const TRIAL_SECONDS = SMOKE_MODE ? 0.05 : 5.0
const RESET_TRIAL_SAMPLES = SMOKE_MODE ? 20 : 1_000

struct CPUFailureBenchmarkTarget
    offset::Float64
end

function (target::CPUFailureBenchmarkTarget)(sample::Float64)::Float64
    return target.offset - abs2(sample)
end

function prepared_sampler(nsamples, threaded, seed)
    algorithm = ImportanceSampling(SphericalGaussian(0.25, 1.5); nsamples)
    return prepare_sampler(
        Random.Xoshiro(seed),
        CPUFailureBenchmarkTarget(0.75),
        algorithm;
        threaded,
    )
end

function run_trial(benchmark)
    return run(
        benchmark;
        samples=TRIAL_SAMPLES,
        seconds=TRIAL_SECONDS,
        evals=1,
    )
end

function run_reset_trial(benchmark)
    tune!(benchmark)
    return run(
        benchmark;
        samples=RESET_TRIAL_SAMPLES,
        seconds=TRIAL_SECONDS,
    )
end

function format_trial(trial)
    estimate = median(trial)
    return "$(BenchmarkTools.prettytime(estimate.time)); " *
           "$(estimate.allocs) allocs; " *
           BenchmarkTools.prettymemory(estimate.memory)
end

function failure_scratch(sampler)
    buffers = getfield(sampler, :random_buffers)
    return getfield(buffers, :failure_scratch)
end

Base.@noinline function reset_and_observe!(scratch)
    ImportanceSamplers._reset_native_failure_scratch!(scratch)
    record = getfield(getfield(scratch, :record), :storage)
    failures = getfield(scratch, :target_failures)
    return record[1], getfield(failures, :first_index)[]
end

Base.@noinline function take_and_observe!(scratch)
    failures = getfield(scratch, :target_failures)
    failure = ImportanceSamplers._take_first_native_target_failure!(failures)
    marker = getfield(failures, :first_index)[]
    return isnothing(failure) ? marker : -1
end

function benchmark_output(io)
    println(io, "# Native CPU failure marker benchmark")
    println(io)
    println(io, "- Julia: `", VERSION, "`")
    println(io, "- CPU: `", Sys.CPU_NAME, "`")
    println(io, "- Julia threads: `", Threads.nthreads(:default), "`")
    println(io, "- seed: `", BENCHMARK_SEED, "`")
    println(io, "- BenchmarkTools complete samples: `", TRIAL_SAMPLES, "`")
    println(io, "- BenchmarkTools reset samples: `", RESET_TRIAL_SAMPLES, "`")
    println(io)
    println(io, "## Clean reset")
    println(io)
    println(io, "| prepared samples | reset | clean take |")
    println(io, "| ---: | --- | --- |")
    for nsamples in RESET_SAMPLE_COUNTS
        sampler = prepared_sampler(nsamples, false, BENCHMARK_SEED + nsamples)
        scratch = failure_scratch(sampler)
        reset_and_observe!(scratch)
        reset_trial = run_reset_trial(@benchmarkable reset_and_observe!($scratch))
        take_trial = run_reset_trial(@benchmarkable take_and_observe!($scratch))
        println(
            io,
            "| ",
            nsamples,
            " | ",
            format_trial(reset_trial),
            " | ",
            format_trial(take_trial),
            " |",
        )
    end
    println(io)
    println(io, "## Complete warm execution")
    println(io)
    println(io, "| mode | prepared samples | median time and allocation |")
    println(io, "| --- | ---: | --- |")
    for threaded in (false, true)
        sampler = prepared_sampler(
            COMPLETE_SAMPLE_COUNT,
            threaded,
            BENCHMARK_SEED + threaded,
        )
        warm = importance_sample!(sampler)
        trial = run_trial(@benchmarkable importance_sample!($sampler))
        mode = warm.diagnostics.execution
        println(
            io,
            "| ",
            mode,
            " | ",
            COMPLETE_SAMPLE_COUNT,
            " | ",
            format_trial(trial),
            " |",
        )
    end
end

benchmark_output(stdout)
