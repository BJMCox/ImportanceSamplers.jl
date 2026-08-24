using CUDA
using cuDNN
using BenchmarkTools
using ImportanceSamplers
using MLDataDevices
using Pkg
using Random

include(joinpath(@__DIR__, "..", "validation", "cuda_plain_is_support.jl"))

const SMOKE_MODE = "--smoke" in ARGS
const BENCHMARK_SEED = 0x6e6174697665626d
const SAMPLE_COUNT = SMOKE_MODE ? 4_096 : 1_048_576
const TRIAL_SAMPLES = SMOKE_MODE ? 3 : 20
const TRIAL_SECONDS = SMOKE_MODE ? 0.2 : 5.0
const BENCHMARK_COMMAND = if SMOKE_MODE
    "Kaimon ex: empty!(ARGS); push!(ARGS, \"--smoke\"); " *
    "include(\"cuda_plain_is.jl\") in the benchmark project"
else
    "Kaimon ex: empty!(ARGS); include(\"cuda_plain_is.jl\") in the benchmark project"
end

function proposal_generation!(samples, normals, location, scale)
    @. samples = muladd(scale, normals, location)
    return nothing
end

function gaussian_logdensity!(output, samples, location, scale)
    half = oftype(location, 0.5)
    log2pi = log(oftype(location, 2pi))
    @. output = -half * abs2((samples - location) / scale) - log(scale) - half * log2pi
    return nothing
end

function synchronized(f, args...)
    value = f(args...)
    CUDA.synchronize()
    return value
end

function run_trial(benchmark)
    return run(
        benchmark;
        samples=TRIAL_SAMPLES,
        seconds=TRIAL_SECONDS,
        evals=1,
    )
end

function trial_record(trial)
    estimate = median(trial)
    return (
        median_seconds=estimate.time / 1.0e9,
        host_allocations=estimate.allocs,
        host_allocated_bytes=estimate.memory,
    )
end

function benchmark_type(device, ::Type{T}) where {T}
    case = scalar_gaussian_case(T)
    location, scale = only(case.context.location), only(case.context.scale)
    algorithm = ImportanceSampling(case.proposal; nsamples=SAMPLE_COUNT)

    cold_sampler = prepare_sampler(
        Xoshiro(BENCHMARK_SEED),
        case.target,
        case.context,
        algorithm;
        threaded=true,
    ) |> device
    CUDA.synchronize()
    cold = @timed synchronized(importance_sample!, cold_sampler)

    sampler = prepare_sampler(
        Xoshiro(BENCHMARK_SEED),
        case.target,
        case.context,
        algorithm;
        threaded=true,
    ) |> device
    warm_result = synchronized(importance_sample!, sampler)

    rng = getfield(sampler, :rng)
    normals = getfield(getfield(sampler, :random_buffers), :normal)
    samples = similar(normals, T, SAMPLE_COUNT)
    target_log = similar(normals, T, SAMPLE_COUNT)
    proposal_log = similar(normals, T, SAMPLE_COUNT)
    logweights = similar(normals, T, SAMPLE_COUNT)
    fill!(logweights, zero(T))
    CUDA.synchronize()

    warm_benchmark = @benchmarkable synchronized(importance_sample!, $sampler)
    rng_benchmark = @benchmarkable synchronized(Random.randn!, $rng, $normals)
    proposal_benchmark =
        @benchmarkable synchronized(proposal_generation!, $samples, $normals, $location, $scale)
    target_benchmark =
        @benchmarkable synchronized(gaussian_logdensity!, $target_log, $samples, $location, $scale)
    density_benchmark =
        @benchmarkable synchronized(gaussian_logdensity!, $proposal_log, $samples, $location, $scale)
    reduction_benchmark = @benchmarkable synchronized(sum, $logweights)
    transfer_benchmark = @benchmarkable synchronized(Array, $samples)
    trials = (
        warm_execution=run_trial(warm_benchmark),
        rng_fill=run_trial(rng_benchmark),
        proposal_generation=run_trial(proposal_benchmark),
        target_evaluation=run_trial(target_benchmark),
        density_evaluation=run_trial(density_benchmark),
        reduction=run_trial(reduction_benchmark),
        transfer=run_trial(transfer_benchmark),
    )

    CUDA.reclaim()
    CUDA.synchronize()
    return (
        scalar_type=T,
        compile_and_first_execution=(
            seconds=cold.time,
            host_allocated_bytes=cold.bytes,
        ),
        phases=map(trial_record, trials),
        device_arrays=(
            normal_bytes=sizeof(T) * length(normals),
            sample_bytes=sizeof(T) * length(warm_result.samples),
            logweight_bytes=sizeof(T) * length(warm_result.logweights),
        ),
    )
end

function main()
    device = cuda_device()
    CUDA.synchronize()

    before = CUDA.memory_info()
    results = map(T -> benchmark_type(device, T), (Float32, Float64))
    after = CUDA.memory_info()
    gpu = CUDA.device()
    return (
        environment=(
            gpu=CUDA.name(gpu),
            capability=CUDA.capability(gpu),
            driver=CUDA.driver_version(),
            runtime=CUDA.runtime_version(),
            total_device_memory=CUDA.total_memory(),
            julia=VERSION,
            packages=cuda_package_versions((
                "Adapt",
                "BenchmarkTools",
                "CUDA",
                "ImportanceSamplers",
                "KernelAbstractions",
                "MLDataDevices",
                "cuDNN",
            )),
            seed=BENCHMARK_SEED,
            sample_count=SAMPLE_COUNT,
            scalar_types=(Float32, Float64),
            smoke=SMOKE_MODE,
            command=BENCHMARK_COMMAND,
        ),
        device_memory=(before=before, after=after),
        results=results,
    )
end

main()
