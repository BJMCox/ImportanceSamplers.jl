using CUDA
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
const SUSTAINED_WARM_EXECUTIONS = parse(
    Int,
    get(
        ENV,
        "IMPORTANCESAMPLERS_CUDA_SUSTAINED_WARM_EXECUTIONS",
        SMOKE_MODE ? "2" : "64",
    ),
)
SUSTAINED_WARM_EXECUTIONS > 0 ||
    error("IMPORTANCESAMPLERS_CUDA_SUSTAINED_WARM_EXECUTIONS must be positive")
const BENCHMARK_COMMAND = if SMOKE_MODE
    "Kaimon ex: empty!(ARGS); push!(ARGS, \"--smoke\"); " *
    "include(\"cuda_plain_is.jl\") in the benchmark project"
else
    "Kaimon ex: empty!(ARGS); include(\"cuda_plain_is.jl\") in the benchmark project"
end

function synthetic_proposal_generation!(samples, normals, location, scale)
    @. samples = muladd(scale, normals, location)
    return nothing
end

function synthetic_gaussian_logdensity!(output, samples, location, scale)
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

function synchronized_sustained_warm_execution(sampler, executions)
    result = importance_sample!(sampler)
    for _ in 2:executions
        result = importance_sample!(sampler)
    end
    CUDA.synchronize()
    return result
end

function synchronized_complete_result_transfer(result)
    host_result = MLDataDevices.cpu_device()(result)
    CUDA.synchronize()
    return host_result
end

function assert_independent_storage(source::AbstractArray, destination::AbstractArray)
    @assert source !== destination
    return nothing
end

function assert_independent_storage(source::NamedTuple, destination::NamedTuple)
    @assert keys(source) == keys(destination)
    for (source_value, destination_value) in
        zip(values(source), values(destination))
        assert_independent_storage(source_value, destination_value)
    end
    return nothing
end

assert_independent_storage(source, destination) = nothing

array_payload_bytes(array::AbstractArray) = sizeof(array)
array_payload_bytes(tuple::NamedTuple) =
    sum(array_payload_bytes, values(tuple); init=0)
array_payload_bytes(value) = 0

function validate_complete_host_result(device_result, host_result)
    required_fields = (:samples, :logweights, :diagnostics, :provenance)
    @assert host_result isa WeightedSamples
    @assert all(field -> hasproperty(host_result, field), required_fields)
    @assert host_result.samples isa Array
    @assert host_result.logweights isa Array
    assert_independent_storage(device_result.samples, host_result.samples)
    assert_independent_storage(device_result.logweights, host_result.logweights)
    assert_independent_storage(device_result.diagnostics, host_result.diagnostics)
    assert_independent_storage(device_result.provenance, host_result.provenance)
    @assert device_result.diagnostics.transfers !== host_result.diagnostics.transfers

    sample_array_bytes = array_payload_bytes(host_result.samples)
    complete_result_array_bytes =
        sample_array_bytes + array_payload_bytes(host_result.logweights) +
        array_payload_bytes(host_result.diagnostics) +
        array_payload_bytes(host_result.provenance)
    @assert complete_result_array_bytes != sample_array_bytes
    return (; sample_array_bytes, complete_result_array_bytes)
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

function throughput_record(trial, samples_per_execution, executions)
    record = trial_record(trial)
    total_samples = samples_per_execution * executions
    return merge(
        record,
        (
            executions_per_evaluation=executions,
            samples_per_evaluation=total_samples,
            samples_per_second=total_samples / record.median_seconds,
        ),
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
    cold_benchmark = @benchmarkable synchronized(importance_sample!, $cold_sampler)
    cold = run(cold_benchmark; samples=1, evals=1, warmup=false)

    sampler = prepare_sampler(
        Xoshiro(BENCHMARK_SEED),
        case.target,
        case.context,
        algorithm;
        threaded=true,
    ) |> device
    warm_result = synchronized(importance_sample!, sampler)
    host_result = synchronized_complete_result_transfer(warm_result)
    transfer_payload = validate_complete_host_result(warm_result, host_result)

    rng = getfield(sampler, :rng)
    normals = getfield(getfield(sampler, :random_buffers), :normal)
    samples = similar(normals, T, SAMPLE_COUNT)
    target_log = similar(normals, T, SAMPLE_COUNT)
    proposal_log = similar(normals, T, SAMPLE_COUNT)
    CUDA.synchronize()

    warm_benchmark = @benchmarkable synchronized(importance_sample!, $sampler)
    sustained_warm_benchmark = @benchmarkable synchronized_sustained_warm_execution(
        $sampler, $SUSTAINED_WARM_EXECUTIONS,
    ) setup=(CUDA.synchronize()) gcsample=true
    rng_benchmark = @benchmarkable synchronized(Random.randn!, $rng, $normals)
    synthetic_proposal_benchmark = @benchmarkable synchronized(
        synthetic_proposal_generation!, $samples, $normals, $location, $scale,
    )
    synthetic_target_benchmark = @benchmarkable synchronized(
        synthetic_gaussian_logdensity!, $target_log, $samples, $location, $scale,
    )
    synthetic_density_benchmark = @benchmarkable synchronized(
        synthetic_gaussian_logdensity!, $proposal_log, $samples, $location, $scale,
    )
    public_reduction_benchmark = @benchmarkable synchronized(lognormalizer, $warm_result)
    complete_result_transfer_benchmark =
        @benchmarkable synchronized_complete_result_transfer($warm_result)
    synthetic_sample_array_transfer_benchmark =
        @benchmarkable synchronized(Array, $samples)
    trials = (
        warm_execution=run_trial(warm_benchmark),
        sustained_warm_throughput=run_trial(sustained_warm_benchmark),
        rng_fill=run_trial(rng_benchmark),
        synthetic_proposal_generation=run_trial(synthetic_proposal_benchmark),
        synthetic_target_evaluation=run_trial(synthetic_target_benchmark),
        synthetic_density_evaluation=run_trial(synthetic_density_benchmark),
        public_log_normalizer=run_trial(public_reduction_benchmark),
        complete_result_transfer=run_trial(complete_result_transfer_benchmark),
        synthetic_sample_array_transfer=
            run_trial(synthetic_sample_array_transfer_benchmark),
    )
    @assert trial_record(trials.complete_result_transfer).host_allocated_bytes !=
        trial_record(trials.synthetic_sample_array_transfer).host_allocated_bytes

    CUDA.reclaim()
    CUDA.synchronize()
    return (
        scalar_type=T,
        first_measured_execution=trial_record(cold),
        measurements=merge(
            map(trial_record, trials),
            (
                sustained_warm_throughput=throughput_record(
                    trials.sustained_warm_throughput,
                    SAMPLE_COUNT,
                    SUSTAINED_WARM_EXECUTIONS,
                ),
            ),
        ),
        additive_decomposition=false,
        complete_result_transfer_payload=transfer_payload,
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
            )),
            seed=BENCHMARK_SEED,
            sample_count=SAMPLE_COUNT,
            sustained_warm_executions=SUSTAINED_WARM_EXECUTIONS,
            scalar_types=(Float32, Float64),
            smoke=SMOKE_MODE,
            command=BENCHMARK_COMMAND,
        ),
        device_memory=(before=before, after=after),
        results=results,
    )
end

main()
