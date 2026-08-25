using BenchmarkTools
using CUDA
using ImportanceSamplers
using MLDataDevices
using Pkg
using Random

include(joinpath(@__DIR__, "..", "cuda_plain_is_support.jl"))
include(joinpath(@__DIR__, "..", "static_mis_capabilities.jl"))

const STATIC_MIS_SEED = 0x7374617469636d69
const CORRECTNESS_SAMPLES = 10_003
const RUN_BENCHMARKS = !("--correctness-only" in ARGS)
const RUN_CORRECTNESS = !("--benchmark-only" in ARGS)
const BENCHMARK_SAMPLES = parse(Int, get(ENV, "STATIC_MIS_BENCHMARK_SAMPLES", "3"))
const BENCHMARK_SECONDS = parse(Float64, get(ENV, "STATIC_MIS_BENCHMARK_SECONDS", "0.25"))
const BENCHMARK_OUTPUT = joinpath(@__DIR__, "cuda_static_mis_results.tsv")

@inline function static_mis_logaddexp(left, right)
    left == -Inf && return right
    right == -Inf && return left
    largest = max(left, right)
    return largest + log1p(exp(min(left, right) - largest))
end

@inline function static_mis_gaussian_logdensity(sample, context, proposal)
    lognormalizer = @inbounds context.lognormalizers[proposal]
    value = lognormalizer
    @inbounds for coordinate in eachindex(sample)
        value -= typeof(value)(0.5) * abs2(
            (sample[coordinate] - context.locations[coordinate, proposal]) /
            context.scales[coordinate, proposal],
        )
    end
    return value
end

@inline function static_mis_mixture_target(sample, context)
    value = oftype(context.logmasses[1], -Inf)
    @inbounds for proposal in axes(context.locations, 2)
        term = context.logmasses[proposal] +
               static_mis_gaussian_logdensity(sample, context, proposal)
        value = static_mis_logaddexp(value, term)
    end
    return value
end

@inline function static_mis_scalar_target(sample, context)
    value = oftype(context.logmasses[1], -Inf)
    @inbounds for proposal in eachindex(context.locations)
        standardized =
            (sample - context.locations[proposal]) / context.scales[proposal]
        term = context.logmasses[proposal] + context.lognormalizers[proposal] -
               typeof(value)(0.5) * abs2(standardized)
        value = static_mis_logaddexp(value, term)
    end
    return value
end

function static_mis_case(::Type{T}, proposal_count, dimension) where {T}
    locations = Matrix{T}(undef, dimension, proposal_count)
    scales = Matrix{T}(undef, dimension, proposal_count)
    proposals = Vector{typeof(SphericalGaussian(zeros(T, dimension), one(T)))}(undef, proposal_count)
    for proposal in 1:proposal_count
        location = T(0.2) .* T.(1:dimension) .+
                   T(0.35) * T(proposal - (proposal_count + 1) / 2)
        scale = T(0.65) + T(0.05) * T(mod(proposal, 4))
        proposals[proposal] = SphericalGaussian(location, scale)
        locations[:, proposal] = location
        scales[:, proposal] .= scale
    end
    masses = T.(mod.(1:proposal_count, 5) .+ 1)
    proposal_count > 2 && (masses[end] = zero(T))
    masses ./= sum(masses)
    active = findall(!iszero, masses)
    active_locations = locations[:, active]
    active_scales = scales[:, active]
    active_masses = masses[active]
    lognormalizers = T[
        -T(0.5) * T(dimension) * log(T(2) * T(pi)) -
        sum(log, view(active_scales, :, proposal)) for
        proposal in axes(active_scales, 2)
    ]
    context = (
        locations=active_locations,
        scales=active_scales,
        lognormalizers=lognormalizers,
        logmasses=log.(active_masses),
    )
    expected_mean = locations * masses
    return ProposalBank(proposals, masses), context, expected_mean
end

function static_mis_direct_case(row, ::Type{T}) where {T}
    bank = row.factory(T)
    direct = row.direct
    isnothing(direct) && error("static-MIS capability row has no direct case")
    layout = direct.sample_layout
    masses = bank.masses
    active = findall(!iszero, masses)
    locations = if layout === :scalar
        T[getfield(proposal, :location) for proposal in bank.proposals]
    elseif layout === :vector
        reduce(hcat, (getfield(proposal, :location) for proposal in bank.proposals))
    else
        error("unknown directly tested sample layout: $layout")
    end
    proposal_scales = T[
        getfield(getfield(proposal, :scale), :scale) for proposal in bank.proposals
    ]
    if layout === :scalar
        active_scales = proposal_scales[active]
        context = (
            locations=locations[active],
            scales=active_scales,
            lognormalizers=-log.(active_scales) .-
                           T(0.5) * log(T(2) * T(pi)),
            logmasses=log.(masses[active]),
        )
        expected_mean = T[sum(locations .* masses)]
    else
        scales = repeat(reshape(proposal_scales, 1, :), size(locations, 1), 1)
        active_scales = scales[:, active]
        context = (
            locations=locations[:, active],
            scales=active_scales,
            lognormalizers=T[
                -T(0.5) * T(size(locations, 1)) * log(T(2) * T(pi)) -
                sum(log, view(active_scales, :, proposal)) for
                proposal in axes(active_scales, 2)
            ],
            logmasses=log.(masses[active]),
        )
        expected_mean = locations * masses
    end
    return bank, context, expected_mean
end

function static_mis_scheme(label, proposal_count)
    label === :stratified_mixture && return StratifiedMixture()
    label === :random_mixture && return RandomMixture()
    label === :standard_mis && return StandardMIS()
    label === :partial_deterministic_mixture && return PartialDeterministicMixture(
        (Tuple(1:2:proposal_count), Tuple(2:2:proposal_count)),
    )
    error("unknown static-MIS scheme metadata label: $label")
end

static_mis_schemes(proposal_count) = Tuple(
    static_mis_scheme(scheme.label, proposal_count) for
    scheme in STATIC_MIS_COMPLETE_SCHEMES
)

scheme_name(::StratifiedMixture) = :stratified_mixture
scheme_name(::RandomMixture) = :random_mixture
scheme_name(::StandardMIS) = :standard_mis
scheme_name(::PartialDeterministicMixture) = :partial_deterministic_mixture

function host_summary(result)
    weights = normalized_weights(result)
    samples = result.samples
    mean, variance = if samples isa AbstractVector
        scalar_mean = sum(samples .* weights)
        [scalar_mean], [sum(abs2.(samples .- scalar_mean) .* weights)]
    else
        vector_mean = vec(samples * weights)
        centered = samples .- vector_mean
        vector_variance = vec(
            sum(abs2.(centered) .* reshape(weights, 1, :); dims=2),
        )
        vector_mean, vector_variance
    end
    ess = inv(sum(abs2, weights))
    mean_se = sqrt.(variance ./ ess)
    linear_weights = exp.(result.logweights)
    normalizer = sum(linear_weights) / length(linear_weights)
    normalizer_variance =
        sum(weight -> abs2(weight - normalizer), linear_weights) /
        length(linear_weights)
    normalizer_se = sqrt(normalizer_variance / length(linear_weights))
    lognormalizer_se = normalizer_se / max(normalizer, floatmin(eltype(linear_weights)))
    return (; mean, mean_se, lognormalizer=lognormalizer(result), lognormalizer_se, ess)
end

function assert_resident(prepared, result)
    method_state = getfield(prepared, :method_state)
    bank = getfield(method_state, :bank)
    design = getfield(method_state, :design)
    buffers = getfield(prepared, :random_buffers)
    arrays = Any[
        bank.locations,
        bank.scales,
        bank.lognormalizers,
        bank.logmasses,
        bank.cdf,
        bank.proposal_ids,
        buffers.uniform,
        buffers.normal,
        buffers.assignments,
        buffers.failure_scratch.record.storage,
        result.samples,
        result.logweights,
        result.provenance.proposal_id,
    ]
    denominator = design.denominator
    if denominator isa ImportanceSamplers._PartialMixtureDenominator
        append!(
            arrays,
            (
                denominator.group_of_slot,
                denominator.offsets,
                denominator.members,
                denominator.logcoefficients,
            ),
        )
    end
    @assert all(array -> array isa CUDA.AnyCuArray, arrays)
    return nothing
end

function assert_assignment_counts(ids, masses, scheme)
    active = findall(!iszero, masses)
    counts = [count(==(proposal), ids) for proposal in eachindex(masses)]
    @assert all(iszero(counts[proposal]) for proposal in findall(iszero, masses))
    if scheme isa RandomMixture
        for proposal in active
            expected = length(ids) * masses[proposal]
            tolerance = 7sqrt(length(ids) * masses[proposal] * (1 - masses[proposal])) + 2
            @assert abs(counts[proposal] - expected) <= tolerance
        end
    else
        @assert all(abs(counts[proposal] - length(ids) * masses[proposal]) <= 1 for proposal in active)
    end
    return counts
end

function correctness_case(
    device,
    ::Type{T},
    scheme,
    case_index;
    row,
) where {T}
    bank, context, expected_mean = static_mis_direct_case(row, T)
    scalar = row.direct.sample_layout === :scalar
    target = scalar ? static_mis_scalar_target : static_mis_mixture_target
    algorithm = ImportanceSampling(
        bank;
        nsamples=CORRECTNESS_SAMPLES,
        mis_scheme=scheme,
    )
    cpu = prepare_sampler(
        Xoshiro(STATIC_MIS_SEED + UInt(case_index)),
        target,
        context,
        algorithm;
        threaded=false,
    )
    gpu_source = prepare_sampler(
        Xoshiro(STATIC_MIS_SEED + UInt(case_index)),
        target,
        context,
        algorithm;
        threaded=true,
    )
    gpu = gpu_source |> device
    first_result = importance_sample!(gpu)
    second_result = importance_sample!(gpu)
    CUDA.synchronize()
    assert_resident(gpu, first_result)
    assert_resident(gpu, second_result)
    @assert first_result.samples !== second_result.samples
    @assert first_result.logweights !== second_result.logweights
    @assert first_result.provenance.proposal_id !== second_result.provenance.proposal_id

    device_weights = normalized_weights(first_result)
    device_normalizer = lognormalizer(first_result)
    @assert device_weights isa CUDA.AnyCuArray
    @assert isapprox(sum(Array(device_weights)), one(T); atol=T(256) * eps(T))
    @assert isfinite(device_normalizer)

    host_gpu = MLDataDevices.cpu_device()(first_result)
    @assert host_gpu.samples isa (scalar ? Vector{T} : Matrix{T})
    @assert host_gpu.logweights isa Vector
    @assert host_gpu.provenance.proposal_id isa Vector{Int}
    @assert size(host_gpu.samples) ==
            (scalar ? (CORRECTNESS_SAMPLES,) : (4, CORRECTNESS_SAMPLES))
    counts = assert_assignment_counts(
        host_gpu.provenance.proposal_id,
        bank.masses,
        scheme,
    )
    if scheme isa Union{StratifiedMixture,RandomMixture}
        @assert maximum(abs, host_gpu.logweights) <= T(4096) * eps(T)
    end

    host_cpu = importance_sample!(cpu)
    gpu_summary = host_summary(host_gpu)
    cpu_summary = host_summary(host_cpu)
    for coordinate in eachindex(expected_mean)
        gpu_tolerance = max(T(7) * gpu_summary.mean_se[coordinate], T(512) * eps(T))
        cpu_tolerance = max(T(7) * cpu_summary.mean_se[coordinate], T(512) * eps(T))
        difference_tolerance = max(
            T(7) * hypot(gpu_summary.mean_se[coordinate], cpu_summary.mean_se[coordinate]),
            T(1024) * eps(T),
        )
        @assert abs(gpu_summary.mean[coordinate] - expected_mean[coordinate]) <= gpu_tolerance
        @assert abs(cpu_summary.mean[coordinate] - expected_mean[coordinate]) <= cpu_tolerance
        @assert abs(gpu_summary.mean[coordinate] - cpu_summary.mean[coordinate]) <= difference_tolerance
    end
    gpu_log_tolerance = max(T(7) * gpu_summary.lognormalizer_se, T(4096) * eps(T))
    cpu_log_tolerance = max(T(7) * cpu_summary.lognormalizer_se, T(4096) * eps(T))
    difference_log_tolerance = max(
        T(7) * hypot(gpu_summary.lognormalizer_se, cpu_summary.lognormalizer_se),
        T(8192) * eps(T),
    )
    @assert abs(gpu_summary.lognormalizer) <= gpu_log_tolerance
    @assert abs(cpu_summary.lognormalizer) <= cpu_log_tolerance
    @assert abs(gpu_summary.lognormalizer - cpu_summary.lognormalizer) <= difference_log_tolerance

    transfers = first_result.diagnostics.transfers
    @assert transfers.count >= 3
    @assert transfers.bytes >= 3sizeof(T) + 3sizeof(UInt64)
    return (
        scalar_type=T,
        sample_layout=row.direct.sample_layout,
        scheme=scheme_name(scheme),
        counts,
        gpu=gpu_summary,
        cpu=cpu_summary,
        transfers=(count=transfers.count, bytes=transfers.bytes),
    )
end

function synchronized_sample!(sampler)
    result = importance_sample!(sampler)
    CUDA.synchronize()
    return result
end

function synchronized_copy(result)
    copied = MLDataDevices.cpu_device()(result)
    CUDA.synchronize()
    return copied
end

function benchmark_case(device, ::Type{T}, scheme, proposal_count, dimension, nsamples) where {T}
    bank, context, _ = static_mis_case(T, proposal_count, dimension)
    sampler = prepare_sampler(
        Xoshiro(STATIC_MIS_SEED),
        static_mis_mixture_target,
        context,
        ImportanceSampling(bank; nsamples, mis_scheme=scheme);
        threaded=true,
    ) |> device
    warm_result = synchronized_sample!(sampler)
    synchronized_copy(warm_result)
    execution = @benchmark synchronized_sample!($sampler) samples=BENCHMARK_SAMPLES seconds=BENCHMARK_SECONDS evals=1
    copy_trial = @benchmark synchronized_copy($warm_result) samples=BENCHMARK_SAMPLES seconds=BENCHMARK_SECONDS evals=1
    execution_estimate = BenchmarkTools.median(execution)
    copy_estimate = BenchmarkTools.median(copy_trial)
    record = (
        scalar_type=string(T),
        scheme=string(scheme_name(scheme)),
        proposal_count,
        dimension,
        nsamples,
        execution_ns=execution_estimate.time,
        samples_per_second=nsamples / (execution_estimate.time / 1.0e9),
        copy_ns=copy_estimate.time,
        copy_bytes=sizeof(warm_result.samples) + sizeof(warm_result.logweights) +
                   sizeof(warm_result.provenance.proposal_id),
    )
    warm_result = nothing
    CUDA.reclaim()
    return record
end

function write_benchmarks(records)
    open(BENCHMARK_OUTPUT, "w") do io
        println(
            io,
            "scalar_type\tscheme\tproposal_count\tdimension\tnsamples\texecution_ns\tsamples_per_second\tcopy_ns\tcopy_bytes",
        )
        for record in records
            println(io, join(values(record), '\t'))
        end
    end
    return nothing
end

function environment_record()
    root = normpath(joinpath(@__DIR__, "..", ".."))
    gpu = CUDA.device()
    return (
        commit=readchomp(
            addenv(
                `git -C $root rev-parse HEAD`,
                "GIT_CONFIG_GLOBAL" => "/dev/null",
            ),
        ),
        gpu=CUDA.name(gpu),
        capability=CUDA.capability(gpu),
        driver=CUDA.driver_version(),
        runtime=CUDA.runtime_version(),
        julia=VERSION,
        packages=cuda_package_versions((
            "Adapt",
            "BenchmarkTools",
            "CUDA",
            "ImportanceSamplers",
            "KernelAbstractions",
            "MLDataDevices",
        )),
        seed=STATIC_MIS_SEED,
        correctness_samples=CORRECTNESS_SAMPLES,
        benchmark_samples=BENCHMARK_SAMPLES,
        benchmark_seconds=BENCHMARK_SECONDS,
        allowscalar=false,
    )
end

function main()
    device = cuda_device()
    environment = environment_record()
    direct_cases = [
        (row=row, type=T, scheme=scheme) for
        row in STATIC_MIS_CAPABILITY_ROWS if !isnothing(row.direct) for
        T in row.direct.types for scheme in row.direct.schemes
    ]
    @assert length(direct_cases) == 16
    correctness = RUN_CORRECTNESS ? [
        correctness_case(
            device,
            case.type,
            static_mis_scheme(case.scheme, 4),
            case_index;
            row=case.row,
        ) for (case_index, case) in enumerate(direct_cases)
    ] : NamedTuple[]
    @assert correctness isa Vector
    @assert length(correctness) == (RUN_CORRECTNESS ? 16 : 0)
    direct_types = unique(case.type for case in direct_cases)
    benchmarks = RUN_BENCHMARKS ? [
        benchmark_case(device, T, scheme, proposal_count, dimension, nsamples) for
        T in direct_types for
        proposal_count in (2, 8, 32) for
        scheme in static_mis_schemes(proposal_count) for
        dimension in (1, 4, 16) for
        nsamples in (10_000, 100_000, 1_000_000)
    ] : NamedTuple[]
    RUN_BENCHMARKS && write_benchmarks(benchmarks)
    return (
        environment,
        correctness_cases=length(correctness),
        correctness,
        benchmark_cases=length(benchmarks),
        benchmark_output=RUN_BENCHMARKS ? BENCHMARK_OUTPUT : nothing,
        benchmark_throughput_extrema=isempty(benchmarks) ? nothing : (
            minimum(record.samples_per_second for record in benchmarks),
            maximum(record.samples_per_second for record in benchmarks),
        ),
    )
end

main()
