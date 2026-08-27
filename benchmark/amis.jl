using BenchmarkTools
using CUDA
using ImportanceSamplers
using LinearAlgebra
using MLDataDevices
using Pkg
using Random

include(joinpath(@__DIR__, "..", "validation", "amis_capabilities.jl"))

const AMIS_BENCHMARK_SEED = 0x616d697362656e63
const AMIS_BENCHMARK_REPLICATES = 5
const AMIS_BENCHMARK_SMOKE = "--smoke" in ARGS
const AMIS_BENCHMARK_LONG = "--long" in ARGS
const AMIS_BENCHMARK_CPU = !("--cuda-only" in ARGS)
const AMIS_BENCHMARK_CUDA = !("--cpu-only" in ARGS)
const AMIS_BENCHMARK_GUARDS_ONLY = "--guards-only" in ARGS
const AMIS_BENCHMARK_FACTOR_ONLY = "--factor-only" in ARGS
AMIS_BENCHMARK_SMOKE && AMIS_BENCHMARK_LONG && error(
    "--smoke and --long are mutually exclusive",
)
const AMIS_BENCHMARK_ROUNDS = AMIS_BENCHMARK_SMOKE ? 2 : 5
const AMIS_BENCHMARK_ROUND_SIZE = AMIS_BENCHMARK_SMOKE ? 256 :
                                  AMIS_BENCHMARK_LONG ? 262_144 : 4_096
const AMIS_GUARD_SAMPLE_COUNT = AMIS_BENCHMARK_SMOKE ? 32_768 : 65_536

struct AMISBenchmarkTarget{T<:AbstractFloat} end

function (::AMISBenchmarkTarget{T})(sample)::T where {T}
    if sample isa Number
        radius = abs2(sample - T(0.25))
        dimension = 1
    else
        radius = zero(T)
        @inbounds for coordinate in eachindex(sample)
            radius += abs2(sample[coordinate] - T(0.25))
        end
        dimension = length(sample)
    end
    return -T(0.5) * radius - T(0.5) * T(dimension) * log(T(2pi))
end

function amis_benchmark_proposal(::Type{T}, geometry) where {T}
    geometry === :scalar && return SphericalGaussian(T(-0.75), T(1.5))
    geometry === :factor || error("unknown AMIS benchmark geometry $geometry")
    return FactorGaussian(
        T[-0.75, 0.5, 1.25, -0.25],
        T[1.3 0 0 0; -0.15 1.1 0 0; 0.1 0.2 1.25 0; 0.05 -0.1 0.15 1.2],
    )
end

function amis_cuda_device()
    CUDA.functional() || error("CUDA is not functional")
    CUDA.allowscalar(false)
    physical = CUDA.device()
    return MLDataDevices.CUDADevice{typeof(physical),Nothing}(physical)
end

function amis_prepare(device_kind, ::Type{T}, geometry, seed) where {T}
    algorithm = AMIS(
        amis_benchmark_proposal(T, geometry);
        rounds=AMIS_BENCHMARK_ROUNDS,
        round_size=AMIS_BENCHMARK_ROUND_SIZE,
    )
    sampler = prepare_sampler(
        Xoshiro(seed),
        AMISBenchmarkTarget{T}(),
        algorithm;
        threaded=device_kind === :cuda,
    )
    return device_kind === :cpu ? sampler : amis_cuda_device()(sampler)
end

function synchronized_run!(sampler, device_kind)
    result = importance_sample!(sampler)
    device_kind === :cuda && CUDA.synchronize()
    return result
end

function array_storage_bytes(value, seen=IdDict{Any,Nothing}())
    value isa AbstractArray || return zero(Int)
    haskey(seen, value) && return zero(Int)
    seen[value] = nothing
    return sizeof(value)
end

function owned_storage_bytes(value, seen=IdDict{Any,Nothing}())
    value isa AbstractArray && return array_storage_bytes(value, seen)
    value isa Union{Number,Symbol,String,Type,Function,Module,Nothing,Missing} &&
        return zero(Int)
    haskey(seen, value) && return zero(Int)
    seen[value] = nothing
    if value isa NamedTuple || value isa Tuple
        return sum(item -> owned_storage_bytes(item, seen), value; init=0)
    end
    isstructtype(typeof(value)) || return zero(Int)
    return sum(
        field -> owned_storage_bytes(getfield(value, field), seen),
        fieldnames(typeof(value));
        init=0,
    )
end

function prepared_storage_record(sampler)
    state = getproperty(sampler, :method_state)
    owned = owned_storage_bytes((
        getproperty(sampler, :algorithm),
        state,
        getproperty(sampler, :random_buffers),
    ))
    workspace = owned_storage_bytes(getproperty(state, :workspace))
    return (; owned_bytes=owned, workspace_bytes=workspace)
end

function one_evaluation(sampler, device_kind)
    result_holder = Ref{Any}()
    evaluations = Ref(0)
    benchmark = @benchmarkable $result_holder[] = synchronized_run!(
        $sampler,
        $device_kind,
    ) setup=($evaluations[] += 1)
    trial = run(benchmark; samples=1, evals=1, warmup=false)
    evaluations[] == 1 || error(
        "BenchmarkTools executed $(evaluations[]) evaluations; expected exactly one",
    )
    estimate = minimum(trial)
    return (
        result=result_holder[],
        seconds=estimate.time / 1.0e9,
        host_allocations=estimate.allocs,
        host_allocated_bytes=estimate.memory,
        benchmarktools_evaluations=evaluations[],
    )
end

function host_logweights(result, device_kind)
    device_kind === :cpu && return copy(result.logweights), (count=0, bytes=0)
    values = Array(result.logweights)
    CUDA.synchronize()
    return values, (count=1, bytes=sizeof(values))
end

function weight_concentration_ess(logweights)
    maximum_logweight = maximum(logweights)
    scaled = exp.(logweights .- maximum_logweight)
    return abs2(sum(scaled)) / sum(abs2, scaled)
end

function scalar_summary(values)
    ordered = sort(collect(values))
    return (minimum=first(ordered), median=ordered[3], maximum=last(ordered))
end

function benchmark_record(sampler, device_kind)
    storage = prepared_storage_record(sampler)
    evaluation = one_evaluation(sampler, device_kind)
    result = evaluation.result
    logweights, measurement_transfers = host_logweights(result, device_kind)
    ess = weight_concentration_ess(logweights)
    seconds = evaluation.seconds
    return (;
        seconds,
        samples_per_second=length(result) / seconds,
        ess,
        ess_per_second=ess / seconds,
        target_evaluations=result.diagnostics.target_evaluations,
        target_evaluations_per_second=result.diagnostics.target_evaluations / seconds,
        proposal_evaluations=result.diagnostics.proposal_evaluations,
        proposal_evaluations_per_second=result.diagnostics.proposal_evaluations / seconds,
        host_allocations=evaluation.host_allocations,
        host_allocated_bytes=evaluation.host_allocated_bytes,
        owned_bytes=storage.owned_bytes,
        workspace_bytes=storage.workspace_bytes,
        explicit_transfer_count=result.diagnostics.transfers.count,
        explicit_transfer_bytes=result.diagnostics.transfers.bytes,
        measurement_transfers,
        benchmarktools_evaluations=evaluation.benchmarktools_evaluations,
    )
end

function summarize_records(records)
    fields = (
        :seconds,
        :samples_per_second,
        :ess,
        :ess_per_second,
        :target_evaluations_per_second,
        :proposal_evaluations_per_second,
        :host_allocations,
        :host_allocated_bytes,
        :owned_bytes,
        :workspace_bytes,
        :explicit_transfer_bytes,
    )
    summaries = map(fields) do field
        scalar_summary(getproperty(record, field) for record in records)
    end
    return NamedTuple{fields}(summaries)
end

function benchmark_amis_row(device_kind, ::Type{T}, geometry) where {T}
    warmup = amis_prepare(device_kind, T, geometry, AMIS_BENCHMARK_SEED)
    synchronized_run!(warmup, device_kind)
    records = ntuple(AMIS_BENCHMARK_REPLICATES) do replicate
        sampler = amis_prepare(
            device_kind,
            T,
            geometry,
            AMIS_BENCHMARK_SEED + UInt(replicate),
        )
        benchmark_record(sampler, device_kind)
    end
    all(record -> record.benchmarktools_evaluations == 1, records) || error(
        "an AMIS benchmark record used more than one evaluation",
    )
    return (;
        device=device_kind,
        scalar_type=T,
        geometry,
        rounds=AMIS_BENCHMARK_ROUNDS,
        round_size=AMIS_BENCHMARK_ROUND_SIZE,
        prepared_samplers=AMIS_BENCHMARK_REPLICATES,
        records,
        summary=summarize_records(records),
    )
end

function guard_factor(::Type{T}, dimension, slot) where {T}
    factor = zeros(T, dimension, dimension)
    for row in 1:dimension
        factor[row, row] = T(1 + 0.02row + 0.01slot)
        row > 1 && (factor[row, row - 1] = T(0.03slot))
    end
    return factor
end

function guard_bank(::Type{T}) where {T}
    dimension = 4
    proposals = [
        FactorGaussian(
            fill(T(0.15 * (slot - 2.5)), dimension),
            guard_factor(T, dimension, slot),
        ) for slot in 1:4
    ]
    return ProposalBank(proposals, T[1, 2, 3, 4])
end

function guard_prepare(method, device_kind, seed)
    T = Float64
    bank = guard_bank(T)
    algorithm = method === :static_mis ?
                ImportanceSampling(
                    bank;
                    nsamples=AMIS_GUARD_SAMPLE_COUNT,
                    mis_scheme=StratifiedMixture(),
                ) :
                DeterministicMixturePMC(
                    bank;
                    rounds=2,
                    round_size=AMIS_GUARD_SAMPLE_COUNT,
                )
    sampler = prepare_sampler(
        Xoshiro(seed),
        AMISBenchmarkTarget{T}(),
        algorithm;
        threaded=true,
    )
    return device_kind === :cpu ? sampler : amis_cuda_device()(sampler)
end

function benchmark_guard(method, device_kind)
    warmup = guard_prepare(method, device_kind, AMIS_BENCHMARK_SEED)
    synchronized_run!(warmup, device_kind)
    records = ntuple(AMIS_BENCHMARK_REPLICATES) do replicate
        sampler = guard_prepare(
            method,
            device_kind,
            AMIS_BENCHMARK_SEED + UInt(replicate),
        )
        evaluation = one_evaluation(sampler, device_kind)
        result = evaluation.result
        (;
            seconds=evaluation.seconds,
            samples_per_second=length(result) / evaluation.seconds,
            host_allocations=evaluation.host_allocations,
            host_allocated_bytes=evaluation.host_allocated_bytes,
            benchmarktools_evaluations=evaluation.benchmarktools_evaluations,
        )
    end
    return (;
        method,
        device=device_kind,
        workload=AMIS_GUARD_SAMPLE_COUNT,
        prepared_samplers=AMIS_BENCHMARK_REPLICATES,
        records,
        samples_per_second=scalar_summary(record.samples_per_second for record in records),
        host_allocations=scalar_summary(record.host_allocations for record in records),
        host_allocated_bytes=scalar_summary(record.host_allocated_bytes for record in records),
    )
end

function compare_guards(base_guards, candidate_guards; threshold=0.05)
    length(base_guards) == length(candidate_guards) || error(
        "base and candidate guard matrices differ",
    )
    rows = map(base_guards, candidate_guards) do base, candidate
        (base.method, base.device) == (candidate.method, candidate.device) || error(
            "base and candidate guard rows are not aligned",
        )
        base_throughput = [record.samples_per_second for record in base.records]
        candidate_throughput = [record.samples_per_second for record in candidate.records]
        slow_replicates = count(
            candidate_throughput .< (1 - threshold) .* base_throughput,
        )
        throughput_ratio = candidate.samples_per_second.median /
                           base.samples_per_second.median
        repeatable_regression = slow_replicates >= 4 && throughput_ratio < 1 - threshold
        new_execution_allocation =
            candidate.host_allocations.median > base.host_allocations.median
        repeatable_regression && error(
            "$(candidate.method) throughput regressed repeatably by more than 5%",
        )
        new_execution_allocation && error(
            "$(candidate.method) introduced a new execution allocation",
        )
        return (;
            method=candidate.method,
            device=candidate.device,
            throughput_ratio,
            slow_replicates,
            repeatable_regression,
            new_execution_allocation,
        )
    end
    return Tuple(rows)
end

function benchmark_environment()
    root = normpath(joinpath(dirname(pathof(ImportanceSamplers)), ".."))
    commit = get(ENV, "IMPORTANCE_SAMPLERS_BENCHMARK_COMMIT", nothing)
    isnothing(commit) && (commit = readchomp(`git -C $root rev-parse HEAD`))
    wanted = Set(("BenchmarkTools", "CUDA", "ImportanceSamplers", "MLDataDevices"))
    packages = sort!(
        [
            (dependency.name, something(dependency.version, "unversioned")) for
            dependency in values(Pkg.dependencies()) if dependency.name in wanted
        ];
        by=first,
    )
    cuda = AMIS_BENCHMARK_CUDA && CUDA.functional() ? (
        gpu=CUDA.name(CUDA.device()),
        capability=CUDA.capability(CUDA.device()),
        driver=CUDA.driver_version(),
        runtime=CUDA.runtime_version(),
        allowscalar=false,
    ) : nothing
    return (;
        commit,
        julia=VERSION,
        cpu=Sys.CPU_NAME,
        cpu_threads=Sys.CPU_THREADS,
        julia_threads=Threads.nthreads(:default),
        cuda,
        packages,
    )
end

function main()
    devices = Symbol[]
    AMIS_BENCHMARK_CPU && push!(devices, :cpu)
    AMIS_BENCHMARK_CUDA && CUDA.functional() && push!(devices, :cuda)
    rows = AMIS_BENCHMARK_GUARDS_ONLY ? () : Tuple(
        benchmark_amis_row(device, T, geometry) for
        (device, T, geometry) in AMIS_BENCHMARK_ROWS if
        device in devices && (!AMIS_BENCHMARK_FACTOR_ONLY || geometry === :factor)
    )
    guards = Tuple(
        benchmark_guard(method, device) for device in devices for
        method in (:static_mis, :dm_pmc)
    )
    return (;
        command="include(\"amis.jl\") in the benchmark project",
        smoke=AMIS_BENCHMARK_SMOKE,
        long=AMIS_BENCHMARK_LONG,
        factor_only=AMIS_BENCHMARK_FACTOR_ONLY,
        warmup=false,
        benchmarktools_evaluations_per_sampler=1,
        environment=benchmark_environment(),
        rows,
        guards,
        unavailable_cuda=AMIS_BENCHMARK_CUDA && !CUDA.functional(),
    )
end

main()
