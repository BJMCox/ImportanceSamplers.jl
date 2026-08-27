using BenchmarkTools
using CUDA
using ImportanceSamplers
using LinearAlgebra
using MLDataDevices
using Pkg
using Random
using Serialization
using Sockets

include(joinpath(@__DIR__, "..", "validation", "amis_capabilities.jl"))

const AMIS_BENCHMARK_SCHEMA_VERSION = 1
const AMIS_BENCHMARK_SEED = 0x616d697362656e63
const AMIS_BENCHMARK_REPLICATES = 5
const AMIS_BENCHMARK_SMOKE = "--smoke" in ARGS
const AMIS_BENCHMARK_LONG = "--long" in ARGS
const AMIS_BENCHMARK_CPU = !("--cuda-only" in ARGS)
const AMIS_BENCHMARK_CUDA = !("--cpu-only" in ARGS)
const AMIS_BENCHMARK_GUARDS_ONLY = "--guards-only" in ARGS
const AMIS_BENCHMARK_FACTOR_ONLY = "--factor-only" in ARGS
const AMIS_BENCHMARK_SCALING = "--scaling" in ARGS
const AMIS_BENCHMARK_THREAD_SCALING = "--thread-scaling" in ARGS
const AMIS_BENCHMARK_COMPARE = "--compare-guards" in ARGS

function benchmark_option(name)
    prefix = "$name="
    matches = [arg[length(prefix) + 1:end] for arg in ARGS if startswith(arg, prefix)]
    length(matches) <= 1 || error("$name may be supplied only once")
    return isempty(matches) ? nothing : only(matches)
end

const AMIS_BENCHMARK_SAVE = benchmark_option("--save")
const AMIS_BENCHMARK_BASE = benchmark_option("--base")
const AMIS_BENCHMARK_CANDIDATE = benchmark_option("--candidate")
const AMIS_BENCHMARK_EXPECTED_BASE = benchmark_option("--expected-base")
const AMIS_BENCHMARK_EXPECTED_CANDIDATE = benchmark_option("--expected-candidate")

AMIS_BENCHMARK_SMOKE && AMIS_BENCHMARK_LONG && error(
    "--smoke and --long are mutually exclusive",
)
count(identity, (
    AMIS_BENCHMARK_GUARDS_ONLY,
    AMIS_BENCHMARK_SCALING,
    AMIS_BENCHMARK_THREAD_SCALING,
    AMIS_BENCHMARK_COMPARE,
)) <= 1 || error(
    "--guards-only, --scaling, --thread-scaling, and --compare-guards are mutually exclusive",
)
AMIS_BENCHMARK_THREAD_SCALING && AMIS_BENCHMARK_CUDA && error(
    "--thread-scaling requires --cpu-only",
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

function amis_benchmark_factor(::Type{T}, dimension) where {T}
    if dimension == 4
        return T[1.3 0 0 0; -0.15 1.1 0 0; 0.1 0.2 1.25 0; 0.05 -0.1 0.15 1.2]
    end
    factor = zeros(T, dimension, dimension)
    for coordinate in 1:dimension
        factor[coordinate, coordinate] = T(1.15 + 0.01coordinate)
        coordinate > 1 && (factor[coordinate, coordinate - 1] = T(0.05))
    end
    return factor
end

function amis_benchmark_proposal(::Type{T}, geometry, dimension) where {T}
    geometry === :scalar && dimension == 1 &&
        return SphericalGaussian(T(-0.75), T(1.5))
    geometry === :factor || error("unknown AMIS benchmark geometry $geometry")
    location = dimension == 4 ? T[-0.75, 0.5, 1.25, -0.25] :
               fill(T(-0.5), dimension)
    return FactorGaussian(location, amis_benchmark_factor(T, dimension))
end

function amis_cuda_device()
    CUDA.functional() || error("CUDA is not functional")
    CUDA.allowscalar(false)
    physical = CUDA.device()
    return MLDataDevices.CUDADevice{typeof(physical),Nothing}(physical)
end

function amis_prepare(device_kind, ::Type{T}, geometry, dimension, schedule, seed) where {T}
    algorithm = AMIS(
        amis_benchmark_proposal(T, geometry, dimension);
        rounds=length(schedule),
        round_size=collect(schedule),
    )
    sampler = prepare_sampler(
        Xoshiro(seed),
        AMISBenchmarkTarget{T}(),
        algorithm;
        threaded=true,
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
    middle = cld(length(ordered), 2)
    return (minimum=first(ordered), median=ordered[middle], maximum=last(ordered))
end

function validate_amis_result(result, device_kind, schedule)
    logweights, measurement_transfers = host_logweights(result, device_kind)
    total_samples = sum(schedule)
    rounds = length(schedule)
    length(result) == total_samples || error("AMIS benchmark sample count is wrong")
    all(isfinite, logweights) || error("AMIS benchmark log weights are nonfinite")
    ess = weight_concentration_ess(logweights)
    isfinite(ess) && ess > 0 || error("AMIS benchmark ESS is invalid")
    result.diagnostics.method === :amis || error("AMIS diagnostics method is wrong")
    result.diagnostics.round_sizes == collect(schedule) || error(
        "AMIS diagnostics schedule is wrong",
    )
    result.diagnostics.target_evaluations == total_samples || error(
        "AMIS target evaluation count is wrong",
    )
    result.diagnostics.proposal_evaluations == rounds * total_samples || error(
        "AMIS proposal evaluation count is wrong",
    )
    result.diagnostics.failures == 0 || error("AMIS diagnostics report a failure")
    expected_execution = Threads.nthreads(:default) > 1 ? :threaded : :serial
    result.diagnostics.execution === expected_execution || error(
        "AMIS benchmark reported $(result.diagnostics.execution) execution; expected $expected_execution",
    )
    length(result.diagnostics.round_ess) == rounds &&
        all(value -> isfinite(value) && value > 0, result.diagnostics.round_ess) || error(
        "AMIS benchmark produced invalid round ESS diagnostics",
    )
    length(result.diagnostics.round_lognormalizers) == rounds &&
        all(isfinite, result.diagnostics.round_lognormalizers) || error(
        "AMIS benchmark produced invalid round log-normalizer diagnostics",
    )
    return (;
        total_samples,
        ess,
        target_evaluations=result.diagnostics.target_evaluations,
        proposal_evaluations=result.diagnostics.proposal_evaluations,
        execution=result.diagnostics.execution,
        measurement_transfers,
    )
end

function benchmark_record(sampler, device_kind, schedule, seed)
    storage = prepared_storage_record(sampler)
    evaluation = one_evaluation(sampler, device_kind)
    evaluation.benchmarktools_evaluations == 1 || error("AMIS row used multiple evaluations")
    validation = validate_amis_result(evaluation.result, device_kind, schedule)
    seconds = evaluation.seconds
    return (;
        seed,
        schedule=Tuple(schedule),
        validation.total_samples,
        seconds,
        samples_per_second=validation.total_samples / seconds,
        validation.ess,
        ess_per_second=validation.ess / seconds,
        validation.target_evaluations,
        target_evaluations_per_second=validation.target_evaluations / seconds,
        validation.proposal_evaluations,
        proposal_evaluations_per_second=validation.proposal_evaluations / seconds,
        validation.execution,
        host_allocations=evaluation.host_allocations,
        host_allocated_bytes=evaluation.host_allocated_bytes,
        owned_bytes=storage.owned_bytes,
        workspace_bytes=storage.workspace_bytes,
        explicit_transfer_count=evaluation.result.diagnostics.transfers.count,
        explicit_transfer_bytes=evaluation.result.diagnostics.transfers.bytes,
        validation.measurement_transfers,
        benchmarktools_evaluations=evaluation.benchmarktools_evaluations,
    )
end

function summarize_records(records)
    fields = (
        :seconds, :samples_per_second, :ess, :ess_per_second,
        :target_evaluations_per_second, :proposal_evaluations_per_second,
        :host_allocations, :host_allocated_bytes, :owned_bytes, :workspace_bytes,
        :explicit_transfer_bytes,
    )
    summaries = map(fields) do field
        scalar_summary(getproperty(record, field) for record in records)
    end
    return NamedTuple{fields}(summaries)
end

function benchmark_amis_row(device_kind, cell)
    warmup = amis_prepare(
        device_kind, cell.scalar_type, cell.geometry, cell.dimension,
        cell.schedule, AMIS_BENCHMARK_SEED,
    )
    synchronized_run!(warmup, device_kind)
    seeds = ntuple(
        replicate -> AMIS_BENCHMARK_SEED + UInt(replicate),
        AMIS_BENCHMARK_REPLICATES,
    )
    records = map(seeds) do seed
        sampler = amis_prepare(
            device_kind, cell.scalar_type, cell.geometry, cell.dimension,
            cell.schedule, seed,
        )
        benchmark_record(sampler, device_kind, cell.schedule, seed)
    end
    executions = unique(record.execution for record in records)
    length(executions) == 1 || error("AMIS benchmark execution modes differ")
    return (;
        label=cell.label,
        device=device_kind,
        scalar_type=cell.scalar_type,
        geometry=cell.geometry,
        dimension=cell.dimension,
        execution=only(executions),
        rounds=length(cell.schedule),
        schedule=Tuple(cell.schedule),
        total_samples=sum(cell.schedule),
        prepared_samplers=AMIS_BENCHMARK_REPLICATES,
        replicate_seeds=seeds,
        records,
        summary=summarize_records(records),
    )
end

function standard_benchmark_cells()
    schedule = ntuple(_ -> AMIS_BENCHMARK_ROUND_SIZE, AMIS_BENCHMARK_ROUNDS)
    return Tuple((;
        label=Symbol(lowercase(string(T)), "_", geometry),
        scalar_type=T,
        geometry,
        dimension=geometry === :scalar ? 1 : 4,
        schedule,
    ) for (T, geometry) in AMIS_BENCHMARK_TYPE_GEOMETRIES if
        !AMIS_BENCHMARK_FACTOR_ONLY || geometry === :factor)
end

function scaling_benchmark_cells()
    return Tuple((;
        label=cell.label,
        scalar_type=Float32,
        geometry=:factor,
        dimension=cell.dimension,
        schedule=cell.schedule,
    ) for cell in AMIS_PERFORMANCE_SCALING_CELLS)
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
        Xoshiro(seed), AMISBenchmarkTarget{T}(), algorithm; threaded=true,
    )
    return device_kind === :cpu ? sampler : amis_cuda_device()(sampler)
end

function validate_guard_result(result, method, device_kind)
    logweights, transfer = host_logweights(result, device_kind)
    total_samples = method === :static_mis ? AMIS_GUARD_SAMPLE_COUNT :
                    2AMIS_GUARD_SAMPLE_COUNT
    length(result) == total_samples || error("$method guard sample count is wrong")
    all(isfinite, logweights) || error("$method guard produced nonfinite log weights")
    ess = weight_concentration_ess(logweights)
    isfinite(ess) && ess > 0 || error("$method guard produced invalid ESS")
    expected_method = method === :static_mis ? :importance_sampling :
                      :deterministic_mixture_pmc
    result.diagnostics.method === expected_method || error("$method diagnostics are wrong")
    expected_execution = Threads.nthreads(:default) > 1 ? :threaded : :serial
    result.diagnostics.execution === expected_execution || error("$method execution is wrong")
    result.diagnostics.failures == 0 || error("$method guard reported a failure")
    if method === :static_mis
        result.diagnostics.nsamples == AMIS_GUARD_SAMPLE_COUNT || error(
            "static MIS diagnostics sample count is wrong",
        )
    else
        result.diagnostics.rounds == 2 || error("DM-PMC guard reported wrong rounds")
        result.diagnostics.round_sizes == fill(AMIS_GUARD_SAMPLE_COUNT, 2) ||
            error("DM-PMC diagnostics schedule is wrong")
    end
    return (; total_samples, ess, measurement_transfers=transfer)
end

function benchmark_guard(method, device_kind)
    warmup = guard_prepare(method, device_kind, AMIS_BENCHMARK_SEED)
    synchronized_run!(warmup, device_kind)
    seeds = ntuple(
        replicate -> AMIS_BENCHMARK_SEED + UInt(replicate),
        AMIS_BENCHMARK_REPLICATES,
    )
    records = map(seeds) do seed
        sampler = guard_prepare(method, device_kind, seed)
        evaluation = one_evaluation(sampler, device_kind)
        validation = validate_guard_result(evaluation.result, method, device_kind)
        (;
            seed,
            seconds=evaluation.seconds,
            samples_per_second=validation.total_samples / evaluation.seconds,
            validation.total_samples,
            validation.ess,
            host_allocations=evaluation.host_allocations,
            host_allocated_bytes=evaluation.host_allocated_bytes,
            validation.measurement_transfers,
            benchmarktools_evaluations=evaluation.benchmarktools_evaluations,
        )
    end
    all(record -> record.benchmarktools_evaluations == 1, records) ||
        error("$method guard used multiple evaluations")
    allocation_sampler = guard_prepare(method, device_kind, first(seeds))
    device_allocated_bytes = if device_kind === :cpu
        missing
    else
        CUDA.synchronize()
        bytes = CUDA.@allocated synchronized_run!(allocation_sampler, :cuda)
        CUDA.synchronize()
        bytes
    end
    return (;
        method,
        device=device_kind,
        workload=AMIS_GUARD_SAMPLE_COUNT,
        prepared_samplers=AMIS_BENCHMARK_REPLICATES,
        allocation_samplers=device_kind === :cuda ? 1 : 0,
        device_allocated_bytes,
        replicate_seeds=seeds,
        records,
        samples_per_second=scalar_summary(record.samples_per_second for record in records),
        host_allocations=scalar_summary(record.host_allocations for record in records),
        host_allocated_bytes=scalar_summary(record.host_allocated_bytes for record in records),
    )
end

function validate_full_commit(commit, label)
    commit isa AbstractString && occursin(r"^[0-9a-f]{40}$", commit) ||
        error("$label must be a full lowercase Git commit")
    return commit
end

function guard_context(run)
    environment = run.environment
    return (;
        run.schema_version,
        environment.hostname,
        environment.julia,
        environment.cpu,
        environment.cpu_threads,
        environment.julia_threads,
        environment.cuda,
        environment.packages,
        run.mode,
        run.seed,
        run.benchmark_replicates,
        run.guard_sample_count,
        run.warmup,
        run.benchmarktools_evaluations_per_sampler,
        run.devices,
    )
end

function compare_guard_runs(
    base,
    candidate;
    expected_base_commit,
    expected_candidate_commit,
    threshold=0.05,
)
    validate_full_commit(expected_base_commit, "expected base commit")
    validate_full_commit(expected_candidate_commit, "expected candidate commit")
    base.environment.commit == expected_base_commit || error("base commit is wrong")
    candidate.environment.commit == expected_candidate_commit || error(
        "candidate commit is wrong",
    )
    guard_context(base) == guard_context(candidate) || error(
        "base and candidate host, Julia, package, hardware, thread, or workload contexts differ",
    )
    base.mode.guards_only || error("guard inputs require --guards-only")
    length(base.guards) == length(candidate.guards) || error(
        "base and candidate guard matrices differ",
    )
    rows = map(base.guards, candidate.guards) do base_guard, candidate_guard
        base_key = (
            base_guard.method, base_guard.device, base_guard.workload,
            base_guard.prepared_samplers, base_guard.allocation_samplers,
            base_guard.replicate_seeds,
        )
        candidate_key = (
            candidate_guard.method, candidate_guard.device, candidate_guard.workload,
            candidate_guard.prepared_samplers, candidate_guard.allocation_samplers,
            candidate_guard.replicate_seeds,
        )
        base_key == candidate_key || error("base and candidate guard rows differ")
        length(base_guard.records) == length(candidate_guard.records) ||
            error("guard replicate counts differ")
        for (base_record, candidate_record) in zip(base_guard.records,
                                                    candidate_guard.records)
            base_record.seed == candidate_record.seed || error("guard seeds differ")
            base_record.benchmarktools_evaluations == 1 &&
                candidate_record.benchmarktools_evaluations == 1 || error(
                "guard records must use one BenchmarkTools evaluation",
            )
        end
        base_throughput = [record.samples_per_second for record in base_guard.records]
        candidate_throughput = [
            record.samples_per_second for record in candidate_guard.records
        ]
        slow_replicates = count(
            candidate_throughput .< (1 - threshold) .* base_throughput,
        )
        throughput_ratio = candidate_guard.samples_per_second.median /
                           base_guard.samples_per_second.median
        repeatable_regression = slow_replicates >= 4 &&
                                throughput_ratio < 1 - threshold
        new_host_allocation =
            candidate_guard.host_allocations.maximum >
            base_guard.host_allocations.maximum ||
            candidate_guard.host_allocated_bytes.maximum >
            base_guard.host_allocated_bytes.maximum
        ismissing(base_guard.device_allocated_bytes) ==
            ismissing(candidate_guard.device_allocated_bytes) ||
            error("CUDA allocation records differ in availability")
        new_device_allocation = !ismissing(candidate_guard.device_allocated_bytes) &&
                                candidate_guard.device_allocated_bytes >
                                base_guard.device_allocated_bytes
        repeatable_regression && error(
            "$(candidate_guard.method) $(candidate_guard.device) throughput regressed repeatably by more than $(100threshold)%",
        )
        new_host_allocation && error(
            "$(candidate_guard.method) $(candidate_guard.device) introduced a host execution allocation",
        )
        new_device_allocation && error(
            "$(candidate_guard.method) $(candidate_guard.device) introduced a CUDA device allocation",
        )
        return (;
            method=candidate_guard.method,
            device=candidate_guard.device,
            throughput_ratio,
            slow_replicates,
            repeatable_regression,
            new_host_allocation,
            new_device_allocation,
        )
    end
    return Tuple(rows)
end

function loaded_checkout_record()
    checkout = dirname(dirname(pathof(ImportanceSamplers)))
    top_level = readchomp(`git -C $checkout rev-parse --show-toplevel`)
    normpath(top_level) == checkout || error("loaded package root is not its Git root")
    commit = readchomp(`git -C $checkout rev-parse HEAD`)
    validate_full_commit(commit, "loaded ImportanceSamplers HEAD")
    return (; checkout, commit)
end

function benchmark_environment(cuda_requested)
    loaded = loaded_checkout_record()
    wanted = Set(("BenchmarkTools", "CUDA", "ImportanceSamplers", "MLDataDevices"))
    packages = Tuple(sort!(
        [
            (name=dependency.name, version=string(dependency.version)) for
            dependency in values(Pkg.dependencies()) if dependency.name in wanted
        ];
        by=row -> row.name,
    ))
    cuda = cuda_requested && CUDA.functional() ? (
        gpu=CUDA.name(CUDA.device()),
        capability=CUDA.capability(CUDA.device()),
        driver=CUDA.driver_version(),
        runtime=CUDA.runtime_version(),
        allowscalar=false,
    ) : nothing
    return (;
        loaded.commit,
        loaded.checkout,
        hostname=gethostname(),
        julia=VERSION,
        cpu=Sys.CPU_NAME,
        cpu_threads=Sys.CPU_THREADS,
        julia_threads=Threads.nthreads(:default),
        cuda,
        packages,
    )
end

function compare_main()
    isnothing(AMIS_BENCHMARK_EXPECTED_BASE) && error(
        "--compare-guards requires --expected-base=<full commit>",
    )
    isnothing(AMIS_BENCHMARK_EXPECTED_CANDIDATE) && error(
        "--compare-guards requires --expected-candidate=<full commit>",
    )
    loaded = loaded_checkout_record()
    loaded.commit == AMIS_BENCHMARK_EXPECTED_CANDIDATE || error(
        "guard comparison must run from the expected candidate checkout",
    )
    isnothing(AMIS_BENCHMARK_BASE) && error("--compare-guards requires --base=<path>")
    isnothing(AMIS_BENCHMARK_CANDIDATE) && error(
        "--compare-guards requires --candidate=<path>",
    )
    base = open(Serialization.deserialize, AMIS_BENCHMARK_BASE)
    candidate = open(Serialization.deserialize, AMIS_BENCHMARK_CANDIDATE)
    rows = compare_guard_runs(
        base,
        candidate;
        expected_base_commit=AMIS_BENCHMARK_EXPECTED_BASE,
        expected_candidate_commit=AMIS_BENCHMARK_EXPECTED_CANDIDATE,
    )
    return (;
        expected_base_commit=AMIS_BENCHMARK_EXPECTED_BASE,
        expected_candidate_commit=AMIS_BENCHMARK_EXPECTED_CANDIDATE,
        comparison_checkout=loaded.checkout,
        rows,
        passed=true,
    )
end

function benchmark_main()
    devices = Symbol[]
    AMIS_BENCHMARK_CPU && push!(devices, :cpu)
    AMIS_BENCHMARK_CUDA && CUDA.functional() && push!(devices, :cuda)
    if AMIS_BENCHMARK_THREAD_SCALING
        devices == [:cpu] || error("thread scaling requires an available CPU-only run")
    end

    standard_cells = standard_benchmark_cells()
    rows = if AMIS_BENCHMARK_GUARDS_ONLY || AMIS_BENCHMARK_SCALING
        ()
    elseif AMIS_BENCHMARK_THREAD_SCALING
        cell = only(filter(cell -> cell.scalar_type === Float32 &&
                                  cell.geometry === :factor, standard_cells))
        (benchmark_amis_row(:cpu, cell),)
    else
        Tuple(benchmark_amis_row(device, cell) for device in devices for
              cell in standard_cells)
    end
    scaling_rows = AMIS_BENCHMARK_SCALING ? Tuple(
        benchmark_amis_row(device, cell) for device in devices for
        cell in scaling_benchmark_cells()
    ) : ()
    guards = AMIS_BENCHMARK_SCALING || AMIS_BENCHMARK_THREAD_SCALING ? () : Tuple(
        benchmark_guard(method, device) for device in devices for
        method in (:static_mis, :dm_pmc)
    )
    run = (;
        schema_version=AMIS_BENCHMARK_SCHEMA_VERSION,
        command="include(\"amis.jl\") in the benchmark project",
        mode=(;
            smoke=AMIS_BENCHMARK_SMOKE,
            long=AMIS_BENCHMARK_LONG,
            scaling=AMIS_BENCHMARK_SCALING,
            thread_scaling=AMIS_BENCHMARK_THREAD_SCALING,
            guards_only=AMIS_BENCHMARK_GUARDS_ONLY,
            factor_only=AMIS_BENCHMARK_FACTOR_ONLY,
        ),
        seed=AMIS_BENCHMARK_SEED,
        benchmark_replicates=AMIS_BENCHMARK_REPLICATES,
        guard_sample_count=AMIS_GUARD_SAMPLE_COUNT,
        warmup=false,
        benchmarktools_evaluations_per_sampler=1,
        devices=Tuple(devices),
        environment=benchmark_environment(AMIS_BENCHMARK_CUDA),
        rows,
        scaling_rows,
        guards,
        unavailable_cuda=AMIS_BENCHMARK_CUDA && !CUDA.functional(),
    )
    if !isnothing(AMIS_BENCHMARK_SAVE)
        open(AMIS_BENCHMARK_SAVE, "w") do io
            Serialization.serialize(io, run)
        end
    end
    return run
end

main() = AMIS_BENCHMARK_COMPARE ? compare_main() : benchmark_main()

main()
