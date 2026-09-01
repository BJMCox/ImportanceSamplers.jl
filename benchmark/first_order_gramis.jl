using BenchmarkTools
using ImportanceSamplers
using LinearAlgebra
using Pkg
using Random

function gram_is_benchmark_option(name)
    prefix = "--$(name)="
    values = [argument[(length(prefix) + 1):end] for argument in ARGS if
              startswith(argument, prefix)]
    length(values) <= 1 || error("repeat benchmark option --$name")
    return isempty(values) ? nothing : only(values)
end

function gram_is_benchmark_positive_integer(name)
    value = gram_is_benchmark_option(name)
    value === nothing && return nothing
    parsed = tryparse(Int, value)
    parsed !== nothing && parsed > 0 ||
        error("--$name must be a positive integer")
    return parsed
end

function gram_is_benchmark_integer_list(name, defaults)
    value = gram_is_benchmark_option(name)
    value === nothing && return defaults
    parsed = tryparse.(Int, split(value, ','))
    all(item -> item !== nothing && item > 0, parsed) ||
        error("--$name must be a comma-separated list of positive integers")
    return Tuple(something.(parsed))
end

function gram_is_benchmark_source_provenance(; required=false)
    value = get(ENV, "IMPORTANCE_SAMPLERS_SOURCE_COMMIT", "")
    supplied = !isempty(value)
    required && !supplied && error(
        "CUDA benchmarks require IMPORTANCE_SAMPLERS_SOURCE_COMMIT",
    )
    supplied && !occursin(r"^[0-9a-f]{40}$", value) && error(
        "IMPORTANCE_SAMPLERS_SOURCE_COMMIT must be 40 lowercase hex characters",
    )
    return (
        environment_variable="IMPORTANCE_SAMPLERS_SOURCE_COMMIT",
        commit=supplied ? value : nothing,
        supplied,
    )
end

const GRAMIS_BENCHMARK_SMOKE = "--smoke" in ARGS
const GRAMIS_BENCHMARK_SCALING = "--scaling" in ARGS
const GRAMIS_BENCHMARK_CUDA_ONLY = "--cuda-only" in ARGS
const GRAMIS_BENCHMARK_CUDA = GRAMIS_BENCHMARK_CUDA_ONLY || "--cuda" in ARGS
const GRAMIS_BENCHMARK_CPU = !GRAMIS_BENCHMARK_CUDA_ONLY
const GRAMIS_BENCHMARK_PROFILE = "--profile" in ARGS
"--cpu-only" in ARGS && GRAMIS_BENCHMARK_CUDA &&
    error("--cpu-only cannot be combined with --cuda or --cuda-only")
GRAMIS_BENCHMARK_PROFILE && !GRAMIS_BENCHMARK_CUDA &&
    error("--profile requires --cuda or --cuda-only")
const GRAMIS_BENCHMARK_SOURCE = gram_is_benchmark_source_provenance(
    required=GRAMIS_BENCHMARK_CUDA,
)

if GRAMIS_BENCHMARK_CUDA
    @eval using CUDA
    @eval using MLDataDevices
    @eval function gram_is_cuda_allocated_warm_execution(sampler)
        return CUDA.@allocated gram_is_synchronized_run!(sampler, :cuda)
    end
    CUDA.functional() || error("CUDA is not functional")
    CUDA.allowscalar(false)
end

const GRAMIS_BENCHMARK_ROUND_SIZE =
    gram_is_benchmark_positive_integer("round-size")
GRAMIS_BENCHMARK_SCALING && GRAMIS_BENCHMARK_ROUND_SIZE !== nothing &&
    error("--scaling is an alias for --round-size=8192; choose one")
GRAMIS_BENCHMARK_SMOKE &&
    (GRAMIS_BENCHMARK_SCALING || GRAMIS_BENCHMARK_ROUND_SIZE !== nothing) &&
    error("--smoke cannot be combined with a round-size override")
const GRAMIS_BENCHMARK_EFFECTIVE_ROUND_SIZE = GRAMIS_BENCHMARK_SCALING ?
                                              8_192 :
                                              GRAMIS_BENCHMARK_ROUND_SIZE
const GRAMIS_BENCHMARK_SEED = 0x6772616d69736265
const GRAMIS_BENCHMARK_TYPES = GRAMIS_BENCHMARK_SMOKE ?
                               (Float32,) : (Float32, Float64)
const GRAMIS_BENCHMARK_DIMENSIONS = GRAMIS_BENCHMARK_SMOKE ?
                                    (4,) : gram_is_benchmark_integer_list(
                                        "dimensions",
                                        (4, 16, 32),
                                    )
const GRAMIS_BENCHMARK_PROPOSALS = GRAMIS_BENCHMARK_SMOKE ?
                                   (4,) : gram_is_benchmark_integer_list(
                                       "proposals",
                                       (4, 16),
                                   )
const GRAMIS_BENCHMARK_TEMPERING = (:inactive, :active)
const GRAMIS_BENCHMARK_SERIAL_ONLY = "--serial-only" in ARGS
const GRAMIS_BENCHMARK_THREADED_ONLY = "--threaded-only" in ARGS
GRAMIS_BENCHMARK_SERIAL_ONLY && GRAMIS_BENCHMARK_THREADED_ONLY &&
    error("choose at most one execution selector")
const GRAMIS_BENCHMARK_CPU_EXECUTIONS = GRAMIS_BENCHMARK_SERIAL_ONLY ?
                                        (:serial,) :
                                        GRAMIS_BENCHMARK_THREADED_ONLY ?
                                        (:threaded,) : (:serial, :threaded)
const GRAMIS_BENCHMARK_FACTOR_EXECUTIONS = (
    FusedFactorExecution(),
    BatchedFactorExecution(),
)
const GRAMIS_BENCHMARK_DEFAULT_SAMPLES = GRAMIS_BENCHMARK_SMOKE ? 3 :
                                         GRAMIS_BENCHMARK_EFFECTIVE_ROUND_SIZE === nothing ?
                                         3 :
                                         GRAMIS_BENCHMARK_EFFECTIVE_ROUND_SIZE <= 65_536 ?
                                         7 : 5
const GRAMIS_BENCHMARK_SAMPLES = something(
    gram_is_benchmark_positive_integer("samples"),
    GRAMIS_BENCHMARK_DEFAULT_SAMPLES,
)

struct GRAMISBenchmarkCPUTarget{T<:AbstractFloat} end
struct GRAMISBenchmarkCUDATarget{T<:AbstractFloat} end

@inline function (::GRAMISBenchmarkCPUTarget{T})(sample) where {T}
    return -T(0.5) * sum(abs2, sample)
end

@inline function (::GRAMISBenchmarkCUDATarget{T})(sample) where {T}
    radius = zero(T)
    @inbounds for coordinate in eachindex(sample)
        radius += abs2(sample[coordinate])
    end
    return -T(0.5) * radius
end

function gram_is_benchmark_gradient!(destination, sample)
    @inbounds for index in 1:length(destination)
        destination[index] = -sample[index]
    end
    return destination
end

function gram_is_benchmark_bank(
    ::Type{T},
    dimension,
    proposal_count,
    tempering,
) where {T}
    amplitude = tempering === :inactive ? sqrt(eps(T)) : T(0.5 / sqrt(dimension))
    proposals = map(1:proposal_count) do proposal_slot
        location = Vector{T}(undef, dimension)
        phase = T(2pi * (proposal_slot - 1) / proposal_count)
        @inbounds for coordinate in 1:dimension
            location[coordinate] = amplitude *
                                   sin(phase + T(0.61 * coordinate))
        end
        FactorGaussian(location, Matrix{T}(I, dimension, dimension))
    end
    return ProposalBank(proposals)
end

function gram_is_benchmark_cell(
    ::Type{T},
    dimension,
    proposal_count,
    tempering,
    execution,
    factor_execution,
    device,
) where {T}
    round_size = GRAMIS_BENCHMARK_EFFECTIVE_ROUND_SIZE === nothing ?
                 proposal_count * (dimension + 8) :
                 GRAMIS_BENCHMARK_EFFECTIVE_ROUND_SIZE
    iszero(round_size % proposal_count) || error(
        "round size $round_size must be divisible by $proposal_count proposals",
    )
    samples_per_proposal = round_size ÷ proposal_count
    threshold = tempering === :inactive ? dimension + 1 : samples_per_proposal - 1
    return (
        device,
        scalar_type=T,
        dimension,
        proposals=proposal_count,
        tempering,
        execution,
        threaded=execution !== :serial,
        factor_execution,
        factor_execution_policy=ImportanceSamplers._factor_execution_name(
            factor_execution,
        ),
        rounds=2,
        round_size,
        covariance_ess_threshold=threshold,
    )
end

function gram_is_cuda_device()
    physical = CUDA.device()
    return MLDataDevices.CUDADevice{typeof(physical),Nothing}(physical)
end

function gram_is_prepare_benchmark(cell, seed)
    target_function = cell.device === :cpu ?
                      GRAMISBenchmarkCPUTarget{cell.scalar_type}() :
                      GRAMISBenchmarkCUDATarget{cell.scalar_type}()
    target = LogTarget(
        target_function;
        grad=gram_is_benchmark_gradient!,
    )
    algorithm = FirstOrderGRAMIS(
        gram_is_benchmark_bank(
            cell.scalar_type,
            cell.dimension,
            cell.proposals,
            cell.tempering,
        );
        rounds=cell.rounds,
        round_size=cell.round_size,
        repulsion_strength=cell.tempering === :inactive ?
                           zero(cell.scalar_type) : cell.scalar_type(0.05),
        covariance_rate=cell.tempering === :inactive ?
                        eps(cell.scalar_type) : one(cell.scalar_type),
        covariance_ess_threshold=cell.covariance_ess_threshold,
    )
    source = prepare_sampler(
        Xoshiro(seed),
        target,
        algorithm;
        threaded=cell.threaded,
        factor_execution=cell.factor_execution,
    )
    if cell.device === :cpu
        return source
    elseif cell.device === :cuda
        sampler = gram_is_cuda_device()(source)
        CUDA.synchronize()
        return sampler
    end
    error("unknown benchmark device $(cell.device)")
end

function gram_is_synchronized_run!(sampler, device)
    result = importance_sample!(sampler)
    device === :cuda && CUDA.synchronize()
    return result
end

function gram_is_host_array(array, device)
    host = device === :cuda ? Array(array) : copy(array)
    device === :cuda && CUDA.synchronize()
    return host
end

function gram_is_transfer_record(transfers)
    names = fieldnames(typeof(transfers.reasons))
    reasons = NamedTuple{names}(map(names) do reason
        record = getfield(transfers.reasons, reason)
        (count=record.count, bytes=record.bytes)
    end)
    reason_count = sum(record.count for record in values(reasons))
    reason_bytes = sum(record.bytes for record in values(reasons))
    reason_count <= transfers.count && reason_bytes <= transfers.bytes || error(
        "reported transfer reasons exceed the aggregate",
    )
    return (
        count=transfers.count,
        bytes=transfers.bytes,
        reasons,
        unattributed=(
            count=transfers.count - reason_count,
            bytes=transfers.bytes - reason_bytes,
        ),
    )
end

function gram_is_result_signature(result, sampler, cell)
    samples = gram_is_host_array(result.samples, cell.device)
    logweights = gram_is_host_array(result.logweights, cell.device)
    rounds = gram_is_host_array(result.provenance.round, cell.device)
    proposal_ids = gram_is_host_array(result.provenance.proposal_id, cell.device)
    status = gram_is_host_array(result.diagnostics.fallback_status, cell.device)
    steps = gram_is_host_array(result.diagnostics.accepted_steps, cell.device)
    trials = gram_is_host_array(result.diagnostics.backtracking_trials, cell.device)
    tempering_powers = gram_is_host_array(
        result.diagnostics.tempering_powers,
        cell.device,
    )
    locations = gram_is_host_array(
        sampler.method_state.committed.locations,
        cell.device,
    )
    factors = gram_is_host_array(
        sampler.method_state.committed.factors,
        cell.device,
    )
    return (;
        samples,
        logweights,
        rounds,
        proposal_ids,
        status,
        steps,
        trials,
        tempering_powers,
        locations,
        factors,
    )
end

function gram_is_validate_benchmark_result(result, sampler, cell)
    expected_samples = cell.rounds * cell.round_size
    length(result) == expected_samples || error("benchmark sample count changed")
    signature = gram_is_result_signature(result, sampler, cell)
    all(isfinite, signature.logweights) || error(
        "benchmark produced nonfinite weights",
    )
    if cell.tempering === :inactive
        all(isone, signature.tempering_powers) ||
            error("inactive tempering cell invoked tempering")
    else
        any(<(one(cell.scalar_type)), signature.tempering_powers) ||
            error("active tempering cell did not invoke tempering")
    end
    maximum_logweight = maximum(signature.logweights)
    weights = exp.(signature.logweights .- maximum_logweight)
    weights ./= sum(weights)
    return (
        expected_samples,
        ess=inv(sum(abs2, weights)),
        signature,
        measurement_transfers=cell.device === :cuda ? (
            count=10,
            bytes=sizeof(signature.samples) + sizeof(signature.logweights) +
                  sizeof(signature.rounds) + sizeof(signature.proposal_ids) +
                  sizeof(signature.status) + sizeof(signature.steps) +
                  sizeof(signature.trials) +
                  sizeof(signature.tempering_powers) +
                  sizeof(signature.locations) + sizeof(signature.factors),
            reason=:post_timing_determinism_and_ess,
        ) : (count=0, bytes=0, reason=:not_required),
        reported_explicit_transfers=gram_is_transfer_record(
            result.diagnostics.transfers,
        ),
    )
end

function gram_is_kernel_family(name)
    lowered = lowercase(name)
    occursin("backtracking", lowered) && return :backtracking
    occursin("tempering", lowered) && return :tempering
    (occursin("mis_round", lowered) || occursin("local_weights", lowered) ||
     occursin("rand", lowered)) && return :sampling
    (occursin("covariance", lowered) || occursin("factor_population", lowered) ||
     occursin("potrf", lowered)) && return :covariance
    (occursin("frozen", lowered) || occursin("precondition", lowered)) &&
        return :derivative
    (occursin("repulsion", lowered) || occursin("whiten", lowered) ||
     occursin("minimum_first_order", lowered)) && return :repulsion
    (occursin("gemm", lowered) || occursin("trsm", lowered)) &&
        return :linear_algebra
    occursin("logweight", lowered) && return :diagnostics
    return :other
end

@assert gram_is_kernel_family("volta_sgemm_128x64") === :linear_algebra
@assert gram_is_kernel_family("minimum_first_order_repulsion_kernel!") ===
        :repulsion

function gram_is_copy_direction(name)
    matched = match(
        r"^\[copy (pageable|pinned|device) to (pageable|pinned|device) memory\]$",
        lowercase(name),
    )
    matched === nothing && return :other
    source, destination = matched.captures
    host_kinds = ("pageable", "pinned")
    source == "device" && destination in host_kinds && return :device_to_host
    source in host_kinds && destination == "device" && return :host_to_device
    source == "device" && destination == "device" && return :device_to_device
    return :other
end

function gram_is_profile_record(profile)
    host = profile.host
    device = profile.device
    first_sync = findfirst(==("cuCtxSynchronize"), host.name)
    first_sync === nothing && error("CUPTI profile has no opening synchronization")
    last_sync = findlast(==("cuCtxSynchronize"), host.name)
    first_sync == last_sync && error("CUPTI profile has no closing synchronization")
    first_id = host.id[first_sync + 1]
    last_id = host.id[last_sync - 1]
    host_indices = findall(index -> first_id <= host.id[index] <= last_id, eachindex(host.id))
    device_indices = findall(
        index -> first_id <= device.id[index] <= last_id,
        eachindex(device.id),
    )
    kernel_indices = filter(device_indices) do index
        !ismissing(device.grid[index])
    end
    families = (
        :sampling,
        :tempering,
        :covariance,
        :derivative,
        :backtracking,
        :repulsion,
        :linear_algebra,
        :diagnostics,
        :other,
    )
    family_launches = Dict(family => 0 for family in families)
    family_seconds = Dict(family => 0.0 for family in families)
    for index in kernel_indices
        family = gram_is_kernel_family(device.name[index])
        family_launches[family] += 1
        family_seconds[family] += device.stop[index] - device.start[index]
    end
    copy_indices = filter(device_indices) do index
        !ismissing(device.size[index])
    end
    directions = (
        :host_to_device,
        :device_to_host,
        :device_to_device,
        :other,
    )
    copy_directions = Dict(
        direction => filter(
            index -> gram_is_copy_direction(device.name[index]) === direction,
            copy_indices,
        ) for direction in directions
    )
    sum(length, values(copy_directions)) == length(copy_indices) || error(
        "CUPTI copy records did not partition by direction",
    )
    backtracking_validation_indices = filter(kernel_indices) do index
        occursin("validate_backtracking_candidates", device.name[index])
    end
    failure_record_bytes = 3sizeof(UInt64)
    failure_record_copy_indices = filter(copy_indices) do index
        device.size[index] == failure_record_bytes &&
            gram_is_copy_direction(device.name[index]) === :device_to_host
    end
    synchronizations = count(index -> begin
        name = host.name[index]
        occursin("synchronize", lowercase(name))
    end, host_indices)
    copy_record(direction) = (
        count=length(copy_directions[direction]),
        bytes=sum(
            (device.size[index] for index in copy_directions[direction]);
            init=0,
        ),
    )
    return (
        kernel_launches=length(kernel_indices),
        synchronizations,
        profile_window=(first_correlation_id=first_id, last_correlation_id=last_id),
        device_kernel_seconds=sum(
            (device.stop[index] - device.start[index] for index in kernel_indices);
            init=0.0,
        ),
        device_kernel_family_launches=NamedTuple{families}(
            Tuple(family_launches[family] for family in families),
        ),
        device_kernel_family_seconds=NamedTuple{families}(
            Tuple(family_seconds[family] for family in families),
        ),
        host_to_device_copies=copy_record(:host_to_device),
        device_to_host_copies=copy_record(:device_to_host),
        device_to_device_copies=copy_record(:device_to_device),
        other_copies=copy_record(:other),
        backtracking_validation_launches=length(backtracking_validation_indices),
        failure_record_copies_upper_bound=(
            count=length(failure_record_copy_indices),
            bytes=sum(
                (device.size[index] for index in failure_record_copy_indices);
                init=0,
            ),
        ),
    )
end

function gram_is_cuda_profile(cell, seed)
    GRAMIS_BENCHMARK_PROFILE || return missing
    sampler = gram_is_prepare_benchmark(cell, seed)
    result = Ref{Any}()
    profile = CUDA.Profile.profile_internally() do
        result[] = gram_is_synchronized_run!(sampler, :cuda)
    end
    result[] === nothing && error("profiled execution did not return a result")
    return gram_is_profile_record(profile)
end

function gram_is_cuda_metrics(cell, seed)
    cell.device === :cuda || return missing
    allocation_sampler = gram_is_prepare_benchmark(cell, seed)
    CUDA.synchronize()
    device_memory_before = CUDA.used_memory()
    device_allocated_bytes = gram_is_cuda_allocated_warm_execution(
        allocation_sampler,
    )
    device_memory_after = CUDA.used_memory()
    profile_metrics = gram_is_cuda_profile(cell, seed)
    return (
        device_allocation=(
            allocated_bytes=device_allocated_bytes,
            pool_used_before=device_memory_before,
            pool_used_after=device_memory_after,
            retained_pool_delta=device_memory_after - device_memory_before,
            peak_memory_measured=false,
        ),
        profile=profile_metrics,
    )
end

function gram_is_benchmark_run(cell, seed)
    result_holder = Ref{Any}()
    sampler_holder = Ref{Any}()
    benchmark = @benchmarkable begin
        $sampler_holder[] = sampler
        $result_holder[] = gram_is_synchronized_run!(sampler, $(cell.device))
    end setup = (sampler = gram_is_prepare_benchmark($cell, $seed))
    trial = run(
        benchmark;
        samples=GRAMIS_BENCHMARK_SAMPLES,
        evals=1,
        warmup=false,
    )
    fastest = minimum(trial)
    estimate = median(trial)
    slowest = maximum(trial)
    validated = gram_is_validate_benchmark_result(
        result_holder[],
        sampler_holder[],
        cell,
    )
    seconds = estimate.time / 1.0e9
    seconds_range = (fastest.time / 1.0e9, slowest.time / 1.0e9)
    cuda = gram_is_cuda_metrics(cell, seed)
    return (
        seconds,
        seconds_range,
        samples_per_second=validated.expected_samples / seconds,
        samples_per_second_range=(
            validated.expected_samples / seconds_range[2],
            validated.expected_samples / seconds_range[1],
        ),
        ess=validated.ess,
        ess_per_second=validated.ess / seconds,
        ess_per_second_range=(
            validated.ess / seconds_range[2],
            validated.ess / seconds_range[1],
        ),
        host_allocations=estimate.allocs,
        host_allocated_bytes=estimate.memory,
        reported_explicit_transfers=validated.reported_explicit_transfers,
        measurement_transfers=validated.measurement_transfers,
        signature=validated.signature,
        cuda,
    )
end

function gram_is_benchmark_cell(cell)
    seed = GRAMIS_BENCHMARK_SEED + UInt64(1)
    warm_sampler = gram_is_prepare_benchmark(cell, seed)
    warm_result = gram_is_synchronized_run!(warm_sampler, cell.device)
    warm_validation = gram_is_validate_benchmark_result(
        warm_result,
        warm_sampler,
        cell,
    )
    run = gram_is_benchmark_run(cell, seed)
    run.signature == warm_validation.signature || error(
        "fresh same-seed benchmark executions are not deterministic",
    )
    return merge(
        cell,
        (
            julia_threads=Threads.nthreads(:default),
            benchmark_samples=GRAMIS_BENCHMARK_SAMPLES,
            deterministic_same_seed=true,
            seconds=run.seconds,
            seconds_range=run.seconds_range,
            samples_per_second=run.samples_per_second,
            samples_per_second_range=run.samples_per_second_range,
            ess=run.ess,
            ess_per_second=run.ess_per_second,
            ess_per_second_range=run.ess_per_second_range,
            host_allocations=run.host_allocations,
            host_allocated_bytes=run.host_allocated_bytes,
            reported_explicit_transfers=run.reported_explicit_transfers,
            measurement_transfers=run.measurement_transfers,
            cuda=run.cuda,
        ),
    )
end

function gram_is_benchmark_configurations()
    return GRAMIS_BENCHMARK_SCALING ?
           ((4, 4), (16, 16), (32, 16)) :
           Tuple(
        (dimension, proposals) for dimension in GRAMIS_BENCHMARK_DIMENSIONS for
        proposals in GRAMIS_BENCHMARK_PROPOSALS
    )
end

function gram_is_benchmark_cells()
    configurations = gram_is_benchmark_configurations()
    cells = Any[]
    if GRAMIS_BENCHMARK_CPU
        append!(
            cells,
            (
                gram_is_benchmark_cell(
                    T,
                    dimension,
                    proposals,
                    tempering,
                    execution,
                    factor_execution,
                    :cpu,
                ) for T in GRAMIS_BENCHMARK_TYPES for
                (dimension, proposals) in configurations for
                tempering in GRAMIS_BENCHMARK_TEMPERING for
                execution in GRAMIS_BENCHMARK_CPU_EXECUTIONS for
                factor_execution in GRAMIS_BENCHMARK_FACTOR_EXECUTIONS
            ),
        )
    end
    if GRAMIS_BENCHMARK_CUDA
        append!(
            cells,
            (
                gram_is_benchmark_cell(
                    T,
                    dimension,
                    proposals,
                    tempering,
                    :cuda,
                    factor_execution,
                    :cuda,
                ) for T in GRAMIS_BENCHMARK_TYPES for
                (dimension, proposals) in configurations for
                tempering in GRAMIS_BENCHMARK_TEMPERING for
                factor_execution in GRAMIS_BENCHMARK_FACTOR_EXECUTIONS
            ),
        )
    end
    return cells
end

function gram_is_package_versions()
    wanted = Set((
        "Adapt",
        "BenchmarkTools",
        "CUDA",
        "ImportanceSamplers",
        "KernelAbstractions",
        "MLDataDevices",
    ))
    return sort!(
        [
            (dependency.name, something(dependency.version, "unversioned")) for
            dependency in values(Pkg.dependencies()) if dependency.name in wanted
        ];
        by=first,
    )
end

function gram_is_benchmark_main()
    configurations = gram_is_benchmark_configurations()
    cells = gram_is_benchmark_cells()
    rows = [gram_is_benchmark_cell(cell) for cell in cells]
    expected_cpu_rows = GRAMIS_BENCHMARK_CPU ?
                        length(GRAMIS_BENCHMARK_TYPES) * length(configurations) *
                        length(GRAMIS_BENCHMARK_TEMPERING) *
                        length(GRAMIS_BENCHMARK_CPU_EXECUTIONS) *
                        length(GRAMIS_BENCHMARK_FACTOR_EXECUTIONS) : 0
    expected_cuda_rows = GRAMIS_BENCHMARK_CUDA ?
                         length(GRAMIS_BENCHMARK_TYPES) * length(configurations) *
                         length(GRAMIS_BENCHMARK_TEMPERING) *
                         length(GRAMIS_BENCHMARK_FACTOR_EXECUTIONS) : 0
    length(rows) == expected_cpu_rows + expected_cuda_rows ||
        error("benchmark matrix is incomplete")
    cuda = GRAMIS_BENCHMARK_CUDA ? (
        gpu=CUDA.name(CUDA.device()),
        capability=CUDA.capability(CUDA.device()),
        driver=CUDA.driver_version(),
        runtime=CUDA.runtime_version(),
        total_device_memory=CUDA.total_memory(),
        memory_info=CUDA.memory_info(),
    ) : nothing
    return (
        environment=(
            julia=VERSION,
            cpu=Sys.CPU_NAME,
            cpu_threads=Sys.CPU_THREADS,
            julia_threads=Threads.nthreads(:default),
            cuda,
            packages=gram_is_package_versions(),
            seed=GRAMIS_BENCHMARK_SEED,
            source=GRAMIS_BENCHMARK_SOURCE,
        ),
        smoke=GRAMIS_BENCHMARK_SMOKE,
        profile=GRAMIS_BENCHMARK_PROFILE,
        round_size_override=GRAMIS_BENCHMARK_EFFECTIVE_ROUND_SIZE,
        tier=GRAMIS_BENCHMARK_EFFECTIVE_ROUND_SIZE === nothing ? :overhead :
             GRAMIS_BENCHMARK_EFFECTIVE_ROUND_SIZE == 8_192 ? :overhead_only :
             GRAMIS_BENCHMARK_EFFECTIVE_ROUND_SIZE == 65_536 ? :broad_matrix :
             GRAMIS_BENCHMARK_EFFECTIVE_ROUND_SIZE == 262_144 ?
             :production_scale :
             GRAMIS_BENCHMARK_EFFECTIVE_ROUND_SIZE == 1_048_576 ?
             :sustained_throughput : :custom,
        measurement_contract=(
            setup_outside_timing=true,
            synchronized_cuda=true,
            evals=1,
            repeated_median=true,
            deterministic_same_seed=true,
            device_kernel_family_attribution=:lower_bound,
            device_kernel_family_excludes=(
                :host_work,
                :transfers,
                :synchronization_barriers,
            ),
            cupti_window=:between_profiler_context_synchronizations,
        ),
        rows,
    )
end

const GRAMIS_BENCHMARK_RESULT = gram_is_benchmark_main()
show(stdout, MIME("text/plain"), GRAMIS_BENCHMARK_RESULT)
println()
GRAMIS_BENCHMARK_RESULT
