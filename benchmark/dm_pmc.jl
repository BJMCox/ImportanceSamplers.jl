using BenchmarkTools
using CUDA
using ImportanceSamplers
using LinearAlgebra
using MLDataDevices
using Pkg
using Random

const DM_PMC_BENCHMARK_SMOKE = "--smoke" in ARGS
const DM_PMC_BENCHMARK_CPU = !("--cuda-only" in ARGS)
const DM_PMC_BENCHMARK_CUDA = !("--cpu-only" in ARGS)
const DM_PMC_BENCHMARK_SEED = 0x646d706d6362656e
const DM_PMC_BENCHMARK_ROUNDS = DM_PMC_BENCHMARK_SMOKE ? 2 : 5
const DM_PMC_BENCHMARK_ROUND_SIZE = DM_PMC_BENCHMARK_SMOKE ? 256 : 10_000
const DM_PMC_BENCHMARK_TRIAL_SECONDS = DM_PMC_BENCHMARK_SMOKE ? 0.05 : 1.0
const DM_PMC_BENCHMARK_PREPARATION_SAMPLES = DM_PMC_BENCHMARK_SMOKE ? 1 : 2
const DM_PMC_BENCHMARK_REPLICATES = DM_PMC_BENCHMARK_SMOKE ? 2 : 5
const DM_PMC_BENCHMARK_TYPES = (Float32, Float64)
const DM_PMC_BENCHMARK_BANKS = (:diagonal, :factor)
const DM_PMC_BENCHMARK_DIMENSIONS = (4, 16)
const DM_PMC_BENCHMARK_PROPOSALS = (4, 16)
const DM_PMC_TRANSFER_REASONS = (
    :failure_snapshot,
    :cdf_maximum,
    :cdf_sum,
    :summary_maximum,
    :summary_scaled_sum,
    :summary_scaled_square_sum,
)

struct DMPMCNormalTarget{T<:AbstractFloat}
    scale::T
end

@inline function (target::DMPMCNormalTarget{T})(sample) where {T}
    squared_radius = zero(T)
    @inbounds for coordinate in eachindex(sample)
        squared_radius += abs2(sample[coordinate] / target.scale)
    end
    return -T(0.5) * squared_radius -
           T(length(sample)) * (log(target.scale) + T(0.5) * log(T(2pi)))
end

function dm_pmc_benchmark_factor(::Type{T}, dimension, slot) where {T}
    factor = zeros(T, dimension, dimension)
    for row in 1:dimension
        factor[row, row] = T(1 + 0.025 * sin(0.4 * slot + 0.3 * row) / sqrt(dimension))
        for column in 1:(row - 1)
            factor[row, column] = T(0.012 / sqrt(dimension)) *
                                  sin(T(0.3 * slot + 0.2 * row - 0.1 * column))
        end
    end
    return factor
end

function dm_pmc_benchmark_locations(::Type{T}, dimension, proposal_count) where {T}
    return [
        T[
            0.25 / sqrt(dimension) * sin(
                2pi * (slot - 1) / proposal_count + 0.37 * coordinate,
            ) for coordinate in 1:dimension
        ] for slot in 1:proposal_count
    ]
end

function dm_pmc_benchmark_bank(
    ::Type{T},
    kind,
    dimension,
    proposal_count,
) where {T}
    locations = dm_pmc_benchmark_locations(T, dimension, proposal_count)
    if kind === :diagonal
        proposals = [
            DiagonalGaussian(
                locations[slot],
                diag(dm_pmc_benchmark_factor(T, dimension, slot)),
            ) for slot in 1:proposal_count
        ]
    elseif kind === :factor
        proposals = [
            FactorGaussian(
                locations[slot],
                dm_pmc_benchmark_factor(T, dimension, slot),
            ) for slot in 1:proposal_count
        ]
    else
        error("unknown benchmark bank kind $kind")
    end
    return ProposalBank(proposals, ones(T, proposal_count))
end

function dm_pmc_cuda_device()
    CUDA.functional() || error("CUDA is not functional")
    physical = CUDA.device()
    CUDA.allowscalar(false)
    return MLDataDevices.CUDADevice{typeof(physical),Nothing}(physical)
end

function dm_pmc_prepare_cell(cell, device_kind, seed=DM_PMC_BENCHMARK_SEED)
    bank = dm_pmc_benchmark_bank(
        cell.scalar_type,
        cell.bank,
        cell.dimension,
        cell.proposals,
    )
    algorithm = DeterministicMixturePMC(
        bank;
        rounds=cell.rounds,
        round_size=cell.round_size,
    )
    source = prepare_sampler(
        Xoshiro(seed),
        DMPMCNormalTarget{cell.scalar_type}(one(cell.scalar_type)),
        algorithm;
        threaded=true,
    )
    if device_kind === :cpu
        return source
    elseif device_kind === :cuda
        sampler = dm_pmc_cuda_device()(source)
        CUDA.synchronize()
        return sampler
    end
    error("unknown benchmark device $device_kind")
end

function dm_pmc_synchronized_run!(sampler, device_kind)
    result = importance_sample!(sampler)
    device_kind === :cuda && CUDA.synchronize()
    return result
end

function dm_pmc_run_preparation_trial(benchmark)
    return run(
        benchmark;
        samples=DM_PMC_BENCHMARK_PREPARATION_SAMPLES,
        seconds=DM_PMC_BENCHMARK_TRIAL_SECONDS,
        evals=1,
    )
end

function dm_pmc_trial_record(trial)
    estimate = median(trial)
    return (;
        median_seconds=estimate.time / 1.0e9,
        host_allocations=estimate.allocs,
        host_allocated_bytes=estimate.memory,
    )
end

function dm_pmc_host_diagnostics(result, device_kind)
    if device_kind === :cpu
        return (
            logweights=copy(result.logweights),
            round_diagnostic_ess=copy(result.diagnostics.round_ess),
            measurement_transfers=(
                flattened_logweights=(count=0, bytes=0, reason=:not_required),
                round_diagnostic_ess=(count=0, bytes=0, reason=:not_required),
            ),
        )
    end
    host_logweights = Array(result.logweights)
    host_round_ess = Array(result.diagnostics.round_ess)
    CUDA.synchronize()
    return (
        logweights=host_logweights,
        round_diagnostic_ess=host_round_ess,
        measurement_transfers=(
            flattened_logweights=(
                count=1,
                bytes=sizeof(host_logweights),
                reason=:benchmark_postrun_flattened_weight_concentration_ess,
            ),
            round_diagnostic_ess=(
                count=1,
                bytes=sizeof(host_round_ess),
                reason=:benchmark_postrun_round_diagnostic_ess,
            ),
        ),
    )
end

function dm_pmc_flattened_weight_concentration_ess(logweights)
    maximum_logweight = maximum(logweights)
    scaled = exp.(logweights .- maximum_logweight)
    return abs2(sum(scaled)) / sum(abs2, scaled)
end

function dm_pmc_transfer_record(transfers)
    names = fieldnames(typeof(transfers.reasons))
    names == DM_PMC_TRANSFER_REASONS || error(
        "reported transfer reasons changed: observed=$names",
    )
    values_by_reason = map(names) do reason
        record = getfield(transfers.reasons, reason)
        (count=record.count, bytes=record.bytes)
    end
    reasons = NamedTuple{names}(values_by_reason)
    sum(record.count for record in values(reasons)) == transfers.count || error(
        "reported transfer reason counts do not sum to the aggregate",
    )
    sum(record.bytes for record in values(reasons)) == transfers.bytes || error(
        "reported transfer reason bytes do not sum to the aggregate",
    )
    return (count=transfers.count, bytes=transfers.bytes, reasons)
end

function dm_pmc_device_allocated_bytes(sampler, device_kind)
    device_kind === :cuda || return missing
    CUDA.synchronize()
    return CUDA.@allocated dm_pmc_synchronized_run!(sampler, :cuda)
end

function dm_pmc_validate_paired_result(result, cell, device_kind)
    length(result) == cell.rounds * cell.round_size || error(
        "benchmark result count does not match its fixed workload",
    )
    diagnostics = dm_pmc_host_diagnostics(result, device_kind)
    all(isfinite, diagnostics.logweights) || error(
        "benchmark produced nonfinite log weights",
    )
    all(isfinite, diagnostics.round_diagnostic_ess) || error(
        "benchmark produced nonfinite per-round diagnostic ESS",
    )
    all(isfinite, result.diagnostics.round_lognormalizers) || error(
        "benchmark produced nonfinite round log normalizers",
    )
    return (;
        total_samples=length(result),
        flattened_weight_concentration_ess=dm_pmc_flattened_weight_concentration_ess(
            diagnostics.logweights,
        ),
        per_round_diagnostic_ess=Tuple(diagnostics.round_diagnostic_ess),
        reported_explicit_transfers=dm_pmc_transfer_record(result.diagnostics.transfers),
        measurement_transfers=diagnostics.measurement_transfers,
        finite_diagnostics=true,
    )
end

function dm_pmc_paired_run(cell, device_kind, seed)
    sampler = dm_pmc_prepare_cell(cell, device_kind, seed)
    result_holder = Ref{Any}()
    benchmark = @benchmarkable $result_holder[] = dm_pmc_synchronized_run!(
        $sampler,
        $device_kind,
    )
    trial = run(
        benchmark;
        samples=1,
        seconds=DM_PMC_BENCHMARK_TRIAL_SECONDS,
        evals=1,
    )
    timing = minimum(trial)
    result = result_holder[]
    validation = dm_pmc_validate_paired_result(result, cell, device_kind)
    seconds = timing.time / 1.0e9
    return merge(
        (;
            seed,
            fixed_initial_population=true,
            warmed_seconds=seconds,
            samples_per_second=validation.total_samples / seconds,
            flattened_weight_concentration_ess_per_second=
                validation.flattened_weight_concentration_ess / seconds,
            host_allocations=timing.allocs,
            host_allocated_bytes=timing.memory,
        ),
        validation,
    )
end

function dm_pmc_summary(values)
    return (minimum=minimum(values), median=median(values), maximum=maximum(values))
end

function dm_pmc_benchmark_cell(cell, device_kind)
    dm_pmc_prepare_cell(cell, device_kind)
    preparation = @benchmarkable dm_pmc_prepare_cell($cell, $device_kind)
    preparation_record = dm_pmc_trial_record(
        dm_pmc_run_preparation_trial(preparation),
    )

    warm_sampler = dm_pmc_prepare_cell(cell, device_kind, DM_PMC_BENCHMARK_SEED)
    dm_pmc_synchronized_run!(warm_sampler, device_kind)
    seeds = ntuple(
        replicate -> DM_PMC_BENCHMARK_SEED + UInt64(replicate - 1),
        DM_PMC_BENCHMARK_REPLICATES,
    )
    paired_runs = Tuple(dm_pmc_paired_run(cell, device_kind, seed) for seed in seeds)
    total_samples = first(paired_runs).total_samples
    all(run -> run.total_samples == total_samples, paired_runs) || error(
        "paired benchmark runs returned different sample counts",
    )
    all(run -> run.reported_explicit_transfers == first(paired_runs).reported_explicit_transfers, paired_runs) || error(
        "paired benchmark runs reported different explicit transfers",
    )
    all(run -> run.measurement_transfers == first(paired_runs).measurement_transfers, paired_runs) || error(
        "paired benchmark runs required different diagnostic transfers",
    )
    warmed_seconds = dm_pmc_summary([run.warmed_seconds for run in paired_runs])
    samples_per_second = dm_pmc_summary([run.samples_per_second for run in paired_runs])
    flattened_ess = dm_pmc_summary([
        run.flattened_weight_concentration_ess for run in paired_runs
    ])
    flattened_ess_per_second = dm_pmc_summary([
        run.flattened_weight_concentration_ess_per_second for run in paired_runs
    ])
    per_round_diagnostic_ess = Tuple(
        dm_pmc_summary([run.per_round_diagnostic_ess[round] for run in paired_runs]) for
        round in 1:cell.rounds
    )
    allocation_sampler = dm_pmc_prepare_cell(cell, device_kind, first(seeds))
    device_allocated_bytes = dm_pmc_device_allocated_bytes(
        allocation_sampler,
        device_kind,
    )
    return (;
        device=device_kind,
        scalar_type=cell.scalar_type,
        bank=cell.bank,
        dimension=cell.dimension,
        proposals=cell.proposals,
        rounds=cell.rounds,
        round_size=cell.round_size,
        total_samples,
        preparation_seconds=preparation_record.median_seconds,
        preparation_host_allocations=preparation_record.host_allocations,
        preparation_host_allocated_bytes=preparation_record.host_allocated_bytes,
        replicate_seeds=seeds,
        paired_runs,
        warmed_seconds=warmed_seconds.median,
        warmed_seconds_variability=(minimum=warmed_seconds.minimum, maximum=warmed_seconds.maximum),
        samples_per_second=samples_per_second.median,
        samples_per_second_variability=(minimum=samples_per_second.minimum, maximum=samples_per_second.maximum),
        flattened_weight_concentration_ess=flattened_ess.median,
        flattened_weight_concentration_ess_variability=(minimum=flattened_ess.minimum, maximum=flattened_ess.maximum),
        flattened_weight_concentration_ess_per_second=flattened_ess_per_second.median,
        flattened_weight_concentration_ess_per_second_variability=(minimum=flattened_ess_per_second.minimum, maximum=flattened_ess_per_second.maximum),
        per_round_diagnostic_ess,
        host_allocations=round(Int, median([run.host_allocations for run in paired_runs])),
        host_allocated_bytes=round(
            Int,
            median([run.host_allocated_bytes for run in paired_runs]),
        ),
        device_allocated_bytes,
        reported_explicit_transfers=first(paired_runs).reported_explicit_transfers,
        measurement_transfers=first(paired_runs).measurement_transfers,
        finite_diagnostics=true,
    )
end

function dm_pmc_benchmark_cases()
    if DM_PMC_BENCHMARK_SMOKE
        return (
            (
                scalar_type=Float32,
                bank=:diagonal,
                dimension=4,
                proposals=4,
                rounds=DM_PMC_BENCHMARK_ROUNDS,
                round_size=DM_PMC_BENCHMARK_ROUND_SIZE,
            ),
            (
                scalar_type=Float64,
                bank=:factor,
                dimension=4,
                proposals=4,
                rounds=DM_PMC_BENCHMARK_ROUNDS,
                round_size=DM_PMC_BENCHMARK_ROUND_SIZE,
            ),
        )
    end
    return Tuple(
        (;
            scalar_type=T,
            bank,
            dimension,
            proposals,
            rounds=DM_PMC_BENCHMARK_ROUNDS,
            round_size=DM_PMC_BENCHMARK_ROUND_SIZE,
        ) for T in DM_PMC_BENCHMARK_TYPES for bank in DM_PMC_BENCHMARK_BANKS for
        dimension in DM_PMC_BENCHMARK_DIMENSIONS for proposals in DM_PMC_BENCHMARK_PROPOSALS
    )
end

function dm_pmc_package_versions()
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

function dm_pmc_environment()
    cuda = DM_PMC_BENCHMARK_CUDA ? (
        gpu=CUDA.name(CUDA.device()),
        capability=CUDA.capability(CUDA.device()),
        driver=CUDA.driver_version(),
        runtime=CUDA.runtime_version(),
    ) : nothing
    return (;
        julia=VERSION,
        cpu=Sys.CPU_NAME,
        cpu_threads=Sys.CPU_THREADS,
        julia_threads=Threads.nthreads(:default),
        cuda,
        packages=dm_pmc_package_versions(),
        seed=DM_PMC_BENCHMARK_SEED,
        smoke=DM_PMC_BENCHMARK_SMOKE,
        rounds=DM_PMC_BENCHMARK_ROUNDS,
        round_size=DM_PMC_BENCHMARK_ROUND_SIZE,
        preparation_trial_samples=DM_PMC_BENCHMARK_PREPARATION_SAMPLES,
        paired_trial_samples=1,
        trial_seconds=DM_PMC_BENCHMARK_TRIAL_SECONDS,
        replicates=DM_PMC_BENCHMARK_REPLICATES,
    )
end

function dm_pmc_benchmark_main()
    devices = Symbol[]
    DM_PMC_BENCHMARK_CPU && push!(devices, :cpu)
    DM_PMC_BENCHMARK_CUDA && push!(devices, :cuda)
    isempty(devices) && error("no benchmark device selected")
    cases = dm_pmc_benchmark_cases()
    rows = [
        dm_pmc_benchmark_cell(cell, device) for device in devices for cell in cases
    ]
    expected_rows = length(devices) * (DM_PMC_BENCHMARK_SMOKE ? 2 : 16)
    length(rows) == expected_rows || error(
        "benchmark matrix is incomplete: observed=$(length(rows)), expected=$expected_rows",
    )
    return (;
        smoke=DM_PMC_BENCHMARK_SMOKE,
        fixed_matrix=!DM_PMC_BENCHMARK_SMOKE,
        source_level_explicit_transfer_accounting=true,
        hidden_cuda_runtime_transfers_instrumented=false,
        flattened_weight_concentration_efficiency_comparative=false,
        flattened_weight_concentration_interpretation=
            :descriptive_paired_observations_only,
        environment=dm_pmc_environment(),
        rows,
    )
end

DM_PMC_BENCHMARK_RESULT = dm_pmc_benchmark_main()
@assert DM_PMC_BENCHMARK_RESULT.smoke == DM_PMC_BENCHMARK_SMOKE
@assert all(row -> row.total_samples == DM_PMC_BENCHMARK_ROUNDS * DM_PMC_BENCHMARK_ROUND_SIZE, DM_PMC_BENCHMARK_RESULT.rows)
@assert all(row -> hasproperty(row, :paired_runs), DM_PMC_BENCHMARK_RESULT.rows)
@assert all(row -> !hasproperty(row, :ess), DM_PMC_BENCHMARK_RESULT.rows)
@assert all(row -> all(run -> run.fixed_initial_population, row.paired_runs), DM_PMC_BENCHMARK_RESULT.rows)
show(stdout, MIME("text/plain"), DM_PMC_BENCHMARK_RESULT)
println()
DM_PMC_BENCHMARK_RESULT
