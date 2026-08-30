using BenchmarkTools
using ImportanceSamplers
using LinearAlgebra
using Random

"--cpu-only" in ARGS || error("first_order_gramis.jl requires --cpu-only")

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

const GRAMIS_BENCHMARK_SMOKE = "--smoke" in ARGS
const GRAMIS_BENCHMARK_SCALING = "--scaling" in ARGS
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
const GRAMIS_BENCHMARK_EXECUTIONS = GRAMIS_BENCHMARK_SERIAL_ONLY ?
                                    (:serial,) :
                                    GRAMIS_BENCHMARK_THREADED_ONLY ?
                                    (:threaded,) : (:serial, :threaded)
const GRAMIS_BENCHMARK_SAMPLES = GRAMIS_BENCHMARK_SMOKE ? 3 :
                                 GRAMIS_BENCHMARK_EFFECTIVE_ROUND_SIZE === nothing ?
                                 3 :
                                 GRAMIS_BENCHMARK_EFFECTIVE_ROUND_SIZE <= 65_536 ?
                                 7 : 5

struct GRAMISBenchmarkTarget{T<:AbstractFloat} end

@inline function (::GRAMISBenchmarkTarget{T})(sample) where {T}
    return -T(0.5) * sum(abs2, sample)
end

function gram_is_benchmark_gradient!(destination, sample)
    @inbounds for index in eachindex(destination, sample)
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
        scalar_type=T,
        dimension,
        proposals=proposal_count,
        tempering,
        threaded=execution === :threaded,
        rounds=2,
        round_size,
        covariance_ess_threshold=threshold,
    )
end

function gram_is_prepare_benchmark(cell, seed)
    target = LogTarget(
        GRAMISBenchmarkTarget{cell.scalar_type}();
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
    return prepare_sampler(
        Xoshiro(seed),
        target,
        algorithm;
        threaded=cell.threaded,
    )
end

function gram_is_validate_benchmark_result(result, cell)
    expected_samples = cell.rounds * cell.round_size
    length(result) == expected_samples || error("benchmark sample count changed")
    all(isfinite, result.logweights) || error("benchmark produced nonfinite weights")
    powers = result.diagnostics.tempering_powers
    if cell.tempering === :inactive
        all(isone, powers) || error("inactive tempering cell invoked tempering")
    else
        any(<(one(cell.scalar_type)), powers) ||
            error("active tempering cell did not invoke tempering")
    end
    return expected_samples
end

function gram_is_benchmark_run(cell, seed)
    result_holder = Ref{Any}()
    benchmark = @benchmarkable begin
        $result_holder[] = importance_sample!(sampler)
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
    total_samples = gram_is_validate_benchmark_result(result_holder[], cell)
    seconds = estimate.time / 1.0e9
    seconds_range = (fastest.time / 1.0e9, slowest.time / 1.0e9)
    weights = normalized_weights(result_holder[])
    ess = inv(sum(abs2, weights))
    return (
        seconds,
        seconds_range,
        samples_per_second=total_samples / seconds,
        samples_per_second_range=(
            total_samples / seconds_range[2],
            total_samples / seconds_range[1],
        ),
        ess,
        ess_per_second=ess / seconds,
        ess_per_second_range=(
            ess / seconds_range[2],
            ess / seconds_range[1],
        ),
        allocations=estimate.allocs,
        allocated_bytes=estimate.memory,
    )
end

function gram_is_benchmark_cell(cell)
    seed = GRAMIS_BENCHMARK_SEED + UInt64(1)
    warm_sampler = gram_is_prepare_benchmark(cell, seed)
    gram_is_validate_benchmark_result(importance_sample!(warm_sampler), cell)
    run = gram_is_benchmark_run(cell, seed)
    return merge(
        cell,
        (
            julia_threads=Threads.nthreads(:default),
            benchmark_samples=GRAMIS_BENCHMARK_SAMPLES,
            run...,
        ),
    )
end

function gram_is_benchmark_main()
    configurations = GRAMIS_BENCHMARK_SCALING ?
                     ((4, 4), (16, 16), (32, 16)) :
                     Tuple(
                         (dimension, proposals) for
                         dimension in GRAMIS_BENCHMARK_DIMENSIONS for
                         proposals in GRAMIS_BENCHMARK_PROPOSALS
                     )
    cells = (
        gram_is_benchmark_cell(T, dimension, proposals, tempering, execution) for
        T in GRAMIS_BENCHMARK_TYPES for
        (dimension, proposals) in configurations for
        tempering in GRAMIS_BENCHMARK_TEMPERING for
        execution in GRAMIS_BENCHMARK_EXECUTIONS
    )
    rows = [gram_is_benchmark_cell(cell) for cell in cells]
    expected_rows = length(GRAMIS_BENCHMARK_TYPES) * length(configurations) *
                    length(GRAMIS_BENCHMARK_TEMPERING) *
                    length(GRAMIS_BENCHMARK_EXECUTIONS)
    length(rows) == expected_rows || error("benchmark matrix is incomplete")
    return (
        environment=(
            julia=VERSION,
            cpu=Sys.CPU_NAME,
            cpu_threads=Sys.CPU_THREADS,
            julia_threads=Threads.nthreads(:default),
        ),
        smoke=GRAMIS_BENCHMARK_SMOKE,
        round_size_override=GRAMIS_BENCHMARK_EFFECTIVE_ROUND_SIZE,
        tier=GRAMIS_BENCHMARK_EFFECTIVE_ROUND_SIZE === nothing ? :overhead :
             GRAMIS_BENCHMARK_EFFECTIVE_ROUND_SIZE == 8_192 ? :medium :
             GRAMIS_BENCHMARK_EFFECTIVE_ROUND_SIZE == 65_536 ? :large :
             GRAMIS_BENCHMARK_EFFECTIVE_ROUND_SIZE == 262_144 ?
             :production_scale :
             GRAMIS_BENCHMARK_EFFECTIVE_ROUND_SIZE == 1_048_576 ?
             :million_scale : :custom,
        rows,
    )
end

const GRAMIS_BENCHMARK_RESULT = gram_is_benchmark_main()
show(stdout, MIME("text/plain"), GRAMIS_BENCHMARK_RESULT)
println()
GRAMIS_BENCHMARK_RESULT
