using BenchmarkTools
using DensityInterface
using ImportanceSamplers
using LinearAlgebra
using Random

const SMOKE_MODE = "--smoke" in ARGS
const BENCHMARK_SEED = 0x4e41544956454734
const DIMENSIONS = (2, 8, 32, 128, 512)
const BATCH_SIZES = (1, 1_000, 100_000)
const FLOAT_TYPES = (Float32, Float64)
const MAX_CHUNK_BYTES = 16 * 1024^2
const MAX_BATCH_BYTES = 256 * 1024^2
const REQUESTED_JULIA_THREADS = 10
const REQUESTED_BLAS_THREADS = 10
const OBSERVED_JULIA_THREADS = Threads.nthreads(:default)
const OBSERVED_BLAS_THREADS_BEFORE = BLAS.get_num_threads()
BLAS.set_num_threads(REQUESTED_BLAS_THREADS)
const OBSERVED_BLAS_THREADS = BLAS.get_num_threads()
const TRIAL_SAMPLES = SMOKE_MODE ? 2 : 5
const TRIAL_SECONDS = SMOKE_MODE ? 0.05 : 2.0
const RESULT_PATH = let index = findfirst(==("--output"), ARGS)
    isnothing(index) ? nothing : ARGS[index + 1]
end

struct CachedInverseFactorGaussian{P,M}
    proposal::P
    inverse_factor::M
end

function lower_factor(rng, ::Type{T}, dimension; diagonal_ratio) where {T}
    factor = zeros(T, dimension, dimension)
    for column in axes(factor, 2), row in column:dimension
        factor[row, column] = T(0.05) * randn(rng, T)
    end
    for index in axes(factor, 1)
        fraction = T(index - 1) / T(max(dimension - 1, 1))
        factor[index, index] += exp(log(T(diagonal_ratio)) * fraction)
    end
    return factor
end

function prepare_cached_inverse_factor(location, factor)
    proposal = FactorGaussian(location, factor)
    inverse_factor = Matrix(inv(LowerTriangular(proposal.scale.factor)))
    return CachedInverseFactorGaussian(proposal, inverse_factor)
end

function DensityInterface.logdensityof(candidate::CachedInverseFactorGaussian, sample)
    difference = copy(sample)
    for index in eachindex(difference, candidate.proposal.location)
        difference[index] -= candidate.proposal.location[index]
    end
    standardized = candidate.inverse_factor * difference
    return candidate.proposal.lognormalizer -
           oftype(candidate.proposal.lognormalizer, 0.5) * sum(abs2, standardized)
end

function factor_quadratic_form(proposal, sample)
    difference = copy(sample)
    for index in eachindex(difference, proposal.location)
        difference[index] -= proposal.location[index]
    end
    ldiv!(LowerTriangular(proposal.scale.factor), difference)
    return sum(abs2, difference)
end

function inverse_quadratic_form(candidate, sample)
    difference = copy(sample)
    for index in eachindex(difference, candidate.proposal.location)
        difference[index] -= candidate.proposal.location[index]
    end
    standardized = candidate.inverse_factor * difference
    return sum(abs2, standardized)
end

function density_sum(proposal, sample_chunks)
    total = zero(eltype(first(sample_chunks)))
    for samples in sample_chunks, column in axes(samples, 2)
        total += DensityInterface.logdensityof(proposal, view(samples, :, column))
    end
    return total
end

function distinct_sample_chunks(rng, location, factor, batch_size)
    T = eltype(location)
    total_bytes = length(location) * batch_size * sizeof(T)
    total_bytes <= MAX_BATCH_BYTES || return nothing, total_bytes
    chunk_columns = max(1, MAX_CHUNK_BYTES ÷ (length(location) * sizeof(T)))
    chunks = Matrix{T}[]
    remaining = batch_size
    while remaining > 0
        columns = min(remaining, chunk_columns)
        standardized = randn(rng, T, length(location), columns)
        samples = similar(standardized)
        mul!(samples, factor, standardized)
        for column in axes(samples, 2), row in eachindex(location)
            samples[row, column] += location[row]
        end
        push!(chunks, samples)
        remaining -= columns
    end
    return chunks, total_bytes
end

function direct_errors(proposal, candidate, sample_chunks)
    maximum_relative_quadratic_error = zero(eltype(proposal.location))
    maximum_absolute_logdensity_error = zero(eltype(proposal.location))
    for samples in sample_chunks, column in axes(samples, 2)
        sample = view(samples, :, column)
        solve_quadratic = factor_quadratic_form(proposal, sample)
        inverse_quadratic = inverse_quadratic_form(candidate, sample)
        maximum_relative_quadratic_error = max(
            maximum_relative_quadratic_error,
            abs(inverse_quadratic - solve_quadratic) /
            max(abs(solve_quadratic), eps(eltype(proposal.location))),
        )
        maximum_absolute_logdensity_error = max(
            maximum_absolute_logdensity_error,
            abs(
                DensityInterface.logdensityof(candidate, sample) -
                DensityInterface.logdensityof(proposal, sample),
            ),
        )
    end
    return maximum_relative_quadratic_error, maximum_absolute_logdensity_error
end

function run_trial(benchmark)
    return run(benchmark; samples=TRIAL_SAMPLES, seconds=TRIAL_SECONDS, evals=1)
end

function trial_metrics(trial, work_items)
    estimate = median(trial)
    seconds = estimate.time / 1.0e9
    return (
        time_ns=estimate.time,
        throughput=work_items / seconds,
        allocations=estimate.allocs,
        allocated_bytes=estimate.memory,
    )
end

function format_metrics(metrics)
    return "$(BenchmarkTools.prettytime(metrics.time_ns)); " *
           "$(round(metrics.throughput; sigdigits=5)) samples/s; " *
           "$(metrics.allocations) allocs; $(BenchmarkTools.prettymemory(metrics.allocated_bytes))"
end

function benchmark_case(rng, ::Type{T}, dimension, batch_size) where {T}
    location = randn(rng, T, dimension)
    factor = lower_factor(rng, T, dimension; diagonal_ratio=T(10))
    sample_chunks, sample_bytes = distinct_sample_chunks(rng, location, factor, batch_size)
    isnothing(sample_chunks) && return (skipped=true, sample_bytes)

    solve_preparation = run_trial(@benchmarkable FactorGaussian($location, $factor))
    inverse_preparation = run_trial(@benchmarkable prepare_cached_inverse_factor($location, $factor))
    solve_proposal = FactorGaussian(location, factor)
    inverse_candidate = prepare_cached_inverse_factor(location, factor)
    density_sum(solve_proposal, sample_chunks)
    density_sum(inverse_candidate, sample_chunks)
    solve_warm = run_trial(@benchmarkable density_sum($solve_proposal, $sample_chunks))
    inverse_warm = run_trial(@benchmarkable density_sum($inverse_candidate, $sample_chunks))
    return (
        skipped=false,
        condition_number=cond(factor),
        sample_bytes,
        solve_preparation=trial_metrics(solve_preparation, 1),
        inverse_preparation=trial_metrics(inverse_preparation, 1),
        solve_warm=trial_metrics(solve_warm, batch_size),
        inverse_warm=trial_metrics(inverse_warm, batch_size),
        solve_retained_bytes=Base.summarysize(solve_proposal),
        inverse_retained_bytes=Base.summarysize(inverse_candidate),
    )
end

function error_case(rng, ::Type{T}, dimension, label, diagonal_ratio) where {T}
    location = randn(rng, T, dimension)
    factor = lower_factor(rng, T, dimension; diagonal_ratio)
    sample_chunks, sample_bytes = distinct_sample_chunks(rng, location, factor, min(1_000, 100_000))
    proposal = FactorGaussian(location, factor)
    candidate = prepare_cached_inverse_factor(location, factor)
    quadratic_error, logdensity_error = direct_errors(proposal, candidate, sample_chunks)
    return (label, cond(factor), sample_bytes, quadratic_error, logdensity_error)
end

function benchmark_command()
    output = isnothing(RESULT_PATH) ? "" : " --output $(RESULT_PATH)"
    smoke = SMOKE_MODE ? " --smoke" : ""
    return "OPENBLAS_NUM_THREADS=$(REQUESTED_BLAS_THREADS) julia --threads=$(REQUESTED_JULIA_THREADS) " *
           "--project=benchmark benchmark/native_gaussian.jl$(smoke)$(output)"
end

function benchmark_output(io)
    println(io, "# Native Gaussian factor benchmark")
    println(io)
    println(io, "- command: `", benchmark_command(), "`")
    println(io, "- seed: `", BENCHMARK_SEED, "`")
    println(io, "- Julia: `", VERSION, "`")
    println(io, "- CPU: `", Sys.CPU_NAME, "`")
    println(io, "- CPU threads: `", Sys.CPU_THREADS, "`")
    println(io, "- requested Julia / observed Julia threads: `", REQUESTED_JULIA_THREADS,
        " / ", OBSERVED_JULIA_THREADS, "`")
    println(io, "- requested BLAS / observed BLAS threads: `", REQUESTED_BLAS_THREADS,
        " / ", OBSERVED_BLAS_THREADS, "` (before script setup: ", OBSERVED_BLAS_THREADS_BEFORE, ")")
    println(io, "- BLAS: `", repr(BLAS.get_config()), "`")
    println(io, "- distinct samples: chunks at most `", MAX_CHUNK_BYTES,
        "` bytes; logical batches over `", MAX_BATCH_BYTES, "` bytes are skipped")
    println(io)
    println(io, "## Throughput, allocation, and retained state")
    println(io)
    println(io, "| type | dimension | batch | actual cond(L) | distinct samples | solve preparation | inverse preparation | solve public density | inverse candidate density | solve retained | inverse retained |")
    println(io, "| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: |")
    rng = Xoshiro(BENCHMARK_SEED)
    for T in FLOAT_TYPES, dimension in DIMENSIONS, batch_size in BATCH_SIZES
        metrics = benchmark_case(rng, T, dimension, batch_size)
        if metrics.skipped
            println(io, "| ", T, " | ", dimension, " | ", batch_size,
                " | — | skipped: distinct samples require ", metrics.sample_bytes,
                " bytes, exceeding the ", MAX_BATCH_BYTES, " byte safety cap | — | — | — | — | — | — |")
            continue
        end
        println(
            io,
            "| ", T, " | ", dimension, " | ", batch_size, " | ",
            metrics.condition_number, " | ", metrics.sample_bytes, " B | ",
            format_metrics(metrics.solve_preparation), " | ",
            format_metrics(metrics.inverse_preparation), " | ",
            format_metrics(metrics.solve_warm), " | ",
            format_metrics(metrics.inverse_warm), " | ",
            metrics.solve_retained_bytes, " B | ", metrics.inverse_retained_bytes, " B |",
        )
    end
    println(io)
    println(io, "## Numerical comparison")
    println(io)
    println(io, "Quadratic forms are compared before the normalizer; log-density error is separate.")
    println(io)
    println(io, "| type | dimension | fixture | actual cond(L) | distinct samples | max relative quadratic-form error | max absolute log-density error |")
    println(io, "| --- | ---: | --- | ---: | ---: | ---: | ---: |")
    for T in FLOAT_TYPES, dimension in DIMENSIONS,
        (label, diagonal_ratio) in (("well-targeted", T(10)),
            ("ill-targeted", T == Float32 ? T(1e4) : T(1e8)))
        label, condition_number, sample_bytes, quadratic_error, logdensity_error =
            error_case(rng, T, dimension, label, diagonal_ratio)
        println(io, "| ", T, " | ", dimension, " | ", label, " | ", condition_number,
            " | ", sample_bytes, " B | ", quadratic_error, " | ", logdensity_error, " |")
    end
end

function main()
    if isnothing(RESULT_PATH)
        benchmark_output(stdout)
    else
        open(RESULT_PATH, "w") do io
            benchmark_output(io)
        end
    end
    return nothing
end

main()
