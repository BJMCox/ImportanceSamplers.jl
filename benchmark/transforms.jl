using BenchmarkTools
using ImportanceSamplers

const IS = ImportanceSamplers
const TRANSFORM_FLOAT_TYPES = (Float32, Float64)
const TRANSFORM_INPUT_PATTERN = (-20, -2, 0, 2, 20)
const TRANSFORM_WORKLOAD_LENGTH = 4_096
const TRANSFORM_BENCHMARK_SAMPLES = "--smoke" in ARGS ? 10 : 100
const TRANSFORM_BENCHMARK_SECONDS = "--smoke" in ARGS ? 0.1 : 1.0
const SIMPLEX_DIMENSIONS = (3, 10, 100, 1_000)

function _transform_inputs(::Type{T}) where {T}
    inputs = Vector{T}(undef, TRANSFORM_WORKLOAD_LENGTH)
    for index in eachindex(inputs)
        inputs[index] = T(TRANSFORM_INPUT_PATTERN[mod1(index, length(TRANSFORM_INPUT_PATTERN))])
    end
    return inputs
end

function _transform_sum(transform, inputs)
    total = zero(eltype(inputs))
    for z in inputs
        x, logabsjac = IS._transform_with_logjac(transform, z)
        total += x + logabsjac
    end
    return total
end

function _transform_trial(transform, inputs)
    return run(
        @benchmarkable _transform_sum($transform, $inputs);
        samples=TRANSFORM_BENCHMARK_SAMPLES,
        seconds=TRANSFORM_BENCHMARK_SECONDS,
    )
end

function _trial_metrics(trial, length)
    estimate = median(trial)
    return (
        time_per_element_ns=estimate.time / length,
        allocations=estimate.allocs,
        allocated_bytes=estimate.memory,
    )
end

function benchmark_scalar_transforms(; repetitions=1)
    return map(TRANSFORM_FLOAT_TYPES) do T
        inputs = _transform_inputs(T)
        positive = [
            _trial_metrics(
                _transform_trial(PositiveTransform(), inputs),
                length(inputs),
            ) for _ in 1:repetitions
        ]
        softplus = [
            _trial_metrics(
                _transform_trial(SoftplusTransform(), inputs),
                length(inputs),
            ) for _ in 1:repetitions
        ]
        (float_type=T, positive, softplus)
    end
end

function _simplex_input(::Type{T}, dimension) where {T}
    return T[T(0.25) * T(sin(index)) for index in 1:(dimension - 1)]
end

function _simplex_forward_trial(transform, input)
    return run(
        @benchmarkable IS._transform_with_logjac($transform, $input) evals = 1;
        samples=TRANSFORM_BENCHMARK_SAMPLES,
        seconds=TRANSFORM_BENCHMARK_SECONDS,
    )
end

function _simplex_inverse_trial(transform, input)
    return run(
        @benchmarkable IS._inverse_with_logjac($transform, $input) evals = 1;
        samples=TRANSFORM_BENCHMARK_SAMPLES,
        seconds=TRANSFORM_BENCHMARK_SECONDS,
    )
end

function _simplex_trial_metrics(trial, dimension)
    estimate = median(trial)
    return (
        time_ns=estimate.time,
        time_per_output_coordinate_ns=estimate.time / dimension,
        allocations=estimate.allocs,
        allocated_bytes=estimate.memory,
    )
end

function benchmark_simplex_transforms(; repetitions=1)
    return [
        begin
            transform = SimplexTransform(dimension)
            unconstrained = _simplex_input(T, dimension)
            simplex, _ = IS._transform_with_logjac(transform, unconstrained)
            forward = [
                _simplex_trial_metrics(
                    _simplex_forward_trial(transform, unconstrained),
                    dimension,
                ) for _ in 1:repetitions
            ]
            inverse = [
                _simplex_trial_metrics(
                    _simplex_inverse_trial(transform, simplex),
                    dimension - 1,
                ) for _ in 1:repetitions
            ]
            (; float_type=T, dimension, forward, inverse)
        end for T in TRANSFORM_FLOAT_TYPES for dimension in SIMPLEX_DIMENSIONS
    ]
end

if abspath(PROGRAM_FILE) == @__FILE__
    (scalar=benchmark_scalar_transforms(), simplex=benchmark_simplex_transforms())
end
