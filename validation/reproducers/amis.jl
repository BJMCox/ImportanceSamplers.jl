using ImportanceSamplers
using LinearAlgebra
using Pkg
using Random
using Test

include(joinpath(@__DIR__, "..", "amis_capabilities.jl"))

const AMIS_CPU_ORACLE_COMMAND =
    "include(\"reproducers/amis.jl\") in the validation project"
const AMIS_CPU_ORACLE_SEED = 0x616d69736f726163

mutable struct LiteralNormalRNG{T<:AbstractFloat} <: Random.AbstractRNG
    batches::Vector{Vector{T}}
    index::Int
end

LiteralNormalRNG(batches::Vector{Vector{T}}) where {T<:AbstractFloat} =
    LiteralNormalRNG{T}(batches, 1)

function Random.randn!(rng::LiteralNormalRNG, destination::AbstractArray)
    source = rng.batches[rng.index]
    length(source) == length(destination) || throw(
        DimensionMismatch("literal normal batch does not match the prepared buffer"),
    )
    copyto!(destination, source)
    rng.index += 1
    return destination
end

struct LiteralGaussian{T,L,F}
    location::L
    factor::F
    lognormalizer::T
end

function literal_gaussian(location::T, scale::T) where {T<:AbstractFloat}
    return LiteralGaussian(location, scale, -T(0.5) * log(T(2pi)) - log(scale))
end

function literal_gaussian(location::Vector{T}, factor::Matrix{T}) where {T<:AbstractFloat}
    lognormalizer = -T(0.5) * T(length(location)) * log(T(2pi))
    for coordinate in eachindex(location)
        lognormalizer -= log(factor[coordinate, coordinate])
    end
    return LiteralGaussian(location, factor, lognormalizer)
end

literal_dimension(proposal::LiteralGaussian{T,T,T}) where {T} = 1
literal_dimension(proposal::LiteralGaussian) = length(proposal.location)

function literal_sample(proposal::LiteralGaussian{T,T,T}, normals, offset) where {T}
    return proposal.location + proposal.factor * normals[offset]
end

function literal_sample(proposal::LiteralGaussian, normals, offset)
    dimension = literal_dimension(proposal)
    sample = similar(proposal.location)
    for row in 1:dimension
        value = proposal.location[row]
        for column in 1:row
            value += proposal.factor[row, column] * normals[offset + column - 1]
        end
        sample[row] = value
    end
    return sample
end

function literal_gaussian_logdensity(
    proposal::LiteralGaussian{T,T,T},
    sample,
) where {T}
    standardized = (sample - proposal.location) / proposal.factor
    return proposal.lognormalizer - T(0.5) * abs2(standardized)
end

function literal_gaussian_logdensity(proposal::LiteralGaussian{T}, sample) where {T}
    dimension = literal_dimension(proposal)
    standardized = zeros(T, dimension)
    for row in 1:dimension
        value = sample[row] - proposal.location[row]
        for column in 1:(row - 1)
            value -= proposal.factor[row, column] * standardized[column]
        end
        standardized[row] = value / proposal.factor[row, row]
    end
    squared_radius = zero(T)
    for coordinate in eachindex(standardized)
        squared_radius += abs2(standardized[coordinate])
    end
    return proposal.lognormalizer - T(0.5) * squared_radius
end

function literal_logaddexp(left::T, right::T) where {T}
    maximum_value = max(left, right)
    return maximum_value + log(exp(left - maximum_value) + exp(right - maximum_value))
end

function literal_normalized_weights(logweights::Vector{T}) where {T}
    maximum_logweight = maximum(logweights)
    weights = exp.(logweights .- maximum_logweight)
    total = zero(T)
    for weight in weights
        total += weight
    end
    weights ./= total
    return weights
end

function literal_cholesky(covariance::Matrix{T}) where {T}
    dimension = size(covariance, 1)
    factor = zeros(T, dimension, dimension)
    for column in 1:dimension, row in column:dimension
        value = covariance[row, column]
        for inner in 1:(column - 1)
            value -= factor[row, inner] * factor[column, inner]
        end
        factor[row, column] = row == column ? sqrt(value) : value / factor[column, column]
    end
    return factor
end

function literal_fit(
    samples::Vector{T},
    logweights::Vector{T},
    previous::LiteralGaussian{T,T,T},
) where {T}
    weights = literal_normalized_weights(logweights)
    mean = zero(T)
    for index in eachindex(samples)
        mean += weights[index] * samples[index]
    end
    variance = zero(T)
    for index in eachindex(samples)
        variance += weights[index] * abs2(samples[index] - mean)
    end
    variance += sqrt(eps(T)) * abs2(previous.factor)
    return literal_gaussian(mean, sqrt(variance))
end

function literal_fit(
    samples::Matrix{T},
    logweights::Vector{T},
    previous::LiteralGaussian{T},
) where {T}
    weights = literal_normalized_weights(logweights)
    dimension, sample_count = size(samples)
    mean = zeros(T, dimension)
    for sample_index in 1:sample_count, coordinate in 1:dimension
        mean[coordinate] += weights[sample_index] * samples[coordinate, sample_index]
    end
    covariance = zeros(T, dimension, dimension)
    for sample_index in 1:sample_count, column in 1:dimension, row in 1:dimension
        covariance[row, column] += weights[sample_index] *
                                   (samples[row, sample_index] - mean[row]) *
                                   (samples[column, sample_index] - mean[column])
    end
    previous_trace = zero(T)
    for column in 1:dimension, row in column:dimension
        previous_trace += abs2(previous.factor[row, column])
    end
    ridge = sqrt(eps(T)) * previous_trace / T(dimension)
    for coordinate in 1:dimension
        covariance[coordinate, coordinate] += ridge
    end
    return literal_gaussian(mean, literal_cholesky(covariance))
end

function literal_target(sample::T, mean::T, factor::T) where {T}
    return literal_gaussian_logdensity(literal_gaussian(mean, factor), sample)
end

function literal_target(sample::AbstractVector{T}, mean, factor) where {T}
    return literal_gaussian_logdensity(
        literal_gaussian(Vector{T}(mean), Matrix{T}(factor)),
        sample,
    )
end

function literal_amis(initial, schedule, normal_batches, target)
    T = typeof(initial.lognormalizer)
    dimension = literal_dimension(initial)
    total_samples = sum(schedule)
    samples = dimension == 1 ? Vector{T}(undef, total_samples) :
              Matrix{T}(undef, dimension, total_samples)
    logtargets = Vector{T}(undef, total_samples)
    lognumerators = fill(T(-Inf), total_samples)
    logweights = Vector{T}(undef, total_samples)
    round_ids = Vector{Int}(undef, total_samples)
    history = LiteralGaussian[initial]
    first_sample = 1

    for round in eachindex(schedule)
        round_size = schedule[round]
        last_sample = first_sample + round_size - 1
        proposal = history[round]
        normals = normal_batches[round]
        for sample_index in first_sample:last_sample
            normal_offset = dimension * (sample_index - first_sample) + 1
            sample = literal_sample(proposal, normals, normal_offset)
            if dimension == 1
                samples[sample_index] = sample
            else
                samples[:, sample_index] .= sample
            end
            logtargets[sample_index] = target(sample)
            round_ids[sample_index] = round
        end

        logcount = log(T(round_size))
        for sample_index in 1:(first_sample - 1)
            sample = dimension == 1 ? samples[sample_index] : view(samples, :, sample_index)
            term = logcount + literal_gaussian_logdensity(proposal, sample)
            lognumerators[sample_index] = literal_logaddexp(
                lognumerators[sample_index],
                term,
            )
        end
        for sample_index in first_sample:last_sample
            sample = dimension == 1 ? samples[sample_index] : view(samples, :, sample_index)
            for proposal_round in 1:round
                term = log(T(schedule[proposal_round])) +
                       literal_gaussian_logdensity(history[proposal_round], sample)
                lognumerators[sample_index] = literal_logaddexp(
                    lognumerators[sample_index],
                    term,
                )
            end
        end
        for sample_index in 1:last_sample
            logweights[sample_index] = logtargets[sample_index] -
                                       lognumerators[sample_index] +
                                       log(T(last_sample))
        end
        push!(
            history,
            literal_fit(
                dimension == 1 ? samples[1:last_sample] : samples[:, 1:last_sample],
                logweights[1:last_sample],
                proposal,
            ),
        )
        first_sample = last_sample + 1
    end
    return (; samples, logtargets, lognumerators, logweights, round_ids, history)
end

function oracle_case(::Type{T}, geometry, schedule) where {T}
    initial = if geometry === :scalar
        literal_gaussian(T(-1.25), T(1.8))
    else
        literal_gaussian(
            T[-1.0, 0.75, 1.5],
            T[1.4 0 0; -0.2 1.2 0; 0.1 0.25 1.35],
        )
    end
    dimension = literal_dimension(initial)
    target_mean = geometry === :scalar ? T(0.75) : T[0.5, -0.25, 1.0]
    target_factor = geometry === :scalar ? T(1.1) :
                    T[0.8 0 0; 0.2 1.1 0; -0.1 0.25 0.9]
    target = sample -> literal_target(sample, target_mean, target_factor)
    source = Xoshiro(AMIS_CPU_ORACLE_SEED + UInt(sizeof(T)) + UInt(dimension) + UInt(sum(schedule)))
    capacity = dimension * maximum(schedule)
    normal_batches = [randn(source, T, capacity) for _ in eachindex(schedule)]
    package_proposal = geometry === :scalar ?
                       SphericalGaussian(initial.location, initial.factor) :
                       FactorGaussian(copy(initial.location), copy(initial.factor))
    sampler = prepare_sampler(
        LiteralNormalRNG(deepcopy(normal_batches)),
        target,
        AMIS(package_proposal; rounds=length(schedule), round_size=schedule);
        threaded=false,
    )
    result = importance_sample!(sampler)
    oracle = literal_amis(initial, schedule, normal_batches, target)
    return (; result, learned=current_proposal(sampler), oracle, target_mean, target_factor)
end

function equation_tolerance(::Type{Float32}, dimension, rounds)
    return 2048 * max(dimension, rounds) * eps(Float32)
end

function equation_tolerance(::Type{Float64}, dimension, rounds)
    return 4096 * max(dimension, rounds) * eps(Float64)
end

function validate_oracle_case(::Type{T}, geometry, schedule) where {T}
    case = oracle_case(T, geometry, schedule)
    result = case.result
    oracle = case.oracle
    dimension = geometry === :scalar ? 1 : length(case.target_mean)
    tolerance = equation_tolerance(T, dimension, length(schedule))
    @test result.samples ≈ oracle.samples rtol=tolerance atol=tolerance
    @test result.logweights ≈ oracle.logweights rtol=tolerance atol=tolerance
    @test result.provenance.round == oracle.round_ids
    final = oracle.history[end]
    if geometry === :scalar
        @test case.learned.location ≈ final.location rtol=tolerance atol=tolerance
        @test case.learned.scale.scale ≈ final.factor rtol=tolerance atol=tolerance
    else
        @test case.learned.location ≈ final.location rtol=tolerance atol=tolerance
        @test case.learned.scale.factor ≈ final.factor rtol=tolerance atol=tolerance
    end
    @test result.diagnostics.target_evaluations == sum(schedule)
    @test result.diagnostics.proposal_evaluations == length(schedule) * sum(schedule)
    recomputed_logweights = literal_final_logweights(
        result.samples,
        sample -> literal_target(sample, case.target_mean, case.target_factor),
        oracle.history,
        schedule,
    )
    @test result.logweights ≈ recomputed_logweights rtol=tolerance atol=tolerance
    analytic = validate_analytic_estimator(case, geometry)
    return (type=T, geometry, schedule=Tuple(schedule), tolerance, analytic)
end

function literal_final_logweights(samples, target, history, schedule)
    T = typeof(first(history).lognormalizer)
    dimension = literal_dimension(first(history))
    total_samples = sum(schedule)
    logweights = Vector{T}(undef, total_samples)
    for sample_index in 1:total_samples
        sample = dimension == 1 ? samples[sample_index] : view(samples, :, sample_index)
        lognumerator = T(-Inf)
        for round in eachindex(schedule)
            term = log(T(schedule[round])) +
                   literal_gaussian_logdensity(history[round], sample)
            lognumerator = literal_logaddexp(lognumerator, term)
        end
        logweights[sample_index] = target(sample) -
                                   lognumerator +
                                   log(T(total_samples))
    end
    return logweights
end

function literal_weighted_moments(samples::Vector{T}, logweights) where {T}
    weights = literal_normalized_weights(logweights)
    mean = zero(T)
    for index in eachindex(samples)
        mean += weights[index] * samples[index]
    end
    variance = zero(T)
    for index in eachindex(samples)
        variance += weights[index] * abs2(samples[index] - mean)
    end
    return mean, variance
end

function literal_weighted_moments(samples::Matrix{T}, logweights) where {T}
    weights = literal_normalized_weights(logweights)
    dimension, sample_count = size(samples)
    mean = zeros(T, dimension)
    for sample_index in 1:sample_count, coordinate in 1:dimension
        mean[coordinate] += weights[sample_index] * samples[coordinate, sample_index]
    end
    covariance = zeros(T, dimension, dimension)
    for sample_index in 1:sample_count, column in 1:dimension, row in 1:dimension
        covariance[row, column] += weights[sample_index] *
                                   (samples[row, sample_index] - mean[row]) *
                                   (samples[column, sample_index] - mean[column])
    end
    return mean, covariance
end

function validate_analytic_estimator(case, geometry)
    T = eltype(case.result.logweights)
    mean, covariance = literal_weighted_moments(
        case.result.samples,
        case.result.logweights,
    )
    expected_covariance = geometry === :scalar ? abs2(case.target_factor) :
                          case.target_factor * transpose(case.target_factor)
    # These deterministic simulation bounds support the exact equation checks;
    # they are not the AMIS correctness oracle.
    mean_tolerance = T(0.08)
    covariance_tolerance = T(0.15)
    normalizer_tolerance = T(0.03)
    @test mean ≈ case.target_mean atol=mean_tolerance rtol=mean_tolerance
    @test covariance ≈ expected_covariance atol=covariance_tolerance rtol=covariance_tolerance
    observed_lognormalizer = lognormalizer(case.result)
    @test observed_lognormalizer ≈ zero(T) atol=normalizer_tolerance
    return (;
        mean,
        covariance,
        expected_mean=case.target_mean,
        expected_covariance,
        observed_lognormalizer,
        expected_lognormalizer=zero(T),
        mean_tolerance,
        covariance_tolerance,
        normalizer_tolerance,
        role=:supporting_simulation_evidence,
    )
end

function scaling_proposal(::Type{T}, dimension) where {T}
    dimension == 1 && return SphericalGaussian(T(-0.5), T(1.4))
    factor = zeros(T, dimension, dimension)
    for coordinate in 1:dimension
        factor[coordinate, coordinate] = T(1.2 + 0.01coordinate)
        coordinate > 1 && (factor[coordinate, coordinate - 1] = T(0.05))
    end
    return FactorGaussian(fill(T(-0.5), dimension), factor)
end

function validate_scaling_case(rounds, dimension)
    T = Float64
    schedule = [17 + 3round for round in 1:rounds]
    total_samples = sum(schedule)
    proposal = scaling_proposal(T, dimension)
    target = dimension == 1 ?
             (sample -> -T(0.5) * abs2(sample) - T(0.5) * log(T(2pi))) :
             (sample -> -T(0.5) * sum(abs2, sample) -
                        T(0.5) * T(dimension) * log(T(2pi)))
    sampler = prepare_sampler(
        Xoshiro(AMIS_CPU_ORACLE_SEED + UInt(rounds + dimension)),
        target,
        AMIS(proposal; rounds, round_size=schedule);
        threaded=false,
    )
    state = getproperty(sampler, :method_state)
    history = getproperty(state, :history)
    workspace = getproperty(state, :workspace)
    history_elements = length(getproperty(history, :means)) +
                       length(getproperty(history, :lognormalizers)) +
                       length(
                           hasproperty(history, :scales) ?
                           getproperty(history, :scales) :
                           getproperty(history, :factors),
                       )
    expected_history_elements = rounds * (dimension^2 + dimension + 1)
    accumulator_elements = sum(
        length(getproperty(workspace, field)) for
        field in (:logtargets, :lognumerators, :logweights, :normalized_weights)
    )
    adaptation_scratch_elements = length(getproperty(workspace, :centered_scaled))
    @test history_elements == expected_history_elements
    @test accumulator_elements == 4total_samples
    @test adaptation_scratch_elements == dimension * total_samples
    @test length(getproperty(workspace, :samples)) == dimension * total_samples

    result = importance_sample!(sampler)
    @test result.diagnostics.proposal_evaluations == rounds * total_samples
    @test result.diagnostics.target_evaluations == total_samples
    return (;
        rounds,
        dimension,
        schedule=Tuple(schedule),
        total_samples,
        proposal_evaluations=result.diagnostics.proposal_evaluations,
        history=(
            elements=history_elements,
            bytes=history_elements * sizeof(T),
            order=:rounds_times_dimension_squared,
        ),
        accumulators=(
            elements=accumulator_elements,
            bytes=accumulator_elements * sizeof(T),
            order=:total_samples,
        ),
        adaptation_scratch=(
            arrays=1,
            elements=adaptation_scratch_elements,
            bytes=adaptation_scratch_elements * sizeof(T),
            shape=(dimension, total_samples),
        ),
    )
end

struct CapabilityTarget{T<:AbstractFloat} end

function (::CapabilityTarget{T})(sample)::T where {T}
    radius = sample isa Number ? abs2(sample) : sum(abs2, sample)
    return -T(0.5) * radius
end

function validate_capability_rows()
    @test Tuple(row.label for row in AMIS_CAPABILITY_ROWS) == (
        :float32_scalar,
        :float64_scalar,
        :float32_factor,
        :float64_factor,
    )
    @test AMIS_BENCHMARK_ROWS == (
        (:cpu, Float32, :scalar),
        (:cpu, Float64, :factor),
        (:cuda, Float32, :factor),
        (:cuda, Float64, :factor),
    )
    for row in AMIS_CAPABILITY_ROWS
        proposal = row.proposal(row.type)
        sampler = prepare_sampler(
            Xoshiro(AMIS_CPU_ORACLE_SEED),
            CapabilityTarget{row.type}(),
            AMIS(proposal; rounds=2, round_size=[7, 9]);
            threaded=false,
        )
        @test length(importance_sample!(sampler)) == 16
    end
    return (
        execution=Tuple((row.label, row.cpu) for row in AMIS_CAPABILITY_ROWS),
        schedules=Tuple(row.label for row in AMIS_SCHEDULE_CAPABILITY_ROWS),
        benchmark_rows=AMIS_BENCHMARK_ROWS,
    )
end

function environment_record()
    root = normpath(joinpath(@__DIR__, "..", ".."))
    wanted = Set(("ImportanceSamplers", "LogExpFunctions", "MLDataDevices"))
    packages = sort!(
        [
            (dependency.name, something(dependency.version, "unversioned")) for
            dependency in values(Pkg.dependencies()) if dependency.name in wanted
        ];
        by=first,
    )
    return (;
        commit=readchomp(`git -C $root rev-parse HEAD`),
        julia=VERSION,
        cpu=Sys.CPU_NAME,
        cpu_threads=Sys.CPU_THREADS,
        julia_threads=Threads.nthreads(:default),
        packages,
    )
end

function main()
    cases = (
        (Float32, :scalar, [257, 257, 257]),
        (Float64, :scalar, [257, 370, 483]),
        (Float32, :factor, [257, 370, 483]),
        (Float64, :factor, [257, 257, 257]),
    )
    rows = map(case -> validate_oracle_case(case...), cases)
    scaling = Tuple(
        validate_scaling_case(rounds, dimension) for
        rounds in AMIS_SCALING_ROUNDS for dimension in AMIS_SCALING_DIMENSIONS
    )
    return (;
        command=AMIS_CPU_ORACLE_COMMAND,
        environment=environment_record(),
        seed=AMIS_CPU_ORACLE_SEED,
        exact_equation_oracle=true,
        analytic_checks=:supporting_simulation_evidence,
        rows,
        scaling,
        capabilities=validate_capability_rows(),
    )
end

main()
