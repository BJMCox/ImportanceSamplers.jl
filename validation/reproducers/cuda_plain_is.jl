using CUDA
using DensityInterface
using ImportanceSamplers
using MLDataDevices
using Pkg
using Random
using Test

include(joinpath(@__DIR__, "..", "cuda_plain_is_support.jl"))
include(joinpath(@__DIR__, "..", "cuda_plain_is_capabilities.jl"))

const VALIDATION_SEED = 0x6e61746976656770
const VALIDATION_SAMPLES = 65_536
const FAILURE_SAMPLES = 4_096
const ONE_THREAD_LAUNCH_MODE = "--one-thread-launch" in ARGS
const VALIDATION_COMMAND =
    "Kaimon ex: include(\"reproducers/cuda_plain_is.jl\") in the validation project"

struct GenericGaussian{T}
    location::T
    scale::T
end

struct UnadaptedCUDAContext{A}
    shift::A
end

struct AdaptedCUDAContext{A}
    shift::A
end

ImportanceSamplers.Adapt.@adapt_structure AdaptedCUDAContext

struct AdaptableCUDAFunction{A<:AbstractVector} <: Function
    offset::A
end

ImportanceSamplers.Adapt.@adapt_structure AdaptableCUDAFunction

@inline function (target::AdaptableCUDAFunction)(sample, context)
    return context.shift[1] + target.offset[1] - abs2(sample) / 2
end

@inline function contextual_scalar_logtarget(sample, context)::Float32
    return context.shift[1] - abs2(sample) / 2
end

function Random.rand(rng::Random.AbstractRNG, proposal::GenericGaussian{T}) where {T}
    return proposal.location + proposal.scale * randn(rng, T)
end

@inline function DensityInterface.logdensityof(proposal::GenericGaussian{T}, sample) where {T}
    standardized = (sample - proposal.location) / proposal.scale
    return -T(0.5) * abs2(standardized) - log(proposal.scale) -
           T(0.5) * log(T(2) * T(pi))
end

@inline function vector_gaussian_logtarget(sample, p)
    T = eltype(sample)
    total = zero(T)
    for index in eachindex(sample)
        total += gaussian_logdensity(sample[index], p.location[index], p.scale[index])
    end
    return total
end

@inline function factor_gaussian_logtarget(sample, p)
    first_standardized = (sample[1] - p.location[1]) / p.factor[1, 1]
    second_standardized = (
        sample[2] - p.location[2] - p.factor[2, 1] * first_standardized
    ) / p.factor[2, 2]
    return p.lognormalizer[1] -
           typeof(first_standardized)(0.5) *
           (abs2(first_standardized) + abs2(second_standardized))
end

@inline function flat_layout_logtarget(sample, p)
    T = eltype(sample.free)
    lower_coordinate = log(sample.lower - p.lower[1])
    upper_coordinate = log(p.upper[1] - sample.upper)
    squared_radius = abs2(sample.free[1]) + abs2(sample.free[2]) +
                     abs2(lower_coordinate) + abs2(upper_coordinate)
    base = -T(2) * log(T(2) * T(pi)) - T(0.5) * squared_radius
    return base - lower_coordinate - upper_coordinate
end

@inline function positive_gaussian_logtarget(sample, p)
    coordinate = log(sample)
    return gaussian_logdensity(coordinate, p.location[1], p.scale[1]) - coordinate
end

@inline function softplus_gaussian_logtarget(sample, p)
    coordinate = sample + log(-expm1(-sample))
    return gaussian_logdensity(coordinate, p.location[1], p.scale[1]) -
           coordinate + sample
end

@inline function lower_bounded_gaussian_logtarget(sample, p)
    coordinate = log(sample - p.lower[1])
    return gaussian_logdensity(coordinate, p.location[1], p.scale[1]) - coordinate
end

@inline function upper_bounded_gaussian_logtarget(sample, p)
    coordinate = log(p.upper[1] - sample)
    return gaussian_logdensity(coordinate, p.location[1], p.scale[1]) - coordinate
end

@inline function interval_gaussian_logtarget(sample, p)
    T = typeof(sample)
    lower_distance = sample - p.lower[1]
    upper_distance = p.upper[1] - sample
    coordinate = log(lower_distance) - log(upper_distance)
    logabsjac = log(lower_distance) + log(upper_distance) -
                log(p.upper[1] - p.lower[1])
    return -T(0.5) * abs2(coordinate) - T(0.5) * log(T(2) * T(pi)) -
           logabsjac
end

@inline function simplex_gaussian_logtarget(sample, p)
    T = eltype(sample)
    dimension = length(sample)
    log_sum = zero(T)
    for index in eachindex(sample)
        log_sum += log(sample[index])
    end
    mean_log = log_sum / T(dimension)
    inverse_root = inv(sqrt(T(dimension)))
    transpose_coefficient = inverse_root / (one(T) - inverse_root)
    last_centered = log(sample[dimension]) - mean_log
    squared_radius = zero(T)
    for index in 1:(dimension - 1)
        coordinate = log(sample[index]) - mean_log +
                     transpose_coefficient * last_centered
        squared_radius += abs2(coordinate)
    end
    base = -T(0.5) * squared_radius -
           T(0.5) * T(dimension - 1) * log(T(2) * T(pi))
    logabsjac = T(0.5) * log(T(dimension)) + log_sum
    return base - logabsjac
end

@inline zero_logtarget(sample::Number) = zero(sample)
@inline zero_logtarget(sample::AbstractVector) = zero(eltype(sample))

function caught(f)
    try
        f()
    catch error
        return error
    end
    return nothing
end

function transfer_tuple(result)
    transfers = result.diagnostics.transfers
    return transfers.count, transfers.bytes
end

scalar_summary(samples) = sum(samples) / length(samples)
vector_summary(samples) = vec(sum(samples; dims=2)) / size(samples, 2)
flat_layout_summary(samples) = vcat(
    vector_summary(samples.free),
    scalar_summary(samples.lower),
    scalar_summary(samples.upper),
)

function lognormal_variance(mean, scale)
    variance = abs2(scale)
    return (exp(variance) - one(mean)) * exp(2mean + variance)
end

function validation_case(
    ::Type{T}, label, proposal, target, context, summarize, expected, variance_trace;
    weight_multiplier=512,
    sample_contract=nothing,
) where {T}
    standard_error = sqrt(variance_trace / T(VALIDATION_SAMPLES))
    return (;
        label, proposal, target, context, summarize, expected, standard_error,
        summary_atol=T(5) * standard_error,
        difference_atol=T(5) * sqrt(T(2)) * standard_error,
        weight_atol=T(weight_multiplier) * eps(T),
        sample_contract,
    )
end

function support_validation_case(
    ::Type{T}, label, proposal, target, context, sample_contract;
    weight_multiplier=512,
) where {T}
    return (;
        label, proposal, target, context,
        summarize=nothing,
        expected=nothing,
        standard_error=nothing,
        summary_atol=nothing,
        difference_atol=nothing,
        weight_atol=T(weight_multiplier) * eps(T),
        sample_contract,
    )
end

validation_case(::Type{T}, case, summarize, expected, variance_trace; kws...) where {T} =
    validation_case(
        T, case.label, case.proposal, case.target, case.context, summarize,
        expected, variance_trace; kws...,
    )

# Vector tolerances use tr(Cov(X)); the interval variance is from deterministic
# quadrature, and the simplex bound is tr(Cov(X)) <= 1 - ||E[X]||² = 2/3.
function validation_cases(::Type{T}) where {T}
    vector_spherical = SphericalGaussian(T[0.4, -0.3, 0.2], T(0.9))
    factor = T[1.25 0; -0.4 0.75]
    factor_proposal = FactorGaussian(T[0.25, -0.5], factor)
    factor_lognormalizer = T[
        -log(T(2) * T(pi)) - log(factor[1, 1]) - log(factor[2, 2]),
    ]
    identity = TransformedProposal(
        SphericalGaussian(T[-0.2, 0.6], T(1.1)),
        IdentityTransform(),
    )
    softplus = TransformedProposal(
        SphericalGaussian(T(0.15), T(0.7)),
        SoftplusTransform(),
    )
    lower = T(-1.25)
    lower_bounded = TransformedProposal(
        SphericalGaussian(T(0.1), T(0.65)),
        IntervalTransform(lower, nothing),
    )
    upper = T(2.5)
    upper_bounded = TransformedProposal(
        SphericalGaussian(T(-0.2), T(0.75)),
        IntervalTransform(nothing, upper),
    )
    flat_lower = T(-1)
    flat_upper = T(2)
    flat_layout = TransformedProposal(
        SphericalGaussian(zeros(T, 4), one(T)),
        (
            free=(1:2 => IdentityTransform()),
            lower=(3 => IntervalTransform(flat_lower, nothing)),
            upper=(4 => IntervalTransform(nothing, flat_upper)),
        ),
    )
    flat_variance = T(2) + T(2) * lognormal_variance(zero(T), one(T))
    return (
        validation_case(
            T, scalar_gaussian_case(T), scalar_summary, T(0.25), abs2(T(1.25)),
        ),
        validation_case(
            T, :vector, DiagonalGaussian(T[0.25, -0.5], T[0.75, 1.25]),
            vector_gaussian_logtarget,
            (location=T[0.25, -0.5], scale=T[0.75, 1.25]),
            vector_summary, T[0.25, -0.5], abs2(T(0.75)) + abs2(T(1.25)),
        ),
        validation_case(
            T, :vector_spherical, vector_spherical, vector_gaussian_logtarget,
            (location=T[0.4, -0.3, 0.2], scale=fill(T(0.9), 3)),
            vector_summary, T[0.4, -0.3, 0.2], T(3) * abs2(T(0.9)),
            sample_contract=(kind=:vector, dimension=3),
        ),
        validation_case(
            T, :factor, factor_proposal, factor_gaussian_logtarget,
            (
                location=T[0.25, -0.5],
                factor,
                lognormalizer=factor_lognormalizer,
            ),
            vector_summary, T[0.25, -0.5], sum(abs2, factor),
            sample_contract=(kind=:vector, dimension=2),
        ),
        validation_case(
            T, :identity, identity, vector_gaussian_logtarget,
            (location=T[-0.2, 0.6], scale=fill(T(1.1), 2)),
            vector_summary, T[-0.2, 0.6], T(2) * abs2(T(1.1)),
            sample_contract=(kind=:vector, dimension=2),
        ),
        support_validation_case(
            T, :softplus, softplus, softplus_gaussian_logtarget,
            (location=T[0.15], scale=T[0.7]),
            (kind=:scalar, lower=zero(T), upper=nothing),
        ),
        validation_case(
            T, :lower_bounded, lower_bounded, lower_bounded_gaussian_logtarget,
            (location=T[0.1], scale=T[0.65], lower=T[lower]),
            scalar_summary,
            lower + exp(T(0.1) + T(0.5) * abs2(T(0.65))),
            lognormal_variance(T(0.1), T(0.65)),
            sample_contract=(kind=:scalar, lower, upper=nothing),
        ),
        validation_case(
            T, :upper_bounded, upper_bounded, upper_bounded_gaussian_logtarget,
            (location=T[-0.2], scale=T[0.75], upper=T[upper]),
            scalar_summary,
            upper - exp(T(-0.2) + T(0.5) * abs2(T(0.75))),
            lognormal_variance(T(-0.2), T(0.75)),
            sample_contract=(kind=:scalar, lower=nothing, upper),
        ),
        validation_case(
            T, :flat_layout, flat_layout, flat_layout_logtarget,
            (lower=T[flat_lower], upper=T[flat_upper]),
            flat_layout_summary,
            T[zero(T), zero(T), flat_lower + exp(T(0.5)), flat_upper - exp(T(0.5))],
            flat_variance,
            sample_contract=(kind=:flat, lower=flat_lower, upper=flat_upper),
        ),
        validation_case(
            T, :positive,
            TransformedProposal(
                SphericalGaussian(T(0.2), T(0.8)),
                PositiveTransform(),
            ),
            positive_gaussian_logtarget, (location=T[0.2], scale=T[0.8]),
            scalar_summary, exp(T(0.2) + T(0.5) * abs2(T(0.8))),
            lognormal_variance(T(0.2), T(0.8)),
        ),
        validation_case(
            T, :interval,
            TransformedProposal(
                SphericalGaussian(zero(T), one(T)),
                IntervalTransform(T(-2), T(3)),
            ),
            interval_gaussian_logtarget, (lower=T[-2], upper=T[3]),
            scalar_summary, T(0.5), T(1.08447589645),
        ),
        validation_case(
            T, :simplex,
            TransformedProposal(
                SphericalGaussian(zeros(T, 2), one(T)),
                SimplexTransform(3),
            ),
            simplex_gaussian_logtarget, NamedTuple(), vector_summary,
            fill(inv(T(3)), 3), T(2) / T(3);
            weight_multiplier=4096,
        ),
    )
end

function validate_sample_contract(samples, ::Type{T}, contract) where {T}
    contract === nothing && return nothing
    if contract.kind === :scalar
        @test samples isa Vector{T}
        @test length(samples) == VALIDATION_SAMPLES
        @test all(isfinite, samples)
        isnothing(contract.lower) || @test all(>(contract.lower), samples)
        isnothing(contract.upper) || @test all(<(contract.upper), samples)
    elseif contract.kind === :vector
        @test samples isa Matrix{T}
        @test size(samples) == (contract.dimension, VALIDATION_SAMPLES)
        @test all(isfinite, samples)
    elseif contract.kind === :flat
        @test keys(samples) == (:free, :lower, :upper)
        @test samples.free isa Matrix{T}
        @test size(samples.free) == (2, VALIDATION_SAMPLES)
        @test samples.lower isa Vector{T}
        @test samples.upper isa Vector{T}
        @test length(samples.lower) == VALIDATION_SAMPLES
        @test length(samples.upper) == VALIDATION_SAMPLES
        @test all(isfinite, samples.free)
        @test all(>(contract.lower), samples.lower)
        @test all(<(contract.upper), samples.upper)
    else
        error("unknown CUDA validation sample contract $(contract.kind)")
    end
    return nothing
end

_all_array_leaves(predicate, array::AbstractArray) = predicate(array)
_all_array_leaves(predicate, arrays::NamedTuple) =
    all(array -> _all_array_leaves(predicate, array), values(arrays))

_explicit_array_transfer(array::AbstractArray) = Array(array)
_explicit_array_transfer(arrays::NamedTuple) = map(_explicit_array_transfer, arrays)

function validate_case(device, ::Type{T}, case, seed) where {T}
    algorithm = ImportanceSampling(case.proposal; nsamples=VALIDATION_SAMPLES)
    cpu = importance_sample(
        Xoshiro(seed),
        case.target,
        case.context,
        algorithm;
        threaded=false,
    )
    prepared = prepare_sampler(
        Xoshiro(seed),
        case.target,
        case.context,
        algorithm;
        threaded=true,
    ) |> device
    @test ImportanceSamplers._backend_state_resident(
        device,
        (prepared.algorithm, prepared.target, prepared.random_buffers),
    )
    gpu = importance_sample!(prepared)

    @test _all_array_leaves(array -> array isa CuArray, gpu.samples)
    @test gpu.logweights isa CuArray
    @test _all_array_leaves(array -> eltype(array) === T, gpu.samples)
    @test eltype(gpu.logweights) === T
    @test transfer_tuple(gpu) == (1, 2sizeof(UInt64))

    resident_view = gpu[1:16]
    @test _all_array_leaves(
        array -> MLDataDevices.get_device(array) isa MLDataDevices.CUDADevice,
        resident_view.samples,
    )
    @test MLDataDevices.get_device(resident_view.logweights) isa MLDataDevices.CUDADevice
    @test transfer_tuple(gpu) == (1, 2sizeof(UInt64))

    resident_normalized = normalized_weights(gpu)
    @test resident_normalized isa CuArray
    @test MLDataDevices.get_device(resident_normalized) isa MLDataDevices.CUDADevice
    @test transfer_tuple(gpu) == (2, 2sizeof(UInt64) + 2sizeof(T))

    host = MLDataDevices.cpu_device()(gpu)
    @test _all_array_leaves(array -> array isa Array, host.samples)
    @test host.logweights isa Vector{T}
    @test host.samples == _explicit_array_transfer(gpu.samples)
    @test host.logweights == Array(gpu.logweights)
    @test transfer_tuple(host) == transfer_tuple(gpu)

    @test maximum(abs, cpu.logweights) <= case.weight_atol
    @test maximum(abs, host.logweights) <= case.weight_atol
    @test abs(lognormalizer(cpu)) <= case.weight_atol
    @test abs(lognormalizer(host)) <= case.weight_atol

    validate_sample_contract(cpu.samples, T, case.sample_contract)
    validate_sample_contract(host.samples, T, case.sample_contract)

    if !isnothing(case.summarize)
        cpu_summary = case.summarize(cpu.samples)
        gpu_summary = case.summarize(host.samples)
        @test isapprox(cpu_summary, case.expected; atol=case.summary_atol, rtol=zero(T))
        @test isapprox(gpu_summary, case.expected; atol=case.summary_atol, rtol=zero(T))
        @test isapprox(cpu_summary, gpu_summary; atol=case.difference_atol, rtol=zero(T))
    end
    return nothing
end

function validate_rejections(device)
    T = Float32
    proposal = SphericalGaussian(zero(T), one(T))
    captured = T[0.25]
    captured_target = (sample, p) -> p.shift[1] + captured[1] - abs2(sample) / 2
    captured_prepared = prepare_sampler(
        Xoshiro(VALIDATION_SEED),
        captured_target,
        (shift=T[0.5],),
        ImportanceSampling(proposal; nsamples=16);
        threaded=true,
    )
    captured_error = caught(() -> device(captured_prepared))
    @test captured_error isa SamplerDeviceError
    @test captured_error.reason === :opaque_host_closure
    @test occursin("pass numerical state through p", sprint(showerror, captured_error))

    generic = GenericGaussian(zero(T), one(T))
    generic_prepared = prepare_sampler(
        Xoshiro(VALIDATION_SEED),
        zero_logtarget,
        ImportanceSampling(generic; nsamples=16);
        threaded=true,
    )
    generic_error = caught(() -> device(generic_prepared))
    @test generic_error isa SamplerDeviceError
    @test generic_error.reason === :generic_proposal_cpu_only

    unadapted_prepared = prepare_sampler(
        Xoshiro(VALIDATION_SEED),
        contextual_scalar_logtarget,
        UnadaptedCUDAContext(T[0.25]),
        ImportanceSampling(proposal; nsamples=16);
        threaded=true,
    )
    expected_unadapted_rng = copy(getfield(unadapted_prepared, :rng))
    unadapted_error = caught(() -> device(unadapted_prepared))
    @test unadapted_error isa SamplerDeviceError
    @test unadapted_error.reason === :kernel_argument_unsupported
    @test rand(getfield(unadapted_prepared, :rng), UInt64) ==
          rand(expected_unadapted_rng, UInt64)

    migration_source = prepare_sampler(
        Xoshiro(VALIDATION_SEED),
        contextual_scalar_logtarget,
        (shift=T[0.25],),
        ImportanceSampling(proposal; nsamples=16);
        threaded=true,
    ) |> device
    migration_error = caught(() -> MLDataDevices.cpu_device()(migration_source))
    @test migration_error isa SamplerDeviceError
    @test migration_error.reason === :prepared_migration_unsupported

    invalid = TransformedProposal(
        SphericalGaussian(zeros(T, 2), T(floatmax(T))),
        SimplexTransform(3),
    )
    invalid_prepared = prepare_sampler(
        Xoshiro(VALIDATION_SEED),
        zero_logtarget,
        ImportanceSampling(invalid; nsamples=FAILURE_SAMPLES);
        threaded=true,
    ) |> device
    invalid_error = caught(() -> importance_sample!(invalid_prepared))
    @test invalid_error isa SamplerExecutionError
    @test invalid_error.phase === :proposal_draw
    @test invalid_error.captured.ex isa InvalidTransformError
    return nothing
end


function validate_context_execution(device)
    T = Float32
    proposal = SphericalGaussian(zero(T), one(T))

    adapted = prepare_sampler(
        Xoshiro(VALIDATION_SEED + UInt64(0x10)),
        contextual_scalar_logtarget,
        AdaptedCUDAContext(T[0.25]),
        ImportanceSampling(proposal; nsamples=16);
        threaded=true,
    ) |> device
    adapted_result = importance_sample!(adapted)
    @test adapted_result.samples isa CuArray
    @test adapted_result.logweights isa CuArray

    named = prepare_sampler(
        Xoshiro(VALIDATION_SEED + UInt64(0x11)),
        contextual_scalar_logtarget,
        (shift=T[0.25],),
        ImportanceSampling(proposal; nsamples=2_048);
        threaded=true,
    ) |> device
    named_result = importance_sample!(named)
    @test named_result.samples isa CuArray
    @test named_result.logweights isa CuArray
    @test length(named_result) == 2_048
    return nothing
end

function validate_public_preserving_device(device, ::Type{T}) where {T}
    proposal = DiagonalGaussian(T[0.25, -0.5], T[0.75, 1.25])
    context = (location=T[0.25, -0.5], scale=T[0.75, 1.25])
    source = prepare_sampler(
        Xoshiro(VALIDATION_SEED + UInt64(sizeof(T))),
        vector_gaussian_logtarget,
        context,
        ImportanceSampling(proposal; nsamples=16);
        threaded=true,
    )
    prepared = device(source)
    prepared_proposal = getfield(getfield(prepared, :algorithm), :proposal)
    prepared_context = getfield(getfield(prepared, :target), :context)
    buffers = getfield(prepared, :random_buffers)

    @test eltype(device) === Nothing
    @test eltype(prepared_proposal.location) === T
    @test eltype(prepared_proposal.scale.scales) === T
    @test eltype(prepared_context.location) === T
    @test eltype(prepared_context.scale) === T
    @test eltype(buffers.uniform) === T
    @test eltype(buffers.normal) === T

    result = importance_sample!(prepared)
    @test eltype(result.samples) === T
    @test eltype(result.logweights) === T
    return nothing
end

function validate_explicit_physical_device()
    physical_devices = collect(CUDA.devices())
    length(physical_devices) >= 2 || return @test_skip "two CUDA devices required"
    caller_device = physical_devices[1]
    selected_device = physical_devices[2]
    CUDA.device!(caller_device)
    device = cuda_device(2)
    @test CUDA.device() == caller_device
    inherited_device = ImportanceSamplers._with_backend_device(device) do
        @test CUDA.device() == selected_device
        return fetch(Threads.@spawn CUDA.device())
    end
    @test inherited_device == caller_device
    @test CUDA.device() == caller_device

    cases = (
        (
            proposal=SphericalGaussian(0.0f0, 1.0f0),
            target=AdaptableCUDAFunction(Float32[0.25]),
            context=(shift=Float32[0.5],),
        ),
        (
            proposal=DiagonalGaussian(
                Float32[0.25, -0.5],
                Float32[0.75, 1.25],
            ),
            target=vector_gaussian_logtarget,
            context=(
                location=Float32[0.25, -0.5],
                scale=Float32[0.75, 1.25],
            ),
        ),
    )

    for (case_index, case) in enumerate(cases)
        source = prepare_sampler(
            Xoshiro(VALIDATION_SEED + UInt64(0x40 + case_index)),
            case.target,
            case.context,
            ImportanceSampling(case.proposal; nsamples=2_048);
            threaded=true,
        )
        prepared = device(source)
        @test CUDA.device() == caller_device

        algorithm = getfield(prepared, :algorithm)
        prepared_target = getfield(prepared, :target)
        buffers = getfield(prepared, :random_buffers)
        scratch = getfield(buffers, :failure_scratch)
        failure_storage = getfield(getfield(scratch, :record), :storage)
        @test ImportanceSamplers._backend_state_resident(
            device,
            (algorithm, prepared_target, buffers),
        )
        for array in values(getfield(prepared_target, :context))
            @test CUDA.device(array) == selected_device
        end
        prepared_proposal = getfield(algorithm, :proposal)
        if case_index == 1
            callable = getfield(prepared_target, :target)
            @test CUDA.device(callable.offset) == selected_device
        else
            @test CUDA.device(prepared_proposal.location) == selected_device
            @test CUDA.device(prepared_proposal.scale.scales) == selected_device
        end
        @test CUDA.device(buffers.uniform) == selected_device
        @test CUDA.device(buffers.normal) == selected_device
        @test CUDA.device(failure_storage) == selected_device

        result = importance_sample!(prepared)
        @test CUDA.device() == caller_device
        @test CUDA.device(result.samples) == selected_device
        @test CUDA.device(result.logweights) == selected_device

        weights = normalized_weights(result)
        @test CUDA.device() == caller_device
        @test CUDA.device(weights) == selected_device
        weight_sum = ImportanceSamplers._with_backend_device(device) do
            return sum(weights)
        end
        @test weight_sum ≈ 1.0f0 atol = 32eps(Float32)
        @test CUDA.device() == caller_device
        @test isfinite(lognormalizer(result))
        @test CUDA.device() == caller_device
    end
    return nothing
end

function validate_public_rng_ownership_and_replay(device)
    T = Float32
    algorithm = ImportanceSampling(
        SphericalGaussian(zero(T), one(T));
        nsamples=32,
    )
    make_prepared() = prepare_sampler(
        Xoshiro(VALIDATION_SEED + UInt64(0x20)),
        zero_logtarget,
        algorithm;
        threaded=true,
    ) |> device

    first_prepared = make_prepared()
    second_prepared = make_prepared()
    first_rng = getfield(first_prepared, :rng)
    second_rng = getfield(second_prepared, :rng)
    default_rng = CUDA.default_rng()
    @test first_rng isa CUDA.RNG
    @test second_rng isa CUDA.RNG
    @test first_rng !== second_rng
    @test first_rng !== default_rng
    @test second_rng !== default_rng

    first_result = importance_sample!(first_prepared)
    second_result = importance_sample!(second_prepared)
    @test Array(first_result.samples) == Array(second_result.samples)
    @test Array(first_result.logweights) == Array(second_result.logweights)

    first_advanced = importance_sample!(first_prepared)
    second_advanced = importance_sample!(second_prepared)
    @test Array(first_advanced.samples) == Array(second_advanced.samples)
    @test Array(first_advanced.logweights) == Array(second_advanced.logweights)
    @test Array(first_advanced.samples) != Array(first_result.samples)
    return nothing
end

function validate_failure_scratch_reuse_and_reset(device)
    T = Float32
    proposal = TransformedProposal(
        DiagonalGaussian(T[0.25], T[1]),
        IdentityTransform(),
    )
    prepared = prepare_sampler(
        Xoshiro(VALIDATION_SEED + UInt64(0x28)),
        zero_logtarget,
        ImportanceSampling(proposal; nsamples=FAILURE_SAMPLES);
        threaded=true,
    ) |> device
    scratch = getfield(getfield(prepared, :random_buffers), :failure_scratch)
    failure_storage = getfield(getfield(scratch, :record), :storage)
    prepared_scale = getfield(
        getfield(getfield(prepared, :algorithm), :proposal).base.scale,
        :scales,
    )

    fill!(prepared_scale, T(Inf))
    failure = try
        importance_sample!(prepared)
        nothing
    catch error
        error
    end
    @test failure isa SamplerExecutionError
    @test failure.phase === :proposal_draw
    failure_snapshot =
        ImportanceSamplers._device_failure_snapshot(getfield(scratch, :record))
    @test failure_snapshot.failure.count == FAILURE_SAMPLES
    @test failure_snapshot.failure.first_logical_index == 1
    @test failure_snapshot.failure.first_block == 1
    @test failure_snapshot.failure.reason_bits == UInt16(0x0800)
    @test failure_snapshot.transfers == (count=1, bytes=2 * sizeof(UInt64))

    fill!(prepared_scale, one(T))
    first_result = importance_sample!(prepared)
    @test getfield(getfield(prepared, :random_buffers), :failure_scratch) ===
          scratch
    @test getfield(getfield(scratch, :record), :storage) === failure_storage
    @test Array(failure_storage) == zeros(UInt64, 2)
    @test (
        first_result.diagnostics.transfers.count,
        first_result.diagnostics.transfers.bytes,
    ) == (1, 2 * sizeof(UInt64))

    second_result = importance_sample!(prepared)
    @test getfield(getfield(scratch, :record), :storage) === failure_storage
    @test Array(failure_storage) == zeros(UInt64, 2)
    @test (
        second_result.diagnostics.transfers.count,
        second_result.diagnostics.transfers.bytes,
    ) == (1, 2 * sizeof(UInt64))
    @test first_result.samples !== second_result.samples
    @test first_result.logweights !== second_result.logweights
    @test first_result.diagnostics.transfers !==
          second_result.diagnostics.transfers
    return nothing
end

function validate_callable_and_view_transfers(device, ::Type{T}) where {T}
    target = AdaptableCUDAFunction(T[0.25])
    prepared = prepare_sampler(
        Xoshiro(VALIDATION_SEED + UInt64(0x30) + UInt64(sizeof(T))),
        target,
        (shift=T[0.5],),
        ImportanceSampling(
            SphericalGaussian(zero(T), one(T));
            nsamples=16,
        );
        threaded=true,
    ) |> device
    transferred_target = getfield(getfield(prepared, :target), :target)
    @test transferred_target isa AdaptableCUDAFunction
    @test transferred_target !== target
    @test transferred_target.offset isa CuArray{T,1}
    @test Array(transferred_target.offset) == T[0.25]
    target.offset[1] = T(9)
    @test Array(transferred_target.offset) == T[0.25]

    result = importance_sample!(prepared)
    resident_view = result[1:8]
    host_view = MLDataDevices.cpu_device()(resident_view)
    @test host_view isa WeightedSampleView
    @test host_view.samples isa Array
    @test host_view.logweights isa Vector{T}
    @test host_view.samples == Array(resident_view.samples)
    @test host_view.logweights == Array(resident_view.logweights)
    @test collect(host_view) == [
        (
            sample=host_view.samples[index],
            logweight=host_view.logweights[index],
            provenance=NamedTuple(),
        ) for index in eachindex(host_view.logweights)
    ]
    @test host_view.samples !== resident_view.samples
    @test host_view.logweights !== resident_view.logweights
    @test host_view.transfers !== resident_view.transfers
    @test (host_view.transfers.count, host_view.transfers.bytes) ==
          (resident_view.transfers.count, resident_view.transfers.bytes)
    return nothing
end

function environment_record()
    gpu = CUDA.device()
    return (
        gpu=CUDA.name(gpu),
        capability=CUDA.capability(gpu),
        driver=CUDA.driver_version(),
        runtime=CUDA.runtime_version(),
        total_device_memory=CUDA.total_memory(),
        julia=VERSION,
        packages=cuda_package_versions((
            "Adapt",
            "CUDA",
            "ImportanceSamplers",
            "KernelAbstractions",
            "MLDataDevices",
        )),
        seed=VALIDATION_SEED,
        sample_count=VALIDATION_SAMPLES,
        failure_sample_count=FAILURE_SAMPLES,
        scalar_types=CUDA_PLAIN_IS_A100_TYPES,
        cases=CUDA_PLAIN_IS_A100_CASES,
        mode=ONE_THREAD_LAUNCH_MODE ? :one_thread_launch : :full_matrix,
        default_threads=Threads.nthreads(:default),
        tolerances=(
            summaries="5 standard errors from analytic variance or trace-variance bounds",
            cpu_cuda_difference="5sqrt(2) standard errors for independent streams",
            interval_variance=1.08447589645,
            logweights=(ordinary="512eps(T)", simplex="4096eps(T)"),
        ),
        commands=(validation=VALIDATION_COMMAND, project=Base.active_project()),
    )
end

function main()
    device = cuda_device()
    if ONE_THREAD_LAUNCH_MODE
        @testset "CUDA one-thread launch contract" begin
            @test Threads.nthreads(:default) == 1
            validate_context_execution(device)
        end
        return environment_record()
    end
    @testset "CUDA plain importance sampling" begin
        for T in CUDA_PLAIN_IS_A100_TYPES
            cases = validation_cases(T)
            Tuple(case.label for case in cases) == CUDA_PLAIN_IS_A100_CASES ||
                error("CUDA validation cases do not match capability metadata")
            @testset "$T $(case.label)" for (case_index, case) in enumerate(cases)
                validate_case(device, T, case, VALIDATION_SEED + UInt64(case_index))
            end
        end
        @testset "fail closed" begin
            validate_rejections(device)
        end
        @testset "context and one-thread launch contracts" begin
            validate_context_execution(device)
        end
        @testset "public preserving device and owned RNG" begin
            for T in CUDA_PLAIN_IS_A100_TYPES
                validate_public_preserving_device(device, T)
                validate_callable_and_view_transfers(device, T)
            end
            validate_public_rng_ownership_and_replay(device)
            validate_failure_scratch_reuse_and_reset(device)
        end
        @testset "explicit physical CUDA device" begin
            validate_explicit_physical_device()
        end
    end
    return environment_record()
end

main()
