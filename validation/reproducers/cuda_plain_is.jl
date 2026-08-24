using CUDA
using cuDNN
using DensityInterface
using ImportanceSamplers
using MLDataDevices
using Pkg
using Random
using Test

include(joinpath(@__DIR__, "..", "cuda_plain_is_support.jl"))

const VALIDATION_SEED = 0x6e61746976656770
const VALIDATION_SAMPLES = 65_536
const FAILURE_SAMPLES = 4_096
const VALIDATION_COMMAND =
    "Kaimon ex: include(\"reproducers/cuda_plain_is.jl\") in the validation project"

struct GenericGaussian{T}
    location::T
    scale::T
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

@inline function positive_gaussian_logtarget(sample, p)
    coordinate = log(sample)
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

function validation_case(
    ::Type{T}, label, proposal, target, context, summarize, expected, summary_atol;
    weight_multiplier=512,
) where {T}
    return (;
        label, proposal, target, context, summarize, expected,
        summary_atol=T(summary_atol),
        weight_atol=T(weight_multiplier) * eps(T),
    )
end

validation_case(::Type{T}, case, summarize, expected, summary_atol; kws...) where {T} =
    validation_case(
        T, case.label, case.proposal, case.target, case.context, summarize,
        expected, summary_atol; kws...,
    )

function validation_cases(::Type{T}) where {T}
    return (
        validation_case(
            T, scalar_gaussian_case(T), scalar_summary, T(0.25), 0.02,
        ),
        validation_case(
            T, :vector, DiagonalGaussian(T[0.25, -0.5], T[0.75, 1.25]),
            vector_gaussian_logtarget,
            (location=T[0.25, -0.5], scale=T[0.75, 1.25]),
            vector_summary, T[0.25, -0.5], 0.02,
        ),
        validation_case(
            T, :positive,
            TransformedProposal(
                SphericalGaussian(T(0.2), T(0.8)),
                PositiveTransform(),
            ),
            positive_gaussian_logtarget, (location=T[0.2], scale=T[0.8]),
            scalar_summary, exp(T(0.2) + T(0.5) * abs2(T(0.8))), 0.04,
        ),
        validation_case(
            T, :interval,
            TransformedProposal(
                SphericalGaussian(zero(T), one(T)),
                IntervalTransform(T(-2), T(3)),
            ),
            interval_gaussian_logtarget, (lower=T[-2], upper=T[3]),
            scalar_summary, T(0.5), 0.01,
        ),
        validation_case(
            T, :simplex,
            TransformedProposal(
                SphericalGaussian(zeros(T, 2), one(T)),
                SimplexTransform(3),
            ),
            simplex_gaussian_logtarget, NamedTuple(), vector_summary,
            fill(inv(T(3)), 3), 0.01;
            weight_multiplier=4096,
        ),
    )
end

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
    @test all(value -> value isa CuArray, values(getfield(prepared.target, :context)))
    gpu = importance_sample!(prepared)

    @test gpu.samples isa CuArray
    @test gpu.logweights isa CuArray
    @test eltype(gpu.samples) === T
    @test eltype(gpu.logweights) === T
    @test transfer_tuple(gpu) == (1, 2sizeof(UInt64))

    resident_view = gpu[1:16]
    @test MLDataDevices.get_device(resident_view.samples) isa MLDataDevices.CUDADevice
    @test MLDataDevices.get_device(resident_view.logweights) isa MLDataDevices.CUDADevice
    @test transfer_tuple(gpu) == (1, 2sizeof(UInt64))

    resident_normalized = normalized_weights(gpu)
    @test resident_normalized isa CuArray
    @test MLDataDevices.get_device(resident_normalized) isa MLDataDevices.CUDADevice
    @test transfer_tuple(gpu) == (2, 2sizeof(UInt64) + 2sizeof(T))

    host = MLDataDevices.cpu_device()(gpu)
    @test host.samples isa Array
    @test host.logweights isa Vector{T}
    @test host.samples == Array(gpu.samples)
    @test host.logweights == Array(gpu.logweights)
    @test transfer_tuple(host) == transfer_tuple(gpu)

    @test maximum(abs, cpu.logweights) <= case.weight_atol
    @test maximum(abs, host.logweights) <= case.weight_atol
    @test abs(lognormalizer(cpu)) <= case.weight_atol
    @test abs(lognormalizer(host)) <= case.weight_atol

    cpu_summary = case.summarize(cpu.samples)
    gpu_summary = case.summarize(host.samples)
    @test isapprox(cpu_summary, case.expected; atol=case.summary_atol, rtol=zero(T))
    @test isapprox(gpu_summary, case.expected; atol=case.summary_atol, rtol=zero(T))
    @test isapprox(cpu_summary, gpu_summary; atol=2case.summary_atol, rtol=zero(T))
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
    @test generic_error.reason === :accelerator_rng_unavailable

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
            "cuDNN",
        )),
        seed=VALIDATION_SEED,
        sample_count=VALIDATION_SAMPLES,
        failure_sample_count=FAILURE_SAMPLES,
        scalar_types=(Float32, Float64),
        cases=(:scalar, :vector, :positive, :interval, :simplex),
        tolerances=(
            summaries=(scalar=0.02, vector=0.02, positive=0.04, interval=0.01, simplex=0.01),
            logweights=(ordinary="512eps(T)", simplex="4096eps(T)"),
        ),
        commands=(validation=VALIDATION_COMMAND, project=Base.active_project()),
    )
end

function main()
    device = cuda_device()
    @testset "CUDA plain importance sampling" begin
        for T in (Float32, Float64)
            @testset "$T $(case.label)" for (case_index, case) in
                                               enumerate(validation_cases(T))
                validate_case(device, T, case, VALIDATION_SEED + UInt64(case_index))
            end
        end
        @testset "fail closed" begin
            validate_rejections(device)
        end
    end
    return environment_record()
end

main()
