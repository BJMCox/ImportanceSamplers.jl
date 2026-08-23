import KernelAbstractions
import Random

struct FusedQuadraticTarget{T}
    offset::T
end

function (target::FusedQuadraticTarget{T})(sample::T)::T where {T}
    return target.offset - abs2(sample)
end

struct FusedTargetFailure <: Exception
    sample::Float64
end

struct FusedFailAtTarget
    sample::Float64
end

struct FusedPhaseOrderTarget end

function (::FusedPhaseOrderTarget)(sample::Float64)::Float64
    sample == 2.0 && throw(FusedTargetFailure(sample))
    return isfinite(sample) ? 0.0 : -Inf
end

function (target::FusedFailAtTarget)(sample::Float64)::Float64
    sample == target.sample && throw(FusedTargetFailure(sample))
    return -abs2(sample)
end

struct FiniteOrMinusInfTarget{T} end

function (::FiniteOrMinusInfTarget{T})(sample::T)::T where {T}
    return isfinite(sample) ? zero(T) : T(-Inf)
end

struct FusedConstantTarget{T} end

function (::FusedConstantTarget{T})(::T)::T where {T}
    return zero(T)
end

mutable struct PrefilledNormalRNG{T} <: Random.AbstractRNG
    values::Vector{T}
    index::Int
end

function _next_prefilled_normal!(rng::PrefilledNormalRNG)
    rng.index += 1
    return rng.values[rng.index]
end

Random.randn(rng::PrefilledNormalRNG{Float32}, ::Type{Float32}) =
    _next_prefilled_normal!(rng)
Random.randn(rng::PrefilledNormalRNG{Float64}, ::Type{Float64}) =
    _next_prefilled_normal!(rng)

function _caught_kernel_execution_error(f)
    try
        f()
    catch error
        return error
    end
    return nothing
end

function _run_native_fused(target, proposal, normal_buffer, threaded)
    sampler = prepare_sampler(
        PrefilledNormalRNG(copy(normal_buffer), 0),
        target,
        ImportanceSampling(proposal; nsamples=length(normal_buffer));
        threaded=threaded,
    )
    return @inferred importance_sample!(sampler)
end

function _native_unfused_reference(target, proposal, normal_buffer::Vector{T}) where {T}
    sampler = prepare_sampler(
        PrefilledNormalRNG(copy(normal_buffer), 0),
        target,
        ImportanceSampling(proposal; nsamples=length(normal_buffer));
        threaded=false,
    )
    samples, logweights = ImportanceSamplers._importance_sample_generic_cpu!(sampler, false)
    return (samples=samples, logweights=logweights)
end

function _check_transformed_native_equivalence(
    ::Type{T},
    transform,
    coordinates::Vector{T},
) where {T<:Union{Float32,Float64}}
    proposal = TransformedProposal(SphericalGaussian(zero(T), one(T)), transform)
    target = FusedConstantTarget{T}()
    fused = @inferred _run_native_fused(target, proposal, coordinates, false)
    reference = @inferred _native_unfused_reference(
        target,
        proposal,
        copy(coordinates),
    )
    tolerance = T(64) * eps(T)
    @test fused.samples == reference.samples
    @test fused.logweights ≈
          reference.logweights rtol = tolerance atol = tolerance
    return maximum(abs, fused.logweights - reference.logweights)
end

function _check_scalar_native_equivalence(::Type{T}) where {T<:Union{Float32,Float64}}
    location = T(0.25)
    scale = T(1.5)
    normals = T[-1.5, -0.25, 0, 0.75, 1.25]
    proposal = TransformedProposal(
        SphericalGaussian(location, scale),
        SoftplusTransform(),
    )
    target = FusedQuadraticTarget(T(0.75))
    fused = @inferred _run_native_fused(target, proposal, normals, false)
    reference = @inferred _native_unfused_reference(target, proposal, copy(normals))

    coordinates = location .+ scale .* normals
    expected_samples = max.(coordinates, zero(T)) .+ log1p.(exp.(-abs.(coordinates)))
    expected_logjacs = coordinates .- expected_samples
    logtwopi = log(T(2) * T(pi))
    expected_base_logs = @. -T(0.5) * logtwopi - log(scale) - T(0.5) * abs2(normals)
    expected_target_logs = @. target.offset - abs2(expected_samples)
    expected_logweights = expected_target_logs .- (expected_base_logs .- expected_logjacs)

    @test fused.samples == expected_samples
    @test fused.samples == reference.samples
    tolerance = T(16) * eps(T)
    @test fused.logweights ≈ expected_logweights rtol = tolerance atol = tolerance
    @test fused.logweights ≈
          reference.logweights rtol = tolerance atol = tolerance
    @test fused.samples !== normals
    @test fused.logweights !== normals

    return fused
end

@testset "portable kernel smoke test" begin
    output = zeros(Int, 4)
    backend = KernelAbstractions.get_backend(output)

    @test backend isa KernelAbstractions.CPU

    ImportanceSamplers._kernel_smoke!(output)
    KernelAbstractions.synchronize(backend)

    @test output == collect(1:4)
end

@testset "native scalar fused execution" begin
    fused32 = _check_scalar_native_equivalence(Float32)
    fused64 = _check_scalar_native_equivalence(Float64)
    @test fused32.samples isa Vector{Float32}
    @test fused32.logweights isa Vector{Float32}
    @test fused64.samples isa Vector{Float64}
    @test fused64.logweights isa Vector{Float64}
end

@testset "native transformed density follows public inverse semantics" begin
    for T in (Float32, Float64)
        bounded_coordinates = T[-10, -8, 8, 10]
        bounded_error = _check_transformed_native_equivalence(
            T,
            IntervalTransform(T(-2), T(3)),
            bounded_coordinates,
        )
        @test bounded_error <= T(64) * eps(T)

        extreme = T === Float32 ? T[-80, -40, 40, 80] : T[-700, -350, 350, 700]
        for transform in (
            PositiveTransform(),
            SoftplusTransform(),
            IntervalTransform(zero(T), nothing),
            IntervalTransform(nothing, zero(T)),
        )
            error = _check_transformed_native_equivalence(T, transform, extreme)
            @test error <= T(64) * eps(T)
        end
    end
end

@testset "native fused failures retain phase and logical index" begin
    target_proposal = SphericalGaussian(0.0, 1.0)
    target = FusedFailAtTarget(2.0)
    target_normals = [-1.0, 0.0, 2.0, 3.0]
    fused_target_failure = _caught_kernel_execution_error() do
        _run_native_fused(target, target_proposal, target_normals, false)
    end
    reference_target_failure = _caught_kernel_execution_error() do
        _native_unfused_reference(target, target_proposal, copy(target_normals))
    end
    for failure in (fused_target_failure, reference_target_failure)
        @test failure isa SamplerExecutionError
        @test failure.phase === :target
        @test failure.sample_index == 3
        @test failure.captured.ex isa FusedTargetFailure
        @test failure.captured.ex.sample == 2.0
    end

    transform_proposal = TransformedProposal(
        SphericalGaussian(0.0, floatmax(Float64)),
        PositiveTransform(),
    )
    transform_target = FusedQuadraticTarget(0.0)
    transform_normals = [0.0, 1.0]
    fused_transform_failure = _caught_kernel_execution_error() do
        _run_native_fused(
            transform_target,
            transform_proposal,
            transform_normals,
            false,
        )
    end
    reference_transform_failure = _caught_kernel_execution_error() do
        _native_unfused_reference(
            transform_target,
            transform_proposal,
            copy(transform_normals),
        )
    end
    for failure in (fused_transform_failure, reference_transform_failure)
        @test failure isa SamplerExecutionError
        @test failure.phase === :proposal_draw
        @test failure.sample_index == 2
        @test failure.captured.ex isa InvalidTransformError
        @test failure.captured.ex.reason === :nonfinite_output
    end

    density_proposal = SphericalGaussian(0.0, 1.0)
    density_target = FiniteOrMinusInfTarget{Float64}()
    density_normals = [0.0, Inf]
    fused_density_failure = _caught_kernel_execution_error() do
        _run_native_fused(density_target, density_proposal, density_normals, false)
    end
    reference_density_failure = _caught_kernel_execution_error() do
        _native_unfused_reference(
            density_target,
            density_proposal,
            copy(density_normals),
        )
    end
    for failure in (fused_density_failure, reference_density_failure)
        @test failure isa SamplerExecutionError
        @test failure.phase === :proposal_logdensity
        @test failure.sample_index == 2
        @test failure.captured.ex isa DomainError
    end

    phase_order_normals = zeros(2_048)
    phase_order_normals[1] = Inf
    phase_order_normals[1_500] = 2.0
    for threaded in (false, true)
        phase_order_failure = _caught_kernel_execution_error() do
            _run_native_fused(
                FusedPhaseOrderTarget(),
                density_proposal,
                phase_order_normals,
                threaded,
            )
        end
        @test phase_order_failure isa SamplerExecutionError
        @test phase_order_failure.phase === :target
        @test phase_order_failure.sample_index == 1_500
        @test phase_order_failure.captured.ex isa FusedTargetFailure
    end
end

@testset "native fused capability and fallback" begin
    scalar = SphericalGaussian(0.0, 1.0)
    transformed_scalar = TransformedProposal(scalar, PositiveTransform())
    vector = SphericalGaussian(zeros(2), 1.0)
    transformed_vector = TransformedProposal(vector, IdentityTransform())
    product = ProductProposal((left=scalar, right=scalar))
    malformed_interval = ImportanceSamplers.TransformedProposal(
        SphericalGaussian(0.0f0, 1.0f0),
        IntervalTransform(0.0, 1.0),
        ImportanceSamplers._PreparedProposalToken(),
    )
    spoofed_transform = IntervalTransform{Float32,Float64,Float64}(0.0, 1.0)
    spoofed_interval = TransformedProposal(
        SphericalGaussian(0.0f0, 1.0f0),
        spoofed_transform,
    )
    malformed_base = ImportanceSamplers._GaussianProposal(
        ImportanceSamplers.GaussianFamily(),
        0.0f0,
        ImportanceSamplers._SphericalGaussianScale(1.0),
        Float32(-0.9189385),
    )

    @test ImportanceSamplers._sampling_execution(scalar, false) isa
          ImportanceSamplers._KernelExecution
    @test ImportanceSamplers._sampling_execution(transformed_scalar, true) isa
          ImportanceSamplers._KernelExecution
    @test ImportanceSamplers._sampling_execution(vector, false) isa
          ImportanceSamplers._SerialCPUExecution
    @test ImportanceSamplers._sampling_execution(transformed_vector, true) isa
          ImportanceSamplers._ThreadedCPUExecution
    @test ImportanceSamplers._sampling_execution(product, true) isa
          ImportanceSamplers._ThreadedCPUExecution
    @test ImportanceSamplers._sampling_execution(malformed_interval, true) isa
          ImportanceSamplers._ThreadedCPUExecution
    @test !ImportanceSamplers._supports_native_fused_cpu(spoofed_interval)
    @test ImportanceSamplers._sampling_execution(spoofed_interval, true) isa
          ImportanceSamplers._ThreadedCPUExecution
    @test !ImportanceSamplers._supports_native_fused_cpu(malformed_base)
    @test ImportanceSamplers._sampling_execution(malformed_base, false) isa
          ImportanceSamplers._SerialCPUExecution
end
