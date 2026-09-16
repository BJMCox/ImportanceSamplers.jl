import KernelAbstractions
import KernelAbstractions: @index, @kernel
import LinearAlgebra
import MLDataDevices
import Random

struct FusedQuadraticTarget{T}
    offset::T
end

function (target::FusedQuadraticTarget{T})(sample::T)::T where {T}
    return target.offset - abs2(sample)
end

struct FusedConstantTarget{T} end

function (::FusedConstantTarget{T})(::T)::T where {T}
    return zero(T)
end

struct FusedVectorTarget{T}
    offset::T
end

function (target::FusedVectorTarget{T})(sample::AbstractVector{T})::T where {T}
    return target.offset - sum(abs2, sample)
end

struct FusedNamedTarget{T} end

function (::FusedNamedTarget{T})(sample::NamedTuple)::T where {T}
    return -sum(abs2, sample.weights) - abs2(sample.rate) - abs2(sample.offset)
end

struct MinusInfVectorTarget{T} end

function (::MinusInfVectorTarget{T})(sample::AbstractVector{T})::T where {T}
    return iszero(first(sample)) ? T(-Inf) : -sum(abs2, sample)
end

struct NativeVectorOnlyTarget end
(::NativeVectorOnlyTarget)(sample::Vector{Float64}) = -sum(abs2, sample)

struct NativeTargetFailure <: Exception
    sample::Float64
end

struct NativeFailAtVectorTarget
    coordinate::Float64
end

function (target::NativeFailAtVectorTarget)(sample::AbstractVector{Float64})::Float64
    first(sample) == target.coordinate && throw(NativeTargetFailure(first(sample)))
    return -sum(abs2, sample)
end

struct NativePhaseOrderTarget end

function (::NativePhaseOrderTarget)(sample::Float64)::Float64
    (sample == 0.0 || sample == 2.0) && throw(NativeTargetFailure(sample))
    return isfinite(sample) ? 0.0 : -Inf
end

struct NativeMaximumTarget{T} end

function (::NativeMaximumTarget{T})(::T)::T where {T}
    return floatmax(T)
end

struct FixedOverflowProposal{T}
    sample::T
end

Random.rand(::Random.AbstractRNG, proposal::FixedOverflowProposal) = proposal.sample
DensityInterface.logdensityof(proposal::FixedOverflowProposal{T}, ::T) where {T} =
    -floatmax(T)

mutable struct NativeCallCounterTarget{T}
    calls::Int
end

function (target::NativeCallCounterTarget{T})(::T)::T where {T}
    target.calls += 1
    return zero(T)
end

struct NativeFailFromScalarTarget
    first_failure::Float64
end

function (target::NativeFailFromScalarTarget)(sample::Float64)::Float64
    sample >= target.first_failure && throw(NativeTargetFailure(sample))
    return -abs2(sample)
end

struct NativeFailAtTwoScalarTarget
    first_failure::Float64
    second_failure::Float64
end

function (target::NativeFailAtTwoScalarTarget)(sample::Float64)::Float64
    (sample == target.first_failure || sample == target.second_failure) &&
        throw(NativeTargetFailure(sample))
    return -abs2(sample)
end

mutable struct NativeResetTarget
    fail::Bool
end

function (target::NativeResetTarget)(sample::Float64)::Float64
    target.fail && sample == 1.0 && throw(NativeTargetFailure(sample))
    return -abs2(sample)
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

mutable struct PrefilledRadialRNG{T} <: Random.AbstractRNG
    uniforms::Vector{T}
    normals::Vector{T}
end

function Random.rand!(rng::PrefilledRadialRNG, values::AbstractArray)
    copyto!(values, rng.uniforms)
    return values
end

function Random.randn!(rng::PrefilledRadialRNG, values::AbstractArray)
    copyto!(values, rng.normals)
    return values
end

mutable struct BulkFillRecorder <: Random.AbstractRNG
    calls::Vector{Symbol}
end

function Random.rand!(rng::BulkFillRecorder, values::AbstractArray)
    push!(rng.calls, :uniform)
    fill!(values, 0.25)
    return values
end

function Random.randn!(rng::BulkFillRecorder, values::AbstractArray)
    push!(rng.calls, :normal)
    fill!(values, -0.25)
    return values
end

struct OwnedBufferAccelerator <: MLDataDevices.AbstractAcceleratorDevice end
MLDataDevices.functional(::OwnedBufferAccelerator) = true
(::OwnedBufferAccelerator)(values::Vector) = copy(values)
(::OwnedBufferAccelerator)(proposal::ImportanceSamplers._GaussianProposal) =
    deepcopy(proposal)

@eval ImportanceSamplers begin
    _owned_backend_rng(::Main.OwnedBufferAccelerator, seed::UInt64) =
        Random.Xoshiro(seed)
    _factor_batch_supported(::Main.OwnedBufferAccelerator) = true
end

struct SharedRNGAccelerator <: MLDataDevices.AbstractAcceleratorDevice end
MLDataDevices.functional(::SharedRNGAccelerator) = true

const SHARED_BACKEND_TEST_RNG = Random.Xoshiro(0)
MLDataDevices.default_device_rng(::SharedRNGAccelerator) =
    SHARED_BACKEND_TEST_RNG

function _caught_kernel_execution_error(f)
    try
        f()
    catch error
        return error
    end
    return nothing
end

@kernel function _exercise_failure_record!(storage)
    slot = @index(Global, Linear)
    logical_index = slot == 1 ? 7 : slot == 2 ? 3 : 5
    block = slot == 1 ? 2 : slot == 2 ? 4 : 1
    reason_bits = slot == 1 ? UInt16(0x0001) :
                  slot == 2 ? UInt16(0x0400) : UInt16(0x0800)
    ImportanceSamplers._record_native_failure!(
        storage, logical_index, block, reason_bits
    )
end

function _run_native_fused(
    target,
    proposal,
    normal_buffer,
    threaded,
    nsamples=length(normal_buffer);
    factor_execution=FusedFactorExecution(),
)
    sampler = prepare_sampler(
        PrefilledNormalRNG(copy(normal_buffer), 0),
        target,
        ImportanceSampling(proposal; nsamples=nsamples);
        factor_execution,
        threaded=threaded,
    )
    return @inferred importance_sample!(sampler)
end

function _run_native_factor_batch(target, proposal, normal_buffer)
    dimension = ImportanceSamplers._proposal_dimension(proposal)
    nsamples = length(normal_buffer) ÷ dimension
    samples = Matrix{eltype(normal_buffer)}(undef, dimension, nsamples)
    logweights = Vector{eltype(normal_buffer)}(undef, nsamples)
    failure_record = ImportanceSamplers._DeviceFailureRecord(zeros(UInt64, 3))
    target_evaluator = ImportanceSamplers._NativeDeviceTarget{
        eltype(logweights),
        typeof(target),
    }(target)
    ImportanceSamplers._launch_native_factor_batch!(
        samples,
        logweights,
        failure_record,
        copy(normal_buffer),
        target_evaluator,
        proposal,
        ImportanceSamplers._SerialCPUExecution(),
    )
    return (; samples, logweights, failures=failure_record.storage)
end

function _native_unfused_reference(
    target, proposal, normal_buffer::Vector{T}, nsamples=length(normal_buffer)
) where {T}
    sampler = prepare_sampler(
        PrefilledNormalRNG(copy(normal_buffer), 0),
        target,
        ImportanceSampling(proposal; nsamples=nsamples);
        threaded=false,
    )
    samples, logweights = ImportanceSamplers._importance_sample_generic_cpu!(sampler, false)
    return (samples=samples, logweights=logweights)
end

function _check_vector_native_equivalence(::Type{T}, proposal, normals, target) where {T}
    dimension = ImportanceSamplers._proposal_dimension(
        proposal isa TransformedProposal ? proposal.base : proposal,
    )
    nsamples = length(normals) ÷ dimension
    fused = @inferred _run_native_fused(target, proposal, normals, false, nsamples)
    reference = @inferred _native_unfused_reference(
        target, proposal, normals, nsamples
    )
    tolerance = T(128) * eps(T)
    @test fused.samples == reference.samples
    @test fused.logweights ≈ reference.logweights rtol = tolerance atol = tolerance
    return fused
end

@testset "native factor batch matches fused factor execution" begin
    for T in (Float32, Float64)
        proposal = FactorGaussian(T[0.25, -0.5], T[1.25 0; -0.4 0.75])
        normals = T[1, -2, -1, 2, 0.5, -0.5, -0.5, 0.5]
        target = FusedVectorTarget(zero(T))
        fused = _run_native_fused(target, proposal, normals, false, 4)
        batch = _run_native_factor_batch(target, proposal, normals)
        selected = _run_native_fused(
            target,
            proposal,
            normals,
            false,
            4;
            factor_execution=BatchedFactorExecution(),
        )
        @test batch.samples ≈ fused.samples rtol = 8eps(T)
        @test batch.logweights ≈ fused.logweights rtol = 32eps(T)
        @test selected.samples ≈ fused.samples rtol = 8eps(T)
        @test selected.logweights ≈ fused.logweights rtol = 32eps(T)
        @test iszero(batch.failures)
    end

    proposal = FactorGaussian(
        [0.3, -0.7],
        [1.0e-4 0.0; 1.0e4 1.0e-4],
    )
    target = FusedVectorTarget(0.0)
    selected = _run_native_fused(
        target,
        proposal,
        [0.25, -0.5, -0.75, 1.25],
        false,
        2;
        factor_execution=BatchedFactorExecution(),
    )
    expected = map(eachcol(selected.samples)) do sample
        target(sample) - DensityInterface.logdensityof(proposal, sample)
    end
    @test selected.logweights ≈ expected rtol = 64eps(Float64)
end

@testset "native factor batch preserves target failure priority" begin
    samples = zeros(1, 1)
    logweights = [-Inf]
    failures = zeros(UInt64, 3)
    target_function = sample -> NaN
    target = ImportanceSamplers._NativeDeviceTarget{
        Float64,
        typeof(target_function),
    }(target_function)
    backend = KernelAbstractions.CPU()
    kernel = ImportanceSamplers._native_factor_batch_finish_kernel!(backend)
    kernel(
        samples,
        logweights,
        failures,
        target;
        ndrange=1,
        workgroupsize=1,
    )
    KernelAbstractions.synchronize(backend)
    decoded = ImportanceSamplers._decode_native_failure(failures[1], failures[2])
    @test decoded.reason_bits == ImportanceSamplers._NATIVE_TARGET_NAN
end

@testset "prepared sampler selects factor execution" begin
    dimension = 32
    nsamples = 4096
    proposal = FactorGaussian(
        zeros(dimension),
        Matrix{Float64}(LinearAlgebra.I, dimension, dimension),
    )
    sampler = prepare_sampler(
        Random.Xoshiro(0x5010),
        FusedVectorTarget(0.0),
        ImportanceSampling(proposal; nsamples);
        factor_execution=BatchedFactorExecution(),
        threaded=true,
    ) |> OwnedBufferAccelerator()

    default_sampler = prepare_sampler(
        Random.Xoshiro(0x5010),
        FusedVectorTarget(0.0),
        ImportanceSampling(proposal; nsamples);
        threaded=true,
    ) |> OwnedBufferAccelerator()
    default_result = importance_sample!(default_sampler)
    @test default_result.diagnostics.factor_execution_policy === :batched

    result = importance_sample!(sampler)
    @test size(result.samples) == (dimension, nsamples)
    @test all(isfinite, result.logweights)

    factor32 = FactorGaussian(
        zeros(Float32, dimension),
        Matrix{Float32}(LinearAlgebra.I, dimension, dimension),
    )
    mixed = prepare_sampler(Random.Xoshiro(41), x -> 1e8 + 0.1,
        ImportanceSampling(ProposalBank([factor32, factor32]); nsamples=32);
        factor_execution=BatchedFactorExecution())
    weights = importance_sample!(mixed)
    expected = 1e8 .+ 0.1 .+ dimension * log(2pi)/2 .+
        vec(sum(abs2, Float64.(weights.samples); dims=1)) ./ 2
    @test weights.logweights ≈ expected atol=1e-4 rtol=0
    @test weights.diagnostics.factor_execution_policy === :batched
end

_view_backed_scale(scale::ImportanceSamplers._SphericalGaussianScale) = scale
_view_backed_scale(scale::ImportanceSamplers._DiagonalGaussianScale) =
    ImportanceSamplers._DiagonalGaussianScale(view(scale.scales, :))
_view_backed_scale(scale::ImportanceSamplers._FactorGaussianScale) =
    ImportanceSamplers._FactorGaussianScale(view(scale.factor, :, :))

function _view_backed_gaussian(proposal)
    return ImportanceSamplers._GaussianProposal(
        proposal.family,
        view(proposal.location, :),
        _view_backed_scale(proposal.scale),
        proposal.lognormalizer,
    )
end

function _view_backed_proposal(proposal::TransformedProposal)
    return ImportanceSamplers.TransformedProposal(
        _view_backed_gaussian(proposal.base),
        proposal.transform,
        ImportanceSamplers._PreparedProposalToken(),
    )
end

_view_backed_proposal(proposal) = _view_backed_gaussian(proposal)

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
        coordinates,
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
    reference = @inferred _native_unfused_reference(target, proposal, normals)

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

@testset "native scalar fused execution" begin
    fused32 = _check_scalar_native_equivalence(Float32)
    fused64 = _check_scalar_native_equivalence(Float64)
    @test fused32.samples isa Vector{Float32}
    @test fused32.logweights isa Vector{Float32}
    @test fused64.samples isa Vector{Float64}
    @test fused64.logweights isa Vector{Float64}
end

@testset "portable Gaussian generation and fused density" begin
    for T in (Float32, Float64)
        normals = T[-1, 0.5, 2, -0.25, 0.75, -1.5]
        target = FusedVectorTarget(T(0.75))
        proposals = (
            SphericalGaussian(T[0.25, -0.5], T(1.5)),
            DiagonalGaussian(T[0.25, -0.5], T[0.5, 2]),
            FactorGaussian(T[0.25, -0.5], T[1.25 0; -0.4 0.75]),
            TransformedProposal(
                SphericalGaussian(T[0.25, -0.5], T(1.5)),
                IdentityTransform(),
            ),
        )

        for proposal in proposals
            @test ImportanceSamplers._sampling_execution(proposal, false) isa
                  ImportanceSamplers._KernelExecution
            fused = _check_vector_native_equivalence(T, proposal, normals, target)
            backend_shaped = @inferred _run_native_fused(
                target, _view_backed_proposal(proposal), normals, false, 3
            )
            @test backend_shaped.samples == fused.samples
            @test backend_shaped.logweights == fused.logweights
        end
    end
end

@testset "portable Student-t generation and fused density" begin
    proposal = SphericalStudentT(1.0, 0.0, 2.0)
    algorithm = ImportanceSampling(proposal; nsamples=2)
    sampler = prepare_sampler(
        PrefilledRadialRNG(Float64[], [1.0, 2.0, -1.0, 4.0]),
        proposal,
        algorithm;
        threaded=false,
    )

    @test ImportanceSamplers._sampling_execution(proposal, false) isa
          ImportanceSamplers._KernelExecution
    result = @inferred importance_sample!(sampler)
    @test result.samples == [1.0, -0.5]
    @test result.logweights ≈ zeros(2) atol=8eps(Float64)

    exhausted_proposal = SphericalStudentT(2.0, 0.0, 1.0)
    exhausted = prepare_sampler(
        PrefilledRadialRNG(ones(8), vcat(0.0, fill(-10.0, 8))),
        exhausted_proposal,
        ImportanceSampling(exhausted_proposal; nsamples=1);
        threaded=false,
    )
    error = try
        importance_sample!(exhausted)
        nothing
    catch caught
        caught
    end
    @test error isa SamplerExecutionError
    @test error.phase == :proposal_draw
end

@testset "portable block transforms produce aligned logical samples" begin
    for T in (Float32, Float64)
        simplex = TransformedProposal(
            SphericalGaussian(zeros(T, 2), one(T)),
            SimplexTransform(3),
        )
        simplex_normals = T[-0.5, 0.25, 0, 0, 0.75, -1.25]
        simplex_result = _check_vector_native_equivalence(
            T,
            simplex,
            simplex_normals,
            FusedVectorTarget(T(0.5)),
        )
        @test all(isone, vec(sum(simplex_result.samples; dims=1)))
        backend_simplex = @inferred _run_native_fused(
            FusedVectorTarget(T(0.5)),
            _view_backed_proposal(simplex),
            simplex_normals,
            false,
            3,
        )
        @test backend_simplex.samples == simplex_result.samples
        @test backend_simplex.logweights == simplex_result.logweights

        factor = T[
            1 0 0 0
            0.25 1.25 0 0
            -0.5 0.2 0.75 0
            0.1 -0.3 0.4 1.5
        ]
        named = TransformedProposal(
            FactorGaussian(zeros(T, 4), factor),
            (
                weights=(1:2 => SimplexTransform(3)),
                rate=(3 => PositiveTransform()),
                offset=(4 => IdentityTransform()),
            ),
        )
        named_normals = T[
            -0.5, 0.25, 0.1, -0.75,
            0, 0, -0.25, 0.5,
            0.75, -1.25, 0.5, 0.25,
        ]
        named_result = _check_vector_native_equivalence(
            T,
            named,
            named_normals,
            FusedNamedTarget{T}(),
        )
        @test all(isone, vec(sum(named_result.samples.weights; dims=1)))
        backend_named = @inferred _run_native_fused(
            FusedNamedTarget{T}(),
            _view_backed_proposal(named),
            named_normals,
            false,
            3,
        )
        @test backend_named.samples == named_result.samples
        @test backend_named.logweights == named_result.logweights
    end
end

@testset "portable failure record and target minus infinity" begin
    failure_scratch = ImportanceSamplers._allocate_native_failure_scratch(zeros(1), 3)
    ImportanceSamplers._reset_native_failure_scratch!(failure_scratch)
    failure_record = getfield(failure_scratch, :record)
    record_backend = KernelAbstractions.get_backend(getfield(failure_record, :storage))
    _exercise_failure_record!(record_backend)(failure_record.storage; ndrange=3)
    KernelAbstractions.synchronize(record_backend)
    snapshot = ImportanceSamplers._device_failure_snapshot(failure_record)
    @test snapshot.failure.count == 3
    @test snapshot.failure.first_logical_index == 3
    @test snapshot.failure.first_block == 4
    @test snapshot.failure.reason_bits == UInt16(0x0400)
    @test snapshot.draw_failure.count == 1
    @test snapshot.draw_failure.first_logical_index == 5
    @test snapshot.draw_failure.first_block == 1
    @test snapshot.draw_failure.reason_bits == UInt16(0x0800)
    @test snapshot.transfers == (count=0, bytes=0)

    for T in (Float32, Float64)
        proposal = SphericalGaussian(zeros(T, 2), one(T))
        valid = @inferred _run_native_fused(
            MinusInfVectorTarget{T}(),
            proposal,
            T[0, 1, 2, 3],
            true,
            2,
        )
        @test valid.logweights[1] === T(-Inf)
        @test isfinite(valid.logweights[2])

        invalid = TransformedProposal(
            SphericalGaussian(zeros(T, 2), T(floatmax(T))),
            SimplexTransform(3),
        )
        failure = _caught_kernel_execution_error() do
            _run_native_fused(
                FusedVectorTarget(zero(T)),
                invalid,
                T[0, 0, 1, 1],
                true,
                2,
            )
        end
        @test failure isa SamplerExecutionError
        @test failure.phase === :proposal_draw
        @test failure.sample_index == 2
        @test failure.captured.ex isa InvalidTransformError
        @test failure.captured.ex.location == 1:2
        @test failure.captured.ex.reason === :nonfinite_output

    end
end

@testset "derived nonfinite log weights fail consistently" begin
    for T in (Float32, Float64)
        invalid_cases = (
            (T(-Inf), T(-Inf), T(NaN)),
            (floatmax(T), -floatmax(T), T(Inf)),
        )
        for (target_log, proposal_log, expected) in invalid_cases
            logweight, reason = ImportanceSamplers._subtract_logweight(
                target_log,
                proposal_log,
            )
            @test isequal(logweight, expected)
            @test !iszero(reason)
        end
        preserved, reason = ImportanceSamplers._subtract_logweight(T(-Inf), T(Inf))
        @test preserved === T(-Inf)
        @test iszero(reason)

        for threaded in (false, true)
            generic_failure = _caught_kernel_execution_error() do
                importance_sample(
                    Random.Xoshiro(0x9301),
                    NativeMaximumTarget{T}(),
                    ImportanceSampling(FixedOverflowProposal(zero(T)); nsamples=2);
                    threaded,
                )
            end
            @test generic_failure isa SamplerExecutionError
            @test (generic_failure.phase, generic_failure.sample_index) == (:logweight, 1)
            @test generic_failure.captured.ex isa DomainError

            overflow_normal = T(0.75) * sqrt(floatmax(T))
            fused_failure = _caught_kernel_execution_error() do
                _run_native_fused(
                    NativeMaximumTarget{T}(),
                    SphericalGaussian(zero(T), one(T)),
                    T[overflow_normal, zero(T)],
                    threaded,
                    2,
                )
            end
            @test fused_failure isa SamplerExecutionError
            @test (fused_failure.phase, fused_failure.sample_index) == (:logweight, 1)
            @test fused_failure.captured.ex isa DomainError
        end
    end
end

@testset "nonfinite Gaussian draws fail before target evaluation" begin
    for T in (Float32, Float64), transformed in (false, true)
        target = NativeCallCounterTarget{T}(0)
        gaussian = SphericalGaussian(zero(T), one(T))
        proposal = transformed ?
                   TransformedProposal(gaussian, IdentityTransform()) : gaussian
        failure = _caught_kernel_execution_error() do
            _run_native_fused(target, proposal, T[Inf], false)
        end
        @test failure isa SamplerExecutionError
        @test (failure.phase, failure.sample_index) == (:proposal_draw, 1)
        @test failure.captured.ex isa DomainError
        @test target.calls == 0
    end
end

@testset "native CPU target failures retain binding, phase, and index" begin
    proposal = SphericalGaussian(zeros(2), 1.0)
    binding_failure = _caught_kernel_execution_error() do
        _run_native_fused(NativeVectorOnlyTarget(), proposal, zeros(2), false, 1)
    end
    @test binding_failure isa SamplerExecutionError
    if binding_failure isa SamplerExecutionError
        @test (binding_failure.phase, binding_failure.sample_index) == (:target, 1)
        @test binding_failure.captured.ex isa ArgumentError
    end

    target_failure = _caught_kernel_execution_error() do
        _run_native_fused(
            NativeFailAtVectorTarget(1.0),
            proposal,
            [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            false,
            3,
        )
    end
    @test target_failure isa SamplerExecutionError
    if target_failure isa SamplerExecutionError
        @test (target_failure.phase, target_failure.sample_index) == (:target, 2)
        @test target_failure.captured.ex isa NativeTargetFailure
        @test target_failure.captured.ex.sample == 1.0
    end

    phase_order_failure = _caught_kernel_execution_error() do
        _run_native_fused(
            NativePhaseOrderTarget(),
            SphericalGaussian(0.0, 1.0),
            [0.0, Inf, 2.0],
            true,
        )
    end
    @test phase_order_failure isa SamplerExecutionError
    if phase_order_failure isa SamplerExecutionError
        @test (phase_order_failure.phase, phase_order_failure.sample_index) ==
              (:proposal_draw, 2)
        @test phase_order_failure.captured.ex isa DomainError
        @test phase_order_failure.captured.ex.val == UInt16(0x0800)
    end

    for threaded in (false, true)
        proposal = SphericalGaussian(0.0, 1.0)
        concurrent = threaded && Threads.nthreads(:default) > 1
        if concurrent
            execution = ImportanceSamplers._sampling_execution(proposal, true)
            @test getfield(execution, :cpu_execution) isa
                  ImportanceSamplers._ThreadedCPUExecution
        end
        nsamples = concurrent ? 2_049 : 4
        second_failure = concurrent ? 2_048.0 : 2.0
        first_failure = _caught_kernel_execution_error() do
            _run_native_fused(
                NativeFailAtTwoScalarTarget(1.0, second_failure),
                proposal,
                collect(0.0:(nsamples - 1)),
                threaded,
            )
        end
        @test first_failure isa SamplerExecutionError
        if first_failure isa SamplerExecutionError
            @test (first_failure.phase, first_failure.sample_index) == (:target, 2)
            @test first_failure.captured.ex isa NativeTargetFailure
            @test first_failure.captured.ex.sample == 1.0
        end
    end
end

@testset "portable native execution allocations" begin
    proposal = DiagonalGaussian([0.25, -0.5], [0.5, 2.0])
    normals = repeat([-1.0, 0.5], 64)
    target = FusedVectorTarget(0.75)
    _run_native_fused(target, proposal, normals, false, 64)
    allocation = @allocated _run_native_fused(
        target,
        proposal,
        normals,
        false,
        64,
    )
    @test allocation <= 12_000
end

@testset "native random buffers are bulk-filled and reused" begin
    proposal = SphericalGaussian(0.25, 1.5)
    algorithm = ImportanceSampling(proposal; nsamples=32)
    rng = Random.Xoshiro(0x9109)
    sampler = prepare_sampler(
        rng,
        FusedQuadraticTarget(0.75),
        algorithm;
        threaded=false,
    )

    buffers = getfield(sampler, :random_buffers)
    normal_buffer = getfield(buffers, :normal)

    first_result = @inferred importance_sample!(sampler)
    second_result = @inferred importance_sample!(sampler)
    @test getfield(getfield(sampler, :random_buffers), :normal) === normal_buffer
    @test first_result.samples != second_result.samples

    replay = @inferred importance_sample!(
        prepare_sampler(
            Random.Xoshiro(0x9109),
            FusedQuadraticTarget(0.75),
            algorithm;
            threaded=false,
        ),
    )
    @test replay.samples == first_result.samples
    @test replay.logweights == first_result.logweights
end

@testset "native failure scratch is reused and results stay fresh" begin
    for nsamples in (1_000, 100_000)
        algorithm = ImportanceSampling(
            SphericalGaussian(0.25, 1.5);
            nsamples=nsamples,
        )
        sampler = prepare_sampler(
            Random.Xoshiro(0x9150 + nsamples),
            FusedQuadraticTarget(0.75),
            algorithm;
            threaded=false,
        )
        scratch = getfield(getfield(sampler, :random_buffers), :failure_scratch)
        failure_storage = getfield(getfield(scratch, :record), :storage)
        target_failures = getfield(scratch, :target_failures)

        target_failure_slots = getfield(target_failures, :slots)
        target_failure_index = getfield(target_failures, :first_index)

        first_result = @inferred importance_sample!(sampler)
        second_result = @inferred importance_sample!(sampler)

        @test getfield(getfield(sampler, :random_buffers), :failure_scratch) ===
              scratch
        @test getfield(getfield(scratch, :record), :storage) === failure_storage
        @test getfield(scratch, :target_failures) === target_failures
        @test length(target_failure_slots) == nsamples
        @test target_failure_index[] == typemax(Int)
        @test first_result.samples !== second_result.samples
        @test first_result.logweights !== second_result.logweights
        @test first_result.diagnostics.transfers !==
              second_result.diagnostics.transfers
        @test first_result.provenance == second_result.provenance == NamedTuple()
    end
end

@testset "native failure scratch resets after failed execution" begin
    nsamples = 3
    target = NativeResetTarget(true)
    sampler = prepare_sampler(
        PrefilledNormalRNG([0.0, 1.0, 2.0, 0.0, 1.0, 2.0], 0),
        target,
        ImportanceSampling(SphericalGaussian(0.0, 1.0); nsamples=nsamples);
        threaded=false,
    )
    scratch = getfield(getfield(sampler, :random_buffers), :failure_scratch)
    failure_storage = getfield(getfield(scratch, :record), :storage)
    target_failures = getfield(scratch, :target_failures)
    target_failure_slots = getfield(target_failures, :slots)
    target_failure_index = getfield(target_failures, :first_index)

    failure = _caught_kernel_execution_error() do
        importance_sample!(sampler)
    end
    @test failure isa SamplerExecutionError
    if failure isa SamplerExecutionError
        @test (failure.phase, failure.sample_index) == (:target, 2)
        @test failure.captured.ex isa NativeTargetFailure
    end
    @test all(isnothing, target_failure_slots)
    @test target_failure_index[] == typemax(Int)

    target.fail = false
    result = @inferred importance_sample!(sampler)
    @test length(result) == nsamples
    @test getfield(getfield(sampler, :random_buffers), :failure_scratch) === scratch
    @test getfield(getfield(scratch, :record), :storage) === failure_storage
    @test getfield(scratch, :target_failures) === target_failures
    @test all(isnothing, target_failure_slots)
    @test target_failure_index[] == typemax(Int)
    @test failure_storage == zeros(UInt64, 3)

    transform_sampler = prepare_sampler(
        PrefilledNormalRNG([Inf, 0.0], 0),
        FusedConstantTarget{Float64}(),
        ImportanceSampling(
            TransformedProposal(
                SphericalGaussian(0.0, 1.0),
                IdentityTransform(),
            );
            nsamples=1,
        );
        threaded=false,
    )
    transform_scratch = getfield(
        getfield(transform_sampler, :random_buffers),
        :failure_scratch,
    )
    transform_storage = getfield(getfield(transform_scratch, :record), :storage)
    transform_failure = _caught_kernel_execution_error() do
        importance_sample!(transform_sampler)
    end
    @test transform_failure isa SamplerExecutionError
    @test (transform_failure.phase, transform_failure.sample_index) ==
          (:proposal_draw, 1)
    @test transform_failure.captured.ex isa DomainError
    @test transform_storage[1] == 1
    @test transform_storage[2] & UInt64(0xffff) == UInt64(0x0800)
    @test transform_storage[3] & UInt64(0xffff) == UInt64(0x0800)

    transform_result = @inferred importance_sample!(transform_sampler)
    @test length(transform_result) == 1
    @test transform_storage == zeros(UInt64, 3)

    mixed_sampler = prepare_sampler(
        PrefilledNormalRNG([Inf, 1.0, 0.0, 0.0], 0),
        NativeFailFromScalarTarget(1.0),
        ImportanceSampling(
            TransformedProposal(
                SphericalGaussian(0.0, 1.0),
                IdentityTransform(),
            );
            nsamples=2,
        );
        threaded=true,
    )
    mixed_scratch = getfield(
        getfield(mixed_sampler, :random_buffers),
        :failure_scratch,
    )
    mixed_failures = getfield(mixed_scratch, :target_failures)
    mixed_failure = _caught_kernel_execution_error() do
        importance_sample!(mixed_sampler)
    end
    @test mixed_failure isa SamplerExecutionError
    if mixed_failure isa SamplerExecutionError
        @test (mixed_failure.phase, mixed_failure.sample_index) == (:proposal_draw, 1)
        @test mixed_failure.captured.ex isa DomainError
    end
    @test all(isnothing, getfield(mixed_failures, :slots))
    @test getfield(mixed_failures, :first_index)[] == typemax(Int)

    mixed_result = @inferred importance_sample!(mixed_sampler)
    @test length(mixed_result) == 2
end

@testset "native CPU target failure marker avoids clean full-array passes" begin
    nsamples = 8
    sampler = prepare_sampler(
        PrefilledNormalRNG(zeros(2nsamples), 0),
        FusedQuadraticTarget(0.75),
        ImportanceSampling(SphericalGaussian(0.0, 1.0); nsamples);
        threaded=false,
    )
    scratch = getfield(getfield(sampler, :random_buffers), :failure_scratch)
    target_failures = getfield(scratch, :target_failures)
    slots = getfield(target_failures, :slots)
    first_index = getfield(target_failures, :first_index)
    stale = SamplerExecutionError(
        :target,
        nsamples,
        CapturedException(NativeTargetFailure(9.0), backtrace()),
    )

    slots[end] = stale
    clean = @inferred importance_sample!(sampler)
    @test length(clean) == nsamples
    @test slots[end] === stale
    @test first_index[] == typemax(Int)

    first_index[] = 2
    ImportanceSamplers._reset_native_failure_scratch!(scratch)
    @test all(isnothing, slots)
    @test first_index[] == typemax(Int)

    recovered = @inferred importance_sample!(sampler)
    @test length(recovered) == nsamples
    @test all(isnothing, slots)
    @test first_index[] == typemax(Int)
end

@testset "native serial and threaded prepared execution remain equivalent" begin
    algorithm = ImportanceSampling(
        DiagonalGaussian([0.25, -0.5], [0.5, 2.0]);
        nsamples=1_000,
    )
    serial = prepare_sampler(
        Random.Xoshiro(0x9151),
        FusedVectorTarget(0.75),
        algorithm;
        threaded=false,
    )
    threaded = prepare_sampler(
        Random.Xoshiro(0x9151),
        FusedVectorTarget(0.75),
        algorithm;
        threaded=true,
    )

    serial_result = @inferred importance_sample!(serial)
    threaded_result = @inferred importance_sample!(threaded)
    @test serial_result.samples == threaded_result.samples
    @test serial_result.logweights == threaded_result.logweights
    @test serial_result.diagnostics.execution === :serial
    @test threaded_result.diagnostics.execution ===
          (Threads.nthreads(:default) > 1 ? :threaded : :serial)
end

@testset "random buffers use the standard bulk fill APIs" begin
    rng = BulkFillRecorder(Symbol[])
    buffers = ImportanceSamplers._RandomBuffers(
        zeros(8),
        zeros(8),
        ImportanceSamplers._NoNativeFailureScratch(),
        nothing,
    )

    @test ImportanceSamplers._fill_random_buffers!(rng, buffers) === buffers
    @test rng.calls == [:uniform, :normal]
end

@testset "accelerator transfer owns one seeded stream and destination buffers" begin
    device = OwnedBufferAccelerator()
    algorithm = ImportanceSampling(
        SphericalGaussian(0.25, 1.5);
        nsamples=32,
    )
    source_rng = Random.Xoshiro(0x9111)
    expected_source = copy(source_rng)
    expected_seed = rand(expected_source, UInt64)
    expected_next = rand(expected_source, UInt64)
    source = prepare_sampler(
        source_rng,
        FusedQuadraticTarget(0.75),
        algorithm;
        threaded=true,
    )

    destination = device(source)
    destination_rng = getfield(destination, :rng)
    destination_buffers = getfield(destination, :random_buffers)
    @test destination_rng isa Random.Xoshiro
    @test destination_rng !== getfield(source, :rng)
    @test getfield(source, :rng) !== source_rng
    @test rand(source_rng, UInt64) == expected_seed
    @test rand(getfield(source, :rng), UInt64) == expected_next
    @test getfield(destination_buffers, :uniform) isa Vector{Float64}
    @test getfield(destination_buffers, :normal) isa Vector{Float64}
    @test getfield(destination_buffers, :normal) !==
          getfield(getfield(source, :random_buffers), :normal)

    normal_buffer = getfield(destination_buffers, :normal)
    expected_buffer = similar(normal_buffer)
    Random.randn!(Random.Xoshiro(expected_seed), expected_buffer)
    ImportanceSamplers._fill_random_buffers!(destination_rng, destination_buffers)
    @test collect(normal_buffer) == collect(expected_buffer)
    @test getfield(destination_buffers, :normal) === normal_buffer

    @test MLDataDevices.default_device_rng(SharedRNGAccelerator()) ===
          MLDataDevices.default_device_rng(SharedRNGAccelerator())
    shared_source = prepare_sampler(
        Random.Xoshiro(0x9112),
        FusedQuadraticTarget(0.75),
        algorithm;
        threaded=true,
    )
    expected_shared_rng = copy(getfield(shared_source, :rng))
    shared_error = _caught_kernel_execution_error() do
        SharedRNGAccelerator()(shared_source)
    end
    @test shared_error isa SamplerDeviceError
    @test shared_error.reason === :accelerator_rng_unavailable
    @test rand(getfield(shared_source, :rng), UInt64) ==
          rand(expected_shared_rng, UInt64)
end

@testset "accelerator launch policy is independent of host threads" begin
    sampler = OwnedBufferAccelerator()(
        prepare_sampler(
            Random.Xoshiro(0x9113),
            FusedQuadraticTarget(0.75),
            ImportanceSampling(
                SphericalGaussian(0.25, 1.5);
                nsamples=2_048,
            );
            threaded=true,
        ),
    )
    result = importance_sample!(sampler)
    @test result.diagnostics.execution === :threaded
    @test length(result) == 2_048
end

@testset "core CUDA RNG rejection without CUDA loaded" begin
    @test Base.get_extension(
        ImportanceSamplers,
        :ImportanceSamplersCUDAExt,
    ) === nothing
    error = _caught_kernel_execution_error() do
        ImportanceSamplers._owned_backend_rng(
            MLDataDevices.CUDADevice(),
            UInt64(0x9191),
        )
    end
    @test error isa SamplerDeviceError
    @test error.reason === :accelerator_rng_unavailable
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

@testset "native density is canonical at the returned finite-precision sample" begin
    for (T, location) in ((Float32, Float32(1e30)), (Float64, Float64(1e300)))
        proposal = SphericalGaussian(location, one(T))
        normals = T[1]
        fused = @inferred _run_native_fused(FusedConstantTarget{T}(), proposal, normals, false)
        reference = @inferred _native_unfused_reference(
            FusedConstantTarget{T}(), proposal, normals
        )
        @test fused.samples == reference.samples == T[location]
        @test fused.logweights == reference.logweights
    end

    for (T, normals) in (
        (Float32, Float32[60, 60]),
        (Float64, Float64[-450, -350]),
    )
        proposal = TransformedProposal(
            SphericalGaussian(zeros(T, 2), one(T)),
            SimplexTransform(3),
        )
        _check_vector_native_equivalence(
            T,
            proposal,
            normals,
            FusedVectorTarget(zero(T)),
        )
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
          ImportanceSamplers._KernelExecution
    @test ImportanceSamplers._sampling_execution(transformed_vector, true) isa
          ImportanceSamplers._KernelExecution
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
