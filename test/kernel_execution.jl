import KernelAbstractions
import KernelAbstractions: @index, @kernel
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
    sample == 2.0 && throw(NativeTargetFailure(sample))
    return isfinite(sample) ? 0.0 : -Inf
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
end

struct SharedRNGAccelerator <: MLDataDevices.AbstractAcceleratorDevice end
MLDataDevices.functional(::SharedRNGAccelerator) = true

const SHARED_BACKEND_TEST_RNG = Random.Xoshiro(0)
MLDataDevices.default_device_rng(::SharedRNGAccelerator) =
    SHARED_BACKEND_TEST_RNG

mutable struct PointerBackedTestRNG <: Random.AbstractRNG
    state::Ptr{Nothing}
end

PointerBackedTestRNG(seed::UInt64) =
    PointerBackedTestRNG(Ptr{Nothing}(seed | one(UInt64)))

struct NestedPointerTestState
    state::Ptr{Nothing}
end

mutable struct NestedPointerBackedTestRNG <: Random.AbstractRNG
    state::NestedPointerTestState
end

NestedPointerBackedTestRNG(seed::UInt64) =
    NestedPointerBackedTestRNG(
        NestedPointerTestState(Ptr{Nothing}(seed | one(UInt64))),
    )

mutable struct ValueBackedTestRNG <: Random.AbstractRNG
    seed::UInt64
    counter::UInt64
end

ValueBackedTestRNG(seed::UInt64) = ValueBackedTestRNG(seed, zero(UInt64))

const CUDA_RNG_TEST_WITNESS =
    Ref{Random.AbstractRNG}(PointerBackedTestRNG(UInt64(1)))
MLDataDevices.default_device_rng(::MLDataDevices.CUDADevice) =
    CUDA_RNG_TEST_WITNESS[]

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
    reason_bits = UInt16(1) << slot
    ImportanceSamplers._record_native_failure!(
        storage, logical_index, block, reason_bits
    )
end

function _run_native_fused(
    target, proposal, normal_buffer, threaded, nsamples=length(normal_buffer)
)
    sampler = prepare_sampler(
        PrefilledNormalRNG(copy(normal_buffer), 0),
        target,
        ImportanceSampling(proposal; nsamples=nsamples);
        threaded=threaded,
    )
    return @inferred importance_sample!(sampler)
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
    failure_record = ImportanceSamplers._allocate_device_failure_record(zeros(1))
    record_backend = KernelAbstractions.get_backend(getfield(failure_record, :storage))
    _exercise_failure_record!(record_backend)(failure_record.storage; ndrange=3)
    KernelAbstractions.synchronize(record_backend)
    snapshot = ImportanceSamplers._device_failure_snapshot(failure_record)
    @test snapshot.count == 3
    @test snapshot.first_logical_index == 3
    @test snapshot.first_block == 4
    @test snapshot.reason_bits == UInt16(1) << 2

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
            [Inf, 0.0, 2.0],
            true,
        )
    end
    @test phase_order_failure isa SamplerExecutionError
    if phase_order_failure isa SamplerExecutionError
        @test (phase_order_failure.phase, phase_order_failure.sample_index) == (:target, 3)
        @test phase_order_failure.captured.ex isa NativeTargetFailure
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

@testset "random buffers use the standard bulk fill APIs" begin
    rng = BulkFillRecorder(Symbol[])
    buffers = ImportanceSamplers._RandomBuffers(zeros(8), zeros(8))

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

    destination = @inferred device(source)
    destination_rng = getfield(destination, :rng)
    destination_buffers = getfield(destination, :random_buffers)
    @test destination_rng isa Random.Xoshiro
    @test destination_rng !== getfield(source, :rng)
    @test rand(source_rng, UInt64) == expected_next
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
    shared_error = _caught_kernel_execution_error() do
        SharedRNGAccelerator()(
            prepare_sampler(
                Random.Xoshiro(0x9112),
                FusedQuadraticTarget(0.75),
                algorithm;
                threaded=true,
            ),
        )
    end
    @test shared_error isa SamplerDeviceError
    @test shared_error.reason === :accelerator_rng_unavailable
end

@testset "CUDA ownership rejects pointer-backed RNG state" begin
    CUDA_RNG_TEST_WITNESS[] = ValueBackedTestRNG(UInt64(1))
    owned = ImportanceSamplers._owned_backend_rng(
        MLDataDevices.CUDADevice(),
        UInt64(0x9191),
    )
    @test owned isa ValueBackedTestRNG
    @test (owned.seed, owned.counter) == (UInt64(0x9191), zero(UInt64))

    for witness in (
        PointerBackedTestRNG(UInt64(1)),
        NestedPointerBackedTestRNG(UInt64(1)),
    )
        CUDA_RNG_TEST_WITNESS[] = witness
        error = _caught_kernel_execution_error() do
            ImportanceSamplers._owned_backend_rng(
                MLDataDevices.CUDADevice(),
                UInt64(0x9191),
            )
        end
        @test error isa SamplerDeviceError
        if error isa SamplerDeviceError
            @test error.reason === :accelerator_rng_unavailable
        end
    end
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
