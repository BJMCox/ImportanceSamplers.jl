using CUDA
using ImportanceSamplers
using LinearAlgebra
using MLDataDevices
using Random
using Test

const IS = ImportanceSamplers

struct CUDAFirstOrderGRAMISTarget{T} end
struct CUDAFirstOrderGRAMISGradient end
struct CUDAFirstOrderGRAMISMeanOnlyTarget{T}
    locations::NTuple{2,T}
end
struct CUDAFirstOrderGRAMISZeroGradient end

function (::CUDAFirstOrderGRAMISTarget{T})(sample)::T where {T}
    value = zero(T)
    @inbounds for row in 1:length(sample)
        value += abs2(sample[row])
    end
    return -T(0.5) * value
end

function (::CUDAFirstOrderGRAMISGradient)(destination, sample)
    @inbounds for row in 1:length(destination)
        destination[row] = -sample[row]
    end
    return destination
end

function (target::CUDAFirstOrderGRAMISMeanOnlyTarget{T})(sample)::T where {T}
    at_left = sample[1] == target.locations[1] && iszero(sample[2])
    at_right = sample[1] == target.locations[2] && iszero(sample[2])
    return at_left || at_right ? zero(T) : T(-Inf)
end

function (::CUDAFirstOrderGRAMISZeroGradient)(destination, sample)
    @inbounds for row in 1:length(destination)
        destination[row] = zero(eltype(destination))
    end
    return destination
end

function gram_is_cuda_device()
    physical = CUDA.device()
    return MLDataDevices.CUDADevice{typeof(physical),Nothing}(physical)
end

function gram_is_covariance_batch(::Type{T}, dimension, proposal_count) where {T}
    covariances = Array{T}(undef, dimension, dimension, proposal_count)
    for proposal_slot in 1:proposal_count
        seed = reshape(
            T.(1:(dimension * dimension)),
            dimension,
            dimension,
        ) / T(dimension + proposal_slot + 3)
        covariances[:, :, proposal_slot] .=
            seed * transpose(seed) + T(dimension + proposal_slot) * I
    end
    return covariances
end

function gram_is_cuda_population_cholesky(::Type{T}, dimension, proposal_count) where {T}
    source = gram_is_covariance_batch(T, dimension, proposal_count)
    covariances = CuArray(source)
    factors = similar(covariances)
    info = CUDA.fill(Int32(-1), proposal_count)
    device = gram_is_cuda_device()

    IS._factor_population!(device, factors, covariances, info)
    CUDA.synchronize()

    wrapper_factors = [CuArray(source[:, :, slot]) for slot in 1:proposal_count]
    wrapper_factors, _ = CUDA.cuSOLVER.potrfBatched!(
        'L',
        wrapper_factors,
    )
    CUDA.synchronize()
    wrapper_lower = cat(
        (tril(Array(factor)) for factor in wrapper_factors)...;
        dims=3,
    )
    return (
        source,
        factors=Array(factors),
        info=Array(info),
        wrapper_factors=wrapper_lower,
    )
end


function gram_is_cuda_preflight(::Type{T}) where {T}
    bank = ProposalBank([
        FactorGaussian(T[-1, 0], T[1 0; 0.1 0.8]),
        FactorGaussian(T[1, 0], T[0.9 0; -0.2 1.1]),
    ])
    source = prepare_sampler(
        Random.Xoshiro(0x4752414d49534355),
        LogTarget(
            CUDAFirstOrderGRAMISTarget{T}();
            grad=CUDAFirstOrderGRAMISGradient(),
        ),
        FirstOrderGRAMIS(
            bank;
            rounds=1,
            round_size=8,
            repulsion_strength=T(0.1),
        );
        threaded=true,
    )
    prepared = gram_is_cuda_device()(source)
    return (
        prepared,
        resident=IS._backend_state_resident(
            prepared.device,
            IS._transferred_backend_state(
                prepared.algorithm,
                prepared.method_state,
                prepared.target,
                prepared.random_buffers,
            ),
        ),
    )
end

function gram_is_cuda_sampler(
    ::Type{T};
    repulsion_strength=zero(T),
    fallback=false,
) where {T}
    bank = ProposalBank([
        FactorGaussian(T[-1, 0], T[1 0; 0.1 0.8]),
        FactorGaussian(T[1, 0], T[0.9 0; -0.2 1.1]),
    ])
    target = fallback ?
             LogTarget(
        CUDAFirstOrderGRAMISMeanOnlyTarget{T}((-one(T), one(T)));
        grad=CUDAFirstOrderGRAMISZeroGradient(),
    ) : LogTarget(
        CUDAFirstOrderGRAMISTarget{T}();
        grad=CUDAFirstOrderGRAMISGradient(),
    )
    source = prepare_sampler(
        Random.Xoshiro(0x4752414d49534532),
        target,
        FirstOrderGRAMIS(
            bank;
            rounds=2,
            round_size=8,
            repulsion_strength=repulsion_strength,
        );
        threaded=true,
    )
    return gram_is_cuda_device()(source)
end

function gram_is_cuda_population_bits(sampler)
    state = sampler.method_state.committed
    return (
        locations=map(bitstring, vec(Array(state.locations))),
        factors=map(bitstring, vec(Array(state.factors))),
        lognormalizers=map(bitstring, Array(state.lognormalizers)),
    )
end

CUDA.allowscalar(false)

@testset "FirstOrderGRAMIS CUDA population Cholesky capabilities" begin
    caller_device = CUDA.device()
    for T in (Float32, Float64), dimension in (
        2,
        IS._GRAMIS_CHOLESKY_WORKGROUP_SIZE + 3,
    )
        result = gram_is_cuda_population_cholesky(T, dimension, 3)
        @test result.info == zeros(Int32, 3)
        @test result.factors ≈ result.wrapper_factors rtol =
            T === Float32 ? 3f-4 : 3e-12
    end
    @test CUDA.device() == caller_device
end


@testset "FirstOrderGRAMIS CUDA preflight and pooled adapters" begin
    caller_device = CUDA.device()
    for T in (Float32, Float64)
        preflight = gram_is_cuda_preflight(T)
        @test preflight.resident

        source = gram_is_covariance_batch(T, 3, 3)
        factors = cat(
            (
                Matrix(cholesky(Hermitian(source[:, :, slot])).L)
                for slot in 1:3
            )...;
            dims=3,
        )
        device_factors = CuArray(factors)
        pooled = CUDA.zeros(T, 3, 3)
        execution = IS._KernelExecution(IS._ThreadedCPUExecution())
        IS._pooled_covariance!(pooled, device_factors, execution)
        pooled_reference = dropdims(sum(source; dims=3); dims=3) / T(3)
        @test Array(pooled) ≈ pooled_reference rtol =
            T === Float32 ? 4f-5 : 4e-13

        IS._factor_pooled_covariance!(pooled)
        means = CuArray(reshape(T.(1:12), 3, 4) / T(7))
        whitened = similar(means)
        IS._whiten_means!(whitened, pooled, means)
        expected = cholesky(Hermitian(pooled_reference)).L \ Array(means)
        @test Array(whitened) ≈ expected rtol =
            T === Float32 ? 5f-5 : 5e-13
    end
    @test CUDA.device() == caller_device
end

@testset "FirstOrderGRAMIS CUDA end-to-end sampler" begin
    caller_device = CUDA.device()
    for (T, strength) in ((Float32, 0f0), (Float64, 0.1))
        sampler = gram_is_cuda_sampler(T; repulsion_strength=strength)
        result = importance_sample!(sampler)
        @test result.diagnostics.transfers.count <= 192
        @test result.diagnostics.transfers.bytes <= 4_096
        if !iszero(strength)
            @test IS._backend_state_resident(
                sampler.device,
                IS._transferred_backend_state(
                    sampler.algorithm,
                    sampler.method_state,
                    sampler.target,
                    sampler.random_buffers,
                ),
            )
        end
    end
    @test CUDA.device() == caller_device
end

@testset "FirstOrderGRAMIS CUDA fallback preserves factors" begin
    caller_device = CUDA.device()
    for T in (Float32, Float64)
        sampler = gram_is_cuda_sampler(T; fallback=true)
        before_factors = map(
            bitstring,
            vec(Array(sampler.method_state.committed.factors)),
        )
        result = importance_sample!(sampler)
        @test all(==(-Inf), Array(result.logweights))
        @test all(
            ==(IS._GRAMIS_ALL_ZERO_LOCAL),
            Array(result.diagnostics.fallback_status),
        )
        @test map(
            bitstring,
            vec(Array(sampler.method_state.committed.factors)),
        ) == before_factors
        @test result.diagnostics.transfers.count <= 192
        @test result.diagnostics.transfers.bytes <= 4_096
    end
    @test CUDA.device() == caller_device
end

@testset "FirstOrderGRAMIS CUDA covariance failure rolls back" begin
    caller_device = CUDA.device()
    sampler = gram_is_cuda_sampler(Float64)
    fill!(sampler.method_state.covariance_rate, NaN)
    before = gram_is_cuda_population_bits(sampler)
    failure = try
        importance_sample!(sampler)
        nothing
    catch error
        error
    end
    @test failure isa FirstOrderGRAMISRoundError
    @test failure.phase === :covariance
    @test failure.diagnostics.transfers.count <= 192
    @test failure.diagnostics.transfers.bytes <= 4_096
    @test gram_is_cuda_population_bits(sampler) == before
    @test CUDA.device() == caller_device
end

@testset "FirstOrderGRAMIS CUDA repulsion failures roll back" begin
    caller_device = CUDA.device()
    for (T, case) in ((Float32, :pooled_covariance), (Float64, :force))
        sampler = gram_is_cuda_sampler(
            T;
            repulsion_strength=T(0.1),
            fallback=case === :pooled_covariance,
        )
        if case === :pooled_covariance
            fill!(sampler.method_state.committed.factors, sqrt(floatmax(T)))
        else
            fill!(sampler.method_state.repulsion_strength, T(Inf))
        end
        before = gram_is_cuda_population_bits(sampler)
        failure = try
            importance_sample!(sampler)
            nothing
        catch error
            error
        end
        @test failure isa FirstOrderGRAMISRoundError
        @test failure.phase === :repulsion
        @test failure.cause.reason === (
            case === :pooled_covariance ?
            :pooled_covariance_nonfinite : :force_nonfinite
        )
        @test failure.diagnostics.transfers.count <= 192
        @test failure.diagnostics.transfers.bytes <= 4_096
        @test gram_is_cuda_population_bits(sampler) == before
    end
    @test CUDA.device() == caller_device
end
