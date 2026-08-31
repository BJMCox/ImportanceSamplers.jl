using CUDA
using ImportanceSamplers
using LinearAlgebra
using MLDataDevices
using Random
using Test

const IS = ImportanceSamplers

struct CUDAFirstOrderGRAMISTarget{T} end
struct CUDAFirstOrderGRAMISGradient end

function (::CUDAFirstOrderGRAMISTarget{T})(sample)::T where {T}
    return -T(0.5) * sum(abs2, sample)
end

function (::CUDAFirstOrderGRAMISGradient)(destination, sample)
    destination .= -sample
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
    wrapper_factors, wrapper_info = CUDA.cuSOLVER.potrfBatched!(
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
        wrapper_info=wrapper_info,
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

CUDA.allowscalar(false)

@testset "FirstOrderGRAMIS CUDA population Cholesky capabilities" begin
    @test CUDA.functional()
    caller_device = CUDA.device()
    for T in (Float32, Float64), dimension in (
        2,
        IS._GRAMIS_CHOLESKY_WORKGROUP_SIZE + 3,
    )
        result = gram_is_cuda_population_cholesky(T, dimension, 3)
        @test result.info == zeros(Int32, 3)
        @test result.wrapper_info == zeros(Int32, 3)
        @test result.factors ≈ result.wrapper_factors rtol =
            T === Float32 ? 3f-4 : 3e-12
        for proposal_slot in 1:3
            factor = result.factors[:, :, proposal_slot]
            @test factor * transpose(factor) ≈
                  result.source[:, :, proposal_slot] rtol =
                T === Float32 ? 5f-4 : 5e-12
        end
    end
    @test CUDA.device() == caller_device
end


@testset "FirstOrderGRAMIS CUDA preflight and pooled adapters" begin
    caller_device = CUDA.device()
    for T in (Float32, Float64)
        preflight = gram_is_cuda_preflight(T)
        @test preflight.resident
        @test preflight.prepared.device.device == caller_device

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
