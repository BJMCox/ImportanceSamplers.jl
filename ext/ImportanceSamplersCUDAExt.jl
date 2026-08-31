module ImportanceSamplersCUDAExt

import CUDA
import ImportanceSamplers
import KernelAbstractions
import LinearAlgebra
import MLDataDevices

ImportanceSamplers._backend_functional(::MLDataDevices.CUDADevice) =
    CUDA.functional()

ImportanceSamplers._factor_batch_supported(::MLDataDevices.CUDADevice) = true
ImportanceSamplers._first_order_gramis_factorization_supported(
    ::MLDataDevices.CUDADevice,
) = true

function ImportanceSamplers._with_backend_device(
    f,
    device::MLDataDevices.CUDADevice,
)
    requested = device.device
    return requested === nothing ? f() : CUDA.device!(f, requested)
end

function ImportanceSamplers._backend_state_resident(
    device::MLDataDevices.CUDADevice,
    state,
)
    requested = something(device.device, CUDA.device())
    return _cuda_state_resident(requested, state)
end

_cuda_state_resident(
    device,
    state::ImportanceSamplers._PreparedFirstOrderGRAMIS,
) = _cuda_state_resident(
    device,
    ImportanceSamplers._first_order_gramis_resident_state(state),
)

_cuda_state_resident(device, array::CUDA.AnyCuArray) =
    CUDA.device(array) == device
_cuda_state_resident(device, range::AbstractRange) = true
_cuda_state_resident(device, array::AbstractArray) = false
_cuda_state_resident(
    device,
    state::Union{Number,AbstractString,Symbol,Type,Module,Nothing},
) = true
_cuda_state_resident(device, state::Tuple) =
    all(value -> _cuda_state_resident(device, value), state)
_cuda_state_resident(device, state::NamedTuple) =
    all(value -> _cuda_state_resident(device, value), values(state))

function _cuda_state_resident(device, state)
    type = typeof(state)
    for field in 1:fieldcount(type)
        isdefined(state, field) || return false
        _cuda_state_resident(device, getfield(state, field)) || return false
    end
    return true
end

ImportanceSamplers._owned_backend_rng(
    ::MLDataDevices.CUDADevice,
    seed::UInt64,
) = CUDA.RNG(seed)

function ImportanceSamplers._amis_potrf!(
    ::MLDataDevices.CUDADevice,
    factor::CUDA.StridedCuMatrix{T},
) where {T<:Union{Float32,Float64}}
    factor, info = CUDA.cuSOLVER.potrf!('L', factor)
    info == 0 || throw(LinearAlgebra.PosDefException(info))
    return factor
end

function ImportanceSamplers._factor_population!(
    ::MLDataDevices.CUDADevice,
    factors::CUDA.StridedCuArray{T,3},
    covariances::CUDA.StridedCuArray{T,3},
    info::CUDA.StridedCuVector{Int32},
) where {T<:Union{Float32,Float64}}
    proposal_count = size(factors, 3)
    workgroupsize = ImportanceSamplers._GRAMIS_CHOLESKY_WORKGROUP_SIZE
    backend = KernelAbstractions.get_backend(factors)
    kernel = ImportanceSamplers._factor_population_kernel!(
        backend,
        workgroupsize,
    )
    kernel(
        factors,
        covariances,
        info;
        ndrange=workgroupsize * proposal_count,
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

function ImportanceSamplers._factor_pooled_covariance!(
    pooled_covariance::CUDA.StridedCuMatrix{T},
) where {T<:Union{Float32,Float64}}
    _, info = CUDA.cuSOLVER.potrf!('L', pooled_covariance)
    iszero(info) || throw(
        ImportanceSamplers._FirstOrderGRAMISRepulsionError(
            0,
            :pooled_factorization_failed,
            info,
        ),
    )
    return nothing
end

function ImportanceSamplers._preflight_first_order_gramis_factorization!(
    device::MLDataDevices.CUDADevice,
    method_state::ImportanceSamplers._PreparedFirstOrderGRAMIS,
)
    T = eltype(method_state.committed.locations)
    dimension = 3
    proposal_count = 2
    prototype = method_state.workspace.covariances
    covariances = similar(
        prototype,
        T,
        dimension,
        dimension,
        proposal_count,
    )
    factors = similar(covariances)
    info = similar(method_state.workspace.factor_info, Int32, proposal_count)
    backend = KernelAbstractions.get_backend(covariances)
    covariance_kernel =
        ImportanceSamplers._factorization_preflight_covariances!(backend)
    covariance_kernel(
        covariances;
        ndrange=length(covariances),
    )
    ImportanceSamplers._factor_population!(
        device,
        factors,
        covariances,
        info,
    )
    any(!iszero, info) && throw(
        ImportanceSamplers.SamplerDeviceError(
            device,
            :kernel_argument_unsupported,
        ),
    )

    pooled = similar(method_state.workspace.pooled_covariance, T, dimension, dimension)
    execution = ImportanceSamplers._KernelExecution(
        ImportanceSamplers._ThreadedCPUExecution(),
    )
    ImportanceSamplers._pooled_covariance!(pooled, factors, execution)
    ImportanceSamplers._factor_pooled_covariance!(pooled)
    means = similar(method_state.workspace.whitened_means, T, dimension, proposal_count)
    fill!(means, one(T))
    whitened = similar(means)
    ImportanceSamplers._whiten_means!(whitened, pooled, means)
    all(isfinite, whitened) || throw(
        ImportanceSamplers.SamplerDeviceError(
            device,
            :kernel_argument_unsupported,
        ),
    )
    return nothing
end

end
