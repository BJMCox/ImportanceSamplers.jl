module ImportanceSamplersCUDAExt

import CUDA
import ImportanceSamplers
import LinearAlgebra
import MLDataDevices

ImportanceSamplers._backend_functional(::MLDataDevices.CUDADevice) =
    CUDA.functional()

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

end
