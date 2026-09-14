module ImportanceSamplersMetalExt

import ImportanceSamplers as IS
import Metal
import MLDataDevices
import Random
import LinearAlgebra

# The Metal compiler miscompiles the combined unordered comparison emitted for
# NaN-or-Inf checks. Keep the target validity predicate in integer arithmetic.
Metal.@device_override function IS._native_target_reason(value::Float32)
    bits = reinterpret(UInt32, value)
    (bits & 0x7fffffff) > 0x7f800000 && return IS._NATIVE_TARGET_NAN
    bits == 0x7f800000 && return IS._NATIVE_TARGET_POSITIVE_INFINITY
    return UInt16(0)
end

function IS._owned_backend_rng(device::MLDataDevices.MetalDevice, seed::UInt64)
    return Random.seed!(MLDataDevices.default_device_rng(device), seed)
end

IS._allocate_failure_storage(::Metal.MetalBackend, prototype, capacity) =
    IS._allocate_portable_failure_storage(prototype, capacity)

function IS._gaussian_potrf!(::MLDataDevices.MetalDevice, factor::Metal.MtlMatrix{Float32})
    LinearAlgebra.cholesky!(LinearAlgebra.Symmetric(factor, :L))
    return factor
end

function IS._factor_pooled_covariance!(factor::Metal.MtlMatrix{Float32})
    LinearAlgebra.cholesky!(LinearAlgebra.Symmetric(factor, :L))
    return nothing
end

end
