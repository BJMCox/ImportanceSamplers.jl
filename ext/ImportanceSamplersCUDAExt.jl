module ImportanceSamplersCUDAExt

import CUDA
import ImportanceSamplers
import MLDataDevices

ImportanceSamplers._owned_backend_rng(
    ::MLDataDevices.CUDADevice,
    seed::UInt64,
) = CUDA.RNG(seed)

end
