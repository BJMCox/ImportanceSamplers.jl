module ImportanceSamplers

import DensityInterface
import KernelAbstractions
import KernelAbstractions: @index, @kernel
import LinearAlgebra
import LogDensityProblems
import LogExpFunctions
import MLDataDevices
import Random

export AbstractImportanceSampler,
    AbstractProposalFamily,
    AbstractRadialProposalFamily,
    AbstractSampleTransform,
    AllZeroWeightsError,
    DiagonalGaussian,
    FactorGaussian,
    ImportanceSampling,
    IdentityTransform,
    IntervalTransform,
    InvalidTransformError,
    LogTarget,
    SamplerAlreadyExecutedError,
    SamplerBusyError,
    SamplerDeviceError,
    SamplerExecutionError,
    SimplexTransform,
    SphericalGaussian,
    PositiveTransform,
    ProductProposal,
    SoftplusTransform,
    TransformedProposal,
    WeightedSamples,
    WeightedSampleView,
    importance_sample,
    importance_sample!,
    lognormalizer,
    normalized_weights,
    prepare_sampler

include("proposals.jl")
include("transforms.jl")
include("proposal_composition.jl")
include("methods/importance_sampling.jl")
include("targets.jl")
include("storage.jl")
include("results.jl")

@kernel function _kernel_smoke_kernel!(output)
    index = @index(Global, Linear)
    output[index] = index
end

function _kernel_smoke!(output)
    backend = KernelAbstractions.get_backend(output)
    return _kernel_smoke_kernel!(backend)(output; ndrange=length(output))
end

end # module ImportanceSamplers
