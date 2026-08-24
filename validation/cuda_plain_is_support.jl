struct ScalarGaussianTarget end

@inline function gaussian_logdensity(coordinate, location, scale)
    standardized = (coordinate - location) / scale
    return -oftype(coordinate, 0.5) * abs2(standardized) - log(scale) -
           oftype(coordinate, 0.5) * log(oftype(coordinate, 2pi))
end

@inline (target::ScalarGaussianTarget)(sample, p) =
    gaussian_logdensity(sample, p.location[1], p.scale[1])

function scalar_gaussian_case(::Type{T}) where {T}
    return (
        label=:scalar,
        proposal=SphericalGaussian(T(0.25), T(1.25)),
        target=ScalarGaussianTarget(),
        context=(location=T[0.25], scale=T[1.25]),
    )
end

function cuda_device()
    CUDA.functional() || error("CUDA is not functional")
    device = MLDataDevices.gpu_device(nothing; force=true)
    MLDataDevices.functional(device) || error("MLDataDevices CUDADevice is not functional")
    eltype(device) === Nothing || error("CUDA device does not preserve scalar types")
    CUDA.allowscalar(false)
    return device
end

function cuda_package_versions(wanted)
    return sort!(
        [
            (dependency.name, something(dependency.version, "unversioned"))
            for dependency in values(Pkg.dependencies())
            if dependency.name in wanted
        ];
        by=first,
    )
end
