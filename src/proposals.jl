"""
    AbstractProposalFamily

Abstract supertype for native proposal-family markers.
"""
abstract type AbstractProposalFamily end

"""
    AbstractRadialProposalFamily <: AbstractProposalFamily

Abstract supertype for radially symmetric native proposal families.
"""
abstract type AbstractRadialProposalFamily <: AbstractProposalFamily end

struct GaussianFamily <: AbstractRadialProposalFamily end

struct _SphericalGaussianScale{T}
    scale::T
end

struct _DiagonalGaussianScale{V}
    scales::V
end

struct _FactorGaussianScale{M}
    factor::M
end

struct _GaussianProposal{F,L,S,T}
    family::F
    location::L
    scale::S
    lognormalizer::T
end

const _NativeGaussianFloat = Union{Float32,Float64}

function _validated_gaussian_location(location::T) where {T<:_NativeGaussianFloat}
    isfinite(location) || throw(ArgumentError("location must be finite"))
    return location
end

function _validated_gaussian_location(
    location::AbstractVector{T},
) where {T<:_NativeGaussianFloat}
    isempty(location) && throw(ArgumentError("location must be nonempty"))
    all(isfinite, location) || throw(ArgumentError("location must contain only finite values"))
    return collect(location)
end

function _validated_gaussian_location(location)
    throw(ArgumentError("location must be a Float32 or Float64 scalar or vector"))
end

_gaussian_float_type(::T) where {T<:_NativeGaussianFloat} = T
_gaussian_float_type(::AbstractVector{T}) where {T<:_NativeGaussianFloat} = T

_gaussian_dimension(::_NativeGaussianFloat) = 1
_gaussian_dimension(location::AbstractVector) = length(location)

function _validated_gaussian_scale(scale::T, ::Type{T}) where {T<:_NativeGaussianFloat}
    isfinite(scale) && scale > zero(T) || throw(
        ArgumentError("scale must be finite and positive"),
    )
    return scale
end

function _validated_gaussian_scale(scale, ::Type{T}) where {T<:_NativeGaussianFloat}
    throw(ArgumentError("scale must have the same Float32 or Float64 type as location"))
end

function _gaussian_lognormalizer(::Type{T}, dimension, logabsdet::T) where {T}
    logtwopi = log(T(2) * T(pi))
    return -T(0.5) * T(dimension) * logtwopi - logabsdet
end

"""
    SphericalGaussian(location, scale)

Construct a normalized scalar or vector Gaussian proposal with finite
`Float32` or `Float64` location and a finite positive scalar standard
deviation of the same type.
"""
function SphericalGaussian(location, scale)
    stored_location = _validated_gaussian_location(location)
    T = _gaussian_float_type(stored_location)
    stored_scale = _validated_gaussian_scale(scale, T)
    dimension = _gaussian_dimension(stored_location)
    scale_storage = _SphericalGaussianScale(stored_scale)
    lognormalizer = _gaussian_lognormalizer(
        T,
        dimension,
        T(dimension) * log(stored_scale),
    )
    return _GaussianProposal(
        GaussianFamily(),
        stored_location,
        scale_storage,
        lognormalizer,
    )
end

function _validated_diagonal_scales(
    scales::AbstractVector{T},
    ::Type{T},
    dimension,
) where {T<:_NativeGaussianFloat}
    length(scales) == dimension || throw(
        ArgumentError("diagonal scales must match the location length"),
    )
    all(scale -> isfinite(scale) && scale > zero(T), scales) || throw(
        ArgumentError("diagonal scales must be finite and positive"),
    )
    return collect(scales)
end

function _validated_diagonal_scales(scales, ::Type{T}, dimension) where {T}
    throw(ArgumentError("diagonal scales must have the same floating type as location"))
end

"""
    DiagonalGaussian(location, scales)

Construct a normalized vector Gaussian proposal with independent finite
positive coordinate scales. Location and scales must use the same `Float32`
or `Float64` element type.
"""
function DiagonalGaussian(location, scales)
    stored_location = _validated_gaussian_location(location)
    stored_location isa AbstractVector || throw(
        ArgumentError("DiagonalGaussian requires a vector location"),
    )
    T = _gaussian_float_type(stored_location)
    stored_scales = _validated_diagonal_scales(
        scales,
        T,
        length(stored_location),
    )
    scale_storage = _DiagonalGaussianScale(stored_scales)
    logabsdet = sum(log, stored_scales)
    lognormalizer = _gaussian_lognormalizer(T, length(stored_location), logabsdet)
    return _GaussianProposal(
        GaussianFamily(),
        stored_location,
        scale_storage,
        lognormalizer,
    )
end

function _copied_gaussian_factor(factor::LinearAlgebra.Cholesky{T}) where {T}
    T <: _NativeGaussianFloat || throw(
        ArgumentError("factor must use Float32 or Float64 elements"),
    )
    return Matrix(factor.L)
end

function _copied_gaussian_factor(factor::AbstractMatrix{T}) where {T}
    T <: _NativeGaussianFloat || throw(
        ArgumentError("factor must use Float32 or Float64 elements"),
    )
    return Matrix(factor)
end

function _copied_gaussian_factor(factor)
    throw(ArgumentError("factor must be a Cholesky factorization or lower-triangular matrix"))
end

function _validate_gaussian_factor(factor::Matrix{T}, location, ::Type{T}) where {T}
    dimension = length(location)
    size(factor) == (dimension, dimension) || throw(
        ArgumentError("factor must be square and match the location length"),
    )
    all(isfinite, factor) || throw(ArgumentError("factor must contain only finite values"))
    for column in axes(factor, 2), row in first(axes(factor, 1)):(column - 1)
        iszero(factor[row, column]) || throw(
            ArgumentError("factor must be lower triangular"),
        )
    end
    all(index -> factor[index, index] > zero(T), axes(factor, 1)) || throw(
        ArgumentError("factor diagonal must be positive"),
    )
    return factor
end

function _validate_gaussian_factor(factor, location, ::Type{T}) where {T}
    throw(ArgumentError("factor must have the same floating type as location"))
end

"""
    FactorGaussian(location, factor)

Construct a normalized vector Gaussian proposal with affine map
`x = location + L * z`, where `z` is standard normal and `L` is a finite
lower-triangular factor with positive diagonal. A `Cholesky` factorization is
accepted through its lower factor.
"""
function FactorGaussian(location, factor)
    stored_location = _validated_gaussian_location(location)
    stored_location isa AbstractVector || throw(
        ArgumentError("FactorGaussian requires a vector location"),
    )
    T = _gaussian_float_type(stored_location)
    stored_factor = _validate_gaussian_factor(
        _copied_gaussian_factor(factor),
        stored_location,
        T,
    )
    logabsdet = sum(index -> log(stored_factor[index, index]), axes(stored_factor, 1))
    scale_storage = _FactorGaussianScale(stored_factor)
    lognormalizer = _gaussian_lognormalizer(T, length(stored_location), logabsdet)
    return _GaussianProposal(
        GaussianFamily(),
        stored_location,
        scale_storage,
        lognormalizer,
    )
end

function _draw_gaussian(
    rng::Random.AbstractRNG,
    location::T,
    scale::_SphericalGaussianScale{T},
) where {T<:_NativeGaussianFloat}
    return location + scale.scale * Random.randn(rng, T)
end

function _draw_gaussian(
    rng::Random.AbstractRNG,
    location::Vector{T},
    scale::_SphericalGaussianScale{T},
) where {T<:_NativeGaussianFloat}
    sample = Random.randn(rng, T, length(location))
    for index in eachindex(sample, location)
        sample[index] = location[index] + scale.scale * sample[index]
    end
    return sample
end

function _draw_gaussian(
    rng::Random.AbstractRNG,
    location::Vector{T},
    scale::_DiagonalGaussianScale{Vector{T}},
) where {T<:_NativeGaussianFloat}
    sample = Random.randn(rng, T, length(location))
    for index in eachindex(sample, location, scale.scales)
        sample[index] = location[index] + scale.scales[index] * sample[index]
    end
    return sample
end

function _draw_gaussian(
    rng::Random.AbstractRNG,
    location::Vector{T},
    scale::_FactorGaussianScale{Matrix{T}},
) where {T<:_NativeGaussianFloat}
    sample = Random.randn(rng, T, length(location))
    LinearAlgebra.lmul!(LinearAlgebra.LowerTriangular(scale.factor), sample)
    for index in eachindex(sample, location)
        sample[index] += location[index]
    end
    return sample
end

function Random.rand(rng::Random.AbstractRNG, proposal::_GaussianProposal)
    return _draw_gaussian(rng, proposal.location, proposal.scale)
end

function _check_gaussian_sample_length(location, sample)
    length(sample) == length(location) || throw(
        DimensionMismatch(
            "sample length $(length(sample)) does not match proposal dimension " *
            "$(length(location))",
        ),
    )
    return nothing
end

function _gaussian_squared_radius(
    location::T,
    scale::_SphericalGaussianScale{T},
    sample::T,
) where {T<:_NativeGaussianFloat}
    standardized = (sample - location) / scale.scale
    return abs2(standardized)
end

function _gaussian_squared_radius(
    location::Vector{T},
    scale::_SphericalGaussianScale{T},
    sample::AbstractVector{T},
) where {T<:_NativeGaussianFloat}
    _check_gaussian_sample_length(location, sample)
    squared_radius = zero(T)
    for index in eachindex(location, sample)
        standardized = (sample[index] - location[index]) / scale.scale
        squared_radius += abs2(standardized)
    end
    return squared_radius
end

function _gaussian_squared_radius(
    location::Vector{T},
    scale::_DiagonalGaussianScale{Vector{T}},
    sample::AbstractVector{T},
) where {T<:_NativeGaussianFloat}
    _check_gaussian_sample_length(location, sample)
    squared_radius = zero(T)
    for index in eachindex(location, scale.scales, sample)
        standardized = (sample[index] - location[index]) / scale.scales[index]
        squared_radius += abs2(standardized)
    end
    return squared_radius
end

function _gaussian_squared_radius(
    location::Vector{T},
    scale::_FactorGaussianScale{Matrix{T}},
    sample::AbstractVector{T},
) where {T<:_NativeGaussianFloat}
    _check_gaussian_sample_length(location, sample)
    standardized = copy(sample)
    for index in eachindex(standardized, location)
        standardized[index] -= location[index]
    end
    LinearAlgebra.ldiv!(LinearAlgebra.LowerTriangular(scale.factor), standardized)
    return sum(abs2, standardized)
end

function DensityInterface.logdensityof(
    proposal::_GaussianProposal,
    sample,
)
    squared_radius = _gaussian_squared_radius(
        proposal.location,
        proposal.scale,
        sample,
    )
    return proposal.lognormalizer - oftype(proposal.lognormalizer, 0.5) * squared_radius
end

_proposal_dimension(proposal::_GaussianProposal) = _gaussian_dimension(proposal.location)
