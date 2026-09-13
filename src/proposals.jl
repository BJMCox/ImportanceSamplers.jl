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

struct StudentTFamily{T} <: AbstractRadialProposalFamily
    dof::T
end

@inline _radial_logdensity(::GaussianFamily, lognormalizer, squared_radius, dimension) =
    lognormalizer - oftype(squared_radius, 0.5) * squared_radius

@inline function _radial_logdensity(family::StudentTFamily, lognormalizer, squared_radius, dimension)
    dof = family.dof
    return lognormalizer - (dof + oftype(dof, dimension)) / oftype(dof, 2) *
           log1p(squared_radius / dof)
end

# Moment updates use covariance. Student-t proposal storage uses elliptical scale.
_covariance_multiplier(::GaussianFamily, ::Type{T}) where {T} = one(T)
_covariance_multiplier(family::StudentTFamily, ::Type{T}) where {T} =
    T(family.dof / (family.dof - 2))
_scale_covariance_factor!(factor, ::GaussianFamily) = factor
function _scale_covariance_factor!(factor, family::StudentTFamily)
    factor .*= sqrt(inv(_covariance_multiplier(family, eltype(factor))))
    return factor
end
_validate_moment_family(::GaussianFamily) = nothing
function _validate_moment_family(family::StudentTFamily)
    family.dof > 2 || throw(ArgumentError("covariance adaptation requires Student-t degrees of freedom > 2"))
    return nothing
end
_radial_lognormalizer(::GaussianFamily, ::Type{T}, dimension, logabsdet) where {T} =
    _gaussian_lognormalizer(T, dimension, logabsdet)
_radial_lognormalizer(family::StudentTFamily, ::Type{T}, dimension, logabsdet) where {T} =
    _student_t_lognormalizer(T, family.dof, dimension, logabsdet)
_radial_proposal(::GaussianFamily, location, scale, lognormalizer) =
    _GaussianProposal(GaussianFamily(), location, scale, lognormalizer)
_radial_proposal(family::StudentTFamily, location, scale, lognormalizer) =
    _StudentTProposal(family, location, scale, lognormalizer)

_prepare_proposal_input(proposal) = proposal
_prepare_proposal_inputs(proposals) = copy(proposals)

"""
    ProductProposal(blocks::NamedTuple)

Construct a CPU proposal for the product of independent named proposal blocks.
Blocks are drawn in field order with the supplied RNG, and their normalized log
densities are summed. `ProductProposal` and transformed named-product layouts
are not supported on CUDA.
"""
struct ProductProposal{B<:NamedTuple}
    blocks::B

    function ProductProposal(blocks::B) where {B<:NamedTuple}
        isempty(blocks) && throw(ArgumentError("product proposal must contain at least one block"))
        all(isconcretetype, fieldtypes(B)) || throw(
            ArgumentError("product proposal blocks must have concrete field types"),
        )
        return new{B}(blocks)
    end
end

_accelerator_proposal_limit(proposal) = :generic_proposal_cpu_only
_accelerator_proposal_limit(::ProductProposal) = :product_proposal_cpu_only

struct _PreparedProposalToken end

"""
    TransformedProposal(base, transform)

Construct a proposal that maps draws from `base` into logical values through
`transform`. Its density uses the exact normalized change of variables
`log q_x(x) = log q_z(z) - log|J(z)|`. The proposal owns this Jacobian;
target densities must use the resulting logical reference measure without
applying it again.
"""
struct TransformedProposal{B,T}
    base::B
    transform::T

    function TransformedProposal(base::B, transform::T, ::_PreparedProposalToken) where {B,T}
        return new{B,T}(base, transform)
    end
end

TransformedProposal(base, transform) =
    _prepare_transformed_proposal(_prepare_proposal_input(base), transform)

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

struct _StudentTProposal{F,L,S,T}
    family::F
    location::L
    scale::S
    lognormalizer::T
end

const _NativeRadialProposal = Union{_GaussianProposal,_StudentTProposal}

_accelerator_proposal_limit(::_GaussianProposal) = nothing
_accelerator_proposal_limit(::_StudentTProposal) = nothing

Adapt.@adapt_structure _SphericalGaussianScale
Adapt.@adapt_structure _DiagonalGaussianScale
Adapt.@adapt_structure _FactorGaussianScale
Adapt.@adapt_structure _GaussianProposal
Adapt.@adapt_structure StudentTFamily
Adapt.@adapt_structure _StudentTProposal

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
deviation of the same type. Its covariance is `scale^2` for a scalar and
`scale^2 I` for a vector.
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
or `Float64` element type. The scales are standard deviations, so the
covariance diagonal is `scales .^ 2`.
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
accepted through its lower factor. The covariance is `L * L'`; pass a factor,
not an inverse covariance. Density evaluation uses triangular solves and never
forms a covariance inverse.
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

function _validated_student_t_dof(dof::T, ::Type{T}) where {T<:_NativeGaussianFloat}
    isfinite(dof) && dof > zero(T) || throw(
        ArgumentError("degrees of freedom must be finite and positive"),
    )
    return dof
end

function _validated_student_t_dof(dof, ::Type{T}) where {T}
    throw(
        ArgumentError(
            "degrees of freedom must have the same Float32 or Float64 type as location",
        ),
    )
end

@inline function _positive_loggamma(value::Float64)
    shifted = value < 1.0 ? value + 1.0 : value
    z = shifted - 1.0
    series = 0.99999999999980993
    coefficients = (
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    )
    for index in eachindex(coefficients)
        series += coefficients[index] / (z + Float64(index))
    end
    shifted_sum = z + 7.5
    result = 0.9189385332046727 +
             (z + 0.5) * log(shifted_sum) - shifted_sum + log(series)
    return value < 1.0 ? result - log(value) : result
end

function _student_t_lognormalizer(
    ::Type{T},
    dof::T,
    dimension,
    logabsdet::T,
) where {T<:_NativeGaussianFloat}
    work_dof = Float64(dof)
    half_dimension = 0.5 * Float64(dimension)
    value = _positive_loggamma(0.5 * (work_dof + Float64(dimension))) -
            _positive_loggamma(0.5 * work_dof) -
            half_dimension * log(work_dof * pi) - Float64(logabsdet)
    return T(value)
end

function _student_t_proposal(dof, location, scale, logabsdet)
    T = _gaussian_float_type(location)
    stored_dof = _validated_student_t_dof(dof, T)
    lognormalizer = _student_t_lognormalizer(
        T,
        stored_dof,
        _gaussian_dimension(location),
        logabsdet,
    )
    return _StudentTProposal(
        StudentTFamily(stored_dof),
        location,
        scale,
        lognormalizer,
    )
end

"""
    SphericalStudentT(dof, location, scale)

Construct a normalized scalar or spherical multivariate Student-t proposal.
`dof`, `location`, and `scale` must share a `Float32` or `Float64` type.
The scale is the standard elliptical scale, not the standard deviation.
"""
function SphericalStudentT(dof, location, scale)
    stored_location = _validated_gaussian_location(location)
    T = _gaussian_float_type(stored_location)
    stored_scale = _validated_gaussian_scale(scale, T)
    dimension = _gaussian_dimension(stored_location)
    return _student_t_proposal(
        dof,
        stored_location,
        _SphericalGaussianScale(stored_scale),
        T(dimension) * log(stored_scale),
    )
end

"""
    DiagonalStudentT(dof, location, scales)

Construct a normalized multivariate Student-t proposal with diagonal elliptical
scales. The scales are not marginal standard deviations.
"""
function DiagonalStudentT(dof, location, scales)
    stored_location = _validated_gaussian_location(location)
    stored_location isa AbstractVector || throw(
        ArgumentError("DiagonalStudentT requires a vector location"),
    )
    T = _gaussian_float_type(stored_location)
    stored_scales = _validated_diagonal_scales(
        scales,
        T,
        length(stored_location),
    )
    return _student_t_proposal(
        dof,
        stored_location,
        _DiagonalGaussianScale(stored_scales),
        sum(log, stored_scales),
    )
end

"""
    FactorStudentT(dof, location, factor)

Construct a normalized multivariate Student-t proposal with elliptical scale
matrix `factor * factor'`. Density evaluation uses triangular solves.
"""
function FactorStudentT(dof, location, factor)
    stored_location = _validated_gaussian_location(location)
    stored_location isa AbstractVector || throw(
        ArgumentError("FactorStudentT requires a vector location"),
    )
    T = _gaussian_float_type(stored_location)
    stored_factor = _validate_gaussian_factor(
        _copied_gaussian_factor(factor),
        stored_location,
        T,
    )
    logabsdet = sum(index -> log(stored_factor[index, index]), axes(stored_factor, 1))
    return _student_t_proposal(
        dof,
        stored_location,
        _FactorGaussianScale(stored_factor),
        logabsdet,
    )
end

function _draw_gaussian(
    rng::Random.AbstractRNG,
    location::T,
    scale::_SphericalGaussianScale{T},
) where {T<:_NativeGaussianFloat}
    return _gaussian_affine_coordinate(
        location,
        scale.scale,
        Random.randn(rng, T),
    )
end

@inline _gaussian_affine_coordinate(location, scale, normal) =
    location + scale * normal

@inline function _gaussian_coordinate(
    location::T,
    scale::_SphericalGaussianScale{T},
    normals,
    offset,
    coordinate,
) where {T<:_NativeGaussianFloat}
    return _gaussian_affine_coordinate(
        location,
        scale.scale,
        @inbounds(normals[offset]),
    )
end

@inline function _gaussian_coordinate(
    location::AbstractVector{T},
    scale::_SphericalGaussianScale{T},
    normals,
    offset,
    coordinate,
) where {T<:_NativeGaussianFloat}
    return _gaussian_affine_coordinate(
        @inbounds(location[coordinate]),
        scale.scale,
        @inbounds(normals[offset + coordinate - 1]),
    )
end

@inline function _gaussian_coordinate(
    location::AbstractVector{T},
    scale::_DiagonalGaussianScale,
    normals,
    offset,
    coordinate,
) where {T<:_NativeGaussianFloat}
    return _gaussian_affine_coordinate(
        @inbounds(location[coordinate]),
        @inbounds(scale.scales[coordinate]),
        @inbounds(normals[offset + coordinate - 1]),
    )
end

@inline function _gaussian_coordinate(
    location::AbstractVector{T},
    scale::_FactorGaussianScale,
    normals,
    offset,
    coordinate,
) where {T<:_NativeGaussianFloat}
    value = zero(T)
    for source_coordinate in 1:coordinate
        value += @inbounds(scale.factor[coordinate, source_coordinate]) *
                 @inbounds(normals[offset + source_coordinate - 1])
    end
    return @inbounds(location[coordinate]) + value
end

@inline function _native_gaussian_squared_radius!(
    location::T,
    scale::_SphericalGaussianScale{T},
    coordinates,
    offset,
) where {T<:_NativeGaussianFloat}
    standardized = (@inbounds coordinates[offset] - location) / scale.scale
    @inbounds coordinates[offset] = standardized
    return abs2(standardized)
end

@inline function _native_standardized_coordinate(
    location::AbstractVector{T},
    scale::_SphericalGaussianScale{T},
    coordinates,
    offset,
    index,
) where {T<:_NativeGaussianFloat}
    return (@inbounds coordinates[offset + index - 1] - location[index]) / scale.scale
end

@inline function _native_standardized_coordinate(
    location::AbstractVector{T},
    scale::_DiagonalGaussianScale,
    coordinates,
    offset,
    index,
) where {T<:_NativeGaussianFloat}
    return (@inbounds coordinates[offset + index - 1] - location[index]) /
           @inbounds(scale.scales[index])
end

@inline function _native_gaussian_squared_radius!(
    location::AbstractVector{T},
    scale::Union{_SphericalGaussianScale,_DiagonalGaussianScale},
    coordinates,
    offset,
) where {T<:_NativeGaussianFloat}
    squared_radius = zero(T)
    for index in eachindex(location)
        standardized = _native_standardized_coordinate(
            location, scale, coordinates, offset, index
        )
        @inbounds coordinates[offset + index - 1] = standardized
        squared_radius += abs2(standardized)
    end
    return squared_radius
end

@inline function _native_gaussian_squared_radius!(
    location::AbstractVector{T},
    scale::_FactorGaussianScale,
    coordinates,
    offset,
) where {T<:_NativeGaussianFloat}
    squared_radius = zero(T)
    for row in eachindex(location)
        standardized = @inbounds coordinates[offset + row - 1] - location[row]
        for column in 1:(row - 1)
            standardized -= @inbounds(scale.factor[row, column]) *
                            @inbounds(coordinates[offset + column - 1])
        end
        standardized /= @inbounds scale.factor[row, row]
        @inbounds coordinates[offset + row - 1] = standardized
        squared_radius += abs2(standardized)
    end
    return squared_radius
end

@inline function _native_gaussian_logdensity!(proposal::_GaussianProposal, coordinates, offset)
    squared_radius = _native_gaussian_squared_radius!(
        proposal.location,
        proposal.scale,
        coordinates,
        offset,
    )
    return proposal.lognormalizer -
           oftype(proposal.lognormalizer, 0.5) * squared_radius
end

@inline _native_proposal_logdensity!(proposal::_GaussianProposal, coordinates, offset) =
    _native_gaussian_logdensity!(proposal, coordinates, offset)

@inline function _native_proposal_logdensity!(proposal::_StudentTProposal, coordinates, offset)
    squared_radius = _native_gaussian_squared_radius!(
        proposal.location,
        proposal.scale,
        coordinates,
        offset,
    )
    return _radial_logdensity(proposal.family, proposal.lognormalizer,
        squared_radius, _gaussian_dimension(proposal.location))
end

function _gaussian_from_normal(
    location::AbstractVector{T},
    scale,
    normals::AbstractVector{T},
) where {T<:_NativeGaussianFloat}
    dimension = _gaussian_dimension(location)
    length(normals) == dimension || throw(
        DimensionMismatch("normal coordinates must match the Gaussian dimension"),
    )
    sample = similar(normals, dimension)
    for coordinate in 1:dimension
        sample[coordinate] = _gaussian_coordinate(
            location,
            scale,
            normals,
            firstindex(normals),
            coordinate,
        )
    end
    return sample
end

function _draw_gaussian(
    rng::Random.AbstractRNG,
    location::Vector{T},
    scale::Union{
        _SphericalGaussianScale{T},
        _DiagonalGaussianScale{Vector{T}},
        _FactorGaussianScale{Matrix{T}},
    },
) where {T<:_NativeGaussianFloat}
    normals = Random.randn(rng, T, length(location))
    return _gaussian_from_normal(location, scale, normals)
end

function Random.rand(rng::Random.AbstractRNG, proposal::_GaussianProposal)
    return _draw_gaussian(rng, proposal.location, proposal.scale)
end

@inline function _gamma_parameters(shape::T) where {T}
    offset = shape - T(1 / 3)
    return offset, inv(sqrt(T(9) * offset))
end

@inline function _gamma_candidate(offset::T, coefficient::T, normal::T, uniform::T) where {T}
    root = one(T) + coefficient * normal
    root > zero(T) || return zero(T), false
    cube = root * root * root
    accepted = uniform < one(T) - T(0.0331) * abs2(abs2(normal)) ||
               log(uniform) < T(0.5) * abs2(normal) +
                              offset * (one(T) - cube + log(cube))
    return offset * cube, accepted
end

function _rand_unit_gamma(rng::Random.AbstractRNG, shape::T) where {T<:_NativeGaussianFloat}
    adjusted_shape = shape < one(T) ? shape + one(T) : shape
    offset, coefficient = _gamma_parameters(adjusted_shape)
    while true
        gamma, accepted = _gamma_candidate(
            offset,
            coefficient,
            Random.randn(rng, T),
            Random.rand(rng, T),
        )
        accepted || continue
        shape >= one(T) && return gamma
        correction = Random.rand(rng, T)
        correction > zero(T) && return gamma * correction^inv(shape)
    end
end

function _student_t_multiplier(
    rng::Random.AbstractRNG,
    dof::T,
) where {T<:_NativeGaussianFloat}
    dof == one(T) && return inv(abs(Random.randn(rng, T)))
    chi_square = T(2) * _rand_unit_gamma(rng, dof / T(2))
    return sqrt(dof / chi_square)
end

function _draw_student_t(
    rng::Random.AbstractRNG,
    proposal::_StudentTProposal{F,T},
) where {F,T<:_NativeGaussianFloat}
    multiplier = _student_t_multiplier(rng, proposal.family.dof)
    normal = Random.randn(rng, T) * multiplier
    return _gaussian_affine_coordinate(
        proposal.location,
        proposal.scale.scale,
        normal,
    )
end

function _draw_student_t(
    rng::Random.AbstractRNG,
    proposal::_StudentTProposal{F,Vector{T}},
) where {F,T<:_NativeGaussianFloat}
    normals = Random.randn(rng, T, length(proposal.location))
    normals .*= _student_t_multiplier(rng, proposal.family.dof)
    return _gaussian_from_normal(proposal.location, proposal.scale, normals)
end

function Random.rand(rng::Random.AbstractRNG, proposal::_StudentTProposal)
    return _draw_student_t(rng, proposal)
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

function DensityInterface.logdensityof(
    proposal::_StudentTProposal,
    sample,
)
    squared_radius = _gaussian_squared_radius(
        proposal.location,
        proposal.scale,
        sample,
    )
    dof = proposal.family.dof
    dimension = _gaussian_dimension(proposal.location)
    return proposal.lognormalizer -
           (dof + typeof(dof)(dimension)) / typeof(dof)(2) *
           log1p(squared_radius / dof)
end

@inline DensityInterface.DensityKind(::_GaussianProposal) = DensityInterface.HasDensity()
@inline DensityInterface.DensityKind(::_StudentTProposal) = DensityInterface.HasDensity()

_proposal_dimension(proposal::_GaussianProposal) = _gaussian_dimension(proposal.location)
_proposal_dimension(proposal::_StudentTProposal) = _gaussian_dimension(proposal.location)

_draw_product_values(rng::Random.AbstractRNG, ::Tuple{}) = ()

function _draw_product_values(rng::Random.AbstractRNG, blocks::Tuple)
    return (Random.rand(rng, first(blocks)), _draw_product_values(rng, Base.tail(blocks))...)
end

function Random.rand(rng::Random.AbstractRNG, proposal::ProductProposal)
    block_values = _draw_product_values(rng, values(proposal.blocks))
    return NamedTuple{keys(proposal.blocks)}(block_values)
end

function DensityInterface.logdensityof(
    proposal::ProductProposal{B},
    sample::NamedTuple{Names},
) where {Names,B<:NamedTuple{Names}}
    block_logs = map(DensityInterface.logdensityof, proposal.blocks, sample)
    return sum(values(block_logs))
end

function DensityInterface.logdensityof(proposal::ProductProposal, sample::NamedTuple)
    throw(ArgumentError("product proposal sample fields must match proposal blocks"))
end

@inline DensityInterface.DensityKind(::ProductProposal) = DensityInterface.HasDensity()
