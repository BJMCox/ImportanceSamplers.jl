"""
    AbstractProposalPopulation

Abstract supertype for explicit populations of normalized proposals.

Population values are sampler configuration, not probability distributions.
Concrete subtypes describe proposals that assignment and denominator schemes can
use separately during preparation.
"""
abstract type AbstractProposalPopulation end

"""
    ProposalBank(proposals)
    ProposalBank(proposals, masses)

Construct an explicit proposal population with copied proposals and normalized
nonnegative masses. Omitted masses are equal. The readable `proposals` and
`masses` fields preserve input order; masses sum to one. A configured positive
mass must remain positive through floating conversion and normalization;
otherwise construction rejects the bank instead of making that proposal inert.

Zero-mass proposals remain in the configuration and retain their stable
one-based IDs, but are neither assigned nor evaluated. Positive-mass proposals
must share a sample dimension. Generic CPU execution additionally requires one
concrete proposal element type. Native Gaussian banks with one scalar/vector
layout and floating type may be packed during preparation.

`ProposalBank` deliberately implements neither `rand` nor
`DensityInterface.logdensityof`: a bank is a proposal population, not a mixture
distribution. [`AbstractMISScheme`](@ref) independently selects both the
assignment law and the density used in each MIS denominator. A pre-existing
mixture object remains one atomic proposal unless its components are explicitly
placed in a bank.
"""
struct ProposalBank{P<:AbstractVector,M<:AbstractVector} <: AbstractProposalPopulation
    proposals::P
    masses::M

    function ProposalBank(
        proposals::P,
        masses::M,
    ) where {P<:AbstractVector,M<:AbstractVector}
        isempty(proposals) && throw(ArgumentError("a proposal bank cannot be empty"))
        length(masses) == length(proposals) || throw(
            ArgumentError("proposal and mass counts must match"),
        )

        mass_type = eltype(M)
        isconcretetype(mass_type) && mass_type <: Real && mass_type !== Bool || throw(
            ArgumentError("mass element type must be a concrete non-Bool real type"),
        )
        floating_type = promote_type(float(mass_type), Float32)
        isconcretetype(floating_type) && floating_type <: AbstractFloat || throw(
            ArgumentError("masses must promote to one concrete floating type"),
        )

        floating_masses = floating_type.(masses)
        _require_preserved_positive_masses(masses, floating_masses, "floating conversion")
        all(isfinite, floating_masses) || throw(ArgumentError("masses must be finite"))
        all(>=(zero(floating_type)), floating_masses) || throw(
            ArgumentError("masses must be nonnegative"),
        )
        maximum_mass = maximum(floating_masses)
        maximum_mass > zero(floating_type) || throw(
            ArgumentError("at least one mass must be positive"),
        )

        scaled_masses = floating_masses ./ maximum_mass
        _require_preserved_positive_masses(
            floating_masses,
            scaled_masses,
            "maximum scaling",
        )
        normalized_masses = scaled_masses ./ sum(scaled_masses)
        _require_preserved_positive_masses(
            scaled_masses,
            normalized_masses,
            "final normalization",
        )
        copied_proposals = copy(proposals)
        return new{typeof(copied_proposals),typeof(normalized_masses)}(
            copied_proposals,
            normalized_masses,
        )
    end
end

function _require_preserved_positive_masses(source, destination, operation)
    for index in eachindex(source, destination)
        source[index] > 0 && iszero(destination[index]) && throw(
            ArgumentError(
                "positive proposal mass at index $index became zero during $operation",
            ),
        )
    end
    return nothing
end

struct _ActiveProposalBank{P,M,C,I}
    proposals::P
    logmasses::M
    cdf::C
    proposal_ids::I
end

struct _ScalarGaussianLayout end
struct _VectorGaussianLayout end

struct _PackedDiagonalGaussianBank{L,S,N,M,C,I,R}
    locations::L
    scales::S
    lognormalizers::N
    logmasses::M
    cdf::C
    proposal_ids::I
    layout::R
end

struct _PackedFactorGaussianBank{L,F,N,M,C,I}
    locations::L
    factors::F
    lognormalizers::N
    logmasses::M
    cdf::C
    proposal_ids::I
end

Adapt.@adapt_structure _PackedDiagonalGaussianBank
Adapt.@adapt_structure _PackedFactorGaussianBank

_active_proposal_count(bank::_ActiveProposalBank) = length(bank.proposals)
_active_proposal_count(bank::_PackedDiagonalGaussianBank) = size(bank.locations, 2)
_active_proposal_count(bank::_PackedFactorGaussianBank) = size(bank.locations, 2)

function _accelerator_proposal_limit(bank::ProposalBank)
    proposal_ids = findall(!iszero, bank.masses)
    proposals = view(bank.proposals, proposal_ids)
    for proposal in proposals
        proposal isa ProductProposal && return :product_proposal_cpu_only
        proposal isa TransformedProposal && return :transformed_proposal_cpu_only
        proposal isa _GaussianProposal || return :generic_proposal_cpu_only
        proposal.scale isa _FactorGaussianScale && return :factor_proposal_cpu_only
        _is_packable_native_gaussian(proposal) || return :generic_proposal_cpu_only
    end

    return nothing
end

function _prepare_active_proposal_metadata(bank::ProposalBank)
    proposal_ids = findall(!iszero, bank.masses)
    sort!(
        proposal_ids;
        by=proposal_id -> bank.masses[proposal_id],
        alg=Base.Sort.MergeSort,
    )
    masses = bank.masses[proposal_ids]
    masses ./= sum(masses)
    cdf = cumsum(masses)
    cdf[end] = one(eltype(cdf))
    logmasses = similar(cdf)
    previous = zero(eltype(cdf))
    for slot in eachindex(cdf)
        boundary = cdf[slot]
        boundary > previous || throw(
            ArgumentError(
                "every positive proposal mass must retain a positive assignment interval",
            ),
        )
        logmasses[slot] = log(boundary - previous)
        previous = boundary
    end
    logmasses .-= LogExpFunctions.logsumexp(logmasses)
    return proposal_ids, logmasses, cdf
end

function _is_packable_native_gaussian(proposal)
    proposal isa _GaussianProposal || return false
    return proposal.scale isa Union{
        _SphericalGaussianScale,
        _DiagonalGaussianScale,
        _FactorGaussianScale,
    }
end

_native_gaussian_pack_kind(::Type) = Val(:dynamic)
_native_gaussian_pack_kind(
    ::Type{<:_GaussianProposal{F,L,S,T}},
) where {F,L,S<:_FactorGaussianScale,T} = Val(:factor)
_native_gaussian_pack_kind(
    ::Type{<:_GaussianProposal{F,L,S,T}},
) where {
    F,
    L,
    S<:Union{_SphericalGaussianScale,_DiagonalGaussianScale},
    T,
} = Val(:diagonal)

function _copy_packed_gaussian_factor!(destination, proposal, slot)
    scale = proposal.scale
    dimension = size(destination, 1)
    if scale isa _SphericalGaussianScale
        for coordinate in 1:dimension
            @inbounds destination[coordinate, coordinate, slot] = scale.scale
        end
    elseif scale isa _DiagonalGaussianScale
        for coordinate in 1:dimension
            @inbounds destination[coordinate, coordinate, slot] =
                scale.scales[coordinate]
        end
    else
        factor = scale.factor
        size(factor) == (dimension, dimension) || throw(
            DimensionMismatch("Gaussian factor must match the proposal dimension"),
        )
        for column in 1:dimension
            for row in 1:(column - 1)
                iszero(@inbounds(factor[row, column])) || throw(
                    ArgumentError("Gaussian factor must be lower triangular"),
                )
            end
            for row in column:dimension
                @inbounds destination[row, column, slot] = factor[row, column]
            end
        end
    end
    return destination
end

function _pack_native_gaussian_storage(
    locations,
    lognormalizers,
    logmasses,
    cdf,
    proposal_ids,
    proposals,
    layout,
    ::Val{:factor},
)
    T = eltype(locations)
    dimension = size(locations, 1)
    factors = zeros(T, dimension, dimension, length(proposals))
    for (slot, proposal) in pairs(proposals)
        _copy_packed_gaussian_factor!(factors, proposal, slot)
    end
    return _PackedFactorGaussianBank(
        locations,
        factors,
        lognormalizers,
        logmasses,
        cdf,
        proposal_ids,
    )
end

function _pack_native_gaussian_storage(
    locations,
    lognormalizers,
    logmasses,
    cdf,
    proposal_ids,
    proposals,
    layout,
    ::Val{:diagonal},
)
    T = eltype(locations)
    dimension = size(locations, 1)
    scales = Matrix{T}(undef, dimension, length(proposals))
    for (slot, proposal) in pairs(proposals)
        if proposal.scale isa _SphericalGaussianScale
            fill!(view(scales, :, slot), proposal.scale.scale)
        else
            copyto!(view(scales, :, slot), proposal.scale.scales)
        end
    end
    return _PackedDiagonalGaussianBank(
        locations,
        scales,
        lognormalizers,
        logmasses,
        cdf,
        proposal_ids,
        layout,
    )
end

function _pack_native_gaussian_storage(
    locations,
    lognormalizers,
    logmasses,
    cdf,
    proposal_ids,
    proposals,
    layout,
    ::Val{:dynamic},
)
    kind = any(proposal -> proposal.scale isa _FactorGaussianScale, proposals) ?
           Val(:factor) : Val(:diagonal)
    return _pack_native_gaussian_storage(
        locations,
        lognormalizers,
        logmasses,
        cdf,
        proposal_ids,
        proposals,
        layout,
        kind,
    )
end

function _pack_native_gaussian_bank(bank::ProposalBank)
    proposal_ids, logmasses, cdf = _prepare_active_proposal_metadata(bank)
    return _pack_native_gaussian_bank(bank, proposal_ids, logmasses, cdf)
end

function _pack_native_gaussian_bank(bank, proposal_ids, logmasses, cdf)
    return _pack_native_gaussian_bank(
        bank,
        proposal_ids,
        logmasses,
        cdf,
        _native_gaussian_pack_kind(eltype(bank.proposals)),
    )
end

function _pack_native_gaussian_bank(
    bank,
    proposal_ids,
    logmasses,
    cdf,
    pack_kind,
)
    proposals = view(bank.proposals, proposal_ids)
    all(_is_packable_native_gaussian, proposals) || return nothing
    eltype(logmasses) <: _NativeGaussianFloat || return nothing

    first_proposal = first(proposals)
    first_location = first_proposal.location
    scalar_layout = first_location isa _NativeGaussianFloat
    layout = scalar_layout ? _ScalarGaussianLayout() : _VectorGaussianLayout()
    T = _gaussian_float_type(first_location)
    dimension = _gaussian_dimension(first_location)

    for proposal in Iterators.drop(proposals, 1)
        location = proposal.location
        (location isa _NativeGaussianFloat) == scalar_layout || throw(
            ArgumentError(
                "positive-mass native Gaussian proposals must share scalar or vector layout",
            ),
        )
        _gaussian_float_type(location) === T || throw(
            ArgumentError(
                "positive-mass native Gaussian proposals must use one floating type",
            ),
        )
        _gaussian_dimension(location) == dimension || throw(
            DimensionMismatch(
                "positive-mass native Gaussian proposals must have one common dimension",
            ),
        )
    end

    locations = Matrix{T}(undef, dimension, length(proposals))
    lognormalizers = Vector{T}(undef, length(proposals))
    for (slot, proposal) in pairs(proposals)
        if scalar_layout
            locations[1, slot] = proposal.location
        else
            copyto!(view(locations, :, slot), proposal.location)
        end
        lognormalizers[slot] = proposal.lognormalizer
    end

    return _pack_native_gaussian_storage(
        locations,
        lognormalizers,
        logmasses,
        cdf,
        proposal_ids,
        proposals,
        layout,
        pack_kind,
    )
end

function _prepare_active_proposal_bank(bank::ProposalBank)
    proposal_ids, logmasses, cdf = _prepare_active_proposal_metadata(bank)
    proposal_type = eltype(bank.proposals)
    native_candidate = Val(
        proposal_type <: _GaussianProposal || !isconcretetype(proposal_type),
    )
    return _prepare_active_proposal_bank(
        bank,
        proposal_ids,
        logmasses,
        cdf,
        native_candidate,
    )
end

function _prepare_active_proposal_bank(
    bank,
    proposal_ids,
    logmasses,
    cdf,
    ::Val{true},
)
    return _prepare_active_proposal_bank(
        bank,
        proposal_ids,
        logmasses,
        cdf,
        _native_gaussian_pack_kind(eltype(bank.proposals)),
    )
end

function _prepare_active_proposal_bank(
    bank,
    proposal_ids,
    logmasses,
    cdf,
    ::Val{:factor},
)
    return _prepare_generic_active_proposal_bank(bank, proposal_ids, logmasses, cdf)
end

function _prepare_active_proposal_bank(
    bank,
    proposal_ids,
    logmasses,
    cdf,
    ::Val{:diagonal},
)
    packed = _pack_native_gaussian_bank(
        bank,
        proposal_ids,
        logmasses,
        cdf,
        Val(:diagonal),
    )
    isnothing(packed) || return packed
    return _prepare_generic_active_proposal_bank(bank, proposal_ids, logmasses, cdf)
end

function _prepare_active_proposal_bank(
    bank,
    proposal_ids,
    logmasses,
    cdf,
    ::Val{:dynamic},
)
    proposals = view(bank.proposals, proposal_ids)
    any(
        proposal -> proposal isa _GaussianProposal &&
                    proposal.scale isa _FactorGaussianScale,
        proposals,
    ) && throw(
        ArgumentError(
            "generic CPU proposal banks require one concrete proposal element type",
        ),
    )
    return _prepare_active_proposal_bank(
        bank,
        proposal_ids,
        logmasses,
        cdf,
        Val(:diagonal),
    )
end

function _prepare_active_proposal_bank(
    bank,
    proposal_ids,
    logmasses,
    cdf,
    ::Val{false},
)
    return _prepare_generic_active_proposal_bank(bank, proposal_ids, logmasses, cdf)
end

function _prepare_generic_active_proposal_bank(bank, proposal_ids, logmasses, cdf)
    proposal_type = eltype(bank.proposals)
    isconcretetype(proposal_type) || throw(
        ArgumentError(
            "generic CPU proposal banks require one concrete proposal element type",
        ),
    )
    proposals = bank.proposals[proposal_ids]
    return _ActiveProposalBank(
        proposals,
        logmasses,
        cdf,
        proposal_ids,
    )
end

function _proposal_dimension(bank::ProposalBank)
    dimension = nothing
    for proposal_id in eachindex(bank.proposals, bank.masses)
        iszero(bank.masses[proposal_id]) && continue
        proposal_dimension = _proposal_dimension(bank.proposals[proposal_id])
        proposal_dimension === nothing && return nothing
        if dimension === nothing
            dimension = proposal_dimension
        elseif proposal_dimension != dimension
            throw(
                DimensionMismatch(
                    "positive-mass proposals must have one common dimension",
                ),
            )
        end
    end
    return dimension
end

function ProposalBank(proposals::AbstractVector)
    isempty(proposals) && throw(ArgumentError("a proposal bank cannot be empty"))
    return ProposalBank(proposals, ones(Float64, length(proposals)))
end

"""
    AbstractMISScheme

Abstract supertype for complete static multiple-importance-sampling schemes.

Every subtype fixes both proposal assignment and the weight denominator. Scheme
values are passed as `mis_scheme` to [`ImportanceSampling`](@ref).
"""
abstract type AbstractMISScheme end

"""
    StratifiedMixture()

Fix stratified proposal assignments from the nominal bank masses before drawing
samples, and divide every target value by the full nominal proposal mixture.
The aggregate mixture must cover the target support. This is the default scheme
for a [`ProposalBank`](@ref).
"""
struct StratifiedMixture <: AbstractMISScheme end

"""
    RandomMixture()

Assign every sample independently from the nominal bank masses, and divide by
the full nominal proposal mixture. The aggregate mixture must cover the target
support. Compared with [`StratifiedMixture`](@ref), this retains iid mixture
selection but usually has more assignment-count variation.
"""
struct RandomMixture <: AbstractMISScheme end

"""
    StandardMIS()

Use stratified proposal assignment and divide each target value only by the
sample's generating proposal density. Every positive-mass proposal must cover
the target support. This costs one proposal-density evaluation per sample but
normally gives higher variance than a valid full-mixture denominator.
"""
struct StandardMIS <: AbstractMISScheme end

"""
    PartialDeterministicMixture(groups)

Use stratified assignment and divide by the nominal mixture within the sample's
generating group. `groups` must partition all stable proposal IDs exactly once;
groups containing only zero-mass proposals are accepted as inert. Each active
group mixture must cover the target support.

This interpolates between [`StandardMIS`](@ref) (singleton groups) and the full
mixture denominator (one group), with proposal-density cost proportional to the
generating group size.
"""
struct PartialDeterministicMixture{G} <: AbstractMISScheme
    groups::G
end
