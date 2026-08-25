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
`masses` fields preserve input order; masses sum to one.

Zero-mass proposals remain in the configuration and retain their stable
one-based IDs, but are neither assigned nor evaluated. Positive-mass proposals
must share a sample dimension. Generic CPU execution additionally requires one
concrete proposal element type. Native spherical and diagonal Gaussian banks
with one scalar/vector layout and floating type may be packed during
preparation.

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
        all(isfinite, floating_masses) || throw(ArgumentError("masses must be finite"))
        all(>=(zero(floating_type)), floating_masses) || throw(
            ArgumentError("masses must be nonnegative"),
        )
        maximum_mass = maximum(floating_masses)
        maximum_mass > zero(floating_type) || throw(
            ArgumentError("at least one mass must be positive"),
        )

        normalized_masses = floating_masses ./ maximum_mass
        normalized_masses ./= sum(normalized_masses)
        copied_proposals = copy(proposals)
        return new{typeof(copied_proposals),typeof(normalized_masses)}(
            copied_proposals,
            normalized_masses,
        )
    end
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

Adapt.@adapt_structure _PackedDiagonalGaussianBank

_active_proposal_count(bank::_ActiveProposalBank) = length(bank.proposals)
_active_proposal_count(bank::_PackedDiagonalGaussianBank) = size(bank.locations, 2)

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
    previous = zero(eltype(cdf))
    for boundary in cdf
        boundary > previous || throw(
            ArgumentError(
                "every positive proposal mass must retain a positive assignment interval",
            ),
        )
        previous = boundary
    end
    return proposal_ids, masses, cdf
end

function _is_packable_native_gaussian(proposal)
    proposal isa _GaussianProposal || return false
    return proposal.scale isa Union{
        _SphericalGaussianScale,
        _DiagonalGaussianScale,
    }
end

function _pack_native_gaussian_bank(bank::ProposalBank)
    proposal_ids, masses, cdf = _prepare_active_proposal_metadata(bank)
    return _pack_native_gaussian_bank(bank, proposal_ids, masses, cdf)
end

function _pack_native_gaussian_bank(bank, proposal_ids, masses, cdf)
    proposals = view(bank.proposals, proposal_ids)
    all(_is_packable_native_gaussian, proposals) || return nothing
    eltype(masses) <: _NativeGaussianFloat || return nothing

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
    scales = Matrix{T}(undef, dimension, length(proposals))
    lognormalizers = Vector{T}(undef, length(proposals))
    for (slot, proposal) in pairs(proposals)
        if scalar_layout
            locations[1, slot] = proposal.location
        else
            copyto!(view(locations, :, slot), proposal.location)
        end
        if proposal.scale isa _SphericalGaussianScale
            fill!(view(scales, :, slot), proposal.scale.scale)
        else
            copyto!(view(scales, :, slot), proposal.scale.scales)
        end
        lognormalizers[slot] = proposal.lognormalizer
    end

    return _PackedDiagonalGaussianBank(
        locations,
        scales,
        lognormalizers,
        log.(masses),
        cdf,
        proposal_ids,
        layout,
    )
end

function _prepare_active_proposal_bank(bank::ProposalBank)
    proposal_ids, masses, cdf = _prepare_active_proposal_metadata(bank)
    proposal_type = eltype(bank.proposals)
    native_candidate = Val(
        proposal_type <: _GaussianProposal || !isconcretetype(proposal_type),
    )
    return _prepare_active_proposal_bank(
        bank,
        proposal_ids,
        masses,
        cdf,
        native_candidate,
    )
end

function _prepare_active_proposal_bank(
    bank,
    proposal_ids,
    masses,
    cdf,
    ::Val{true},
)
    packed = _pack_native_gaussian_bank(bank, proposal_ids, masses, cdf)
    isnothing(packed) || return packed
    return _prepare_generic_active_proposal_bank(bank, proposal_ids, masses, cdf)
end

function _prepare_active_proposal_bank(
    bank,
    proposal_ids,
    masses,
    cdf,
    ::Val{false},
)
    return _prepare_generic_active_proposal_bank(bank, proposal_ids, masses, cdf)
end

function _prepare_generic_active_proposal_bank(bank, proposal_ids, masses, cdf)
    proposal_type = eltype(bank.proposals)
    isconcretetype(proposal_type) || throw(
        ArgumentError(
            "generic CPU proposal banks require one concrete proposal element type",
        ),
    )
    proposals = bank.proposals[proposal_ids]
    return _ActiveProposalBank(
        proposals,
        log.(masses),
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
