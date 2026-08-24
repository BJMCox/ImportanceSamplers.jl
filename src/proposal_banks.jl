"""
    AbstractProposalPopulation

Abstract supertype for explicit populations of normalized proposals.
"""
abstract type AbstractProposalPopulation end

"""
    ProposalBank(proposals)
    ProposalBank(proposals, masses)

Construct an explicit proposal population with copied proposals and normalized
nonnegative masses. Omitted masses are equal. Zero-mass proposals remain in the
bank configuration.
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

function ProposalBank(proposals::AbstractVector)
    isempty(proposals) && throw(ArgumentError("a proposal bank cannot be empty"))
    return ProposalBank(proposals, ones(Float64, length(proposals)))
end

"""
    AbstractMISScheme

Abstract supertype for complete static multiple-importance-sampling schemes.
"""
abstract type AbstractMISScheme end

"""Use stratified proposal assignment and the full nominal mixture denominator."""
struct StratifiedMixture <: AbstractMISScheme end

"""Use independent mixture assignment and the full nominal mixture denominator."""
struct RandomMixture <: AbstractMISScheme end

"""Use stratified assignment and each sample's generating-proposal denominator."""
struct StandardMIS <: AbstractMISScheme end

"""
    PartialDeterministicMixture(groups)

Use stratified assignment and the nominal mixture within each proposal group.
"""
struct PartialDeterministicMixture{G} <: AbstractMISScheme
    groups::G
end
