import DensityInterface
import Random

struct DMPMCTarget{T<:AbstractFloat} end

(::DMPMCTarget{T})(sample) where {T} = -T(0.5) * sum(abs2, sample)

mutable struct DMPMCGenericProposal
    draw_count::Base.RefValue{Int}
end

DMPMCGenericProposal() = DMPMCGenericProposal(Ref(0))

function Random.rand(rng::Random.AbstractRNG, proposal::DMPMCGenericProposal)
    proposal.draw_count[] += 1
    return Random.randn(rng)
end

DensityInterface.logdensityof(::DMPMCGenericProposal, sample::Real) = -abs2(sample) / 2

function dm_pmc_counts_by_proposal_id(prepared_bank, plan, configured_count)
    counts = zeros(Int, configured_count, size(plan.counts, 2))
    for (slot, proposal_id) in pairs(prepared_bank.proposal_ids)
        counts[proposal_id, :] .= plan.counts[slot, :]
    end
    return counts
end
