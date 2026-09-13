import DensityInterface
import KernelAbstractions
import LogExpFunctions
import Random

struct DMPMCTarget{T<:AbstractFloat} end

(::DMPMCTarget{T})(sample) where {T} = -T(0.5) * sum(abs2, sample)

mutable struct DMPMCPrefilledRNG{T<:AbstractFloat} <: Random.AbstractRNG
    normal_batches::Vector{Vector{T}}
    uniform_batches::Vector{Vector{T}}
    normal_index::Int
    uniform_index::Int
end

function DMPMCPrefilledRNG(
    normal_batches::Vector{Vector{T}},
    uniform_batches::Vector{Vector{T}},
) where {T<:AbstractFloat}
    return DMPMCPrefilledRNG{T}(normal_batches, uniform_batches, 1, 1)
end

function Random.randn!(rng::DMPMCPrefilledRNG, destination::AbstractArray)
    batch = rng.normal_batches[rng.normal_index]
    length(batch) >= length(destination) || throw(
        DimensionMismatch("prefilled normal batch is too short"),
    )
    copyto!(destination, 1, batch, 1, length(destination))
    rng.normal_index += 1
    return destination
end

function Random.rand!(rng::DMPMCPrefilledRNG, destination::AbstractArray)
    batch = rng.uniform_batches[rng.uniform_index]
    length(batch) >= length(destination) || throw(
        DimensionMismatch("prefilled uniform batch is too short"),
    )
    copyto!(destination, 1, batch, 1, length(destination))
    rng.uniform_index += 1
    return destination
end

struct DMPMCResultFailurePrototype{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    storage::A
end

struct DMPMCResultFailureArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    storage::A
end

for ArrayType in (DMPMCResultFailurePrototype, DMPMCResultFailureArray)
    @eval begin
        Base.size(array::$ArrayType) = size(array.storage)
        Base.IndexStyle(::Type{<:$ArrayType}) = IndexLinear()
        Base.getindex(array::$ArrayType, indices...) = getindex(array.storage, indices...)
        Base.setindex!(array::$ArrayType, value, indices...) =
            setindex!(array.storage, value, indices...)
        KernelAbstractions.get_backend(array::$ArrayType) =
            KernelAbstractions.get_backend(array.storage)
    end
end

function Base.similar(
    array::DMPMCResultFailurePrototype,
    ::Type{T},
    dimensions::Dims{N},
) where {T,N}
    return DMPMCResultFailureArray(similar(array.storage, T, dimensions))
end

Base.require_one_based_indexing(::DMPMCResultFailureArray) =
    error("intentional DM-PMC result validation failure")

function dm_pmc_result_failure_sampler(sampler)
    old_buffers = sampler.random_buffers
    buffers = ImportanceSamplers._DMPMCRandomBuffers(
        DMPMCResultFailurePrototype(old_buffers.normals),
        old_buffers.resampling_uniforms,
        old_buffers.failure_scratch,
        old_buffers.radial,
    )
    return ImportanceSamplers._PreparedImportanceSampler(
        sampler.rng,
        buffers,
        sampler.target,
        sampler.algorithm,
        sampler.method_state,
        sampler.device,
        sampler.factor_execution,
        sampler.threaded,
        false,
        false,
    )
end

struct DMPMCMixtureTarget{T,V<:AbstractVector{T}}
    locations::V
    scales::V
    logcoefficients::V
end

function dm_pmc_normal_logdensity(location::T, scale::T, sample::T) where {T}
    return -log(scale) - T(0.5) * log(T(2) * T(pi)) -
           T(0.5) * abs2((sample - location) / scale)
end

function (target::DMPMCMixtureTarget{T})(sample) where {T}
    value = T(-Inf)
    for slot in eachindex(target.locations, target.scales, target.logcoefficients)
        term = target.logcoefficients[slot] + dm_pmc_normal_logdensity(
            target.locations[slot],
            target.scales[slot],
            T(sample),
        )
        value = LogExpFunctions.logaddexp(value, term)
    end
    return value
end

mutable struct DMPMCFailAfterTarget{T<:AbstractFloat}
    evaluations::Base.RefValue{Int}
    fail_after::Int
end

DMPMCFailAfterTarget(::Type{T}, fail_after) where {T<:AbstractFloat} =
    DMPMCFailAfterTarget{T}(Ref(0), fail_after)

function (target::DMPMCFailAfterTarget{T})(sample) where {T}
    target.evaluations[] += 1
    target.evaluations[] >= target.fail_after && error("intentional DM-PMC target failure")
    return -T(0.5) * sum(abs2, sample)
end

mutable struct DMPMCNegativeInfinityAfterTarget{T<:AbstractFloat}
    evaluations::Base.RefValue{Int}
    switch_after::Int
end

DMPMCNegativeInfinityAfterTarget(::Type{T}, switch_after) where {T<:AbstractFloat} =
    DMPMCNegativeInfinityAfterTarget{T}(Ref(0), switch_after)

function (target::DMPMCNegativeInfinityAfterTarget{T})(sample) where {T}
    target.evaluations[] += 1
    target.evaluations[] > target.switch_after && return T(-Inf)
    return -T(0.5) * sum(abs2, sample)
end

function dm_pmc_multinomial_oracle(cdf, uniforms)
    last_index = lastindex(cdf)
    return [
        min(searchsortedlast(cdf, uniform) + 1, last_index) for
        uniform in uniforms
    ]
end

function dm_pmc_scalar_oracle(
    initial_locations,
    scales,
    proposal_ids,
    plan,
    normal_batches,
    uniform_batches,
    target,
    resampling=:global,
)
    T = eltype(initial_locations)
    locations = copy(initial_locations)
    samples = T[]
    logweights = T[]
    rounds = Int[]
    generated_ids = Int[]
    round_ancestors = Vector{Vector{Int}}()
    for round in eachindex(normal_batches, uniform_batches)
        round_size = plan.schedule[round]
        assignments = view(plan.assignments, 1:round_size, round)
        round_samples = Vector{T}(undef, round_size)
        round_logweights = Vector{T}(undef, round_size)
        for sample_index in 1:round_size
            slot = assignments[sample_index]
            sample = locations[slot] + scales[slot] * normal_batches[round][sample_index]
            round_samples[sample_index] = sample
            denominator = T(-Inf)
            for denominator_slot in eachindex(locations)
                term = plan.logcoefficients[denominator_slot, round] +
                       dm_pmc_normal_logdensity(
                    locations[denominator_slot],
                    scales[denominator_slot],
                    sample,
                )
                denominator = LogExpFunctions.logaddexp(denominator, term)
            end
            round_logweights[sample_index] = target(sample) - denominator
        end
        ancestors = if resampling === :global
            normalized = exp.(
                round_logweights .- LogExpFunctions.logsumexp(round_logweights),
            )
            cdf = cumsum(normalized)
            cdf[end] = one(T)
            dm_pmc_multinomial_oracle(
                cdf,
                view(uniform_batches[round], 1:length(locations)),
            )
        else
            map(eachindex(locations)) do slot
                indices = findall(==(slot), assignments)
                local_logweights = round_logweights[indices]
                normalized = exp.(
                    local_logweights .- LogExpFunctions.logsumexp(local_logweights),
                )
                cdf = cumsum(normalized)
                cdf[end] = one(T)
                indices[only(dm_pmc_multinomial_oracle(
                    cdf,
                    view(uniform_batches[round], slot:slot),
                ))]
            end
        end
        locations .= round_samples[ancestors]
        append!(samples, round_samples)
        append!(logweights, round_logweights)
        append!(rounds, fill(round, round_size))
        append!(generated_ids, proposal_ids[assignments])
        push!(round_ancestors, ancestors)
    end
    return (; samples, logweights, rounds, proposal_ids=generated_ids, locations, round_ancestors)
end

struct DMPMCLeftEmptyTarget{T<:AbstractFloat} end

function (::DMPMCLeftEmptyTarget{T})(sample) where {T}
    sample < zero(sample) && return T(-Inf)
    return -T(0.5) * abs2(sample)
end

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
