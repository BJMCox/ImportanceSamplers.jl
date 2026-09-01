struct FirstOrderGRAMISTarget{T} end

function (::FirstOrderGRAMISTarget{T})(sample) where {T}
    return -T(0.5) * sum(abs2, sample)
end

function first_order_gramis_gradient!(destination, sample)
    destination .= -sample
    return destination
end

mutable struct FirstOrderGRAMISSchedule{T}
    values::Vector{T}
    calls::Vector{Int}
end

function (schedule::FirstOrderGRAMISSchedule)(round)
    push!(schedule.calls, round)
    return schedule.values[round]
end

struct FirstOrderGRAMISNonGaussian end

mutable struct FirstOrderGRAMISPrefilledRNG{T} <: Random.AbstractRNG
    batches::Vector{Vector{T}}
    next_batch::Int
end

FirstOrderGRAMISPrefilledRNG(batches::Vector{Vector{T}}) where {T} =
    FirstOrderGRAMISPrefilledRNG{T}(batches, 1)

function Random.randn!(rng::FirstOrderGRAMISPrefilledRNG, destination::AbstractArray)
    batch = rng.batches[rng.next_batch]
    length(batch) == length(destination) || throw(
        DimensionMismatch("prefilled GRAMIS normal batch has the wrong length"),
    )
    copyto!(destination, 1, batch, 1, length(destination))
    rng.next_batch += 1
    return destination
end

struct FirstOrderGRAMISShiftedTarget{T}
    center::T
end

function (target::FirstOrderGRAMISShiftedTarget{T})(sample) where {T}
    offset = only(sample) - target.center
    return -T(0.5) * abs2(offset)
end

function first_order_gramis_shifted_gradient!(destination, sample)
    destination[1] = eltype(destination)(0.5) - only(sample)
    return destination
end

mutable struct FirstOrderGRAMISCountingShiftedTarget{T}
    center::T
    calls::Int
end


function (target::FirstOrderGRAMISCountingShiftedTarget{T})(sample) where {T}
    target.calls += 1
    offset = only(sample) - target.center
    return -T(0.5) * abs2(offset)
end

mutable struct FirstOrderGRAMISCountingGradient
    calls::Int
end


function (gradient::FirstOrderGRAMISCountingGradient)(destination, sample)
    gradient.calls += 1
    destination[1] = eltype(destination)(0.5) - only(sample)
    return destination
end

function first_order_gramis_two_proposal_bank(::Type{T}=Float64) where {T}
    return ProposalBank([
        FactorGaussian(T[-2], reshape(T[0.75], 1, 1)),
        FactorGaussian(T[2], reshape(T[1.25], 1, 1)),
    ])
end

struct FirstOrderGRAMISMeanOnlyTarget{T}
    left::T
    right::T
end

function (target::FirstOrderGRAMISMeanOnlyTarget{T})(sample) where {T}
    value = only(sample)
    return value == target.left || value == target.right ? zero(T) : T(-Inf)
end

function first_order_gramis_zero_gradient!(destination, sample)
    fill!(destination, zero(eltype(destination)))
    return destination
end

function first_order_gramis_bank(::Type{T}=Float64) where {T}
    return ProposalBank([
        SphericalGaussian(T[-2, 0], T(0.75)),
        DiagonalGaussian(T[0, 2], T[1.25, 0.5]),
        FactorGaussian(T[2, 0], T[1 0; 0.25 1.5]),
    ])
end

function prepare_first_order_gramis(algorithm; threaded=false)
    target = LogTarget(
        FirstOrderGRAMISTarget{eltype(first(algorithm.bank.proposals).location)}();
        grad=first_order_gramis_gradient!,
    )
    return prepare_sampler(
        Random.Xoshiro(0x4752414d4953),
        target,
        algorithm;
        threaded,
    )
end
