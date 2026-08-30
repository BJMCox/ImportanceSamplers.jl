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
