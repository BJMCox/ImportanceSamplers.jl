mutable struct _ValidatedAMISToken end
const _VALIDATED_AMIS_TOKEN = _ValidatedAMISToken()

"""
    AMIS(proposal; rounds, round_size)

Configure adaptive multiple importance sampling with a fixed round schedule.
`proposal` must be a native `Float32` or `Float64` scalar spherical, vector
spherical, diagonal, or factor Gaussian or Student-t. Student-t fits require
`nu > 2`, keep `nu` fixed, and convert fitted covariance to Student-t scale.
Loading Distributions.jl accepts its supported normal and Student-t forms
through native conversion. `round_size` is either one positive `Int` repeated for every
round or a positive `Vector{Int}` with one entry per round.

Each successful result reports `diagnostics.method == :amis`.
`diagnostics.round_ess[t]` and `diagnostics.round_lognormalizers[t]` summarize
all samples retained through round `t` using their retrospective weights after
that round. Each round log normalizer is a numerical estimate; no finite-sample
unbiasedness or generic consistency guarantee is claimed.
"""
struct AMIS{P,S} <: _MomentAdaptiveSampler
    proposal::P
    rounds::Int
    round_size::S

    function AMIS(
        proposal::P,
        rounds::Int,
        round_size::S,
        token::_ValidatedAMISToken,
    ) where {P,S}
        token === _VALIDATED_AMIS_TOKEN || throw(
            ArgumentError("invalid internal algorithm-construction token"),
        )
        return new{P,S}(proposal, rounds, round_size)
    end
end

function AMIS(proposal; rounds, round_size)
    schedule = _validate_adaptive_schedule(rounds, round_size)
    prepared_proposal = _prepare_proposal_input(proposal)
    _validate_moment_proposal(prepared_proposal)
    return AMIS(prepared_proposal, rounds, schedule, _VALIDATED_AMIS_TOKEN)
end


function _retarget_algorithm(
    sampler::_PreparedImportanceSampler{R,B,T,A},
) where {R,B,T,A<:AMIS}
    algorithm = sampler.algorithm
    return AMIS(
        current_proposal(MLDataDevices.cpu_device(), sampler);
        rounds=algorithm.rounds,
        round_size=algorithm.round_size,
    )
end

"""
    AMISRoundError

Exception thrown when an AMIS call fails before its learned proposal can be
committed. `round` and `phase` locate the failure, `cause` stores the
underlying exception, and `diagnostics` reports the requested `round_size`,
`completed_rounds`, and `cumulative_sample_count` retained through those
completed rounds. Factorization failures additionally report the smallest
stable covariance summary: its `minimum_diagonal` and
`maximum_absolute_entry`. `diagnostics.transfers` counts any explicit
failure-only device scalar transfers. The prepared sampler retains the proposal
committed by its previous successful call.
"""
struct AMISRoundError{E,D<:NamedTuple} <: Exception
    round::Int
    phase::Symbol
    cause::E
    diagnostics::D
end

function Base.showerror(io::IO, error::AMISRoundError)
    print(io, "AMIS failed in round ", error.round, " during ", error.phase, ": ")
    showerror(io, error.cause)
end

function _copy_algorithm(device, algorithm::AMIS)
    proposal = _copy_to_device(device, algorithm.proposal)
    _validate_moment_proposal(proposal)
    round_size = algorithm.round_size isa Vector ?
                 copy(algorithm.round_size) : algorithm.round_size
    return AMIS(
        proposal,
        algorithm.rounds,
        round_size,
        _VALIDATED_AMIS_TOKEN,
    )
end

function _preflight_accelerator_method(
    device,
    target,
    algorithm::AMIS,
    method_state::_PreparedMomentSampler,
    random_buffers::_RandomBuffers,
    factor_execution,
)
    return _preflight_amis_kernels(
        device,
        target,
        method_state,
        random_buffers,
        factor_execution,
    )
end
