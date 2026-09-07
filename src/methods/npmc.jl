"""
    NPMC(proposal; rounds, round_size)

Configure nonlinear population Monte Carlo with a native scalar or vector
Gaussian proposal and a fixed round schedule, as accepted by [`AMIS`](@ref).
Each round fits a full Gaussian to that round's samples using weights clipped
at the `isqrt(n)`-th largest weight, where `n` is the round size. The fit adds
the same scale-aware covariance ridge as AMIS.

Returned samples retain their raw `logtarget - logproposal` weights and round
provenance. This is the N-PMC adaptation variant, not the original paper's
transformed-weight estimator. `diagnostics.round_ess` and
`diagnostics.round_lognormalizers` summarize each round's raw weights.
`diagnostics.adaptation_ess` summarizes the clipped weights used for fitting.
The final fitted proposal persists after a successful call. A failed call
retains the last committed proposal and throws [`NPMCRoundError`](@ref).
"""
struct NPMC{P,S} <: _AdaptiveGaussianSampler
    proposal::P
    rounds::Int
    round_size::S

    function NPMC(proposal; rounds, round_size)
        schedule = _validate_adaptive_schedule(rounds, round_size)
        prepared = _prepare_proposal_input(proposal)
        _validate_adaptive_gaussian_proposal(prepared)
        return new{typeof(prepared),typeof(schedule)}(prepared, rounds, schedule)
    end
end

"""
    NPMCRoundError

An N-PMC round failed before committing its fitted proposal. `round`, `phase`,
`cause`, and `diagnostics` follow the [`AMISRoundError`](@ref) contract.
Clipping to an all-zero adaptation population throws this error with an
[`AllZeroWeightsError`](@ref) cause, even when some raw weights are nonzero.
"""
struct NPMCRoundError{E,D<:NamedTuple} <: Exception
    round::Int
    phase::Symbol
    cause::E
    diagnostics::D
end

function Base.showerror(io::IO, error::NPMCRoundError)
    print(io, "NPMC failed in round ", error.round, " during ", error.phase, ": ")
    showerror(io, error.cause)
end

function _copy_algorithm(device, algorithm::NPMC)
    return NPMC(
        _copy_to_device(device, algorithm.proposal);
        rounds=algorithm.rounds,
        round_size=algorithm.round_size,
    )
end

function _retarget_algorithm(
    sampler::_PreparedImportanceSampler{R,B,T,A},
) where {R,B,T,A<:NPMC}
    algorithm = sampler.algorithm
    return NPMC(
        current_proposal(MLDataDevices.cpu_device(), sampler);
        rounds=algorithm.rounds,
        round_size=algorithm.round_size,
    )
end
