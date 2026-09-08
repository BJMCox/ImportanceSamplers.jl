"""A failed LAIS round, retaining the pre-call bank and transition state."""
struct LAISRoundError{E,D<:NamedTuple} <: Exception
    round::Int
    phase::Symbol
    cause::E
    diagnostics::D
end

function Base.showerror(io::IO, error::LAISRoundError)
    print(io, "LAIS failed in round ", error.round, " during ", error.phase, ": ")
    showerror(io, error.cause)
end

_population_round_error(::LAIS) = LAISRoundError
_population_round_phase(::LAIS) = :weight_summary
_population_mis_adaptation(::_PreparedLAIS, round_views) = nothing

function _population_round_views(state::_PreparedLAIS, round)
    count = state.plan.schedule[round]
    workspace = state.workspace
    return (
        round_size=count,
        samples=_sample_view(workspace.round_samples, 1:count),
        logweights=view(workspace.round_logweights, 1:count),
        proposal_ids=view(workspace.round_proposal_ids, 1:count),
        assignments=view(state.plan.assignments, 1:count, round),
    )
end

function _reset_population_run!(state::_PreparedLAIS, run, committed)
    copyto!(state.run_transition, state.transition)
    return nothing
end

function _before_population_round!(sampler, state::_PreparedLAIS, target, round, execution, transfers)
    transition!(state.run_transition, target, sampler.rng, execution, transfers)
    return nothing
end

_advance_population!(sampler, state::_PreparedLAIS, views, round, execution, transfers) = nothing
_commit_population_round!(::_PreparedLAIS, bank) = nothing

function _commit_population_run!(state::_PreparedLAIS)
    state.bank, state.run_bank = state.run_bank, state.bank
    state.transition, state.run_transition = state.run_transition, state.transition
    return nothing
end

function _population_diagnostics(sampler, state::_PreparedLAIS, execution, schedule,
    round_ess, round_lognormalizers, transfers)
    names = (:initial_target_evaluations, :warmup_target_evaluations,
        :production_target_evaluations, :warmup_proposals, :production_proposals, :accepted)
    after = transition_diagnostics(state.run_transition, transfers)
    before = transition_diagnostics(state.transition, transfers)
    transition = NamedTuple{names}(map(name -> getproperty(after, name) - getproperty(before, name), names))
    lower_count = sum(schedule)
    return (
        method=:lais, execution=_execution_name(execution), threaded=sampler.threaded,
        factor_execution_policy=_factor_execution_name(sampler.device, sampler.factor_execution),
        rounds=sampler.algorithm.rounds, round_sizes=collect(schedule),
        round_ess=round_ess, round_lognormalizers=round_lognormalizers,
        target_evaluations=lower_count + transition.initial_target_evaluations +
            transition.warmup_target_evaluations + transition.production_target_evaluations,
        proposal_evaluations=length(sampler.algorithm.bank.proposals) * lower_count,
        transition=transition, failures=0, transfers=transfers,
    )
end

_importance_sample_cpu!(sampler, state::_PreparedLAIS, threaded) =
    _importance_sample_fixed_population!(sampler, state, threaded)
