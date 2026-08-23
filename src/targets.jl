"""
    LogTarget(logdensity)

Mark `logdensity` as a package callable target.

This explicit wrapper takes priority over target interfaces advertised by the
wrapped object. The callable must accept `logdensity(sample)` in the
context-free API or `logdensity(sample, p)` when an explicit context is passed,
and must return a `Float32` or `Float64` log density.
"""
struct LogTarget{F}
    logdensity::F
end

abstract type _BoundTarget end

struct _BoundContextFreeTarget{F} <: _BoundTarget
    logdensity::F
end

struct _BoundContextualTarget{F,P} <: _BoundTarget
    logdensity::F
    context::P
end

struct _BoundLogDensityProblemsTarget{T} <: _BoundTarget
    target::T
end

struct _BoundDensityInterfaceTarget{T} <: _BoundTarget
    target::T
end

@inline (target::_BoundContextFreeTarget)(sample) = target.logdensity(sample)
@inline (target::_BoundContextualTarget)(sample) = target.logdensity(sample, target.context)
@inline (target::_BoundLogDensityProblemsTarget)(sample) =
    LogDensityProblems.logdensity(target.target, sample)
@inline (target::_BoundDensityInterfaceTarget)(sample) =
    DensityInterface.logdensityof(target.target, sample)

_bind_resolved_target(target::_BoundTarget, sample) = target

_proposal_dimension(proposal) = nothing

function _prepare_target(target::LogTarget, proposal)
    return _ContextFreePreparedTarget(target.logdensity)
end

function _prepare_target(target::LogTarget, context, proposal)
    return _ContextualPreparedTarget(target.logdensity, context)
end

function _prepare_target(target, proposal)
    capabilities = LogDensityProblems.capabilities(target)
    if capabilities !== nothing
        capabilities isa LogDensityProblems.LogDensityOrder || throw(
            ArgumentError(
                "LogDensityProblems.capabilities returned an invalid value for " *
                "$(typeof(target))",
            ),
        )
        _validate_target_dimension(target, proposal)
        return _BoundLogDensityProblemsTarget(target)
    end

    density_kind = DensityInterface.DensityKind(target)
    if density_kind isa DensityInterface.IsOrHasDensity
        return _BoundDensityInterfaceTarget(target)
    elseif !(density_kind isa DensityInterface.NoDensity)
        throw(
            ArgumentError(
                "DensityInterface.DensityKind returned an invalid value for " *
                "$(typeof(target))",
            ),
        )
    end

    return _ContextFreePreparedTarget(target)
end

function _prepare_target(target, context, proposal)
    return _ContextualPreparedTarget(target, context)
end

function _bind_target(target, proposal, samples)
    prepared_target = _prepare_target(target, proposal)
    return _bind_resolved_target(prepared_target, _sample_at(samples, 1))
end

function _bind_target(target, context, proposal, samples)
    prepared_target = _prepare_target(target, context, proposal)
    return _bind_resolved_target(prepared_target, _sample_at(samples, 1))
end

function _bind_context_free_callable(logdensity, sample)
    applicable(logdensity, sample) || throw(
        ArgumentError(
            "target of type $(typeof(logdensity)) is not callable with one sample argument",
        ),
    )
    return _BoundContextFreeTarget(logdensity)
end

function _bind_contextual_callable(logdensity, context, sample)
    applicable(logdensity, sample, context) || throw(
        ArgumentError(
            "an explicit p requires a target callable as logtarget(sample, p); " *
            "target type $(typeof(logdensity)) does not support that form",
        ),
    )
    return _BoundContextualTarget(logdensity, context)
end

function _validate_target_dimension(target, proposal)
    proposal_dimension = _proposal_dimension(proposal)
    proposal_dimension === nothing && return nothing
    proposal_dimension isa Integer && proposal_dimension >= 0 || throw(
        ArgumentError("proposal dimension must be a nonnegative integer or nothing"),
    )

    target_dimension = LogDensityProblems.dimension(target)
    target_dimension == proposal_dimension || throw(
        DimensionMismatch(
            "target dimension $target_dimension does not match proposal dimension " *
            "$proposal_dimension",
        ),
    )
    return nothing
end
