"""
    LogTarget(logdensity)
    LogTarget(logdensity, adtype; grad=nothing)
    LogTarget(logdensity; grad=nothing)

Mark `logdensity` as a package callable target.

This explicit wrapper takes priority over target interfaces advertised by the
wrapped object. The callable must accept `logdensity(sample)` in the
context-free API or `logdensity(sample, p)` when an explicit context is passed,
and must return a `Float32` or `Float64` unnormalized log density. For CUDA,
use a device-compatible callable and pass numerical arrays through `p`; opaque
closure captures cannot be transferred reliably.

An explicit `grad` takes priority over `adtype`. CPU automatic gradients use
DifferentiationInterface. FirstOrderGRAMIS also supports reverse-mode
`ADTypes.AutoEnzyme()` on CUDA when Enzyme is loaded. It batches the existing
scalar logdensity callable internally; `p` remains constant during differentiation.
The callable and its operations must support Enzyme's device differentiation.
Structured GPU contexts can require Enzyme's runtime-activity mode; the supplied
AD mode is preserved, not changed implicitly.
Other accelerator AD backends and out-of-place explicit gradients are unsupported.
"""
struct LogTarget{F,A<:ADTypes.AbstractADType,G}
    logdensity::F
    adtype::A
    grad::G
end

LogTarget(f, adtype::ADTypes.AbstractADType; grad=nothing) =
    LogTarget(f, adtype, grad)
LogTarget(f; grad=nothing) = LogTarget(f, ADTypes.NoAutoDiff(), grad)

struct _NoTargetContext end

struct _PreparedLogTarget{F,P,A<:ADTypes.AbstractADType,G}
    logdensity::F
    context::P
    adtype::A
    gradient::G
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

Adapt.@adapt_structure LogTarget
Adapt.@adapt_structure _PreparedLogTarget
Adapt.@adapt_structure _BoundContextFreeTarget
Adapt.@adapt_structure _BoundContextualTarget
Adapt.@adapt_structure _BoundLogDensityProblemsTarget
Adapt.@adapt_structure _BoundDensityInterfaceTarget

@inline (target::_BoundContextFreeTarget)(sample) = target.logdensity(sample)
@inline (target::_BoundContextualTarget)(sample) = target.logdensity(sample, target.context)
@inline (target::_BoundLogDensityProblemsTarget)(sample) =
    LogDensityProblems.logdensity(target.target, sample)
@inline (target::_BoundDensityInterfaceTarget)(sample) =
    DensityInterface.logdensityof(target.target, sample)

_bind_resolved_target(target::_BoundTarget, sample) = target

function _has_explicit_function_adapt_rule(device, ::Type{F}) where {F<:Function}
    method = which(Adapt.adapt_structure, Tuple{typeof(device),F})
    device_function_method =
        which(Adapt.adapt_structure, Tuple{typeof(device),typeof(identity)})
    return method !== device_function_method
end

function _copy_target_callable(
    device::MLDataDevices.AbstractDevice,
    target::F,
) where {F<:Function}
    _has_explicit_function_adapt_rule(device, F) || return target
    return _copy_to_device(device, target)
end

_copy_target_callable(device::MLDataDevices.AbstractDevice, target) =
    _copy_to_device(device, target)

function _transfer_prepared_target(
    device::MLDataDevices.AbstractDevice,
    target::_PreparedLogTarget,
)
    return _PreparedLogTarget(
        _copy_target_callable(device, target.logdensity),
        _copy_to_device(device, target.context),
        _copy_to_device(device, target.adtype),
        _copy_target_callable(device, target.gradient),
    )
end

_transfer_prepared_target(device::MLDataDevices.AbstractDevice, target::_BoundTarget) =
    _copy_to_device(device, target)

function _target_has_opaque_host_closure(target::_PreparedLogTarget)
    return _has_opaque_host_closure(target.logdensity) ||
           _has_opaque_host_closure(target.gradient) ||
           _has_opaque_host_closure(target.context)
end

function _target_has_opaque_host_closure(
    target::_PreparedLogTarget,
    device::MLDataDevices.AbstractDevice,
)
    return _target_callable_has_opaque_host_closure(target.logdensity, device) ||
           _target_callable_has_opaque_host_closure(target.gradient, device) ||
           _has_opaque_host_closure(target.context)
end

function _target_callable_has_opaque_host_closure(
    target::F,
    device::MLDataDevices.AbstractDevice,
) where {F<:Function}
    _has_explicit_function_adapt_rule(device, F) && return false
    return _has_opaque_host_closure(target)
end

_target_callable_has_opaque_host_closure(target, device) =
    _has_opaque_host_closure(target)

function _target_has_opaque_host_closure(target::_BoundLogDensityProblemsTarget)
    return _has_opaque_host_closure(target.target)
end

function _target_has_opaque_host_closure(target::_BoundDensityInterfaceTarget)
    return _has_opaque_host_closure(target.target)
end

_target_has_opaque_host_closure(
    target::_BoundTarget,
    device::MLDataDevices.AbstractDevice,
) = _target_has_opaque_host_closure(target)

function _target_transfer_rewrites_opaque_closure(target::_PreparedLogTarget)
    return _field_transfer_rewrites_opaque_closure(target.logdensity) ||
           _field_transfer_rewrites_opaque_closure(target.gradient) ||
           _field_transfer_rewrites_opaque_closure(target.context)
end

_field_transfer_rewrites_opaque_closure(value) =
    !(value isa Function) && _has_opaque_host_closure(value)

function _target_transfer_rewrites_opaque_closure(target::_BoundTarget)
    return _target_has_opaque_host_closure(target)
end

_has_opaque_host_closure(target) = _has_opaque_host_closure(target, IdSet())

function _has_opaque_host_closure(target::Function, seen)
    return !isbitstype(typeof(target))
end

_has_opaque_host_closure(target::Type, seen) = false
_has_opaque_host_closure(target::Module, seen) = false

_collection_eltype_is_closure_free(::Type{T}) where {T} =
    T <: Number || isbitstype(T)

function _already_visited!(target, seen)
    Base.ismutable(target) || return false
    target in seen && return true
    push!(seen, target)
    return false
end

function _has_opaque_host_closure(target::AbstractArray{T}, seen) where {T}
    _collection_eltype_is_closure_free(T) && return false
    _already_visited!(target, seen) && return false
    for index in eachindex(target)
        isassigned(target, index) || continue
        _has_opaque_host_closure(target[index], seen) && return true
    end
    return false
end

function _has_opaque_host_closure(target::AbstractDict{K,V}, seen) where {K,V}
    scan_keys = !_collection_eltype_is_closure_free(K)
    scan_values = !_collection_eltype_is_closure_free(V)
    (scan_keys || scan_values) || return false
    _already_visited!(target, seen) && return false
    for (key, value) in target
        scan_keys && _has_opaque_host_closure(key, seen) && return true
        scan_values && _has_opaque_host_closure(value, seen) && return true
    end
    return false
end

function _has_opaque_host_closure(target::AbstractSet{T}, seen) where {T}
    _collection_eltype_is_closure_free(T) && return false
    _already_visited!(target, seen) && return false
    for value in target
        _has_opaque_host_closure(value, seen) && return true
    end
    return false
end

function _has_opaque_host_closure(
    target::Union{
        Number,
        Random.AbstractRNG,
        AbstractString,
        Symbol,
        Nothing,
        Missing,
        Val,
        AbstractRange,
    },
    seen,
)
    return false
end

function _has_opaque_host_closure(target::Union{Tuple,NamedTuple}, seen)
    return any(value -> _has_opaque_host_closure(value, seen), target)
end

function _has_opaque_host_closure(target, seen)
    isbits(target) && return false
    _already_visited!(target, seen) && return false
    for field in 1:fieldcount(typeof(target))
        _has_opaque_host_closure(getfield(target, field), seen) && return true
    end
    return false
end

_proposal_dimension(proposal) = nothing

function _prepare_target(target::LogTarget, proposal)
    return _PreparedLogTarget(
        target.logdensity,
        _NoTargetContext(),
        target.adtype,
        target.grad,
    )
end

function _prepare_target(target::LogTarget, context, proposal)
    return _PreparedLogTarget(
        target.logdensity,
        context,
        target.adtype,
        target.grad,
    )
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

    return _PreparedLogTarget(
        target,
        _NoTargetContext(),
        ADTypes.NoAutoDiff(),
        nothing,
    )
end

function _prepare_target(target, context, proposal)
    return _PreparedLogTarget(target, context, ADTypes.NoAutoDiff(), nothing)
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
