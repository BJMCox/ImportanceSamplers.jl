abstract type _BoundGradient end

struct _BoundInPlaceGradient{T} <: _BoundGradient
    target::T
end

struct _BoundOutOfPlaceGradient{T} <: _BoundGradient
    target::T
end

struct _BoundLDPGradient{T} <: _BoundGradient
    target::T
end

struct _BoundDIGradient{T,A,C} <: _BoundGradient
    target::T
    backend::A
    preparation::C
end

struct _DIPreparationPool{P}
    preparations::Vector{P}
    thread_slots::Vector{Int}
end

const _GRADIENT_SOURCE_MESSAGE =
    "a gradient requires LogTarget(logdensity; grad=grad), " *
    "LogTarget(logdensity, adtype), or a bare first-order " *
    "LogDensityProblems target"

_gradient_source(::LogTarget{F,A,G}) where {F,A,G} = :explicit
_gradient_source(::LogTarget{F,A,Nothing}) where {F,A<:ADTypes.NoAutoDiff} =
    throw(ArgumentError(_GRADIENT_SOURCE_MESSAGE))
_gradient_source(::LogTarget{F,A,Nothing}) where {F,A} = :ad

function _gradient_source(target::_BoundLogDensityProblemsTarget)
    return _gradient_source(target.target)
end

function _gradient_source(target)
    capabilities = LogDensityProblems.capabilities(target)
    return _gradient_source(target, capabilities)
end

function _gradient_source(::Any, ::LogDensityProblems.LogDensityOrder{K}) where {K}
    K >= 1 && return :logdensityproblems
    throw(ArgumentError(_GRADIENT_SOURCE_MESSAGE))
end

_gradient_source(::Any, ::Any) =
    throw(ArgumentError(_GRADIENT_SOURCE_MESSAGE))

function _prepare_bound_gradient(
    target::_PreparedLogTarget{F,P,A,G},
    sample,
    worker_count,
) where {F,P,A,G}
    _validate_gradient_worker_count(worker_count)
    return _prepare_explicit_gradient(target, sample)
end

function _prepare_bound_gradient(
    target::_PreparedLogTarget{F,P,A,Nothing},
    sample,
    worker_count,
) where {F,P,A<:ADTypes.NoAutoDiff}
    _validate_gradient_worker_count(worker_count)
    throw(ArgumentError(_GRADIENT_SOURCE_MESSAGE))
end

function _prepare_bound_gradient(
    target::_PreparedLogTarget{F,P,A,Nothing},
    sample,
    worker_count,
) where {F,P,A}
    _validate_gradient_worker_count(worker_count)
    _reject_cpu_gradient_on_accelerator(sample)
    preparation = _prepare_di_pool(target, sample, worker_count)
    return _BoundDIGradient(target, target.adtype, preparation)
end

function _prepare_bound_gradient(
    target::_BoundLogDensityProblemsTarget,
    sample,
    worker_count,
)
    _validate_gradient_worker_count(worker_count)
    _gradient_source(target)
    _reject_cpu_gradient_on_accelerator(sample)
    return _BoundLDPGradient(target)
end

function _prepare_bound_gradient(
    ::_BoundDensityInterfaceTarget,
    ::Any,
    worker_count,
)
    _validate_gradient_worker_count(worker_count)
    throw(ArgumentError(_GRADIENT_SOURCE_MESSAGE))
end

function _validate_gradient_worker_count(worker_count)
    worker_count isa Integer && worker_count > 0 || throw(
        ArgumentError("gradient worker count must be a positive integer"),
    )
    return nothing
end

function _prepare_explicit_gradient(
    target::_PreparedLogTarget{F,_NoTargetContext},
    sample,
) where {F}
    destination = similar(sample)
    if applicable(target.gradient, destination, sample)
        return _BoundInPlaceGradient(target)
    elseif applicable(target.gradient, sample)
        _reject_out_of_place_gradient_on_accelerator(sample)
        return _BoundOutOfPlaceGradient(target)
    end
    _reject_cpu_restricted_explicit_gradient(target, sample)
    throw(ArgumentError(_GRADIENT_SOURCE_MESSAGE))
end

function _prepare_explicit_gradient(target::_PreparedLogTarget, sample)
    destination = similar(sample)
    if applicable(target.gradient, destination, sample, target.context)
        return _BoundInPlaceGradient(target)
    elseif applicable(target.gradient, sample, target.context)
        _reject_out_of_place_gradient_on_accelerator(sample)
        return _BoundOutOfPlaceGradient(target)
    end
    _reject_cpu_restricted_explicit_gradient(target, sample)
    throw(ArgumentError(_GRADIENT_SOURCE_MESSAGE))
end

function _reject_cpu_restricted_explicit_gradient(target, sample)
    device = MLDataDevices.get_device(sample)
    device isa MLDataDevices.AbstractCPUDevice && return nothing

    cpu_sample = Array{eltype(sample)}(undef, size(sample))
    cpu_destination = similar(cpu_sample)
    return _reject_cpu_restricted_explicit_gradient(
        target,
        cpu_destination,
        cpu_sample,
        target.context,
        device,
    )
end

function _reject_cpu_restricted_explicit_gradient(
    target,
    destination,
    sample,
    ::_NoTargetContext,
    device,
)
    applicable(target.gradient, destination, sample) && throw(
        SamplerDeviceError(device, :gradient_source_cpu_only),
    )
    applicable(target.gradient, sample) && throw(
        SamplerDeviceError(device, :out_of_place_gradient_cpu_only),
    )
    return nothing
end

function _reject_cpu_restricted_explicit_gradient(
    target,
    destination,
    sample,
    context,
    device,
)
    applicable(target.gradient, destination, sample, context) && throw(
        SamplerDeviceError(device, :gradient_source_cpu_only),
    )
    applicable(target.gradient, sample, context) && throw(
        SamplerDeviceError(device, :out_of_place_gradient_cpu_only),
    )
    return nothing
end

function _reject_cpu_gradient_on_accelerator(sample)
    device = MLDataDevices.get_device(sample)
    device isa MLDataDevices.AbstractCPUDevice && return nothing
    throw(SamplerDeviceError(device, :gradient_source_cpu_only))
end

function _reject_out_of_place_gradient_on_accelerator(sample)
    device = MLDataDevices.get_device(sample)
    device isa MLDataDevices.AbstractCPUDevice && return nothing
    throw(SamplerDeviceError(device, :out_of_place_gradient_cpu_only))
end

function _prepare_di_pool(target, sample, worker_count)
    first_preparation = _prepare_di_gradient(target, target.context, sample)
    preparations = Vector{typeof(first_preparation)}(undef, worker_count)
    preparations[1] = first_preparation
    for worker in 2:worker_count
        preparations[worker] = _prepare_di_gradient(target, target.context, sample)
    end

    thread_slots = zeros(Int, Threads.maxthreadid())
    if worker_count == 1
        fill!(thread_slots, 1)
    else
        default_thread_ids = Threads.threadpooltids(:default)
        worker_count == length(default_thread_ids) || throw(
            ArgumentError(
                "threaded gradient worker count must match the default thread pool",
            ),
        )
        for (slot, thread_id) in enumerate(default_thread_ids)
            thread_slots[thread_id] = slot
        end
    end
    return _DIPreparationPool(preparations, thread_slots)
end

function _prepare_di_gradient(
    target,
    ::_NoTargetContext,
    sample,
)
    return DifferentiationInterface.prepare_gradient(
        target.logdensity,
        target.adtype,
        sample,
    )
end

function _prepare_di_gradient(target, context, sample)
    return DifferentiationInterface.prepare_gradient(
        target.logdensity,
        target.adtype,
        sample,
        DifferentiationInterface.Constant(context),
    )
end

@inline function _gradient!(destination, gradient::_BoundInPlaceGradient, sample)
    return _explicit_inplace_gradient!(
        destination,
        gradient.target,
        gradient.target.context,
        sample,
    )
end

@inline function _explicit_inplace_gradient!(
    destination,
    target,
    ::_NoTargetContext,
    sample,
)
    target.gradient(destination, sample)
    return destination
end

@inline function _explicit_inplace_gradient!(
    destination,
    target,
    context,
    sample,
)
    target.gradient(destination, sample, context)
    return destination
end

@inline function _gradient!(destination, gradient::_BoundOutOfPlaceGradient, sample)
    return _explicit_out_of_place_gradient!(
        destination,
        gradient.target,
        gradient.target.context,
        sample,
    )
end

@inline function _explicit_out_of_place_gradient!(
    destination,
    target,
    ::_NoTargetContext,
    sample,
)
    copyto!(destination, target.gradient(sample))
    return destination
end

@inline function _explicit_out_of_place_gradient!(
    destination,
    target,
    context,
    sample,
)
    copyto!(destination, target.gradient(sample, context))
    return destination
end

@inline function _gradient!(destination, gradient::_BoundLDPGradient, sample)
    _, result = LogDensityProblems.logdensity_and_gradient(
        gradient.target.target,
        sample,
    )
    copyto!(destination, result)
    return destination
end

@inline function _gradient!(destination, gradient::_BoundDIGradient, sample)
    preparation = _di_preparation(gradient.preparation)
    return _di_gradient!(
        destination,
        gradient.target,
        gradient.backend,
        preparation,
        gradient.target.context,
        sample,
    )
end

@inline function _di_preparation(pool::_DIPreparationPool)
    thread_id = Threads.threadid()
    thread_id <= length(pool.thread_slots) || throw(
        ArgumentError("gradient evaluation must run on a prepared worker"),
    )
    slot = @inbounds pool.thread_slots[thread_id]
    slot > 0 || throw(
        ArgumentError("gradient evaluation must run on a default-pool worker"),
    )
    return @inbounds pool.preparations[slot]
end

@inline function _di_gradient!(
    destination,
    target,
    backend,
    preparation,
    ::_NoTargetContext,
    sample,
)
    DifferentiationInterface.gradient!(
        target.logdensity,
        destination,
        preparation,
        backend,
        sample,
    )
    return destination
end

@inline function _di_gradient!(
    destination,
    target,
    backend,
    preparation,
    context,
    sample,
)
    DifferentiationInterface.gradient!(
        target.logdensity,
        destination,
        preparation,
        backend,
        sample,
        DifferentiationInterface.Constant(context),
    )
    return destination
end
