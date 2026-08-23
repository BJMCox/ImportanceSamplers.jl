"""
    AbstractImportanceSampler

Abstract supertype for importance-sampling algorithm configurations.

Algorithms are immutable descriptions. Execution state, RNG ownership, and a
bound target belong to the object returned by [`prepare_sampler`](@ref).
"""
abstract type AbstractImportanceSampler end

"""
    ImportanceSampling(proposal; nsamples)

Configure plain, single-proposal importance sampling.

`proposal` must implement `rand(rng, proposal)` and
`DensityInterface.logdensityof(proposal, sample)` for the same normalized
measure. `nsamples` must be a positive `Int` and is the exact number of samples
returned by every run.
"""
struct ImportanceSampling{P} <: AbstractImportanceSampler
    proposal::P
    nsamples::Int
end

function ImportanceSampling(proposal; nsamples)
    nsamples isa Int && nsamples > 0 || throw(ArgumentError("nsamples must be a positive Int"))
    return ImportanceSampling(proposal, nsamples)
end

"""
    SamplerBusyError

Exception thrown when a prepared sampler is entered while it is already
running. Prepared samplers are mutable, single-owner, and non-reentrant.
"""
struct SamplerBusyError <: Exception end

function Base.showerror(io::IO, ::SamplerBusyError)
    print(io, "the prepared sampler is already running")
end

"""
    SamplerAlreadyExecutedError

Exception thrown when device transfer is requested after a prepared sampler's
first execution has begun. Device placement is fixed for the lifetime of an
executed prepared sampler.
"""
struct SamplerAlreadyExecutedError <: Exception end

function Base.showerror(io::IO, ::SamplerAlreadyExecutedError)
    print(io, "a prepared sampler cannot be transferred after execution has begun")
end

"""
    SamplerDeviceError

Exception thrown when a requested device cannot execute a prepared sampler.
The `reason` field identifies the rejected capability.
"""
struct SamplerDeviceError{D} <: Exception
    device::D
    reason::Symbol
end

function Base.showerror(io::IO, error::SamplerDeviceError)
    message = if error.reason === :serial_accelerator
        "threaded=false has no accelerator execution contract"
    elseif error.reason === :opaque_host_closure
        "an opaque target closure reachable through target state cannot be " *
        "transferred without inspecting or reconstructing its captures; " *
        "pass numerical state through p"
    elseif error.reason === :backend_unavailable
        "the requested device backend is unavailable or nonfunctional"
    elseif error.reason === :accelerator_rng_unavailable
        "accelerator random-buffer support is not available"
    elseif error.reason === :product_proposal_cpu_only
        "ProductProposal is CPU-only"
    elseif error.reason === :rng_not_cloneable
        "the prepared RNG does not provide an independent copy"
    else
        "the requested device is unsupported"
    end
    print(
        io,
        "prepared-sampler device transfer failed for ",
        typeof(error.device),
        ": ",
        message,
    )
end

"""
    SamplerExecutionError

Exception wrapping a failure from one logical sample during proposal drawing,
target evaluation, or proposal-density evaluation.

The `phase` and `sample_index` fields locate the failure. `captured` preserves
the original exception and backtrace. A failed run never returns a partial
result.
"""
struct SamplerExecutionError <: Exception
    phase::Symbol
    sample_index::Int
    captured::CapturedException
end

function Base.showerror(io::IO, error::SamplerExecutionError)
    print(
        io,
        "importance-sampler execution failed during ",
        error.phase,
        " at logical sample ",
        error.sample_index,
        ": ",
    )
    showerror(io, error.captured)
end

struct _ContextFreePreparedTarget{T}
    target::T
end

struct _ContextualPreparedTarget{T,P}
    target::T
    context::P
end

mutable struct _PreparedImportanceSampler{R,T,A,D}
    rng::R
    target::T
    algorithm::A
    device::D
    threaded::Bool
    running::Bool
    executed::Bool
end

"""
    prepare_sampler(rng, logtarget, algorithm; threaded=true)
    prepare_sampler(rng, logtarget, p, algorithm; threaded=true)

Bind a target, optional context `p`, algorithm, CPU execution policy, and RNG
into a reusable prepared sampler.

Known target interfaces and known LogDensityProblems dimensions are resolved
during preparation. Callable applicability and scalar log-density types are
validated on the first retained execution, when the logical sample type is
available. The target must return a `Float32` or `Float64` log density.

The prepared sampler retains and advances the supplied RNG; treat that RNG as
owned by the sampler after preparation. A prepared sampler is mutable and
non-reentrant. Repeated calls to [`importance_sample!`](@ref) create separate,
noncumulative results that own their arrays.

Preparation always produces a CPU sampler. Apply an explicit
`MLDataDevices.AbstractDevice` value to the complete prepared sampler before
its first execution to request transfer. Set `threaded=false` for serial CPU
evaluation. With `threaded=true`, a one-thread Julia process falls back to the
serial path.
"""
function prepare_sampler(
    rng::Random.AbstractRNG,
    logtarget,
    algorithm::ImportanceSampling;
    threaded=true,
)
    target = _ContextFreePreparedTarget(logtarget)
    return _prepare_importance_sampler(rng, target, algorithm, threaded)
end

function prepare_sampler(
    rng::Random.AbstractRNG,
    logtarget,
    context,
    algorithm::ImportanceSampling;
    threaded=true,
)
    target = _ContextualPreparedTarget(logtarget, context)
    return _prepare_importance_sampler(rng, target, algorithm, threaded)
end

function _prepare_importance_sampler(rng, target, algorithm, threaded)
    threaded isa Bool || throw(ArgumentError("threaded must be Bool"))
    prepared_target = _resolve_prepared_target(target, algorithm.proposal)
    return _PreparedImportanceSampler(
        rng,
        prepared_target,
        algorithm,
        MLDataDevices.CPUDevice(),
        threaded,
        false,
        false,
    )
end

function _copy_to_device(device, value)
    return device(deepcopy(value))
end

function _copy_algorithm(device, algorithm::ImportanceSampling)
    proposal = _copy_to_device(device, algorithm.proposal)
    return ImportanceSampling(proposal, algorithm.nsamples)
end

function _clone_rng(device, rng::Random.AbstractRNG)
    cloned = try
        copy(rng)
    catch
        throw(SamplerDeviceError(device, :rng_not_cloneable))
    end
    cloned isa Random.AbstractRNG && cloned !== rng || throw(
        SamplerDeviceError(device, :rng_not_cloneable),
    )
    return cloned
end

function _transfer_prepared_sampler(
    device::MLDataDevices.AbstractCPUDevice,
    sampler::_PreparedImportanceSampler,
)
    sampler.executed && throw(SamplerAlreadyExecutedError())
    MLDataDevices.functional(device) || throw(
        SamplerDeviceError(device, :backend_unavailable),
    )
    _target_transfer_rewrites_opaque_closure(sampler.target) && throw(
        SamplerDeviceError(device, :opaque_host_closure),
    )
    return _PreparedImportanceSampler(
        _clone_rng(device, sampler.rng),
        _transfer_prepared_target(device, sampler.target),
        _copy_algorithm(device, sampler.algorithm),
        device,
        sampler.threaded,
        false,
        false,
    )
end

function _transfer_prepared_sampler(
    device::MLDataDevices.AbstractAcceleratorDevice,
    sampler::_PreparedImportanceSampler,
)
    sampler.executed && throw(SamplerAlreadyExecutedError())
    sampler.threaded || throw(SamplerDeviceError(device, :serial_accelerator))
    _target_has_opaque_host_closure(sampler.target) && throw(
        SamplerDeviceError(device, :opaque_host_closure),
    )
    proposal_limit = _accelerator_proposal_limit(sampler.algorithm.proposal)
    isnothing(proposal_limit) || throw(SamplerDeviceError(device, proposal_limit))
    MLDataDevices.functional(device) || throw(
        SamplerDeviceError(device, :backend_unavailable),
    )
    throw(SamplerDeviceError(device, :accelerator_rng_unavailable))
end

function _transfer_prepared_sampler(
    device::MLDataDevices.AbstractDevice,
    sampler::_PreparedImportanceSampler,
)
    sampler.executed && throw(SamplerAlreadyExecutedError())
    throw(SamplerDeviceError(device, :unsupported_device))
end

function (device::MLDataDevices.AbstractDevice)(sampler::_PreparedImportanceSampler)
    return _transfer_prepared_sampler(device, sampler)
end

"""
    importance_sample(rng, logtarget, algorithm; threaded=true)
    importance_sample(rng, logtarget, p, algorithm; threaded=true)

Run one complete plain-importance-sampling estimator.

This is the one-shot form of [`prepare_sampler`](@ref) followed by
[`importance_sample!`](@ref). `logtarget` returns a log density, not a linear
density. The contextual overload calls `logtarget(sample, p)`.

The result stores canonical raw log weights
`logtarget(sample) - logdensityof(proposal, sample)`. Use
[`normalized_weights`](@ref) for weights that sum to one and
[`lognormalizer`](@ref) for the complete estimator's log normalizer.
"""
function importance_sample(
    rng::Random.AbstractRNG,
    logtarget,
    algorithm::ImportanceSampling;
    threaded=true,
)
    sampler = prepare_sampler(
        rng,
        logtarget,
        algorithm;
        threaded=threaded,
    )
    return importance_sample!(sampler)
end

function importance_sample(
    rng::Random.AbstractRNG,
    logtarget,
    context,
    algorithm::ImportanceSampling;
    threaded=true,
)
    sampler = prepare_sampler(
        rng,
        logtarget,
        context,
        algorithm;
        threaded=threaded,
    )
    return importance_sample!(sampler)
end

"""
    importance_sample!(sampler)

Execute one complete estimator run using a prepared sampler.

The bang records that the sampler's RNG stream and running state are mutated.
It also permanently fixes device placement when execution begins, whether the
run succeeds or fails.
Each returned [`WeightedSamples`](@ref) owns its storage and cannot be changed
by later runs. Concurrent calls on the same sampler throw
[`SamplerBusyError`](@ref); use separate prepared samplers and RNGs for
concurrent top-level runs.
"""
function importance_sample!(sampler::_PreparedImportanceSampler)
    sampler.running && throw(SamplerBusyError())
    sampler.executed = true
    sampler.running = true
    try
        threaded = sampler.threaded && Threads.nthreads(:default) > 1
        return _importance_sample_cpu!(sampler, threaded)
    finally
        sampler.running = false
    end
end

function _importance_sample_cpu!(sampler, threaded)
    execution = _sampling_execution(sampler.algorithm.proposal, threaded)
    samples, logweights = _importance_sample!(sampler, execution)
    diagnostics = (
        method=:importance_sampling,
        execution=_execution_name(execution),
        threaded=sampler.threaded,
        nsamples=sampler.algorithm.nsamples,
        failures=0,
        transfers=(count=0, bytes=0),
    )
    return _adopt_weighted_samples(samples, logweights; diagnostics=diagnostics)
end

function _importance_sample_generic_cpu!(sampler, threaded)
    samples = _draw_prepared_batch(sampler)
    target = _bind_prepared_target(sampler.target, samples)
    log_type = _resolve_logweight_type(
        target,
        sampler.algorithm.proposal,
        samples,
    )
    logweights = if threaded
        _evaluate_logweights_threaded(
            log_type,
            target,
            sampler.algorithm.proposal,
            samples,
        )
    else
        _evaluate_logweights(
            log_type,
            target,
            sampler.algorithm.proposal,
            samples,
        )
    end
    return samples, logweights
end

function _evaluate_logweights_threaded(
    ::Type{T},
    target,
    proposal,
    samples,
) where {T<:Union{Float32,Float64}}
    nsamples = _sample_count(samples)
    target_logs = Vector{T}(undef, nsamples)
    proposal_logs = Vector{T}(undef, nsamples)
    _evaluate_logs_threaded!(target_logs, target, samples, Val(:target))
    _evaluate_logs_threaded!(
        proposal_logs,
        proposal,
        samples,
        Val(:proposal_logdensity),
    )
    _construct_logweights!(proposal_logs, target_logs)
    return proposal_logs
end

function _evaluate_logs_threaded!(
    logs::Vector{T},
    evaluator,
    samples,
    phase::Val,
) where {T}
    ntasks = min(Threads.nthreads(:default), length(logs))
    chunk_size = cld(length(logs), ntasks)
    tasks = Task[]
    for first_index in 1:chunk_size:length(logs)
        last_index = min(first_index + chunk_size - 1, length(logs))
        let indices = first_index:last_index
            task = Threads.@spawn _evaluate_log_chunk!(
                logs,
                evaluator,
                samples,
                indices,
                phase,
            )
            push!(tasks, task)
        end
    end

    failure = nothing
    for task in tasks
        task_failure = fetch(task)::Union{Nothing,SamplerExecutionError}
        failure === nothing && task_failure !== nothing && (failure = task_failure)
    end
    failure === nothing || throw(failure)
    return logs
end

function _evaluate_log_chunk!(
    logs::Vector{T},
    evaluator,
    samples,
    indices,
    phase::Val{P},
) where {T,P}
    sample_index = first(indices)
    try
        for index in indices
            sample_index = index
            _evaluate_log_sample!(
                logs,
                evaluator,
                samples,
                sample_index,
                phase,
            )
        end
    catch error
        error isa SamplerExecutionError && return error
        return SamplerExecutionError(
            P,
            sample_index,
            CapturedException(error, catch_backtrace()),
        )
    end
    return nothing
end

function _phase_logdensity(target, sample, sample_index, ::Val{:target})
    return _target_logdensity(target, sample, sample_index)
end

function _phase_logdensity(
    proposal,
    sample,
    sample_index,
    ::Val{:proposal_logdensity},
)
    return _proposal_logdensity(proposal, sample, sample_index)
end

function _evaluate_log_sample!(
    logs::Vector{T},
    evaluator,
    samples,
    sample_index,
    phase::Val{P},
) where {T,P}
    sample = _sample_at(samples, sample_index)
    logdensity = _phase_logdensity(evaluator, sample, sample_index, phase)
    logs[sample_index] = _lossless_log_convert(T, logdensity, P, sample_index)
    return logs
end

function _draw_prepared_batch(sampler)
    proposal = sampler.algorithm.proposal
    nsamples = sampler.algorithm.nsamples
    first_sample = _capture_sampler_failure(:proposal_draw, 1) do
        rand(sampler.rng, proposal)
    end
    batch = _capture_sampler_failure(:proposal_draw, 1) do
        _allocate_batch(first_sample, nsamples)
    end
    _capture_sampler_failure(:proposal_draw, 1) do
        _store_sample!(batch, first_sample, 1)
    end

    for sample_index in 2:nsamples
        _capture_sampler_failure(:proposal_draw, sample_index) do
            sample = rand(sampler.rng, proposal)
            _store_sample!(batch, sample, sample_index)
        end
    end
    return batch
end

function _resolve_prepared_target(target::_ContextFreePreparedTarget, proposal)
    return _prepare_target(target.target, proposal)
end

function _resolve_prepared_target(target::_ContextualPreparedTarget, proposal)
    return _prepare_target(target.target, target.context, proposal)
end

function _bind_resolved_target(target::_ContextFreePreparedTarget, sample)
    return _bind_context_free_callable(target.target, sample)
end

function _bind_resolved_target(target::_ContextualPreparedTarget, sample)
    return _bind_contextual_callable(target.target, target.context, sample)
end

function _bind_prepared_target(target, samples)
    return _capture_sampler_failure(:target, 1) do
        _bind_resolved_target(target, _sample_at(samples, 1))
    end
end

function _resolve_logweight_type(target, proposal, samples)
    return _resolve_logweight_type(target, proposal, typeof(_sample_at(samples, 1)))
end

function _resolve_logweight_type(target, proposal, sample_type::Type)
    target_type = _capture_sampler_failure(:target, 1) do
        inferred = Base.promote_op(target, sample_type)
        _canonical_inferred_log_type(inferred, "target")
    end
    proposal_type = _capture_sampler_failure(:proposal_logdensity, 1) do
        inferred = Base.promote_op(
            DensityInterface.logdensityof,
            typeof(proposal),
            sample_type,
        )
        _canonical_inferred_log_type(inferred, "proposal")
    end
    return target_type === Float64 || proposal_type === Float64 ? Float64 : Float32
end

function _canonical_inferred_log_type(inferred, label)
    inferred !== Union{} && inferred <: Union{Float32,Float64} || throw(
        ArgumentError(
            "$label log-density return type must be provably limited to Float32 and Float64; inferred $inferred",
        ),
    )
    return Float64 <: inferred ? Float64 : Float32
end

function _evaluate_logweights(
    ::Type{T},
    target,
    proposal,
    samples,
) where {T<:Union{Float32,Float64}}
    nsamples = _sample_count(samples)
    target_logs = Vector{T}(undef, nsamples)
    proposal_logs = Vector{T}(undef, nsamples)
    _evaluate_target_logs!(target_logs, target, samples)
    _evaluate_proposal_logs!(proposal_logs, proposal, samples)
    _construct_logweights!(proposal_logs, target_logs)
    return proposal_logs
end

function _evaluate_target_logs!(target_logs::Vector{T}, target, samples) where {T}
    phase = Val(:target)
    for sample_index in eachindex(target_logs)
        _evaluate_log_sample!(target_logs, target, samples, sample_index, phase)
    end
    return target_logs
end

function _evaluate_proposal_logs!(proposal_logs::Vector{T}, proposal, samples) where {T}
    phase = Val(:proposal_logdensity)
    for sample_index in eachindex(proposal_logs)
        _evaluate_log_sample!(
            proposal_logs,
            proposal,
            samples,
            sample_index,
            phase,
        )
    end
    return proposal_logs
end

function _construct_logweights!(logweights::Vector{T}, target_logs::Vector{T}) where {T}
    axes(logweights) == axes(target_logs) || throw(
        DimensionMismatch("target and proposal log arrays must be aligned"),
    )
    for sample_index in eachindex(logweights, target_logs)
        logweights[sample_index] = target_logs[sample_index] - logweights[sample_index]
    end
    return logweights
end

function _target_logdensity(target, sample, sample_index)
    return _capture_sampler_failure(:target, sample_index) do
        value = target(sample)
        _validate_target_logdensity(value)
        value
    end
end

function _proposal_logdensity(proposal, sample, sample_index)
    return _capture_sampler_failure(:proposal_logdensity, sample_index) do
        value = DensityInterface.logdensityof(proposal, sample)
        _validate_proposal_logdensity(value)
        value
    end
end

function _validate_target_logdensity(value)
    (typeof(value) === Float32 || typeof(value) === Float64) || throw(
        ArgumentError("target log density must be Float32 or Float64"),
    )
    (isnan(value) || value == Inf) && throw(
        DomainError(value, "target log density may not be NaN or +Inf"),
    )
    return nothing
end

function _validate_proposal_logdensity(value)
    (typeof(value) === Float32 || typeof(value) === Float64) || throw(
        ArgumentError("proposal log density must be Float32 or Float64"),
    )
    (isnan(value) || value == -Inf) && throw(
        DomainError(
            value,
            "proposal log density at a generated sample may not be NaN or -Inf",
        ),
    )
    return nothing
end

function _lossless_log_convert(::Type{T}, value, phase, sample_index) where {T}
    return _capture_sampler_failure(phase, sample_index) do
        promote_type(T, typeof(value)) === T || throw(
            ArgumentError(
                "log-density type $(typeof(value)) would narrow when stored as $T",
            ),
        )
        convert(T, value)
    end
end

function _capture_sampler_failure(f, phase::Symbol, sample_index::Int)
    try
        return f()
    catch error
        throw(
            SamplerExecutionError(
                phase,
                sample_index,
                CapturedException(error, catch_backtrace()),
            ),
        )
    end
end
