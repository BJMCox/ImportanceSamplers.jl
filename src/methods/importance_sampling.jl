struct _SingleProposalScheme end
struct _SingleProposalMethodState end

"""
    AbstractImportanceSampler

Abstract supertype for importance-sampling algorithm configurations.

Algorithms are immutable descriptions. Execution state, RNG ownership, and a
bound target belong to the object returned by [`prepare_sampler`](@ref).
"""
abstract type AbstractImportanceSampler end

mutable struct _ValidatedImportanceSamplingToken end
const _VALIDATED_IMPORTANCE_SAMPLING_TOKEN = _ValidatedImportanceSamplingToken()

"""
    ImportanceSampling(proposal; nsamples)
    ImportanceSampling(bank::ProposalBank; nsamples, mis_scheme=StratifiedMixture())

Configure plain or static multiple importance sampling.

`proposal` must implement `rand(rng, proposal)` and
`DensityInterface.logdensityof(proposal, sample)` for the same normalized
measure. `nsamples` must be a positive `Int` and is the exact number of samples
returned by every run. Generic proposals execute on CPU; the native Gaussian
and transform subset also supports prepared CUDA execution.

When the first argument is a [`ProposalBank`](@ref), `mis_scheme` selects one of
the four complete assignment/denominator contracts described by
[`AbstractMISScheme`](@ref); it defaults to [`StratifiedMixture`](@ref). The
result stores canonical raw log weights and stable one-based generating
proposal IDs in `result.provenance.proposal_id`. Bank capabilities are validated
during preparation, before the supplied RNG is consumed.
"""
struct ImportanceSampling{P,S} <: AbstractImportanceSampler
    proposal::P
    nsamples::Int
    mis_scheme::S

    function ImportanceSampling(
        proposal::P,
        nsamples::Int,
        mis_scheme::S,
        token::_ValidatedImportanceSamplingToken,
    ) where {P,S}
        token === _VALIDATED_IMPORTANCE_SAMPLING_TOKEN || throw(
            ArgumentError("invalid internal algorithm-construction token"),
        )
        return new{P,S}(proposal, nsamples, mis_scheme)
    end
end

function ImportanceSampling(proposal; nsamples)
    nsamples isa Int && nsamples > 0 || throw(ArgumentError("nsamples must be a positive Int"))
    return ImportanceSampling(
        proposal,
        nsamples,
        _SingleProposalScheme(),
        _VALIDATED_IMPORTANCE_SAMPLING_TOKEN,
    )
end

function ImportanceSampling(
    bank::ProposalBank;
    nsamples,
    mis_scheme::AbstractMISScheme=StratifiedMixture(),
)
    nsamples isa Int && nsamples > 0 || throw(ArgumentError("nsamples must be a positive Int"))
    return ImportanceSampling(
        bank,
        nsamples,
        mis_scheme,
        _VALIDATED_IMPORTANCE_SAMPLING_TOKEN,
    )
end

_algorithm_proposal(algorithm::ImportanceSampling) = algorithm.proposal
_algorithm_sample_budget(algorithm::ImportanceSampling) = algorithm.nsamples

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
The `reason` field identifies the rejected capability, such as an unavailable
backend, unspecified scalar policy, opaque target closure, unsupported proposal,
or missing device RNG.
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
    elseif error.reason === :device_residency_mismatch
        "prepared state is not resident on the requested device"
    elseif error.reason === :scalar_policy_unspecified
        "the accelerator scalar policy is unspecified; construct a preserving " *
        "device whose eltype policy is Nothing, Float32, or Float64"
    elseif error.reason === :generic_proposal_cpu_only
        "generic proposals are CPU-only"
    elseif error.reason === :product_proposal_cpu_only
        "ProductProposal is CPU-only"
    elseif error.reason === :factor_proposal_cpu_only
        "factor Gaussian proposal banks are CPU-only"
    elseif error.reason === :transformed_proposal_cpu_only
        "transformed proposal banks are CPU-only"
    elseif error.reason === :kernel_argument_unsupported
        "the target or context does not have a supported accelerator kernel " *
        "argument representation"
    elseif error.reason === :prepared_migration_unsupported
        "accelerator-resident prepared samplers cannot be transferred"
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
target evaluation, proposal-density evaluation, or log-weight construction.

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

mutable struct _PreparedImportanceSampler{R,B,T,A,M,D}
    rng::R
    random_buffers::B
    target::T
    algorithm::A
    method_state::M
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
its first execution to request transfer. Transfer recursively moves proposal,
device-adaptable callable state, and every numerical array in `p`; opaque
closure captures cannot be moved reliably and are rejected for accelerator
execution. An accelerator whose public `eltype(device)` is `Missing` is rejected
as `:scalar_policy_unspecified`; construct a preserving device with
an explicit non-`Missing` scalar policy. The accelerator guide shows CUDA
construction for the current or a selected physical device without auto-selection.
Native CUDA execution requires `threaded=true` and keeps returned arrays on the
device. Set `threaded=false` for serial CPU evaluation. On CPU,
`threaded=true` falls back to serial execution when Julia has one default
thread; accelerator launch policy does not depend on host thread count.
"""
function prepare_sampler(
    rng::Random.AbstractRNG,
    logtarget,
    algorithm::AbstractImportanceSampler;
    threaded=true,
)
    target = _ContextFreePreparedTarget(logtarget)
    return _prepare_importance_sampler(rng, target, algorithm, threaded)
end

function prepare_sampler(
    rng::Random.AbstractRNG,
    logtarget,
    context,
    algorithm::AbstractImportanceSampler;
    threaded=true,
)
    target = _ContextualPreparedTarget(logtarget, context)
    return _prepare_importance_sampler(rng, target, algorithm, threaded)
end

function _prepare_importance_sampler(rng, target, algorithm, threaded)
    threaded isa Bool || throw(ArgumentError("threaded must be Bool"))
    proposal = _algorithm_proposal(algorithm)
    sample_budget = _algorithm_sample_budget(algorithm)
    prepared_target = _resolve_prepared_target(target, proposal)
    method_state = _prepare_method_state(algorithm, prepared_target)
    device = MLDataDevices.CPUDevice()
    random_buffers = _allocate_random_buffers(
        device,
        proposal,
        method_state,
        sample_budget,
    )
    return _PreparedImportanceSampler(
        rng,
        random_buffers,
        prepared_target,
        algorithm,
        method_state,
        device,
        threaded,
        false,
        false,
    )
end

_prepare_method_state(
    ::ImportanceSampling{P,_SingleProposalScheme},
) where {P} = _SingleProposalMethodState()

_prepare_method_state(algorithm, target) = _prepare_method_state(algorithm)

function _copy_to_device(device, value)
    return device(deepcopy(value))
end

function _copy_algorithm(device, algorithm::ImportanceSampling)
    proposal = _copy_to_device(device, algorithm.proposal)
    mis_scheme = _copy_to_device(device, algorithm.mis_scheme)
    return ImportanceSampling(
        proposal,
        algorithm.nsamples,
        mis_scheme,
        _VALIDATED_IMPORTANCE_SAMPLING_TOKEN,
    )
end

_copy_accelerator_algorithm(device, algorithm, method_state) =
    _copy_algorithm(device, algorithm)

_prepare_transferred_method_state(device, algorithm, method_state) =
    _prepare_method_state(algorithm)

_accelerator_method_state_limit(method_state) = nothing

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

_backend_functional(device) = MLDataDevices.functional(device)
_with_backend_device(f, device) = f()
_backend_state_resident(device, state) = true

function _validate_backend_state(device, state)
    _backend_state_resident(device, state) || throw(
        SamplerDeviceError(device, :device_residency_mismatch),
    )
    return nothing
end

_prepared_backend_state(sampler, method_state) = (
    sampler.algorithm,
    method_state,
    sampler.target,
    sampler.random_buffers,
    sampler.rng,
)

_transferred_backend_state(algorithm, method_state, target, random_buffers) =
    (algorithm, method_state, target, random_buffers)

function _preflight_accelerator_method(
    device,
    target,
    algorithm,
    ::_SingleProposalMethodState,
    random_buffers,
)
    return _preflight_native_kernel_target(
        device,
        target,
        algorithm.proposal,
        random_buffers,
    )
end

function _transfer_prepared_sampler(
    device::MLDataDevices.AbstractCPUDevice,
    sampler::_PreparedImportanceSampler,
)
    sampler.device isa MLDataDevices.AbstractAcceleratorDevice && throw(
        SamplerDeviceError(device, :prepared_migration_unsupported),
    )
    sampler.executed && throw(SamplerAlreadyExecutedError())
    _backend_functional(device) || throw(
        SamplerDeviceError(device, :backend_unavailable),
    )
    _target_transfer_rewrites_opaque_closure(sampler.target) && throw(
        SamplerDeviceError(device, :opaque_host_closure),
    )
    algorithm = _copy_algorithm(device, sampler.algorithm)
    proposal = _algorithm_proposal(algorithm)
    transferred_target = _transfer_prepared_target(device, sampler.target)
    method_state = _prepare_method_state(algorithm, transferred_target)
    random_buffers = _allocate_random_buffers(
        device,
        proposal,
        method_state,
        _algorithm_sample_budget(algorithm),
    )
    return _PreparedImportanceSampler(
        _clone_rng(device, sampler.rng),
        random_buffers,
        transferred_target,
        algorithm,
        method_state,
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
    sampler.device isa MLDataDevices.AbstractAcceleratorDevice && throw(
        SamplerDeviceError(device, :prepared_migration_unsupported),
    )
    if applicable(Base.eltype, device) && Base.eltype(device) === Missing
        throw(SamplerDeviceError(device, :scalar_policy_unspecified))
    end
    sampler.executed && throw(SamplerAlreadyExecutedError())
    sampler.threaded || throw(SamplerDeviceError(device, :serial_accelerator))
    _target_has_opaque_host_closure(sampler.target, device) && throw(
        SamplerDeviceError(device, :opaque_host_closure),
    )
    _backend_functional(device) || throw(
        SamplerDeviceError(device, :backend_unavailable),
    )
    proposal_limit = _accelerator_proposal_limit(
        _algorithm_proposal(sampler.algorithm),
    )
    isnothing(proposal_limit) || throw(SamplerDeviceError(device, proposal_limit))
    method_state_limit = _accelerator_method_state_limit(sampler.method_state)
    isnothing(method_state_limit) || throw(
        SamplerDeviceError(device, method_state_limit),
    )
    return _with_backend_device(device) do
        algorithm = _copy_accelerator_algorithm(
            device,
            sampler.algorithm,
            sampler.method_state,
        )
        target = _transfer_prepared_target(device, sampler.target)
        method_state = _prepare_transferred_method_state(
            device,
            algorithm,
            sampler.method_state,
        )
        random_buffers = _allocate_random_buffers(
            device,
            _algorithm_proposal(algorithm),
            method_state,
            _algorithm_sample_budget(algorithm),
        )
        random_buffers isa Union{
            _RandomBuffers,
            _PackedStaticMISRandomBuffers,
            _DMPMCRandomBuffers,
        } || throw(
            SamplerDeviceError(device, :accelerator_rng_unavailable),
        )
        _validate_backend_state(
            device,
            _transferred_backend_state(
                algorithm,
                method_state,
                target,
                random_buffers,
            ),
        )
        _preflight_accelerator_method(
            device,
            target,
            algorithm,
            method_state,
            random_buffers,
        )
        source_rng = _clone_rng(device, sampler.rng)
        seed = try
            Random.rand(source_rng, UInt64)
        catch
            throw(SamplerDeviceError(device, :accelerator_rng_unavailable))
        end
        backend_rng = try
            _owned_backend_rng(device, seed)
        catch
            throw(SamplerDeviceError(device, :accelerator_rng_unavailable))
        end
        destination = _PreparedImportanceSampler(
            backend_rng,
            random_buffers,
            target,
            algorithm,
            method_state,
            device,
            sampler.threaded,
            false,
            false,
        )
        sampler.rng = source_rng
        return destination
    end
end

function _transfer_prepared_sampler(
    device::MLDataDevices.AbstractDevice,
    sampler::_PreparedImportanceSampler,
)
    sampler.device isa MLDataDevices.AbstractAcceleratorDevice && throw(
        SamplerDeviceError(device, :prepared_migration_unsupported),
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

Run one complete importance-sampling estimator.

This is the one-shot form of [`prepare_sampler`](@ref) followed by
[`importance_sample!`](@ref). `logtarget` returns a log density, not a linear
density. The contextual overload calls `logtarget(sample, p)`. The one-shot
form executes on CPU; apply a device to a prepared sampler for CUDA execution.

The result stores the algorithm's canonical raw log weights. Use
[`normalized_weights`](@ref) for weights that sum to one and [`lognormalizer`](@ref)
for the complete estimator's log normalizer.
"""
function importance_sample(
    rng::Random.AbstractRNG,
    logtarget,
    algorithm::AbstractImportanceSampler;
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
    algorithm::AbstractImportanceSampler;
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
by later runs. Concurrent use of one sampler is unsupported; use separate
prepared samplers and RNGs. An entry that observes the sampler already busy,
including recursive re-entry, throws [`SamplerBusyError`](@ref); this check does
not synchronize simultaneous callers. CUDA results remain device-resident
until an explicit transfer such as `result |> MLDataDevices.cpu_device()`.
"""
function importance_sample!(sampler::_PreparedImportanceSampler)
    sampler.running && throw(SamplerBusyError())
    sampler.executed = true
    sampler.running = true
    try
        return _with_backend_device(sampler.device) do
            _validate_backend_state(
                sampler.device,
                _prepared_backend_state(sampler, sampler.method_state),
            )
            _reset_native_failure_scratch!(
                _native_failure_scratch(sampler.random_buffers),
            )
            threaded =
                sampler.device isa MLDataDevices.AbstractAcceleratorDevice ||
                sampler.threaded && Threads.nthreads(:default) > 1
            return _importance_sample_cpu!(sampler, threaded)
        end
    finally
        sampler.running = false
    end
end

function _importance_sample_cpu!(sampler, threaded)
    return _importance_sample_cpu!(sampler, sampler.method_state, threaded)
end

function _importance_sample_cpu!(sampler, ::_SingleProposalMethodState, threaded)
    execution = _sampling_execution(sampler.algorithm.proposal, threaded)
    samples, logweights, transfers = _importance_sample!(sampler, execution)
    diagnostics = (
        method=:importance_sampling,
        execution=_execution_name(execution),
        threaded=sampler.threaded,
        nsamples=sampler.algorithm.nsamples,
        failures=0,
        transfers=transfers,
    )
    if execution isa _KernelExecution
        return _adopt_validated_weighted_samples(
            samples,
            logweights;
            diagnostics=diagnostics,
        )
    end
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
        logweight, reason = _subtract_logweight(
            target_logs[sample_index],
            logweights[sample_index],
        )
        iszero(reason) || _throw_invalid_logweight(logweight, sample_index)
        logweights[sample_index] = logweight
    end
    return logweights
end

@noinline function _throw_invalid_logweight(logweight, sample_index)
    return _capture_sampler_failure(:logweight, sample_index) do
        throw(DomainError(logweight, "derived log weight may not be NaN or +Inf"))
    end
end

@inline function _subtract_logweight(target_log, proposal_log)
    logweight = target_log - proposal_log
    reason = (isnan(logweight) || logweight == Inf) ?
             _NATIVE_LOGWEIGHT_INVALID : UInt16(0)
    return logweight, reason
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
