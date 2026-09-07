_gaussian_adaptation_count(algorithm, state, round) =
    length(_gaussian_summary_indices(algorithm, state, round))

struct _FixedMISAssignments
    slot::Int
    length::Int
end

const _GAUSSIAN_COVARIANCE_INVALID = UInt16(0x2000)

struct _GaussianStageError{E} <: Exception
    phase::Symbol
    cause::E
end

struct _GaussianFitFailure{E}
    cause::E
end

@noinline function _throw_gaussian_stage(phase, cause)
    cause isa _GaussianStageError && throw(cause)
    throw(_GaussianStageError(phase, cause))
end

Adapt.@adapt_structure _GaussianScalarHistory
Adapt.@adapt_structure _GaussianFactorHistory

Base.@propagate_inbounds Base.getindex(assignments::_FixedMISAssignments, index) =
    assignments.slot
Base.length(assignments::_FixedMISAssignments) = assignments.length

@inline _mis_dimension(::_GaussianScalarHistory) = 1
@inline _mis_dimension(history::_GaussianFactorHistory) = size(history.means, 1)

@inline function _native_gaussian_coordinate(
    history::_GaussianScalarHistory,
    normals,
    offset,
    coordinate,
    proposal_slot,
)
    return _gaussian_affine_coordinate(
        @inbounds(history.means[proposal_slot]),
        @inbounds(history.scales[proposal_slot]),
        @inbounds(normals[offset]),
    )
end

@inline function _native_gaussian_coordinate(
    history::_GaussianFactorHistory,
    normals,
    offset,
    coordinate,
    proposal_slot,
)
    value = @inbounds history.means[coordinate, proposal_slot]
    for column in 1:coordinate
        value += @inbounds(
            history.factors[coordinate, column, proposal_slot] *
            normals[offset + column - 1]
        )
    end
    return value
end

@inline function _mis_proposal_logdensity(
    ::Type{T},
    history::_GaussianScalarHistory,
    sample,
    proposal_slot,
    solve_scratch,
    sample_index,
) where {T}
    return convert(
        T,
        _packed_gaussian_logdensity(history, sample, proposal_slot),
    )
end

@inline function _mis_proposal_logdensity(
    ::Type{T},
    history::_GaussianFactorHistory,
    sample,
    proposal_slot,
    solve_scratch,
    sample_index,
) where {T}
    return convert(
        T,
        _packed_gaussian_logdensity!(
            history,
            sample,
            proposal_slot,
            solve_scratch,
            sample_index,
        ),
    )
end

@kernel function _gaussian_batch_target_kernel!(
    logtargets,
    lognumerators,
    logweights,
    round_ids,
    samples,
    target,
    round,
    failure_storage,
)
    sample_index = @index(Global, Linear)
    T = eltype(logweights)
    @inbounds logtargets[sample_index] = T(-Inf)
    @inbounds lognumerators[sample_index] = T(-Inf)
    @inbounds logweights[sample_index] = T(-Inf)
    @inbounds round_ids[sample_index] = 0
    if !_factor_batch_sample_finite(samples, sample_index)
        _record_native_failure!(
            failure_storage,
            sample_index,
            0,
            _NATIVE_GENERATED_NONFINITE,
        )
    else
        target_log, reason, failed = target(
            _native_sample_at(samples, sample_index),
            sample_index,
        )
        if failed
            iszero(reason) ||
                _record_native_failure!(failure_storage, sample_index, 0, reason)
        else
            @inbounds logtargets[sample_index] = target_log
            @inbounds round_ids[sample_index] = round
        end
    end
end

function _preflight_gaussian_fit_kernels(device, method_state, buffers)
    history = method_state.history
    workspace = method_state.workspace
    backend = KernelAbstractions.get_backend(workspace.samples)
    representative_round = findmax(method_state.schedule)[2]
    last_sample = method_state.offsets[representative_round + 1] - 1
    if history isa _GaussianScalarHistory
        ridge_kernel = _add_gaussian_scalar_ridge_kernel!(backend)
        for argument in (workspace.covariance, history.scales, representative_round)
            _preflight_kernel_argument(device, ridge_kernel, argument)
        end
        finish_kernel = _finish_gaussian_scalar_candidate_kernel!(backend)
        for argument in (
            workspace.candidate_scale,
            workspace.candidate_lognormalizer,
            workspace.covariance,
            buffers.failure_scratch.record.storage,
            last_sample + 1,
        )
            _preflight_kernel_argument(device, finish_kernel, argument)
        end
    else
        ridge_kernel = _add_gaussian_factor_ridge_kernel!(backend)
        for argument in (
            workspace.covariance,
            history.factors,
            representative_round,
        )
            _preflight_kernel_argument(device, ridge_kernel, argument)
        end
        finish_kernel = _finish_gaussian_factor_candidate_kernel!(backend)
        for argument in (
            workspace.candidate_mean,
            workspace.candidate_scale,
            workspace.candidate_lognormalizer,
            buffers.failure_scratch.record.storage,
            last_sample + 1,
        )
            _preflight_kernel_argument(device, finish_kernel, argument)
        end
    end
    return nothing
end

function _normalize_gaussian_weights!(normalized_weights, logweights, sample_count)
    T = eltype(normalized_weights)
    maximum_logweight = maximum(view(logweights, 1:sample_count))
    maximum_logweight == -Inf && throw(AllZeroWeightsError())
    total = zero(T)
    @inbounds for sample_index in 1:sample_count
        weight = convert(
            T,
            exp(logweights[sample_index] - maximum_logweight),
        )
        normalized_weights[sample_index] = weight
        total += weight
    end
    isfinite(total) && total > zero(T) || throw(AllZeroWeightsError())
    inverse_total = inv(total)
    @inbounds for sample_index in 1:sample_count
        normalized_weights[sample_index] *= inverse_total
    end
    return nothing
end

function _normalize_gaussian_weights!(
    normalized_weights,
    logweights,
    sample_count,
    transfers::_ResultTransferCounter,
)
    T = eltype(normalized_weights)
    active_logweights = view(logweights, 1:sample_count)
    maximum_logweight = maximum(active_logweights)
    _record_device_scalar_transfer!(
        transfers,
        logweights,
        eltype(logweights),
        Val(:logweight_maximum),
    )
    maximum_logweight == -Inf && throw(AllZeroWeightsError())
    active_weights = view(normalized_weights, 1:sample_count)
    active_weights .= exp.(active_logweights .- maximum_logweight)
    total = sum(active_weights)
    _record_device_scalar_transfer!(
        transfers,
        normalized_weights,
        T,
        Val(:logweight_scaled_sum),
    )
    isfinite(total) && total > zero(T) || throw(AllZeroWeightsError())
    scaled_square_sum = sum(abs2, active_weights)
    _record_device_scalar_transfer!(
        transfers,
        normalized_weights,
        T,
        Val(:logweight_scaled_square_sum),
    )
    active_weights ./= total
    return _logweight_summary(
        maximum_logweight,
        total,
        scaled_square_sum,
        sample_count,
    )
end

@inline function _gaussian_factor_candidate_valid(
    candidate_mean,
    candidate_factor,
    candidate_lognormalizer,
)
    valid = isfinite(candidate_lognormalizer)
    @inbounds for coordinate in eachindex(candidate_mean)
        valid &= isfinite(candidate_mean[coordinate])
    end
    T = eltype(candidate_factor)
    dimension = size(candidate_factor, 1)
    @inbounds for column in 1:dimension, row in column:dimension
        factor_entry = candidate_factor[row, column]
        valid &= isfinite(factor_entry)
        valid &= row != column || factor_entry > zero(T)
    end
    return valid
end

function _fit_gaussian_proposal!(
    workspace::_GaussianMomentWorkspace,
    history::_GaussianScalarHistory,
    previous_slot,
    sample_count,
)
    samples = workspace.samples
    weights = workspace.normalized_weights
    centered_scaled = workspace.centered_scaled
    phase = :moment
    try
    _normalize_gaussian_weights!(weights, workspace.logweights, sample_count)

    T = eltype(samples)
    mean = zero(T)
    @inbounds for sample_index in 1:sample_count
        mean += weights[sample_index] * samples[sample_index]
    end
    variance = zero(T)
    @inbounds for sample_index in 1:sample_count
        centered = (samples[sample_index] - mean) * sqrt(weights[sample_index])
        centered_scaled[sample_index] = centered
        variance += abs2(centered)
    end
    previous_variance = abs2(@inbounds(history.scales[previous_slot]))
    variance += _scale_aware_ridge(
        previous_variance,
        1,
        sqrt(eps(T)),
    )
    workspace.covariance[1] = variance

    phase = :factorization
    isfinite(variance) && variance > zero(T) || throw(
        LinearAlgebra.PosDefException(1),
    )
    return SphericalGaussian(mean, sqrt(variance))
    catch cause
        _throw_gaussian_stage(phase, cause)
    end
end

function _fit_gaussian_proposal!(
    workspace::_GaussianMomentWorkspace,
    history::_GaussianFactorHistory,
    previous_slot,
    sample_count,
)
    samples = workspace.samples
    weights = workspace.normalized_weights
    centered_scaled = workspace.centered_scaled
    covariance = workspace.covariance
    candidate_mean = workspace.candidate_mean
    candidate_factor = workspace.candidate_scale
    T = eltype(samples)
    dimension = size(samples, 1)
    phase = :moment
    try
    _normalize_gaussian_weights!(weights, workspace.logweights, sample_count)

    fill!(candidate_mean, zero(T))
    @inbounds for sample_index in 1:sample_count
        weight = weights[sample_index]
        for coordinate in 1:dimension
            candidate_mean[coordinate] +=
                weight * samples[coordinate, sample_index]
        end
    end
    @inbounds for sample_index in 1:sample_count
        scale = sqrt(weights[sample_index])
        for coordinate in 1:dimension
            centered_scaled[coordinate, sample_index] =
                (samples[coordinate, sample_index] - candidate_mean[coordinate]) *
                scale
        end
    end
    centered = view(centered_scaled, :, 1:sample_count)
    LinearAlgebra.mul!(covariance, centered, transpose(centered))

    previous_trace = zero(T)
    @inbounds for column in 1:dimension, row in column:dimension
        previous_trace += abs2(history.factors[row, column, previous_slot])
    end
    ridge = _scale_aware_ridge(
        previous_trace,
        dimension,
        sqrt(eps(T)),
    )
    @inbounds for coordinate in 1:dimension
        covariance[coordinate, coordinate] += ridge
    end

    phase = :factorization
    copyto!(candidate_factor, covariance)
    _gaussian_potrf!(MLDataDevices.CPUDevice(), candidate_factor)
    @inbounds for column in 1:dimension, row in 1:(column - 1)
        candidate_factor[row, column] = zero(T)
    end

    logabsdet = zero(T)
    @inbounds for coordinate in 1:dimension
        logabsdet += log(abs(candidate_factor[coordinate, coordinate]))
    end
    candidate_lognormalizer =
        _gaussian_lognormalizer(T, dimension, logabsdet)
    _gaussian_factor_candidate_valid(
        candidate_mean,
        candidate_factor,
        candidate_lognormalizer,
    ) || throw(
        ArgumentError(
            "fitted factor candidate must have a finite mean, finite lower " *
            "factor, positive diagonal, and finite lognormalizer",
        ),
    )
    workspace.candidate_lognormalizer[1] = candidate_lognormalizer
    return nothing
    catch cause
        _throw_gaussian_stage(phase, cause)
    end
end

function _gaussian_potrf!(
    ::MLDataDevices.AbstractCPUDevice,
    factor::StridedMatrix{T},
) where {T<:Union{Float32,Float64}}
    factor, info = LinearAlgebra.LAPACK.potrf!('L', factor)
    iszero(info) || throw(LinearAlgebra.PosDefException(info))
    return factor
end

@kernel function _add_gaussian_scalar_ridge_kernel!(covariance, scales, previous_slot)
    T = eltype(covariance)
    covariance[1] += _scale_aware_ridge(
        abs2(scales[previous_slot]),
        1,
        sqrt(eps(T)),
    )
end

@kernel function _add_gaussian_factor_ridge_kernel!(
    covariance,
    factors,
    previous_slot,
)
    T = eltype(covariance)
    dimension = size(covariance, 1)
    previous_trace = zero(T)
    for column in 1:dimension, row in column:dimension
        previous_trace += abs2(factors[row, column, previous_slot])
    end
    ridge = _scale_aware_ridge(
        previous_trace,
        dimension,
        sqrt(eps(T)),
    )
    for coordinate in 1:dimension
        covariance[coordinate, coordinate] += ridge
    end
end

@kernel function _finish_gaussian_scalar_candidate_kernel!(
    candidate_scale,
    candidate_lognormalizer,
    covariance,
    failure_storage,
    failure_index,
)
    T = eltype(candidate_scale)
    variance = covariance[1]
    if isfinite(variance) && variance > zero(T)
        scale = sqrt(variance)
        candidate_scale[1] = scale
        candidate_lognormalizer[1] = _gaussian_lognormalizer(T, 1, log(scale))
    else
        _record_native_failure!(
            failure_storage,
            failure_index,
            0,
            _GAUSSIAN_COVARIANCE_INVALID,
        )
    end
end

@kernel function _finish_gaussian_factor_candidate_kernel!(
    candidate_mean,
    candidate_factor,
    candidate_lognormalizer,
    failure_storage,
    failure_index,
)
    index = @index(Global, Linear)
    dimension = size(candidate_factor, 1)
    row = mod1(index, dimension)
    column = cld(index, dimension)
    row < column && (candidate_factor[row, column] = zero(eltype(candidate_factor)))
    if index == 1
        T = eltype(candidate_factor)
        logabsdet = zero(T)
        for coordinate in 1:dimension
            logabsdet += log(abs(candidate_factor[coordinate, coordinate]))
        end
        lognormalizer =
            _gaussian_lognormalizer(T, dimension, logabsdet)
        candidate_lognormalizer[1] = lognormalizer
        _gaussian_factor_candidate_valid(
            candidate_mean,
            candidate_factor,
            lognormalizer,
        ) || _record_native_failure!(
            failure_storage,
            failure_index,
            0,
            _GAUSSIAN_COVARIANCE_INVALID,
        )
    end
end

function _fit_gaussian_proposal!(
    device::MLDataDevices.AbstractAcceleratorDevice,
    workspace::_GaussianMomentWorkspace,
    history::_GaussianScalarHistory,
    previous_slot,
    sample_count,
    transfers,
    failure_storage,
)
    samples = view(reshape(workspace.samples, 1, :), :, 1:sample_count)
    weights = view(workspace.normalized_weights, 1:sample_count)
    centered = view(
        reshape(workspace.centered_scaled, 1, :),
        :,
        1:sample_count,
    )
    backend = KernelAbstractions.get_backend(workspace.covariance)
    phase = :moment
    try
    summary = _normalize_gaussian_weights!(
        workspace.normalized_weights,
        workspace.logweights,
        sample_count,
        transfers,
    )
    LinearAlgebra.mul!(workspace.candidate_mean, samples, weights)
    centered .= (samples .- reshape(workspace.candidate_mean, 1, 1)) .*
                sqrt.(reshape(weights, 1, :))
    covariance = reshape(workspace.covariance, 1, 1)
    LinearAlgebra.mul!(covariance, centered, transpose(centered))

    ridge_kernel = _add_gaussian_scalar_ridge_kernel!(backend)
    ridge_kernel(
        workspace.covariance,
        history.scales,
        previous_slot;
        ndrange=1,
    )
    KernelAbstractions.synchronize(backend)

    phase = :factorization
    finish_kernel = _finish_gaussian_scalar_candidate_kernel!(backend)
    finish_kernel(
        workspace.candidate_scale,
        workspace.candidate_lognormalizer,
        workspace.covariance,
        failure_storage,
        sample_count + 1;
        ndrange=1,
    )
    KernelAbstractions.synchronize(backend)
    return summary
    catch cause
        _throw_gaussian_stage(phase, cause)
    end
end

function _fit_gaussian_proposal!(
    device::MLDataDevices.AbstractAcceleratorDevice,
    workspace::_GaussianMomentWorkspace,
    history::_GaussianFactorHistory,
    previous_slot,
    sample_count,
    transfers,
    failure_storage,
)
    samples = view(workspace.samples, :, 1:sample_count)
    weights = view(workspace.normalized_weights, 1:sample_count)
    centered = view(workspace.centered_scaled, :, 1:sample_count)
    backend = KernelAbstractions.get_backend(workspace.covariance)
    phase = :moment
    try
    summary = _normalize_gaussian_weights!(
        workspace.normalized_weights,
        workspace.logweights,
        sample_count,
        transfers,
    )
    LinearAlgebra.mul!(workspace.candidate_mean, samples, weights)
    centered .= (samples .- reshape(workspace.candidate_mean, :, 1)) .*
                sqrt.(reshape(weights, 1, :))
    LinearAlgebra.mul!(workspace.covariance, centered, transpose(centered))

    ridge_kernel = _add_gaussian_factor_ridge_kernel!(backend)
    ridge_kernel(
        workspace.covariance,
        history.factors,
        previous_slot;
        ndrange=1,
    )
    KernelAbstractions.synchronize(backend)

    phase = :factorization
    copyto!(workspace.candidate_scale, workspace.covariance)
    _gaussian_potrf!(device, workspace.candidate_scale)
    finish_kernel = _finish_gaussian_factor_candidate_kernel!(backend)
    finish_kernel(
        workspace.candidate_mean,
        workspace.candidate_scale,
        workspace.candidate_lognormalizer,
        failure_storage,
        sample_count + 1;
        ndrange=length(workspace.candidate_scale),
    )
    KernelAbstractions.synchronize(backend)
    return summary
    catch cause
        _throw_gaussian_stage(phase, cause)
    end
end

function _store_gaussian_candidate!(
    history::_GaussianScalarHistory,
    slot,
    workspace::_GaussianMomentWorkspace,
)
    copyto!(view(history.means, slot:slot), workspace.candidate_mean)
    copyto!(view(history.scales, slot:slot), workspace.candidate_scale)
    copyto!(
        view(history.lognormalizers, slot:slot),
        workspace.candidate_lognormalizer,
    )
    return nothing
end

function _store_gaussian_candidate!(
    history::_GaussianFactorHistory,
    slot,
    workspace::_GaussianMomentWorkspace,
)
    copyto!(view(history.means, :, slot), workspace.candidate_mean)
    copyto!(view(history.factors, :, :, slot), workspace.candidate_scale)
    copyto!(
        view(history.lognormalizers, slot:slot),
        workspace.candidate_lognormalizer,
    )
    return nothing
end

function _store_gaussian_proposal!(
    history::_GaussianScalarHistory,
    slot,
    proposal::_GaussianProposal,
)
    history.means[slot] = proposal.location
    history.scales[slot] = proposal.scale.scale
    history.lognormalizers[slot] = proposal.lognormalizer
    return nothing
end

function _store_gaussian_proposal!(
    history::_GaussianFactorHistory,
    slot,
    proposal::_GaussianProposal,
)
    copyto!(view(history.means, :, slot), proposal.location)
    copyto!(view(history.factors, :, :, slot), proposal.scale.factor)
    history.lognormalizers[slot] = proposal.lognormalizer
    return nothing
end

function _store_gaussian_workspace_candidate!(workspace, proposal::_GaussianProposal)
    workspace.candidate_mean[1] = proposal.location
    workspace.candidate_scale[1] = proposal.scale.scale
    workspace.candidate_lognormalizer[1] = proposal.lognormalizer
    return nothing
end

function _reset_gaussian_history!(history::_GaussianScalarHistory, rounds)
    slots = 2:rounds
    isempty(slots) && return nothing
    fill!(view(history.means, slots), zero(eltype(history.means)))
    fill!(view(history.scales, slots), zero(eltype(history.scales)))
    fill!(
        view(history.lognormalizers, slots),
        zero(eltype(history.lognormalizers)),
    )
    return nothing
end

function _reset_gaussian_history!(history::_GaussianFactorHistory, rounds)
    slots = 2:rounds
    isempty(slots) && return nothing
    fill!(view(history.means, :, slots), zero(eltype(history.means)))
    fill!(
        view(history.factors, :, :, slots),
        zero(eltype(history.factors)),
    )
    fill!(
        view(history.lognormalizers, slots),
        zero(eltype(history.lognormalizers)),
    )
    return nothing
end

function _with_gaussian_workspace_authority(
    method_state::_PreparedAdaptiveGaussian{S,O,L,H,W},
    authority::Bool,
) where {S,O,L,H,W}
    return _PreparedAdaptiveGaussian{S,O,L,H,W}(
        method_state.schedule,
        method_state.offsets,
        method_state.logcounts,
        method_state.history,
        method_state.workspace,
        authority,
    )
end

function _gaussian_round_phase(cause::SamplerExecutionError, default)
    phase = cause.phase
    phase === :proposal_draw && return :sampling
    phase === :target && return :target
    phase === :proposal_logdensity && return :denominator
    phase === :logweight && return :weight
    return default
end

_gaussian_round_phase(cause::_GaussianStageError, default) = cause.phase
_gaussian_round_cause(cause) = cause
_gaussian_round_cause(cause::_GaussianStageError) = cause.cause

function _gaussian_round_phase(cause, default)
    cause isa AllZeroWeightsError && default === :factorization && return :moment
    return default
end

function _gaussian_covariance_diagnostics(covariance, transfers)
    diagonal = covariance isa AbstractVector ? covariance :
               view(covariance, LinearAlgebra.diagind(covariance))
    minimum_diagonal = minimum(diagonal)
    maximum_absolute_entry = maximum(abs, covariance)
    T = eltype(covariance)
    _record_device_scalar_transfer!(
        transfers,
        covariance,
        T,
        Val(:covariance_diagnostic),
    )
    _record_device_scalar_transfer!(
        transfers,
        covariance,
        T,
        Val(:covariance_diagnostic),
    )
    return (; minimum_diagonal, maximum_absolute_entry)
end

function _capture_gaussian_round(
    f,
    algorithm,
    round,
    phase,
    round_size,
    completed_rounds,
    cumulative_sample_count,
    covariance,
    transfers,
)
    try
        return f()
    catch cause
        cause isa _gaussian_round_error(algorithm) && rethrow()
        failure_phase = _gaussian_round_phase(cause, phase)
        covariance_diagnostics = failure_phase === :factorization ?
                                 _gaussian_covariance_diagnostics(
            covariance,
            transfers,
        ) : nothing
        throw(
            _gaussian_round_error(algorithm)(
                round,
                failure_phase,
                _gaussian_round_cause(cause),
                (
                    round_size=round_size,
                    completed_rounds=completed_rounds,
                    cumulative_sample_count=cumulative_sample_count,
                    covariance=covariance_diagnostics,
                    transfers=transfers,
                ),
            ),
        )
    end
end

function _capture_gaussian_round(
    f,
    algorithm,
    state::_PreparedAdaptiveGaussian,
    transfers,
    round,
    phase,
    completed_rounds,
    covariance=nothing,
)
    cumulative_sample_count = state.offsets[completed_rounds + 1] - 1
    return _capture_gaussian_round(
        f,
        algorithm,
        round,
        phase,
        state.schedule[round],
        completed_rounds,
        cumulative_sample_count,
        covariance,
        transfers,
    )
end

function _gaussian_failure_snapshot!(transfers, failure_record)
    snapshot = _device_failure_snapshot(failure_record)
    _record_reported_transfer!(
        transfers,
        snapshot.transfers.count,
        snapshot.transfers.bytes,
        Val(:failure_snapshot),
    )
    return snapshot
end

function _importance_sample_cpu!(sampler, committed_state::_PreparedAdaptiveGaussian, threaded)
    algorithm = sampler.algorithm
    execution = threaded ? _ThreadedCPUExecution() : _SerialCPUExecution()
    history = committed_state.history
    workspace = committed_state.workspace
    buffers = sampler.random_buffers
    schedule = committed_state.schedule
    rounds = length(schedule)
    total_samples = last(committed_state.offsets) - 1
    round_ids = similar(workspace.logweights, Int, total_samples)
    transfers = _ResultTransferCounter(0, 0)
    accelerator = sampler.device isa MLDataDevices.AbstractAcceleratorDevice
    round_ess = Vector{eltype(workspace.logweights)}(undef, rounds)
    round_lognormalizers = similar(round_ess)
    adaptation_ess = _allocate_gaussian_adaptation_ess(algorithm, round_ess)
    workspace_candidate = accelerator || history isa _GaussianFactorHistory
    _reset_gaussian_history!(history, rounds)
    method_state = if committed_state.committed_in_workspace
        _capture_gaussian_round(
            algorithm, committed_state, transfers, 1, :result_construction, 0,
        ) do
            _store_gaussian_candidate!(history, 1, workspace)
            KernelAbstractions.synchronize(
                KernelAbstractions.get_backend(history.means),
            )
        end
        run_state = _with_gaussian_workspace_authority(committed_state, false)
        sampler.method_state = run_state
        run_state
    else
        committed_state
    end

    target = _capture_gaussian_round(algorithm, method_state, transfers, 1, :target, 0) do
        binding_sample = history isa _GaussianScalarHistory ?
                         zero(eltype(history.means)) : view(history.means, :, 1)
        _bind_resolved_target(sampler.target, binding_sample)
    end
    target_evaluator, target_failures = _capture_gaussian_round(
        algorithm, method_state, transfers, 1, :target, 0,
    ) do
        _native_target_evaluator(
            KernelAbstractions.get_backend(buffers.normal),
            target,
            eltype(workspace.logweights),
            buffers.failure_scratch.target_failures,
        )
    end

    final_proposal = nothing
    for round in eachindex(schedule)
        deferred_accelerator_snapshot = accelerator
        _capture_gaussian_round(algorithm, method_state, transfers, round, :sampling, round - 1) do
            Random.randn!(sampler.rng, buffers.normal)
        end
        _capture_gaussian_round(algorithm, method_state, transfers, round, :sampling, round - 1) do
            _launch_adaptive_gaussian_round!(
                algorithm,
                workspace.samples,
                workspace.logtargets,
                workspace.lognumerators,
                workspace.logweights,
                round_ids,
                buffers.failure_scratch.record.storage,
                buffers.normal,
                target_evaluator,
                history,
                method_state.logcounts,
                method_state.offsets,
                round,
                workspace.centered_scaled,
                execution,
                sampler.device,
                sampler.factor_execution,
            )
            if !deferred_accelerator_snapshot
                snapshot = _gaussian_failure_snapshot!(
                    transfers,
                    buffers.failure_scratch.record,
                )
                _throw_native_failures(
                    snapshot.failure,
                    snapshot.draw_failure,
                    target_failures,
                    _NoSampleTransform(),
                )
            end
        end
        fit_workspace = _capture_gaussian_round(
            algorithm, method_state, transfers, round, :moment, round - 1,
        ) do
            _gaussian_adaptation_workspace!(algorithm, sampler.device, method_state, round)
        end
        fit_count = _gaussian_adaptation_count(algorithm, method_state, round)
        fit_result = if deferred_accelerator_snapshot
            try
                _fit_gaussian_proposal!(
                    sampler.device,
                    fit_workspace,
                    history,
                    round,
                    fit_count,
                    transfers,
                    buffers.failure_scratch.record.storage,
                )
            catch cause
                _GaussianFitFailure(cause)
            end
        else
            final_proposal = _capture_gaussian_round(
                algorithm,
                method_state,
                transfers,
                round,
                :factorization,
                round - 1,
                workspace.covariance,
            ) do
                _fit_gaussian_proposal!(
                    fit_workspace,
                    history,
                    round,
                    fit_count,
                )
            end
            nothing
        end
        if deferred_accelerator_snapshot
            snapshot = _gaussian_failure_snapshot!(
                transfers,
                buffers.failure_scratch.record,
            )
            _capture_gaussian_round(
                algorithm, method_state, transfers, round, :sampling, round - 1,
            ) do
                snapshot.failure.reason_bits == _GAUSSIAN_COVARIANCE_INVALID ||
                    _throw_native_failures(
                        snapshot.failure,
                        snapshot.draw_failure,
                        target_failures,
                        _NoSampleTransform(),
                    )
            end
            _capture_gaussian_round(
                algorithm,
                method_state,
                transfers,
                round,
                :factorization,
                round - 1,
                workspace.covariance,
            ) do
                fit_result isa _GaussianFitFailure && throw(fit_result.cause)
                if snapshot.failure.reason_bits == _GAUSSIAN_COVARIANCE_INVALID
                    throw(LinearAlgebra.PosDefException(1))
                end
            end
        end
        _capture_gaussian_round(
            algorithm,
            method_state,
            transfers,
            round,
            :result_construction,
            round - 1,
        ) do
            if accelerator && algorithm isa AMIS
                round_ess[round] = fit_result.ess
                round_lognormalizers[round] = fit_result.lognormalizer
            else
                summary = _logweight_summary(
                    view(
                        workspace.logweights,
                        _gaussian_summary_indices(algorithm, method_state, round),
                    ),
                    transfers,
                )
                round_ess[round] = summary.ess
                round_lognormalizers[round] = summary.lognormalizer
            end
            if algorithm isa NPMC
                adaptation_ess[round] = accelerator ? fit_result.ess :
                    inv(sum(abs2, view(fit_workspace.normalized_weights, 1:fit_count)))
            end
        end
        if round < rounds
            if workspace_candidate
                _store_gaussian_candidate!(history, round + 1, workspace)
            else
                _store_gaussian_proposal!(history, round + 1, final_proposal)
            end
        end
    end

    diagnostics = (
        method=_gaussian_method_name(algorithm),
        execution=_execution_name(execution),
        threaded=sampler.threaded,
        factor_execution_policy=_factor_execution_name(
            sampler.device,
            sampler.factor_execution,
        ),
        rounds=rounds,
        round_sizes=collect(schedule),
        round_ess=round_ess,
        round_lognormalizers=round_lognormalizers,
        target_evaluations=total_samples,
        proposal_evaluations=_gaussian_proposal_evaluations(algorithm, total_samples),
        failures=0,
        transfers=transfers,
        _gaussian_extra_diagnostics(algorithm, adaptation_ess, schedule)...,
    )
    result = _capture_gaussian_round(
        algorithm, method_state, transfers, rounds, :result_construction, rounds,
    ) do
        _adopt_validated_weighted_samples(
            copy(workspace.samples),
            copy(workspace.logweights);
            provenance=(round=round_ids,),
            diagnostics=diagnostics,
        )
    end
    _capture_gaussian_round(
        algorithm, method_state, transfers, rounds, :result_construction, rounds,
    ) do
        workspace_candidate ||
            _store_gaussian_workspace_candidate!(workspace, final_proposal)
        KernelAbstractions.synchronize(
            KernelAbstractions.get_backend(workspace.candidate_mean),
        )
    end
    sampler.method_state = _with_gaussian_workspace_authority(method_state, true)
    return result
end
