struct _AMISMixtureDenominator{L}
    logcounts::L
    round::Int
end

struct _FixedMISAssignments
    slot::Int
    length::Int
end

Adapt.@adapt_structure _AMISScalarHistory
Adapt.@adapt_structure _AMISFactorHistory
Adapt.@adapt_structure _AMISMixtureDenominator

Base.@propagate_inbounds Base.getindex(assignments::_FixedMISAssignments, index) =
    assignments.slot
Base.length(assignments::_FixedMISAssignments) = assignments.length

@inline _mis_dimension(::_AMISScalarHistory) = 1
@inline _mis_dimension(history::_AMISFactorHistory) = size(history.means, 1)

@inline function _native_gaussian_coordinate(
    history::_AMISScalarHistory,
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
    history::_AMISFactorHistory,
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
    history::_AMISScalarHistory,
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
    history::_AMISFactorHistory,
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

@inline _mis_term_bounds(
    history,
    denominator::_AMISMixtureDenominator,
    generating_slot,
) = (1, denominator.round)

@inline function _mis_denominator_term(
    ::Type{T},
    history,
    denominator::_AMISMixtureDenominator,
    term_index,
) where {T}
    return term_index, convert(
        T,
        @inbounds(denominator.logcounts[term_index]),
    )
end

@kernel function _append_logmixture_kernel!(
    lognumerators,
    samples,
    history,
    slot,
    logcounts,
    failure_storage,
    solve_scratch,
)
    sample_index = @index(Global, Linear)
    logdensity = _mis_proposal_logdensity(
        eltype(lognumerators),
        history,
        _native_sample_at(samples, sample_index),
        slot,
        solve_scratch,
        sample_index,
    )
    reason = _native_proposal_reason(logdensity)
    if iszero(reason)
        value = _append_logmixture(
            @inbounds(lognumerators[sample_index]),
            @inbounds(logcounts[slot]),
            logdensity,
        )
        reason = _native_proposal_reason(value)
        iszero(reason) && (@inbounds lognumerators[sample_index] = value)
    end
    iszero(reason) || _record_native_failure!(
        failure_storage,
        sample_index,
        0,
        reason,
    )
end

function _launch_append_logmixture!(
    lognumerators,
    samples,
    history,
    slot,
    logcounts,
    failure_storage,
    solve_scratch,
    execution,
)
    backend = KernelAbstractions.get_backend(lognumerators)
    kernel = _append_logmixture_kernel!(backend)
    kernel(
        lognumerators,
        samples,
        history,
        slot,
        logcounts,
        failure_storage,
        solve_scratch;
        ndrange=length(lognumerators),
        workgroupsize=_native_workgroupsize(
            execution,
            length(lognumerators),
        ),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function _form_amis_logweights_kernel!(
    logweights,
    logtargets,
    lognumerators,
    logtotal,
    failure_storage,
)
    sample_index = @index(Global, Linear)
    value, reason = _logweight_from_logmixture(
        @inbounds(logtargets[sample_index]),
        @inbounds(lognumerators[sample_index]),
        logtotal,
    )
    if iszero(reason)
        @inbounds logweights[sample_index] = value
    else
        _record_native_failure!(
            failure_storage,
            sample_index,
            0,
            reason,
        )
    end
end

function _launch_form_amis_logweights!(
    logweights,
    logtargets,
    lognumerators,
    logtotal,
    failure_storage,
    execution,
)
    backend = KernelAbstractions.get_backend(logweights)
    kernel = _form_amis_logweights_kernel!(backend)
    kernel(
        logweights,
        logtargets,
        lognumerators,
        logtotal,
        failure_storage;
        ndrange=length(logweights),
        workgroupsize=_native_workgroupsize(execution, length(logweights)),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

function _launch_prefilled_amis_round!(
    samples,
    logtargets,
    lognumerators,
    logweights,
    round_ids,
    failure_storage,
    normal_buffer,
    target,
    history,
    logcounts,
    offsets,
    round,
    solve_scratch,
    execution,
)
    first_sample = offsets[round]
    last_sample = offsets[round + 1] - 1
    old_sample_count = first_sample - 1
    if !iszero(old_sample_count)
        old_indices = 1:old_sample_count
        _launch_append_logmixture!(
            view(lognumerators, old_indices),
            _sample_view(samples, old_indices),
            history,
            round,
            logcounts,
            failure_storage,
            solve_scratch,
            execution,
        )
    end

    new_indices = first_sample:last_sample
    logtotal = log(eltype(logcounts)(last_sample))
    output = _AMISRoundOutput(
        view(logtargets, new_indices),
        view(lognumerators, new_indices),
        view(logweights, new_indices),
        view(round_ids, new_indices),
        logtotal,
        round,
    )
    _launch_mis_round!(
        _sample_view(samples, new_indices),
        output,
        failure_storage,
        normal_buffer,
        target,
        history,
        _FixedMISAssignments(round, length(new_indices)),
        _AMISMixtureDenominator(logcounts, round),
        solve_scratch,
        execution,
    )

    current_indices = 1:last_sample
    _launch_form_amis_logweights!(
        view(logweights, current_indices),
        view(logtargets, current_indices),
        view(lognumerators, current_indices),
        logtotal,
        failure_storage,
        execution,
    )
    return nothing
end

function _normalize_amis_weights!(normalized_weights, logweights, sample_count)
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

function _fit_amis_proposal!(
    workspace::_AMISWorkspace,
    history::_AMISScalarHistory,
    previous_slot,
    sample_count,
)
    samples = workspace.samples
    weights = workspace.normalized_weights
    centered_scaled = workspace.centered_scaled
    _normalize_amis_weights!(weights, workspace.logweights, sample_count)

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
    variance += sqrt(eps(T)) * previous_variance
    isfinite(variance) && variance > zero(T) || throw(
        LinearAlgebra.PosDefException(1),
    )
    workspace.covariance[1] = variance
    return SphericalGaussian(mean, sqrt(variance))
end

function _fit_amis_proposal!(
    workspace::_AMISWorkspace,
    history::_AMISFactorHistory,
    previous_slot,
    sample_count,
)
    samples = workspace.samples
    weights = workspace.normalized_weights
    centered_scaled = workspace.centered_scaled
    covariance = workspace.covariance
    _normalize_amis_weights!(weights, workspace.logweights, sample_count)

    T = eltype(samples)
    dimension = size(samples, 1)
    mean = zeros(T, dimension)
    @inbounds for sample_index in 1:sample_count
        weight = weights[sample_index]
        for coordinate in 1:dimension
            mean[coordinate] += weight * samples[coordinate, sample_index]
        end
    end
    @inbounds for sample_index in 1:sample_count
        scale = sqrt(weights[sample_index])
        for coordinate in 1:dimension
            centered_scaled[coordinate, sample_index] =
                (samples[coordinate, sample_index] - mean[coordinate]) * scale
        end
    end
    centered = view(centered_scaled, :, 1:sample_count)
    LinearAlgebra.mul!(covariance, centered, transpose(centered))

    previous_trace = zero(T)
    @inbounds for column in 1:dimension, row in column:dimension
        previous_trace += abs2(history.factors[row, column, previous_slot])
    end
    ridge = sqrt(eps(T)) * previous_trace / T(dimension)
    @inbounds for coordinate in 1:dimension
        covariance[coordinate, coordinate] += ridge
    end

    candidate_factor, info = LinearAlgebra.LAPACK.potrf!('L', covariance)
    iszero(info) || throw(LinearAlgebra.PosDefException(info))
    @inbounds for column in 1:dimension, row in 1:(column - 1)
        candidate_factor[row, column] = zero(T)
    end
    return FactorGaussian(mean, candidate_factor)
end

function _store_amis_proposal!(
    history::_AMISScalarHistory,
    slot,
    proposal::_GaussianProposal,
)
    history.means[slot] = proposal.location
    history.scales[slot] = proposal.scale.scale
    history.lognormalizers[slot] = proposal.lognormalizer
    return nothing
end

function _store_amis_proposal!(
    history::_AMISFactorHistory,
    slot,
    proposal::_GaussianProposal,
)
    copyto!(view(history.means, :, slot), proposal.location)
    copyto!(view(history.factors, :, :, slot), proposal.scale.factor)
    history.lognormalizers[slot] = proposal.lognormalizer
    return nothing
end

function _reset_amis_history!(history::_AMISScalarHistory)
    length(history.means) == 1 && return nothing
    fill!(view(history.means, 2:lastindex(history.means)), zero(eltype(history.means)))
    fill!(view(history.scales, 2:lastindex(history.scales)), zero(eltype(history.scales)))
    fill!(
        view(history.lognormalizers, 2:lastindex(history.lognormalizers)),
        zero(eltype(history.lognormalizers)),
    )
    return nothing
end

function _reset_amis_history!(history::_AMISFactorHistory)
    size(history.means, 2) == 1 && return nothing
    fill!(view(history.means, :, 2:size(history.means, 2)), zero(eltype(history.means)))
    fill!(
        view(history.factors, :, :, 2:size(history.factors, 3)),
        zero(eltype(history.factors)),
    )
    fill!(
        view(history.lognormalizers, 2:lastindex(history.lognormalizers)),
        zero(eltype(history.lognormalizers)),
    )
    return nothing
end

function _capture_amis_round(f, round, phase, round_size, completed_rounds)
    try
        return f()
    catch cause
        cause isa AMISRoundError && rethrow()
        throw(
            AMISRoundError(
                round,
                phase,
                cause,
                (round_size=round_size, completed_rounds=completed_rounds),
            ),
        )
    end
end

function _build_amis_result(target, samples, logweights, round_ids, diagnostics)
    return _adopt_validated_weighted_samples(
        copy(samples),
        copy(logweights);
        provenance=(round=round_ids,),
        diagnostics=diagnostics,
    )
end

function _importance_sample_cpu!(sampler, method_state::_PreparedAMIS, threaded)
    execution = threaded ? _ThreadedCPUExecution() : _SerialCPUExecution()
    history = method_state.history
    workspace = method_state.workspace
    buffers = sampler.random_buffers
    schedule = method_state.schedule
    rounds = length(schedule)
    total_samples = last(method_state.offsets) - 1
    round_ids = Vector{Int}(undef, total_samples)
    _reset_amis_history!(history)

    target = _capture_amis_round(1, :sample_and_weight, schedule[1], 0) do
        binding_sample = history isa _AMISScalarHistory ?
                         zero(eltype(history.means)) : view(history.means, :, 1)
        _bind_resolved_target(sampler.target, binding_sample)
    end
    target_evaluator, target_failures = _capture_amis_round(
        1,
        :sample_and_weight,
        schedule[1],
        0,
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
        round_size = schedule[round]
        _capture_amis_round(round, :normal_buffer, round_size, round - 1) do
            Random.randn!(sampler.rng, buffers.normal)
        end
        _capture_amis_round(round, :sample_and_weight, round_size, round - 1) do
            _launch_prefilled_amis_round!(
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
            )
            snapshot = _device_failure_snapshot(buffers.failure_scratch.record)
            _throw_native_failures(
                snapshot.failure,
                snapshot.draw_failure,
                target_failures,
                _NoSampleTransform(),
            )
        end
        final_proposal = _capture_amis_round(
            round,
            :fit_proposal,
            round_size,
            round - 1,
        ) do
            _fit_amis_proposal!(workspace, history, round, method_state.offsets[round + 1] - 1)
        end
        round < rounds && _store_amis_proposal!(history, round + 1, final_proposal)
    end

    diagnostics = (
        method=:adaptive_multiple_importance_sampling,
        execution=_execution_name(execution),
        threaded=sampler.threaded,
        rounds=rounds,
        round_sizes=copy(schedule),
        target_evaluations=total_samples,
        proposal_evaluations=rounds * total_samples,
        failures=0,
        transfers=(count=0, bytes=0),
    )
    result = _capture_amis_round(
        rounds,
        :result_construction,
        schedule[end],
        rounds,
    ) do
        _build_amis_result(
            sampler.target,
            workspace.samples,
            workspace.logweights,
            round_ids,
            diagnostics,
        )
    end
    _store_amis_proposal!(history, 1, final_proposal)
    return result
end
