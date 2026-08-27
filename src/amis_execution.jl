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
