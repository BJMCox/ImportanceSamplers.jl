@inline _mis_proposal_logdensity(
    ::Type{T},
    bank,
    sample,
    proposal_slot,
    solve_scratch,
    sample_index,
) where {T} = _mis_proposal_logdensity(T, bank, sample, proposal_slot)

@inline function _mis_proposal_logdensity(
    ::Type{T},
    bank::_PackedFactorBank,
    sample,
    proposal_slot,
    solve_scratch,
    sample_index,
) where {T}
    return convert(
        T,
        _packed_gaussian_logdensity!(
            bank,
            sample,
            proposal_slot,
            solve_scratch,
            sample_index,
        ),
    )
end

@inline function _mis_logdenominator_core(
    ::Type{T},
    bank,
    denominator,
    generating_slot,
    sample,
    solve_scratch,
    sample_index,
) where {T}
    value = T(-Inf)
    generating_logdensity = zero(T)
    first_term, last_term = _mis_term_bounds(bank, denominator, generating_slot)
    for term_index in first_term:last_term
        proposal_slot, logmass = _mis_denominator_term(
            T,
            bank,
            denominator,
            term_index,
        )
        logdensity = _mis_proposal_logdensity(
            T,
            bank,
            sample,
            proposal_slot,
            solve_scratch,
            sample_index,
        )
        proposal_slot == generating_slot && (generating_logdensity = logdensity)
        value = LogExpFunctions.logaddexp(value, logmass + logdensity)
    end
    reason = (
        isnan(generating_logdensity) || generating_logdensity == -Inf ||
        isnan(value) || value == -Inf
    ) ? _NATIVE_PROPOSAL_INVALID : UInt16(0)
    return value, generating_logdensity, reason
end

struct _RealizedMixtureDenominator{L}
    logcoefficients::L
    round::Int
end
Adapt.@adapt_structure _RealizedMixtureDenominator

_factor_batch_logcoefficients(bank, denominator::_RealizedMixtureDenominator) =
    view(denominator.logcoefficients, :, denominator.round)

@inline _mis_term_bounds(bank, ::_RealizedMixtureDenominator, generating_slot) =
    (1, _active_proposal_count(bank))

@inline function _mis_denominator_term(
    ::Type{T},
    bank,
    denominator::_RealizedMixtureDenominator,
    term_index,
) where {T}
    return term_index, convert(
        T,
        @inbounds(denominator.logcoefficients[term_index, denominator.round]),
    )
end

@inline _append_logmixture(lognumerator, logcoefficient, logdensity) =
    LogExpFunctions.logaddexp(lognumerator, logcoefficient + logdensity)

@inline _logweight_from_logmixture(logtarget, lognumerator, logtotal) =
    _subtract_logweight(logtarget, lognumerator - logtotal)

struct _NoMISSolveScratch end

@inline function _fused_mis_solve_scratch(
    solve_scratch,
    backend,
)
    if !(solve_scratch isa AbstractMatrix) ||
       backend isa KernelAbstractions.CPU
        return solve_scratch
    end

    dimension, capacity = size(solve_scratch)
    sample_major = reshape(solve_scratch, capacity, dimension)
    return PermutedDimsArray(sample_major, (2, 1))
end

struct _MISAdaptationOutput{T,Q}
    logtargets::T
    generating_logdensities::Q
end

Adapt.@adapt_structure _MISAdaptationOutput

@inline _store_mis_adaptation!(::Nothing, args...) = nothing

@inline function _store_mis_adaptation!(
    output::_MISAdaptationOutput,
    sample_index,
    logtarget,
    generating_logdensity,
)
    @inbounds output.logtargets[sample_index] = logtarget
    @inbounds output.generating_logdensities[sample_index] = generating_logdensity
    return nothing
end

@inline function _store_mis_adaptation!(
    output::_MISAdaptationOutput,
    sample_index,
    value,
    ::Val{:target},
)
    @inbounds output.logtargets[sample_index] = value
    return nothing
end

@inline function _store_mis_adaptation!(
    output::_MISAdaptationOutput,
    assignments,
    sample_index,
    proposal_slot,
    value,
    ::Val{:generating},
)
    if _factor_batch_is_generating(assignments, sample_index, proposal_slot)
        @inbounds output.generating_logdensities[sample_index] = value
    end
    return nothing
end

struct _MISRoundOutput{W,I,A}
    logweights::W
    proposal_ids::I
    adaptation::A
end

_MISRoundOutput(logweights, proposal_ids) =
    _MISRoundOutput(logweights, proposal_ids, nothing)

struct _GaussianRoundOutput{T,N,W,R,L}
    logtargets::T
    lognumerators::N
    logweights::W
    round_ids::R
    logtotal::L
    round::Int
end

Adapt.@adapt_structure _MISRoundOutput
Adapt.@adapt_structure _GaussianRoundOutput

@inline _factor_batch_is_generating(::Nothing, sample_index, proposal_slot) = false
@inline _factor_batch_is_generating(assignments, sample_index, proposal_slot) =
    @inbounds(assignments[sample_index]) == proposal_slot
@inline _factor_batch_sample_valid(::Nothing, sample_index) = true
@inline _factor_batch_sample_valid(proposal_ids, sample_index) =
    !iszero(@inbounds(proposal_ids[sample_index]))
@inline _factor_batch_invalidate_sample!(::Nothing, sample_index) = nothing
@inline function _factor_batch_invalidate_sample!(proposal_ids, sample_index)
    @inbounds proposal_ids[sample_index] = 0
    return nothing
end
@inline _factor_batch_slot_value(value::Number, proposal_slot) = value
@inline _factor_batch_slot_value(values, proposal_slot) =
    @inbounds values[proposal_slot]

@kernel function _append_factor_batch_logmixture_kernel!(
    lognumerators,
    standardized,
    lognormalizers,
    family,
    logcoefficients,
    proposal_slot,
    assignments,
    valid_samples,
    adaptation,
    failure_storage,
)
    sample_index = @index(Global, Linear)
    if _factor_batch_sample_valid(valid_samples, sample_index)
        T = eltype(lognumerators)
        squared_radius = zero(T)
        @inbounds for coordinate in axes(standardized, 1)
            squared_radius += abs2(standardized[coordinate, sample_index])
        end
        logdensity = _radial_logdensity(_radial_family_at(family, proposal_slot),
            _factor_batch_slot_value(lognormalizers, proposal_slot),
            squared_radius, size(standardized, 1))
        _store_mis_adaptation!(
            adaptation,
            assignments,
            sample_index,
            proposal_slot,
            logdensity,
            Val(:generating),
        )
        reason = (
            isnan(logdensity) ||
            logdensity == -Inf && _factor_batch_is_generating(
                assignments,
                sample_index,
                proposal_slot,
            )
        ) ? _NATIVE_PROPOSAL_INVALID : UInt16(0)
        if iszero(reason)
            value = _append_logmixture(
                @inbounds(lognumerators[sample_index]),
                _factor_batch_slot_value(logcoefficients, proposal_slot),
                logdensity,
            )
            reason = isnan(value) ? _NATIVE_PROPOSAL_INVALID : UInt16(0)
            iszero(reason) && (@inbounds lognumerators[sample_index] = value)
        end
        if !iszero(reason)
            _factor_batch_invalidate_sample!(valid_samples, sample_index)
            _record_native_failure!(
                failure_storage,
                sample_index,
                0,
                reason,
            )
        end
    end
end

function _launch_factor_batch_logmixture!(
    lognumerators,
    solve_scratch,
    samples,
    factor_source,
    proposal_slot,
    logcoefficients,
    failure_storage,
    execution,
    assignments=nothing,
    valid_samples=nothing,
    adaptation=nothing,
)
    solved = _factor_batch_solve!(solve_scratch, samples, factor_source, proposal_slot)
    backend = KernelAbstractions.get_backend(lognumerators)
    kernel = _append_factor_batch_logmixture_kernel!(backend)
    kernel(
        lognumerators,
        solved,
        _factor_batch_lognormalizers(factor_source),
        factor_source.family,
        logcoefficients,
        proposal_slot,
        assignments,
        valid_samples,
        adaptation,
        failure_storage;
        ndrange=length(lognumerators),
        workgroupsize=_native_workgroupsize(execution, length(lognumerators)),
    )
    return nothing
end

_factor_batch_logcoefficients(bank, denominator) = nothing
_factor_batch_logcoefficients(bank, ::_FullMixtureDenominator) = bank.logmasses

function _use_factor_batch_mis_path(
    device,
    bank,
    denominator,
    ::Type{T},
    factor_execution,
) where {T}
    return false
end

function _use_factor_batch_mis_path(
    device,
    bank::_PackedFactorBank,
    denominator,
    ::Type{T},
    factor_execution,
) where {T}
    return (T === eltype(bank.locations) || factor_execution isa BatchedFactorExecution) &&
           !isnothing(_factor_batch_logcoefficients(bank, denominator)) &&
           _use_factor_batch_path(device, bank, factor_execution)
end

function _use_factor_batch_mis_path(
    device,
    bank::_PackedFactorBank,
    ::_EqualAllocationGeneratingDenominator,
    ::Type{T},
    factor_execution,
) where {T}
    return (T === eltype(bank.locations) || factor_execution isa BatchedFactorExecution) &&
           _use_factor_batch_path(device, bank, factor_execution)
end

@kernel function _factor_batch_mis_draw_target_kernel!(
    samples,
    logtargets,
    proposal_ids,
    failure_storage,
    normal_buffer,
    target,
    bank,
    assignments,
    adaptation,
)
    sample_index = @index(Global, Linear)
    T = eltype(logtargets)
    @inbounds logtargets[sample_index] = T(-Inf)
    @inbounds proposal_ids[sample_index] = 0
    _store_mis_adaptation!(adaptation, sample_index, T(-Inf), T(-Inf))
    generating_slot = @inbounds assignments[sample_index]
    dimension = _mis_dimension(bank)
    normal_offset = (sample_index - 1) * dimension + 1
    valid = _native_store_gaussian!(
        samples,
        sample_index,
        bank,
        normal_buffer,
        normal_offset,
        generating_slot,
    )
    if !valid
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
            iszero(reason) || _record_native_failure!(
                failure_storage,
                sample_index,
                0,
                reason,
            )
        else
            @inbounds logtargets[sample_index] = target_log
            @inbounds proposal_ids[sample_index] = bank.proposal_ids[generating_slot]
            _store_mis_adaptation!(
                adaptation,
                sample_index,
                target_log,
                Val(:target),
            )
        end
    end
end

@kernel function _finish_factor_batch_mis_kernel!(
    logweights,
    logdenominators,
    proposal_ids,
    failure_storage,
)
    sample_index = @index(Global, Linear)
    if !iszero(@inbounds(proposal_ids[sample_index]))
        denominator = @inbounds logdenominators[sample_index]
        reason = _native_proposal_reason(denominator)
        if iszero(reason)
            value, reason = _subtract_logweight(
                @inbounds(logweights[sample_index]),
                denominator,
            )
            iszero(reason) && (@inbounds logweights[sample_index] = value)
        end
        iszero(reason) || _record_native_failure!(
            failure_storage,
            sample_index,
            0,
            reason,
        )
    end
end

@kernel function _finish_equal_allocation_generating_batch_kernel!(
    logweights,
    standardized,
    lognormalizers,
    family,
    proposal_slot,
    proposal_ids,
    failure_storage,
    sample_offset,
)
    local_index = @index(Global, Linear)
    if !iszero(@inbounds(proposal_ids[local_index]))
        T = eltype(logweights)
        squared_radius = zero(T)
        @inbounds for coordinate in axes(standardized, 1)
            squared_radius += abs2(standardized[coordinate, local_index])
        end
        logdensity = _radial_logdensity(_radial_family_at(family, proposal_slot),
            T(@inbounds(lognormalizers[proposal_slot])), squared_radius, size(standardized, 1))
        reason = _native_proposal_reason(logdensity)
        if iszero(reason)
            value, reason = _subtract_logweight(
                @inbounds(logweights[local_index]),
                logdensity,
            )
            iszero(reason) && (@inbounds logweights[local_index] = value)
        end
        iszero(reason) || _record_native_failure!(
            failure_storage,
            sample_offset + local_index,
            0,
            reason,
        )
    end
end

function _launch_factor_batch_mis_round!(
    samples,
    output::_MISRoundOutput,
    failure_storage,
    normal_buffer,
    target,
    bank::_PackedFactorBank,
    assignments,
    denominator::_EqualAllocationGeneratingDenominator,
    solve_scratch,
    execution,
)
    if target isa _NativeBatchTarget
        _launch_factor_batch_mis_round!(samples, output, failure_storage, normal_buffer,
            _deferred_target(target), bank, assignments, denominator, solve_scratch, execution)
        return _finish_batch_weights!(output, target.target, samples, failure_storage, execution)
    end
    backend = KernelAbstractions.get_backend(normal_buffer)
    sample_count = length(output.logweights)
    proposal_count = size(bank.locations, 2)
    group_size, remainder = divrem(sample_count, proposal_count)
    iszero(remainder) || error(
        "equal-allocation factor batching requires equal contiguous groups",
    )

    draw_kernel = _factor_batch_mis_draw_target_kernel!(backend)
    draw_kernel(
        samples,
        output.logweights,
        output.proposal_ids,
        failure_storage,
        normal_buffer,
        target,
        bank,
        assignments,
        output.adaptation;
        ndrange=sample_count,
        workgroupsize=_native_workgroupsize(execution, sample_count),
    )

    finish_kernel = _finish_equal_allocation_generating_batch_kernel!(backend)
    for proposal_slot in axes(bank.locations, 2)
        first_sample = (proposal_slot - 1) * group_size + 1
        last_sample = proposal_slot * group_size
        group = first_sample:last_sample
        group_samples = view(samples, :, group)
        group_scratch = view(solve_scratch, :, group)
        _factor_batch_solve!(
            group_scratch,
            group_samples,
            bank,
            proposal_slot,
        )
        group_logweights = view(output.logweights, group)
        group_proposal_ids = view(output.proposal_ids, group)
        finish_kernel(
            group_logweights,
            group_scratch,
            bank.lognormalizers,
            bank.family,
            proposal_slot,
            group_proposal_ids,
            failure_storage,
            first_sample - 1;
            ndrange=group_size,
            workgroupsize=_native_workgroupsize(execution, group_size),
        )
    end
    KernelAbstractions.synchronize(backend)
    return nothing
end

function _launch_factor_batch_mis_round!(
    samples,
    output::_MISRoundOutput,
    failure_storage,
    normal_buffer,
    target,
    bank::_PackedFactorBank,
    assignments,
    denominator,
    solve_scratch,
    execution,
)
    if target isa _NativeBatchTarget
        _launch_factor_batch_mis_round!(samples, output, failure_storage, normal_buffer,
            _deferred_target(target), bank, assignments, denominator, solve_scratch, execution)
        return _finish_batch_weights!(output, target.target, samples, failure_storage, execution)
    end
    backend = KernelAbstractions.get_backend(normal_buffer)
    sample_count = length(output.logweights)
    draw_kernel = _factor_batch_mis_draw_target_kernel!(backend)
    draw_kernel(
        samples,
        output.logweights,
        output.proposal_ids,
        failure_storage,
        normal_buffer,
        target,
        bank,
        assignments,
        output.adaptation;
        ndrange=sample_count,
        workgroupsize=_native_workgroupsize(execution, sample_count),
    )

    # Reuse consumed normals only when their precision preserves the weights.
    logdenominators = eltype(normal_buffer) === eltype(output.logweights) ?
        view(normal_buffer, 1:sample_count) : similar(output.logweights)
    fill!(logdenominators, eltype(logdenominators)(-Inf))
    logcoefficients = _factor_batch_logcoefficients(bank, denominator)
    isnothing(logcoefficients) && error("unsupported factor-batch denominator")
    round_solve_scratch = view(solve_scratch, :, 1:sample_count)
    for proposal_slot in axes(bank.locations, 2)
        _launch_factor_batch_logmixture!(
            logdenominators,
            round_solve_scratch,
            samples,
            bank,
            proposal_slot,
            logcoefficients,
            failure_storage,
            execution,
            assignments,
            output.proposal_ids,
            output.adaptation,
        )
    end

    finish_kernel = _finish_factor_batch_mis_kernel!(backend)
    finish_kernel(
        output.logweights,
        logdenominators,
        output.proposal_ids,
        failure_storage;
        ndrange=sample_count,
        workgroupsize=_native_workgroupsize(execution, sample_count),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

@inline _mis_dimension(bank) = size(bank.locations, 1)

function _allocate_mis_solve_scratch(
    prototype,
    bank::_PackedDiagonalBank,
    nsamples,
)
    return _NoMISSolveScratch()
end

function _allocate_mis_solve_scratch(
    prototype,
    bank::_PackedFactorBank,
    nsamples,
)
    return similar(
        prototype,
        eltype(bank.locations),
        size(bank.locations, 1),
        nsamples,
    )
end

@inline function _mis_round_values!(
    ::Type{T},
    sample_index,
    samples,
    failure_storage,
    normal_buffer,
    target,
    bank,
    assignments,
    denominator_policy,
    solve_scratch,
) where {T}
    generating_slot = @inbounds assignments[sample_index]
    dimension = _mis_dimension(bank)
    normal_offset = (sample_index - 1) * dimension + 1
    valid = _native_store_gaussian!(
        samples,
        sample_index,
        bank,
        normal_buffer,
        normal_offset,
        generating_slot,
    )
    if !valid
        _record_native_failure!(
            failure_storage,
            sample_index,
            0,
            _NATIVE_GENERATED_NONFINITE,
        )
        return false, T(-Inf), T(-Inf), generating_slot, T(-Inf)
    else
        sample = _native_sample_at(samples, sample_index)
        target_log, target_reason, target_failed = target(sample, sample_index)
        if target_failed
            iszero(target_reason) || _record_native_failure!(
                failure_storage,
                sample_index,
                0,
                target_reason,
            )
            return false, T(-Inf), T(-Inf), generating_slot, T(-Inf)
        else
            denominator, generating_logdensity, denominator_reason =
                _mis_logdenominator_core(
                    typeof(target_log),
                    bank,
                    denominator_policy,
                    generating_slot,
                    sample,
                    solve_scratch,
                    sample_index,
                )
            if !iszero(denominator_reason)
                _record_native_failure!(
                    failure_storage,
                    sample_index,
                    0,
                    denominator_reason,
                )
                return false, T(-Inf), T(-Inf), generating_slot, T(-Inf)
            end
            return (
                true,
                target_log,
                denominator,
                generating_slot,
                generating_logdensity,
            )
        end
    end
end

@kernel function _mis_round_launch_kernel!(
    samples,
    logweights,
    proposal_ids,
    failure_storage,
    normal_buffer,
    target,
    bank,
    assignments,
    denominator,
    solve_scratch,
    adaptation,
)
    sample_index = @index(Global, Linear)
    T = eltype(logweights)
    @inbounds logweights[sample_index] = T(-Inf)
    @inbounds proposal_ids[sample_index] = 0
    _store_mis_adaptation!(adaptation, sample_index, T(-Inf), T(-Inf))
    valid, target_log, logdenominator, generating_slot, generating_logdensity =
        _mis_round_values!(
            T,
            sample_index,
            samples,
            failure_storage,
            normal_buffer,
            target,
            bank,
            assignments,
            denominator,
            solve_scratch,
        )
    if valid
        logweight, reason = _subtract_logweight(target_log, logdenominator)
        if iszero(reason)
            @inbounds logweights[sample_index] = logweight
            @inbounds proposal_ids[sample_index] = bank.proposal_ids[generating_slot]
            _store_mis_adaptation!(
                adaptation,
                sample_index,
                target_log,
                generating_logdensity,
            )
        else
            _record_native_failure!(failure_storage, sample_index, 0, reason)
        end
    end
end

@kernel function _gaussian_round_launch_kernel!(
    samples,
    logtargets,
    lognumerators,
    logweights,
    round_ids,
    logtotal,
    round,
    failure_storage,
    normal_buffer,
    target,
    history,
    assignments,
    denominator,
    solve_scratch,
)
    sample_index = @index(Global, Linear)
    T = eltype(logweights)
    @inbounds logtargets[sample_index] = T(-Inf)
    @inbounds lognumerators[sample_index] = zero(T)
    @inbounds logweights[sample_index] = T(-Inf)
    round_ids[sample_index] = 0
    valid, target_log, lognumerator, _, _ = _mis_round_values!(
        T,
        sample_index,
        samples,
        failure_storage,
        normal_buffer,
        target,
        history,
        assignments,
        denominator,
        solve_scratch,
    )
    if valid
        logweight, reason = _logweight_from_logmixture(
            target_log,
            lognumerator,
            logtotal,
        )
        if iszero(reason)
            @inbounds logtargets[sample_index] = target_log
            @inbounds lognumerators[sample_index] = lognumerator
            @inbounds logweights[sample_index] = logweight
            @inbounds round_ids[sample_index] = round
        else
            _record_native_failure!(failure_storage, sample_index, 0, reason)
        end
    end
end

@inline function _mis_round_kernel_arguments(
    samples,
    output::_MISRoundOutput,
    failure_storage,
    normal_buffer,
    target,
    bank,
    assignments,
    denominator,
    solve_scratch,
)
    return (
        samples,
        output.logweights,
        output.proposal_ids,
        failure_storage,
        normal_buffer,
        target,
        bank,
        assignments,
        denominator,
        solve_scratch,
        output.adaptation,
    )
end

function _launch_mis_round!(
    samples,
    output::_MISRoundOutput,
    failure_storage,
    normal_buffer,
    target,
    bank,
    assignments,
    denominator,
    solve_scratch,
    execution,
)
    if target isa _NativeBatchTarget
        _launch_mis_round!(samples, output, failure_storage, normal_buffer,
            _deferred_target(target), bank, assignments, denominator, solve_scratch, execution)
        return _finish_batch_weights!(output, target.target, samples, failure_storage, execution)
    end
    backend = KernelAbstractions.get_backend(normal_buffer)
    launch_scratch = _fused_mis_solve_scratch(
        solve_scratch,
        backend,
    )
    kernel = _mis_round_launch_kernel!(backend)
    kernel(
        _mis_round_kernel_arguments(
            samples,
            output,
            failure_storage,
            normal_buffer,
            target,
            bank,
            assignments,
            denominator,
            launch_scratch,
        )...;
        ndrange=length(output.logweights),
        workgroupsize=_native_workgroupsize(
            execution,
            length(output.logweights),
        ),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

function _launch_mis_round!(
    samples,
    output::_GaussianRoundOutput,
    failure_storage,
    normal_buffer,
    target,
    history,
    assignments,
    denominator,
    solve_scratch,
    execution,
)
    if target isa _NativeBatchTarget
        _launch_mis_round!(samples, output, failure_storage, normal_buffer,
            _deferred_target(target), history, assignments, denominator, solve_scratch, execution)
        return _finish_batch_weights!(output, target.target, samples, failure_storage, execution)
    end
    backend = KernelAbstractions.get_backend(normal_buffer)
    launch_scratch = _fused_mis_solve_scratch(
        solve_scratch,
        backend,
    )
    kernel = _gaussian_round_launch_kernel!(backend)
    kernel(
        samples,
        output.logtargets,
        output.lognumerators,
        output.logweights,
        output.round_ids,
        output.logtotal,
        output.round,
        failure_storage,
        normal_buffer,
        target,
        history,
        assignments,
        denominator,
        launch_scratch;
        ndrange=length(output.logweights),
        workgroupsize=_native_workgroupsize(
            execution,
            length(output.logweights),
        ),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end
