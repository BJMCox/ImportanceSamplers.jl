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
    bank::_PackedFactorGaussianBank,
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

@inline _append_logmixture(lognumerator, logcoefficient, logdensity) =
    LogExpFunctions.logaddexp(lognumerator, logcoefficient + logdensity)

@inline _logweight_from_logmixture(logtarget, lognumerator, logtotal) =
    _subtract_logweight(logtarget, lognumerator - logtotal)

struct _NoMISSolveScratch end

struct _MISRoundOutput{W,I}
    logweights::W
    proposal_ids::I
end

struct _AMISRoundOutput{T,N,W,R,L}
    logtargets::T
    lognumerators::N
    logweights::W
    round_ids::R
    logtotal::L
    round::Int
end

@inline _mis_dimension(bank) = size(bank.locations, 1)

function _allocate_mis_solve_scratch(
    prototype,
    bank::_PackedDiagonalGaussianBank,
    nsamples,
)
    return _NoMISSolveScratch()
end

function _allocate_mis_solve_scratch(
    prototype,
    bank::_PackedFactorGaussianBank,
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
        return false, T(-Inf), T(-Inf), generating_slot
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
            return false, T(-Inf), T(-Inf), generating_slot
        else
            denominator, _, denominator_reason = _mis_logdenominator_core(
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
                return false, T(-Inf), T(-Inf), generating_slot
            end
            return true, target_log, denominator, generating_slot
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
)
    sample_index = @index(Global, Linear)
    valid, target_log, logdenominator, generating_slot = _mis_round_values!(
        eltype(logweights),
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
        else
            _record_native_failure!(failure_storage, sample_index, 0, reason)
        end
    end
end

@kernel function _amis_round_launch_kernel!(
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
    valid, target_log, lognumerator, _ = _mis_round_values!(
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
    backend = KernelAbstractions.get_backend(normal_buffer)
    kernel = _mis_round_launch_kernel!(backend)
    kernel(
        samples,
        output.logweights,
        output.proposal_ids,
        failure_storage,
        normal_buffer,
        target,
        bank,
        assignments,
        denominator,
        solve_scratch;
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
    output::_AMISRoundOutput,
    failure_storage,
    normal_buffer,
    target,
    history,
    assignments,
    denominator,
    solve_scratch,
    execution,
)
    backend = KernelAbstractions.get_backend(normal_buffer)
    kernel = _amis_round_launch_kernel!(backend)
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
        solve_scratch;
        ndrange=length(output.logweights),
        workgroupsize=_native_workgroupsize(
            execution,
            length(output.logweights),
        ),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end
