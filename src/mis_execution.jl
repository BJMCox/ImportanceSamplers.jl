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

struct _NoMISSolveScratch end

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

@kernel function _mis_round_kernel!(
    samples,
    logweights,
    proposal_ids,
    failure_storage,
    normal_buffer,
    target,
    bank,
    assignments,
    denominator_policy,
    solve_scratch,
)
    sample_index = @index(Global, Linear)
    generating_slot = @inbounds assignments[sample_index]
    dimension = size(bank.locations, 1)
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
        sample = _native_sample_at(samples, sample_index)
        target_log, target_reason, target_failed = target(sample, sample_index)
        if target_failed
            iszero(target_reason) || _record_native_failure!(
                failure_storage,
                sample_index,
                0,
                target_reason,
            )
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
            else
                logweight, logweight_reason = _subtract_logweight(
                    target_log,
                    denominator,
                )
                if iszero(logweight_reason)
                    @inbounds logweights[sample_index] = logweight
                    @inbounds proposal_ids[sample_index] =
                        bank.proposal_ids[generating_slot]
                else
                    _record_native_failure!(
                        failure_storage,
                        sample_index,
                        0,
                        logweight_reason,
                    )
                end
            end
        end
    end
end

function _launch_mis_round!(
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
    execution,
)
    backend = KernelAbstractions.get_backend(normal_buffer)
    kernel = _mis_round_kernel!(backend)
    kernel(
        samples,
        logweights,
        proposal_ids,
        failure_storage,
        normal_buffer,
        target,
        bank,
        assignments,
        denominator,
        solve_scratch;
        ndrange=length(logweights),
        workgroupsize=_native_workgroupsize(execution, length(logweights)),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end
