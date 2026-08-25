function _allocate_random_buffers(
    ::MLDataDevices.AbstractCPUDevice,
    ::ProposalBank,
    method_state::_PreparedStaticMIS{<:_PackedDiagonalGaussianBank},
    nsamples,
)
    bank = method_state.bank
    uniform = Vector{eltype(bank.cdf)}(undef, nsamples)
    normal = Vector{eltype(bank.locations)}(
        undef,
        size(bank.locations, 1) * nsamples,
    )
    assignments = Vector{Int}(undef, nsamples)
    failure_scratch = _allocate_native_failure_scratch(normal, nsamples)
    return _PackedStaticMISRandomBuffers(
        uniform,
        normal,
        assignments,
        failure_scratch,
    )
end

@inline function _static_mis_assignment(cdf, uniform)
    for slot in eachindex(cdf)
        uniform <= @inbounds(cdf[slot]) && return slot
    end
    return lastindex(cdf)
end

@inline function _static_mis_assignment_uniform(
    ::_StratifiedAssignment,
    uniform,
    sample_index,
    nsamples,
)
    return ((sample_index - 1) + uniform) / nsamples
end

@inline _static_mis_assignment_uniform(
    ::_RandomAssignment,
    uniform,
    sample_index,
    nsamples,
) = uniform

@kernel function _static_mis_assignment_kernel!(
    assignments,
    uniforms,
    cdf,
    assignment,
)
    sample_index = @index(Global, Linear)
    uniform = _static_mis_assignment_uniform(
        assignment,
        @inbounds(uniforms[sample_index]),
        sample_index,
        length(assignments),
    )
    @inbounds assignments[sample_index] = _static_mis_assignment(cdf, uniform)
end

@inline function _packed_full_mis_logdenominator(
    ::Type{L},
    bank,
    generating_slot,
    sample,
) where {L}
    denominator = L(-Inf)
    generating_logdensity = zero(L)
    for proposal_slot in axes(bank.locations, 2)
        proposal_logdensity = convert(
            L,
            _packed_gaussian_logdensity(bank, sample, proposal_slot),
        )
        isnan(proposal_logdensity) && return denominator, _NATIVE_PROPOSAL_INVALID
        proposal_slot == generating_slot &&
            (generating_logdensity = proposal_logdensity)
        logterm = convert(L, @inbounds(bank.logmasses[proposal_slot])) +
                  proposal_logdensity
        denominator = LogExpFunctions.logaddexp(denominator, logterm)
    end
    reason = (
        isnan(generating_logdensity) || generating_logdensity == -Inf ||
        isnan(denominator) || denominator == -Inf
    ) ? _NATIVE_PROPOSAL_INVALID : UInt16(0)
    return denominator, reason
end

@inline function _packed_partial_mis_logdenominator(
    ::Type{L},
    bank,
    denominator_policy::_PartialMixtureDenominator,
    generating_slot,
    sample,
) where {L}
    group = @inbounds denominator_policy.group_of_slot[generating_slot]
    first_member = @inbounds denominator_policy.offsets[group]
    last_member = @inbounds(denominator_policy.offsets[group + 1]) - 1
    denominator = L(-Inf)
    generating_logdensity = zero(L)
    for member_index in first_member:last_member
        proposal_slot = @inbounds denominator_policy.members[member_index]
        proposal_logdensity = convert(
            L,
            _packed_gaussian_logdensity(bank, sample, proposal_slot),
        )
        isnan(proposal_logdensity) && return denominator, _NATIVE_PROPOSAL_INVALID
        proposal_slot == generating_slot &&
            (generating_logdensity = proposal_logdensity)
        logterm = convert(
            L,
            @inbounds(denominator_policy.logcoefficients[member_index]),
        ) + proposal_logdensity
        denominator = LogExpFunctions.logaddexp(denominator, logterm)
    end
    reason = (
        isnan(generating_logdensity) || generating_logdensity == -Inf ||
        isnan(denominator) || denominator == -Inf
    ) ? _NATIVE_PROPOSAL_INVALID : UInt16(0)
    return denominator, reason
end

@inline function _packed_mis_logdenominator(
    ::Type{L},
    bank,
    ::_FullMixtureDenominator,
    generating_slot,
    sample,
) where {L}
    return _packed_full_mis_logdenominator(L, bank, generating_slot, sample)
end

@inline function _packed_mis_logdenominator(
    ::Type{L},
    bank,
    ::_GeneratingDenominator,
    generating_slot,
    sample,
) where {L}
    denominator = convert(
        L,
        _packed_gaussian_logdensity(bank, sample, generating_slot),
    )
    reason = (isnan(denominator) || denominator == -Inf) ?
             _NATIVE_PROPOSAL_INVALID : UInt16(0)
    return denominator, reason
end

@inline function _packed_mis_logdenominator(
    ::Type{L},
    bank,
    denominator::_PartialMixtureDenominator,
    generating_slot,
    sample,
) where {L}
    return _packed_partial_mis_logdenominator(
        L,
        bank,
        denominator,
        generating_slot,
        sample,
    )
end

@kernel function _static_mis_sampling_kernel!(
    samples,
    logweights,
    proposal_ids,
    failure_storage,
    normal_buffer,
    target,
    bank,
    assignments,
    denominator_policy,
)
    sample_index = @index(Global, Linear)
    generating_slot = @inbounds assignments[sample_index]
    dimension = size(bank.locations, 1)
    normal_offset = (sample_index - 1) * dimension + 1
    valid = _store_packed_gaussian!(
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
            denominator, denominator_reason = _packed_mis_logdenominator(
                typeof(target_log),
                bank,
                denominator_policy,
                generating_slot,
                sample,
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

function _allocate_packed_static_mis_samples(
    bank::_PackedDiagonalGaussianBank{L,S,N,M,C,I,<:_ScalarGaussianLayout},
    nsamples,
) where {L,S,N,M,C,I}
    return Vector{eltype(bank.locations)}(undef, nsamples)
end

function _allocate_packed_static_mis_samples(
    bank::_PackedDiagonalGaussianBank{L,S,N,M,C,I,<:_VectorGaussianLayout},
    nsamples,
) where {L,S,N,M,C,I}
    return Matrix{eltype(bank.locations)}(
        undef,
        size(bank.locations, 1),
        nsamples,
    )
end

function _resolve_packed_static_mis_logweight_type(target, bank, sample_type)
    target_type = _capture_sampler_failure(:target, 1) do
        inferred = Base.promote_op(target, sample_type)
        _canonical_inferred_log_type(inferred, "target")
    end
    return promote_type(
        target_type,
        eltype(bank.lognormalizers),
        eltype(bank.logmasses),
    )
end

function _launch_packed_static_mis!(
    samples,
    logweights,
    proposal_ids,
    buffers,
    target,
    method_state,
    execution,
)
    backend = KernelAbstractions.get_backend(buffers.normal)
    assignment_kernel = _static_mis_assignment_kernel!(backend)
    assignment_kernel(
        buffers.assignments,
        buffers.uniform,
        method_state.bank.cdf,
        method_state.design.assignment;
        ndrange=length(logweights),
        workgroupsize=_native_workgroupsize(execution, length(logweights)),
    )
    KernelAbstractions.synchronize(backend)

    sampling_kernel = _static_mis_sampling_kernel!(backend)
    sampling_kernel(
        samples,
        logweights,
        proposal_ids,
        buffers.failure_scratch.record.storage,
        buffers.normal,
        target,
        method_state.bank,
        buffers.assignments,
        method_state.design.denominator;
        ndrange=length(logweights),
        workgroupsize=_native_workgroupsize(execution, length(logweights)),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

function _importance_sample_cpu!(
    sampler,
    method_state::_PreparedStaticMIS{<:_PackedDiagonalGaussianBank},
    threaded,
)
    cpu_execution = threaded ? _ThreadedCPUExecution() : _SerialCPUExecution()
    buffers = _capture_sampler_failure(:proposal_draw, 1) do
        _fill_random_buffers!(sampler.rng, sampler.random_buffers)
    end
    nsamples = sampler.algorithm.nsamples
    samples = _allocate_packed_static_mis_samples(method_state.bank, nsamples)
    binding_sample = _native_binding_sample(samples)
    target = _capture_sampler_failure(:target, 1) do
        _bind_resolved_target(sampler.target, binding_sample)
    end
    log_type = _resolve_packed_static_mis_logweight_type(
        target,
        method_state.bank,
        typeof(binding_sample),
    )
    logweights = Vector{log_type}(undef, nsamples)
    proposal_ids = Vector{Int}(undef, nsamples)
    failure_scratch = buffers.failure_scratch
    target_evaluator, target_failures = _native_target_evaluator(
        KernelAbstractions.get_backend(buffers.normal),
        target,
        log_type,
        failure_scratch.target_failures,
    )
    _launch_packed_static_mis!(
        samples,
        logweights,
        proposal_ids,
        buffers,
        target_evaluator,
        method_state,
        cpu_execution,
    )
    snapshot = _device_failure_snapshot(failure_scratch.record)
    _throw_native_failures(
        snapshot.failure,
        target_failures,
        _NoSampleTransform(),
    )
    diagnostics = (
        method=:importance_sampling,
        mis_scheme=_mis_scheme_name(
            method_state.design.assignment,
            method_state.design.denominator,
        ),
        execution=_execution_name(cpu_execution),
        threaded=sampler.threaded,
        nsamples=nsamples,
        failures=0,
        transfers=snapshot.transfers,
    )
    return _adopt_validated_weighted_samples(
        samples,
        logweights;
        provenance=(proposal_id=proposal_ids,),
        diagnostics=diagnostics,
    )
end
