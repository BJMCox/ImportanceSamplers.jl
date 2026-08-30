@kernel function _first_order_gramis_local_weights_kernel!(
    local_logweights,
    generating_logdensities,
    proposal_ids,
    round_ids,
    round,
    failure_storage,
)
    sample_index = @index(Global, Linear)
    @inbounds round_ids[sample_index] = 0
    if !iszero(@inbounds(proposal_ids[sample_index]))
        local_logweight, reason = _subtract_logweight(
            @inbounds(local_logweights[sample_index]),
            @inbounds(generating_logdensities[sample_index]),
        )
        if iszero(reason)
            @inbounds local_logweights[sample_index] = local_logweight
            @inbounds round_ids[sample_index] = round
        else
            _record_native_failure!(failure_storage, sample_index, 0, reason)
        end
    end
end

function _first_order_gramis_sample_round!(
    samples,
    returned_logweights,
    local_logweights,
    generating_logdensities,
    proposal_ids,
    round_ids,
    failure_storage,
    normal_buffer,
    target,
    bank,
    assignments,
    denominator,
    solve_scratch,
    round,
    execution,
    device,
    factor_execution,
)
    adaptation = _MISAdaptationOutput(
        local_logweights,
        generating_logdensities,
    )
    output = _MISRoundOutput(
        returned_logweights,
        proposal_ids,
        adaptation,
    )
    launch = _use_factor_batch_mis_path(
        device,
        bank,
        denominator,
        eltype(returned_logweights),
        factor_execution,
    ) ? _launch_factor_batch_mis_round! : _launch_mis_round!
    launch(
        samples,
        output,
        failure_storage,
        normal_buffer,
        target,
        bank,
        assignments,
        denominator,
        solve_scratch,
        execution,
    )

    backend = KernelAbstractions.get_backend(local_logweights)
    kernel = _first_order_gramis_local_weights_kernel!(backend)
    kernel(
        local_logweights,
        generating_logdensities,
        proposal_ids,
        round_ids,
        round,
        failure_storage;
        ndrange=length(local_logweights),
        workgroupsize=_native_workgroupsize(execution, length(local_logweights)),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end
