function _allocate_random_buffers(
    ::MLDataDevices.AbstractCPUDevice,
    ::ProposalBank,
    method_state::_PreparedStaticMIS{<:Union{
        _PackedDiagonalGaussianBank,
        _PackedFactorGaussianBank,
    }},
    nsamples,
)
    bank = method_state.bank
    return _allocate_packed_static_mis_random_buffers(
        bank.locations,
        bank,
        nsamples,
    )
end

function _copy_accelerator_algorithm(
    device,
    algorithm::ImportanceSampling{<:ProposalBank},
    ::_PreparedStaticMIS{<:Union{
        _PackedDiagonalGaussianBank,
        _PackedFactorGaussianBank,
    }},
)
    return deepcopy(algorithm)
end

_accelerator_method_state_limit(
    ::_PreparedStaticMIS{<:Union{
        _PackedDiagonalGaussianBank,
        _PackedFactorGaussianBank,
    }},
) = nothing

function _prepare_transferred_method_state(
    device,
    algorithm,
    method_state::_PreparedStaticMIS{<:Union{
        _PackedDiagonalGaussianBank,
        _PackedFactorGaussianBank,
    }},
    _transferred_target,
)
    transferred_bank = _copy_packed_gaussian_bank(device, method_state.bank)
    design = method_state.design
    transferred_design = _PreparedMISDesign(
        design.assignment,
        _transfer_static_mis_denominator(device, design.denominator),
    )
    return _PreparedStaticMIS(transferred_bank, transferred_design)
end

_transfer_static_mis_denominator(device, denominator) = denominator

function _transfer_static_mis_denominator(
    device,
    denominator::_PartialMixtureDenominator,
)
    return _PartialMixtureDenominator(
        _copy_to_device(device, denominator.group_of_slot),
        _copy_to_device(device, denominator.offsets),
        _copy_to_device(device, denominator.members),
        _copy_to_device(device, denominator.logcoefficients),
    )
end

_transferred_backend_state(
    algorithm,
    method_state::_PreparedStaticMIS{<:Union{
        _PackedDiagonalGaussianBank,
        _PackedFactorGaussianBank,
    }},
    target,
    random_buffers,
) = (method_state, target, random_buffers)

_prepared_backend_state(
    sampler,
    method_state::_PreparedStaticMIS{<:Union{
        _PackedDiagonalGaussianBank,
        _PackedFactorGaussianBank,
    }},
) = (
    method_state,
    sampler.target,
    sampler.random_buffers,
    sampler.rng,
)

function _preflight_accelerator_method(
    device,
    target,
    algorithm,
    method_state::_PreparedStaticMIS{<:Union{
        _PackedDiagonalGaussianBank,
        _PackedFactorGaussianBank,
    }},
    random_buffers,
    factor_execution,
)
    return _preflight_packed_static_mis_kernel_target(
        device,
        target,
        method_state,
        random_buffers,
        factor_execution,
    )
end

function _allocate_random_buffers(
    device::MLDataDevices.AbstractAcceleratorDevice,
    ::ProposalBank,
    method_state::_PreparedStaticMIS{<:Union{
        _PackedDiagonalGaussianBank,
        _PackedFactorGaussianBank,
    }},
    nsamples,
)
    bank = method_state.bank
    prototype = device(Vector{eltype(bank.locations)}(undef, 0))
    return _allocate_packed_static_mis_random_buffers(prototype, bank, nsamples)
end

function _allocate_packed_static_mis_random_buffers(
    prototype,
    bank::Union{_PackedDiagonalGaussianBank,_PackedFactorGaussianBank},
    nsamples,
)
    uniform = similar(prototype, eltype(bank.cdf), nsamples)
    normal = similar(
        prototype,
        eltype(bank.locations),
        size(bank.locations, 1) * nsamples,
    )
    assignments = similar(prototype, Int, nsamples)
    solve_scratch = _allocate_mis_solve_scratch(prototype, bank, nsamples)
    failure_scratch = _allocate_native_failure_scratch(normal, nsamples)
    return _PackedStaticMISRandomBuffers(
        uniform,
        normal,
        assignments,
        solve_scratch,
        failure_scratch,
    )
end

@kernel function _static_mis_assignment_kernel!(
    assignments,
    uniforms,
    cdf,
    assignment,
)
    sample_index = @index(Global, Linear)
    @inbounds assignments[sample_index] = _static_mis_assignment(
        assignment,
        cdf,
        @inbounds(uniforms[sample_index]),
        sample_index,
        length(assignments),
    )
end

function _allocate_packed_static_mis_samples(
    prototype,
    bank::_PackedDiagonalGaussianBank{L,S,N,M,C,I,<:_ScalarGaussianLayout},
    nsamples,
) where {L,S,N,M,C,I}
    return similar(prototype, eltype(bank.locations), nsamples)
end

function _allocate_packed_static_mis_samples(
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

function _allocate_packed_static_mis_samples(
    prototype,
    bank::_PackedDiagonalGaussianBank{L,S,N,M,C,I,<:_VectorGaussianLayout},
    nsamples,
) where {L,S,N,M,C,I}
    return similar(
        prototype,
        eltype(bank.locations),
        size(bank.locations, 1),
        nsamples,
    )
end

function _preflight_packed_static_mis_kernel_target(
    device,
    target,
    method_state::_PreparedStaticMIS{<:Union{
        _PackedDiagonalGaussianBank,
        _PackedFactorGaussianBank,
    }},
    buffers::_PackedStaticMISRandomBuffers,
    factor_execution,
)
    bank = method_state.bank
    samples = _allocate_packed_static_mis_samples(buffers.normal, bank, 1)
    binding_sample = _native_binding_sample(samples)
    bound_target = _bind_resolved_target(target, binding_sample)
    log_type = _resolve_packed_static_mis_logweight_type(
        bound_target,
        bank,
        typeof(binding_sample),
    )
    logweights = similar(buffers.normal, log_type, 1)
    proposal_ids = similar(buffers.assignments, Int, 1)
    target_argument = _NativeDeviceTarget{log_type,typeof(bound_target)}(bound_target)
    backend = KernelAbstractions.get_backend(buffers.normal)

    assignment_kernel = _static_mis_assignment_kernel!(backend)
    for argument in (
        buffers.assignments,
        buffers.uniform,
        bank.cdf,
        method_state.design.assignment,
    )
        _preflight_kernel_argument(device, assignment_kernel, argument)
    end

    sampling_kernel = _mis_round_launch_kernel!(backend)
    for argument in (
        samples,
        logweights,
        proposal_ids,
        buffers.failure_scratch.record.storage,
        buffers.normal,
        target_argument,
        bank,
        buffers.assignments,
        method_state.design.denominator,
        buffers.solve_scratch,
        nothing,
    )
        _preflight_kernel_argument(device, sampling_kernel, argument)
    end
    if _use_factor_batch_mis_path(
        device,
        bank,
        method_state.design.denominator,
        log_type,
        factor_execution,
    )
        batch_kernel = _factor_batch_mis_draw_target_kernel!(backend)
        _preflight_kernel_argument(device, batch_kernel, target_argument)
    end
    return nothing
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
    device,
    factor_execution,
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

    launch = _use_factor_batch_mis_path(
        device,
        method_state.bank,
        method_state.design.denominator,
        eltype(logweights),
        factor_execution,
    ) ? _launch_factor_batch_mis_round! : _launch_mis_round!
    launch(
        samples,
        _MISRoundOutput(logweights, proposal_ids, nothing),
        buffers.failure_scratch.record.storage,
        buffers.normal,
        target,
        method_state.bank,
        buffers.assignments,
        method_state.design.denominator,
        buffers.solve_scratch,
        execution,
    )
    return nothing
end

function _importance_sample_cpu!(
    sampler,
    method_state::_PreparedStaticMIS{<:Union{
        _PackedDiagonalGaussianBank,
        _PackedFactorGaussianBank,
    }},
    threaded,
)
    cpu_execution = threaded ? _ThreadedCPUExecution() : _SerialCPUExecution()
    buffers = _capture_sampler_failure(:proposal_draw, 1) do
        _fill_random_buffers!(sampler.rng, sampler.random_buffers)
    end
    nsamples = sampler.algorithm.nsamples
    samples = _allocate_packed_static_mis_samples(
        buffers.normal,
        method_state.bank,
        nsamples,
    )
    binding_sample = _native_binding_sample(samples)
    target = _capture_sampler_failure(:target, 1) do
        _bind_resolved_target(sampler.target, binding_sample)
    end
    log_type = _resolve_packed_static_mis_logweight_type(
        target,
        method_state.bank,
        typeof(binding_sample),
    )
    logweights = similar(buffers.normal, log_type, nsamples)
    proposal_ids = similar(buffers.assignments, Int, nsamples)
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
        sampler.device,
        sampler.factor_execution,
    )
    snapshot = _device_failure_snapshot(failure_scratch.record)
    _throw_native_failures(
        snapshot.failure,
        snapshot.draw_failure,
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
        factor_execution_policy=_factor_execution_name(sampler.factor_execution),
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
