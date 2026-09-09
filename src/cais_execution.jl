"""
    CAISRoundError

Exception thrown when a CAIS round cannot complete. `round` and `phase` locate
the failure, `cause` stores the underlying exception, and `diagnostics` reports
the requested round size and number of completed rounds. The prepared sampler
retains its complete pre-call proposal population.
"""
struct CAISRoundError{E,D<:NamedTuple} <: Exception
    round::Int
    phase::Symbol
    cause::E
    diagnostics::D
end

function Base.showerror(io::IO, error::CAISRoundError)
    print(io, "CAIS failed in round ", error.round, " during ", error.phase, ": ")
    showerror(io, error.cause)
end

struct _CAISTemperingError <: Exception
    proposal_slot::Int
end

function Base.showerror(io::IO, error::_CAISTemperingError)
    print(io, "CAIS tempering failed for proposal ", error.proposal_slot)
end

struct _CAISCovarianceError <: Exception
    proposal_slot::Int
    info::Int
end

function Base.showerror(io::IO, error::_CAISCovarianceError)
    print(
        io,
        "CAIS covariance factorization failed for proposal ",
        error.proposal_slot,
        " (info=",
        error.info,
        ')',
    )
end

_population_round_error(::CAIS) = CAISRoundError
_population_round_phase(::CAIS) = :covariance
_population_round_views(method_state::_PreparedCAIS, round) =
    _cais_round_views(method_state, round)
_population_mis_adaptation(::_PreparedCAIS, views) = nothing
_population_denominator(::_PreparedCAIS, round) =
    _EqualAllocationGeneratingDenominator()

function _population_execution(sampler, threaded, ::_PreparedCAIS)
    cpu_execution = threaded ? _ThreadedCPUExecution() : _SerialCPUExecution()
    return sampler.device isa MLDataDevices.AbstractAcceleratorDevice ?
           _KernelExecution(cpu_execution) : cpu_execution
end

function _copy_cais_bank!(
    destination::_PackedDiagonalGaussianBank,
    source::_PackedDiagonalGaussianBank,
)
    copyto!(destination.locations, source.locations)
    copyto!(destination.scales, source.scales)
    copyto!(destination.lognormalizers, source.lognormalizers)
    return nothing
end

function _copy_cais_bank!(
    destination::_PackedFactorGaussianBank,
    source::_PackedFactorGaussianBank,
)
    copyto!(destination.locations, source.locations)
    copyto!(destination.factors, source.factors)
    copyto!(destination.lognormalizers, source.lognormalizers)
    return nothing
end

function _reset_population_run!(method_state::_PreparedCAIS, run, committed)
    return _copy_cais_bank!(run, committed)
end

function _commit_population_round!(method_state::_PreparedCAIS, run)
    _copy_cais_bank!(run, method_state.candidate_bank)
    return nothing
end

@kernel function _install_cais_scalar_factors!(scales, fitted_factors)
    proposal_slot = @index(Global, Linear)
    @inbounds scales[1, proposal_slot] = fitted_factors[1, 1, proposal_slot]
end

@kernel function _update_cais_lognormalizers_kernel!(
    lognormalizers,
    fitted_factors,
)
    proposal_slot = @index(Global, Linear)
    T = eltype(lognormalizers)
    logabsdet = zero(T)
    for coordinate in axes(fitted_factors, 1)
        logabsdet += log(@inbounds(fitted_factors[coordinate, coordinate, proposal_slot]))
    end
    @inbounds lognormalizers[proposal_slot] =
        _gaussian_lognormalizer(T, size(fitted_factors, 1), logabsdet)
end

function _install_cais_factors!(
    candidate::_PackedDiagonalGaussianBank,
    fitted_factors,
    execution,
)
    backend = KernelAbstractions.get_backend(candidate.scales)
    kernel = _install_cais_scalar_factors!(backend)
    kernel(
        candidate.scales,
        fitted_factors;
        ndrange=size(candidate.scales, 2),
        workgroupsize=_population_workgroupsize(
            execution,
            backend,
            size(candidate.scales, 2),
        ),
    )
    return nothing
end

function _install_cais_factors!(
    candidate::_PackedFactorGaussianBank,
    fitted_factors,
    execution,
)
    copyto!(candidate.factors, fitted_factors)
    return nothing
end

function _update_cais_lognormalizers!(
    candidate,
    fitted_factors,
    ::Union{_SerialCPUExecution,_ThreadedCPUExecution},
)
    T = eltype(candidate.lognormalizers)
    dimension = size(fitted_factors, 1)
    @inbounds for proposal_slot in eachindex(candidate.lognormalizers)
        logabsdet = zero(T)
        for coordinate in axes(fitted_factors, 1)
            logabsdet += log(fitted_factors[coordinate, coordinate, proposal_slot])
        end
        candidate.lognormalizers[proposal_slot] =
            _gaussian_lognormalizer(T, dimension, logabsdet)
    end
    return nothing
end

function _update_cais_lognormalizers!(candidate, fitted_factors, execution::_KernelExecution)
    backend = KernelAbstractions.get_backend(candidate.lognormalizers)
    kernel = _update_cais_lognormalizers_kernel!(backend)
    proposal_count = length(candidate.lognormalizers)
    kernel(
        candidate.lognormalizers,
        fitted_factors;
        ndrange=proposal_count,
        workgroupsize=_population_workgroupsize(
            execution,
            backend,
            proposal_count,
        ),
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

function _cais_failure_vector(device, values, transfers)
    if device isa MLDataDevices.AbstractAcceleratorDevice
        copied = _with_backend_device(device) do
            Array(values)
        end
        _record_reported_transfer!(
            transfers,
            1,
            sizeof(eltype(values)) * length(values),
            Val(:covariance_diagnostic),
        )
        return copied
    end
    return values
end

function _throw_cais_weight_failure(device, status, transfers)
    failure_status = _cais_failure_vector(device, status, transfers)
    for proposal_slot in eachindex(failure_status)
        value = failure_status[proposal_slot]
        value == _POPULATION_COVARIANCE_READY && continue
        value == _POPULATION_ALL_ZERO_LOCAL && throw(AllZeroWeightsError())
        throw(_CAISTemperingError(proposal_slot))
    end
    return nothing
end

function _throw_cais_factor_failure(device, info, transfers)
    failure_info = _cais_failure_vector(device, info, transfers)
    for proposal_slot in eachindex(failure_info)
        iszero(failure_info[proposal_slot]) || throw(
            _CAISCovarianceError(proposal_slot, Int(failure_info[proposal_slot])),
        )
    end
    return nothing
end

function _advance_population!(
    sampler,
    method_state::_PreparedCAIS,
    views,
    round,
    execution,
    transfers,
)
    workspace = method_state.workspace
    candidate = method_state.candidate_bank

    _population_local_weights!(
        workspace.normalized_weights,
        workspace.local_ess,
        workspace.tempering_powers,
        workspace.factor_status,
        views.logweights,
        workspace.local_starts,
        method_state.plan.counts,
        method_state.covariance_ess_threshold,
        round,
        method_state.tempering_tolerance,
        method_state.tempering_max_iterations,
        execution,
        candidate.locations,
        views.samples,
    )
    KernelAbstractions.synchronize(
        KernelAbstractions.get_backend(workspace.normalized_weights),
    )
    _throw_cais_weight_failure(sampler.device, workspace.factor_status, transfers)
    _fit_population_covariances!(
        workspace.covariances,
        workspace.covariance_centres,
        workspace.normalized_weights,
        workspace.tempering_powers,
        workspace.factor_status,
        views.samples,
        method_state.run_bank.locations,
        method_state.run_bank,
        workspace.local_starts,
        method_state.plan.counts,
        round,
        execution,
    )
    _factor_population_covariances!(
        workspace.fitted_factors,
        workspace.covariances,
        workspace.factor_info,
        workspace.factor_status,
        execution,
    )
    _throw_cais_factor_failure(sampler.device, workspace.factor_info, transfers)
    _install_cais_factors!(candidate, workspace.fitted_factors, execution)
    _update_cais_lognormalizers!(candidate, workspace.fitted_factors, execution)
    copyto!(view(workspace.local_ess_history, :, round), workspace.local_ess)
    copyto!(
        view(workspace.tempering_power_history, :, round),
        workspace.tempering_powers,
    )
    return nothing
end

function _population_diagnostics(
    sampler,
    method_state::_PreparedCAIS,
    execution,
    schedule,
    round_ess,
    round_lognormalizers,
    transfers,
)
    return (
        method=:cais,
        execution=_execution_name(execution),
        threaded=sampler.threaded,
        factor_execution_policy=_use_factor_batch_mis_path(
            sampler.device,
            method_state.bank,
            _EqualAllocationGeneratingDenominator(),
            eltype(method_state.workspace.round_logweights),
            sampler.factor_execution,
        ) ? :batched : :fused,
        rounds=sampler.algorithm.rounds,
        round_sizes=collect(schedule),
        round_ess,
        round_lognormalizers,
        local_ess=copy(method_state.workspace.local_ess_history),
        tempering_powers=copy(method_state.workspace.tempering_power_history),
        target_evaluations=sum(schedule),
        proposal_evaluations=sum(schedule),
        denominator_evaluations=sum(schedule),
        failures=0,
        transfers,
    )
end

function _importance_sample_cpu!(sampler, method_state::_PreparedCAIS, threaded)
    return _importance_sample_fixed_population!(sampler, method_state, threaded)
end

function _preflight_accelerator_method(
    device,
    target,
    algorithm::CAIS,
    method_state::_PreparedCAIS,
    buffers::_PopulationNormalBuffers,
    factor_execution,
)
    workspace = method_state.workspace
    bank = method_state.bank
    representative_round = findmax(method_state.plan.schedule)[2]
    views = _cais_round_views(method_state, representative_round)
    bound_target = _bind_resolved_target(target, _population_binding_sample(bank))
    log_type = eltype(views.logweights)
    target_argument = _NativeDeviceTarget{log_type,typeof(bound_target)}(bound_target)
    backend = KernelAbstractions.get_backend(buffers.normals)
    denominator = _EqualAllocationGeneratingDenominator()
    output = _MISRoundOutput(
        views.logweights,
        views.proposal_ids,
        nothing,
    )
    if _use_factor_batch_mis_path(
        device,
        bank,
        denominator,
        log_type,
        factor_execution,
    )
        draw_kernel = _factor_batch_mis_draw_target_kernel!(backend)
        for argument in (
            views.samples,
            output.logweights,
            output.proposal_ids,
            buffers.failure_scratch.record.storage,
            buffers.normals,
            target_argument,
            bank,
            views.assignments,
            output.adaptation,
        )
            _preflight_kernel_argument(device, draw_kernel, argument)
        end
        group_size = views.round_size ÷ size(bank.locations, 2)
        group = 1:group_size
        finish_kernel =
            _finish_equal_allocation_generating_batch_kernel!(backend)
        for argument in (
            view(output.logweights, group),
            view(workspace.solve_scratch, :, group),
            bank.lognormalizers,
            1,
            view(output.proposal_ids, group),
            buffers.failure_scratch.record.storage,
            0,
        )
            _preflight_kernel_argument(device, finish_kernel, argument)
        end
    else
        round_kernel = _mis_round_launch_kernel!(backend)
        for argument in _mis_round_kernel_arguments(
            views.samples,
            output,
            buffers.failure_scratch.record.storage,
            buffers.normals,
            target_argument,
            bank,
            views.assignments,
            denominator,
            _fused_mis_solve_scratch(workspace.solve_scratch, backend),
        )
            _preflight_kernel_argument(device, round_kernel, argument)
        end
    end
    for (kernel, arguments) in (
        (
            _cooperative_population_local_weights_kernel!(
                backend,
                _POPULATION_REDUCTION_WORKGROUP_SIZE,
            ),
            (
                workspace.normalized_weights,
                workspace.local_ess,
                workspace.tempering_powers,
                workspace.factor_status,
                views.logweights,
                workspace.local_starts,
                method_state.plan.counts,
                method_state.covariance_ess_threshold,
                representative_round,
                method_state.tempering_tolerance,
                method_state.tempering_max_iterations,
                method_state.candidate_bank.locations,
                views.samples,
            ),
        ),
        (
            _population_covariance_centres_kernel!(
                backend,
                _POPULATION_REDUCTION_WORKGROUP_SIZE,
            ),
            (
                workspace.covariance_centres,
                workspace.normalized_weights,
                workspace.tempering_powers,
                workspace.factor_status,
                views.samples,
                bank.locations,
                workspace.local_starts,
                method_state.plan.counts,
                representative_round,
            ),
        ),
        (
            _population_covariances_kernel!(backend),
            (
                workspace.covariances,
                workspace.covariance_centres,
                workspace.normalized_weights,
                workspace.factor_status,
                views.samples,
                bank,
                workspace.local_starts,
                method_state.plan.counts,
                representative_round,
            ),
        ),
        (
            _factor_population_covariances_kernel!(
                backend,
                _POPULATION_CHOLESKY_WORKGROUP_SIZE,
            ),
            (
                workspace.fitted_factors,
                workspace.covariances,
                workspace.factor_info,
                workspace.factor_status,
            ),
        ),
    )
        for argument in arguments
            _preflight_kernel_argument(device, kernel, argument)
        end
    end
    return nothing
end
