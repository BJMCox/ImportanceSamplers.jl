abstract type _MomentAdaptiveSampler <: AbstractImportanceSampler end

function _validate_moment_proposal(proposal::_NativeRadialProposal)
    _validate_moment_family(proposal.family)
    location = proposal.location
    scale = proposal.scale
    if location isa _NativeGaussianFloat
        scale isa _SphericalGaussianScale{typeof(location)} || throw(
            ArgumentError(
                "adaptive moment sampling requires matching native Float32 or Float64 Gaussian or Student-t storage",
            ),
        )
        return nothing
    end
    location isa AbstractVector{<:_NativeGaussianFloat} || throw(
        ArgumentError(
            "adaptive moment sampling requires a native Float32 or Float64 Gaussian or Student-t proposal",
        ),
    )
    isempty(location) && throw(ArgumentError("adaptive proposal location must be nonempty"))
    T = eltype(location)
    supported = scale isa _SphericalGaussianScale{T} ||
                scale isa _DiagonalGaussianScale{<:AbstractVector{T}} ||
                scale isa _FactorGaussianScale{<:AbstractMatrix{T}}
    supported || throw(
        ArgumentError(
            "adaptive moment sampling requires matching native Float32 or Float64 Gaussian or Student-t storage",
        ),
    )
    return nothing
end

function _validate_moment_proposal(proposal)
    throw(
        ArgumentError(
            "adaptive moment sampling requires a native Float32 or Float64 spherical, " *
            "diagonal, or factor Gaussian or Student-t proposal",
        ),
    )
end

_algorithm_proposal(algorithm::_MomentAdaptiveSampler) = algorithm.proposal
_algorithm_sample_budget(algorithm::_MomentAdaptiveSampler) =
    _adaptive_sample_budget(algorithm.rounds, algorithm.round_size)

struct _ScalarProposalHistory{M,S,N,R}
    means::M
    scales::S
    lognormalizers::N
    family::R
end

struct _FactorProposalHistory{M,F,N,R}
    means::M
    factors::F
    lognormalizers::N
    family::R
end

struct _MomentWorkspace{S,T,N,W,P,C,V,M,F,L}
    samples::S
    logtargets::T
    lognumerators::N
    logweights::W
    normalized_weights::P
    centered_scaled::C
    covariance::V
    candidate_mean::M
    candidate_scale::F
    candidate_lognormalizer::L
end

struct _PreparedMomentSampler{S,O,L,H,W}
    schedule::S
    offsets::O
    logcounts::L
    history::H
    workspace::W
    committed_in_workspace::Bool
end

function _gaussian_storage_prototype(proposal::_NativeRadialProposal)
    location = proposal.location
    return location isa _NativeGaussianFloat ?
           Vector{typeof(location)}(undef, 0) : location
end

function _allocate_gaussian_history(prototype, proposal::_NativeRadialProposal, rounds)
    location = proposal.location
    T = location isa _NativeGaussianFloat ? typeof(location) : eltype(location)
    lognormalizers = similar(prototype, T, rounds)
    lognormalizers[1] = proposal.lognormalizer
    if location isa _NativeGaussianFloat
        means = similar(prototype, T, rounds)
        scales = similar(prototype, T, rounds)
        means[1] = location
        scales[1] = proposal.scale.scale
        return _ScalarProposalHistory(means, scales, lognormalizers, proposal.family)
    end

    dimension = length(location)
    means = similar(prototype, T, dimension, rounds)
    factors = similar(prototype, T, dimension, dimension, rounds)
    copyto!(view(means, :, 1), location)
    _store_gaussian_factor!(view(factors, :, :, 1), proposal.scale)
    return _FactorProposalHistory(means, factors, lognormalizers, proposal.family)
end

function _store_gaussian_factor!(factor, scale::_SphericalGaussianScale)
    fill!(factor, zero(eltype(factor)))
    for index in axes(factor, 1)
        factor[index, index] = scale.scale
    end
    return factor
end

function _store_gaussian_factor!(factor, scale::_DiagonalGaussianScale)
    fill!(factor, zero(eltype(factor)))
    for index in axes(factor, 1)
        factor[index, index] = scale.scales[index]
    end
    return factor
end

function _store_gaussian_factor!(factor, scale::_FactorGaussianScale)
    copyto!(factor, scale.factor)
    return factor
end

function _allocate_gaussian_workspace(
    prototype,
    proposal::_NativeRadialProposal,
    capacity,
    ::Type{L},
) where {L<:_NativeGaussianFloat}
    location = proposal.location
    T = location isa _NativeGaussianFloat ? typeof(location) : eltype(location)
    logtargets = similar(prototype, L, capacity)
    lognumerators = similar(prototype, L, capacity)
    logweights = similar(prototype, L, capacity)
    normalized_weights = similar(prototype, T, capacity)
    if location isa _NativeGaussianFloat
        samples = similar(prototype, T, capacity)
        centered_scaled = similar(prototype, T, capacity)
        covariance = similar(prototype, T, 1)
        candidate_mean = similar(prototype, T, 1)
        candidate_scale = similar(prototype, T, 1)
    else
        dimension = length(location)
        samples = similar(prototype, T, dimension, capacity)
        centered_scaled = similar(prototype, T, dimension, capacity)
        covariance = similar(prototype, T, dimension, dimension)
        candidate_mean = similar(prototype, T, dimension)
        candidate_scale = similar(prototype, T, dimension, dimension)
    end
    candidate_lognormalizer = similar(prototype, T, 1)
    return _MomentWorkspace(
        samples,
        logtargets,
        lognumerators,
        logweights,
        normalized_weights,
        centered_scaled,
        covariance,
        candidate_mean,
        candidate_scale,
        candidate_lognormalizer,
    )
end

function _initialize_gaussian_covariance!(
    covariance,
    history::_ScalarProposalHistory,
)
    covariance[1] = abs2(history.scales[1]) * _covariance_multiplier(history.family, eltype(covariance))
    return covariance
end

function _initialize_gaussian_covariance!(
    covariance,
    history::_FactorProposalHistory,
)
    factor = view(history.factors, :, :, 1)
    LinearAlgebra.mul!(covariance, factor, transpose(factor))
    covariance .*= _covariance_multiplier(history.family, eltype(covariance))
    return covariance
end

function _prepare_gaussian_state(algorithm::_MomentAdaptiveSampler, ::Type{L}) where {L}
    schedule = _resolve_adaptive_schedule(algorithm.rounds, algorithm.round_size)
    offsets = cumsum(vcat(1, schedule))
    proposal = algorithm.proposal
    prototype = _gaussian_storage_prototype(proposal)
    T = _native_fused_float_type(proposal)
    logcounts = similar(prototype, T, algorithm.rounds)
    for round in eachindex(schedule)
        logcounts[round] = log(T(schedule[round]))
    end
    history = _allocate_gaussian_history(prototype, proposal, algorithm.rounds)
    workspace = _allocate_gaussian_workspace(
        prototype,
        proposal,
        offsets[end] - 1,
        L,
    )
    _initialize_gaussian_covariance!(workspace.covariance, history)
    return _PreparedMomentSampler(
        schedule,
        offsets,
        logcounts,
        history,
        workspace,
        false,
    )
end

function _prepare_method_state(algorithm::_MomentAdaptiveSampler)
    return _prepare_gaussian_state(
        algorithm,
        _native_fused_float_type(algorithm.proposal),
    )
end

function _prepare_method_state(algorithm::_MomentAdaptiveSampler, prepared_target)
    proposal = algorithm.proposal
    binding_sample = proposal.location isa _NativeGaussianFloat ?
                     zero(proposal.location) : view(proposal.location, :)
    target = _bind_resolved_target(prepared_target, binding_sample)
    log_type = _resolve_native_logweight_type(
        target,
        proposal,
        typeof(binding_sample),
    )
    return _prepare_gaussian_state(algorithm, log_type)
end

function _allocate_random_buffers(
    ::MLDataDevices.AbstractDevice,
    ::_NativeRadialProposal,
    method_state::_PreparedMomentSampler,
    sample_budget,
)
    prototype = method_state.workspace.samples
    T = eltype(method_state.history.means)
    maximum_round_size = maximum(method_state.schedule)
    dimension = method_state.history isa _ScalarProposalHistory ?
                1 : size(method_state.history.means, 1)
    uniform = similar(prototype, T, 0)
    normal = similar(prototype, T, dimension * maximum_round_size)
    failure_scratch = _allocate_native_failure_scratch(
        normal,
        maximum_round_size,
    )
    return _RandomBuffers(uniform, normal, failure_scratch,
        _allocate_radial_buffers(prototype, method_state.history.family, maximum_round_size))
end

function _copy_accelerator_algorithm(
    device,
    algorithm::_MomentAdaptiveSampler,
    ::_PreparedMomentSampler,
)
    return deepcopy(algorithm)
end

function _copy_gaussian_history(device, history::_ScalarProposalHistory)
    return _ScalarProposalHistory(
        _copy_to_device(device, history.means),
        _copy_to_device(device, history.scales),
        _copy_to_device(device, history.lognormalizers),
        history.family,
    )
end

function _copy_gaussian_history(device, history::_FactorProposalHistory)
    return _FactorProposalHistory(
        _copy_to_device(device, history.means),
        _copy_to_device(device, history.factors),
        _copy_to_device(device, history.lognormalizers),
        history.family,
    )
end

function _copy_gaussian_workspace(device, workspace::_MomentWorkspace)
    return _MomentWorkspace(
        _copy_to_device(device, workspace.samples),
        _copy_to_device(device, workspace.logtargets),
        _copy_to_device(device, workspace.lognumerators),
        _copy_to_device(device, workspace.logweights),
        _copy_to_device(device, workspace.normalized_weights),
        _copy_to_device(device, workspace.centered_scaled),
        _copy_to_device(device, workspace.covariance),
        _copy_to_device(device, workspace.candidate_mean),
        _copy_to_device(device, workspace.candidate_scale),
        _copy_to_device(device, workspace.candidate_lognormalizer),
    )
end

function _prepare_transferred_method_state(
    device,
    algorithm::_MomentAdaptiveSampler,
    method_state::_PreparedMomentSampler,
    _transferred_target,
)
    transferred = _PreparedMomentSampler(
        _HostIntSequence(method_state.schedule),
        _HostIntSequence(method_state.offsets),
        _copy_to_device(device, method_state.logcounts),
        _copy_gaussian_history(device, method_state.history),
        _copy_gaussian_workspace(device, method_state.workspace),
        method_state.committed_in_workspace,
    )
    _preflight_gaussian_factorization!(device, transferred)
    return transferred
end

_preflight_gaussian_factorization!(
    device,
    method_state::_PreparedMomentSampler{S,O,L,H,W},
) where {S,O,L,H<:_ScalarProposalHistory,W} =
    nothing

function _preflight_gaussian_factorization!(
    device,
    method_state::_PreparedMomentSampler{S,O,L,H,W},
) where {S,O,L,H<:_FactorProposalHistory,W}
    _gaussian_potrf!(device, method_state.workspace.covariance)
    return nothing
end

function _gaussian_potrf!(device, factor)
    throw(SamplerDeviceError(device, :accelerator_factorization_unavailable))
end

_transferred_backend_state(
    algorithm,
    method_state::_PreparedMomentSampler,
    target,
    random_buffers,
) = (method_state, target, random_buffers)

_prepared_backend_state(sampler, method_state::_PreparedMomentSampler) = (
    method_state,
    sampler.target,
    sampler.random_buffers,
    sampler.rng,
)

"""
    current_proposal(sampler)

Return an independent native Gaussian or Student-t snapshot of the proposal committed by a
CPU-prepared [`AMIS`](@ref) or [`NPMC`](@ref) sampler. For an accelerator-prepared sampler, pass
an explicit preserving CPU destination. A successful call commits its final
fitted proposal for the next call; a failed call leaves this snapshot unchanged.
"""
function current_proposal(
    sampler::_PreparedImportanceSampler{R,B,T,A,M,D},
) where {R,B,T,A<:_MomentAdaptiveSampler,M<:_PreparedMomentSampler,D}
    sampler.device isa MLDataDevices.AbstractAcceleratorDevice && throw(
        ArgumentError(
            "current_proposal(sampler) does not copy accelerator state " *
            "implicitly; call current_proposal(cpu_device(), sampler) " *
            "to request an explicit CPU snapshot",
        ),
    )
    method_state = sampler.method_state
    return _moment_proposal_snapshot(method_state)
end

function current_proposal(
    destination::MLDataDevices.AbstractCPUDevice,
    sampler::_PreparedImportanceSampler{R,B,T,A,M,D},
) where {R,B,T,A<:_MomentAdaptiveSampler,M<:_PreparedMomentSampler,D}
    if applicable(eltype, destination)
        policy = eltype(destination)
        policy in (Missing, Nothing) || throw(
            ArgumentError(
                "current_proposal requires a preserving CPU destination; " *
                "use MLDataDevices.cpu_device() without a scalar conversion",
            ),
        )
    end
    method_state = sampler.method_state
    parameters = _with_backend_device(sampler.device) do
        _copy_gaussian_snapshot_parameters(destination, method_state)
    end
    return _moment_proposal_snapshot(method_state.history.family, parameters...)
end

function current_proposal(
    destination::MLDataDevices.AbstractDevice,
    sampler::_PreparedImportanceSampler{R,B,T,A,M,D},
) where {R,B,T,A<:_MomentAdaptiveSampler,M<:_PreparedMomentSampler,D}
    throw(
        ArgumentError(
            "current_proposal requires a CPU destination; got " *
            string(typeof(destination)),
        ),
    )
end

function _gaussian_snapshot_parameters(
    method_state::_PreparedMomentSampler{S,O,L,H,W},
) where {S,O,L,H<:_ScalarProposalHistory,W}
    if method_state.committed_in_workspace
        workspace = method_state.workspace
        return workspace.candidate_mean, workspace.candidate_scale
    end
    history = method_state.history
    return view(history.means, 1:1), view(history.scales, 1:1)
end

function _gaussian_snapshot_parameters(
    method_state::_PreparedMomentSampler{S,O,L,H,W},
) where {S,O,L,H<:_FactorProposalHistory,W}
    if method_state.committed_in_workspace
        workspace = method_state.workspace
        return workspace.candidate_mean, workspace.candidate_scale
    end
    history = method_state.history
    return view(history.means, :, 1), view(history.factors, :, :, 1)
end

function _copy_gaussian_snapshot_parameters(destination, method_state::_PreparedMomentSampler)
    parameters = _gaussian_snapshot_parameters(method_state)
    return map(parameter -> destination(Array(parameter)), parameters)
end

_moment_proposal_snapshot(method_state::_PreparedMomentSampler) =
    _moment_proposal_snapshot(method_state.history.family, _gaussian_snapshot_parameters(method_state)...)

_moment_proposal_snapshot(::GaussianFamily, means::AbstractVector, scales::AbstractVector) =
    SphericalGaussian(means[1], scales[1])
_moment_proposal_snapshot(family::StudentTFamily, means::AbstractVector, scales::AbstractVector) =
    SphericalStudentT(family.dof, means[1], scales[1])
_moment_proposal_snapshot(::GaussianFamily, mean::AbstractVector, factor::AbstractMatrix) =
    FactorGaussian(mean, factor)
_moment_proposal_snapshot(family::StudentTFamily, mean::AbstractVector, factor::AbstractMatrix) =
    FactorStudentT(family.dof, mean, factor)
