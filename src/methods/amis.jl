mutable struct _ValidatedAMISToken end
const _VALIDATED_AMIS_TOKEN = _ValidatedAMISToken()

"""
    AMIS(proposal; rounds, round_size)

Configure adaptive multiple importance sampling with a fixed round schedule.
`proposal` must be a native `Float32` or `Float64` scalar spherical Gaussian,
or a vector spherical, diagonal, or factor Gaussian. `round_size` is either one
positive `Int` repeated for every round or a positive `Vector{Int}` with one
entry per round.
"""
struct AMIS{P,S} <: AbstractImportanceSampler
    proposal::P
    rounds::Int
    round_size::S

    function AMIS(
        proposal::P,
        rounds::Int,
        round_size::S,
        token::_ValidatedAMISToken,
    ) where {P,S}
        token === _VALIDATED_AMIS_TOKEN || throw(
            ArgumentError("invalid internal algorithm-construction token"),
        )
        return new{P,S}(proposal, rounds, round_size)
    end
end

function AMIS(proposal; rounds, round_size)
    schedule = _validate_adaptive_schedule(rounds, round_size)
    _validate_amis_proposal(proposal)
    return AMIS(proposal, rounds, schedule, _VALIDATED_AMIS_TOKEN)
end

function _validate_amis_proposal(proposal::_GaussianProposal)
    location = proposal.location
    scale = proposal.scale
    if location isa _NativeGaussianFloat
        scale isa _SphericalGaussianScale{typeof(location)} || throw(
            ArgumentError(
                "AMIS requires matching native Float32 or Float64 Gaussian storage",
            ),
        )
        return nothing
    end
    location isa AbstractVector{<:_NativeGaussianFloat} || throw(
        ArgumentError(
            "AMIS requires a native Float32 or Float64 Gaussian proposal",
        ),
    )
    isempty(location) && throw(ArgumentError("AMIS proposal location must be nonempty"))
    T = eltype(location)
    supported = scale isa _SphericalGaussianScale{T} ||
                scale isa _DiagonalGaussianScale{<:AbstractVector{T}} ||
                scale isa _FactorGaussianScale{<:AbstractMatrix{T}}
    supported || throw(
        ArgumentError(
            "AMIS requires matching native Float32 or Float64 Gaussian storage",
        ),
    )
    return nothing
end

function _validate_amis_proposal(proposal)
    throw(
        ArgumentError(
            "AMIS requires a native Float32 or Float64 spherical, " *
            "diagonal, or factor Gaussian proposal",
        ),
    )
end

_algorithm_proposal(algorithm::AMIS) = algorithm.proposal
_algorithm_sample_budget(algorithm::AMIS) =
    sum(_resolve_adaptive_schedule(algorithm.rounds, algorithm.round_size))

struct _AMISScalarHistory{M,S,N}
    means::M
    scales::S
    lognormalizers::N
end

struct _AMISFactorHistory{M,F,N}
    means::M
    factors::F
    lognormalizers::N
end

struct _AMISWorkspace{S,T,N,W,P,C,V}
    samples::S
    logtargets::T
    lognumerators::N
    logweights::W
    normalized_weights::P
    centered_scaled::C
    covariance::V
end

struct _PreparedAMIS{S,O,L,H,W}
    schedule::S
    offsets::O
    logcounts::L
    history::H
    workspace::W
end

function _amis_storage_prototype(proposal::_GaussianProposal)
    location = proposal.location
    return location isa _NativeGaussianFloat ?
           Vector{typeof(location)}(undef, 0) : location
end

function _allocate_amis_history(prototype, proposal::_GaussianProposal, rounds)
    location = proposal.location
    T = location isa _NativeGaussianFloat ? typeof(location) : eltype(location)
    lognormalizers = similar(prototype, T, rounds)
    lognormalizers[1] = proposal.lognormalizer
    if location isa _NativeGaussianFloat
        means = similar(prototype, T, rounds)
        scales = similar(prototype, T, rounds)
        means[1] = location
        scales[1] = proposal.scale.scale
        return _AMISScalarHistory(means, scales, lognormalizers)
    end

    dimension = length(location)
    means = similar(prototype, T, dimension, rounds)
    factors = similar(prototype, T, dimension, dimension, rounds)
    copyto!(view(means, :, 1), location)
    _store_amis_factor!(view(factors, :, :, 1), proposal.scale)
    return _AMISFactorHistory(means, factors, lognormalizers)
end

function _store_amis_factor!(factor, scale::_SphericalGaussianScale)
    fill!(factor, zero(eltype(factor)))
    for index in axes(factor, 1)
        factor[index, index] = scale.scale
    end
    return factor
end

function _store_amis_factor!(factor, scale::_DiagonalGaussianScale)
    fill!(factor, zero(eltype(factor)))
    for index in axes(factor, 1)
        factor[index, index] = scale.scales[index]
    end
    return factor
end

function _store_amis_factor!(factor, scale::_FactorGaussianScale)
    copyto!(factor, scale.factor)
    return factor
end

function _allocate_amis_workspace(
    prototype,
    proposal::_GaussianProposal,
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
    else
        dimension = length(location)
        samples = similar(prototype, T, dimension, capacity)
        centered_scaled = similar(prototype, T, dimension, capacity)
        covariance = similar(prototype, T, dimension, dimension)
    end
    return _AMISWorkspace(
        samples,
        logtargets,
        lognumerators,
        logweights,
        normalized_weights,
        centered_scaled,
        covariance,
    )
end

function _prepare_amis_state(algorithm::AMIS, ::Type{L}) where {L}
    schedule = _resolve_adaptive_schedule(algorithm.rounds, algorithm.round_size)
    offsets = cumsum(vcat(1, schedule))
    proposal = algorithm.proposal
    prototype = _amis_storage_prototype(proposal)
    T = _native_fused_float_type(proposal)
    logcounts = similar(prototype, T, algorithm.rounds)
    for round in eachindex(schedule)
        logcounts[round] = log(T(schedule[round]))
    end
    history = _allocate_amis_history(prototype, proposal, algorithm.rounds)
    workspace = _allocate_amis_workspace(
        prototype,
        proposal,
        offsets[end] - 1,
        L,
    )
    return _PreparedAMIS(
        schedule,
        offsets,
        logcounts,
        history,
        workspace,
    )
end

function _prepare_method_state(algorithm::AMIS)
    return _prepare_amis_state(
        algorithm,
        _native_fused_float_type(algorithm.proposal),
    )
end

function _prepare_method_state(algorithm::AMIS, prepared_target)
    proposal = algorithm.proposal
    binding_sample = proposal.location isa _NativeGaussianFloat ?
                     zero(proposal.location) : view(proposal.location, :)
    target = _bind_resolved_target(prepared_target, binding_sample)
    log_type = _resolve_native_logweight_type(
        target,
        proposal,
        typeof(binding_sample),
    )
    return _prepare_amis_state(algorithm, log_type)
end

function _allocate_random_buffers(
    ::MLDataDevices.AbstractDevice,
    ::_GaussianProposal,
    method_state::_PreparedAMIS,
    sample_budget,
)
    prototype = method_state.workspace.samples
    T = eltype(method_state.history.means)
    maximum_round_size = maximum(method_state.schedule)
    dimension = method_state.history isa _AMISScalarHistory ?
                1 : size(method_state.history.means, 1)
    uniform = similar(prototype, T, 0)
    normal = similar(prototype, T, dimension * maximum_round_size)
    failure_scratch = _allocate_native_failure_scratch(
        normal,
        maximum_round_size,
    )
    return _RandomBuffers(uniform, normal, failure_scratch)
end

function _copy_algorithm(device, algorithm::AMIS)
    proposal = _copy_to_device(device, algorithm.proposal)
    _validate_amis_proposal(proposal)
    round_size = algorithm.round_size isa Vector ?
                 copy(algorithm.round_size) : algorithm.round_size
    return AMIS(
        proposal,
        algorithm.rounds,
        round_size,
        _VALIDATED_AMIS_TOKEN,
    )
end

function _copy_accelerator_algorithm(
    device,
    algorithm::AMIS,
    ::_PreparedAMIS,
)
    return deepcopy(algorithm)
end

function _copy_amis_history(device, history::_AMISScalarHistory)
    return _AMISScalarHistory(
        _copy_to_device(device, history.means),
        _copy_to_device(device, history.scales),
        _copy_to_device(device, history.lognormalizers),
    )
end

function _copy_amis_history(device, history::_AMISFactorHistory)
    return _AMISFactorHistory(
        _copy_to_device(device, history.means),
        _copy_to_device(device, history.factors),
        _copy_to_device(device, history.lognormalizers),
    )
end

function _copy_amis_workspace(device, workspace::_AMISWorkspace)
    return _AMISWorkspace(
        _copy_to_device(device, workspace.samples),
        _copy_to_device(device, workspace.logtargets),
        _copy_to_device(device, workspace.lognumerators),
        _copy_to_device(device, workspace.logweights),
        _copy_to_device(device, workspace.normalized_weights),
        _copy_to_device(device, workspace.centered_scaled),
        _copy_to_device(device, workspace.covariance),
    )
end

function _prepare_transferred_method_state(
    device,
    algorithm::AMIS,
    method_state::_PreparedAMIS,
)
    return _PreparedAMIS(
        Tuple(method_state.schedule),
        Tuple(method_state.offsets),
        _copy_to_device(device, method_state.logcounts),
        _copy_amis_history(device, method_state.history),
        _copy_amis_workspace(device, method_state.workspace),
    )
end

_transferred_backend_state(
    algorithm,
    method_state::_PreparedAMIS,
    target,
    random_buffers,
) = (method_state, target, random_buffers)

_prepared_backend_state(sampler, method_state::_PreparedAMIS) = (
    method_state,
    sampler.target,
    sampler.random_buffers,
    sampler.rng,
)

function _preflight_accelerator_method(
    device,
    target,
    algorithm::AMIS,
    method_state::_PreparedAMIS,
    random_buffers::_RandomBuffers,
)
    throw(SamplerDeviceError(device, :accelerator_factorization_unavailable))
end
