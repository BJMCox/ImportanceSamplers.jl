"""Abstract supertype for batched MCMC transitions used by [`LAIS`](@ref)."""
abstract type AbstractMCMCTransition end

"""
    RandomWalkMetropolis(covariance)

Configure independent Gaussian random-walk Metropolis chains. `covariance` is
an isotropic variance, an SPD matrix, a Cholesky factorization, or a vector of
these inputs with one entry per chain. Use `Diagonal` for a diagonal covariance.
This covariance is independent of the lower importance proposals.
"""
struct RandomWalkMetropolis{C} <: AbstractMCMCTransition
    covariance::C
end

"""Tune for `steps` upper-only moves, then freeze. Reuse does not repeat completed warmup."""
struct WarmupTuning
    steps::Int
    function WarmupTuning(steps::Integer)
        steps >= 0 || throw(ArgumentError("warmup steps must be nonnegative"))
        return new(steps)
    end
end

"""Tune on each production move, retaining the diminishing schedule across calls."""
struct ContinuousTuning end

"""
    RAM(covariance; tuning, target_acceptance=0.234, decay=0.6)

Robust adaptive Metropolis with Gaussian increments and rank-one covariance
updates. Covariance inputs follow [`RandomWalkMetropolis`](@ref). Choose
[`WarmupTuning`](@ref) or [`ContinuousTuning`](@ref) explicitly. Adaptation uses
the acceptance probability, including on rejected moves, with gain `k^(-decay)`.
Lower importance-proposal covariances stay fixed.
"""
struct RAM{C,S,T,D} <: AbstractMCMCTransition
    covariance::C
    tuning::S
    target_acceptance::T
    decay::D
    function RAM(covariance; tuning::Union{WarmupTuning,ContinuousTuning},
        target_acceptance::Real=0.234, decay::Real=0.6)
        0 < target_acceptance < 1 || throw(ArgumentError("target acceptance must lie in (0, 1)"))
        0.5 < decay < 1 || throw(ArgumentError("RAM decay must lie in (0.5, 1)"))
        return new{typeof(covariance),typeof(tuning),typeof(target_acceptance),typeof(decay)}(
            covariance, tuning, target_acceptance, decay)
    end
end

mutable struct _RAMState{W,S,T,D}
    walk::W
    tuning::S
    target_acceptance::T
    decay::T
    direction::D
    n_tuned::Int
end

mutable struct _RandomWalkState{C,F,L,N,U,A,E}
    centres::C
    factors::F
    logtargets::L
    candidates::C
    normals::N
    uniforms::U
    accepted::A
    failure_scratch::E
    cache_valid::Bool
    initial_evaluations::Int
    steps::Int
end

_transition_dimension(::AbstractVector) = 1
_transition_dimension(centres::AbstractMatrix) = size(centres, 1)
_transition_count(centres::AbstractVector) = length(centres)
_transition_count(centres::AbstractMatrix) = size(centres, 2)

function _transition_factor(covariance::Real, ::Type{T}, dimension) where {T}
    variance = T(covariance)
    isfinite(variance) && variance > zero(T) || throw(
        ArgumentError("transition variance must be finite and positive"),
    )
    return Matrix(LinearAlgebra.Diagonal(fill(sqrt(variance), dimension)))
end

function _transition_factor(covariance::AbstractMatrix, ::Type{T}, dimension) where {T}
    size(covariance) == (dimension, dimension) || throw(
        DimensionMismatch("transition covariance must match the sample dimension"),
    )
    matrix = Matrix{T}(covariance)
    all(isfinite, matrix) && LinearAlgebra.issymmetric(matrix) || throw(
        ArgumentError("transition covariance must be finite and symmetric"),
    )
    return Matrix(LinearAlgebra.cholesky(LinearAlgebra.Symmetric(matrix)).L)
end

function _transition_factor(covariance::LinearAlgebra.Cholesky, ::Type{T}, dimension) where {T}
    size(covariance) == (dimension, dimension) || throw(
        DimensionMismatch("transition factor must match the sample dimension"),
    )
    factor = Matrix{T}(covariance.L)
    LinearAlgebra.issuccess(covariance) && all(isfinite, factor) &&
        all(>(zero(T)), LinearAlgebra.diag(factor)) || throw(
        ArgumentError("transition factor must be finite and positive definite"),
    )
    return factor
end

"""Prepare typed transition state whose centres alias the supplied sample batch."""
function prepare_transition(transition::RandomWalkMetropolis, centres, target, ::Type{L}) where {L}
    T = eltype(centres)
    dimension = _transition_dimension(centres)
    count = _transition_count(centres)
    covariance = transition.covariance
    factors = Array{T}(undef, dimension, dimension, count)
    if covariance isa AbstractVector
        length(covariance) == count || throw(
            DimensionMismatch("provide one transition covariance per chain"),
        )
        for chain in 1:count
            copyto!(view(factors, :, :, chain), _transition_factor(covariance[chain], T, dimension))
        end
    else
        factor = _transition_factor(covariance, T, dimension)
        for chain in 1:count
            copyto!(view(factors, :, :, chain), factor)
        end
    end
    return _RandomWalkState(
        centres, factors, zeros(L, count), similar(centres),
        Matrix{T}(undef, dimension, count), Vector{T}(undef, count),
        zeros(Int, count), _allocate_native_failure_scratch(centres, count),
        false, 0, 0,
    )
end

"""Return the transition's centre batch without copying it."""
transition_centres(state::_RandomWalkState) = state.centres
transition_centres(state::_RAMState) = transition_centres(state.walk)

function prepare_transition(transition::RAM, centres, target, ::Type{L}) where {L}
    walk = prepare_transition(RandomWalkMetropolis(transition.covariance), centres, target, L)
    T = eltype(centres)
    acceptance, decay = T(transition.target_acceptance), T(transition.decay)
    zero(T) < acceptance < one(T) && T(0.5) < decay < one(T) || throw(
        ArgumentError("RAM controls must remain inside their bounds in the sample precision"))
    return _RAMState(walk, transition.tuning, acceptance, decay, similar(walk.normals), 0)
end

function Base.copyto!(destination::_RAMState, source::_RAMState)
    _copy_transition_arrays!(destination, source)
    _copy_transition_counters!(destination, source)
    return destination
end

_copy_transition_arrays!(destination::_RAMState, source::_RAMState) =
    _copy_transition_arrays!(destination.walk, source.walk)
function _copy_transition_counters!(destination::_RAMState, source::_RAMState)
    _copy_transition_counters!(destination.walk, source.walk)
    destination.n_tuned = source.n_tuned
    return destination
end

function _retarget_factors(destination, state::_RandomWalkState)
    factors = destination(Array(state.factors))
    return [LinearAlgebra.Cholesky(copy(view(factors, :, :, chain)), 'L', 0)
            for chain in axes(factors, 3)]
end

"""Build a fresh transition configuration from learned factors; reset caches and tuning on preparation."""
retarget_transition(::RandomWalkMetropolis, state::_RandomWalkState, destination) =
    RandomWalkMetropolis(_retarget_factors(destination, state))

function retarget_transition(::RAM, state::_RAMState, destination)
    return RAM(_retarget_factors(destination, state.walk);
        tuning=state.tuning, target_acceptance=state.target_acceptance, decay=state.decay)
end

function Base.copyto!(destination::_RandomWalkState, source::_RandomWalkState)
    _copy_transition_arrays!(destination, source)
    _copy_transition_counters!(destination, source)
    return destination
end

function _copy_transition_arrays!(destination::_RandomWalkState, source::_RandomWalkState)
    copyto!(destination.centres, source.centres)
    copyto!(destination.factors, source.factors)
    copyto!(destination.logtargets, source.logtargets)
    copyto!(destination.accepted, source.accepted)
    return destination
end

function _copy_transition_counters!(destination::_RandomWalkState, source::_RandomWalkState)
    destination.cache_valid = source.cache_valid
    destination.initial_evaluations = source.initial_evaluations
    destination.steps = source.steps
    return destination
end

"""Return transition work and acceptance counts without exposing implementation state."""
transition_diagnostics(state::_RandomWalkState) = _walk_diagnostics(state, sum(state.accepted))
_walk_diagnostics(state, accepted) = (
    initial_target_evaluations=state.initial_evaluations,
    warmup_target_evaluations=0,
    production_target_evaluations=state.steps * length(state.logtargets),
    warmup_proposals=0,
    production_proposals=state.steps * length(state.logtargets),
    accepted=accepted,
)

_transition_warmup(state::_RAMState{W,<:WarmupTuning}) where {W} = state.n_tuned
_transition_warmup(state::_RAMState) = 0

function transition_diagnostics(state::_RAMState)
    return _ram_diagnostics(state, transition_diagnostics(state.walk))
end

function _ram_diagnostics(state::_RAMState, counts)
    warmup = _transition_warmup(state) * length(state.walk.logtargets)
    return merge(counts, (
        warmup_target_evaluations=warmup,
        warmup_proposals=warmup,
        production_target_evaluations=counts.production_target_evaluations - warmup,
        production_proposals=counts.production_proposals - warmup,
    ))
end

transition_diagnostics(state, transfers) = transition_diagnostics(state)
function transition_diagnostics(state::_RandomWalkState, transfers)
    counts = transition_diagnostics(state)
    _record_device_scalar_transfer!(transfers, state.accepted, eltype(state.accepted))
    return counts
end
transition_diagnostics(state::_RAMState, transfers) =
    _ram_diagnostics(state, transition_diagnostics(state.walk, transfers))

function Adapt.adapt_structure(to, state::_RandomWalkState)
    normals = Adapt.adapt(to, state.normals)
    return _RandomWalkState(
        Adapt.adapt(to, state.centres), Adapt.adapt(to, state.factors),
        Adapt.adapt(to, state.logtargets), Adapt.adapt(to, state.candidates),
        normals, Adapt.adapt(to, state.uniforms), Adapt.adapt(to, state.accepted),
        _allocate_native_failure_scratch(normals, length(state.logtargets)),
        state.cache_valid, state.initial_evaluations, state.steps,
    )
end

function Adapt.adapt_structure(to, state::_RAMState)
    walk = Adapt.adapt(to, state.walk)
    T = eltype(walk.normals)
    acceptance, decay = T(state.target_acceptance), T(state.decay)
    zero(T) < acceptance < one(T) && T(0.5) < decay < one(T) || throw(
        ArgumentError("RAM controls must remain inside their bounds in the sample precision"))
    return _RAMState(walk, state.tuning, acceptance, decay, similar(walk.normals), state.n_tuned)
end

_transition_foreach(f, indices, ::_SerialCPUExecution) = foreach(f, indices)
_transition_foreach(f, indices, ::_ThreadedCPUExecution) = _threaded_foreach(f, indices)

"""Advance the chain batch once, using coordinator-owned bulk random draws."""
transition!(state::_RandomWalkState, target, rng, execution, transfers) =
    _transition_batch!(state, target, rng, execution, transfers, nothing)

function transition!(state::_RAMState{W,<:WarmupTuning}, target, rng, execution, transfers) where {W}
    backend = KernelAbstractions.get_backend(state.walk.normals)
    return _warmup_transition!(backend, state, target, rng, execution, transfers)
end

function _warmup_transition!(::KernelAbstractions.CPU, state, target, rng, execution, transfers)
    while state.n_tuned < state.tuning.steps
        _transition_batch!(state.walk, target, rng, execution, transfers, state)
        state.n_tuned += 1
    end
    return transition!(state.walk, target, rng, execution, transfers)
end

function transition!(state::_RAMState{W,ContinuousTuning}, target, rng, execution, transfers) where {W}
    _transition_batch!(state.walk, target, rng, execution, transfers, state)
    state.n_tuned += 1
    return nothing
end

_transition_adaptation(::Nothing) = nothing
function _transition_adaptation(state::_RAMState)
    T = eltype(state.walk.normals)
    return (direction=state.direction, target_acceptance=state.target_acceptance,
        gain=T(state.n_tuned + 1)^(-state.decay))
end

_adapt_transition!(::Nothing, batch, chain, alpha) = true

function _adapt_transition!(update, batch, chain, alpha)
    T = eltype(batch.normals)
    dimension = size(batch.normals, 1)
    norm = zero(T)
    for row in 1:dimension
        norm = hypot(norm, batch.normals[row, chain])
    end
    isfinite(norm) && norm > zero(T) || return false
    coefficient = update.gain * (T(alpha) - update.target_acceptance)
    for row in 1:dimension
        value = zero(T)
        for column in 1:row
            value += batch.factors[row, column, chain] * (batch.normals[column, chain] / norm)
        end
        update.direction[row, chain] = sqrt(abs(coefficient)) * value
    end
    return _transition_rankone!(batch.factors, update.direction, chain, coefficient < zero(T))
end

function _transition_rankone_diagonal(diagonal, x, downdate)
    ratio = x / diagonal
    if downdate
        abs(ratio) < one(ratio) || return oftype(diagonal, NaN), ratio
        next = diagonal * sqrt((one(ratio) - ratio) * (one(ratio) + ratio))
    else
        next = hypot(diagonal, x)
    end
    return next, ratio
end

function _transition_rankone_entry(factor, value, c, ratio, downdate)
    signed = downdate ? -ratio * value : ratio * value
    updated = (factor + signed) / c
    return updated, c * value - ratio * updated
end

# Lower-Cholesky rank-one update/downdate. Scratch is consumed, not allocated.
function _transition_rankone!(factors, direction, chain, downdate)
    for column in axes(direction, 1)
        diagonal = factors[column, column, chain]
        x = direction[column, chain]
        next, ratio = _transition_rankone_diagonal(diagonal, x, downdate)
        isfinite(next) && next > zero(next) || return false
        c = next / diagonal
        factors[column, column, chain] = next
        for row in (column + 1):size(direction, 1)
            value = direction[row, chain]
            factor, next_direction = _transition_rankone_entry(
                factors[row, column, chain], value, c, ratio, downdate)
            isfinite(factor) || return false
            factors[row, column, chain] = factor
            direction[row, chain] = next_direction
        end
    end
    return true
end

_transition_arrays(state::_RandomWalkState) = (
    centres=state.centres, factors=state.factors, logtargets=state.logtargets,
    candidates=state.candidates, normals=state.normals, uniforms=state.uniforms,
    accepted=state.accepted,
)

function _transition_step!(batch, evaluator, update, chain)
    dimension = size(batch.normals, 1)
    for row in 1:dimension
        increment = zero(eltype(batch.normals))
        for column in 1:row
            increment += batch.factors[row, column, chain] * batch.normals[column, chain]
        end
        value = _population_sample_coordinate(batch.centres, row, chain) + increment
        isfinite(value) || return _NATIVE_GENERATED_NONFINITE, true
        _store_population_location!(batch.candidates, row, chain, value)
    end
    value, reason, failed = evaluator(_native_sample_at(batch.candidates, chain), chain)
    failed && return reason, true
    logacceptance = min(zero(value), value - batch.logtargets[chain])
    _adapt_transition!(update, batch, chain, exp(logacceptance)) ||
        return _NATIVE_PROPOSAL_INVALID, true
    if log(batch.uniforms[chain]) < logacceptance
        for row in 1:dimension
            _store_population_location!(batch.centres, row, chain,
                _population_sample_coordinate(batch.candidates, row, chain))
        end
        batch.logtargets[chain] = value
        batch.accepted[chain] += 1
    end
    return UInt16(0), false
end

function _transition_batch!(state::_RandomWalkState, target, rng, execution, transfers, adaptation)
    backend = KernelAbstractions.get_backend(state.normals)
    return _transition_batch!(backend, state, target, rng, execution, transfers,
        _transition_adaptation(adaptation))
end

function _transition_batch!(::KernelAbstractions.CPU, state, target, rng, execution, transfers, update)
    count = length(state.logtargets)
    evaluator, failures = _native_target_evaluator(
        KernelAbstractions.CPU(), target, eltype(state.logtargets), state.failure_scratch.target_failures,
    )
    _reset_native_target_failures!(failures)
    if !state.cache_valid
        _transition_foreach(1:count, execution) do chain
            value, _, _ = evaluator(_native_sample_at(state.centres, chain), chain)
            state.logtargets[chain] = value
        end
        failure = _take_first_native_target_failure!(failures)
        isnothing(failure) || throw(failure)
        invalid = findfirst(!isfinite, state.logtargets)
        isnothing(invalid) || throw(DomainError(
            (chain=invalid, logtarget=state.logtargets[invalid]),
            "initial chain log targets must be finite"))
        state.cache_valid = true
        state.initial_evaluations += count
    end
    Random.randn!(rng, state.normals)
    Random.rand!(rng, state.uniforms)
    batch = _transition_arrays(state)
    _transition_foreach(1:count, execution) do chain
        try
            reason, failed = _transition_step!(batch, evaluator, update, chain)
            failed && !iszero(reason) && throw(DomainError(reason,
                "transition candidate or RAM factor update is invalid"))
        catch cause
            _record_cpu_failure!(failures, chain, :transition, cause, catch_backtrace())
        end
    end
    failure = _take_first_native_target_failure!(failures)
    isnothing(failure) || throw(failure)
    state.steps += 1
    return nothing
end
