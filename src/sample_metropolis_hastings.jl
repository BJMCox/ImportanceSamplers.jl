"""
    SampleMetropolisHastings(proposal; moves=1)

Configure an interacting Sample Metropolis-Hastings transition for a population
of LAIS centres. `proposal` is a fixed independent native Gaussian or Student-t
proposal. Each move draws one candidate, selects one old centre with probability
proportional to `proposal(x) / target(x)`, and applies the population acceptance
probability `S / (S + r_candidate - min(r_candidate, r_1, ..., r_N))`, where
`r_i = proposal(x_i) / target(x_i)` and `S = sum(r_i)` over the old population.
Set `moves` explicitly to perform an ordered sweep before each lower LAIS round.
One move is the I²-MAIS paper default; `moves=N` means `N` ordered replacement
attempts, not one guaranteed update of every centre.

The upper proposal and centre bank must have the same scalar/vector layout,
dimension, and `Float32` or `Float64` precision. Initial centres require finite
target and proposal log densities. Candidates outside target support are rejected.
Successful calls retain centres and cached ratios; retargeting starts a fresh
target cache, and failed LAIS calls preserve the pre-call committed state without
restoring the RNG.
"""
struct SampleMetropolisHastings{P} <: AbstractMCMCTransition
    proposal::P
    moves::Int

    function SampleMetropolisHastings(proposal; moves=1)
        prepared = _prepare_proposal_input(proposal)
        prepared isa _NativeRadialProposal || throw(ArgumentError(
            "SampleMetropolisHastings requires a native Gaussian or Student-t proposal"))
        moves isa Integer && !(moves isa Bool) && moves > 0 || throw(
            ArgumentError("SampleMetropolisHastings moves must be a positive integer"))
        return new{typeof(prepared)}(prepared, Int(moves))
    end
end

mutable struct _SampleMetropolisHastingsState{P,C,R,S,W,N,U,D,Q,A,E}
    proposal::P
    centres::C
    logratios::R
    candidates::S
    candidate_logweights::W
    normals::N
    proposal_uniforms::U
    density_scratch::D
    decision_uniforms::Q
    accepted::A
    failure_scratch::E
    moves::Int
    capacity::Int
    cache_valid::Bool
    initial_evaluations::Int
    production_evaluations::Int
end

function _smh_scratch_capacity(proposal, centres, ::Type{L}, moves) where {L}
    T = eltype(centres)
    dimension = _transition_dimension(centres)
    bytes_per_move = sizeof(T) * (
        dimension + _native_normal_count(proposal, 1) +
        _native_uniform_count(proposal, 1) + 2) + sizeof(L)
    return min(moves, 256, max(1, (1 << 20) ÷ bytes_per_move))
end

function _allocate_smh_scratch(proposal, centres, ::Type{L}, capacity) where {L}
    T = eltype(centres)
    count = _transition_count(centres)
    dimension = _transition_dimension(centres)
    normals = similar(centres, T, _native_normal_count(proposal, capacity))
    proposal_uniforms = similar(
        centres, T, _native_uniform_count(proposal, capacity))
    candidates = _allocate_native_samples(normals, proposal, capacity)
    candidate_logweights = similar(centres, L, capacity)
    density_scratch = similar(centres, T, dimension * count)
    decision_uniforms = similar(centres, T, 2, capacity)
    failure_scratch = _allocate_native_failure_scratch(normals, max(count, capacity))
    return (; candidates, candidate_logweights, normals, proposal_uniforms,
        density_scratch, decision_uniforms, failure_scratch)
end

function _validate_smh_layout(proposal, centres)
    location = proposal.location
    T = eltype(centres)
    _native_fused_float_type(proposal) === T || throw(ArgumentError(
        "upper proposal and LAIS centres must use the same floating precision"))
    if centres isa AbstractVector
        location isa _NativeGaussianFloat || throw(DimensionMismatch(
            "a scalar LAIS bank requires a scalar upper proposal"))
    else
        location isa AbstractVector || throw(DimensionMismatch(
            "a vector LAIS bank requires a vector upper proposal"))
        length(location) == size(centres, 1) || throw(DimensionMismatch(
            "upper proposal and LAIS centres must have the same dimension"))
    end
    return nothing
end

function prepare_transition(transition::SampleMetropolisHastings, centres, target,
    ::Type{L}) where {L}
    proposal = transition.proposal
    _validate_smh_layout(proposal, centres)
    capacity = _smh_scratch_capacity(proposal, centres, L, transition.moves)
    scratch = _allocate_smh_scratch(proposal, centres, L, capacity)
    return _SampleMetropolisHastingsState(
        proposal, centres, zeros(L, _transition_count(centres)),
        scratch.candidates, scratch.candidate_logweights, scratch.normals,
        scratch.proposal_uniforms, scratch.density_scratch,
        scratch.decision_uniforms, zeros(Int, 1), scratch.failure_scratch,
        transition.moves, capacity, false, 0, 0,
    )
end

transition_centres(state::_SampleMetropolisHastingsState) = state.centres

function Base.copyto!(destination::_SampleMetropolisHastingsState,
    source::_SampleMetropolisHastingsState)
    copyto!(destination.centres, source.centres)
    copyto!(destination.logratios, source.logratios)
    copyto!(destination.accepted, source.accepted)
    destination.cache_valid = source.cache_valid
    destination.initial_evaluations = source.initial_evaluations
    destination.production_evaluations = source.production_evaluations
    return destination
end

function retarget_transition(::SampleMetropolisHastings,
    state::_SampleMetropolisHastingsState, destination)
    proposal = _copy_to_device(destination, state.proposal)
    return SampleMetropolisHastings(proposal; moves=state.moves)
end

function transition_diagnostics(state::_SampleMetropolisHastingsState)
    return (
        initial_target_evaluations=state.initial_evaluations,
        warmup_target_evaluations=0,
        production_target_evaluations=state.production_evaluations,
        warmup_proposals=0,
        production_proposals=state.production_evaluations,
        accepted=sum(state.accepted),
    )
end

function transition_diagnostics(state::_SampleMetropolisHastingsState, transfers)
    counts = transition_diagnostics(state)
    _record_device_scalar_transfer!(transfers, state.accepted, eltype(state.accepted))
    return counts
end

function Adapt.adapt_structure(to, state::_SampleMetropolisHastingsState)
    proposal = Adapt.adapt(to, state.proposal)
    centres = Adapt.adapt(to, state.centres)
    logratios = Adapt.adapt(to, state.logratios)
    accepted = Adapt.adapt(to, state.accepted)
    scratch = _allocate_smh_scratch(
        proposal, centres, eltype(logratios), state.capacity)
    return _SampleMetropolisHastingsState(
        proposal, centres, logratios, scratch.candidates,
        scratch.candidate_logweights, scratch.normals,
        scratch.proposal_uniforms, scratch.density_scratch,
        scratch.decision_uniforms, accepted, scratch.failure_scratch,
        state.moves, state.capacity, state.cache_valid,
        state.initial_evaluations, state.production_evaluations,
    )
end

@kernel function _smh_cache_kernel!(centres, logratios, scratch, evaluator,
    proposal, failures)
    slot = @index(Global, Linear)
    if slot <= length(logratios)
        dimension = _transition_dimension(centres)
        offset = (slot - 1) * dimension + 1
        target_log, target_reason, target_failed =
            evaluator(_native_sample_at(centres, slot), slot)
        if target_failed
            iszero(target_reason) ||
                _record_native_failure!(failures, slot, 0, target_reason)
        else
            proposal_log, proposal_reason = _native_generated_logdensity(
                proposal, _NoSampleTransform(), _native_sample_at(centres, slot),
                scratch, offset)
            ratio = proposal_log - target_log
            reason = !isfinite(target_log) ? _NATIVE_LOGWEIGHT_INVALID :
                     (!iszero(proposal_reason) || !isfinite(proposal_log)) ?
                     _NATIVE_PROPOSAL_INVALID :
                     !isfinite(ratio) ? _NATIVE_LOGWEIGHT_INVALID : UInt16(0)
            if iszero(reason)
                logratios[slot] = ratio
            else
                _record_native_failure!(failures, slot, 0, reason)
            end
        end
    end
end

@inline function _smh_logaddexp(left, right)
    largest = max(left, right)
    return largest + log1p(exp(min(left, right) - largest))
end

@inline _smh_valid_uniform(uniform) =
    isfinite(uniform) && zero(uniform) <= uniform < one(uniform)

@inline function _smh_scaled_ratio(logratio, old_maximum)
    difference = logratio - old_maximum
    return isfinite(difference) ?
           (exp(difference), UInt16(0)) :
           (zero(difference), _NATIVE_LOGWEIGHT_INVALID)
end

function _smh_population_summary(logratios)
    old_maximum = maximum(logratios)
    old_minimum = minimum(logratios)
    scaled_sum = zero(eltype(logratios))
    for logratio in logratios
        scaled, status = _smh_scaled_ratio(logratio, old_maximum)
        iszero(status) || throw(DomainError(
            (logratio=logratio, maximum=old_maximum),
            "finite SMH log-ratio subtraction overflowed"))
        scaled_sum += scaled
    end
    isfinite(scaled_sum) && scaled_sum > zero(scaled_sum) || throw(DomainError(
        scaled_sum, "SMH old-population ratio sum is invalid"))
    return old_maximum, old_minimum, scaled_sum
end

function _smh_selected_slot(logratios, old_maximum, scaled_sum, uniform)
    threshold = uniform * scaled_sum
    cumulative = zero(scaled_sum)
    for slot in eachindex(logratios)
        scaled, status = _smh_scaled_ratio(logratios[slot], old_maximum)
        iszero(status) || return 0, status
        cumulative += scaled
        threshold < cumulative && return slot, UInt16(0)
    end
    return lastindex(logratios), UInt16(0)
end

function _smh_logacceptance(old_maximum, old_minimum, scaled_sum, candidate_logratio)
    candidate_logratio <= old_minimum &&
        return zero(candidate_logratio), UInt16(0)
    candidate_offset = candidate_logratio - old_maximum
    minimum_offset = old_minimum - candidate_logratio
    isfinite(candidate_offset) && isfinite(minimum_offset) ||
        return zero(candidate_logratio), _NATIVE_LOGWEIGHT_INVALID
    log_numerator = log(scaled_sum)
    log_difference = candidate_offset + log(-expm1(minimum_offset))
    (isnan(log_difference) || log_difference == Inf) &&
        return zero(candidate_logratio), _NATIVE_LOGWEIGHT_INVALID
    logacceptance = log_numerator -
                    _smh_logaddexp(log_numerator, log_difference)
    (isnan(logacceptance) || logacceptance > zero(logacceptance)) &&
        return zero(candidate_logratio), _NATIVE_LOGWEIGHT_INVALID
    return logacceptance, UInt16(0)
end

function _smh_ordered_cpu!(state, steps)
    dimension = _transition_dimension(state.centres)
    for move in 1:steps
        candidate_logweight = state.candidate_logweights[move]
        candidate_logweight == -Inf && continue
        isfinite(candidate_logweight) || throw(DomainError(
            candidate_logweight, "SMH candidate log ratio is invalid"))
        old_maximum, old_minimum, scaled_sum =
            _smh_population_summary(state.logratios)
        selection_uniform = state.decision_uniforms[1, move]
        _smh_valid_uniform(selection_uniform) || throw(DomainError(
            selection_uniform, "SMH selection uniforms must lie in [0, 1)"))
        selected, status = _smh_selected_slot(
            state.logratios, old_maximum, scaled_sum, selection_uniform)
        iszero(status) || throw(DomainError(
            (maximum=old_maximum, sum=scaled_sum),
            "finite SMH categorical subtraction overflowed"))
        candidate_logratio = -candidate_logweight
        logacceptance, status = _smh_logacceptance(
            old_maximum, old_minimum, scaled_sum, candidate_logratio)
        iszero(status) || throw(DomainError(
            (candidate=candidate_logratio, maximum=old_maximum,
             minimum=old_minimum), "SMH acceptance arithmetic is invalid"))
        acceptance_uniform = state.decision_uniforms[2, move]
        _smh_valid_uniform(acceptance_uniform) || throw(DomainError(
            acceptance_uniform, "SMH acceptance uniforms must lie in [0, 1)"))
        if log(acceptance_uniform) < logacceptance
            for coordinate in 1:dimension
                _store_population_location!(state.centres, coordinate, selected,
                    _population_sample_coordinate(state.candidates, coordinate, move))
            end
            state.logratios[selected] = candidate_logratio
            state.accepted[1] += 1
        end
    end
    return nothing
end

@kernel function _smh_ordered_kernel!(centres, logratios, candidates,
    candidate_logweights, decision_uniforms, accepted, failures, steps)
    lane = @index(Local, Linear)
    maxima = @localmem eltype(logratios) (_LOCAL_REDUCTION_WORKGROUP_SIZE,)
    minima = @localmem eltype(logratios) (_LOCAL_REDUCTION_WORKGROUP_SIZE,)
    sums = @localmem eltype(logratios) (_LOCAL_REDUCTION_WORKGROUP_SIZE,)
    invalid = @localmem Bool (_LOCAL_REDUCTION_WORKGROUP_SIZE,)
    status = @localmem UInt16 (1,)
    selected = @localmem Int (1,)
    @uniform lanes = @groupsize()[1]
    dimension = _transition_dimension(centres)
    for move in 1:steps
        if lane == 1
            status[1] = 0
            selected[1] = 0
        end
        @synchronize()
        candidate_logweight = candidate_logweights[move]
        candidate_logweight == -Inf && continue
        local_maximum = eltype(logratios)(-Inf)
        local_minimum = eltype(logratios)(Inf)
        for slot in lane:lanes:length(logratios)
            value = logratios[slot]
            local_maximum = max(local_maximum, value)
            local_minimum = min(local_minimum, value)
        end
        maxima[lane] = local_maximum
        minima[lane] = local_minimum
        @synchronize()
        for offset in _LOCAL_REDUCTION_OFFSETS
            if offset < lanes
                if lane <= offset
                    maxima[lane] = max(maxima[lane], maxima[lane + offset])
                    minima[lane] = min(minima[lane], minima[lane + offset])
                end
                @synchronize()
            end
        end
        local_sum = zero(eltype(logratios))
        local_invalid = !isfinite(candidate_logweight)
        for slot in lane:lanes:length(logratios)
            scaled, scaled_status = _smh_scaled_ratio(logratios[slot], maxima[1])
            local_invalid |= !iszero(scaled_status)
            local_sum += scaled
        end
        sums[lane] = local_sum
        invalid[lane] = local_invalid
        @synchronize()
        for offset in _LOCAL_REDUCTION_OFFSETS
            if offset < lanes
                if lane <= offset
                    sums[lane] += sums[lane + offset]
                    invalid[lane] |= invalid[lane + offset]
                end
                @synchronize()
            end
        end
        if lane == 1
            selection_uniform = decision_uniforms[1, move]
            acceptance_uniform = decision_uniforms[2, move]
            if invalid[1] || !(isfinite(sums[1]) && sums[1] > zero(sums[1])) ||
               !_smh_valid_uniform(selection_uniform) ||
               !_smh_valid_uniform(acceptance_uniform)
                status[1] = _NATIVE_LOGWEIGHT_INVALID
            else
                selected[1], status[1] = _smh_selected_slot(
                    logratios, maxima[1], sums[1], selection_uniform)
                candidate_logratio = -candidate_logweight
                logacceptance, acceptance_status = _smh_logacceptance(
                    maxima[1], minima[1], sums[1], candidate_logratio)
                status[1] |= acceptance_status
                if iszero(status[1]) && log(acceptance_uniform) < logacceptance
                    logratios[selected[1]] = candidate_logratio
                    accepted[1] += 1
                else
                    selected[1] = 0
                end
            end
            iszero(status[1]) ||
                _record_native_failure!(failures, move, 0, status[1])
        end
        @synchronize()
        if selected[1] > 0
            for coordinate in lane:lanes:dimension
                _store_population_location!(centres, coordinate, selected[1],
                    _population_sample_coordinate(candidates, coordinate, move))
            end
        end
        @synchronize()
        iszero(status[1]) || break
    end
end

function _check_smh_failures!(state, transfers)
    snapshot = _device_failure_snapshot(state.failure_scratch.record)
    _record_reported_transfer!(transfers, snapshot.transfers.count,
        snapshot.transfers.bytes, Val(:failure_snapshot))
    _throw_native_failures(snapshot.failure, snapshot.draw_failure,
        state.failure_scratch.target_failures, _NoSampleTransform())
    return nothing
end

function _initialize_smh_cache!(state, target, execution, transfers)
    state.cache_valid && return nothing
    backend = KernelAbstractions.get_backend(state.normals)
    evaluator, _ = _native_target_evaluator(
        backend, target, eltype(state.logratios),
        state.failure_scratch.target_failures)
    _reset_native_failure_scratch!(state.failure_scratch)
    kernel = _smh_cache_kernel!(backend)
    count = length(state.logratios)
    kernel(state.centres, state.logratios, state.density_scratch, evaluator,
        state.proposal, state.failure_scratch.record.storage;
        ndrange=count, workgroupsize=_native_workgroupsize(execution, count))
    KernelAbstractions.synchronize(backend)
    _check_smh_failures!(state, transfers)
    state.cache_valid = true
    state.initial_evaluations += count
    return nothing
end

function _smh_candidate_batch!(state, target, rng, execution, transfers, steps)
    normal_count = _native_normal_count(state.proposal, steps)
    uniform_count = _native_uniform_count(state.proposal, steps)
    normal_count > 0 && Random.randn!(rng, view(state.normals, 1:normal_count))
    uniform_count > 0 &&
        Random.rand!(rng, view(state.proposal_uniforms, 1:uniform_count))
    Random.rand!(rng, view(state.decision_uniforms, :, 1:steps))
    backend = KernelAbstractions.get_backend(state.normals)
    evaluator, _ = _native_target_evaluator(
        backend, target, eltype(state.candidate_logweights),
        state.failure_scratch.target_failures)
    _reset_native_failure_scratch!(state.failure_scratch)
    _launch_native_fused!(state.candidates,
        view(state.candidate_logweights, 1:steps), state.failure_scratch.record,
        view(state.proposal_uniforms, 1:uniform_count),
        view(state.normals, 1:normal_count), evaluator, state.proposal,
        _NoSampleTransform(), execution)
    _check_smh_failures!(state, transfers)
    return nothing
end

function _smh_ordered_batch!(::KernelAbstractions.CPU, state, steps, transfers)
    return _smh_ordered_cpu!(state, steps)
end

function _smh_ordered_batch!(backend, state, steps, transfers)
    _reset_native_failure_scratch!(state.failure_scratch)
    kernel = _smh_ordered_kernel!(backend)
    workload = max(length(state.logratios), _transition_dimension(state.centres))
    lanes = min(_LOCAL_REDUCTION_WORKGROUP_SIZE, nextpow(2, workload))
    kernel(state.centres, state.logratios, state.candidates,
        state.candidate_logweights, state.decision_uniforms, state.accepted,
        state.failure_scratch.record.storage, steps;
        ndrange=lanes, workgroupsize=lanes)
    KernelAbstractions.synchronize(backend)
    _check_smh_failures!(state, transfers)
    return nothing
end

function transition!(state::_SampleMetropolisHastingsState, target, rng,
    execution, transfers)
    _initialize_smh_cache!(state, target, execution, transfers)
    backend = KernelAbstractions.get_backend(state.normals)
    completed = 0
    while completed < state.moves
        steps = min(state.capacity, state.moves - completed)
        _smh_candidate_batch!(state, target, rng, execution, transfers, steps)
        _smh_ordered_batch!(backend, state, steps, transfers)
        state.production_evaluations += steps
        completed += steps
    end
    return nothing
end

function _preflight_transition(device, state::_SampleMetropolisHastingsState, target)
    backend = KernelAbstractions.get_backend(state.normals)
    evaluator = _NativeDeviceTarget{eltype(state.logratios),typeof(target)}(target)
    cache_kernel = _smh_cache_kernel!(backend)
    for argument in (state.centres, state.logratios, state.density_scratch,
        evaluator, state.proposal, state.failure_scratch.record.storage)
        _preflight_kernel_argument(device, cache_kernel, argument)
    end
    fused_kernel = _native_fused_kernel!(backend)
    normal_count = _native_normal_count(state.proposal, state.capacity)
    uniform_count = _native_uniform_count(state.proposal, state.capacity)
    for argument in (state.candidates,
        view(state.candidate_logweights, 1:state.capacity),
        state.failure_scratch.record.storage,
        view(state.proposal_uniforms, 1:uniform_count),
        view(state.normals, 1:normal_count), evaluator, state.proposal,
        _NoSampleTransform())
        _preflight_kernel_argument(device, fused_kernel, argument)
    end
    ordered_kernel = _smh_ordered_kernel!(backend)
    for argument in (state.centres, state.logratios, state.candidates,
        state.candidate_logweights, state.decision_uniforms, state.accepted,
        state.failure_scratch.record.storage, state.capacity)
        _preflight_kernel_argument(device, ordered_kernel, argument)
    end
    return nothing
end
