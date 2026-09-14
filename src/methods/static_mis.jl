struct _StratifiedAssignment end
struct _RandomAssignment end
struct _FullMixtureDenominator end
struct _GeneratingDenominator end
struct _EqualAllocationGeneratingDenominator end

struct _PartialMixtureDenominator{G,O,I,C}
    group_of_slot::G
    offsets::O
    members::I
    logcoefficients::C
end

Adapt.@adapt_structure _PartialMixtureDenominator

struct _PreparedMISDesign{A,D}
    assignment::A
    denominator::D
end

Adapt.@adapt_structure _PreparedMISDesign

struct _PreparedStaticMIS{B,D}
    bank::B
    design::D
end

Adapt.@adapt_structure _PreparedStaticMIS

_accelerator_method_state_limit(::_PreparedStaticMIS) = :generic_proposal_cpu_only

struct _StaticMISRandomBuffers{A}
    assignments::A
end

struct _StaticMISDenominatorEvaluator{T,B,A,D}
    bank::B
    assignments::A
    denominator::D
end

function _StaticMISDenominatorEvaluator(
    ::Type{T},
    bank::B,
    assignments::A,
    denominator::D,
) where {T,B,A,D}
    return _StaticMISDenominatorEvaluator{T,B,A,D}(
        bank,
        assignments,
        denominator,
    )
end

function _prepare_static_mis_bank(bank::ProposalBank)
    proposal_type = eltype(bank.proposals)
    native_candidate = Val(
        proposal_type <: _NativeRadialProposal || !isconcretetype(proposal_type),
    )
    return _prepare_static_mis_bank(
        bank,
        native_candidate,
    )
end

function _prepare_static_mis_bank(bank, ::Val{true})
    return _prepare_static_mis_bank(
        bank,
        _native_radial_pack_kind(eltype(bank.proposals)),
    )
end

_prepare_static_mis_bank(bank, ::Val{false}) =
    _prepare_active_proposal_bank(bank)

_prepare_static_mis_bank(bank, ::Val{:diagonal}) =
    _prepare_active_proposal_bank(bank)

function _prepare_static_mis_bank(bank, ::Val{:factor})
    proposal_ids, logmasses, cdf = _prepare_active_proposal_metadata(bank)
    packed = _pack_native_radial_bank(
        bank,
        proposal_ids,
        logmasses,
        cdf,
        Val(:factor),
    )
    isnothing(packed) || return packed
    return _prepare_generic_active_proposal_bank(
        bank,
        proposal_ids,
        logmasses,
        cdf,
    )
end

function _prepare_static_mis_bank(bank, ::Val{:dynamic})
    packed = _pack_native_radial_bank(bank)
    isnothing(packed) || return packed
    return _prepare_active_proposal_bank(bank)
end

function _prepare_method_state(
    algorithm::ImportanceSampling{<:ProposalBank,<:StratifiedMixture},
)
    bank = _prepare_static_mis_bank(algorithm.proposal)
    design = _PreparedMISDesign(
        _StratifiedAssignment(),
        _FullMixtureDenominator(),
    )
    return _PreparedStaticMIS(bank, design)
end

function _prepare_method_state(
    algorithm::ImportanceSampling{<:ProposalBank,<:RandomMixture},
)
    bank = _prepare_static_mis_bank(algorithm.proposal)
    design = _PreparedMISDesign(
        _RandomAssignment(),
        _FullMixtureDenominator(),
    )
    return _PreparedStaticMIS(bank, design)
end

function _prepare_method_state(
    algorithm::ImportanceSampling{<:ProposalBank,<:StandardMIS},
)
    bank = _prepare_static_mis_bank(algorithm.proposal)
    design = _PreparedMISDesign(
        _StratifiedAssignment(),
        _GeneratingDenominator(),
    )
    return _PreparedStaticMIS(bank, design)
end

function _prepare_method_state(
    algorithm::ImportanceSampling{<:ProposalBank,<:PartialDeterministicMixture},
)
    bank = _prepare_static_mis_bank(algorithm.proposal)
    denominator = _compile_partial_denominator(
        algorithm.proposal,
        bank,
        algorithm.mis_scheme.groups,
    )
    design = _PreparedMISDesign(_StratifiedAssignment(), denominator)
    return _PreparedStaticMIS(bank, design)
end

function _prepare_method_state(
    algorithm::ImportanceSampling{<:ProposalBank,<:AbstractMISScheme},
)
    throw(
        ArgumentError(
            "$(typeof(algorithm.mis_scheme)) static-MIS execution is not implemented",
        ),
    )
end

function _allocate_random_buffers(
    ::MLDataDevices.AbstractCPUDevice,
    ::ProposalBank,
    ::_PreparedStaticMIS,
    nsamples,
)
    assignments = Vector{Int}(undef, nsamples)
    return _StaticMISRandomBuffers(assignments)
end

_native_failure_scratch(::_StaticMISRandomBuffers) = _NoNativeFailureScratch()

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

@inline function _static_mis_assignment(
    assignment,
    cdf,
    uniform,
    sample_index,
    nsamples,
)
    mapped_uniform = _static_mis_assignment_uniform(
        assignment,
        uniform,
        sample_index,
        nsamples,
    )
    first_slot = firstindex(cdf)
    last_slot = lastindex(cdf)
    while first_slot < last_slot
        middle_slot = first_slot + (last_slot - first_slot) ÷ 2
        if mapped_uniform <= @inbounds(cdf[middle_slot])
            last_slot = middle_slot
        else
            first_slot = middle_slot + 1
        end
    end
    return first_slot
end

function _compile_assignments!(rng, assignments, assignment, cdf)
    nsamples = length(assignments)
    for sample_index in eachindex(assignments)
        assignments[sample_index] = _capture_sampler_failure(
            :proposal_draw,
            sample_index,
        ) do
            _static_mis_assignment(
                assignment,
                cdf,
                rand(rng),
                sample_index,
                nsamples,
            )
        end
    end
    return assignments
end

_mis_scheme_name(::_RandomAssignment, ::_FullMixtureDenominator) =
    :random_mixture
_mis_scheme_name(::_StratifiedAssignment, ::_FullMixtureDenominator) =
    :stratified_mixture
_mis_scheme_name(::_StratifiedAssignment, ::_GeneratingDenominator) =
    :standard_mis
_mis_scheme_name(::_StratifiedAssignment, ::_PartialMixtureDenominator) =
    :partial_deterministic_mixture

function _compile_partial_denominator(bank, active_bank, groups)
    isempty(groups) && throw(ArgumentError("partial-MIS groups cannot be empty"))
    seen = falses(length(bank.proposals))
    active_slot = zeros(Int, length(bank.proposals))
    for (slot, proposal_id) in pairs(active_bank.proposal_ids)
        active_slot[proposal_id] = slot
    end

    group_of_slot = zeros(Int, _active_proposal_count(active_bank))
    offsets = Int[1]
    members = Int[]
    logcoefficients = Vector{eltype(active_bank.logmasses)}()
    active_group = 0
    for group in groups
        isempty(group) && throw(ArgumentError("partial-MIS groups cannot be empty"))
        first_member = length(members) + 1
        for proposal_id in group
            proposal_id isa Integer && proposal_id !== true && proposal_id !== false || throw(
                ArgumentError("partial-MIS proposal IDs must be integers"),
            )
            1 <= proposal_id <= length(seen) || throw(
                ArgumentError("partial-MIS proposal ID $proposal_id is out of range"),
            )
            id = Int(proposal_id)
            seen[id] && throw(
                ArgumentError("partial-MIS proposal ID $id appears more than once"),
            )
            seen[id] = true
            slot = active_slot[id]
            iszero(slot) && continue
            push!(members, slot)
        end
        first_member > length(members) && continue

        active_group += 1
        group_logmass = eltype(active_bank.logmasses)(-Inf)
        for member_index in first_member:length(members)
            slot = members[member_index]
            group_logmass = LogExpFunctions.logaddexp(
                group_logmass,
                active_bank.logmasses[slot],
            )
        end
        for member_index in first_member:length(members)
            slot = members[member_index]
            group_of_slot[slot] = active_group
            push!(
                logcoefficients,
                active_bank.logmasses[slot] - group_logmass,
            )
        end
        push!(offsets, length(members) + 1)
    end
    all(seen) || throw(
        ArgumentError("partial-MIS groups must include every proposal ID exactly once"),
    )
    return _PartialMixtureDenominator(
        group_of_slot,
        offsets,
        members,
        logcoefficients,
    )
end

function _draw_static_mis_batch!(sampler, method_state::_PreparedStaticMIS)
    bank = method_state.bank
    assignments = sampler.random_buffers.assignments
    _compile_assignments!(
        sampler.rng,
        assignments,
        method_state.design.assignment,
        bank.cdf,
    )

    first_slot = assignments[1]
    first_sample = _capture_sampler_failure(:proposal_draw, 1) do
        rand(sampler.rng, bank.proposals[first_slot])
    end
    samples = _capture_sampler_failure(:proposal_draw, 1) do
        _allocate_batch(first_sample, length(assignments))
    end
    _capture_sampler_failure(:proposal_draw, 1) do
        _store_sample!(samples, first_sample, 1)
    end

    for sample_index in 2:length(assignments)
        slot = assignments[sample_index]
        _capture_sampler_failure(:proposal_draw, sample_index) do
            sample = rand(sampler.rng, bank.proposals[slot])
            _store_sample!(samples, sample, sample_index)
        end
    end

    proposal_ids = Vector{Int}(undef, length(assignments))
    for sample_index in eachindex(proposal_ids)
        proposal_ids[sample_index] = bank.proposal_ids[assignments[sample_index]]
    end
    return samples, proposal_ids
end

function _resolve_static_mis_logweight_type(target, bank, samples)
    sample_type = typeof(_sample_at(samples, 1))
    target_type = _capture_sampler_failure(:target, 1) do
        inferred = Base.promote_op(target, sample_type)
        _canonical_inferred_log_type(inferred, "target")
    end
    proposal_type = _capture_sampler_failure(:proposal_logdensity, 1) do
        inferred = Base.promote_op(
            DensityInterface.logdensityof,
            eltype(bank.proposals),
            sample_type,
        )
        _canonical_inferred_log_type(inferred, "proposal")
    end
    mass_type = eltype(bank.logmasses)
    return promote_type(target_type, proposal_type, mass_type)
end

function _phase_logdensity(
    evaluator::_StaticMISDenominatorEvaluator{T},
    sample,
    sample_index,
    ::Val{:proposal_logdensity},
) where {T}
    return _mis_logdenominator(
        T,
        evaluator.bank,
        evaluator.denominator,
        evaluator.assignments[sample_index],
        sample,
        sample_index,
    )
end

@inline function _mis_proposal_logdensity(
    ::Type{T},
    bank::_ActiveProposalBank,
    sample,
    proposal_slot,
) where {T}
    value = DensityInterface.logdensityof(
        @inbounds(bank.proposals[proposal_slot]),
        sample,
    )
    _validate_mixture_proposal_logdensity(value)
    return _convert_static_mis_logdensity(T, value)
end

@inline _mis_term_bounds(bank, ::_FullMixtureDenominator, generating_slot) =
    (firstindex(bank.logmasses), lastindex(bank.logmasses))

@inline function _mis_term_bounds(
    bank,
    denominator::_PartialMixtureDenominator,
    generating_slot,
)
    group = @inbounds denominator.group_of_slot[generating_slot]
    first_member = @inbounds denominator.offsets[group]
    return first_member, @inbounds(denominator.offsets[group + 1]) - 1
end

@inline _mis_term_bounds(bank, ::_GeneratingDenominator, generating_slot) =
    (generating_slot, generating_slot)
@inline _mis_term_bounds(
    bank,
    ::_EqualAllocationGeneratingDenominator,
    generating_slot,
) = (generating_slot, generating_slot)

@inline function _mis_denominator_term(
    ::Type{T},
    bank,
    ::_FullMixtureDenominator,
    term_index,
) where {T}
    return term_index, convert(T, @inbounds(bank.logmasses[term_index]))
end

@inline function _mis_denominator_term(
    ::Type{T},
    bank,
    denominator::_PartialMixtureDenominator,
    term_index,
) where {T}
    return @inbounds(denominator.members[term_index]), convert(
        T,
        @inbounds(denominator.logcoefficients[term_index]),
    )
end

@inline _mis_denominator_term(
    ::Type{T},
    bank,
    ::_GeneratingDenominator,
    term_index,
) where {T} = (term_index, zero(T))

@inline _mis_denominator_term(
    ::Type{T},
    bank,
    ::_EqualAllocationGeneratingDenominator,
    term_index,
) where {T} = (term_index, zero(T))

function _mis_logdenominator(
    ::Type{T},
    bank,
    denominator,
    generating_slot,
    sample,
    sample_index,
) where {T}
    return _capture_sampler_failure(:proposal_logdensity, sample_index) do
        value, generating_logdensity, reason = _mis_logdenominator_core(
            T,
            bank,
            denominator,
            generating_slot,
            sample,
            nothing,
            sample_index,
        )
        if !iszero(reason)
            _validate_generating_logdensity(generating_logdensity)
            _validate_reduced_mis_denominator(value)
        end
        value
    end
end

function _validate_mixture_proposal_logdensity(value)
    (typeof(value) === Float32 || typeof(value) === Float64) || throw(
        ArgumentError("proposal log density must be Float32 or Float64"),
    )
    isnan(value) && throw(
        DomainError(value, "proposal mixture terms may not be NaN"),
    )
    return nothing
end

function _convert_static_mis_logdensity(::Type{T}, value) where {T}
    promote_type(T, typeof(value)) === T || throw(
        ArgumentError(
            "proposal log-density type $(typeof(value)) would narrow when stored as $T",
        ),
    )
    return convert(T, value)
end

function _validate_generating_logdensity(value)
    (isnan(value) || value == -Inf) && throw(
        DomainError(
            value,
            "generating proposal log density may not be NaN or -Inf at its own sample",
        ),
    )
    return nothing
end

function _validate_reduced_mis_denominator(value)
    (isnan(value) || value == -Inf) && throw(
        DomainError(
            value,
            "the reduced MIS denominator may not be NaN or -Inf",
        ),
    )
    return nothing
end

function _evaluate_static_mis_logweights!(
    ::Type{T},
    target,
    denominator,
    samples,
    threaded,
) where {T<:AbstractFloat}
    nsamples = _sample_count(samples)
    target_logs = Vector{T}(undef, nsamples)
    logweights = Vector{T}(undef, nsamples)
    if threaded
        _evaluate_logs_threaded!(
            target_logs,
            target,
            samples,
            Val(:target),
        )
        _evaluate_logs_threaded!(
            logweights,
            denominator,
            samples,
            Val(:proposal_logdensity),
        )
    else
        _evaluate_target_logs!(target_logs, target, samples)
        _evaluate_proposal_logs!(logweights, denominator, samples)
    end
    _construct_logweights!(logweights, target_logs)
    return logweights
end

function _importance_sample_cpu!(sampler, method_state::_PreparedStaticMIS, threaded)
    execution = threaded ? _ThreadedCPUExecution() : _SerialCPUExecution()
    samples, proposal_ids = _draw_static_mis_batch!(sampler, method_state)
    target = _bind_prepared_target(sampler.target, samples)
    log_type = _resolve_static_mis_logweight_type(
        target,
        method_state.bank,
        samples,
    )
    denominator = _StaticMISDenominatorEvaluator(
        log_type,
        method_state.bank,
        sampler.random_buffers.assignments,
        method_state.design.denominator,
    )
    logweights = _evaluate_static_mis_logweights!(
        log_type,
        target,
        denominator,
        samples,
        threaded,
    )
    diagnostics = (
        method=:importance_sampling,
        mis_scheme=_mis_scheme_name(
            method_state.design.assignment,
            method_state.design.denominator,
        ),
        execution=_execution_name(execution),
        threaded=sampler.threaded,
        factor_execution_policy=_factor_execution_name(
            sampler.device,
            sampler.factor_execution,
        ),
        nsamples=sampler.algorithm.nsamples,
        failures=0,
        transfers=_ResultTransferCounter(0, 0),
    )
    samples = _map_result_samples(
        sampler.target,
        samples,
        _native_failure_scratch(sampler.random_buffers),
        diagnostics.transfers,
        sampler.threaded,
    )
    return _adopt_weighted_samples(
        samples,
        logweights;
        provenance=(proposal_id=proposal_ids,),
        diagnostics=diagnostics,
    )
end
