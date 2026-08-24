struct _StratifiedAssignment end
struct _FullMixtureDenominator end

struct _PreparedMISDesign{A,D}
    assignment::A
    denominator::D
end

struct _PreparedStaticMIS{B,D}
    bank::B
    design::D
end

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

function _prepare_method_state(
    algorithm::ImportanceSampling{<:ProposalBank,<:StratifiedMixture},
)
    bank = _prepare_active_proposal_bank(algorithm.proposal)
    design = _PreparedMISDesign(
        _StratifiedAssignment(),
        _FullMixtureDenominator(),
    )
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

function _compile_assignments!(rng, assignments, ::_StratifiedAssignment, cdf)
    nsamples = length(assignments)
    for sample_index in eachindex(assignments)
        assignments[sample_index] = _capture_sampler_failure(
            :proposal_draw,
            sample_index,
        ) do
            uniform = ((sample_index - 1) + rand(rng)) / nsamples
            searchsortedfirst(cdf, uniform)
        end
    end
    return assignments
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

function _mis_logdenominator(
    ::Type{T},
    bank,
    ::_FullMixtureDenominator,
    generating_slot,
    sample,
    sample_index,
) where {T}
    return _capture_sampler_failure(:proposal_logdensity, sample_index) do
        denominator = T(-Inf)
        generating_logdensity = zero(T)
        for slot in eachindex(bank.proposals)
            value = DensityInterface.logdensityof(bank.proposals[slot], sample)
            _validate_mixture_proposal_logdensity(value)
            converted = _convert_static_mis_logdensity(T, value)
            slot == generating_slot && (generating_logdensity = converted)
            logterm = convert(T, bank.logmasses[slot]) + converted
            denominator = LogExpFunctions.logaddexp(denominator, logterm)
        end
        _validate_reduced_mis_denominator(denominator)
        _validate_generating_logdensity(generating_logdensity)
        denominator
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
        mis_scheme=:stratified_mixture,
        execution=_execution_name(execution),
        threaded=sampler.threaded,
        nsamples=sampler.algorithm.nsamples,
        failures=0,
        transfers=(count=0, bytes=0),
    )
    return _adopt_weighted_samples(
        samples,
        logweights;
        provenance=(proposal_id=proposal_ids,),
        diagnostics=diagnostics,
    )
end
