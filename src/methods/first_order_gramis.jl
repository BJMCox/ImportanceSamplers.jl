mutable struct _ValidatedFirstOrderGRAMISToken end
const _VALIDATED_FIRST_ORDER_GRAMIS_TOKEN = _ValidatedFirstOrderGRAMISToken()

"""
    FirstOrderGRAMIS(bank; rounds, round_size, repulsion_strength,
                     covariance_ess_threshold=nothing, covariance_rate=1,
                     covariance_regularization=nothing,
                     tempering_tolerance=1e-4,
                     tempering_max_iterations=16, repulsion_softening=1,
                     max_backtracking_trials=20)

Configure the package's first-order GRAMIS-CAIS hybrid for a fixed population
of at least two equally weighted native Gaussian proposals. Spherical,
diagonal, and factor Gaussians may be mixed when they share one positive
dimension and one `Float32` or `Float64` scalar type. Initial means must be
distinct. Preparation canonicalizes every proposal to a dense lower-factor
representation while preserving configured masses and stable proposal IDs.

`round_size` is the total sample count in each round and may be one positive
integer or a positive integer vector with one entry per round. Balanced
deterministic allocation requires at least `d + 2` samples from every proposal
in every round. `repulsion_strength` and `covariance_rate` accept a scalar,
per-round vector, or callable `round -> value`; schedules restart at round one
for each sampling call. Repulsion strength must be finite and nonnegative, and
each covariance rate must satisfy `0 < rate <= 1`.

`covariance_ess_threshold` may be `nothing`, an absolute integer, a fraction in
`(0, 1)`, or a callable `(round, proposal, m, d) -> threshold`. The resolved
threshold must satisfy `d + 1 <= threshold < m`. The default is
`max(d + 1, ceil(Int, 0.3m))`. The default covariance regularization is
`sqrt(eps(T))`; an explicit value must be finite and nonnegative.
`tempering_tolerance` must lie strictly between zero and one,
`tempering_max_iterations` and `max_backtracking_trials` must be positive
integers exactly representable as `Int`, and `repulsion_softening` must be
finite and positive. Real controls are converted to the proposal scalar type
`T` during preparation, with ordinary floating-point rounding allowed.

The target must provide an explicit gradient, a first-order
LogDensityProblems interface, or an AD backend through [`LogTarget`](@ref).
This type adds configuration and fixed prepared state; sampling execution is
provided by the method execution layer.
"""
struct FirstOrderGRAMIS{B<:ProposalBank,S,R,E,C,V,T,I,P,J} <:
       AbstractImportanceSampler
    bank::B
    rounds::Int
    round_size::S
    repulsion_strength::R
    covariance_ess_threshold::E
    covariance_rate::C
    covariance_regularization::V
    tempering_tolerance::T
    tempering_max_iterations::I
    repulsion_softening::P
    max_backtracking_trials::J

    function FirstOrderGRAMIS(
        bank::B,
        rounds::Int,
        round_size::S,
        repulsion_strength::R,
        covariance_ess_threshold::E,
        covariance_rate::C,
        covariance_regularization::V,
        tempering_tolerance::T,
        tempering_max_iterations::I,
        repulsion_softening::P,
        max_backtracking_trials::J,
        token::_ValidatedFirstOrderGRAMISToken,
    ) where {B<:ProposalBank,S,R,E,C,V,T,I,P,J}
        token === _VALIDATED_FIRST_ORDER_GRAMIS_TOKEN || throw(
            ArgumentError("invalid internal algorithm-construction token"),
        )
        return new{B,S,R,E,C,V,T,I,P,J}(
            bank,
            rounds,
            round_size,
            repulsion_strength,
            covariance_ess_threshold,
            covariance_rate,
            covariance_regularization,
            tempering_tolerance,
            tempering_max_iterations,
            repulsion_softening,
            max_backtracking_trials,
        )
    end
end

function FirstOrderGRAMIS(
    bank::ProposalBank;
    rounds,
    round_size,
    repulsion_strength,
    covariance_ess_threshold=nothing,
    covariance_rate=1,
    covariance_regularization=nothing,
    tempering_tolerance=1e-4,
    tempering_max_iterations=16,
    repulsion_softening=1,
    max_backtracking_trials=20,
)
    validated_rounds = _first_order_gramis_positive_int(rounds, "rounds")
    validated_round_size = _first_order_gramis_round_size(
        validated_rounds,
        round_size,
    )
    _validate_first_order_gramis_bank(bank)
    return FirstOrderGRAMIS(
        bank,
        validated_rounds,
        validated_round_size,
        _copy_first_order_gramis_input(repulsion_strength),
        _copy_first_order_gramis_input(covariance_ess_threshold),
        _copy_first_order_gramis_input(covariance_rate),
        covariance_regularization,
        tempering_tolerance,
        tempering_max_iterations,
        repulsion_softening,
        max_backtracking_trials,
        _VALIDATED_FIRST_ORDER_GRAMIS_TOKEN,
    )
end

_copy_first_order_gramis_input(value::AbstractVector) = collect(value)
_copy_first_order_gramis_input(value) = value

function _first_order_gramis_positive_int(value, name)
    value isa Integer && !(value isa Bool) || throw(
        ArgumentError("$name must be a positive integer"),
    )
    converted = try
        Int(value)
    catch
        throw(ArgumentError("$name must be exactly representable as Int"))
    end
    converted == value || throw(
        ArgumentError("$name must be exactly representable as Int"),
    )
    converted > 0 || throw(ArgumentError("$name must be positive"))
    return converted
end

function _first_order_gramis_round_size(rounds, round_size::Integer)
    return _first_order_gramis_positive_int(round_size, "round_size")
end

function _first_order_gramis_round_size(rounds, round_size::AbstractVector)
    length(round_size) == rounds || throw(
        DimensionMismatch("round_size must contain one entry per round"),
    )
    return map(eachindex(round_size)) do round
        _first_order_gramis_positive_int(
            round_size[round],
            "round_size entry $round",
        )
    end
end

function _first_order_gramis_round_size(rounds, round_size)
    throw(ArgumentError("round_size must be a positive integer or integer vector"))
end

function _validate_first_order_gramis_bank(bank::ProposalBank)
    proposal_count = length(bank.proposals)
    proposal_count >= 2 || throw(
        ArgumentError("FirstOrderGRAMIS requires at least two proposals"),
    )
    first_mass = first(bank.masses)
    all(==(first_mass), bank.masses) || throw(
        ArgumentError("FirstOrderGRAMIS requires equal proposal masses"),
    )
    first_mass > zero(first_mass) || throw(
        ArgumentError("FirstOrderGRAMIS proposal masses must be positive"),
    )

    first_proposal = first(bank.proposals)
    _validate_first_order_gramis_proposal(first_proposal)
    first_location = first_proposal.location
    T = _gaussian_float_type(first_location)
    dimension = _gaussian_dimension(first_location)
    dimension > 0 || throw(
        ArgumentError("FirstOrderGRAMIS proposal dimension must be positive"),
    )
    for proposal in Iterators.drop(bank.proposals, 1)
        _validate_first_order_gramis_proposal(proposal)
        location = proposal.location
        _gaussian_float_type(location) === T || throw(
            ArgumentError("FirstOrderGRAMIS proposals must use one floating type"),
        )
        _gaussian_dimension(location) == dimension || throw(
            DimensionMismatch("FirstOrderGRAMIS proposals must share one dimension"),
        )
    end

    for right in 2:proposal_count
        right_location = bank.proposals[right].location
        for left in 1:(right - 1)
            _first_order_gramis_same_location(
                bank.proposals[left].location,
                right_location,
            ) && throw(
                ArgumentError("FirstOrderGRAMIS initial proposal means must be distinct"),
            )
        end
    end
    return nothing
end

function _validate_first_order_gramis_proposal(proposal)
    proposal isa _GaussianProposal && _is_packable_native_gaussian(proposal) || throw(
        ArgumentError(
            "FirstOrderGRAMIS requires native Float32 or Float64 spherical, " *
            "diagonal, or factor Gaussian proposals",
        ),
    )
    return nothing
end

_first_order_gramis_same_location(
    left::_NativeGaussianFloat,
    right::_NativeGaussianFloat,
) = left == right

_first_order_gramis_same_location(left::_NativeGaussianFloat, right::AbstractVector) =
    length(right) == 1 && left == right[1]

_first_order_gramis_same_location(left::AbstractVector, right::_NativeGaussianFloat) =
    length(left) == 1 && left[1] == right

_first_order_gramis_same_location(left::AbstractVector, right::AbstractVector) =
    left == right

function _first_order_gramis_default_threshold(sample_count)
    quotient, remainder = divrem(sample_count, 10)
    return 3 * quotient + cld(3 * remainder, 10)
end

struct _FirstOrderGRAMISWorkspace{S,L,I,N,Q,O,C,G,A,P,R,F,J,E,W,B}
    samples::S
    round_logweights::L
    local_logweights::L
    generating_logdensities::L
    round_proposal_ids::I
    round_ids::I
    normalized_weights::N
    solve_scratch::Q
    local_starts::O
    covariances::C
    pooled_covariance::S
    whitened_means::G
    gradients::G
    frozen_values::P
    candidate_values::P
    moves::G
    active_mask::A
    steps::P
    repulsion::R
    factor_status::F
    factor_info::J
    local_ess::E
    tempering_powers::W
    backtracking_trials::B
    collision_counts::B
end

mutable struct _PreparedFirstOrderGRAMIS{
    B,
    P,
    R,
    C,
    E,
    G,
    W,
}
    committed::B
    run::B
    candidate::B
    plan::P
    repulsion_strength::R
    covariance_rate::R
    covariance_ess_threshold::C
    covariance_regularization::E
    tempering_tolerance::E
    tempering_max_iterations::Int
    repulsion_softening::E
    max_backtracking_trials::Int
    serial_gradient::G
    threaded_gradient::G
    active_repulsion_rounds::Vector{Int}
    workspace::W
end

function _first_order_gramis_factor_bank(bank::ProposalBank)
    _validate_first_order_gramis_bank(bank)
    proposal_count = length(bank.proposals)
    first_location = first(bank.proposals).location
    T = _gaussian_float_type(first_location)
    proposal_ids = collect(1:proposal_count)
    logmass = -log(T(proposal_count))
    logmasses = fill(logmass, proposal_count)
    cdf = collect(T, (1:proposal_count) ./ proposal_count)
    cdf[end] = one(T)
    dimension = _gaussian_dimension(first_location)
    locations = Matrix{T}(undef, dimension, proposal_count)
    factors = zeros(T, dimension, dimension, proposal_count)
    lognormalizers = Vector{T}(undef, proposal_count)
    for (slot, proposal) in pairs(bank.proposals)
        if proposal.location isa _NativeGaussianFloat
            locations[1, slot] = proposal.location
        else
            copyto!(view(locations, :, slot), proposal.location)
        end
        _copy_packed_gaussian_factor!(factors, proposal, slot)
        lognormalizers[slot] = proposal.lognormalizer
    end
    return _PackedFactorGaussianBank(
        locations,
        factors,
        lognormalizers,
        logmasses,
        cdf,
        proposal_ids,
    )
end

function _first_order_gramis_state_bank(bank::_PackedFactorGaussianBank)
    return _PackedFactorGaussianBank(
        copy(bank.locations),
        copy(bank.factors),
        copy(bank.lognormalizers),
        bank.logmasses,
        bank.cdf,
        bank.proposal_ids,
    )
end

function _first_order_gramis_real_control(::Type{T}, value, name, predicate) where {T}
    value isa Real && !(value isa Bool) || throw(
        ArgumentError("$name must be real"),
    )
    converted = try
        T(value)
    catch
        throw(ArgumentError("$name must be convertible to $T"))
    end
    isfinite(converted) && predicate(converted) || throw(
        ArgumentError("$name is outside its permitted finite range"),
    )
    return converted
end

function _resolve_first_order_gramis_schedule(
    ::Type{T},
    input,
    rounds,
    name,
    predicate,
) where {T}
    values = Vector{T}(undef, rounds)
    if input isa AbstractVector
        length(input) == rounds || throw(
            DimensionMismatch("$name must contain one entry per round"),
        )
        for round in 1:rounds
            values[round] = _first_order_gramis_real_control(
                T,
                input[round],
                "$name entry $round",
                predicate,
            )
        end
    elseif input isa Real
        value = _first_order_gramis_real_control(T, input, name, predicate)
        fill!(values, value)
    else
        for round in 1:rounds
            raw = try
                input(round)
            catch error
                throw(ArgumentError("$name callable failed for round $round: $error"))
            end
            values[round] = _first_order_gramis_real_control(
                T,
                raw,
                "$name result for round $round",
                predicate,
            )
        end
    end
    return values
end

function _resolve_first_order_gramis_threshold(
    ::Type{T},
    input,
    round,
    proposal,
    sample_count,
    dimension,
) where {T}
    raw = if input === nothing
        nothing
    elseif input isa Function || !(input isa Union{Integer,Real})
        try
            input(round, proposal, sample_count, dimension)
        catch error
            throw(
                ArgumentError(
                    "covariance_ess_threshold callable failed for round $round, " *
                    "proposal $proposal: $error",
                ),
            )
        end
    else
        input
    end

    threshold = if raw === nothing
        max(dimension + 1, _first_order_gramis_default_threshold(sample_count))
    elseif raw isa Integer && !(raw isa Bool)
        try
            converted = Int(raw)
            converted == raw || throw(InexactError(:Int, Int, raw))
            converted
        catch
            throw(
                ArgumentError(
                    "covariance_ess_threshold must be exactly representable as Int",
                ),
            )
        end
    elseif raw isa Real && !(raw isa Bool)
        fraction = _first_order_gramis_real_control(
            T,
            raw,
            "covariance_ess_threshold fraction",
            value -> zero(T) < value < one(T),
        )
        max(dimension + 1, ceil(Int, fraction * T(sample_count)))
    else
        throw(
            ArgumentError(
                "covariance_ess_threshold must resolve to nothing, an integer, or a fraction",
            ),
        )
    end

    dimension + 1 <= threshold < sample_count || throw(
        ArgumentError(
            "covariance_ess_threshold must satisfy d + 1 <= threshold < m " *
            "for round $round, proposal $proposal",
        ),
    )
    return threshold
end

function _resolve_first_order_gramis_thresholds(
    ::Type{T},
    input,
    plan,
    dimension,
) where {T}
    proposal_count, rounds = size(plan.counts)
    thresholds = Matrix{Int}(undef, proposal_count, rounds)
    for round in 1:rounds
        for proposal in 1:proposal_count
            thresholds[proposal, round] = _resolve_first_order_gramis_threshold(
                T,
                input,
                round,
                proposal,
                plan.counts[proposal, round],
                dimension,
            )
        end
    end
    return thresholds
end

function _first_order_gramis_group_starts(counts)
    starts = similar(counts)
    for round in axes(counts, 2)
        first_sample = 1
        for proposal in axes(counts, 1)
            starts[proposal, round] = first_sample
            first_sample += counts[proposal, round]
        end
    end
    return starts
end

function _allocate_first_order_gramis_workspace(bank, plan, ::Type{L}) where {L}
    T = eltype(bank.locations)
    dimension, proposal_count = size(bank.locations)
    capacity = maximum(plan.schedule)
    prototype = bank.locations
    return _FirstOrderGRAMISWorkspace(
        similar(prototype, T, dimension, capacity),
        similar(prototype, L, capacity),
        similar(prototype, L, capacity),
        similar(prototype, L, capacity),
        similar(prototype, Int, capacity),
        similar(prototype, Int, capacity),
        similar(prototype, T, capacity),
        _allocate_mis_solve_scratch(prototype, bank, capacity),
        _first_order_gramis_group_starts(plan.counts),
        similar(prototype, T, dimension, dimension, proposal_count),
        similar(prototype, T, dimension, dimension),
        similar(prototype, T, dimension, proposal_count),
        similar(prototype, T, dimension, proposal_count),
        similar(prototype, T, proposal_count),
        similar(prototype, T, proposal_count),
        similar(prototype, T, dimension, proposal_count),
        similar(prototype, Bool, proposal_count),
        similar(prototype, T, proposal_count),
        similar(prototype, T, dimension, proposal_count),
        similar(prototype, UInt8, proposal_count),
        similar(prototype, Int32, proposal_count),
        similar(prototype, T, proposal_count),
        similar(prototype, T, proposal_count),
        similar(prototype, Int, proposal_count),
        similar(prototype, Int, proposal_count),
    )
end

function _prepare_first_order_gramis_state(
    algorithm::FirstOrderGRAMIS,
    committed::_PackedFactorGaussianBank,
    ::Type{L},
    serial_gradient,
    threaded_gradient,
) where {L}
    run = _first_order_gramis_state_bank(committed)
    candidate = _first_order_gramis_state_bank(committed)
    T = eltype(committed.locations)
    schedule = _resolve_adaptive_schedule(algorithm.rounds, algorithm.round_size)
    proposal_count = size(committed.locations, 2)
    equal_masses = fill(one(T), proposal_count)
    plan = _deterministic_allocation_plan(committed, equal_masses, schedule)
    dimension = size(committed.locations, 1)
    for round in eachindex(schedule)
        minimum(view(plan.counts, :, round)) >= dimension + 2 || throw(
            ArgumentError(
                "round $round must assign at least d + 2 samples to every proposal",
            ),
        )
    end

    repulsion_strength = _resolve_first_order_gramis_schedule(
        T,
        algorithm.repulsion_strength,
        algorithm.rounds,
        "repulsion_strength",
        value -> value >= zero(T),
    )
    covariance_rate = _resolve_first_order_gramis_schedule(
        T,
        algorithm.covariance_rate,
        algorithm.rounds,
        "covariance_rate",
        value -> zero(T) < value <= one(T),
    )
    covariance_ess_threshold = _resolve_first_order_gramis_thresholds(
        T,
        algorithm.covariance_ess_threshold,
        plan,
        dimension,
    )
    covariance_regularization = algorithm.covariance_regularization === nothing ?
                                sqrt(eps(T)) : _first_order_gramis_real_control(
        T,
        algorithm.covariance_regularization,
        "covariance_regularization",
        value -> value >= zero(T),
    )
    tempering_tolerance = _first_order_gramis_real_control(
        T,
        algorithm.tempering_tolerance,
        "tempering_tolerance",
        value -> zero(T) < value < one(T),
    )
    tempering_max_iterations = _first_order_gramis_positive_int(
        algorithm.tempering_max_iterations,
        "tempering_max_iterations",
    )
    repulsion_softening = _first_order_gramis_real_control(
        T,
        algorithm.repulsion_softening,
        "repulsion_softening",
        value -> value > zero(T),
    )
    max_backtracking_trials = _first_order_gramis_positive_int(
        algorithm.max_backtracking_trials,
        "max_backtracking_trials",
    )
    iszero(ldexp(one(T), 1 - max_backtracking_trials)) && throw(
        ArgumentError(
            "max_backtracking_trials requests a final step that underflows to zero in $T",
        ),
    )
    workspace = _allocate_first_order_gramis_workspace(committed, plan, L)
    active_repulsion_rounds = findall(!iszero, repulsion_strength)
    return _PreparedFirstOrderGRAMIS(
        committed,
        run,
        candidate,
        plan,
        repulsion_strength,
        covariance_rate,
        covariance_ess_threshold,
        covariance_regularization,
        tempering_tolerance,
        tempering_max_iterations,
        repulsion_softening,
        max_backtracking_trials,
        serial_gradient,
        threaded_gradient,
        active_repulsion_rounds,
        workspace,
    )
end

function _prepare_method_state(algorithm::FirstOrderGRAMIS, prepared_target)
    packed = _first_order_gramis_factor_bank(algorithm.bank)
    binding_sample = view(packed.locations, :, 1)
    bound_target = _bind_resolved_target(prepared_target, binding_sample)
    log_type = _resolve_packed_static_mis_logweight_type(
        bound_target,
        packed,
        typeof(binding_sample),
    )
    serial_gradient = _prepare_bound_gradient(prepared_target, binding_sample, 1)
    worker_count = max(1, length(Threads.threadpooltids(:default)))
    threaded_gradient = _prepare_bound_gradient(
        prepared_target,
        binding_sample,
        worker_count,
    )
    return _prepare_first_order_gramis_state(
        algorithm,
        packed,
        log_type,
        serial_gradient,
        threaded_gradient,
    )
end

_algorithm_proposal(algorithm::FirstOrderGRAMIS) = algorithm.bank
_algorithm_sample_budget(algorithm::FirstOrderGRAMIS) =
    _adaptive_sample_budget(algorithm.rounds, algorithm.round_size)

function _allocate_random_buffers(
    ::MLDataDevices.AbstractDevice,
    ::ProposalBank,
    method_state::_PreparedFirstOrderGRAMIS,
    sample_budget,
)
    prototype = method_state.committed.locations
    T = eltype(prototype)
    dimension = size(prototype, 1)
    capacity = maximum(method_state.plan.schedule)
    uniform = similar(prototype, T, 0)
    normal = similar(prototype, T, dimension * capacity)
    failure_scratch = _allocate_native_failure_scratch(normal, capacity)
    return _RandomBuffers(uniform, normal, failure_scratch)
end

function _copy_algorithm(device, algorithm::FirstOrderGRAMIS)
    copied_bank = _copy_dm_pmc_bank(device, algorithm.bank)
    return FirstOrderGRAMIS(
        copied_bank;
        rounds=algorithm.rounds,
        round_size=algorithm.round_size,
        repulsion_strength=algorithm.repulsion_strength,
        covariance_ess_threshold=algorithm.covariance_ess_threshold,
        covariance_rate=algorithm.covariance_rate,
        covariance_regularization=algorithm.covariance_regularization,
        tempering_tolerance=algorithm.tempering_tolerance,
        tempering_max_iterations=algorithm.tempering_max_iterations,
        repulsion_softening=algorithm.repulsion_softening,
        max_backtracking_trials=algorithm.max_backtracking_trials,
    )
end

_copy_algorithm(
    ::MLDataDevices.CPUDevice{Missing},
    algorithm::FirstOrderGRAMIS,
) = deepcopy(algorithm)

function _copy_accelerator_algorithm(
    device,
    algorithm::FirstOrderGRAMIS,
    ::_PreparedFirstOrderGRAMIS,
)
    return deepcopy(algorithm)
end

function _copy_first_order_gramis_workspace(device, workspace)
    return _FirstOrderGRAMISWorkspace(
        _copy_to_device(device, workspace.samples),
        _copy_to_device(device, workspace.round_logweights),
        _copy_to_device(device, workspace.local_logweights),
        _copy_to_device(device, workspace.generating_logdensities),
        _copy_to_device(device, workspace.round_proposal_ids),
        _copy_to_device(device, workspace.round_ids),
        _copy_to_device(device, workspace.normalized_weights),
        _copy_to_device(device, workspace.solve_scratch),
        _copy_to_device(device, workspace.local_starts),
        _copy_to_device(device, workspace.covariances),
        _copy_to_device(device, workspace.pooled_covariance),
        _copy_to_device(device, workspace.whitened_means),
        _copy_to_device(device, workspace.gradients),
        _copy_to_device(device, workspace.frozen_values),
        _copy_to_device(device, workspace.candidate_values),
        _copy_to_device(device, workspace.moves),
        _copy_to_device(device, workspace.active_mask),
        _copy_to_device(device, workspace.steps),
        _copy_to_device(device, workspace.repulsion),
        _copy_to_device(device, workspace.factor_status),
        _copy_to_device(device, workspace.factor_info),
        _copy_to_device(device, workspace.local_ess),
        _copy_to_device(device, workspace.tempering_powers),
        _copy_to_device(device, workspace.backtracking_trials),
        _copy_to_device(device, workspace.collision_counts),
    )
end

function _prepare_transferred_method_state(
    device,
    algorithm::FirstOrderGRAMIS,
    method_state::_PreparedFirstOrderGRAMIS,
    transferred_target,
)
    committed = _copy_packed_gaussian_bank(device, method_state.committed)
    run = _first_order_gramis_state_bank(committed)
    candidate = _first_order_gramis_state_bank(committed)
    plan = method_state.plan
    transferred_plan = _DeterministicAllocationPlan(
        Tuple(plan.schedule),
        _copy_to_device(device, plan.counts),
        _copy_to_device(device, plan.assignments),
        _copy_to_device(device, plan.logcoefficients),
        Tuple(plan.offsets),
    )
    binding_sample = view(committed.locations, :, 1)
    bound_gradient = _prepare_bound_gradient(transferred_target, binding_sample, 1)
    return _PreparedFirstOrderGRAMIS(
        committed,
        run,
        candidate,
        transferred_plan,
        _copy_to_device(device, method_state.repulsion_strength),
        _copy_to_device(device, method_state.covariance_rate),
        _copy_to_device(device, method_state.covariance_ess_threshold),
        method_state.covariance_regularization,
        method_state.tempering_tolerance,
        method_state.tempering_max_iterations,
        method_state.repulsion_softening,
        method_state.max_backtracking_trials,
        bound_gradient,
        bound_gradient,
        copy(method_state.active_repulsion_rounds),
        _copy_first_order_gramis_workspace(device, method_state.workspace),
    )
end

_transferred_backend_state(
    algorithm,
    method_state::_PreparedFirstOrderGRAMIS,
    target,
    random_buffers,
) = (method_state, target, random_buffers)

_prepared_backend_state(
    sampler,
    method_state::_PreparedFirstOrderGRAMIS,
) = (method_state, sampler.target, sampler.random_buffers, sampler.rng)

_first_order_gramis_resident_state(method_state::_PreparedFirstOrderGRAMIS) =
    (
        method_state.committed,
        method_state.run,
        method_state.candidate,
        method_state.plan,
        method_state.repulsion_strength,
        method_state.covariance_rate,
        method_state.covariance_ess_threshold,
        method_state.serial_gradient,
        method_state.threaded_gradient,
        method_state.workspace,
    )

function _preflight_first_order_gramis_factorization!(device, method_state)
    throw(SamplerDeviceError(device, :first_order_gramis_accelerator_unavailable))
end

function _preflight_first_order_gramis_kernel_arguments(
    device,
    kernel,
    arguments,
)
    for argument in arguments
        _preflight_kernel_argument(device, kernel, argument)
    end
    return nothing
end

function _preflight_accelerator_method(
    device,
    target,
    algorithm::FirstOrderGRAMIS,
    method_state::_PreparedFirstOrderGRAMIS,
    random_buffers,
    factor_execution,
)
    workspace = method_state.workspace
    bank = method_state.committed
    binding_sample = view(bank.locations, :, 1)
    bound_target = _bind_resolved_target(target, binding_sample)
    log_type = eltype(workspace.round_logweights)
    target_argument = _NativeDeviceTarget{log_type,typeof(bound_target)}(
        bound_target,
    )
    backend = KernelAbstractions.get_backend(workspace.samples)

    sample_kernel = _mis_round_launch_kernel!(backend)
    preflight_output = _MISRoundOutput(
        view(workspace.round_logweights, 1:1),
        view(workspace.round_proposal_ids, 1:1),
        _MISAdaptationOutput(
            workspace.local_logweights,
            workspace.generating_logdensities,
        ),
    )
    _preflight_first_order_gramis_kernel_arguments(
        device,
        sample_kernel,
        _mis_round_kernel_arguments(
            view(workspace.samples, :, 1:1),
            preflight_output,
            random_buffers.failure_scratch.record.storage,
            view(random_buffers.normal, 1:size(bank.locations, 1)),
            target_argument,
            bank,
            view(method_state.plan.assignments, 1:1, 1),
            _RealizedMixtureDenominator(method_state.plan.logcoefficients, 1),
            workspace.solve_scratch,
        ),
    )

    local_weights_kernel = _first_order_gramis_local_weights_kernel!(backend)
    _preflight_first_order_gramis_kernel_arguments(
        device,
        local_weights_kernel,
        _first_order_gramis_local_weight_arguments(
            workspace.local_logweights,
            workspace.generating_logdensities,
            workspace.round_proposal_ids,
            workspace.round_ids,
            1,
            random_buffers.failure_scratch.record.storage,
        ),
    )

    covariance_kernels = (
        _cooperative_local_weights_kernel!(
            backend,
            _GRAMIS_REDUCTION_WORKGROUP_SIZE,
        ),
        _fit_local_covariances_kernel!(backend),
        _blend_local_covariances_kernel!(backend),
    )
    covariance_arguments = _first_order_gramis_covariance_kernel_arguments(
        method_state,
        1,
    )
    for (kernel, arguments) in zip(covariance_kernels, covariance_arguments)
        _preflight_first_order_gramis_kernel_arguments(
            device,
            kernel,
            arguments,
        )
    end

    factor_kernel = _factor_population_kernel!(
        backend,
        _GRAMIS_CHOLESKY_WORKGROUP_SIZE,
    )
    _preflight_first_order_gramis_kernel_arguments(
        device,
        factor_kernel,
        _first_order_gramis_factor_arguments(method_state),
    )

    repulsion_kernel = _repulsion_force_kernel!(backend)
    _preflight_first_order_gramis_kernel_arguments(
        device,
        repulsion_kernel,
        _repulsion_force_arguments(
            workspace.repulsion,
            workspace.collision_counts,
            bank.locations,
            workspace.whitened_means,
            method_state.repulsion_strength,
            1,
            method_state.repulsion_softening,
        ),
    )

    _preflight_first_order_gramis_factorization!(device, method_state)
    _execute_first_order_gramis_live_preflight!(
        device,
        method_state,
        bound_target,
        random_buffers,
    )
    return nothing
end

function _first_order_gramis_preserving_cpu_destination(destination)
    if applicable(eltype, destination)
        policy = eltype(destination)
        policy in (Missing, Nothing) || throw(
            ArgumentError(
                "current_proposal requires a preserving CPU destination; " *
                "use MLDataDevices.cpu_device() without a scalar conversion",
            ),
        )
    end
    return destination
end

"""
    current_proposal(sampler)
    current_proposal(destination, sampler)

Return an independent `ProposalBank` of dense [`FactorGaussian`](@ref)
snapshots from the population committed by a prepared [`FirstOrderGRAMIS`](@ref)
sampler. Configured masses and stable proposal IDs are preserved. The returned
locations, factors, log normalizers, proposal vector, and masses do not alias
the sampler's committed or scratch arrays.

The one-argument form is CPU-only and never hides a device transfer. For an
accelerator-prepared sampler, pass an explicit preserving CPU destination such
as `MLDataDevices.cpu_device()`. Scalar-converting and non-CPU destinations are
rejected. Only committed locations, factors, log normalizers, and stable IDs
are copied from accelerator state.
"""
function current_proposal(
    sampler::_PreparedImportanceSampler{R,B,T,A,M,D},
) where {R,B,T,A<:FirstOrderGRAMIS,M<:_PreparedFirstOrderGRAMIS,D}
    sampler.device isa MLDataDevices.AbstractAcceleratorDevice && throw(
        ArgumentError(
            "current_proposal(sampler) does not copy accelerator state " *
            "implicitly; call current_proposal(cpu_device(), sampler) " *
            "to request an explicit CPU snapshot",
        ),
    )
    return current_proposal(MLDataDevices.cpu_device(), sampler)
end

function current_proposal(
    destination::MLDataDevices.AbstractCPUDevice,
    sampler::_PreparedImportanceSampler{R,B,T,A,M,D},
) where {R,B,T,A<:FirstOrderGRAMIS,M<:_PreparedFirstOrderGRAMIS,D}
    _first_order_gramis_preserving_cpu_destination(destination)
    committed = sampler.method_state.committed
    if sampler.device isa MLDataDevices.AbstractCPUDevice
        return _first_order_gramis_snapshot(
            committed.locations,
            committed.factors,
            committed.lognormalizers,
            committed.proposal_ids,
            sampler.algorithm.bank.masses,
        )
    end
    parameters = _with_backend_device(sampler.device) do
        (
            destination(Array(committed.locations)),
            destination(Array(committed.factors)),
            destination(Array(committed.lognormalizers)),
            destination(Array(committed.proposal_ids)),
        )
    end
    return _first_order_gramis_snapshot(
        parameters...,
        sampler.algorithm.bank.masses,
    )
end

function current_proposal(
    destination::MLDataDevices.AbstractDevice,
    sampler::_PreparedImportanceSampler{R,B,T,A,M,D},
) where {R,B,T,A<:FirstOrderGRAMIS,M<:_PreparedFirstOrderGRAMIS,D}
    throw(
        ArgumentError(
            "current_proposal requires a CPU destination; got " *
            string(typeof(destination)),
        ),
    )
end

function _first_order_gramis_snapshot(
    locations::AbstractMatrix{T},
    factors::AbstractArray{T,3},
    lognormalizers::AbstractVector{T},
    proposal_ids,
    configured_masses,
) where {T<:_NativeGaussianFloat}
    D = _GaussianProposal{
        GaussianFamily,
        Vector{T},
        _FactorGaussianScale{Matrix{T}},
        T,
    }
    proposals = Vector{D}(undef, length(configured_masses))
    for (slot, proposal_id) in pairs(proposal_ids)
        proposals[proposal_id] = _GaussianProposal(
            GaussianFamily(),
            copy(view(locations, :, slot)),
            _FactorGaussianScale(copy(view(factors, :, :, slot))),
            lognormalizers[slot],
        )
    end
    snapshot = ProposalBank(proposals, configured_masses)
    copyto!(snapshot.masses, configured_masses)
    return snapshot
end
