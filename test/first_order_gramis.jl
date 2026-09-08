using Test
using ImportanceSamplers
import ADTypes
import DifferentiationInterface
import LinearAlgebra
import MLDataDevices
import Random

include("support/first_order_gramis.jl")

const GRAMISIS = ImportanceSamplers

mutable struct FirstOrderGRAMISDITrace
    next_preparation::Threads.Atomic{Int}
    lock::ReentrantLock
    calls::Vector{Tuple{Int,Any}}
end

FirstOrderGRAMISDITrace() = FirstOrderGRAMISDITrace(
    Threads.Atomic{Int}(0),
    ReentrantLock(),
    Tuple{Int,Any}[],
)

struct FirstOrderGRAMISRecordingAD <: ADTypes.AbstractADType
    trace::FirstOrderGRAMISDITrace
end

ADTypes.mode(::FirstOrderGRAMISRecordingAD) = ADTypes.ForwardMode()

mutable struct FirstOrderGRAMISRecordingPreparation
    id::Int
    trace::FirstOrderGRAMISDITrace
end

function DifferentiationInterface.prepare_gradient(
    target,
    backend::FirstOrderGRAMISRecordingAD,
    sample,
)
    id = Threads.atomic_add!(backend.trace.next_preparation, 1) + 1
    return FirstOrderGRAMISRecordingPreparation(id, backend.trace)
end

function DifferentiationInterface.gradient!(
    target,
    destination,
    preparation::FirstOrderGRAMISRecordingPreparation,
    backend::FirstOrderGRAMISRecordingAD,
    sample,
)
    thread_id = Threads.threadid()
    lock(preparation.trace.lock) do
        push!(preparation.trace.calls, (thread_id, preparation))
    end
    destination .= -sample
    return destination
end

@testset "FirstOrderGRAMIS constructor and schedules" begin
    for T in (Float32, Float64)
        bank = first_order_gramis_bank(T)
        round_sizes = [24, 30, 36]
        repulsion = FirstOrderGRAMISSchedule(T[0.1, 0.2, 0.3], Int[])
        covariance_rate = T[1, 0.75, 0.5]
        algorithm = @inferred FirstOrderGRAMIS(
            bank;
            rounds=3,
            round_size=round_sizes,
            repulsion_strength=repulsion,
            covariance_rate,
        )

        @test algorithm isa AbstractImportanceSampler
        @test algorithm.bank === bank
        @test algorithm.rounds == 3
        @test algorithm.round_size == [24, 30, 36]
        @test algorithm.round_size !== round_sizes
        round_sizes[1] = 1
        @test algorithm.round_size == [24, 30, 36]

        sampler = @inferred prepare_first_order_gramis(algorithm)
        state = sampler.method_state
        @test state.repulsion_strength == T[0.1, 0.2, 0.3]
        @test state.covariance_rate == covariance_rate
        @test repulsion.calls == [1, 2, 3]
        @test state.plan.schedule == [24, 30, 36]
        @test GRAMISIS._algorithm_sample_budget(algorithm) == 90

        second = prepare_first_order_gramis(algorithm)
        @test second.method_state.repulsion_strength == state.repulsion_strength
        @test repulsion.calls == [1, 2, 3, 1, 2, 3]

        scalar = prepare_first_order_gramis(FirstOrderGRAMIS(
            bank;
            rounds=2,
            round_size=24,
            repulsion_strength=T(0.25),
        ))
        @test scalar.method_state.repulsion_strength == fill(T(0.25), 2)
        @test scalar.method_state.covariance_rate == ones(T, 2)
    end
end

@testset "FirstOrderGRAMIS proposal validation" begin
    good = first_order_gramis_bank()
    constructor(bank; round_size=24) = FirstOrderGRAMIS(
        bank;
        rounds=1,
        round_size,
        repulsion_strength=0.1,
    )

    @test_throws ArgumentError constructor(ProposalBank(good.proposals[1:1]))
    @test_throws ArgumentError constructor(ProposalBank(good.proposals, [1, 1, 2]))
    @test_throws ArgumentError constructor(ProposalBank([
        SphericalGaussian([-1.0, 0.0], 1.0),
        SphericalGaussian([-1.0, 0.0], 2.0),
    ]))
    @test_throws DimensionMismatch constructor(ProposalBank(Any[
        SphericalGaussian([-1.0], 1.0),
        SphericalGaussian([1.0, 0.0], 1.0),
    ]))
    @test_throws ArgumentError constructor(ProposalBank(Any[
        SphericalGaussian(Float32[-1, 0], 1.0f0),
        SphericalGaussian(Float64[1, 0], 1.0),
    ]))
    @test_throws ArgumentError constructor(ProposalBank(Any[
        SphericalGaussian([-1.0, 0.0], 1.0),
        FirstOrderGRAMISNonGaussian(),
    ]))

    empty_location = GRAMISIS._GaussianProposal(
        GRAMISIS.GaussianFamily(),
        Float64[],
        GRAMISIS._SphericalGaussianScale(1.0),
        0.0,
    )
    @test_throws ArgumentError constructor(ProposalBank(Any[
        empty_location,
        empty_location,
    ]))

    too_small = constructor(good; round_size=11)
    @test_throws ArgumentError prepare_first_order_gramis(too_small)
    @test minimum(prepare_first_order_gramis(
        constructor(good; round_size=12),
    ).method_state.plan.counts) == 4

    scalar_one_dimensional = SphericalGaussian(-1.0, 0.75)
    vector_one_dimensional = FactorGaussian([1.0], reshape([1.25], 1, 1))
    mixed_one_dimensional = ProposalBank(
        Union{typeof(scalar_one_dimensional), typeof(vector_one_dimensional)}[
            scalar_one_dimensional,
            vector_one_dimensional,
        ],
    )
    mixed_sampler = @inferred prepare_first_order_gramis(
        constructor(mixed_one_dimensional; round_size=6),
    )
    @test mixed_sampler.method_state.committed.locations == reshape([-1.0, 1.0], 1, 2)
    @test size(mixed_sampler.method_state.committed.factors) == (1, 1, 2)
    @test current_proposal(mixed_sampler).proposals[1].location == [-1.0]
    @test_throws ArgumentError constructor(ProposalBank(Any[
        SphericalGaussian(0.0, 1.0),
        SphericalGaussian([0.0], 1.0),
    ]))
end

@testset "FirstOrderGRAMIS numeric controls" begin
    for T in (Float32, Float64)
        bank = first_order_gramis_bank(T)
        function prepare_controls(; kwargs...)
            algorithm = FirstOrderGRAMIS(
                bank;
                rounds=2,
                round_size=T === Float32 ? Int16[15, 16] : UInt8[15, 16],
                repulsion_strength=Float64(0.01),
                kwargs...,
            )
            return prepare_first_order_gramis(algorithm)
        end

        state = (@inferred prepare_controls(
            covariance_ess_threshold=0.75,
            covariance_regularization=Float64(0.01),
            tempering_tolerance=Float64(0.01),
            tempering_max_iterations=UInt8(7),
            repulsion_softening=Float64(2),
            max_backtracking_trials=Int16(9),
        )).method_state
        @test state.repulsion_strength == fill(T(0.01), 2)
        @test state.covariance_regularization === T(0.01)
        @test state.tempering_tolerance === T(0.01)
        @test state.tempering_max_iterations === 7
        @test state.repulsion_softening === T(2)
        @test state.max_backtracking_trials === 9
        @test state.covariance_ess_threshold == [4 4; 4 5; 4 4]

        default_state = prepare_controls().method_state
        @test default_state.covariance_regularization === sqrt(eps(T))
        @test default_state.covariance_ess_threshold == [3 3; 3 3; 3 3]
        callable_state = prepare_controls(
            covariance_ess_threshold=(round, proposal, m, d) -> d + 1,
            covariance_rate=round -> round == 1 ? one(T) : T(0.5),
        ).method_state
        @test callable_state.covariance_ess_threshold == fill(3, 3, 2)
        @test callable_state.covariance_rate == T[1, 0.5]

        for bad in (-1, Inf, NaN)
            @test_throws ArgumentError prepare_controls(covariance_regularization=bad)
        end
        for bad in (0, 1, -0.1, Inf, NaN)
            @test_throws ArgumentError prepare_controls(tempering_tolerance=bad)
        end
        for bad in (0, -1, true, big(typemax(Int)) + 1)
            @test_throws ArgumentError prepare_controls(tempering_max_iterations=bad)
            @test_throws ArgumentError prepare_controls(max_backtracking_trials=bad)
        end
        for bad in (0, -1, Inf, NaN)
            @test_throws ArgumentError prepare_controls(repulsion_softening=bad)
        end
        for bad in (-0.1, Inf, NaN)
            @test_throws ArgumentError prepare_first_order_gramis(FirstOrderGRAMIS(
                bank;
                rounds=1,
                round_size=15,
                repulsion_strength=bad,
            ))
        end
        for bad in (0, -0.1, 1.1, Inf, NaN)
            @test_throws ArgumentError prepare_controls(covariance_rate=bad)
        end
        for bad in (0, 1, -0.1, 1.1, Inf, NaN)
            @test_throws ArgumentError prepare_controls(covariance_ess_threshold=bad)
        end
        @test_throws ArgumentError prepare_controls(covariance_ess_threshold=2)
        @test_throws ArgumentError prepare_controls(covariance_ess_threshold=5)

        last_nonzero_trial = T === Float32 ? 150 : 1075
        first_zero_trial = last_nonzero_trial + 1
        boundary_state = prepare_controls(
            max_backtracking_trials=last_nonzero_trial,
        ).method_state
        @test boundary_state.max_backtracking_trials == last_nonzero_trial
        @test !iszero(ldexp(one(T), 1 - last_nonzero_trial))
        @test iszero(ldexp(one(T), 1 - first_zero_trial))
        @test_throws ArgumentError prepare_controls(
            max_backtracking_trials=first_zero_trial,
        )
    end

    @test_throws ArgumentError FirstOrderGRAMIS(
        first_order_gramis_bank();
        rounds=true,
        round_size=24,
        repulsion_strength=0.1,
    )
    @test_throws ArgumentError FirstOrderGRAMIS(
        first_order_gramis_bank();
        rounds=1,
        round_size=24.0,
        repulsion_strength=0.1,
    )
    for rounds in (0, -1, 1.0)
        @test_throws ArgumentError FirstOrderGRAMIS(
            first_order_gramis_bank();
            rounds,
            round_size=24,
            repulsion_strength=0.1,
        )
    end
    @test_throws DimensionMismatch FirstOrderGRAMIS(
        first_order_gramis_bank();
        rounds=2,
        round_size=[24],
        repulsion_strength=0.1,
    )
    @test_throws ArgumentError FirstOrderGRAMIS(
        first_order_gramis_bank();
        rounds=2,
        round_size=[24, 0],
        repulsion_strength=0.1,
    )
end

@testset "FirstOrderGRAMIS fixed state and independent snapshots" begin
    algorithm = FirstOrderGRAMIS(
        first_order_gramis_bank();
        rounds=3,
        round_size=[15, 16, 17],
        repulsion_strength=[0.1, 0.2, 0.3],
    )
    sampler = @inferred prepare_first_order_gramis(algorithm)
    state = sampler.method_state
    committed = state.committed
    run = state.run
    candidate = state.candidate
    workspace = state.workspace

    @test sampler.target === state.serial_gradient.target
    @test sampler.target === state.threaded_gradient.target
    @test state.serial_gradient.target.context ===
          state.threaded_gradient.target.context

    for field in (:locations, :factors, :lognormalizers)
        @test getfield(committed, field) !== getfield(run, field)
        @test getfield(committed, field) !== getfield(candidate, field)
        @test getfield(run, field) !== getfield(candidate, field)
    end
    for field in (:logmasses, :cdf, :proposal_ids)
        @test getfield(committed, field) === getfield(run, field)
        @test getfield(committed, field) === getfield(candidate, field)
    end

    @test workspace.frozen_values !== workspace.candidate_values
    @test workspace.moves !== workspace.gradients
    @test workspace.moves !== workspace.repulsion
    @test workspace.moves !== candidate.locations

    snapshot = @inferred current_proposal(sampler)
    explicit_snapshot = @inferred current_proposal(MLDataDevices.cpu_device(), sampler)
    @test snapshot isa ProposalBank
    @test all(proposal -> proposal.scale isa GRAMISIS._FactorGaussianScale, snapshot.proposals)
    @test snapshot.masses == algorithm.bank.masses
    @test explicit_snapshot.masses == algorithm.bank.masses
    @test snapshot.proposals !== explicit_snapshot.proposals
    @test snapshot.masses !== explicit_snapshot.masses

    retained_locations = copy(committed.locations)
    retained_factors = copy(committed.factors)
    retained_lognormalizers = copy(committed.lognormalizers)
    snapshot.proposals[1].location[1] = 1.0e6
    snapshot.proposals[1].scale.factor[1, 1] = 1.0e6
    snapshot.masses[1] = 0
    @test committed.locations == retained_locations
    @test committed.factors == retained_factors
    @test committed.lognormalizers == retained_lognormalizers

    independent_snapshot = current_proposal(sampler)
    snapshot_location = copy(independent_snapshot.proposals[1].location)
    snapshot_factor = copy(independent_snapshot.proposals[1].scale.factor)
    run.locations[1, 1] = -2.0e6
    run.factors[1, 1, 1] = 2.0e6
    candidate.locations[1, 1] = -3.0e6
    candidate.factors[1, 1, 1] = 3.0e6
    workspace.samples[1, 1] = 4.0e6
    workspace.covariances[1, 1, 1] = 5.0e6
    @test independent_snapshot.proposals[1].location == snapshot_location
    @test independent_snapshot.proposals[1].scale.factor == snapshot_factor
end

function first_order_gramis_logaddexp(left, right)
    maximum_value = max(left, right)
    return maximum_value + log(exp(left - maximum_value) + exp(right - maximum_value))
end

function first_order_gramis_mixture_logdensity(bank, sample, counts)
    total = sum(counts)
    value = -Inf
    for (proposal, count) in zip(bank.proposals, counts)
        term = log(count / total) +
               GRAMISIS.DensityInterface.logdensityof(proposal, sample)
        value = value == -Inf ? term : first_order_gramis_logaddexp(value, term)
    end
    return value
end

function first_order_gramis_round_summary(logweights)
    maximum_logweight = maximum(logweights)
    scaled = exp.(logweights .- maximum_logweight)
    return (
        ess=sum(scaled)^2 / sum(abs2, scaled),
        lognormalizer=maximum_logweight + log(sum(scaled)) - log(length(logweights)),
    )
end

function first_order_gramis_assert_same_diagnostics(serial, threaded)
    @test serial.execution === :serial
    @test serial.threaded === false
    @test threaded.execution === :threaded
    @test threaded.threaded === true
    for field in propertynames(serial)
        field in (:execution, :threaded, :transfers) && continue
        @test getproperty(serial, field) == getproperty(threaded, field)
    end
    @test serial.transfers.count == threaded.transfers.count
    @test serial.transfers.bytes == threaded.transfers.bytes
    for reason in propertynames(serial.transfers.reasons)
        @test getproperty(serial.transfers.reasons, reason) ==
              getproperty(threaded.transfers.reasons, reason)
    end
    return nothing
end

function first_order_gramis_assert_same_population(serial, threaded)
    @test serial.masses == threaded.masses
    @test length(serial.proposals) == length(threaded.proposals)
    for (serial_proposal, threaded_proposal) in
        zip(serial.proposals, threaded.proposals)
        @test serial_proposal.location == threaded_proposal.location
        @test serial_proposal.scale.factor == threaded_proposal.scale.factor
        @test serial_proposal.lognormalizer == threaded_proposal.lognormalizer
    end
    return nothing
end

@testset "FirstOrderGRAMIS serial and threaded public execution are deterministic" begin
    for T in (Float32, Float64)
        bank = first_order_gramis_bank(T)
        round_sizes = [12, 15]
        batches = [
            [T(mod(index, 7) - 3) / T(4) for index in 1:(2 * round_size)] for
            round_size in round_sizes
        ]
        algorithm = FirstOrderGRAMIS(
            bank;
            rounds=2,
            round_size=round_sizes,
            repulsion_strength=T[0, 0.1],
            covariance_ess_threshold=3,
        )
        target = LogTarget(
            FirstOrderGRAMISTarget{T}();
            grad=first_order_gramis_gradient!,
        )
        serial = @inferred prepare_sampler(
            FirstOrderGRAMISPrefilledRNG(deepcopy(batches)),
            target,
            algorithm;
            threaded=false,
        )
        threaded = @inferred prepare_sampler(
            FirstOrderGRAMISPrefilledRNG(deepcopy(batches)),
            target,
            algorithm;
            threaded=true,
        )

        serial_result = @inferred importance_sample!(serial)
        threaded_result = @inferred importance_sample!(threaded)

        # Threading changes only independent outer-loop scheduling; every
        # floating-point reduction retains the same within-slot operation order.
        @test serial_result.samples == threaded_result.samples
        @test serial_result.logweights == threaded_result.logweights
        @test serial_result.provenance == threaded_result.provenance
        first_order_gramis_assert_same_diagnostics(
            serial_result.diagnostics,
            threaded_result.diagnostics,
        )
        first_order_gramis_assert_same_population(
            current_proposal(serial),
            current_proposal(threaded),
        )
    end
end

@testset "FirstOrderGRAMIS threaded public execution uses private DI preparations" begin
    T = Float64
    worker_count = Threads.nthreads(:default)
    proposal_count = 8max(worker_count, 2)
    dimension = 2
    proposals = map(1:proposal_count) do proposal_slot
        phase = T(2pi * (proposal_slot - 1) / proposal_count)
        FactorGaussian(
            T[2cos(phase), 2sin(phase)],
            Matrix{T}(LinearAlgebra.I, dimension, dimension),
        )
    end
    bank = ProposalBank(proposals)
    round_size = proposal_count * (dimension + 2)
    round_sizes = [round_size, round_size]
    batches = [
        [T(mod(index, 17) - 8) / T(8) for index in 1:(dimension * round_size)]
        for _ in round_sizes
    ]
    algorithm = FirstOrderGRAMIS(
        bank;
        rounds=2,
        round_size=round_sizes,
        repulsion_strength=zero(T),
        covariance_ess_threshold=dimension + 1,
    )
    serial_trace = FirstOrderGRAMISDITrace()
    threaded_trace = FirstOrderGRAMISDITrace()
    serial = prepare_sampler(
        FirstOrderGRAMISPrefilledRNG(deepcopy(batches)),
        LogTarget(
            FirstOrderGRAMISTarget{T}(),
            FirstOrderGRAMISRecordingAD(serial_trace),
        ),
        algorithm;
        threaded=false,
    )
    threaded = prepare_sampler(
        FirstOrderGRAMISPrefilledRNG(deepcopy(batches)),
        LogTarget(
            FirstOrderGRAMISTarget{T}(),
            FirstOrderGRAMISRecordingAD(threaded_trace),
        ),
        algorithm;
        threaded=true,
    )

    serial_result = importance_sample!(serial)
    threaded_result = importance_sample!(threaded)
    @test serial_result.samples == threaded_result.samples
    @test serial_result.logweights == threaded_result.logweights
    @test serial_result.provenance == threaded_result.provenance
    first_order_gramis_assert_same_diagnostics(
        serial_result.diagnostics,
        threaded_result.diagnostics,
    )
    first_order_gramis_assert_same_population(
        current_proposal(serial),
        current_proposal(threaded),
    )

    threaded_pool = threaded.method_state.threaded_gradient.preparation
    serial_only_preparation = only(
        threaded.method_state.serial_gradient.preparation.preparations,
    )
    calls = copy(threaded_trace.calls)
    observed_thread_ids = sort!(unique(first.(calls)))
    @test observed_thread_ids == sort!(collect(Threads.threadpooltids(:default)))
    @test length(threaded_pool.preparations) == worker_count
    for thread_id in observed_thread_ids
        preparations = last.(filter(call -> first(call) == thread_id, calls))
        @test !isempty(preparations)
        @test all(preparation -> preparation === first(preparations), preparations)
        slot = threaded_pool.thread_slots[thread_id]
        @test first(preparations) === threaded_pool.preparations[slot]
        @test first(preparations) !== serial_only_preparation
    end
    used_preparations = [
        only(unique(last.(filter(call -> first(call) == thread_id, calls)))) for
        thread_id in observed_thread_ids
    ]
    @test all(
        left == right || used_preparations[left] !== used_preparations[right] for
        left in eachindex(used_preparations) for right in eachindex(used_preparations)
    )
end

@testset "FirstOrderGRAMIS two-round execution is causal and retains only q3" begin
    T = Float64
    bank = first_order_gramis_two_proposal_bank(T)
    target_value = FirstOrderGRAMISShiftedTarget(T(0.5))
    target = LogTarget(target_value; grad=first_order_gramis_shifted_gradient!)
    first_normals = T[-1, 0, 1, -1, 0, 1]
    second_normals = T[-1.5, -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5]

    one_round = prepare_sampler(
        FirstOrderGRAMISPrefilledRNG([copy(first_normals)]),
        target,
        FirstOrderGRAMIS(
            bank;
            rounds=1,
            round_size=6,
            repulsion_strength=zero(T),
            covariance_ess_threshold=2,
        );
        factor_execution=BatchedFactorExecution(),
        threaded=false,
    )
    q1 = current_proposal(one_round)
    first_result = importance_sample!(one_round)
    q2 = current_proposal(one_round)

    counted_target = FirstOrderGRAMISCountingShiftedTarget(T(0.5), 0)
    counted_gradient = FirstOrderGRAMISCountingGradient(0)
    two_round = prepare_sampler(
        FirstOrderGRAMISPrefilledRNG([
            first_normals,
            second_normals,
        ]),
        LogTarget(counted_target; grad=counted_gradient),
        FirstOrderGRAMIS(
            bank;
            rounds=2,
            round_size=[6, 8],
            repulsion_strength=zero(T),
            covariance_ess_threshold=2,
        );
        factor_execution=BatchedFactorExecution(),
        threaded=false,
    )
    result = @inferred importance_sample!(two_round)
    q3 = current_proposal(two_round)

    @test result isa WeightedSamples
    @test size(result.samples) == (1, 14)
    @test length(result.logweights) == 14
    @test result.provenance.round == vcat(fill(1, 6), fill(2, 8))
    @test result.provenance.proposal_id == vcat(
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 1, 2, 2, 2, 2],
    )
    @test result.samples[:, 1:6] == first_result.samples

    round_two_assignments = view(two_round.method_state.plan.assignments, 1:8, 2)
    expected_round_two = Matrix{T}(undef, 1, 8)
    for sample_index in 1:8
        proposal = q2.proposals[round_two_assignments[sample_index]]
        expected_round_two[1, sample_index] = proposal.location[1] +
                                              proposal.scale.factor[1, 1] *
                                              second_normals[sample_index]
    end
    @test result.samples[:, 7:14] ≈ expected_round_two rtol = 8eps(T)

    for round in 1:2
        indices = round == 1 ? (1:6) : (7:14)
        frozen = round == 1 ? q1 : q2
        counts = round == 1 ? [3, 3] : [4, 4]
        expected_logweights = map(indices) do sample_index
            sample = view(result.samples, :, sample_index)
            target_value(sample) -
            first_order_gramis_mixture_logdensity(frozen, sample, counts)
        end
        @test result.logweights[indices] ≈ expected_logweights rtol = 16eps(T)
        summary = first_order_gramis_round_summary(expected_logweights)
        @test result.diagnostics.round_ess[round] ≈ summary.ess rtol = 16eps(T)
        @test result.diagnostics.round_lognormalizers[round] ≈
              summary.lognormalizer rtol = 16eps(T)
    end

    q3_denominators = map(7:14) do sample_index
        sample = view(result.samples, :, sample_index)
        first_order_gramis_mixture_logdensity(q3, sample, [4, 4])
    end
    returned_denominators = map(7:14) do sample_index
        sample = view(result.samples, :, sample_index)
        target_value(sample) - result.logweights[sample_index]
    end
    @test any(!isapprox(left, right; rtol=64eps(T), atol=0) for
              (left, right) in zip(q3_denominators, returned_denominators))

    diagnostics = result.diagnostics
    @test diagnostics.method === :first_order_gramis
    @test diagnostics.round_sizes == [6, 8]
    @test size(diagnostics.local_ess) == (2, 2)
    @test size(diagnostics.tempering_powers) == (2, 2)
    @test size(diagnostics.fallback_status) == (2, 2)
    @test size(diagnostics.accepted_steps) == (2, 2)
    @test size(diagnostics.backtracking_trials) == (2, 2)
    @test size(diagnostics.collision_counts) == (2, 2)
    @test isempty(diagnostics.minimum_whitened_pair_distance.round)
    @test isempty(diagnostics.minimum_whitened_pair_distance.value)
    @test diagnostics.target_evaluations ==
          14 + 2 * 2 + sum(diagnostics.backtracking_trials)
    @test diagnostics.gradient_evaluations == 2 * 2
    @test counted_target.calls == diagnostics.target_evaluations
    @test counted_gradient.calls == diagnostics.gradient_evaluations
    @test diagnostics.proposal_evaluations == 2 * 14
    @test diagnostics.denominator_evaluations == 14
    @test diagnostics.failures == 0
    @test diagnostics.transfers.count == 0
    @test diagnostics.transfers.bytes == 0
end

@testset "FirstOrderGRAMIS successful results remain independent across calls" begin
    algorithm = FirstOrderGRAMIS(
        first_order_gramis_two_proposal_bank();
        rounds=2,
        round_size=[6, 8],
        repulsion_strength=[0.0, 0.1],
        covariance_ess_threshold=2,
    )
    sampler = prepare_sampler(
        Random.Xoshiro(0x7461736b3130),
        LogTarget(
            FirstOrderGRAMISShiftedTarget(0.5);
            grad=first_order_gramis_shifted_gradient!,
        ),
        algorithm;
        threaded=false,
    )
    first = importance_sample!(sampler)
    retained = deepcopy((
        samples=first.samples,
        logweights=first.logweights,
        provenance=first.provenance,
        local_ess=first.diagnostics.local_ess,
        tempering_powers=first.diagnostics.tempering_powers,
        fallback_status=first.diagnostics.fallback_status,
        accepted_steps=first.diagnostics.accepted_steps,
        backtracking_trials=first.diagnostics.backtracking_trials,
        collision_counts=first.diagnostics.collision_counts,
    ))
    second = importance_sample!(sampler)

    @test first.samples == retained.samples
    @test first.logweights == retained.logweights
    @test first.provenance == retained.provenance
    for field in keys(retained)[4:end]
        @test getproperty(first.diagnostics, field) == getproperty(retained, field)
        @test getproperty(first.diagnostics, field) !==
              getproperty(second.diagnostics, field)
    end
    @test first.samples !== second.samples
    @test first.logweights !== second.logweights
    @test first.provenance.round !== second.provenance.round
    @test second.diagnostics.round_sizes == [6, 8]
    @test second.diagnostics.minimum_whitened_pair_distance.round == [2]
    @test length(second.diagnostics.minimum_whitened_pair_distance.value) == 1
end

function first_order_gramis_steady_state_allocation(::Type{T}) where {T}
    sampler = prepare_sampler(
        Random.Xoshiro(0x7461736b3131),
        LogTarget(
            FirstOrderGRAMISShiftedTarget(T(0.5));
            grad=first_order_gramis_shifted_gradient!,
        ),
        FirstOrderGRAMIS(
            first_order_gramis_two_proposal_bank(T);
            rounds=2,
            round_size=[60, 80],
            repulsion_strength=T[0, 0.1],
            covariance_ess_threshold=2,
        );
        threaded=false,
    )
    warm_result = importance_sample!(sampler)
    holder = Ref{typeof(warm_result)}()
    holder[] = importance_sample!(sampler)
    allocation = @allocated holder[] = importance_sample!(sampler)
    return allocation, Base.summarysize(holder[])
end

@testset "FirstOrderGRAMIS public execution has bounded steady-state allocation" begin
    for T in (Float32, Float64)
        allocation, owned_summary =
            first_order_gramis_steady_state_allocation(T)
        @test allocation <= owned_summary + 2_048
    end
end

@testset "FirstOrderGRAMIS repeated calls reuse learned state and restart rounds" begin
    T = Float64
    bank = first_order_gramis_two_proposal_bank(T)
    batches = [
        T[-1, 0, 1, -1, 0, 1],
        T[-2, -1, 0, 0, 1, 2],
    ]
    sampler = prepare_sampler(
        FirstOrderGRAMISPrefilledRNG(deepcopy(batches)),
        LogTarget(
            FirstOrderGRAMISShiftedTarget(T(0.5));
            grad=first_order_gramis_shifted_gradient!,
        ),
        FirstOrderGRAMIS(
            bank;
            rounds=1,
            round_size=6,
            repulsion_strength=zero(T),
            covariance_ess_threshold=2,
        );
        threaded=false,
    )
    first = importance_sample!(sampler)
    q2 = current_proposal(sampler)
    second = importance_sample!(sampler)

    expected = Matrix{T}(undef, 1, 6)
    assignments = view(sampler.method_state.plan.assignments, 1:6, 1)
    for sample_index in 1:6
        proposal = q2.proposals[assignments[sample_index]]
        expected[1, sample_index] = proposal.location[1] +
                                    proposal.scale.factor[1, 1] *
                                    batches[2][sample_index]
    end
    @test second.samples ≈ expected rtol = 8eps(T)
    @test first.provenance.round == fill(1, 6)
    @test second.provenance.round == fill(1, 6)
    @test first.diagnostics.round_sizes == [6]
    @test second.diagnostics.round_sizes == [6]
end

@testset "FirstOrderGRAMIS successful local fallbacks remain nonfailures" begin
    T = Float64
    bank = first_order_gramis_two_proposal_bank(T)
    before_factors = [copy(proposal.scale.factor) for proposal in bank.proposals]
    sampler = prepare_sampler(
        FirstOrderGRAMISPrefilledRNG([fill(one(T), 6)]),
        LogTarget(
            FirstOrderGRAMISMeanOnlyTarget(T(-2), T(2));
            grad=first_order_gramis_zero_gradient!,
        ),
        FirstOrderGRAMIS(
            bank;
            rounds=1,
            round_size=6,
            repulsion_strength=zero(T),
            covariance_ess_threshold=2,
        );
        threaded=false,
    )
    result = importance_sample!(sampler)

    @test all(==(-Inf), result.logweights)
    @test [proposal.scale.factor for proposal in current_proposal(sampler).proposals] ==
          before_factors
    @test result.diagnostics.fallbacks.all_zero_local_weights == 2
end

@testset "FirstOrderGRAMIS public covariance blend and scale-aware ridge" begin
    T = Float64
    rate = T(0.5)
    regularization = T(0.1)
    bank = first_order_gramis_two_proposal_bank(T)
    normals = T[-1, 0, 1, 2, -2, -1, 0, 1]
    target = FirstOrderGRAMISTarget{T}()
    sampler = prepare_sampler(
        FirstOrderGRAMISPrefilledRNG([normals]),
        LogTarget(target; grad=first_order_gramis_zero_gradient!),
        FirstOrderGRAMIS(
            bank;
            rounds=1,
            round_size=8,
            repulsion_strength=zero(T),
            covariance_ess_threshold=2,
            covariance_rate=rate,
            covariance_regularization=regularization,
        );
        threaded=false,
    )
    result = importance_sample!(sampler)
    learned = current_proposal(sampler)

    for (proposal_id, (initial, fitted)) in enumerate(
        zip(bank.proposals, learned.proposals),
    )
        indices = findall(==(proposal_id), result.provenance.proposal_id)
        samples = vec(result.samples[:, indices])
        local_logs = [
            target([sample]) -
            GRAMISIS.DensityInterface.logdensityof(initial, [sample]) for
            sample in samples
        ]
        power = result.diagnostics.tempering_powers[proposal_id, 1]
        weights = exp.(power .* local_logs .- maximum(power .* local_logs))
        weights ./= sum(weights)
        center = power < one(T) ? sum(weights .* samples) : only(initial.location)
        covariance = sum(weights .* abs2.(samples .- center))
        old_covariance = abs2(only(initial.scale.factor))
        expected = (one(T) - rate) * old_covariance + rate * covariance +
                   regularization * old_covariance
        @test abs2(only(fitted.scale.factor)) ≈ expected rtol=32eps(T)
    end
end

@testset "FirstOrderGRAMIS finite extreme tempering retains the public factor" begin
    T = Float64
    bank = first_order_gramis_two_proposal_bank(T)
    before = copy(first(bank.proposals).scale.factor)
    normals = T[-1, 1, 0, 0, -1, 0, 1, 2]
    extreme_target(sample) = only(sample) < 0 ?
                             (only(sample) < T(-2.5) ? floatmax(T) : -floatmax(T)) :
                             -sum(abs2, sample) / T(2)
    sampler = prepare_sampler(
        FirstOrderGRAMISPrefilledRNG([normals]),
        LogTarget(extreme_target; grad=first_order_gramis_zero_gradient!),
        FirstOrderGRAMIS(
            bank;
            rounds=1,
            round_size=8,
            repulsion_strength=zero(T),
            covariance_ess_threshold=3,
            tempering_max_iterations=16,
        );
        threaded=false,
    )

    result = importance_sample!(sampler)
    @test first(current_proposal(sampler).proposals).scale.factor == before
    @test result.diagnostics.fallbacks.tempering == 1
end

@testset "FirstOrderGRAMIS degenerate covariance failure preserves the proposal" begin
    T = Float64
    recovery_normals = T[-1, 0, 1, 2, -2, -1, 1, 2]
    rng = FirstOrderGRAMISPrefilledRNG([zeros(T, 8), recovery_normals])
    target = FirstOrderGRAMISTarget{T}()
    sampler = prepare_sampler(
        rng,
        LogTarget(target; grad=first_order_gramis_zero_gradient!),
        FirstOrderGRAMIS(
            first_order_gramis_two_proposal_bank(T);
            rounds=1,
            round_size=8,
            repulsion_strength=zero(T),
            covariance_ess_threshold=2,
            covariance_regularization=zero(T),
        );
        threaded=false,
    )
    before = current_proposal(sampler)
    failure = try
        importance_sample!(sampler)
        nothing
    catch error
        error
    end
    retained = current_proposal(sampler)

    @test failure isa FirstOrderGRAMISRoundError
    @test failure.round == 1
    @test [proposal.location for proposal in retained.proposals] ==
          [proposal.location for proposal in before.proposals]
    @test [proposal.scale.factor for proposal in retained.proposals] ==
          [proposal.scale.factor for proposal in before.proposals]
    @test rng.next_batch == 2

    control = prepare_sampler(
        FirstOrderGRAMISPrefilledRNG([copy(recovery_normals)]),
        LogTarget(target; grad=first_order_gramis_zero_gradient!),
        FirstOrderGRAMIS(
            before;
            rounds=1,
            round_size=8,
            repulsion_strength=zero(T),
            covariance_ess_threshold=2,
            covariance_regularization=zero(T),
        );
        threaded=false,
    )
    recovered_result = importance_sample!(sampler)
    control_result = importance_sample!(control)
    recovered = current_proposal(sampler)
    expected = current_proposal(control)

    @test recovered_result.samples == control_result.samples
    @test recovered_result.logweights == control_result.logweights
    @test [proposal.location for proposal in recovered.proposals] ==
          [proposal.location for proposal in expected.proposals]
    @test [proposal.scale.factor for proposal in recovered.proposals] ==
          [proposal.scale.factor for proposal in expected.proposals]
end
