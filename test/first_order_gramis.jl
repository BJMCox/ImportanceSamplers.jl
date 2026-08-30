using Test
using ImportanceSamplers
import MLDataDevices
import Random

include("support/first_order_gramis.jl")

const GRAMISIS = ImportanceSamplers

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

    @test state isa GRAMISIS._PreparedFirstOrderGRAMIS
    @test workspace isa GRAMISIS._FirstOrderGRAMISWorkspace
    @test committed isa GRAMISIS._PackedFactorGaussianBank
    @test run isa GRAMISIS._PackedFactorGaussianBank
    @test candidate isa GRAMISIS._PackedFactorGaussianBank
    @test size(committed.locations) == (2, 3)
    @test size(committed.factors) == (2, 2, 3)
    @test size(committed.lognormalizers) == (3,)

    for field in (:locations, :factors, :lognormalizers)
        @test getfield(committed, field) !== getfield(run, field)
        @test getfield(committed, field) !== getfield(candidate, field)
        @test getfield(run, field) !== getfield(candidate, field)
    end
    for field in (:logmasses, :cdf, :proposal_ids)
        @test getfield(committed, field) === getfield(run, field)
        @test getfield(committed, field) === getfield(candidate, field)
    end

    @test size(workspace.samples) == (2, 17)
    @test size(workspace.local_logweights) == (17,)
    @test size(workspace.normalized_weights) == (17,)
    @test size(workspace.local_starts) == (3,)
    @test size(workspace.covariances) == (2, 2, 3)
    @test size(workspace.gradients) == (2, 3)
    @test size(workspace.frozen_values) == (3,)
    @test size(workspace.candidate_values) == (3,)
    @test size(workspace.moves) == (2, 3)
    @test size(workspace.active_mask) == (3,)
    @test size(workspace.steps) == (3,)
    @test size(workspace.repulsion) == (2, 3)
    @test size(workspace.factor_status) == (3,)
    @test size(workspace.local_ess) == (3,)
    @test size(workspace.tempering_powers) == (3,)
    @test size(workspace.backtracking_trials) == (3,)
    @test size(workspace.collision_counts) == (3,)
    @test workspace.frozen_values !== workspace.candidate_values
    @test workspace.moves !== workspace.gradients
    @test workspace.moves !== workspace.repulsion
    @test workspace.moves !== candidate.locations

    for type in (typeof(state), typeof(workspace), typeof(committed))
        @test isconcretetype(type)
        @test all(isconcretetype, fieldtypes(type))
        @test !(Any in fieldtypes(type))
    end

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
