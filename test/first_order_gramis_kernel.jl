using Test
using ImportanceSamplers
import LinearAlgebra
import Random

const GRAMISKernelIS = ImportanceSamplers

struct GRAMISKernelTarget{T} end

function (::GRAMISKernelTarget{T})(sample)::T where {T}
    value = sample isa Real ? sample : only(sample)
    return -abs2(value) / T(2)
end

function gram_is_factor_policy_result(algorithm, policy)
    return importance_sample(
        Random.Xoshiro(0x4752414d49534b45),
        GRAMISKernelTarget{Float64}(),
        algorithm;
        threaded=false,
        factor_execution=policy,
    )
end

@testset "shared factor-policy output characterization before GRAMIS adaptation" begin
    bank = ProposalBank(
        [
            FactorGaussian([-1.0], reshape([0.75], 1, 1)),
            FactorGaussian([1.0], reshape([1.25], 1, 1)),
        ],
        [1.0, 1.0],
    )
    algorithms = (
        static=ImportanceSampling(
            bank;
            nsamples=8,
            mis_scheme=StratifiedMixture(),
        ),
        dm_pmc=DeterministicMixturePMC(bank; rounds=1, round_size=8),
        amis=AMIS(first(bank.proposals); rounds=1, round_size=8),
    )

    for (name, algorithm) in pairs(algorithms)
        fused = gram_is_factor_policy_result(algorithm, FusedFactorExecution())
        batched = gram_is_factor_policy_result(algorithm, BatchedFactorExecution())

        @test fused.samples == batched.samples
        @test fused.logweights == batched.logweights
        @test fused.provenance == batched.provenance
        @test fused.diagnostics.factor_execution_policy === :fused
        @test batched.diagnostics.factor_execution_policy === :batched
        @test length(fused) == 8
    end
end

function gram_is_scalar_logdensity(sample, location, scale)
    T = promote_type(typeof(sample), typeof(location), typeof(scale))
    standardized = (sample - location) / scale
    return -log(scale) - log(T(2pi)) / T(2) - abs2(standardized) / T(2)
end

function gram_is_one_round_oracle(
    ::Type{T},
    locations,
    scales,
    normals,
    assignments,
    counts,
) where {T}
    sample_count = length(assignments)
    samples = Vector{T}(undef, sample_count)
    logtargets = Vector{T}(undef, sample_count)
    returned_logweights = Vector{T}(undef, sample_count)
    local_logweights = Vector{T}(undef, sample_count)
    generating_logdensities = Vector{T}(undef, sample_count)
    for sample_index in eachindex(assignments)
        slot = assignments[sample_index]
        sample = locations[slot] + scales[slot] * normals[sample_index]
        logtarget = -abs2(sample) / T(3) + sample / T(5) - T(0.7)
        logmixture = T(-Inf)
        for proposal_slot in eachindex(locations)
            logcoefficient = log(T(counts[proposal_slot]) / T(sample_count))
            logdensity = gram_is_scalar_logdensity(
                sample,
                locations[proposal_slot],
                scales[proposal_slot],
            )
            logmixture = GRAMISKernelIS.LogExpFunctions.logaddexp(
                logmixture,
                logcoefficient + logdensity,
            )
        end
        generating_logdensity = gram_is_scalar_logdensity(
            sample,
            locations[slot],
            scales[slot],
        )
        samples[sample_index] = sample
        logtargets[sample_index] = logtarget
        returned_logweights[sample_index] = logtarget - logmixture
        local_logweights[sample_index] = logtarget - generating_logdensity
        generating_logdensities[sample_index] = generating_logdensity
    end
    return (;
        samples,
        logtargets,
        returned_logweights,
        local_logweights,
        generating_logdensities,
        proposal_ids=assignments,
    )
end

struct GRAMISFrozenRoundTarget{T}
    calls::Threads.Atomic{Int}
end

GRAMISFrozenRoundTarget{T}() where {T} =
    GRAMISFrozenRoundTarget{T}(Threads.Atomic{Int}(0))

function (target::GRAMISFrozenRoundTarget{T})(sample)::T where {T}
    Threads.atomic_add!(target.calls, 1)
    value = only(sample)
    return -abs2(value) / T(3) + value / T(5) - T(0.7)
end

function run_gram_is_frozen_round(
    ::Type{T},
    execution,
    factor_execution;
    failed_sample=nothing,
) where {T}
    locations = T[-2, 0.5, 2.5]
    scales = T[0.75, 1.25, 0.5]
    bank = GRAMISKernelIS._first_order_gramis_factor_bank(ProposalBank([
        FactorGaussian(T[locations[1]], reshape(T[scales[1]], 1, 1)),
        FactorGaussian(T[locations[2]], reshape(T[scales[2]], 1, 1)),
        FactorGaussian(T[locations[3]], reshape(T[scales[3]], 1, 1)),
    ]))
    normals = T[-1.5, -0.25, 0.5, 1.25, -1, 0.25, 1.5, -0.75, 0, 1]
    assignments = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    counts = [4, 3, 3]
    sample_count = length(assignments)
    samples = fill(T(101), 1, sample_count)
    returned_logweights = fill(T(102), sample_count)
    local_logweights = fill(T(103), sample_count)
    generating_logdensities = fill(T(104), sample_count)
    proposal_ids = fill(105, sample_count)
    round_ids = fill(106, sample_count)
    failures = zeros(UInt64, 3)
    sampled_normals = copy(normals)
    isnothing(failed_sample) || (sampled_normals[failed_sample] = T(Inf))
    denominator = GRAMISKernelIS._RealizedMixtureDenominator(
        reshape(log.(T.(counts) ./ T(sample_count)), :, 1),
        1,
    )
    counted_target = GRAMISFrozenRoundTarget{T}()
    target = GRAMISKernelIS._NativeDeviceTarget{
        T,
        GRAMISFrozenRoundTarget{T},
    }(counted_target)
    GRAMISKernelIS._first_order_gramis_sample_round!(
        samples,
        returned_logweights,
        local_logweights,
        generating_logdensities,
        proposal_ids,
        round_ids,
        failures,
        sampled_normals,
        target,
        bank,
        assignments,
        denominator,
        GRAMISKernelIS._allocate_mis_solve_scratch(normals, bank, sample_count),
        7,
        execution,
        GRAMISKernelIS.MLDataDevices.CPUDevice(),
        factor_execution,
    )
    oracle = gram_is_one_round_oracle(
        T,
        locations,
        scales,
        normals,
        assignments,
        counts,
    )
    return (;
        samples,
        returned_logweights,
        local_logweights,
        generating_logdensities,
        proposal_ids,
        round_ids,
        failures,
        target_calls=counted_target.calls[],
        oracle,
    )
end

@testset "failed frozen-round samples clear fused and batched adaptation state" begin
    failed_sample = 3
    for T in (Float32, Float64)
        results = (
            fused_serial=run_gram_is_frozen_round(
                T,
                GRAMISKernelIS._SerialCPUExecution(),
                FusedFactorExecution();
                failed_sample,
            ),
            fused_threaded=run_gram_is_frozen_round(
                T,
                GRAMISKernelIS._ThreadedCPUExecution(),
                FusedFactorExecution();
                failed_sample,
            ),
            batched_serial=run_gram_is_frozen_round(
                T,
                GRAMISKernelIS._SerialCPUExecution(),
                BatchedFactorExecution();
                failed_sample,
            ),
            batched_threaded=run_gram_is_frozen_round(
                T,
                GRAMISKernelIS._ThreadedCPUExecution(),
                BatchedFactorExecution();
                failed_sample,
            ),
        )

        for result in results
            decoded = GRAMISKernelIS._decode_native_failure(
                result.failures[1],
                result.failures[2],
            )
            @test result.returned_logweights[failed_sample] == T(-Inf)
            @test result.local_logweights[failed_sample] == T(-Inf)
            @test result.generating_logdensities[failed_sample] == T(-Inf)
            @test iszero(result.proposal_ids[failed_sample])
            @test iszero(result.round_ids[failed_sample])
            @test decoded.first_logical_index == failed_sample
            @test decoded.reason_bits == GRAMISKernelIS._NATIVE_GENERATED_NONFINITE
        end

        reference = results.batched_serial
        for result in results
            @test result.returned_logweights[failed_sample] ==
                  reference.returned_logweights[failed_sample]
            @test result.local_logweights[failed_sample] ==
                  reference.local_logweights[failed_sample]
            @test result.generating_logdensities[failed_sample] ==
                  reference.generating_logdensities[failed_sample]
            @test result.proposal_ids[failed_sample] ==
                  reference.proposal_ids[failed_sample]
            @test result.round_ids[failed_sample] == reference.round_ids[failed_sample]
            @test result.failures == reference.failures
        end
    end
end

@testset "FirstOrderGRAMIS frozen round uses realized and local denominators" begin
    for T in (Float32, Float64), factor_execution in (
        FusedFactorExecution(),
        BatchedFactorExecution(),
    )
        serial = run_gram_is_frozen_round(
            T,
            GRAMISKernelIS._SerialCPUExecution(),
            factor_execution,
        )
        threaded = run_gram_is_frozen_round(
            T,
            GRAMISKernelIS._ThreadedCPUExecution(),
            factor_execution,
        )
        oracle = serial.oracle

        @test vec(serial.samples) ≈ oracle.samples rtol = 8eps(T)
        @test serial.returned_logweights ≈
              oracle.returned_logweights rtol = 32eps(T)
        @test serial.local_logweights ≈ oracle.local_logweights rtol = 32eps(T)
        @test serial.generating_logdensities ≈
              oracle.generating_logdensities rtol = 16eps(T)
        @test serial.local_logweights .+ serial.generating_logdensities ≈
              oracle.logtargets rtol = 16eps(T)
        @test serial.proposal_ids == oracle.proposal_ids
        @test serial.round_ids == fill(7, 10)
        @test iszero(serial.failures)
        @test serial.target_calls == 10
        @test serial.samples == threaded.samples
        @test serial.returned_logweights == threaded.returned_logweights
        @test serial.local_logweights == threaded.local_logweights
        @test serial.generating_logdensities == threaded.generating_logdensities
        @test serial.proposal_ids == threaded.proposal_ids
        @test serial.round_ids == threaded.round_ids
        @test serial.failures == threaded.failures
        @test threaded.target_calls == 10

        returned_denominators = oracle.logtargets .- serial.returned_logweights
        local_denominators = oracle.logtargets .- serial.local_logweights
        @test returned_denominators ≈
              oracle.logtargets .- oracle.returned_logweights rtol = 16eps(T)
        @test local_denominators ≈ oracle.generating_logdensities rtol = 16eps(T)
        @test any(!isapprox(returned_denominators[index], local_denominators[index]) for
                  index in eachindex(returned_denominators))
    end
end

@testset "realized-mixture finite-law conditional unbiasedness identity" begin
    counts = [4, 3, 3]
    total = 10
    proposals = Rational{Int}[
        1//2 1//4 1//4
        1//4 1//2 1//4
        1//4 1//4 1//2
    ]
    pi_h = Rational{Int}[2//5, -1//7, 3//11]
    psi = [
        sum((counts[slot] // total) * proposals[slot, point] for slot in 1:3) for
        point in 1:3
    ]
    pointwise = [
        sum(
            (counts[slot] // total) * proposals[slot, point] *
            pi_h[point] / psi[point] for slot in 1:3
        ) for point in 1:3
    ]

    @test vec(sum(proposals; dims=2)) == fill(1//1, 3)
    @test psi == Rational{Int}[7//20, 13//40, 13//40]
    @test pointwise == pi_h
    @test sum(pointwise) == 204//385

    round_sizes = [10, 13]
    round_sums = Rational{Int}[17//6, -5//4]
    combined = sum(
        (round_sizes[round] // sum(round_sizes)) *
        (round_sums[round] / round_sizes[round]) for round in 1:2
    )
    @test combined == sum(round_sums) / sum(round_sizes) == 19//276
end

function gram_is_local_covariance_fixture(::Type{T}) where {T}
    bank = ProposalBank([
        FactorGaussian(T[10], reshape(T[2], 1, 1)),
        FactorGaussian(T[20], reshape(T[3], 1, 1)),
        FactorGaussian(T[100], reshape(T[4], 1, 1)),
    ])
    algorithm = FirstOrderGRAMIS(
        bank;
        rounds=1,
        round_size=12,
        repulsion_strength=zero(T),
        covariance_ess_threshold=3,
    )
    state = GRAMISKernelIS._prepare_method_state(algorithm)
    state.workspace.samples .= reshape(
        T[0, 2, 4, 6, 0, 2, 4, 6, 0, 1, 2, 4],
        1,
        :,
    )
    state.workspace.local_logweights .= T[
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -Inf,
        log(T(8)),
        0,
        0,
        0,
    ]
    return state
end

@testset "FirstOrderGRAMIS CAIS covariance centering oracle" begin
    for T in (Float32, Float64), execution in (
        GRAMISKernelIS._SerialCPUExecution(),
        GRAMISKernelIS._ThreadedCPUExecution(),
    )
        state = gram_is_local_covariance_fixture(T)
        locations = copy(state.run.locations)
        factors = copy(state.run.factors)
        lognormalizers = copy(state.run.lognormalizers)

        @test @inferred(
            GRAMISKernelIS._fit_local_covariances!(state, 1, execution)
        ) === nothing

        workspace = state.workspace
        @test workspace.factor_status == fill(
            GRAMISKernelIS._GRAMIS_COVARIANCE_READY,
            3,
        )
        @test workspace.local_ess[1] == T(4)
        @test workspace.local_ess[2] == T(3)
        @test workspace.tempering_powers[1:2] == ones(T, 2)
        @test workspace.normalized_weights[1:4] == fill(T(0.25), 4)
        @test workspace.normalized_weights[5:8] == T[1 / 3, 1 / 3, 1 / 3, 0]
        @test workspace.covariances[1, 1, 1] ≈ T(54) rtol = 8eps(T)
        @test workspace.covariances[1, 1, 2] ≈ T(980 / 3) rtol = 8eps(T)

        expected_power = T(0.5283203125)
        @test workspace.tempering_powers[3] == expected_power
        @test workspace.local_ess[3] >= T(3)
        upper_power = T(0.52838134765625)
        upper_ratio = T(8)^upper_power
        upper_ess = abs2(upper_ratio + T(3)) / (abs2(upper_ratio) + T(3))
        @test upper_ess < T(3)
        @test workspace.covariances[1, 1, 3] ≈
              T(2.1388893102660655) rtol = 64eps(T)

        @test state.run.locations == locations
        @test state.run.factors == factors
        @test state.run.lognormalizers == lognormalizers
    end
end

function gram_is_stability_fixture(::Type{T}, execution) where {T}
    bank = ProposalBank([
        FactorGaussian(T[0, 0], T[2 0; 1 3]),
        FactorGaussian(T[10, 10], T[1 0; 0 1]),
    ])
    state = GRAMISKernelIS._prepare_method_state(FirstOrderGRAMIS(
        bank;
        rounds=1,
        round_size=8,
        repulsion_strength=zero(T),
        covariance_ess_threshold=3,
        tempering_tolerance=T(1.0e-8),
        tempering_max_iterations=16,
    ))
    state.workspace.samples .= T[
        -1 1 0 0 8 10 12 10
        0 0 -1 1 10 8 10 12
    ]
    state.workspace.local_logweights .= T[
        floatmax(T),
        -floatmax(T),
        -floatmax(T),
        -floatmax(T),
        0,
        0,
        0,
        0,
    ]
    GRAMISKernelIS._fit_local_covariances!(state, 1, execution)
    return state
end

@testset "FirstOrderGRAMIS bounded tempering stability" begin
    serial = gram_is_stability_fixture(
        Float64,
        GRAMISKernelIS._SerialCPUExecution(),
    )
    threaded = gram_is_stability_fixture(
        Float64,
        GRAMISKernelIS._ThreadedCPUExecution(),
    )

    for state in (serial, threaded)
        workspace = state.workspace
        @test state.plan.counts[:, 1] == [4, 4]
        @test state.covariance_ess_threshold[:, 1] == [3, 3]
        @test workspace.factor_status == UInt8[
            GRAMISKernelIS._GRAMIS_TEMPERING_FALLBACK,
            GRAMISKernelIS._GRAMIS_COVARIANCE_READY,
        ]
        @test workspace.tempering_powers == [0.0, 1.0]
        @test workspace.local_ess == [1.0, 4.0]
        @test workspace.covariances[:, :, 1] == [4.0 2.0; 2.0 10.0]
        @test workspace.covariances[:, :, 2] == [2.0 0.0; 0.0 2.0]
        @test all(
            covariance -> LinearAlgebra.issymmetric(covariance),
            eachslice(workspace.covariances; dims=3),
        )
    end

    @test serial.workspace.normalized_weights ==
          threaded.workspace.normalized_weights
    @test serial.workspace.local_ess == threaded.workspace.local_ess
    @test serial.workspace.tempering_powers ==
          threaded.workspace.tempering_powers
    @test serial.workspace.factor_status == threaded.workspace.factor_status
    @test serial.workspace.covariances == threaded.workspace.covariances
end

@testset "FirstOrderGRAMIS m - 1 raw ESS threshold" begin
    T = Float64
    bank = ProposalBank([
        FactorGaussian(T[0, 0], T[1 0; 0 1]),
        FactorGaussian(T[10, 10], T[1 0; 0 1]),
    ])
    state = GRAMISKernelIS._prepare_method_state(FirstOrderGRAMIS(
        bank;
        rounds=1,
        round_size=10,
        repulsion_strength=zero(T),
        covariance_ess_threshold=4,
    ))
    state.workspace.samples .= 0
    state.workspace.local_logweights .= [
        0,
        0,
        0,
        0,
        -Inf,
        0,
        0,
        0,
        0,
        0,
    ]

    GRAMISKernelIS._fit_local_covariances!(
        state,
        1,
        GRAMISKernelIS._SerialCPUExecution(),
    )

    @test state.plan.counts[:, 1] == [5, 5]
    @test state.covariance_ess_threshold[:, 1] == [4, 4]
    @test state.workspace.local_ess == [4.0, 5.0]
    @test state.workspace.tempering_powers == [1.0, 1.0]
    @test state.workspace.factor_status == fill(
        GRAMISKernelIS._GRAMIS_COVARIANCE_READY,
        2,
    )
end

@testset "FirstOrderGRAMIS computes local group starts once" begin
    counts = [3 5; 4 3; 5 4]
    starts = zeros(Int, 3)

    @test @inferred(GRAMISKernelIS._local_group_starts!(
        starts,
        counts,
        2,
        GRAMISKernelIS._SerialCPUExecution(),
    )) === nothing
    @test starts == [1, 6, 9]
end

@testset "FirstOrderGRAMIS Float32 accepted ESS matches published weights" begin
    T = Float32
    bank = ProposalBank([
        FactorGaussian(T[0], reshape(T[1], 1, 1)),
        FactorGaussian(T[10], reshape(T[1], 1, 1)),
    ])
    state = GRAMISKernelIS._prepare_method_state(FirstOrderGRAMIS(
        bank;
        rounds=1,
        round_size=16,
        repulsion_strength=zero(T),
        covariance_ess_threshold=3,
    ))
    state.workspace.samples .= zero(T)
    state.workspace.local_logweights .= T[
        -2.2072423,
        -2.450517,
        0.959949,
        -1.0785025,
        2.3158128,
        -2.0148082,
        0.7158158,
        -2.0185878,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]

    GRAMISKernelIS._fit_local_covariances!(
        state,
        1,
        GRAMISKernelIS._SerialCPUExecution(),
    )

    workspace = state.workspace
    accepted_weights = view(workspace.normalized_weights, 1:8)
    accepted_power_ess = GRAMISKernelIS._gramis_power_ess(
        workspace.local_logweights,
        1,
        8,
        workspace.tempering_powers[1],
        T,
    )
    @test workspace.factor_status[1] ==
          GRAMISKernelIS._GRAMIS_COVARIANCE_READY
    @test workspace.local_ess[1] >= T(3)
    @test workspace.local_ess[1] == accepted_power_ess
    @test sum(accepted_weights) ≈ one(T) rtol = 4eps(T)
    maximum_index = argmax(view(workspace.local_logweights, 1:8))
    for index in eachindex(accepted_weights)
        expected_ratio = exp(
            workspace.tempering_powers[1] *
            (workspace.local_logweights[index] -
             workspace.local_logweights[maximum_index]),
        )
        @test accepted_weights[index] / accepted_weights[maximum_index] ≈
              expected_ratio rtol = 8eps(T)
    end
end
