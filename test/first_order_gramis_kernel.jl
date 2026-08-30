using Test
using ImportanceSamplers
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
