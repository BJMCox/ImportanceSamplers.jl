using Test
using ImportanceSamplers
import Random

include("support/dm_pmc.jl")

const DMPMCKernelIS = ImportanceSamplers

@testset "DM-PMC private execution types" begin
    @test isdefined(DMPMCKernelIS, :_DMPMCRandomBuffers)
    @test isdefined(DMPMCKernelIS, :_DMPMCWorkspace)
    @test isdefined(DMPMCKernelIS, :_RealizedMixtureDenominator)
end

@testset "realized mixture denominator characterization" begin
    for T in (Float32, Float64)
        bank = DMPMCKernelIS._pack_native_gaussian_bank(
            ProposalBank(
                [SphericalGaussian(T(-1), one(T)), SphericalGaussian(T(1), one(T))],
                T[1, 1],
            ),
        )
        logcoefficients = T[
            log(T(3) / T(4)) log(T(1) / T(4))
            log(T(1) / T(4)) log(T(3) / T(4))
        ]
        left = DMPMCKernelIS._RealizedMixtureDenominator(logcoefficients, 1)
        right = DMPMCKernelIS._RealizedMixtureDenominator(logcoefficients, 2)
        sample = T[-1]
        no_scratch = DMPMCKernelIS._NoMISSolveScratch()
        left_value, left_generating, left_reason =
            DMPMCKernelIS._mis_logdenominator_core(
                T,
                bank,
                left,
                1,
                sample,
                no_scratch,
                1,
            )
        right_value, right_generating, right_reason =
            DMPMCKernelIS._mis_logdenominator_core(
                T,
                bank,
                right,
                1,
                sample,
                no_scratch,
                1,
            )
        lognormalizer = -T(0.5) * log(T(2pi))

        @test DMPMCKernelIS._factor_batch_logcoefficients(bank, left) ==
              view(logcoefficients, :, 1)
        @test DMPMCKernelIS._mis_term_bounds(bank, left, 1) == (1, 2)
        @test DMPMCKernelIS._mis_denominator_term(T, bank, left, 2) ==
              (2, logcoefficients[2, 1])
        @test left_value ≈
              lognormalizer + log(T(3) / T(4) + T(1) / T(4) * exp(T(-2)))
        @test right_value ≈
              lognormalizer + log(T(1) / T(4) + T(3) / T(4) * exp(T(-2)))
        @test left_generating == right_generating == lognormalizer
        @test iszero(left_reason)
        @test iszero(right_reason)
    end
end

function run_dm_pmc_prefilled_round(bank, normals, assignments, target, execution)
    T = eltype(bank.locations)
    samples = Vector{T}(undef, length(assignments))
    logweights = Vector{T}(undef, length(assignments))
    proposal_ids = Vector{Int}(undef, length(assignments))
    failure_storage = zeros(UInt64, 3)
    denominator = DMPMCKernelIS._RealizedMixtureDenominator(
        reshape(T[log(T(3) / T(4)), log(T(1) / T(4))], 2, 1),
        1,
    )
    target_evaluator = DMPMCKernelIS._NativeDeviceTarget{T,typeof(target)}(target)
    DMPMCKernelIS._launch_mis_round!(
        samples,
        DMPMCKernelIS._MISRoundOutput(logweights, proposal_ids),
        failure_storage,
        normals,
        target_evaluator,
        bank,
        assignments,
        denominator,
        DMPMCKernelIS._allocate_mis_solve_scratch(
            normals,
            bank,
            length(assignments),
        ),
        execution,
    )
    return (; samples, logweights, proposal_ids, failure_storage)
end

function dm_pmc_prefilled_launch_allocated!(
    result,
    bank,
    normals,
    assignments,
    target,
)
    T = eltype(result.logweights)
    denominator = DMPMCKernelIS._RealizedMixtureDenominator(
        reshape(T[log(T(3) / T(4)), log(T(1) / T(4))], 2, 1),
        1,
    )
    target_evaluator = DMPMCKernelIS._NativeDeviceTarget{T,typeof(target)}(target)
    output = DMPMCKernelIS._MISRoundOutput(
        result.logweights,
        result.proposal_ids,
    )
    solve_scratch = DMPMCKernelIS._allocate_mis_solve_scratch(
        normals,
        bank,
        length(assignments),
    )
    fill!(result.failure_storage, zero(UInt64))
    return @allocated DMPMCKernelIS._launch_mis_round!(
        result.samples,
        output,
        result.failure_storage,
        normals,
        target_evaluator,
        bank,
        assignments,
        denominator,
        solve_scratch,
        DMPMCKernelIS._SerialCPUExecution(),
    )
end

@testset "DM-PMC realized-count denominator reuses the MIS round" begin
    for T in (Float32, Float64)
        bank = DMPMCKernelIS._pack_native_gaussian_bank(
            ProposalBank(
                [SphericalGaussian(T(-1), one(T)), SphericalGaussian(T(1), one(T))],
                T[1, 1],
            ),
        )
        normals = T[-1, 0, 1, 0.5]
        assignments = [1, 1, 1, 2]
        target = DMPMCMixtureTarget(
            T[-1, 1],
            T[1, 1],
            T[log(T(3) / T(4)), log(T(1) / T(4))],
        )
        serial = @inferred run_dm_pmc_prefilled_round(
            bank,
            normals,
            assignments,
            target,
            DMPMCKernelIS._SerialCPUExecution(),
        )
        threaded = @inferred run_dm_pmc_prefilled_round(
            bank,
            normals,
            assignments,
            target,
            DMPMCKernelIS._ThreadedCPUExecution(),
        )

        @test serial.samples == T[-2, -1, 0, 1.5]
        @test all(iszero, serial.logweights)
        @test serial.proposal_ids == [1, 1, 1, 2]
        @test iszero(serial.failure_storage)
        @test threaded == serial
        @test dm_pmc_prefilled_launch_allocated!(
            serial,
            bank,
            normals,
            assignments,
            target,
        ) == 320

        failed_normals = copy(normals)
        failed_normals[3] = T(Inf)
        serial_failure = run_dm_pmc_prefilled_round(
            bank,
            failed_normals,
            assignments,
            target,
            DMPMCKernelIS._SerialCPUExecution(),
        )
        threaded_failure = run_dm_pmc_prefilled_round(
            bank,
            failed_normals,
            assignments,
            target,
            DMPMCKernelIS._ThreadedCPUExecution(),
        )
        decoded = DMPMCKernelIS._decode_native_failure(
            serial_failure.failure_storage[1],
            serial_failure.failure_storage[2],
        )
        @test decoded.first_logical_index == 3
        @test decoded.reason_bits == DMPMCKernelIS._NATIVE_GENERATED_NONFINITE
        @test threaded_failure.failure_storage ==
              serial_failure.failure_storage
    end
end

@testset "DM-PMC multinomial selection and gather kernels" begin
    for T in (Float32, Float64)
        cdf = T[0.2, 0.5, 0.9, 0.9]
        uniforms = T[0.1, 0.2, 0.2001, 0.95]
        samples = reshape(T[1, 2, 3, 4, 11, 12, 13, 14], 2, 4)
        ancestors = zeros(Int, 4)
        candidates = zeros(T, 2, 4)
        expected_ancestors = dm_pmc_multinomial_oracle(
            T[0.2, 0.5, 0.9, 1],
            uniforms,
        )

        DMPMCKernelIS._resample_and_gather!(
            cdf,
            uniforms,
            ancestors,
            samples,
            candidates,
            DMPMCKernelIS._SerialCPUExecution(),
        )

        @test cdf[end] == one(T)
        @test ancestors == expected_ancestors == [1, 2, 2, 4]
        @test candidates == samples[:, expected_ancestors]
        @test count(==(2), ancestors) == 2
    end
end

@testset "DM-PMC multinomial CDF plateaus use strict upper bounds" begin
    for T in (Float32, Float64)
        samples = T[10, 20, 30, 40]
        cases = (
            (
                cdf=T[0, 0, 0.5, 1],
                uniforms=T[0, prevfloat(T(0.5)), T(0.5), prevfloat(one(T))],
                ancestors=[3, 3, 4, 4],
            ),
            (
                cdf=T[0.25, 1, 1, 1],
                uniforms=T[0, T(0.25), nextfloat(T(0.25)), prevfloat(one(T))],
                ancestors=[1, 2, 2, 2],
            ),
            (
                cdf=T[0.25, 0.5, 0.75, 1],
                uniforms=T[one(T)],
                ancestors=[4],
            ),
        )
        for case in cases
            ancestors = zeros(Int, length(case.uniforms))
            candidates = zeros(T, length(case.uniforms))

            DMPMCKernelIS._resample_and_gather!(
                copy(case.cdf),
                case.uniforms,
                ancestors,
                samples,
                candidates,
                DMPMCKernelIS._SerialCPUExecution(),
            )

            @test ancestors == case.ancestors
            @test candidates == samples[case.ancestors]
        end
    end
end

@testset "DM-PMC round ESS is stable for extreme finite weights" begin
    for T in (Float32, Float64)
        equal_positive = DMPMCKernelIS._logweight_summary(
            T[floatmax(T), floatmax(T)],
        )
        equal_negative = DMPMCKernelIS._logweight_summary(
            T[-floatmax(T), -floatmax(T)],
        )
        concentrated = DMPMCKernelIS._logweight_summary(
            T[floatmax(T), -floatmax(T)],
        )
        all_zero = DMPMCKernelIS._logweight_summary(T[-Inf, -Inf])

        @test equal_positive.ess === T(2)
        @test equal_negative.ess === T(2)
        @test concentrated.ess === one(T)
        @test all_zero.ess === zero(T)
        @test isfinite(equal_positive.lognormalizer)
        @test isfinite(equal_negative.lognormalizer)
        @test isfinite(concentrated.lognormalizer)
        @test all_zero.lognormalizer === T(-Inf)
    end
end

@testset "DM-PMC reported explicit transfers retain bounded reasons" begin
    transfers = DMPMCKernelIS._ResultTransferCounter(0, 0)
    DMPMCKernelIS._record_reported_transfer!(
        transfers,
        1,
        3sizeof(UInt64),
        Val(:failure_snapshot),
    )
    for reason in (
        Val(:cdf_maximum),
        Val(:cdf_sum),
        Val(:logweight_maximum),
        Val(:logweight_scaled_sum),
        Val(:logweight_scaled_square_sum),
    )
        DMPMCKernelIS._record_scalar_transfer!(transfers, Float32, reason)
    end

    @test transfers.count == 6
    @test transfers.bytes == 3sizeof(UInt64) + 5sizeof(Float32)
    @test fieldnames(typeof(transfers.reasons)) == (
        :failure_snapshot,
        :cdf_maximum,
        :cdf_sum,
        :logweight_maximum,
        :logweight_scaled_sum,
        :logweight_scaled_square_sum,
        :covariance_diagnostic,
    )
    @test transfers.reasons.failure_snapshot.count == 1
    @test transfers.reasons.failure_snapshot.bytes == 3sizeof(UInt64)
    for reason in fieldnames(typeof(transfers.reasons))[2:(end - 1)]
        @test getfield(transfers.reasons, reason).count == 1
        @test getfield(transfers.reasons, reason).bytes == sizeof(Float32)
    end
    @test iszero(transfers.reasons.covariance_diagnostic.count)
    @test iszero(transfers.reasons.covariance_diagnostic.bytes)
end
