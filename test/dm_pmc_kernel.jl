using Test
using ImportanceSamplers
import Random

include("support/dm_pmc.jl")

const DMPMCKernelIS = ImportanceSamplers

@testset "DM-PMC private execution types" begin
    @test isdefined(DMPMCKernelIS, :_DMPMCRandomBuffers)
    @test isdefined(DMPMCKernelIS, :_DMPMCWorkspace)
    @test isdefined(DMPMCKernelIS, :_DMPMCRoundDenominator)
end

function run_dm_pmc_prefilled_round(bank, normals, assignments, target, execution)
    T = eltype(bank.locations)
    samples = Vector{T}(undef, length(assignments))
    logweights = Vector{T}(undef, length(assignments))
    proposal_ids = Vector{Int}(undef, length(assignments))
    failure_storage = zeros(UInt64, 3)
    denominator = DMPMCKernelIS._DMPMCRoundDenominator(
        reshape(T[log(T(3) / T(4)), log(T(1) / T(4))], 2, 1),
        1,
    )
    target_evaluator = DMPMCKernelIS._NativeDeviceTarget{T,typeof(target)}(target)
    DMPMCKernelIS._launch_mis_round!(
        samples,
        logweights,
        proposal_ids,
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

        DMPMCKernelIS._launch_dm_pmc_resampling!(
            cdf,
            uniforms,
            ancestors,
            samples,
            candidates,
            DMPMCKernelIS._SerialCPUExecution(),
        )

        @test cdf[end] == one(T)
        @test ancestors == expected_ancestors == [1, 1, 2, 4]
        @test candidates == samples[:, expected_ancestors]
        @test count(==(1), ancestors) == 2
    end
end
