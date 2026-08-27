using Test
using ImportanceSamplers
import LogExpFunctions

const AMISKernelIS = ImportanceSamplers

mutable struct AMISKernelCountingTarget{T}
    calls::Vector{Int}
end

struct AMISKernelTarget{T} end

function (::AMISKernelTarget{T})(sample)::T where {T}
    value = sample isa Real ? sample : only(sample)
    return -abs2(value) / T(3)
end

function (target::AMISKernelCountingTarget{T})(sample)::T where {T}
    target.calls[1] += 1
    value = sample isa Real ? sample : only(sample)
    return -abs2(value) / T(3)
end

struct AMISKernelCountingHistory{H,C}
    history::H
    calls::C
end

if isdefined(AMISKernelIS, :_launch_prefilled_amis_round!)
    @eval AMISKernelIS begin
        @inline _mis_dimension(history::Main.AMISKernelCountingHistory) =
            _mis_dimension(history.history)

        @inline function _native_gaussian_coordinate(
            history::Main.AMISKernelCountingHistory,
            normals,
            offset,
            coordinate,
            proposal_slot,
        )
            return _native_gaussian_coordinate(
                history.history,
                normals,
                offset,
                coordinate,
                proposal_slot,
            )
        end

        @inline function _mis_proposal_logdensity(
            ::Type{T},
            history::Main.AMISKernelCountingHistory,
            sample,
            proposal_slot,
            solve_scratch,
            sample_index,
        ) where {T}
            @inbounds history.calls[proposal_slot] += 1
            return _mis_proposal_logdensity(
                T,
                history.history,
                sample,
                proposal_slot,
                solve_scratch,
                sample_index,
            )
        end
    end
end

function amis_kernel_logdensity(sample, mean, scale)
    T = promote_type(typeof(sample), typeof(mean), typeof(scale))
    standardized = (sample - mean) / scale
    return -log(scale) - log(T(2pi)) / T(2) - abs2(standardized) / T(2)
end

function configure_amis_kernel_state(::Type{T}, ::Val{F}) where {T,F}
    proposal = F ?
               FactorGaussian(T[-1], reshape(T[1], 1, 1)) :
               SphericalGaussian(T(-1), one(T))
    state = AMISKernelIS._prepare_method_state(
        AMIS(proposal; rounds=2, round_size=[2, 3]),
    )
    history = state.history
    if F
        history.means[:, 2] .= T(1)
        history.factors[:, :, 2] .= reshape(T[1.5], 1, 1)
    else
        history.means[2] = T(1)
        history.scales[2] = T(1.5)
    end
    history.lognormalizers[2] = -log(T(1.5)) - log(T(2pi)) / T(2)
    return state
end

function run_amis_kernel_rounds(::Type{T}, factor_history::Val, execution) where {T}
    state = configure_amis_kernel_state(T, factor_history)
    workspace = state.workspace
    round_ids = Vector{Int}(undef, 5)
    failures = zeros(UInt64, 3)
    target = AMISKernelIS._NativeDeviceTarget{
        T,
        AMISKernelTarget{T},
    }(AMISKernelTarget{T}())
    first_normals = T[-0.5, 0.75]
    second_normals = T[-1, 0.25, 1]

    AMISKernelIS._launch_prefilled_amis_round!(
        workspace.samples,
        workspace.logtargets,
        workspace.lognumerators,
        workspace.logweights,
        round_ids,
        failures,
        first_normals,
        target,
        state.history,
        state.logcounts,
        state.offsets,
        1,
        workspace.centered_scaled,
        execution,
    )
    first_round_lognumerators = copy(view(workspace.lognumerators, 1:2))
    AMISKernelIS._launch_prefilled_amis_round!(
        workspace.samples,
        workspace.logtargets,
        workspace.lognumerators,
        workspace.logweights,
        round_ids,
        failures,
        second_normals,
        target,
        state.history,
        state.logcounts,
        state.offsets,
        2,
        workspace.centered_scaled,
        execution,
    )
    return (
        samples=copy(workspace.samples),
        logtargets=copy(workspace.logtargets),
        lognumerators=copy(workspace.lognumerators),
        logweights=copy(workspace.logweights),
        round_ids,
        failures,
        first_round_lognumerators,
    )
end

function amis_kernel_scalar_values(samples)
    return samples isa AbstractVector ? samples : vec(samples)
end

@testset "prefilled AMIS retrospective log mixture" begin
    for T in (Float32, Float64)
        executions = (
            (
                @inferred(run_amis_kernel_rounds(
                    T,
                    Val(false),
                    AMISKernelIS._SerialCPUExecution(),
                )),
                @inferred(run_amis_kernel_rounds(
                    T,
                    Val(false),
                    AMISKernelIS._ThreadedCPUExecution(),
                )),
            ),
            (
                @inferred(run_amis_kernel_rounds(
                    T,
                    Val(true),
                    AMISKernelIS._SerialCPUExecution(),
                )),
                @inferred(run_amis_kernel_rounds(
                    T,
                    Val(true),
                    AMISKernelIS._ThreadedCPUExecution(),
                )),
            ),
        )
        for (serial, threaded) in executions
            samples = amis_kernel_scalar_values(serial.samples)
            expected = [
                LogExpFunctions.logaddexp(
                    log(T(2)) +
                    amis_kernel_logdensity(sample, T(-1), one(T)),
                    log(T(3)) +
                    amis_kernel_logdensity(sample, T(1), T(1.5)),
                ) for sample in samples
            ]

            @test serial.lognumerators ≈ expected rtol = 16eps(T)
            @test serial.logweights ≈
                  serial.logtargets .- expected .+ log(T(5)) rtol = 16eps(T)
            @test serial.lognumerators[1:2] !=
                  serial.first_round_lognumerators
            @test serial.round_ids == [1, 1, 2, 2, 2]
            @test iszero(serial.failures)
            @test threaded.samples == serial.samples
            @test threaded.logtargets == serial.logtargets
            @test threaded.lognumerators == serial.lognumerators
            @test threaded.logweights == serial.logweights
            @test threaded.round_ids == serial.round_ids
            @test threaded.failures == serial.failures
        end
    end
end

@testset "prefilled AMIS evaluates every final sample-proposal pair once" begin
    T = Float64
    state = configure_amis_kernel_state(T, Val(false))
    calls = zeros(Int, 2)
    history = AMISKernelCountingHistory(state.history, calls)
    workspace = state.workspace
    target_calls = [0]
    target = AMISKernelIS._NativeDeviceTarget{
        T,
        AMISKernelCountingTarget{T},
    }(AMISKernelCountingTarget{T}(target_calls))
    round_ids = Vector{Int}(undef, 5)
    failures = zeros(UInt64, 3)

    for (round, normals) in ((1, T[-0.5, 0.75]), (2, T[-1, 0.25, 1]))
        AMISKernelIS._launch_prefilled_amis_round!(
            workspace.samples,
            workspace.logtargets,
            workspace.lognumerators,
            workspace.logweights,
            round_ids,
            failures,
            normals,
            target,
            history,
            state.logcounts,
            state.offsets,
            round,
            workspace.centered_scaled,
            AMISKernelIS._SerialCPUExecution(),
        )
    end

    @test calls == [5, 5]
    @test sum(calls) == 10
    @test only(target_calls) == 5
    @test iszero(failures)
end

@testset "AMIS log-weight failure reasons stay in device storage" begin
    logweights = fill(-123.0, 2)
    failures = zeros(UInt64, 3)
    AMISKernelIS._launch_form_amis_logweights!(
        logweights,
        [0.0, 0.0],
        [-Inf, 0.0],
        0.0,
        failures,
        AMISKernelIS._SerialCPUExecution(),
    )
    decoded = AMISKernelIS._decode_native_failure(failures[1], failures[2])

    @test decoded.count == 1
    @test decoded.first_logical_index == 1
    @test decoded.reason_bits == AMISKernelIS._NATIVE_LOGWEIGHT_INVALID
    @test logweights == [-123.0, 0.0]
    @test AMISKernelIS._logweight_from_logmixture(0.0, -Inf, 0.0)[2] ==
          AMISKernelIS._NATIVE_LOGWEIGHT_INVALID
end
