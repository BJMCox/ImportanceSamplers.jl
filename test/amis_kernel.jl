using Test
using ImportanceSamplers
import KernelAbstractions
import LogExpFunctions
import Random

const AMISKernelIS = ImportanceSamplers

mutable struct AMISKernelCountingTarget{T}
    calls::Vector{Int}
end

mutable struct AMISPrefilledRNG{T} <: Random.AbstractRNG
    batches::Vector{Vector{T}}
    index::Int
end

AMISPrefilledRNG(batches::Vector{Vector{T}}) where {T} =
    AMISPrefilledRNG{T}(batches, 1)

function Random.randn!(rng::AMISPrefilledRNG, destination::AbstractArray)
    batch = rng.batches[rng.index]
    length(batch) >= length(destination) || throw(
        DimensionMismatch("prefilled AMIS normal batch is too short"),
    )
    copyto!(destination, 1, batch, 1, length(destination))
    rng.index += 1
    return destination
end

mutable struct AMISCountingTarget{T}
    calls::Int
end

function (target::AMISCountingTarget{T})(sample)::T where {T}
    target.calls += 1
    return -abs2(T(sample) - T(0.75)) / T(3)
end

struct AMISKernelTarget{T} end
struct AMISKernelVectorTarget{T} end
struct AMISProposalTarget{T} end
struct AMISKernelStageFailureExecution
    phase::Symbol
end

function (::AMISKernelTarget{T})(sample)::T where {T}
    value = sample isa Real ? sample : only(sample)
    return -abs2(value) / T(3)
end

function (::AMISKernelVectorTarget{T})(sample)::T where {T}
    return -sum(abs2, sample) / T(2)
end

function (::AMISProposalTarget{T})(sample)::T where {T}
    return amis_kernel_logdensity(T(sample), zero(T), one(T))
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

        _native_workgroupsize(
            execution::Main.AMISKernelStageFailureExecution,
            nsamples,
        ) = _native_workgroupsize(_SerialCPUExecution(), nsamples)

        function _launch_append_logmixture!(
            lognumerators,
            samples,
            history,
            slot,
            logcounts,
            failure_storage,
            solve_scratch,
            execution::Main.AMISKernelStageFailureExecution,
        )
            execution.phase === :denominator &&
                error("intentional AMIS denominator launch failure")
            return _launch_append_logmixture!(
                lognumerators,
                samples,
                history,
                slot,
                logcounts,
                failure_storage,
                solve_scratch,
                _SerialCPUExecution(),
            )
        end

        function _launch_mis_round!(
            samples,
            output::_AMISRoundOutput,
            failure_storage,
            normal_buffer,
            target,
            history,
            assignments,
            denominator,
            solve_scratch,
            execution::Main.AMISKernelStageFailureExecution,
        )
            execution.phase === :sampling &&
                error("intentional AMIS sampling launch failure")
            return _launch_mis_round!(
                samples,
                output,
                failure_storage,
                normal_buffer,
                target,
                history,
                assignments,
                denominator,
                solve_scratch,
                _SerialCPUExecution(),
            )
        end

        function _launch_form_amis_logweights!(
            logweights,
            logtargets,
            lognumerators,
            logtotal,
            failure_storage,
            execution::Main.AMISKernelStageFailureExecution,
        )
            execution.phase === :weight &&
                error("intentional AMIS weight launch failure")
            return _launch_form_amis_logweights!(
                logweights,
                logtargets,
                lognumerators,
                logtotal,
                failure_storage,
                _SerialCPUExecution(),
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

@testset "AMIS old-sample zero mixture contributions are valid" begin
    for T in (Float32, Float64), factor in (Val(false), Val(true))
        state = configure_amis_kernel_state(T, factor)
        history = state.history
        if factor isa Val{true}
            history.means[1, 2] = floatmax(T)
            history.factors[1, 1, 2] = one(T)
        else
            history.means[2] = floatmax(T)
            history.scales[2] = one(T)
        end
        for execution in (
            AMISKernelIS._SerialCPUExecution(),
            AMISKernelIS._ThreadedCPUExecution(),
        )
            lognumerators = T[1, 2]
            failures = zeros(UInt64, 3)
            AMISKernelIS._launch_append_logmixture!(
                lognumerators,
                T[-1, 1],
                history,
                2,
                state.logcounts,
                failures,
                state.workspace.centered_scaled,
                execution,
            )

            @test lognumerators == T[1, 2]
            @test iszero(failures)
        end
    end
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

@testset "AMIS native failures map to binding phases" begin
    empty = (
        count=UInt64(0),
        first_logical_index=0,
        first_block=0,
        reason_bits=UInt16(0),
    )
    cases = (
        (reason=AMISKernelIS._NATIVE_GENERATED_NONFINITE, phase=:sampling),
        (reason=AMISKernelIS._NATIVE_TARGET_NAN, phase=:target),
        (reason=AMISKernelIS._NATIVE_PROPOSAL_INVALID, phase=:denominator),
        (reason=AMISKernelIS._NATIVE_LOGWEIGHT_INVALID, phase=:weight),
    )
    for case in cases
        snapshot = (
            count=UInt64(1),
            first_logical_index=2,
            first_block=0,
            reason_bits=case.reason,
        )
        failure = try
            AMISKernelIS._throw_native_failures(
                snapshot,
                empty,
                AMISKernelIS._NoNativeTargetFailures(),
                AMISKernelIS._NoSampleTransform(),
            )
            nothing
        catch cause
            cause
        end

        @test failure isa SamplerExecutionError
        @test AMISKernelIS._amis_round_phase(failure, :sampling) === case.phase
    end

    storage = zeros(UInt64, 3)
    AMISKernelIS._record_native_failure!(
        storage,
        2,
        0,
        AMISKernelIS._NATIVE_TARGET_NAN,
    )
    AMISKernelIS._record_native_failure!(
        storage,
        4,
        0,
        AMISKernelIS._AMIS_COVARIANCE_INVALID,
    )
    packed = AMISKernelIS._device_failure_snapshot(
        AMISKernelIS._DeviceFailureRecord(storage),
    )
    combined_failure = try
        AMISKernelIS._capture_amis_round(
            1,
            :sampling,
            3,
            0,
            0,
            nothing,
            AMISKernelIS._ResultTransferCounter(0, 0),
        ) do
            packed.failure.reason_bits == AMISKernelIS._AMIS_COVARIANCE_INVALID ||
                AMISKernelIS._throw_native_failures(
                    packed.failure,
                    packed.draw_failure,
                    AMISKernelIS._NoNativeTargetFailures(),
                    AMISKernelIS._NoSampleTransform(),
                )
        end
        nothing
    catch cause
        cause
    end
    @test packed.failure.count == 2
    @test packed.failure.first_logical_index == 2
    @test packed.failure.reason_bits == AMISKernelIS._NATIVE_TARGET_NAN
    @test iszero(storage[3])
    @test combined_failure isa AMISRoundError
    @test combined_failure.phase === :target
    @test combined_failure.cause isa SamplerExecutionError
    @test combined_failure.cause.sample_index == 2
end

@testset "AMIS launch exceptions retain their actual stage" begin
    for phase in (:sampling, :denominator, :weight)
        state = configure_amis_kernel_state(Float64, Val(false))
        workspace = state.workspace
        round_ids = zeros(Int, 5)
        failure_storage = zeros(UInt64, 3)
        target = AMISKernelIS._NativeDeviceTarget{Float64,AMISKernelTarget{Float64}}(
            AMISKernelTarget{Float64}(),
        )
        round = phase === :denominator ? 2 : 1
        if round == 2
            AMISKernelIS._launch_prefilled_amis_round!(
                workspace.samples,
                workspace.logtargets,
                workspace.lognumerators,
                workspace.logweights,
                round_ids,
                failure_storage,
                Float64[-0.5, 0.75],
                target,
                state.history,
                state.logcounts,
                state.offsets,
                1,
                workspace.centered_scaled,
                AMISKernelIS._SerialCPUExecution(),
            )
        end
        normals = round == 1 ? Float64[-0.5, 0.75] : Float64[-1, 0.25, 1]
        failure = try
            AMISKernelIS._capture_amis_round(
                round,
                :sampling,
                state.schedule[round],
                round - 1,
                state.offsets[round] - 1,
                nothing,
                AMISKernelIS._ResultTransferCounter(0, 0),
            ) do
                AMISKernelIS._launch_prefilled_amis_round!(
                    workspace.samples,
                    workspace.logtargets,
                    workspace.lognumerators,
                    workspace.logweights,
                    round_ids,
                    failure_storage,
                    normals,
                    target,
                    state.history,
                    state.logcounts,
                    state.offsets,
                    round,
                    workspace.centered_scaled,
                    AMISKernelStageFailureExecution(phase),
                )
            end
            nothing
        catch cause
            cause
        end

        @test failure isa AMISRoundError
        @test failure.phase === phase
        @test failure.cause isa ErrorException
        @test occursin("$(phase) launch failure", failure.cause.msg)
    end
end

@testset "AMIS factor candidate failure stays in device storage" begin
    for T in (Float32, Float64)
        mean = T[0]
        factor = reshape(T[Inf], 1, 1)
        lognormalizer = zeros(T, 1)
        failures = zeros(UInt64, 3)
        backend = KernelAbstractions.get_backend(factor)
        kernel = AMISKernelIS._finish_amis_factor_candidate_kernel!(backend)
        kernel(
            mean,
            factor,
            lognormalizer,
            failures,
            4;
            ndrange=1,
        )
        KernelAbstractions.synchronize(backend)
        decoded = AMISKernelIS._decode_native_failure(
            failures[1],
            failures[2],
        )

        @test decoded.count == 1
        @test decoded.first_logical_index == 4
        @test decoded.reason_bits == AMISKernelIS._AMIS_COVARIANCE_INVALID
        @test !isfinite(only(lognormalizer))
        @test iszero(failures[3])
    end
end

@testset "AMIS diagnostics summarize each retrospective retained prefix" begin
    expected = (
        Float32=(
            ess=Float32[2.0, 3.472985029220581],
            lognormalizers=Float32[0.0, -0.22794675827026367],
        ),
        Float64=(
            ess=Float64[2.0, 3.473043499457537],
            lognormalizers=Float64[0.0, -0.2279354492780643],
        ),
    )
    for T in (Float32, Float64)
        batches = [
            T[-2, 2],
            T[0, 1],
            T[-1, 1],
            T[-0.5, 0.5],
        ]
        sampler = @inferred prepare_sampler(
            AMISPrefilledRNG(batches),
            AMISProposalTarget{T}(),
            AMIS(SphericalGaussian(zero(T), one(T)); rounds=2, round_size=2);
            threaded=false,
        )

        first = @inferred importance_sample!(sampler)
        expected_type = getproperty(expected, nameof(T))
        first_ess = copy(first.diagnostics.round_ess)
        first_lognormalizers = copy(first.diagnostics.round_lognormalizers)

        @test first.diagnostics.method === :amis
        @test first.diagnostics.round_ess ≈ expected_type.ess rtol = 16eps(T)
        @test first.diagnostics.round_lognormalizers ≈
              expected_type.lognormalizers rtol = 16eps(T)
        @test eltype(first.diagnostics.round_ess) === T
        @test eltype(first.diagnostics.round_lognormalizers) === T

        second = @inferred importance_sample!(sampler)

        @test second.diagnostics.method === :amis
        @test eltype(second.diagnostics.round_ess) === T
        @test eltype(second.diagnostics.round_lognormalizers) === T
        @test first.diagnostics.round_ess == first_ess
        @test first.diagnostics.round_lognormalizers == first_lognormalizers
        @test first.diagnostics.round_ess !== second.diagnostics.round_ess
        @test first.diagnostics.round_lognormalizers !==
              second.diagnostics.round_lognormalizers
    end
end

@testset "complete CPU AMIS is retrospective, transactional, and reusable" begin
    for T in (Float32, Float64)
        schedule = [3, 3]
        batches = [
            T[-1, 0.25, 1.5],
            T[-0.75, 1.25, 9],
            T[0.5, -1, 2],
            T[-1.5, 0.75, 8],
        ]
        proposal = SphericalGaussian(T(-0.5), T(1.25))
        target = AMISCountingTarget{T}(0)
        rng = AMISPrefilledRNG(deepcopy(batches))
        sampler = @inferred prepare_sampler(
            rng,
            target,
            AMIS(proposal; rounds=2, round_size=schedule);
            threaded=false,
        )

        first = @inferred importance_sample!(sampler)
        first_snapshot = (
            samples=copy(first.samples),
            logweights=copy(first.logweights),
            rounds=copy(first.provenance.round),
        )
        learned_after_first = @inferred current_proposal(sampler)
        @test sampler.method_state.committed_in_workspace
        q2_mean = sampler.method_state.history.means[2]
        q2_scale = sampler.method_state.history.scales[2]
        q1_logs = [
            amis_kernel_logdensity(sample, proposal.location, proposal.scale.scale) for
            sample in first.samples
        ]
        q2_logs = [
            amis_kernel_logdensity(sample, q2_mean, q2_scale) for
            sample in first.samples
        ]
        final_denominators = [
            LogExpFunctions.logaddexp(log(T(3)) + q1, log(T(3)) + q2) -
            log(T(6)) for (q1, q2) in zip(q1_logs, q2_logs)
        ]
        expected_final = [
            -abs2(T(sample) - T(0.75)) / T(3) - denominator for
            (sample, denominator) in zip(first.samples, final_denominators)
        ]
        provisional_old = [
            -abs2(T(sample) - T(0.75)) / T(3) - q1_logs[index] for
            (index, sample) in pairs(first.samples[1:3])
        ]

        @test length(first) == sum(schedule)
        @test first.provenance.round == repeat(1:2; inner=3)
        @test first.logweights ≈ expected_final rtol = 64eps(T)
        @test all(
            index -> first.logweights[index] != provisional_old[index],
            eachindex(provisional_old),
        )
        @test first.diagnostics.target_evaluations == sum(schedule)
        @test first.diagnostics.proposal_evaluations == 2 * sum(schedule)
        @test target.calls == sum(schedule)
        @test learned_after_first !== proposal
        @test learned_after_first.location isa T
        @test learned_after_first.scale.scale isa T

        first_second_sample = learned_after_first.location +
                              learned_after_first.scale.scale * batches[3][1]
        second = @inferred importance_sample!(sampler)

        @test sampler.method_state.committed_in_workspace
        @test sampler.method_state.history.means[1] == learned_after_first.location
        @test sampler.method_state.history.scales[1] ==
              learned_after_first.scale.scale
        @test second.samples[1] ≈ first_second_sample rtol = 8eps(T)
        @test first.samples == first_snapshot.samples
        @test first.logweights == first_snapshot.logweights
        @test first.provenance.round == first_snapshot.rounds
        @test first.samples !== second.samples
        @test first.logweights !== second.logweights
        @test first.provenance.round !== second.provenance.round
        @test first.samples !== sampler.method_state.workspace.samples
        @test first.logweights !== sampler.method_state.workspace.logweights
    end
end

@testset "CPU AMIS factor proposal snapshots are independent and exact" begin
    T = Float32
    proposal = FactorGaussian(T[-1, 0.5], T[1 0; 0.25 1.5])
    sampler = @inferred prepare_sampler(
        AMISPrefilledRNG([T[-1, 0, 1, 0.5, -0.5, 1.5]]),
        AMISKernelVectorTarget{T}(),
        AMIS(proposal; rounds=1, round_size=3);
        threaded=false,
    )
    @inferred importance_sample!(sampler)
    first_snapshot = @inferred current_proposal(sampler)
    second_snapshot = @inferred current_proposal(sampler)
    @test sampler.method_state.committed_in_workspace

    @test eltype(first_snapshot.location) === T
    @test eltype(first_snapshot.scale.factor) === T
    @test first_snapshot !== second_snapshot
    @test first_snapshot.location !== second_snapshot.location
    @test first_snapshot.scale.factor !== second_snapshot.scale.factor
    first_snapshot.location[1] = T(100)
    first_snapshot.scale.factor[1, 1] = T(100)
    third_snapshot = current_proposal(sampler)
    @test third_snapshot.location == second_snapshot.location
    @test third_snapshot.scale.factor == second_snapshot.scale.factor
end
