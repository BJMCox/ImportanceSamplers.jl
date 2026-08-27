using Test
using ImportanceSamplers
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
    committed_mean = copy(sampler.method_state.history.means[:, 1])
    committed_factor = copy(sampler.method_state.history.factors[:, :, 1])

    @test eltype(first_snapshot.location) === T
    @test eltype(first_snapshot.scale.factor) === T
    @test first_snapshot !== second_snapshot
    @test first_snapshot.location !== second_snapshot.location
    @test first_snapshot.scale.factor !== second_snapshot.scale.factor
    first_snapshot.location[1] = T(100)
    first_snapshot.scale.factor[1, 1] = T(100)
    @test sampler.method_state.history.means[:, 1] == committed_mean
    @test sampler.method_state.history.factors[:, :, 1] == committed_factor
end
