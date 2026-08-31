using Test
using ImportanceSamplers
import DensityInterface
import KernelAbstractions
import LinearAlgebra
import MLDataDevices
import Random
import Random: rand

struct TestOffsetArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::A
    offsets::NTuple{N,Int}
end

mutable struct AMISFailureRNG{T} <: Random.AbstractRNG
    batches::Vector{Vector{T}}
    index::Int
end

AMISFailureRNG(batches::Vector{Vector{T}}) where {T} =
    AMISFailureRNG{T}(batches, 1)

function Random.randn!(rng::AMISFailureRNG, destination::AbstractArray)
    batch = rng.batches[rng.index]
    copyto!(destination, 1, batch, 1, length(destination))
    rng.index += 1
    return destination
end

mutable struct AMISFailingTarget{T}
    calls::Int
    fail_at::Int
end

function (target::AMISFailingTarget{T})(sample)::T where {T}
    target.calls += 1
    target.calls == target.fail_at && error("intentional AMIS target failure")
    return -abs2(T(sample)) / T(2)
end

struct AMISZeroTarget{T} end

struct AMISMomentFailureMatrix{T,A<:AbstractMatrix{T}} <: AbstractMatrix{T}
    storage::A
end

struct AMISResultFailureArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    storage::A
end

mutable struct AMISPublicationFailure
    writes::Int
    fail_at::Int
    visited::UInt8
end

struct AMISPublicationFailureArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    storage::A
    failure::AMISPublicationFailure
    write::Int
end

Base.size(array::AMISPublicationFailureArray) = size(array.storage)
Base.IndexStyle(::Type{<:AMISPublicationFailureArray}) = IndexLinear()
Base.getindex(array::AMISPublicationFailureArray, indices...) =
    getindex(array.storage, indices...)

function fail_amis_publication!(array::AMISPublicationFailureArray)
    failure = array.failure
    visited = UInt8(1) << (array.write - 1)
    iszero(failure.visited & visited) || return nothing
    failure.visited |= visited
    failure.writes += 1
    array.write == failure.fail_at &&
        error("intentional AMIS publication failure $(failure.fail_at)")
    return nothing
end

function Base.setindex!(
    array::AMISPublicationFailureArray,
    value,
    indices...,
)
    fail_amis_publication!(array)
    return setindex!(array.storage, value, indices...)
end

Base.size(matrix::AMISMomentFailureMatrix) = size(matrix.storage)
Base.IndexStyle(::Type{<:AMISMomentFailureMatrix}) = IndexCartesian()
Base.getindex(matrix::AMISMomentFailureMatrix, i::Int, j::Int) = matrix.storage[i, j]
Base.setindex!(matrix::AMISMomentFailureMatrix, value, i::Int, j::Int) =
    setindex!(matrix.storage, value, i, j)

Base.size(array::AMISResultFailureArray) = size(array.storage)
Base.IndexStyle(::Type{<:AMISResultFailureArray}) = IndexLinear()
Base.getindex(array::AMISResultFailureArray, indices...) =
    getindex(array.storage, indices...)
Base.setindex!(array::AMISResultFailureArray, value, indices...) =
    setindex!(array.storage, value, indices...)
Base.copy(::AMISResultFailureArray) = error("intentional AMIS result copy failure")

struct GRAMISInvalidDiagnosticPrototype{A<:AbstractVector{Float64}} <:
       AbstractVector{Float64}
    storage::A
end

Base.size(array::GRAMISInvalidDiagnosticPrototype) = size(array.storage)
Base.IndexStyle(::Type{<:GRAMISInvalidDiagnosticPrototype}) = IndexLinear()
Base.getindex(array::GRAMISInvalidDiagnosticPrototype, index::Int) =
    array.storage[index]
Base.setindex!(array::GRAMISInvalidDiagnosticPrototype, value, index::Int) =
    setindex!(array.storage, value, index)
function Base.similar(
    array::GRAMISInvalidDiagnosticPrototype,
    ::Type{T},
    dimensions::Int...,
) where {T}
    length(dimensions) == 2 && return Matrix{Any}(undef, dimensions...)
    return similar(array.storage, T, dimensions...)
end

KernelAbstractions.get_backend(::GRAMISInvalidDiagnosticPrototype) =
    KernelAbstractions.CPU()

mutable struct GRAMISOneShotFailure
    armed::Bool
end

struct GRAMISResultAllocationFailureArray{T,N,A<:AbstractArray{T,N}} <:
       AbstractArray{T,N}
    storage::A
    failure::GRAMISOneShotFailure
end

Base.size(array::GRAMISResultAllocationFailureArray) = size(array.storage)
Base.IndexStyle(::Type{<:GRAMISResultAllocationFailureArray}) = IndexCartesian()
Base.getindex(array::GRAMISResultAllocationFailureArray, indices...) =
    getindex(array.storage, indices...)
Base.setindex!(array::GRAMISResultAllocationFailureArray, value, indices...) =
    setindex!(array.storage, value, indices...)
KernelAbstractions.get_backend(::GRAMISResultAllocationFailureArray) =
    KernelAbstractions.CPU()
function Base.similar(
    array::GRAMISResultAllocationFailureArray,
    ::Type{T},
    dimensions::Int...,
) where {T}
    if array.failure.armed
        array.failure.armed = false
        error("intentional GRAMIS result allocation failure")
    end
    return similar(array.storage, T, dimensions...)
end

struct GRAMISCollectFailureSchedule{A<:AbstractVector{Int}} <:
       AbstractVector{Int}
    storage::A
    failure::GRAMISOneShotFailure
end

Base.size(schedule::GRAMISCollectFailureSchedule) = size(schedule.storage)
Base.IndexStyle(::Type{<:GRAMISCollectFailureSchedule}) = IndexLinear()
Base.getindex(schedule::GRAMISCollectFailureSchedule, index::Int) =
    schedule.storage[index]
function Base.collect(schedule::GRAMISCollectFailureSchedule)
    if schedule.failure.armed
        schedule.failure.armed = false
        error("intentional GRAMIS diagnostic assembly failure")
    end
    return collect(schedule.storage)
end

mutable struct GRAMISTransactionTarget{T}
    fail::Bool
    calls::Int
end

mutable struct GRAMISPhaseTarget{T}
    calls::Int
    fail_at::Int
end

mutable struct GRAMISProposalInjectionTarget{T}
    calls::Int
    inject_at::Int
end

function (target::GRAMISProposalInjectionTarget{T})(sample)::T where {T}
    target.calls += 1
    target.calls == target.inject_at && (sample[1] = T(NaN))
    return zero(T)
end

function (target::GRAMISPhaseTarget{T})(sample)::T where {T}
    target.calls += 1
    target.calls == target.fail_at &&
        error("intentional GRAMIS phase target failure")
    return -T(0.5) * sum(abs2, sample)
end

mutable struct GRAMISPhaseGradient
    calls::Int
    fail_at::Int
end

function (gradient::GRAMISPhaseGradient)(destination, sample)
    gradient.calls += 1
    if gradient.calls == gradient.fail_at
        fill!(destination, eltype(destination)(NaN))
        return destination
    end
    destination .= -sample
    return destination
end

mutable struct GRAMISWriteFailure
    armed::Bool
    writes::Int
    fail_at::Int
end

struct GRAMISResultCopyFailureArray{T,N,A<:AbstractArray{T,N}} <:
       AbstractArray{T,N}
    storage::A
    failure::GRAMISWriteFailure
    output::Bool
end

Base.size(array::GRAMISResultCopyFailureArray) = size(array.storage)
Base.IndexStyle(::Type{<:GRAMISResultCopyFailureArray}) = IndexCartesian()
Base.getindex(array::GRAMISResultCopyFailureArray, indices...) =
    getindex(array.storage, indices...)
function Base.setindex!(
    array::GRAMISResultCopyFailureArray,
    value,
    indices...,
)
    if array.output && array.failure.armed
        array.failure.writes += 1
        if array.failure.writes == array.failure.fail_at
            array.failure.armed = false
            error("intentional GRAMIS round-two diagnostic copy failure")
        end
    end
    return setindex!(array.storage, value, indices...)
end
KernelAbstractions.get_backend(::GRAMISResultCopyFailureArray) =
    KernelAbstractions.CPU()
function Base.similar(
    array::GRAMISResultCopyFailureArray,
    ::Type{T},
    dimensions::Int...,
) where {T}
    return GRAMISResultCopyFailureArray(
        similar(array.storage, T, dimensions...),
        array.failure,
        true,
    )
end

function (target::GRAMISTransactionTarget{T})(sample)::T where {T}
    target.calls += 1
    target.fail && error("intentional FirstOrderGRAMIS target failure")
    return -T(0.5) * sum(abs2, sample)
end

function gram_is_transaction_gradient!(destination, sample)
    destination .= -sample
    return destination
end

function LinearAlgebra.mul!(
    destination::AMISMomentFailureMatrix,
    left,
    right,
)
    error("intentional AMIS moment multiplication failure")
end

function (::AMISZeroTarget{T})(sample)::T where {T}
    return zero(T)
end

function amis_result_copy_failure_sampler(sampler)
    old_state = sampler.method_state
    old_workspace = old_state.workspace
    workspace = ImportanceSamplers._AMISWorkspace(
        AMISResultFailureArray(old_workspace.samples),
        old_workspace.logtargets,
        old_workspace.lognumerators,
        old_workspace.logweights,
        old_workspace.normalized_weights,
        old_workspace.centered_scaled,
        old_workspace.covariance,
        old_workspace.candidate_mean,
        old_workspace.candidate_scale,
        old_workspace.candidate_lognormalizer,
    )
    state = ImportanceSamplers._PreparedAMIS(
        old_state.schedule,
        old_state.offsets,
        old_state.logcounts,
        old_state.history,
        workspace,
        old_state.committed_in_workspace,
    )
    return ImportanceSamplers._PreparedImportanceSampler(
        sampler.rng,
        sampler.random_buffers,
        sampler.target,
        sampler.algorithm,
        state,
        sampler.device,
        sampler.factor_execution,
        sampler.threaded,
        false,
        false,
    )
end

function amis_proposal_bits(proposal)
    location = proposal.location isa Number ?
               bitstring(proposal.location) : map(bitstring, proposal.location)
    scale = if proposal.scale isa ImportanceSamplers._SphericalGaussianScale
        bitstring(proposal.scale.scale)
    else
        map(bitstring, proposal.scale.factor)
    end
    return location, scale, bitstring(proposal.lognormalizer)
end

function assert_amis_transaction_failure(
    sampler,
    expected_phase,
    expected_cause,
    expected_round,
    expected_rng_index,
)
    before = amis_proposal_bits(current_proposal(sampler))
    rng_index = sampler.rng.index
    failure = caught_exception(() -> importance_sample!(sampler))

    @test failure isa AMISRoundError
    @test failure.round == expected_round
    @test failure.phase === expected_phase
    @test failure.cause isa expected_cause
    @test failure.diagnostics.round_size == sampler.method_state.schedule[expected_round]
    @test failure.diagnostics.completed_rounds == expected_round - 1 ||
          expected_phase === :result_construction
    completed_rounds = failure.diagnostics.completed_rounds
    @test failure.diagnostics.cumulative_sample_count ==
          sum(view(sampler.method_state.schedule, 1:completed_rounds); init=0)
    @test hasproperty(failure.diagnostics, :covariance)
    @test hasproperty(failure.diagnostics, :transfers)
    if expected_phase === :factorization
        @test failure.diagnostics.covariance isa NamedTuple
        @test keys(failure.diagnostics.covariance) == (
            :minimum_diagonal,
            :maximum_absolute_entry,
        )
    else
        @test isnothing(failure.diagnostics.covariance)
    end
    @test amis_proposal_bits(current_proposal(sampler)) == before
    @test sampler.rng.index == expected_rng_index > rng_index
    @test !sampler.running
    return failure
end

function amis_publication_failure_sampler(fail_at)
    T = Float64
    base = prepare_sampler(
        AMISFailureRNG(fill(T[-1, 0, 1, 0, -1, 1], 2)),
        AMISZeroTarget{T}(),
        AMIS(FactorGaussian(zeros(T, 2), T[1 0; 0 1]); rounds=1, round_size=3);
        threaded=false,
    )
    old_state = base.method_state
    failure = AMISPublicationFailure(0, fail_at, 0x00)
    old_history = old_state.history
    history = ImportanceSamplers._AMISFactorHistory(
        AMISPublicationFailureArray(old_history.means, failure, 1),
        AMISPublicationFailureArray(old_history.factors, failure, 2),
        AMISPublicationFailureArray(old_history.lognormalizers, failure, 3),
    )
    state = ImportanceSamplers._PreparedAMIS(
        old_state.schedule,
        old_state.offsets,
        old_state.logcounts,
        history,
        old_state.workspace,
        old_state.committed_in_workspace,
    )
    sampler = ImportanceSamplers._PreparedImportanceSampler(
        base.rng,
        base.random_buffers,
        base.target,
        base.algorithm,
        state,
        base.device,
        base.factor_execution,
        base.threaded,
        false,
        false,
    )
    return sampler, failure
end

function captured_sampler_execution_error(phase)
    try
        error("intentional AMIS $(phase) failure")
    catch cause
        return SamplerExecutionError(
            phase,
            1,
            CapturedException(cause, catch_backtrace()),
        )
    end
end

struct ThreadedTargetFailure <: Exception
    sample_index::Int
end

struct ThreadedProposalFailure <: Exception
    sample_index::Int
end

mutable struct ThreadFailureTarget
    tasks::Vector{Union{Nothing,Task}}
    fail_enabled::Base.RefValue{Bool}
    fail_at::Int
end

function (target::ThreadFailureTarget)(sample::Float64)::Float64
    sample_index = Int(sample)
    target.tasks[sample_index] = current_task()
    if target.fail_enabled[] && sample_index == target.fail_at
        throw(ThreadedTargetFailure(sample_index))
    end
    return 0.0
end

mutable struct ThreadFailureProposal
    draw_index::Int
    nsamples::Int
    density_tasks::Vector{Union{Nothing,Task}}
    density_seen::Vector{Bool}
    fail_density_at::Int
end

function ThreadFailureProposal(nsamples::Int; fail_density_at=0)
    return ThreadFailureProposal(
        0,
        nsamples,
        fill(nothing, nsamples),
        fill(false, nsamples),
        fail_density_at,
    )
end

function rand(::Random.AbstractRNG, proposal::ThreadFailureProposal)
    proposal.draw_index += 1
    return Float64(mod1(proposal.draw_index, proposal.nsamples))
end

function DensityInterface.logdensityof(
    proposal::ThreadFailureProposal,
    sample::Float64,
)
    sample_index = Int(sample)
    proposal.density_tasks[sample_index] = current_task()
    proposal.density_seen[sample_index] = true
    sample_index == proposal.fail_density_at &&
        throw(ThreadedProposalFailure(sample_index))
    return 0.0
end

TestOffsetArray(parent::AbstractArray{T,N}, offsets::NTuple{N,Int}) where {T,N} =
    TestOffsetArray{T,N,typeof(parent)}(parent, offsets)

Base.size(array::TestOffsetArray) = size(array.parent)
Base.IndexStyle(::Type{<:TestOffsetArray}) = IndexCartesian()

function Base.axes(array::TestOffsetArray{T,N}) where {T,N}
    return ntuple(N) do dimension
        parent_axis = axes(array.parent, dimension)
        (first(parent_axis) + array.offsets[dimension]):(
            last(parent_axis) + array.offsets[dimension]
        )
    end
end

function Base.getindex(array::TestOffsetArray{T,N}, indices::Vararg{Int,N}) where {T,N}
    parent_indices = ntuple(
        dimension -> indices[dimension] - array.offsets[dimension],
        N,
    )
    return array.parent[parent_indices...]
end

@testset "result construction failures" begin
    @test_throws ArgumentError WeightedSamples([1.0, 2.0], [0.0])
    @test_throws ArgumentError WeightedSamples(Float64[], Float64[])
    @test_throws ArgumentError WeightedSamples([1.0], [NaN])
    @test_throws ArgumentError WeightedSamples([1.0], [Inf])
    @test_throws ArgumentError WeightedSamples(
        [1.0, 2.0],
        [0.0, 0.0];
        provenance=(proposal_id=[1],),
    )
    @test_throws ArgumentError WeightedSamples(
        [1.0],
        [0.0];
        provenance=[1],
    )
    @test_throws ArgumentError WeightedSamples(
        [1.0],
        [0.0];
        diagnostics=Dict(:method => :plain_is),
    )
    @test_throws ArgumentError WeightedSamples([[1.0], [2.0]], [0.0, 0.0])
    @test_throws ArgumentError WeightedSamples(["a", "b"], [0.0, 0.0])
    @test_throws ArgumentError WeightedSamples(Real[1.0, 2.0], [0.0, 0.0])
    @test_throws ArgumentError WeightedSamples(reshape(["a", "b"], 1, 2), [0.0, 0.0])
    @test_throws ArgumentError WeightedSamples(NamedTuple(), [0.0])
    @test_throws ArgumentError WeightedSamples((value=Ref(1),), [0.0])
    @test_throws ArgumentError WeightedSamples((value=Float64[],), Float64[])
    @test_throws ArgumentError WeightedSamples(
        (value=[1.0], nested=NamedTuple()),
        [0.0],
    )
    @test_throws ArgumentError WeightedSamples(
        [1.0],
        [0.0];
        diagnostics=(state=Ref(1),),
    )
    @test_throws ArgumentError WeightedSamples(
        [1.0, 2.0],
        [0.0, 0.0];
        provenance=(proposal_id=[[1], [2]],),
    )

    @test_throws MethodError WeightedSamples(
        [1.0],
        [Inf],
        NamedTuple(),
        NamedTuple(),
        Val(:owned),
    )
    @test_throws MethodError WeightedSampleView(
        view([1.0], 1:1),
        view([Inf], 1:1),
        NamedTuple(),
    )

    forged_token = ImportanceSamplers._ValidatedResultToken()
    @test_throws ArgumentError ImportanceSamplers.WeightedSamples{Any}(
        [1.0],
        [Inf],
        NamedTuple(),
        NamedTuple(),
        forged_token,
    )
    @test_throws ArgumentError ImportanceSamplers.WeightedSampleView{Any}(
        view([1.0], 1:1),
        view([Inf], 1:1),
        NamedTuple(),
        ImportanceSamplers._ResultTransferCounter(0, 0),
        forged_token,
    )
end

@testset "result selection failures" begin
    result = WeightedSamples([1.0, 2.0, 3.0], [-3.0, -2.0, -1.0])
    @test_throws BoundsError result[0]
    @test_throws BoundsError result[4]
    @test_throws BoundsError result[0:1]
    @test_throws BoundsError result[2:4]
    @test_throws BoundsError result[[1, 4]]
    @test_throws BoundsError result[Bool[true, false]]
    @test_throws ArgumentError result[false:true]
    @test isempty(Test.detect_ambiguities(ImportanceSamplers; recursive=false))
end

@testset "one-based result storage" begin
    offset_samples = TestOffsetArray([1.0, 2.0], (-1,))
    offset_matrix_samples = TestOffsetArray([1.0 2.0; 3.0 4.0], (0, -1))
    offset_logweights = TestOffsetArray([-2.0, -1.0], (-1,))
    offset_provenance = TestOffsetArray([10, 20], (-1,))
    offset_diagnostics = TestOffsetArray([2, 1], (-1,))

    @test_throws ArgumentError WeightedSamples(offset_samples, [-2.0, -1.0])
    @test_throws ArgumentError WeightedSamples(offset_matrix_samples, [-2.0, -1.0])
    @test_throws ArgumentError WeightedSamples([1.0, 2.0], offset_logweights)
    @test_throws ArgumentError WeightedSamples(
        (location=[1.0, 2.0], state=(scale=offset_samples,)),
        [-2.0, -1.0],
    )
    @test_throws ArgumentError WeightedSamples(
        [1.0, 2.0],
        [-2.0, -1.0];
        provenance=(origin=(proposal_id=offset_provenance,),),
    )
    @test_throws ArgumentError WeightedSamples(
        [1.0, 2.0],
        [-2.0, -1.0];
        diagnostics=(trace=offset_diagnostics,),
    )
end

@testset "all-zero weight mass" begin
    result = WeightedSamples(
        [10.0, 20.0, 30.0],
        fill(-Inf, 3);
        diagnostics=(method=:plain_is,),
    )

    @test result.samples == [10.0, 20.0, 30.0]
    @test result.logweights == fill(-Inf, 3)
    @test result.provenance == NamedTuple()
    @test lognormalizer(result) == -Inf
    @test_throws AllZeroWeightsError normalized_weights(result)

    zero_mass_view = result[1:2]
    @test zero_mass_view.samples == [10.0, 20.0]
    @test zero_mass_view.logweights == fill(-Inf, 2)
    @test_throws AllZeroWeightsError normalized_weights(zero_mass_view)
    @test_throws ArgumentError lognormalizer(zero_mass_view)
end

struct PreparedTargetFailure <: Exception
    sample::Float64
end

struct PreparedProposalFailure <: Exception
    sample::Float64
end

mutable struct FailureSequenceProposal{F}
    values::Vector{Float64}
    logdensity::F
    draw_index::Int
    draw_count::Int
    density_count::Int
    fail_draw_at::Int
end

function FailureSequenceProposal(
    values::Vector{Float64}, logdensity; fail_draw_at=0
)
    return FailureSequenceProposal(values, logdensity, 0, 0, 0, fail_draw_at)
end

struct UnprovablePreparedTarget
    logdensity::Function
end

(target::UnprovablePreparedTarget)(sample) = target.logdensity(sample)

struct IncompatiblePreparedTarget end
(::IncompatiblePreparedTarget)(::Matrix{Float64}) = 0.0

mutable struct UnprovablePreparedProposal
    logdensity::Function
    draw_count::Int
end

function rand(::Random.AbstractRNG, proposal::UnprovablePreparedProposal)
    proposal.draw_count += 1
    return Float64(proposal.draw_count)
end

function DensityInterface.logdensityof(
    proposal::UnprovablePreparedProposal, sample::Float64
)
    return proposal.logdensity(sample)
end

function rand(::Random.AbstractRNG, proposal::FailureSequenceProposal)
    proposal.draw_index += 1
    proposal.draw_count += 1
    proposal.draw_index == proposal.fail_draw_at &&
        throw(PreparedProposalFailure(Float64(proposal.draw_index)))
    return proposal.values[mod1(proposal.draw_index, length(proposal.values))]
end

function DensityInterface.logdensityof(
    proposal::FailureSequenceProposal, sample::Float64
)
    proposal.density_count += 1
    return proposal.logdensity(sample)
end

function caught_exception(f)
    try
        f()
    catch error
        return error
    end
    return nothing
end

@testset "prepared execution failures" begin
    for bad_value in (NaN, Inf)
        proposal = FailureSequenceProposal([1.0, 2.0, 3.0], _ -> 0.0)
        target = sample -> sample == 2.0 ? bad_value : 0.0
        sampler = prepare_sampler(
            Random.Xoshiro(808),
            target,
            ImportanceSampling(proposal; nsamples=3);
            threaded=false,
        )
        failure = caught_exception(() -> importance_sample!(sampler))
        @test failure isa SamplerExecutionError
        @test failure.phase === :target
        @test failure.sample_index == 2
        @test failure.captured isa CapturedException
        @test failure.captured.ex isa DomainError
        @test proposal.draw_count == 3
        @test proposal.density_count == 0
    end

    for bad_value in (NaN, -Inf)
        proposal = FailureSequenceProposal(
            [1.0, 2.0, 3.0],
            sample -> sample == 2.0 ? bad_value : 0.0,
        )
        sampler = prepare_sampler(
            Random.Xoshiro(909),
            _ -> 0.0,
            ImportanceSampling(proposal; nsamples=3);
            threaded=false,
        )
        failure = caught_exception(() -> importance_sample!(sampler))
        @test failure isa SamplerExecutionError
        @test failure.phase === :proposal_logdensity
        @test failure.sample_index == 2
        @test failure.captured isa CapturedException
        @test failure.captured.ex isa DomainError
        @test proposal.draw_count == 3
        @test proposal.density_count == 2
    end

    target_proposal = FailureSequenceProposal([1.0, 2.0, 3.0], _ -> 0.0)
    target_sampler = prepare_sampler(
        Random.Xoshiro(1001),
        sample -> sample == 2.0 ? throw(PreparedTargetFailure(sample)) : 0.0,
        ImportanceSampling(target_proposal; nsamples=3);
        threaded=false,
    )
    target_failure = caught_exception(() -> importance_sample!(target_sampler))
    @test target_failure isa SamplerExecutionError
    @test target_failure.phase === :target
    @test target_failure.sample_index == 2
    @test target_failure.captured isa CapturedException
    @test target_failure.captured.ex isa PreparedTargetFailure
    @test target_failure.captured.ex.sample == 2.0
    @test target_proposal.density_count == 0

    density_proposal = FailureSequenceProposal(
        [1.0, 2.0, 3.0],
        sample -> sample == 2.0 ? throw(PreparedProposalFailure(sample)) : 0.0,
    )
    density_sampler = prepare_sampler(
        Random.Xoshiro(1002),
        _ -> 0.0,
        ImportanceSampling(density_proposal; nsamples=3);
        threaded=false,
    )
    density_failure = caught_exception(() -> importance_sample!(density_sampler))
    @test density_failure isa SamplerExecutionError
    @test density_failure.phase === :proposal_logdensity
    @test density_failure.sample_index == 2
    @test density_failure.captured isa CapturedException
    @test density_failure.captured.ex isa PreparedProposalFailure
    @test density_failure.captured.ex.sample == 2.0

    draw_proposal = FailureSequenceProposal(
        [1.0, 2.0, 3.0],
        _ -> 0.0;
        fail_draw_at=2,
    )
    draw_sampler = prepare_sampler(
        Random.Xoshiro(1003),
        _ -> 0.0,
        ImportanceSampling(draw_proposal; nsamples=3);
        threaded=false,
    )
    draw_failure = caught_exception(() -> importance_sample!(draw_sampler))
    @test draw_failure isa SamplerExecutionError
    @test draw_failure.phase === :proposal_draw
    @test draw_failure.sample_index == 2
    @test draw_failure.captured isa CapturedException
    @test draw_failure.captured.ex isa PreparedProposalFailure
    @test draw_proposal.draw_count == 2
    @test draw_proposal.density_count == 0

    unprovable_target_proposal = FailureSequenceProposal([1.0], _ -> 0.0)
    unprovable_target_sampler = prepare_sampler(
        Random.Xoshiro(1102),
        UnprovablePreparedTarget(_ -> 0.0),
        ImportanceSampling(unprovable_target_proposal; nsamples=1);
        threaded=false,
    )
    unprovable_target_failure = caught_exception(
        () -> importance_sample!(unprovable_target_sampler)
    )
    @test unprovable_target_failure isa SamplerExecutionError
    @test unprovable_target_failure.phase === :target
    @test unprovable_target_failure.sample_index == 1
    @test unprovable_target_failure.captured.ex isa ArgumentError
    @test unprovable_target_proposal.density_count == 0

    incompatible_target_proposal = FailureSequenceProposal([1.0], _ -> 0.0)
    incompatible_target_sampler = prepare_sampler(
        Random.Xoshiro(1104),
        IncompatiblePreparedTarget(),
        ImportanceSampling(incompatible_target_proposal; nsamples=1);
        threaded=false,
    )
    incompatible_target_failure = caught_exception(
        () -> importance_sample!(incompatible_target_sampler)
    )
    @test incompatible_target_failure isa SamplerExecutionError
    @test incompatible_target_failure.phase === :target
    @test incompatible_target_failure.sample_index == 1
    @test incompatible_target_failure.captured.ex isa ArgumentError
    @test incompatible_target_proposal.draw_count == 1
    @test incompatible_target_proposal.density_count == 0

    unprovable_proposal = UnprovablePreparedProposal(_ -> 0.0, 0)
    unprovable_proposal_sampler = prepare_sampler(
        Random.Xoshiro(1103),
        _ -> 0.0,
        ImportanceSampling(unprovable_proposal; nsamples=1);
        threaded=false,
    )
    unprovable_proposal_failure = caught_exception(
        () -> importance_sample!(unprovable_proposal_sampler)
    )
    @test unprovable_proposal_failure isa SamplerExecutionError
    @test unprovable_proposal_failure.phase === :proposal_logdensity
    @test unprovable_proposal_failure.sample_index == 1
    @test unprovable_proposal_failure.captured.ex isa ArgumentError
    @test unprovable_proposal.draw_count == 1

    busy_proposal = FailureSequenceProposal([1.0], _ -> 0.0)
    busy_sampler = prepare_sampler(
        Random.Xoshiro(1201),
        _ -> 0.0,
        ImportanceSampling(busy_proposal; nsamples=1);
        threaded=false,
    )
    setfield!(busy_sampler, :running, true)
    @test_throws SamplerBusyError importance_sample!(busy_sampler)
    setfield!(busy_sampler, :running, false)

    reentrant_proposal = FailureSequenceProposal([1.0, 2.0], _ -> 0.0)
    sampler_ref = Ref{Any}()
    reenter = Ref(true)
    reentrant_target = function (_)
        if reenter[]
            reenter[] = false
            importance_sample!(sampler_ref[])
        end
        return 0.0
    end
    reentrant_sampler = prepare_sampler(
        Random.Xoshiro(1202),
        reentrant_target,
        ImportanceSampling(reentrant_proposal; nsamples=1);
        threaded=false,
    )
    sampler_ref[] = reentrant_sampler
    reentrant_failure = caught_exception(() -> importance_sample!(reentrant_sampler))
    @test reentrant_failure isa SamplerExecutionError
    @test reentrant_failure.phase === :target
    @test reentrant_failure.sample_index == 1
    @test reentrant_failure.captured.ex isa SamplerBusyError
    @test getfield(reentrant_sampler, :running) === false

    recovered_result = importance_sample!(reentrant_sampler)
    @test recovered_result.samples == [2.0]
    @test recovered_result.logweights == [0.0]
end

@testset "threaded execution failures" begin
    nsamples = 32
    target_fail_at = 19
    target_proposal = ThreadFailureProposal(nsamples)
    target_tasks = Union{Nothing,Task}[nothing for _ in 1:nsamples]
    fail_target = Ref(true)
    target = ThreadFailureTarget(target_tasks, fail_target, target_fail_at)
    target_sampler = prepare_sampler(
        Random.Xoshiro(1501),
        target,
        ImportanceSampling(target_proposal; nsamples=nsamples);
        threaded=true,
    )
    target_failure = caught_exception(() -> importance_sample!(target_sampler))
    @test target_failure isa SamplerExecutionError
    @test target_failure.phase === :target
    @test target_failure.sample_index == target_fail_at
    @test target_failure.captured isa CapturedException
    @test target_failure.captured.ex isa ThreadedTargetFailure
    @test target_failure.captured.ex.sample_index == target_fail_at
    @test !any(target_proposal.density_seen)
    @test getfield(target_sampler, :running) === false

    fail_target[] = false
    recovered_target = importance_sample!(target_sampler)
    @test length(recovered_target) == nsamples

    proposal_fail_at = 23
    proposal = ThreadFailureProposal(nsamples; fail_density_at=proposal_fail_at)
    proposal_sampler = prepare_sampler(
        Random.Xoshiro(1502),
        _ -> 0.0,
        ImportanceSampling(proposal; nsamples=nsamples);
        threaded=true,
    )
    proposal_failure = caught_exception(() -> importance_sample!(proposal_sampler))
    @test proposal_failure isa SamplerExecutionError
    @test proposal_failure.phase === :proposal_logdensity
    @test proposal_failure.sample_index == proposal_fail_at
    @test proposal_failure.captured isa CapturedException
    @test proposal_failure.captured.ex isa ThreadedProposalFailure
    @test proposal_failure.captured.ex.sample_index == proposal_fail_at
    @test getfield(proposal_sampler, :running) === false

    if Threads.nthreads(:default) > 1
        recorded_target_tasks = [task for task in target_tasks if task !== nothing]
        recorded_proposal_tasks =
            [task for task in proposal.density_tasks if task !== nothing]
        caller_task = current_task()
        @test all(!=(caller_task), recorded_target_tasks)
        @test all(!=(caller_task), recorded_proposal_tasks)
        @test all(
            task -> Threads.threadpool(task) === :default,
            recorded_target_tasks,
        )
        @test all(
            task -> Threads.threadpool(task) === :default,
            recorded_proposal_tasks,
        )
        @test length(unique(recorded_target_tasks)) > 1
        @test length(unique(recorded_proposal_tasks)) > 1
    end
end

@testset "AMIS round errors expose the binding phase table" begin
    cases = (
        (cause=captured_sampler_execution_error(:proposal_draw), phase=:sampling),
        (cause=captured_sampler_execution_error(:target), phase=:target),
        (
            cause=captured_sampler_execution_error(:proposal_logdensity),
            phase=:denominator,
        ),
        (cause=captured_sampler_execution_error(:logweight), phase=:weight),
        (cause=AllZeroWeightsError(), phase=:moment),
        (cause=LinearAlgebra.PosDefException(1), phase=:factorization),
        (cause=ErrorException("result"), phase=:result_construction),
    )
    for case in cases
        covariance = case.phase === :factorization ? reshape([2.0], 1, 1) : nothing
        transfers = ImportanceSamplers._ResultTransferCounter(0, 0)
        failure = caught_exception() do
            ImportanceSamplers._capture_amis_round(
                2,
                case.phase,
                5,
                1,
                3,
                covariance,
                transfers,
            ) do
                throw(case.cause)
            end
        end

        @test failure isa AMISRoundError
        @test failure.round == 2
        @test failure.phase === case.phase
        @test failure.cause === case.cause
        @test keys(failure.diagnostics) == (
            :round_size,
            :completed_rounds,
            :cumulative_sample_count,
            :covariance,
            :transfers,
        )
        @test failure.diagnostics.round_size == 5
        @test failure.diagnostics.completed_rounds == 1
        @test failure.diagnostics.cumulative_sample_count == 3
        @test failure.diagnostics.transfers === transfers
        if case.phase === :factorization
            @test failure.diagnostics.covariance == (
                minimum_diagonal=2.0,
                maximum_absolute_entry=2.0,
            )
        else
            @test isnothing(failure.diagnostics.covariance)
        end
    end
end

@testset "AMIS whole-call transaction failures preserve the committed proposal" begin
    T = Float64
    algorithm = AMIS(SphericalGaussian(T(0), T(1)); rounds=2, round_size=[3, 3])
    batches = [T[-1, 0, 1], T[-0.5, 0.5, 1.5]]

    target_sampler = prepare_sampler(
        AMISFailureRNG(deepcopy(batches)),
        AMISFailingTarget{T}(0, 4),
        algorithm;
        threaded=false,
    )
    target_failure = assert_amis_transaction_failure(
        target_sampler,
        :target,
        SamplerExecutionError,
        2,
        3,
    )
    @test target_failure.cause.captured.ex isa ErrorException

    denominator_sampler = prepare_sampler(
        AMISFailureRNG([
            T[2sqrt(floatmax(T)), 0],
            T[-1.0e50, 1.0e50],
        ]),
        AMISZeroTarget{T}(),
        AMIS(SphericalGaussian(zero(T), T(1.0e-100)); rounds=1, round_size=2);
        threaded=false,
    )
    denominator_failure = assert_amis_transaction_failure(
        denominator_sampler,
        :denominator,
        SamplerExecutionError,
        1,
        2,
    )
    @test denominator_failure.cause.phase === :proposal_logdensity
    @test denominator_failure.cause.sample_index == 1
    @test denominator_failure.cause.captured.ex isa DomainError
    @test importance_sample!(denominator_sampler) isa WeightedSamples
    @test denominator_sampler.rng.index == 3

    sampling_sampler = prepare_sampler(
        AMISFailureRNG(deepcopy(batches[1:1])),
        AMISZeroTarget{T}(),
        algorithm;
        threaded=false,
    )
    assert_amis_transaction_failure(
        sampling_sampler,
        :sampling,
        BoundsError,
        2,
        2,
    )

    zero_sampler = prepare_sampler(
        AMISFailureRNG(deepcopy(batches)),
        _ -> T(-Inf),
        algorithm;
        threaded=false,
    )
    assert_amis_transaction_failure(
        zero_sampler,
        :moment,
        AllZeroWeightsError,
        1,
        2,
    )

    invalid_sampler = prepare_sampler(
        AMISFailureRNG(deepcopy(batches)),
        _ -> T(NaN),
        algorithm;
        threaded=false,
    )
    invalid_failure = assert_amis_transaction_failure(
        invalid_sampler,
        :target,
        SamplerExecutionError,
        1,
        2,
    )
    @test invalid_failure.cause.captured.ex isa DomainError

    tiny = T(1.0e-200)
    covariance_sampler = prepare_sampler(
        AMISFailureRNG([zeros(T, 6)]),
        _ -> zero(T),
        AMIS(
            FactorGaussian(zeros(T, 2), T[tiny 0; 0 tiny]);
            rounds=1,
            round_size=3,
        );
        threaded=false,
    )
    assert_amis_transaction_failure(
        covariance_sampler,
        :factorization,
        LinearAlgebra.PosDefException,
        1,
        2,
    )

    moment_base = prepare_sampler(
        AMISFailureRNG([T[-1, 0, 1, 0, -1, 1]]),
        AMISZeroTarget{T}(),
        AMIS(FactorGaussian(zeros(T, 2), T[1 0; 0 1]); rounds=1, round_size=3);
        threaded=false,
    )
    old_workspace = moment_base.method_state.workspace
    moment_workspace = ImportanceSamplers._AMISWorkspace(
        old_workspace.samples,
        old_workspace.logtargets,
        old_workspace.lognumerators,
        old_workspace.logweights,
        old_workspace.normalized_weights,
        old_workspace.centered_scaled,
        AMISMomentFailureMatrix(old_workspace.covariance),
        old_workspace.candidate_mean,
        old_workspace.candidate_scale,
        old_workspace.candidate_lognormalizer,
    )
    moment_state = ImportanceSamplers._PreparedAMIS(
        moment_base.method_state.schedule,
        moment_base.method_state.offsets,
        moment_base.method_state.logcounts,
        moment_base.method_state.history,
        moment_workspace,
        moment_base.method_state.committed_in_workspace,
    )
    moment_sampler = ImportanceSamplers._PreparedImportanceSampler(
        moment_base.rng,
        moment_base.random_buffers,
        moment_base.target,
        moment_base.algorithm,
        moment_state,
        moment_base.device,
        moment_base.factor_execution,
        moment_base.threaded,
        false,
        false,
    )
    moment_failure = assert_amis_transaction_failure(
        moment_sampler,
        :moment,
        ErrorException,
        1,
        2,
    )
    @test moment_failure.cause.msg ==
          "intentional AMIS moment multiplication failure"

    for F in (Float32, Float64)
        huge_factor = F(2) * sqrt(floatmax(F))
        overflow_sampler = prepare_sampler(
            AMISFailureRNG([F[-1, 0, 1]]),
            AMISZeroTarget{F}(),
            AMIS(
                FactorGaussian(F[0], reshape(F[huge_factor], 1, 1));
                rounds=1,
                round_size=3,
            );
            threaded=false,
        )
        assert_amis_transaction_failure(
            overflow_sampler,
            :factorization,
            ArgumentError,
            1,
            2,
        )
    end

    result_sampler = amis_result_copy_failure_sampler(prepare_sampler(
        AMISFailureRNG(deepcopy(batches)),
        AMISZeroTarget{T}(),
        algorithm;
        threaded=false,
    ))
    result_failure = assert_amis_transaction_failure(
        result_sampler,
        :result_construction,
        ErrorException,
        2,
        3,
    )
    @test result_failure.diagnostics.completed_rounds == 2
    @test result_failure.cause.msg == "intentional AMIS result copy failure"
    @test covariance_sampler.method_state.workspace.covariance == zeros(T, 2, 2)
    @test covariance_sampler.running === false
end

@testset "AMIS run-start proposal authority copy is atomic" begin
    for fail_at in (2, 3)
        sampler, injected = amis_publication_failure_sampler(fail_at)
        @test importance_sample!(sampler) isa WeightedSamples
        @test sampler.method_state.committed_in_workspace
        before = amis_proposal_bits(current_proposal(sampler))

        failure = caught_exception(() -> importance_sample!(sampler))

        @test failure isa AMISRoundError
        if failure isa AMISRoundError
            @test failure.round == 1
            @test failure.phase === :result_construction
            @test failure.cause isa ErrorException
            @test failure.cause.msg ==
                  "intentional AMIS publication failure $fail_at"
        end
        @test amis_proposal_bits(current_proposal(sampler)) == before
        @test injected.writes == fail_at
        @test !sampler.running
        @test sampler.method_state.committed_in_workspace
    end
end

@testset "AMIS later run failure preserves the workspace proposal" begin
    T = Float64
    target = AMISFailingTarget{T}(0, typemax(Int))
    sampler = prepare_sampler(
        AMISFailureRNG(fill(T[-1, 0, 1], 2)),
        target,
        AMIS(SphericalGaussian(zero(T), one(T)); rounds=1, round_size=3);
        threaded=false,
    )
    @test importance_sample!(sampler) isa WeightedSamples
    @test sampler.method_state.committed_in_workspace
    before = amis_proposal_bits(current_proposal(sampler))
    target.fail_at = target.calls + 1

    failure = caught_exception(() -> importance_sample!(sampler))

    @test failure isa AMISRoundError
    @test failure.phase === :target
    @test amis_proposal_bits(current_proposal(sampler)) == before
    @test !sampler.method_state.committed_in_workspace
    @test !sampler.running
end

@testset "CPU factor AMIS stages, carries over, and rolls back" begin
    for T in (Float32, Float64)
        batches = [
            T[-1, 1],
            T[-0.5, 0.5],
            T[0.25, -0.75],
            T[0.5, -0.25],
        ]
        proposal = FactorGaussian(T[0.5], reshape(T[1.25], 1, 1))
        algorithm = AMIS(proposal; rounds=2, round_size=[2, 2])
        sampler = prepare_sampler(
            AMISFailureRNG(deepcopy(batches)),
            AMISZeroTarget{T}(),
            algorithm;
            threaded=false,
        )

        first = importance_sample!(sampler)
        history = sampler.method_state.history
        staged_mean = history.means[1, 2]
        staged_factor = history.factors[1, 1, 2]
        @test first.samples[1, 3] ≈
              staged_mean + staged_factor * batches[2][1] rtol = 8eps(T)
        @test history.lognormalizers[2] ≈
              -log(T(2pi)) / T(2) - log(staged_factor) rtol = 8eps(T)

        committed = current_proposal(sampler)
        @test sampler.method_state.committed_in_workspace
        second = importance_sample!(sampler)
        @test second.samples[1, 1] ≈
              committed.location[1] +
              committed.scale.factor[1, 1] * batches[3][1] rtol = 8eps(T)
        @test sampler.method_state.committed_in_workspace

        rollback_sampler = amis_result_copy_failure_sampler(prepare_sampler(
            AMISFailureRNG(deepcopy(batches[1:2])),
            AMISZeroTarget{T}(),
            algorithm;
            threaded=false,
        ))
        rollback_failure = assert_amis_transaction_failure(
            rollback_sampler,
            :result_construction,
            ErrorException,
            2,
            3,
        )
        @test rollback_failure.diagnostics.completed_rounds == 2
        @test rollback_failure.cause.msg ==
              "intentional AMIS result copy failure"
    end
end

function gram_is_all_zero_local_state(all_groups)
    T = Float64
    bank = ProposalBank([
        FactorGaussian(T[0, 0], T[2 0; 1 3]),
        FactorGaussian(T[10, 10], T[1 0; 0.5 2]),
    ])
    state = ImportanceSamplers._prepare_method_state(FirstOrderGRAMIS(
        bank;
        rounds=1,
        round_size=8,
        repulsion_strength=zero(T),
        covariance_ess_threshold=3,
    ))
    state.workspace.samples .= T[
        -1 1 0 0 8 10 12 10
        0 0 -1 1 10 8 10 12
    ]
    local_logweights = all_groups ?
                       fill(T(-Inf), 8) :
                       T[-Inf, -Inf, -Inf, -Inf, 0, 0, 0, 0]
    state.workspace.local_logweights .= local_logweights
    return state
end

@testset "FirstOrderGRAMIS all-zero local covariance fallback" begin
    expected_covariances = (
        [4.0 2.0; 2.0 10.0],
        [1.0 0.5; 0.5 4.25],
    )
    for all_groups in (false, true)
        state = gram_is_all_zero_local_state(all_groups)
        candidate_locations = copy(state.candidate.locations)
        candidate_factors = copy(state.candidate.factors)
        candidate_lognormalizers = copy(state.candidate.lognormalizers)

        ImportanceSamplers._fit_local_covariances!(
            state,
            1,
            ImportanceSamplers._SerialCPUExecution(),
        )

        @test state.workspace.factor_status[1] ==
              ImportanceSamplers._GRAMIS_ALL_ZERO_LOCAL
        @test state.workspace.covariances[:, :, 1] == expected_covariances[1]
        @test state.workspace.local_ess[1] == 0
        @test state.workspace.tempering_powers[1] == 0
        if all_groups
            @test state.workspace.factor_status == fill(
                ImportanceSamplers._GRAMIS_ALL_ZERO_LOCAL,
                2,
            )
            @test state.workspace.covariances[:, :, 2] == expected_covariances[2]
            @test state.workspace.local_ess == zeros(2)
            @test state.workspace.tempering_powers == zeros(2)
        else
            @test state.workspace.factor_status[2] ==
                  ImportanceSamplers._GRAMIS_COVARIANCE_READY
        end
        @test state.candidate.locations == candidate_locations
        @test state.candidate.factors == candidate_factors
        @test state.candidate.lognormalizers == candidate_lognormalizers
    end
end

struct GRAMISDerivativeFailureValue{T}
    value::T
end

struct GRAMISDerivativeFailureGradient{T}
    value::T
end

(target::GRAMISDerivativeFailureValue)(sample) = target.value

function (gradient::GRAMISDerivativeFailureGradient)(destination, sample)
    fill!(destination, gradient.value)
    return destination
end

function gram_is_failure_derivative(::Type{T}, value, gradient_value, execution) where {T}
    locations = reshape(T[1, 2], 1, :)
    proposal = FactorGaussian(T[0], reshape(T[1], 1, 1))
    prepared = ImportanceSamplers._prepare_target(
        LogTarget(
            GRAMISDerivativeFailureValue{T}(T(value));
            grad=GRAMISDerivativeFailureGradient{T}(T(gradient_value)),
        ),
        proposal,
    )
    target = ImportanceSamplers._bind_resolved_target(
        prepared,
        view(locations, :, 1),
    )
    worker_count = execution isa ImportanceSamplers._SerialCPUExecution ?
                   1 : length(Threads.threadpooltids(:default))
    gradient = ImportanceSamplers._prepare_bound_gradient(
        prepared,
        view(locations, :, 1),
        worker_count,
    )
    values = zeros(T, 2)
    gradients = zeros(T, 1, 2)
    failure = caught_exception() do
        ImportanceSamplers._evaluate_frozen_gradients!(
            values,
            gradients,
            target,
            gradient,
            locations,
            execution,
        )
    end
    return (; failure, locations)
end

@testset "FirstOrderGRAMIS derivative failures are typed" begin
    for T in (Float32, Float64), execution in (
        ImportanceSamplers._SerialCPUExecution(),
        ImportanceSamplers._ThreadedCPUExecution(),
    )
        for value in (T(-Inf), T(NaN), T(Inf))
            result = gram_is_failure_derivative(T, value, zero(T), execution)
            @test result.failure isa
                  ImportanceSamplers._FirstOrderGRAMISDerivativeError
            if result.failure isa
               ImportanceSamplers._FirstOrderGRAMISDerivativeError
                @test result.failure.proposal_slot == 1
                @test result.failure.reason === :frozen_value_nonfinite
            end
            @test result.locations == reshape(T[1, 2], 1, :)
        end

        for gradient_value in (T(-Inf), T(NaN), T(Inf))
            result = gram_is_failure_derivative(
                T,
                zero(T),
                gradient_value,
                execution,
            )
            @test result.failure isa
                  ImportanceSamplers._FirstOrderGRAMISDerivativeError
            if result.failure isa
               ImportanceSamplers._FirstOrderGRAMISDerivativeError
                @test result.failure.proposal_slot == 1
                @test result.failure.reason === :gradient_nonfinite
            end
            @test result.locations == reshape(T[1, 2], 1, :)
        end

        moves = zeros(T, 1, 2)
        gradients = fill(T(2), 1, 2)
        factors = reshape(fill(floatmax(T), 2), 1, 1, 2)
        failure = caught_exception() do
            ImportanceSamplers._precondition_gradients!(
                moves,
                gradients,
                factors,
                execution,
            )
        end
        @test failure isa ImportanceSamplers._FirstOrderGRAMISDerivativeError
        if failure isa ImportanceSamplers._FirstOrderGRAMISDerivativeError
            @test failure.proposal_slot == 1
            @test failure.reason === :move_nonfinite
        end
    end
end

@testset "FirstOrderGRAMIS rejects -Inf and fails on invalid candidate values" begin
    for T in (Float32, Float64), execution in (
        ImportanceSamplers._SerialCPUExecution(),
        ImportanceSamplers._ThreadedCPUExecution(),
    )
        locations = reshape(T[0, 10], 1, :)
        moves = ones(T, 1, 2)
        frozen_values = zeros(T, 2)
        candidate_locations = similar(locations)
        candidate_values = similar(frozen_values)
        active_mask = similar(frozen_values, Bool)
        steps = similar(frozen_values)
        trials = similar(frozen_values, Int)

        ImportanceSamplers._backtrack_means!(
            candidate_locations,
            candidate_values,
            active_mask,
            steps,
            trials,
            GRAMISDerivativeFailureValue{T}(T(-Inf)),
            frozen_values,
            locations,
            moves,
            2,
            execution,
        )
        @test candidate_locations == locations
        @test candidate_values == frozen_values
        @test steps == zeros(T, 2)
        @test trials == fill(2, 2)

        for candidate_value in (T(NaN), T(Inf))
            failure = caught_exception() do
                ImportanceSamplers._backtrack_means!(
                    candidate_locations,
                    candidate_values,
                    active_mask,
                    steps,
                    trials,
                    GRAMISDerivativeFailureValue{T}(candidate_value),
                    frozen_values,
                    locations,
                    moves,
                    2,
                    execution,
                )
            end
            @test failure isa ImportanceSamplers._FirstOrderGRAMISDerivativeError
            if failure isa ImportanceSamplers._FirstOrderGRAMISDerivativeError
                @test failure.proposal_slot == 1
                @test failure.reason === :candidate_value_nonfinite
            end
        end
    end
end

@testset "FirstOrderGRAMIS zero step is exhaustion-only" begin
    for T in (Float32, Float64)
        max_trials = T === Float32 ? 150 : 1075
        locations = reshape(T[0, 10], 1, :)
        moves = ones(T, 1, 2)
        frozen_values = zeros(T, 2)
        candidate_locations = similar(locations)
        candidate_values = similar(frozen_values)
        active_mask = similar(frozen_values, Bool)
        steps = similar(frozen_values)
        trials = similar(frozen_values, Int)

        ImportanceSamplers._backtrack_means!(
            candidate_locations,
            candidate_values,
            active_mask,
            steps,
            trials,
            GRAMISDerivativeFailureValue{T}(T(-Inf)),
            frozen_values,
            locations,
            moves,
            max_trials,
            ImportanceSamplers._SerialCPUExecution(),
        )

        @test !iszero(ldexp(one(T), 1 - max_trials))
        @test candidate_locations == locations
        @test candidate_values == frozen_values
        @test steps == zeros(T, 2)
        @test trials == fill(max_trials, 2)
    end
end

function gram_is_transaction_sampler(; invalid_diagnostics=false)
    T = Float64
    bank = ProposalBank([
        FactorGaussian(T[-2], reshape(T[0.75], 1, 1)),
        FactorGaussian(T[2], reshape(T[1.25], 1, 1)),
    ])
    target = GRAMISTransactionTarget{T}(false, 0)
    sampler = prepare_sampler(
        AMISFailureRNG(fill(T[-1, 0, 1, -1, 0, 1], 8)),
        LogTarget(target; grad=gram_is_transaction_gradient!),
        FirstOrderGRAMIS(
            bank;
            rounds=1,
            round_size=6,
            repulsion_strength=zero(T),
            covariance_ess_threshold=2,
        );
        threaded=false,
    )
    invalid_diagnostics || return sampler, target

    state = sampler.method_state
    workspace = state.workspace
    workspace_values = map(fieldnames(typeof(workspace))) do name
        name === :local_ess ?
        GRAMISInvalidDiagnosticPrototype(workspace.local_ess) :
        getfield(workspace, name)
    end
    invalid_workspace = ImportanceSamplers._FirstOrderGRAMISWorkspace(
        workspace_values...,
    )
    invalid_state = ImportanceSamplers._PreparedFirstOrderGRAMIS(
        state.committed,
        state.run,
        state.candidate,
        state.plan,
        state.repulsion_strength,
        state.covariance_rate,
        state.covariance_ess_threshold,
        state.covariance_regularization,
        state.tempering_tolerance,
        state.tempering_max_iterations,
        state.repulsion_softening,
        state.max_backtracking_trials,
        state.serial_gradient,
        state.threaded_gradient,
        state.active_repulsion_rounds,
        invalid_workspace,
    )
    invalid_sampler = ImportanceSamplers._PreparedImportanceSampler(
        sampler.rng,
        sampler.random_buffers,
        sampler.target,
        sampler.algorithm,
        invalid_state,
        sampler.device,
        sampler.factor_execution,
        sampler.threaded,
        false,
        false,
    )
    return invalid_sampler, target
end

function gram_is_two_round_phase_sampler(; diagnostic_copy_failure=false)
    T = Float64
    bank = ProposalBank([
        FactorGaussian(T[-2], reshape(T[0.75], 1, 1)),
        FactorGaussian(T[2], reshape(T[1.25], 1, 1)),
    ])
    target = GRAMISPhaseTarget{T}(0, typemax(Int))
    gradient = GRAMISPhaseGradient(0, typemax(Int))
    sampler = prepare_sampler(
        AMISFailureRNG(fill(T[-1, 0, 1, -1, 0, 1], 16)),
        LogTarget(target; grad=gradient),
        FirstOrderGRAMIS(
            bank;
            rounds=2,
            round_size=6,
            repulsion_strength=zero(T),
            covariance_ess_threshold=2,
        );
        threaded=false,
    )
    diagnostic_copy_failure || return sampler, target, gradient

    state = sampler.method_state
    write_failure = GRAMISWriteFailure(true, 0, 7)
    workspace_values = map(fieldnames(typeof(state.workspace))) do name
        value = getfield(state.workspace, name)
        name in (:samples, :pooled_covariance) ?
        GRAMISResultCopyFailureArray(value, write_failure, false) : value
    end
    workspace = ImportanceSamplers._FirstOrderGRAMISWorkspace(
        workspace_values...,
    )
    return gram_is_rebuild_sampler(sampler, workspace), target, gradient
end

function gram_is_proposal_phase_sampler()
    T = Float64
    bank = ProposalBank([
        FactorGaussian(T[-2], reshape(T[0.75], 1, 1)),
        FactorGaussian(T[2], reshape(T[1.25], 1, 1)),
    ])
    target = GRAMISProposalInjectionTarget{T}(0, 19)
    return prepare_sampler(
        AMISFailureRNG(fill(T[-1, 0, 1, -1, 0, 1], 16)),
        LogTarget(target; grad=gram_is_transaction_gradient!),
        FirstOrderGRAMIS(
            bank;
            rounds=2,
            round_size=6,
            repulsion_strength=zero(T),
            covariance_ess_threshold=2,
        );
        threaded=false,
    )
end

function gram_is_population_bits(sampler)
    proposal = current_proposal(sampler)
    return map(proposal.proposals) do component
        (
            map(bitstring, component.location),
            map(bitstring, component.scale.factor),
            bitstring(component.lognormalizer),
        )
    end
end

function gram_is_pointer_roles_valid(sampler)
    state = sampler.method_state
    banks = (state.committed, state.run, state.candidate)
    return banks[1] !== banks[2] && banks[1] !== banks[3] &&
           banks[2] !== banks[3] &&
           banks[1].locations !== banks[2].locations &&
           banks[1].locations !== banks[3].locations &&
           banks[2].locations !== banks[3].locations
end

function gram_is_rebuild_sampler(sampler, workspace)
    state = sampler.method_state
    rebuilt_state = ImportanceSamplers._PreparedFirstOrderGRAMIS(
        state.committed,
        state.run,
        state.candidate,
        state.plan,
        state.repulsion_strength,
        state.covariance_rate,
        state.covariance_ess_threshold,
        state.covariance_regularization,
        state.tempering_tolerance,
        state.tempering_max_iterations,
        state.repulsion_softening,
        state.max_backtracking_trials,
        state.serial_gradient,
        state.threaded_gradient,
        state.active_repulsion_rounds,
        workspace,
    )
    return ImportanceSamplers._PreparedImportanceSampler(
        sampler.rng,
        sampler.random_buffers,
        sampler.target,
        sampler.algorithm,
        rebuilt_state,
        sampler.device,
        sampler.factor_execution,
        sampler.threaded,
        false,
        sampler.executed,
    )
end

function gram_is_rebuild_sampler(sampler, workspace, plan)
    state = sampler.method_state
    rebuilt_state = ImportanceSamplers._PreparedFirstOrderGRAMIS(
        state.committed,
        state.run,
        state.candidate,
        plan,
        state.repulsion_strength,
        state.covariance_rate,
        state.covariance_ess_threshold,
        state.covariance_regularization,
        state.tempering_tolerance,
        state.tempering_max_iterations,
        state.repulsion_softening,
        state.max_backtracking_trials,
        state.serial_gradient,
        state.threaded_gradient,
        state.active_repulsion_rounds,
        workspace,
    )
    return ImportanceSamplers._PreparedImportanceSampler(
        sampler.rng,
        sampler.random_buffers,
        sampler.target,
        sampler.algorithm,
        rebuilt_state,
        sampler.device,
        sampler.factor_execution,
        sampler.threaded,
        false,
        sampler.executed,
    )
end

function gram_is_full_error_diagnostics(
    failure,
    round,
    completed,
    round_size,
    cause_message,
)
    @test failure isa FirstOrderGRAMISRoundError
    @test failure.round == round
    @test failure.phase === :result_construction
    @test failure.cause isa ErrorException
    @test failure.cause.msg == cause_message
    @test failure.diagnostics.round_size == round_size
    @test failure.diagnostics.completed_rounds == completed
    @test failure.diagnostics.covariance === nothing
    @test failure.diagnostics.derivative === nothing
    @test failure.diagnostics.transfers isa
          ImportanceSamplers._ResultTransferCounter
    @test failure.diagnostics.transfers.count == 0
    @test failure.diagnostics.transfers.bytes == 0
    @test failure.diagnostics.pre_call_state_preserved === true
end

@testset "FirstOrderGRAMIS CPU failure seams preserve zero transfers" begin
    device = MLDataDevices.CPUDevice()
    execution = ImportanceSamplers._SerialCPUExecution()

    transfers = ImportanceSamplers._ResultTransferCounter(0, 0)
    proposal_failure = caught_exception() do
        ImportanceSamplers._add_first_order_gramis_repulsion!(
            device,
            reshape([NaN, 0.0], 1, 2),
            zeros(1, 2),
            transfers,
            execution,
        )
    end
    @test proposal_failure isa ImportanceSamplers._FirstOrderGRAMISProposalError
    @test proposal_failure.proposal_slot == 1
    @test proposal_failure.reason === :location_nonfinite
    @test isnan(proposal_failure.value)
    @test transfers.count == 0
    @test transfers.bytes == 0

    sampler, _ = gram_is_transaction_sampler()
    candidate = sampler.method_state.candidate
    candidate.factors[1, 1, 1] = NaN
    transfers = ImportanceSamplers._ResultTransferCounter(0, 0)
    factor_failure = caught_exception() do
        ImportanceSamplers._validate_first_order_gramis_candidate_factors!(
            device,
            candidate,
            zeros(UInt8, 2),
            transfers,
            execution,
        )
    end
    @test factor_failure isa ImportanceSamplers._FirstOrderGRAMISProposalError
    @test factor_failure.proposal_slot == 1
    @test factor_failure.reason === :factor_nonfinite
    @test isnan(factor_failure.value)
    @test transfers.count == 0
    @test transfers.bytes == 0

    transfers = ImportanceSamplers._ResultTransferCounter(0, 0)
    covariance_failure = caught_exception() do
        ImportanceSamplers._throw_first_order_gramis_covariance_failure(
            device,
            [-1, 0],
            transfers,
            execution,
        )
    end
    @test covariance_failure isa
          ImportanceSamplers._FirstOrderGRAMISCovarianceError
    @test covariance_failure.proposal_slot == 1
    @test covariance_failure.info == -1
    @test transfers.count == 0
    @test transfers.bytes == 0
end

function gram_is_result_bits(result)
    return (
        samples=map(bitstring, result.samples),
        logweights=map(bitstring, result.logweights),
        provenance=deepcopy(result.provenance),
        local_ess=map(bitstring, result.diagnostics.local_ess),
        tempering_powers=map(bitstring, result.diagnostics.tempering_powers),
        fallback_status=copy(result.diagnostics.fallback_status),
        accepted_steps=map(bitstring, result.diagnostics.accepted_steps),
        backtracking_trials=copy(result.diagnostics.backtracking_trials),
        collision_counts=copy(result.diagnostics.collision_counts),
    )
end

@testset "FirstOrderGRAMIS public round errors translate private phases" begin
    sampler, _ = gram_is_transaction_sampler()
    state = sampler.method_state
    transfers = ImportanceSamplers._ResultTransferCounter(0, 0)
    causes = (
        (
            ImportanceSamplers._FirstOrderGRAMISDerivativeError(
                1,
                :gradient_nonfinite,
                NaN,
            ),
            :derivative,
        ),
        (
            ImportanceSamplers._FirstOrderGRAMISRepulsionError(
                1,
                :force_nonfinite,
                NaN,
            ),
            :repulsion,
        ),
        (ImportanceSamplers._FirstOrderGRAMISCovarianceError(1, 2), :covariance),
        (
            ImportanceSamplers._FirstOrderGRAMISProposalError(
                1,
                :location_nonfinite,
                NaN,
            ),
            :proposal,
        ),
        (captured_sampler_execution_error(:target), :target),
        (captured_sampler_execution_error(:proposal_logdensity), :denominator),
        (captured_sampler_execution_error(:logweight), :weight),
    )

    for (cause, expected_phase) in causes
        failure = caught_exception() do
            ImportanceSamplers.@_capture_first_order_gramis_round(
                state,
                transfers,
                1,
                :backtracking,
                0,
                throw(cause),
            )
        end
        @test failure isa FirstOrderGRAMISRoundError
        @test failure.round == 1
        @test failure.phase === expected_phase
        @test failure.cause === cause
        @test failure.diagnostics.round_size == 6
        @test failure.diagnostics.completed_rounds == 0
        @test failure.diagnostics.transfers === transfers
        @test failure.diagnostics.pre_call_state_preserved === true
    end
end

@testset "FirstOrderGRAMIS calls commit once and recover after failures" begin
    sampler, target = gram_is_transaction_sampler()
    initial = gram_is_population_bits(sampler)
    target.fail = true
    first_failure = caught_exception(() -> importance_sample!(sampler))
    @test first_failure isa FirstOrderGRAMISRoundError
    @test first_failure.phase === :target
    @test first_failure.diagnostics.completed_rounds == 0
    @test gram_is_population_bits(sampler) == initial
    @test sampler.rng.index == 2

    target.fail = false
    first = importance_sample!(sampler)
    committed = gram_is_population_bits(sampler)
    @test committed != initial
    retained_result = gram_is_result_bits(first)

    target.fail = true
    later_failure = caught_exception(() -> importance_sample!(sampler))
    @test later_failure isa FirstOrderGRAMISRoundError
    @test later_failure.phase === :target
    @test gram_is_population_bits(sampler) == committed
    @test sampler.rng.index == 4
    @test gram_is_result_bits(first) == retained_result

    target.fail = false
    recovered = importance_sample!(sampler)
    @test recovered isa WeightedSamples
    @test length(recovered) == 6
    @test sampler.rng.index == 5
    @test !sampler.running
end

@testset "FirstOrderGRAMIS result construction failure rolls back" begin
    sampler, _ = gram_is_transaction_sampler(; invalid_diagnostics=true)
    before = gram_is_population_bits(sampler)
    failure = caught_exception(() -> importance_sample!(sampler))

    @test failure isa FirstOrderGRAMISRoundError
    @test failure.phase === :result_construction
    @test failure.cause isa ArgumentError
    @test failure.diagnostics.completed_rounds == 1
    @test failure.diagnostics.pre_call_state_preserved === true
    @test gram_is_population_bits(sampler) == before
    @test sampler.rng.index == 2
    @test !sampler.running

    invalid_workspace = sampler.method_state.workspace
    workspace_values = map(fieldnames(typeof(invalid_workspace))) do name
        name === :local_ess ?
        invalid_workspace.local_ess.storage :
        getfield(invalid_workspace, name)
    end
    valid_workspace = ImportanceSamplers._FirstOrderGRAMISWorkspace(
        workspace_values...,
    )
    recovered_sampler = gram_is_rebuild_sampler(sampler, valid_workspace)
    recovered = importance_sample!(recovered_sampler)
    @test recovered isa WeightedSamples
    @test gram_is_population_bits(recovered_sampler) != before
    @test recovered_sampler.rng.index == 3
end

@testset "FirstOrderGRAMIS result allocation failures use the public contract" begin
    base_sampler, _ = gram_is_transaction_sampler()
    state = base_sampler.method_state
    failure_switch = GRAMISOneShotFailure(true)
    workspace_values = map(fieldnames(typeof(state.workspace))) do name
        value = getfield(state.workspace, name)
        name in (:samples, :pooled_covariance) ?
        GRAMISResultAllocationFailureArray(value, failure_switch) : value
    end
    workspace = ImportanceSamplers._FirstOrderGRAMISWorkspace(
        workspace_values...,
    )
    sampler = gram_is_rebuild_sampler(base_sampler, workspace)
    before = gram_is_population_bits(sampler)

    failure = caught_exception(() -> importance_sample!(sampler))
    gram_is_full_error_diagnostics(
        failure,
        1,
        0,
        6,
        "intentional GRAMIS result allocation failure",
    )
    @test occursin("result allocation", sprint(showerror, failure.cause))
    @test gram_is_population_bits(sampler) == before
    @test sampler.rng.index == 1
    @test sampler.method_state.committed !== sampler.method_state.run
    @test sampler.method_state.run !== sampler.method_state.candidate

    recovered = importance_sample!(sampler)
    @test recovered isa WeightedSamples
    @test sampler.rng.index == 2
    @test gram_is_population_bits(sampler) != before
end

@testset "FirstOrderGRAMIS diagnostic assembly failures use the public contract" begin
    base_sampler, _ = gram_is_transaction_sampler()
    state = base_sampler.method_state
    plan = state.plan
    schedule = GRAMISCollectFailureSchedule(
        plan.schedule,
        GRAMISOneShotFailure(true),
    )
    failing_plan = ImportanceSamplers._DeterministicAllocationPlan(
        schedule,
        plan.counts,
        plan.assignments,
        plan.logcoefficients,
        plan.offsets,
    )
    sampler = gram_is_rebuild_sampler(base_sampler, state.workspace, failing_plan)
    before = gram_is_population_bits(sampler)

    failure = caught_exception(() -> importance_sample!(sampler))
    gram_is_full_error_diagnostics(
        failure,
        1,
        1,
        6,
        "intentional GRAMIS diagnostic assembly failure",
    )
    @test occursin("diagnostic assembly", sprint(showerror, failure.cause))
    @test gram_is_population_bits(sampler) == before
    @test sampler.rng.index == 2
    @test sampler.method_state.committed !== sampler.method_state.run
    @test sampler.method_state.run !== sampler.method_state.candidate

    recovered = importance_sample!(sampler)
    @test recovered isa WeightedSamples
    @test sampler.rng.index == 3
    @test gram_is_population_bits(sampler) != before
end

@testset "FirstOrderGRAMIS round-two failures preserve post-swap transactions" begin
    cases = (
        (name=:sampling, expected=:sampling, diagnostic=false),
        (name=:target, expected=:target, diagnostic=false),
        (name=:derivative, expected=:derivative, diagnostic=false),
        (name=:covariance, expected=:covariance, diagnostic=false),
        (name=:repulsion, expected=:repulsion, diagnostic=false),
        (name=:diagnostics, expected=:diagnostics, diagnostic=true),
    )

    for case in cases
        sampler, target, gradient = gram_is_two_round_phase_sampler(
            ; diagnostic_copy_failure=case.diagnostic,
        )
        original_second_batch = copy(sampler.rng.batches[2])
        original_covariance_rate = sampler.method_state.covariance_rate[2]
        original_repulsion_strength = sampler.method_state.repulsion_strength[2]
        if case.name === :sampling
            sampler.rng.batches[2] = [0.0]
        elseif case.name === :target
            target.fail_at = 11
        elseif case.name === :derivative
            gradient.fail_at = 3
        elseif case.name === :covariance
            sampler.method_state.covariance_rate[2] = NaN
        elseif case.name === :repulsion
            sampler.method_state.repulsion_strength[2] = Inf
            push!(sampler.method_state.active_repulsion_rounds, 2)
        end

        before = gram_is_population_bits(sampler)
        failure = caught_exception(() -> importance_sample!(sampler))
        @test failure isa FirstOrderGRAMISRoundError
        @test failure.round == 2
        @test failure.phase === case.expected
        @test failure.diagnostics.round_size == 6
        @test failure.diagnostics.completed_rounds ==
              (case.name === :diagnostics ? 2 : 1)
        @test failure.diagnostics.transfers isa
              ImportanceSamplers._ResultTransferCounter
        @test failure.diagnostics.transfers.count == 0
        @test failure.diagnostics.transfers.bytes == 0
        @test failure.diagnostics.pre_call_state_preserved === true
        @test failure.cause !== nothing
        if case.name === :covariance
            @test failure.cause isa
                  ImportanceSamplers._FirstOrderGRAMISCovarianceError
            @test failure.cause.proposal_slot == 1
            @test failure.cause.info == -1
            @test failure.diagnostics.covariance !== nothing
            @test failure.diagnostics.covariance.proposal_slot == 1
            @test failure.diagnostics.covariance.info == -1
            @test failure.diagnostics.derivative === nothing
        elseif case.name === :derivative
            @test failure.cause isa
                  ImportanceSamplers._FirstOrderGRAMISDerivativeError
            @test failure.cause.proposal_slot == 1
            @test failure.cause.reason === :gradient_nonfinite
            @test isnan(failure.cause.value)
            @test failure.diagnostics.covariance === nothing
            @test failure.diagnostics.derivative !== nothing
            @test failure.diagnostics.derivative.proposal_slot == 1
            @test failure.diagnostics.derivative.reason === :gradient_nonfinite
            @test isnan(failure.diagnostics.derivative.value)
        elseif case.name === :repulsion
            @test failure.cause isa
                  ImportanceSamplers._FirstOrderGRAMISRepulsionError
            @test failure.cause.proposal_slot == 1
            @test failure.cause.reason === :force_nonfinite
            @test failure.cause.value == Inf
            @test failure.diagnostics.covariance === nothing
            @test failure.diagnostics.derivative === nothing
        elseif case.name === :target
            @test failure.cause isa SamplerExecutionError
            @test failure.cause.phase === :target
            @test failure.cause.sample_index == 1
            @test failure.cause.captured.ex isa ErrorException
            @test failure.cause.captured.ex.msg ==
                  "intentional GRAMIS phase target failure"
            @test failure.diagnostics.covariance === nothing
            @test failure.diagnostics.derivative === nothing
        elseif case.name === :sampling
            @test failure.cause isa BoundsError
            @test failure.cause.a === sampler.rng.batches[2]
            @test failure.cause.i == 1:6
            @test failure.diagnostics.covariance === nothing
            @test failure.diagnostics.derivative === nothing
        elseif case.name === :diagnostics
            @test failure.cause isa ErrorException
            @test failure.cause.msg ==
                  "intentional GRAMIS round-two diagnostic copy failure"
            @test failure.diagnostics.covariance === nothing
            @test failure.diagnostics.derivative === nothing
        else
            @test failure.diagnostics.covariance === nothing
            @test failure.diagnostics.derivative === nothing
        end
        @test gram_is_population_bits(sampler) == before
        @test gram_is_pointer_roles_valid(sampler)
        @test sampler.rng.index == (case.name === :sampling ? 2 : 3)

        sampler.rng.batches[2] = original_second_batch
        target.fail_at = typemax(Int)
        gradient.fail_at = typemax(Int)
        sampler.method_state.covariance_rate[2] = original_covariance_rate
        sampler.method_state.repulsion_strength[2] = original_repulsion_strength
        recovered = importance_sample!(sampler)
        @test recovered isa WeightedSamples
        @test length(recovered) == 12
        @test gram_is_population_bits(sampler) != before
        @test gram_is_pointer_roles_valid(sampler)
        @test sampler.rng.index == (case.name === :sampling ? 4 : 5)
    end


    sampler = gram_is_proposal_phase_sampler()
    before = gram_is_population_bits(sampler)
    failure = caught_exception(() -> importance_sample!(sampler))
    @test failure isa FirstOrderGRAMISRoundError
    @test failure.round == 2
    @test failure.phase === :proposal
    @test failure.cause isa ImportanceSamplers._FirstOrderGRAMISProposalError
    @test failure.cause.proposal_slot == 1
    @test failure.cause.reason === :location_nonfinite
    @test isnan(failure.cause.value)
    @test failure.diagnostics.round_size == 6
    @test failure.diagnostics.completed_rounds == 1
    @test failure.diagnostics.covariance === nothing
    @test failure.diagnostics.derivative === nothing
    @test failure.diagnostics.transfers.count == 0
    @test failure.diagnostics.transfers.bytes == 0
    @test failure.diagnostics.pre_call_state_preserved === true
    @test gram_is_population_bits(sampler) == before
    @test gram_is_pointer_roles_valid(sampler)
    @test sampler.rng.index == 3

    recovered = importance_sample!(sampler)
    @test recovered isa WeightedSamples
    @test length(recovered) == 12
    @test gram_is_population_bits(sampler) != before
    @test gram_is_pointer_roles_valid(sampler)
    @test sampler.rng.index == 5
end
