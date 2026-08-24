using Test
using ImportanceSamplers
import DensityInterface
import MLDataDevices
import Random
import Random: rand

struct TestOffsetArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::A
    offsets::NTuple{N,Int}
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
