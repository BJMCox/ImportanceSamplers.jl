using Test
using ImportanceSamplers
import DensityInterface
import Random
import Random: rand, randn

mutable struct ThreadRecordingRNG{R<:Random.AbstractRNG} <: Random.AbstractRNG
    inner::R
    access_tasks::Vector{Task}
end

mutable struct ThreadRecordingProposal
    draw_index::Int
    draw_tasks::Vector{Task}
    density_tasks::Vector{Task}
    density_draw_counts::Vector{Int}
    target_complete::Vector{Bool}
    target_complete_seen::Vector{Bool}
end

function ThreadRecordingProposal(nsamples::Int)
    return ThreadRecordingProposal(
        0,
        Task[],
        Vector{Task}(undef, nsamples),
        Vector{Int}(undef, nsamples),
        fill(false, nsamples),
        Vector{Bool}(undef, nsamples),
    )
end

function rand(rng::ThreadRecordingRNG, proposal::ThreadRecordingProposal)
    push!(rng.access_tasks, current_task())
    proposal.draw_index += 1
    push!(proposal.draw_tasks, current_task())
    return proposal.draw_index + rand(rng.inner) / 4
end

function DensityInterface.logdensityof(
    proposal::ThreadRecordingProposal,
    sample::Float64,
)
    sample_index = floor(Int, sample)
    proposal.density_tasks[sample_index] = current_task()
    proposal.density_draw_counts[sample_index] = proposal.draw_index
    proposal.target_complete_seen[sample_index] = all(proposal.target_complete)
    return -sample
end

mutable struct NativeThreadRecordingRNG{R<:Random.AbstractRNG} <: Random.AbstractRNG
    inner::R
    access_tasks::Vector{Task}
end

function randn(rng::NativeThreadRecordingRNG, ::Type{Float64})
    push!(rng.access_tasks, current_task())
    return randn(rng.inner, Float64)
end

mutable struct NativeThreadTarget
    lock::ReentrantLock
    tasks::Vector{Task}
end

function (target::NativeThreadTarget)(sample::Float64)::Float64
    lock(target.lock)
    try
        push!(target.tasks, current_task())
    finally
        unlock(target.lock)
    end
    return -abs2(sample) / 2
end

@testset "CPU threading ($(Threads.nthreads(:default)) default threads)" begin
    nsamples = 64
    initial_state = Random.Xoshiro(1401)
    serial_rng = ThreadRecordingRNG(copy(initial_state), Task[])
    threaded_rng = ThreadRecordingRNG(copy(initial_state), Task[])
    serial_proposal = ThreadRecordingProposal(nsamples)
    threaded_proposal = ThreadRecordingProposal(nsamples)

    serial_target_tasks = Vector{Task}(undef, nsamples)
    serial_target_draw_counts = Vector{Int}(undef, nsamples)
    serial_target = function (sample)
        sample_index = floor(Int, sample)
        serial_target_tasks[sample_index] = current_task()
        serial_target_draw_counts[sample_index] = serial_proposal.draw_index
        return -abs2(sample) / 2
    end
    threaded_target_tasks = Vector{Task}(undef, nsamples)
    threaded_target_draw_counts = Vector{Int}(undef, nsamples)
    threaded_target = function (sample)
        sample_index = floor(Int, sample)
        threaded_target_tasks[sample_index] = current_task()
        threaded_target_draw_counts[sample_index] = threaded_proposal.draw_index
        threaded_proposal.target_complete[sample_index] = true
        return -abs2(sample) / 2
    end

    caller_task = current_task()
    serial_result = importance_sample!(
        prepare_sampler(
            serial_rng,
            serial_target,
            ImportanceSampling(serial_proposal; nsamples=nsamples);
            threaded=false,
        ),
    )
    threaded_sampler = prepare_sampler(
        threaded_rng,
        threaded_target,
        ImportanceSampling(threaded_proposal; nsamples=nsamples);
        threaded=true,
    )
    threaded_result = @inferred importance_sample!(threaded_sampler)

    @test serial_result.samples == threaded_result.samples
    @test serial_result.logweights == threaded_result.logweights
    @test normalized_weights(serial_result) == normalized_weights(threaded_result)
    @test lognormalizer(serial_result) ≈ lognormalizer(threaded_result) rtol = 8eps()
    @test sum(normalized_weights(serial_result)) ≈
          sum(normalized_weights(threaded_result)) rtol = 8eps()

    @test all(==(caller_task), serial_rng.access_tasks)
    @test all(==(caller_task), threaded_rng.access_tasks)
    @test all(==(caller_task), serial_proposal.draw_tasks)
    @test all(==(caller_task), threaded_proposal.draw_tasks)
    @test serial_target_draw_counts == fill(nsamples, nsamples)
    @test threaded_target_draw_counts == fill(nsamples, nsamples)
    @test serial_proposal.density_draw_counts == fill(nsamples, nsamples)
    @test threaded_proposal.density_draw_counts == fill(nsamples, nsamples)

    @test all(threaded_proposal.target_complete_seen)
    @test serial_result.diagnostics.execution === :serial
    expected_execution = Threads.nthreads(:default) > 1 ? :threaded : :serial
    @test threaded_result.diagnostics.execution === expected_execution
    @test threaded_result.diagnostics.threaded === true

    if Threads.nthreads(:default) > 1
        @test all(!=(caller_task), threaded_target_tasks)
        @test all(!=(caller_task), threaded_proposal.density_tasks)
        @test all(task -> Threads.threadpool(task) === :default, threaded_target_tasks)
        @test all(
            task -> Threads.threadpool(task) === :default,
            threaded_proposal.density_tasks,
        )
        @test length(unique(threaded_target_tasks)) > 1
        @test length(unique(threaded_proposal.density_tasks)) > 1
    else
        @test all(==(caller_task), threaded_target_tasks)
        @test all(==(caller_task), threaded_proposal.density_tasks)
    end
end

@testset "native CPU buffer and execution equivalence" begin
    nsamples = 2_048
    initial_state = Random.Xoshiro(0x8108)
    serial_rng = NativeThreadRecordingRNG(copy(initial_state), Task[])
    threaded_rng = NativeThreadRecordingRNG(copy(initial_state), Task[])
    serial_target = NativeThreadTarget(ReentrantLock(), Task[])
    threaded_target = NativeThreadTarget(ReentrantLock(), Task[])
    proposal = SphericalGaussian(0.5, 1.25)
    algorithm = ImportanceSampling(proposal; nsamples=nsamples)
    caller_task = current_task()

    serial_result = @inferred importance_sample!(
        prepare_sampler(serial_rng, serial_target, algorithm; threaded=false),
    )
    threaded_result = @inferred importance_sample!(
        prepare_sampler(threaded_rng, threaded_target, algorithm; threaded=true),
    )

    expected_rng = copy(initial_state)
    expected_samples = Vector{Float64}(undef, nsamples)
    for index in eachindex(expected_samples)
        expected_samples[index] = 0.5 + 1.25 * randn(expected_rng, Float64)
    end

    @test serial_result.samples == expected_samples
    @test threaded_result.samples == expected_samples
    @test serial_result.logweights == threaded_result.logweights
    expected_next = rand(expected_rng)
    @test rand(serial_rng.inner) == expected_next
    @test rand(threaded_rng.inner) == expected_next
    @test all(==(caller_task), serial_rng.access_tasks)
    @test all(==(caller_task), threaded_rng.access_tasks)
    @test all(==(caller_task), serial_target.tasks)
    @test serial_result.diagnostics.execution === :serial
    expected_execution = Threads.nthreads(:default) > 1 ? :threaded : :serial
    @test threaded_result.diagnostics.execution === expected_execution

    if Threads.nthreads(:default) > 1
        @test all(!=(caller_task), threaded_target.tasks)
        @test all(task -> Threads.threadpool(task) === :default, threaded_target.tasks)
        @test length(unique(threaded_target.tasks)) > 1
    else
        @test all(==(caller_task), threaded_target.tasks)
    end
end
