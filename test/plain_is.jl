import DensityInterface
import LogDensityProblems
import MLDataDevices
import Random
import Random: rand

const IS = ImportanceSamplers

struct PriorityTarget end
(::PriorityTarget)(sample) = 10.0 + sum(sample)
DensityInterface.DensityKind(::PriorityTarget) = DensityInterface.HasDensity()
DensityInterface.logdensityof(::PriorityTarget, sample) = 20.0 + sum(sample)
LogDensityProblems.capabilities(::Type{PriorityTarget}) =
    LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.dimension(::PriorityTarget) = 2
LogDensityProblems.logdensity(::PriorityTarget, sample) = 30.0 + sum(sample)

struct DensityTarget end
DensityInterface.DensityKind(::DensityTarget) = DensityInterface.IsDensity()
DensityInterface.logdensityof(::DensityTarget, sample) = 40.0 + sum(sample)

struct TargetExecutionFailure <: Exception end

struct VectorOnlyTarget end
(::VectorOnlyTarget)(sample::Vector{Float64}) = -sum(abs2, sample)

struct MatrixOnlyTarget end
(::MatrixOnlyTarget)(samples::Matrix{Float64}) = -sum(abs2, samples)

struct ThrowingCallableDensity end
(::ThrowingCallableDensity)(sample) = throw(TargetExecutionFailure())
DensityInterface.DensityKind(::ThrowingCallableDensity) = DensityInterface.HasDensity()
DensityInterface.logdensityof(::ThrowingCallableDensity, sample) = 50.0 + sum(sample)

struct WrongDimensionTarget end
LogDensityProblems.capabilities(::Type{WrongDimensionTarget}) =
    LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.dimension(::WrongDimensionTarget) = 3
LogDensityProblems.logdensity(::WrongDimensionTarget, sample) = -sum(abs2, sample)

struct CountingDensityTarget
    ldp_selections::Base.RefValue{Int}
    density_selections::Base.RefValue{Int}
end

function LogDensityProblems.capabilities(target::CountingDensityTarget)
    target.ldp_selections[] += 1
    return nothing
end

function DensityInterface.DensityKind(target::CountingDensityTarget)
    target.density_selections[] += 1
    return DensityInterface.IsDensity()
end

DensityInterface.logdensityof(::CountingDensityTarget, sample) = -abs2(sample)

struct AlternatingLeafTypeProposal
    draw_count::Base.RefValue{Int}
end

function rand(rng::Random.AbstractRNG, proposal::AlternatingLeafTypeProposal)
    proposal.draw_count[] += 1
    value = rand(rng)
    return isone(proposal.draw_count[]) ? value : Float32(value)
end

struct AlternatingStructureProposal
    draw_count::Base.RefValue{Int}
end

function rand(rng::Random.AbstractRNG, proposal::AlternatingStructureProposal)
    proposal.draw_count[] += 1
    value = rand(rng)
    return isone(proposal.draw_count[]) ? (value=value,) : (value=[value],)
end

IS._proposal_dimension(proposal::TestVectorProposal) = length(proposal.location)

@testset "target binding" begin
    proposal = TestVectorProposal(zeros(2))
    sample_batch = reshape([0.25, -0.5], 2, 1)
    sample = IS._sample_at(sample_batch, 1)

    context_free = sample -> -sum(abs2, sample)
    bound_context_free = IS._bind_prepared_target(
        IS._prepare_target(context_free, proposal), sample_batch
    )
    @test bound_context_free(sample) == -0.3125

    contextual = (sample, context) -> context.shift - sum(abs2, sample)
    context = (shift=2.0,)
    bound_contextual = IS._bind_prepared_target(
        IS._prepare_target(contextual, context, proposal), sample_batch
    )
    @test bound_contextual(sample) == 1.6875

    @test_throws SamplerExecutionError IS._bind_prepared_target(
        IS._prepare_target(context_free, context, proposal), sample_batch
    )

    @test_throws SamplerExecutionError IS._bind_prepared_target(
        IS._prepare_target(VectorOnlyTarget(), proposal), sample_batch
    )
    @test_throws SamplerExecutionError IS._bind_prepared_target(
        IS._prepare_target(MatrixOnlyTarget(), proposal), sample_batch
    )

    priority_target = PriorityTarget()
    wrapped = IS._bind_prepared_target(
        IS._prepare_target(LogTarget(priority_target), proposal), sample_batch
    )
    @test wrapped(sample) == 9.75

    bound_ldp = IS._bind_prepared_target(
        IS._prepare_target(priority_target, proposal), sample_batch
    )
    @test bound_ldp(sample) == 29.75

    bound_density = IS._bind_prepared_target(
        IS._prepare_target(DensityTarget(), proposal), sample_batch
    )
    @test bound_density(sample) == 39.75

    callable_density = IS._bind_prepared_target(
        IS._prepare_target(ThrowingCallableDensity(), proposal), sample_batch
    )
    @test callable_density(sample) == 49.75

    throwing = IS._bind_prepared_target(
        IS._prepare_target(LogTarget(ThrowingCallableDensity()), proposal), sample_batch
    )
    @test_throws TargetExecutionFailure throwing(sample)

    @test_throws DimensionMismatch IS._bind_prepared_target(
        IS._prepare_target(WrongDimensionTarget(), proposal), sample_batch
    )
end

function _draw_test_batch(rng, proposal, nsamples)
    sampler = prepare_sampler(
        rng,
        identity,
        ImportanceSampling(proposal; nsamples=nsamples);
        threaded=false,
    )
    return IS._draw_prepared_batch(sampler)
end

@testset "generic batch storage" begin
    scalar_proposal = TestScalarProposal(1.0)
    scalar_batch = _draw_test_batch(Random.Xoshiro(11), scalar_proposal, 5)
    @test scalar_batch isa Vector{Float64}
    @test scalar_proposal.draw_count[] == 5
    @test IS._sample_count(scalar_batch) == 5
    @test [IS._sample_at(scalar_batch, i) for i in 1:5] == scalar_proposal.history

    vector_proposal = TestVectorProposal(Float32[1, -1])
    vector_batch = _draw_test_batch(Random.Xoshiro(22), vector_proposal, 4)
    @test vector_batch isa Matrix{Float32}
    @test size(vector_batch) == (2, 4)
    @test vector_proposal.draw_count[] == 4
    @test IS._sample_count(vector_batch) == 4
    @test [copy(IS._sample_at(vector_batch, i)) for i in 1:4] == vector_proposal.history

    abstract_elements = TestAbstractElementVectorProposal()
    @test_throws SamplerExecutionError _draw_test_batch(
        Random.Xoshiro(23), abstract_elements, 2
    )
    @test abstract_elements.draw_count[] == 1

    named_proposal = TestNamedProposal(Float64)
    named_batch = _draw_test_batch(Random.Xoshiro(33), named_proposal, 3)
    @test named_batch isa NamedTuple
    @test named_batch.location isa Vector{Float64}
    @test named_batch.state.position isa Matrix{Float64}
    @test named_batch.state.scale isa Vector{Float64}
    @test named_proposal.draw_count[] == 3
    @test IS._sample_count(named_batch) == 3
    retained_named = [IS._sample_at(named_batch, i) for i in 1:3]
    @test [sample.location for sample in retained_named] ==
          [sample.location for sample in named_proposal.history]
    @test [copy(sample.state.position) for sample in retained_named] ==
          [sample.state.position for sample in named_proposal.history]
    @test [sample.state.scale for sample in retained_named] ==
          [sample.state.scale for sample in named_proposal.history]

    leaf_mismatch = AlternatingLeafTypeProposal(Ref(0))
    @test_throws SamplerExecutionError _draw_test_batch(
        Random.Xoshiro(44), leaf_mismatch, 2
    )
    @test leaf_mismatch.draw_count[] == 2

    structure_mismatch = AlternatingStructureProposal(Ref(0))
    @test_throws SamplerExecutionError _draw_test_batch(
        Random.Xoshiro(55), structure_mismatch, 2
    )
    @test structure_mismatch.draw_count[] == 2

    length_mismatch = TestChangingVectorLengthProposal()
    @test_throws SamplerExecutionError _draw_test_batch(
        Random.Xoshiro(66), length_mismatch, 2
    )
    @test length_mismatch.draw_count[] == 2

    empty_named = TestEmptyNamedProposal()
    @test_throws SamplerExecutionError _draw_test_batch(
        Random.Xoshiro(77), empty_named, 1
    )
    @test empty_named.draw_count[] == 1

    nested_empty_named = TestNestedEmptyNamedProposal()
    @test_throws SamplerExecutionError _draw_test_batch(
        Random.Xoshiro(88), nested_empty_named, 1
    )
    @test nested_empty_named.draw_count[] == 1

    reused_buffer = TestReusedVectorBufferProposal(zeros(2))
    reused_batch = _draw_test_batch(Random.Xoshiro(99), reused_buffer, 3)
    @test reused_buffer.draw_count[] == 3
    @test [copy(IS._sample_at(reused_batch, i)) for i in 1:3] ==
          reused_buffer.history
    first_retained = copy(IS._sample_at(reused_batch, 1))
    fill!(reused_buffer.buffer, 0.0)
    @test IS._sample_at(reused_batch, 1) == first_retained
end

mutable struct PreparedSequenceProposal{T<:AbstractFloat,L<:AbstractFloat}
    values::Vector{T}
    logdensities::Vector{L}
    draw_index::Int
    draw_tasks::Vector{Task}
    density_tasks::Vector{Task}
    phase_trace::Vector{Tuple{Symbol,T}}
end

function PreparedSequenceProposal(values::Vector{T}, logdensities::Vector{L}) where {T,L}
    length(values) == length(logdensities) || throw(ArgumentError("fixture mismatch"))
    return PreparedSequenceProposal(
        values,
        logdensities,
        0,
        Task[],
        Task[],
        Tuple{Symbol,T}[],
    )
end

function rand(::Random.AbstractRNG, proposal::PreparedSequenceProposal)
    proposal.draw_index += 1
    push!(proposal.draw_tasks, current_task())
    return proposal.values[proposal.draw_index]
end

function DensityInterface.logdensityof(
    proposal::PreparedSequenceProposal, sample::AbstractFloat
)
    push!(proposal.density_tasks, current_task())
    push!(proposal.phase_trace, (:proposal, sample))
    sample_index = findfirst(isequal(sample), proposal.values)
    sample_index === nothing && error("sample is absent from the fixture")
    return proposal.logdensities[sample_index]
end

struct UnsupportedFactorExecution <: ImportanceSamplers._AbstractFactorExecution end

@testset "prepared serial importance sampling" begin
    identity_proposal = TestScalarProposal(0.0)
    identity_algorithm = ImportanceSampling(identity_proposal; nsamples=8)
    identity_rng = Random.Xoshiro(101)
    identity_sampler = prepare_sampler(
        identity_rng,
        sample -> DensityInterface.logdensityof(identity_proposal, sample),
        identity_algorithm;
        threaded=false,
    )

    @test getfield(identity_sampler, :rng) === identity_rng
    @test getfield(identity_sampler, :device) isa MLDataDevices.CPUDevice
    @test getfield(identity_sampler, :threaded) === false
    @test nameof(typeof(identity_sampler)) ∉ names(ImportanceSamplers)

    identity_result = importance_sample!(identity_sampler)
    @test length(identity_result) == 8
    @test identity_result.logweights == zeros(Float64, 8)
    @test identity_result.logweights isa Vector{Float64}
    @test identity_result.diagnostics.method === :importance_sampling
    @test identity_result.diagnostics.execution === :serial
    @test identity_result.diagnostics.threaded === false
    @test identity_result.diagnostics.factor_execution_policy === :fused
    @test identity_result.diagnostics.nsamples == 8
    @test identity_result.diagnostics.failures == 0
    @test identity_result.diagnostics.transfers.count == 0
    @test identity_result.diagnostics.transfers.bytes == 0

    batched_sampler = @inferred prepare_sampler(
        Random.Xoshiro(101),
        sample -> DensityInterface.logdensityof(identity_proposal, sample),
        identity_algorithm;
        factor_execution=BatchedFactorExecution(),
        threaded=false,
    )
    @test getfield(batched_sampler, :factor_execution) isa BatchedFactorExecution
    @test importance_sample!(batched_sampler).diagnostics.factor_execution_policy ===
          :batched
    @test_throws ArgumentError prepare_sampler(
        Random.Xoshiro(101),
        sample -> DensityInterface.logdensityof(identity_proposal, sample),
        identity_algorithm;
        factor_execution=UnsupportedFactorExecution(),
        threaded=false,
    )
    @test_throws ArgumentError prepare_sampler(
        Random.Xoshiro(101),
        sample -> DensityInterface.logdensityof(identity_proposal, sample),
        identity_algorithm;
        factor_execution=:batched,
        threaded=false,
    )

    @testset "preparation resolves known targets once" begin
        mismatched_proposal = TestVectorProposal(zeros(2))
        @test_throws DimensionMismatch prepare_sampler(
            Random.Xoshiro(102),
            WrongDimensionTarget(),
            ImportanceSampling(mismatched_proposal; nsamples=2);
            threaded=false,
        )
        @test mismatched_proposal.draw_count[] == 0

        counted_target = CountingDensityTarget(Ref(0), Ref(0))
        counted_proposal = TestScalarProposal(0.0)
        counted_sampler = prepare_sampler(
            Random.Xoshiro(103),
            counted_target,
            ImportanceSampling(counted_proposal; nsamples=2);
            threaded=false,
        )
        @test counted_target.ldp_selections[] == 1
        @test counted_target.density_selections[] == 1
        importance_sample!(counted_sampler)
        importance_sample!(counted_sampler)
        @test counted_target.ldp_selections[] == 1
        @test counted_target.density_selections[] == 1

        priority_proposal = TestVectorProposal(zeros(2))
        priority_result = importance_sample(
            Random.Xoshiro(104),
            LogTarget(PriorityTarget()),
            ImportanceSampling(priority_proposal; nsamples=2);
            threaded=false,
        )
        expected_priority_logs = [
            10.0 + sum(sample) -
            DensityInterface.logdensityof(priority_proposal, sample)
            for sample in eachcol(priority_result.samples)
        ]
        @test priority_result.logweights == expected_priority_logs

        contextual_proposal = TestScalarProposal(0.0)
        contextual_result = importance_sample(
            Random.Xoshiro(105),
            (sample, p) -> p.shift - abs2(sample),
            (shift=2.5,),
            ImportanceSampling(contextual_proposal; nsamples=2);
            threaded=false,
        )
        @test contextual_result.logweights == [
            2.5 - abs2(sample) -
            DensityInterface.logdensityof(contextual_proposal, sample)
            for sample in contextual_result.samples
        ]
    end

    base_proposal = TestScalarProposal(0.5)
    shifted_proposal = TestScalarProposal(0.5)
    base_sampler = prepare_sampler(
        Random.Xoshiro(202),
        sample -> DensityInterface.logdensityof(base_proposal, sample),
        ImportanceSampling(base_proposal; nsamples=7);
        threaded=false,
    )
    shift = 3.25
    shifted_sampler = prepare_sampler(
        Random.Xoshiro(202),
        (sample, context) ->
            DensityInterface.logdensityof(shifted_proposal, sample) + context.shift,
        (shift=shift,),
        ImportanceSampling(shifted_proposal; nsamples=7);
        threaded=false,
    )
    base_result = importance_sample!(base_sampler)
    shifted_result = importance_sample!(shifted_sampler)
    @test base_result.samples == shifted_result.samples
    @test normalized_weights(base_result) ≈ normalized_weights(shifted_result) rtol = 4eps()
    @test lognormalizer(shifted_result) == lognormalizer(base_result) + shift

    mixed_proposal = PreparedSequenceProposal([1.0, 2.0, 3.0], zeros(3))
    mixed_sampler = prepare_sampler(
        Random.Xoshiro(303),
        sample -> sample == 2.0 ? -Inf : sample,
        ImportanceSampling(mixed_proposal; nsamples=3);
        threaded=false,
    )
    mixed_result = importance_sample!(mixed_sampler)
    @test mixed_result.samples == [1.0, 2.0, 3.0]
    @test mixed_result.logweights == [1.0, -Inf, 3.0]

    raw_proposal = PreparedSequenceProposal(
        [1.0, 2.0, 3.0],
        [0.25, -0.5, 1.5],
    )
    coordinator_task = current_task()
    target_tasks = Task[]
    raw_target = function (sample)
        push!(target_tasks, current_task())
        return 2sample
    end
    raw_sampler = prepare_sampler(
        Random.Xoshiro(404),
        raw_target,
        ImportanceSampling(raw_proposal; nsamples=3);
        threaded=false,
    )
    raw_result = importance_sample!(raw_sampler)
    @test raw_result.logweights == [1.75, 4.5, 4.5]
    @test all(==(coordinator_task), raw_proposal.draw_tasks)
    @test all(==(coordinator_task), target_tasks)
    @test all(==(coordinator_task), raw_proposal.density_tasks)

    float32_proposal = PreparedSequenceProposal(Float32[1, 2], zeros(Float32, 2))
    float32_sampler = prepare_sampler(
        Random.Xoshiro(505),
        identity,
        ImportanceSampling(float32_proposal; nsamples=2);
        threaded=false,
    )
    float32_result = importance_sample!(float32_sampler)
    @test float32_result.logweights == Float32[1, 2]
    @test float32_result.logweights isa Vector{Float32}

    @testset "typed serial hot path inference" begin
        inferred_float32_target = IS._bind_prepared_target(
            IS._prepare_target(identity, float32_proposal), Float32[1, 2]
        )
        inferred_float32_weights = @inferred IS._evaluate_logweights(
            Float32,
            inferred_float32_target,
            float32_proposal,
            Float32[1, 2],
        )
        @test inferred_float32_weights == Float32[1, 2]

        inference_float64_proposal = PreparedSequenceProposal(
            [1.0, 2.0],
            zeros(2),
        )
        inferred_float64_target = IS._bind_prepared_target(
            IS._prepare_target(identity, inference_float64_proposal), [1.0, 2.0]
        )
        inferred_float64_weights = @inferred IS._evaluate_logweights(
            Float64,
            inferred_float64_target,
            inference_float64_proposal,
            [1.0, 2.0],
        )
        @test inferred_float64_weights == [1.0, 2.0]
    end

    @testset "log type resolution is order independent" begin
        target_type_cases = (
            (Union{Float32,Float64}[1.0f0, 2.0], [1.0, 2.0]),
            (Union{Float32,Float64}[2.0, 1.0f0], [2.0, 1.0]),
        )
        for (target_logs, expected_weights) in target_type_cases
            proposal = PreparedSequenceProposal([1.0, 2.0], zeros(Float32, 2))
            target = sample -> target_logs[Int(sample)]
            sampler = prepare_sampler(
                Random.Xoshiro(506),
                target,
                ImportanceSampling(proposal; nsamples=2);
                threaded=false,
            )
            result = importance_sample!(sampler)
            @test result.logweights == expected_weights
            @test result.logweights isa Vector{Float64}
        end

        proposal_type_cases = (
            (Union{Float32,Float64}[1.0f0, 2.0], [-1.0, -2.0]),
            (Union{Float32,Float64}[2.0, 1.0f0], [-2.0, -1.0]),
        )
        for (proposal_logs, expected_weights) in proposal_type_cases
            proposal = PreparedSequenceProposal([1.0, 2.0], proposal_logs)
            sampler = prepare_sampler(
                Random.Xoshiro(507),
                _ -> 0.0f0,
                ImportanceSampling(proposal; nsamples=2);
                threaded=false,
            )
            result = importance_sample!(sampler)
            @test result.logweights == expected_weights
            @test result.logweights isa Vector{Float64}
        end
    end

    @testset "serial phase order" begin
        phase_proposal = PreparedSequenceProposal([1.0, 2.0, 3.0], zeros(3))
        phase_target = function (sample)
            push!(phase_proposal.phase_trace, (:target, sample))
            return sample
        end
        phase_sampler = prepare_sampler(
            Random.Xoshiro(508),
            phase_target,
            ImportanceSampling(phase_proposal; nsamples=3);
            threaded=false,
        )
        phase_result = importance_sample!(phase_sampler)
        @test phase_result.logweights == [1.0, 2.0, 3.0]
        @test phase_proposal.phase_trace == [
            (:target, 1.0),
            (:target, 2.0),
            (:target, 3.0),
            (:proposal, 1.0),
            (:proposal, 2.0),
            (:proposal, 3.0),
        ]
    end

    infinite_proposal = PreparedSequenceProposal([1.0], [Inf])
    infinite_sampler = prepare_sampler(
        Random.Xoshiro(606),
        _ -> 0.0,
        ImportanceSampling(infinite_proposal; nsamples=1);
        threaded=false,
    )
    infinite_result = importance_sample!(infinite_sampler)
    @test infinite_result.logweights == [-Inf]
    @test lognormalizer(infinite_result) == -Inf

    stream_proposal = TestScalarProposal(-0.25)
    stream_algorithm = ImportanceSampling(stream_proposal; nsamples=4)
    supplied_rng = Random.Xoshiro(707)
    expected_rng = Random.Xoshiro(707)
    expected_proposal = TestScalarProposal(-0.25)
    expected_first = [rand(expected_rng, expected_proposal) for _ in 1:4]
    expected_second = [rand(expected_rng, expected_proposal) for _ in 1:4]
    stream_sampler = prepare_sampler(
        supplied_rng,
        sample -> DensityInterface.logdensityof(stream_proposal, sample),
        stream_algorithm;
        threaded=false,
    )
    first_result = importance_sample!(stream_sampler)
    saved_first_samples = copy(first_result.samples)
    saved_first_logweights = copy(first_result.logweights)
    second_result = importance_sample!(stream_sampler)

    @test first_result.samples == expected_first
    @test second_result.samples == expected_second
    @test first_result.samples == saved_first_samples
    @test first_result.logweights == saved_first_logweights
    @test first_result.samples !== second_result.samples
    @test first_result.logweights !== second_result.logweights
    @test rand(supplied_rng) == rand(expected_rng)
end

@testset "one-shot importance sampling" begin
    context_free_state = Random.Xoshiro(1301)
    one_shot_proposal = TestScalarProposal(0.25)
    prepared_proposal = TestScalarProposal(0.25)
    one_shot_algorithm = ImportanceSampling(one_shot_proposal; nsamples=9)
    prepared_algorithm = ImportanceSampling(prepared_proposal; nsamples=9)
    one_shot_result = importance_sample(
        copy(context_free_state),
        sample -> DensityInterface.logdensityof(one_shot_proposal, sample),
        one_shot_algorithm;
        threaded=false,
    )
    prepared_sampler = prepare_sampler(
        copy(context_free_state),
        sample -> DensityInterface.logdensityof(prepared_proposal, sample),
        prepared_algorithm;
        threaded=false,
    )
    prepared_result = importance_sample!(prepared_sampler)

    @test one_shot_result.samples == prepared_result.samples
    @test one_shot_result.logweights == prepared_result.logweights
    @test normalized_weights(one_shot_result) == normalized_weights(prepared_result)
    @test lognormalizer(one_shot_result) == lognormalizer(prepared_result)

    contextual_state = Random.Xoshiro(1302)
    contextual_one_shot_proposal = TestScalarProposal(-0.5)
    contextual_prepared_proposal = TestScalarProposal(-0.5)
    context = (shift=2.75,)
    contextual_one_shot = importance_sample(
        copy(contextual_state),
        (sample, p) ->
            DensityInterface.logdensityof(contextual_one_shot_proposal, sample) +
            p.shift,
        context,
        ImportanceSampling(contextual_one_shot_proposal; nsamples=7);
        threaded=false,
    )
    contextual_sampler = prepare_sampler(
        copy(contextual_state),
        (sample, p) ->
            DensityInterface.logdensityof(contextual_prepared_proposal, sample) +
            p.shift,
        context,
        ImportanceSampling(contextual_prepared_proposal; nsamples=7);
        threaded=false,
    )
    contextual_prepared = importance_sample!(contextual_sampler)

    @test contextual_one_shot.samples == contextual_prepared.samples
    @test contextual_one_shot.logweights == contextual_prepared.logweights
    @test normalized_weights(contextual_one_shot) ==
          normalized_weights(contextual_prepared)
    @test lognormalizer(contextual_one_shot) == lognormalizer(contextual_prepared)

    inference_proposal = TestScalarProposal(0.0)
    inference_target = sample -> DensityInterface.logdensityof(inference_proposal, sample)
    inferred_algorithm = @inferred ImportanceSampling(inference_proposal; nsamples=4)
    inferred_sampler = @inferred prepare_sampler(
        Random.Xoshiro(1303),
        inference_target,
        inferred_algorithm;
        threaded=false,
    )
    inferred_result = @inferred importance_sample!(inferred_sampler)
    @test (@inferred inferred_result[1]) == first(inferred_result)
    @test (@inferred normalized_weights(inferred_result)) == fill(0.25, 4)
    @test (@inferred lognormalizer(inferred_result)) == 0.0

    inferred_one_shot_proposal = TestScalarProposal(0.0)
    inferred_one_shot_target =
        sample -> DensityInterface.logdensityof(inferred_one_shot_proposal, sample)
    inferred_one_shot = @inferred importance_sample(
        Random.Xoshiro(1304),
        inferred_one_shot_target,
        ImportanceSampling(inferred_one_shot_proposal; nsamples=4);
        threaded=false,
    )
    @test inferred_one_shot.logweights == zeros(4)
end

@testset "normalized positive transformed proposal through plain IS" begin
    proposal = @inferred TransformedProposal(
        SphericalGaussian(0.0, 1.0),
        PositiveTransform(),
    )
    lognormal_target(x)::Float64 =
        -0.5 * abs2(log(x)) - log(x) - 0.5 * log(2pi)
    sampler = @inferred prepare_sampler(
        Random.Xoshiro(0x7102),
        lognormal_target,
        ImportanceSampling(proposal; nsamples=64);
        threaded=false,
    )
    result = @inferred importance_sample!(sampler)
    @test maximum(abs, result.logweights) <= 4eps()
    @test abs(lognormalizer(result)) <= 4eps()
end
