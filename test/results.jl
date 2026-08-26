using Test
using ImportanceSamplers

struct TestMutableBackedNumber <: Number
    payload::Vector{Int}
end

@testset "validated result adoption" begin
    samples = [1.0, 2.0]
    logweights = [-2.0, -1.0]
    provenance = (proposal_id=[1, 2],)
    diagnostics = (method=:importance_sampling, trace=[1, 2])

    adopted = ImportanceSamplers._adopt_weighted_samples(
        samples,
        logweights;
        provenance=provenance,
        diagnostics=diagnostics,
    )
    @test adopted.samples === samples
    @test adopted.logweights === logweights
    @test adopted.provenance === provenance
    @test adopted.diagnostics.method === diagnostics.method
    @test adopted.diagnostics.trace === diagnostics.trace
    @test adopted.diagnostics.transfers.count == 0
    @test adopted.diagnostics.transfers.bytes == 0

    public_result = WeightedSamples(
        samples,
        logweights;
        provenance=provenance,
        diagnostics=diagnostics,
    )
    @test public_result.samples !== samples
    @test public_result.logweights !== logweights
    @test public_result.provenance !== provenance
    @test public_result.diagnostics !== diagnostics
    @test public_result.diagnostics.trace !== diagnostics.trace
    @test public_result.diagnostics.transfers !== adopted.diagnostics.transfers

    reported = ImportanceSamplers._ResultTransferCounter(0, 0)
    ImportanceSamplers._record_reported_transfer!(
        reported,
        1,
        3sizeof(UInt64),
        Val(:failure_snapshot),
    )
    reported_copy = ImportanceSamplers._transfer_result_storage(nothing, reported)
    @test reported_copy !== reported
    @test reported_copy.count == reported.count == 1
    @test reported_copy.bytes == reported.bytes == 3sizeof(UInt64)
    @test reported_copy.reasons !== reported.reasons
    @test reported_copy.reasons.failure_snapshot !==
          reported.reasons.failure_snapshot
    @test reported_copy.reasons.failure_snapshot.count == 1
    @test reported_copy.reasons.failure_snapshot.bytes == 3sizeof(UInt64)

    @test_throws ArgumentError ImportanceSamplers._adopt_weighted_samples(
        [1.0],
        [Inf],
    )
end

@testset "complete weighted results" begin
    source_samples = [10.0, 20.0, 30.0]
    source_logweights = [-2.0, -1.0, 0.0]
    diagnostics = (method=:plain_is, evaluations=3)

    result = WeightedSamples(
        source_samples,
        source_logweights;
        diagnostics=diagnostics,
    )

    @test result.samples == [10.0, 20.0, 30.0]
    @test result.logweights == [-2.0, -1.0, 0.0]
    @test result.provenance == NamedTuple()
    @test result.diagnostics.method === diagnostics.method
    @test result.diagnostics.evaluations == diagnostics.evaluations
    @test result.diagnostics.transfers.count == 0
    @test result.diagnostics.transfers.bytes == 0
    @test result.samples !== source_samples
    @test result.logweights !== source_logweights

    source_samples[1] = -100.0
    source_logweights[1] = -100.0
    @test result.samples == [10.0, 20.0, 30.0]
    @test result.logweights == [-2.0, -1.0, 0.0]

    @test length(result) == 3
    @test result[2] == (
        sample=20.0,
        logweight=-1.0,
        provenance=NamedTuple(),
    )
    @test collect(result) == [
        (sample=10.0, logweight=-2.0, provenance=NamedTuple()),
        (sample=20.0, logweight=-1.0, provenance=NamedTuple()),
        (sample=30.0, logweight=0.0, provenance=NamedTuple()),
    ]

    vector_samples = [1.0 2.0 3.0; 10.0 20.0 30.0]
    provenance = (proposal_id=[11, 12, 13], round=[1, 1, 2])
    vector_result = WeightedSamples(
        vector_samples,
        [-3.0, -2.0, -1.0];
        provenance=provenance,
        diagnostics=(method=:multiple_is,),
    )
    indexed = vector_result[2]
    @test indexed.sample == [2.0, 20.0]
    @test indexed.sample isa SubArray
    @test indexed.logweight == -2.0
    @test indexed.provenance == (proposal_id=12, round=1)
    @test vector_result.provenance !== provenance
    @test vector_result.provenance.proposal_id !== provenance.proposal_id

    named_samples = (
        location=[1.0, 2.0, 3.0],
        state=(position=[1.0 2.0 3.0; 4.0 5.0 6.0], scale=[0.1, 0.2, 0.3]),
    )
    named_provenance = (
        origin=(proposal_id=[11, 12, 13], round=[1, 1, 2]),
    )
    diagnostic_trace = [3, 2, 1]
    named_result = WeightedSamples(
        named_samples,
        [-2.0, -1.0, 0.0];
        provenance=named_provenance,
        diagnostics=(method=:plain_is, trace=diagnostic_trace),
    )
    @test named_result[3].sample.location == 3.0
    @test named_result[3].sample.state.position == [3.0, 6.0]
    @test named_result[3].sample.state.position isa SubArray
    @test named_result[3].sample.state.scale == 0.3
    @test named_result[3].provenance == (origin=(proposal_id=13, round=2),)

    named_samples.location[1] = -10.0
    named_samples.state.position[1, 1] = -20.0
    named_samples.state.scale[1] = -30.0
    named_provenance.origin.proposal_id[1] = -40
    named_provenance.origin.round[1] = -50
    diagnostic_trace[1] = -60
    @test named_result[1].sample.location == 1.0
    @test named_result[1].sample.state.position == [1.0, 4.0]
    @test named_result[1].sample.state.scale == 0.1
    @test named_result[1].provenance == (origin=(proposal_id=11, round=1),)
    @test named_result.diagnostics.trace == [3, 2, 1]

    mutable_backed_samples = [
        TestMutableBackedNumber([1]),
        TestMutableBackedNumber([2]),
    ]
    mutable_backed_diagnostic = TestMutableBackedNumber([3])
    mutable_backed_result = WeightedSamples(
        mutable_backed_samples,
        [-2.0, -1.0];
        diagnostics=(counter=mutable_backed_diagnostic,),
    )
    mutable_backed_samples[1].payload[1] = -1
    mutable_backed_diagnostic.payload[1] = -3
    @test mutable_backed_result.samples[1].payload == [1]
    @test mutable_backed_result.diagnostics.counter.payload == [3]
end

@testset "weighted sample views" begin
    samples = [1.0 2.0 3.0 4.0; 10.0 20.0 30.0 40.0]
    result = WeightedSamples(
        samples,
        [-4.0, -3.0, -2.0, -1.0];
        provenance=(proposal_id=[10, 20, 30, 40],),
    )

    range_view = result[2:4]
    @test range_view isa WeightedSampleView
    @test length(range_view) == 3
    @test range_view.samples isa SubArray
    @test range_view.logweights isa SubArray
    @test range_view.provenance.proposal_id isa SubArray
    @test range_view[1].sample == [2.0, 20.0]
    @test range_view[1].logweight == -3.0
    @test range_view[1].provenance == (proposal_id=20,)

    index_selector = [4, 1, 3]
    index_view = result[index_selector]
    @test index_view isa WeightedSampleView
    @test [copy(record.sample) for record in index_view] == [
        [4.0, 40.0],
        [1.0, 10.0],
        [3.0, 30.0],
    ]
    @test [record.provenance.proposal_id for record in index_view] == [40, 10, 30]

    mask_selector = Bool[true, false, true, false]
    mask_view = result[mask_selector]
    @test mask_view isa WeightedSampleView
    @test [copy(record.sample) for record in mask_view] == [
        [1.0, 10.0],
        [3.0, 30.0],
    ]
    @test mask_view.logweights == [-4.0, -2.0]

    index_selector .= [2, 3, 1]
    mask_selector .= Bool[false, true, false, true]
    @test [copy(record.sample) for record in index_view] == [
        [4.0, 40.0],
        [1.0, 10.0],
        [3.0, 30.0],
    ]
    @test [copy(record.sample) for record in mask_view] == [
        [1.0, 10.0],
        [3.0, 30.0],
    ]

    nested_view = index_view[2:3]
    @test nested_view isa WeightedSampleView
    @test [copy(record.sample) for record in nested_view] == [
        [1.0, 10.0],
        [3.0, 30.0],
    ]

    result.samples[1, 2] = 200.0
    result.logweights[2] = -30.0
    result.provenance.proposal_id[2] = 200
    @test range_view[1].sample == [200.0, 20.0]
    @test range_view[1].logweight == -30.0
    @test range_view[1].provenance == (proposal_id=200,)

    recursive_result = WeightedSamples(
        (
            location=[1.0, 2.0, 3.0],
            state=(position=[1.0 2.0 3.0; 4.0 5.0 6.0], scale=[0.1, 0.2, 0.3]),
        ),
        [-3.0, -2.0, -1.0];
        provenance=(
            origin=(proposal_id=[10, 20, 30], coordinate=[1 2 3; 4 5 6]),
        ),
    )
    recursive_view = recursive_result[[3, 1]]
    @test recursive_view.samples.location isa SubArray
    @test recursive_view.samples.state.position isa SubArray
    @test recursive_view.provenance.origin.proposal_id isa SubArray
    @test recursive_view[1].sample.location == 3.0
    @test recursive_view[1].sample.state.position == [3.0, 6.0]
    @test recursive_view[1].provenance == (
        origin=(proposal_id=30, coordinate=[3, 6]),
    )
    recursive_result.samples.state.position[1, 3] = 300.0
    recursive_result.provenance.origin.coordinate[1, 3] = 300
    @test recursive_view[1].sample.state.position == [300.0, 6.0]
    @test recursive_view[1].provenance.origin.coordinate == [300, 6]

    empty_index_view = result[Int[]]
    empty_range_view = result[2:1]
    @test empty_index_view isa WeightedSampleView
    @test empty_range_view isa WeightedSampleView
    @test isempty(empty_index_view)
    @test isempty(empty_range_view)
    @test collect(empty_index_view) == []
    @test collect(empty_range_view) == []
    @test_throws ArgumentError lognormalizer(range_view)
end

@testset "stable weight reductions" begin
    extreme_logs = [1000.0, 999.0, -Inf]
    result = WeightedSamples([1.0, 2.0, 3.0], extreme_logs)

    expected_sum_log = 1000.0 + log1p(exp(-1.0))
    @test lognormalizer(result) ≈ expected_sum_log - log(3.0)
    @test normalized_weights(result) ≈ [
        1 / (1 + exp(-1.0)),
        1 / (1 + exp(1.0)),
        0.0,
    ]
    @test sum(normalized_weights(result)) ≈ 1.0

    focused = result[1:2]
    @test normalized_weights(focused) ≈ [
        1 / (1 + exp(-1.0)),
        1 / (1 + exp(1.0)),
    ]

    float32_result = WeightedSamples(Float32[1, 2], Float32[100, 99])
    @test lognormalizer(float32_result) isa Float32
    @test eltype(normalized_weights(float32_result)) === Float32
end

@testset "typed result iteration" begin
    scalar_result = WeightedSamples([1.0, 2.0], [-2.0, -1.0])
    scalar_record = @inferred scalar_result[1]
    scalar_iteration = @inferred iterate(scalar_result)
    @test first(scalar_iteration) == scalar_record
    @test eltype(scalar_result) === typeof(scalar_record)
    @test eltype(collect(scalar_result)) === typeof(scalar_record)
    @test eltype(collect(scalar_result)) !== Any

    matrix_result = WeightedSamples(
        [1.0 2.0; 3.0 4.0],
        [-2.0, -1.0];
        provenance=(proposal_id=[10, 20],),
    )
    matrix_record = @inferred matrix_result[1]
    matrix_iteration = @inferred iterate(matrix_result)
    @test first(matrix_iteration) == matrix_record
    @test eltype(matrix_result) === typeof(matrix_record)
    @test eltype(collect(matrix_result)) === typeof(matrix_record)
    @test eltype(collect(matrix_result)) !== Any

    recursive_result = WeightedSamples(
        (
            location=[1.0, 2.0],
            state=(position=[1.0 2.0; 3.0 4.0], scale=[0.1, 0.2]),
        ),
        [-2.0, -1.0];
        provenance=(origin=(proposal_id=[10, 20],),),
    )
    recursive_record = @inferred recursive_result[1]
    recursive_iteration = @inferred iterate(recursive_result)
    @test first(recursive_iteration) == recursive_record
    @test eltype(recursive_result) === typeof(recursive_record)
    @test eltype(collect(recursive_result)) === typeof(recursive_record)
    @test eltype(collect(recursive_result)) !== Any

    empty_index_view = recursive_result[Int[]]
    nonempty_index_view = recursive_result[[1]]
    empty_range_view = recursive_result[2:1]
    nonempty_range_view = recursive_result[1:1]
    @test eltype(empty_index_view) === eltype(nonempty_index_view)
    @test eltype(empty_range_view) === eltype(nonempty_range_view)
    @test eltype(collect(empty_index_view)) === eltype(empty_index_view)
    @test eltype(collect(empty_range_view)) === eltype(empty_range_view)
    @test eltype(empty_index_view) !== Any
    @test eltype(empty_range_view) !== Any
end
