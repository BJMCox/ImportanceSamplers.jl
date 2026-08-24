using Test
using ImportanceSamplers
import MLDataDevices
import LogExpFunctions
import Statistics
struct ResultTestDevice <: MLDataDevices.AbstractAcceleratorDevice
    id::Int
end
mutable struct ResultMutableNumber <: Number
    value::Int
end
struct ResultBackendArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    storage::A
    device::ResultTestDevice
end
ResultBackendArray(storage::AbstractArray{T,N}, device=ResultTestDevice(1)) where {T,N} =
    ResultBackendArray{T,N,typeof(storage)}(storage, device)
const result_backend_reductions = Ref(0)
const result_backend_copies = Ref(0)
const result_backend_deepcopies = Ref(0)
const result_backend_materializations = Ref(0)
const result_backend_payload_bytes = Ref(0)
const forbid_result_backend_scalar_access = Ref(true)
Base.size(array::ResultBackendArray) = size(array.storage)
Base.IndexStyle(::Type{<:ResultBackendArray}) = IndexLinear()
function Base.getindex(array::ResultBackendArray{T}, indices::Int...)::T where {T}
    forbid_result_backend_scalar_access[] && error("forbidden host scalar indexing")
    return array.storage[indices...]
end
function Base.iterate(array::ResultBackendArray, state...)
    forbid_result_backend_scalar_access[] && error("forbidden host iteration")
    return iterate(array.storage, state...)
end
function Base.mapreduce(f, op, array::ResultBackendArray; kwargs...)
    result_backend_reductions[] += 1
    result = mapreduce(f, op, array.storage; kwargs...)
    result_backend_payload_bytes[] += sizeof(typeof(result))
    return result
end
function LogExpFunctions.logsumexp(array::ResultBackendArray)
    result_backend_reductions[] += 1
    result = LogExpFunctions.logsumexp(array.storage)
    result_backend_payload_bytes[] += sizeof(typeof(result))
    return result
end
function Base.copy(array::ResultBackendArray)
    result_backend_copies[] += 1
    return ResultBackendArray(copy(array.storage), array.device)
end
function Base.deepcopy_internal(array::ResultBackendArray, stackdict::IdDict)
    result_backend_deepcopies[] += 1
    return ResultBackendArray(Base.deepcopy_internal(array.storage, stackdict), array.device)
end
Base.view(array::ResultBackendArray, indices...) = ResultBackendArray(
    view(array.storage, map(_backend_selector, indices)...),
    array.device,
)
function Base.Array(array::ResultBackendArray)
    result_backend_materializations[] += 1
    return Array(array.storage)
end
_backend_selector(selector::ResultBackendArray) = selector.storage
_backend_selector(selector) = selector
function Base.Broadcast.broadcasted(f, array::ResultBackendArray, args...)
    return ResultBackendArray(
        broadcast(f, array.storage, map(_backend_selector, args)...),
        array.device,
    )
end
MLDataDevices.get_device(array::ResultBackendArray) = array.device
MLDataDevices.get_device_type(::ResultBackendArray) = ResultTestDevice
function (device::ResultTestDevice)(array::ResultBackendArray)
    return device == array.device ?
           array : ResultBackendArray(copy(array.storage), device)
end
_caught(f) = try
    f()
    nothing
catch error
    error
end

@testset "device result validation" begin
    samples = ResultBackendArray([1.0, 2.0, 3.0])
    logweights = ResultBackendArray([-3.0, -2.0, -1.0])
    result = @inferred ImportanceSamplers._adopt_weighted_samples(samples, logweights)
    @test result.samples === samples
    @test result.logweights === logweights
    @test (result.diagnostics.transfers.count, result.diagnostics.transfers.bytes) ==
          (1, sizeof(Bool))
    for args in (
        (samples, [-3.0, -2.0, -1.0], NamedTuple()),
        (samples, logweights, (proposal_id=[1, 2, 3],)),
        (
            samples,
            ResultBackendArray([-3.0, -2.0, -1.0], ResultTestDevice(2)),
            NamedTuple(),
        ),
        (
            (left=samples, right=ResultBackendArray([4.0, 5.0, 6.0], ResultTestDevice(2))),
            logweights,
            NamedTuple(),
        ),
        (
            samples,
            logweights,
            (proposal_id=ResultBackendArray([1, 2, 3], ResultTestDevice(2)),),
        ),
    )
        error = _caught() do
            ImportanceSamplers._adopt_weighted_samples(args[1], args[2]; provenance=args[3])
        end
        @test error isa ArgumentError
        @test occursin("same device", sprint(showerror, error))
    end
    result_backend_reductions[] = 0
    failure_record = ImportanceSamplers._DeviceFailureRecord(
        ResultBackendArray(zeros(UInt64, 2)),
    )
    result_backend_materializations[] = 0
    failure_payload_bytes =
        length(failure_record.storage) * sizeof(eltype(failure_record.storage))
    @test result_backend_materializations[] == 0
    failure_snapshot = ImportanceSamplers._device_failure_snapshot(failure_record)
    cpu_failure_snapshot = ImportanceSamplers._device_failure_snapshot(
        ImportanceSamplers._DeviceFailureRecord(zeros(UInt64, 2)),
    )
    @test result_backend_materializations[] == 1
    @test failure_snapshot.failure.count == 0
    @test failure_snapshot.transfers == (count=1, bytes=failure_payload_bytes)
    @test cpu_failure_snapshot.transfers == (count=0, bytes=0)
    trusted = ImportanceSamplers._adopt_validated_weighted_samples(
        ResultBackendArray([1.0, 2.0]),
        ResultBackendArray([-2.0, -1.0]);
        diagnostics=(transfers=failure_snapshot.transfers,),
    )
    @test result_backend_reductions[] == 0
    @test (trusted.diagnostics.transfers.count, trusted.diagnostics.transfers.bytes) ==
          (failure_snapshot.transfers.count, failure_snapshot.transfers.bytes)
end

@testset "device result access and views" begin
    result = ImportanceSamplers._adopt_weighted_samples(
        ResultBackendArray([10.0, 20.0, 30.0]),
        ResultBackendArray([-3.0, -2.0, -1.0]);
        provenance=(proposal_id=ResultBackendArray([1, 2, 3]),),
    )
    for operation in (
        () -> result[1],
        () -> iterate(result),
        () -> Statistics.quantile(result, 0.5),
        () -> Statistics.median(result),
    )
        error = _caught(operation)
        @test error isa ArgumentError
        @test occursin("transfer the result to CPU", sprint(showerror, error))
    end
    for (selector, expected) in (
        (2:3, [20.0, 30.0]),
        (ResultBackendArray([3, 1]), [30.0, 10.0]),
        (ResultBackendArray(Bool[true, false, true]), [10.0, 30.0]),
    )
        result_view = result[selector]
        @test result_view isa WeightedSampleView
        @test all(
            leaf -> leaf isa ResultBackendArray,
            (result_view.samples, result_view.logweights, result_view.provenance.proposal_id),
        )
        @test Array(result_view.samples) == expected
        @test_throws ArgumentError result_view[1]
    end
    @test normalized_weights(result[2:3]) isa ResultBackendArray
    @test result.diagnostics.transfers.count == 2
end

@testset "device result reductions" begin
    result = ImportanceSamplers._adopt_weighted_samples(
        ResultBackendArray([1.0, 2.0, 3.0]),
        ResultBackendArray([-3.0, -2.0, -1.0]),
    )
    result_backend_payload_bytes[] = 0
    before = (result.diagnostics.transfers.count, result.diagnostics.transfers.bytes)
    weights = @inferred normalized_weights(result)
    @test weights isa ResultBackendArray
    @test Array(weights) ≈ exp.([-2.0, -1.0, 0.0]) ./ sum(exp.([-2.0, -1.0, 0.0]))
    @test (result.diagnostics.transfers.count, result.diagnostics.transfers.bytes) .- before ==
          (1, result_backend_payload_bytes[])
    @test result_backend_payload_bytes[] == 2sizeof(Float64)
    result_backend_payload_bytes[] = 0
    before = (result.diagnostics.transfers.count, result.diagnostics.transfers.bytes)
    @test (@inferred lognormalizer(result)) ≈
          -1 + log(exp(-2.0) + exp(-1.0) + 1) - log(3.0)
    @test (result.diagnostics.transfers.count, result.diagnostics.transfers.bytes) .- before ==
          (1, result_backend_payload_bytes[])
    @test result_backend_payload_bytes[] == 2sizeof(Float64)
    float32 = ImportanceSamplers._adopt_weighted_samples(
        ResultBackendArray(Float32[1, 2]),
        ResultBackendArray(Float32[-2, -1]),
    )
    result_backend_payload_bytes[] = 0
    normalized_weights(float32)
    lognormalizer(float32)
    @test (float32.diagnostics.transfers.count, float32.diagnostics.transfers.bytes) ==
          (3, result_backend_payload_bytes[] + sizeof(Bool))
    @test result_backend_payload_bytes[] == 4sizeof(Float32)
    stable = ImportanceSamplers._adopt_weighted_samples(
        ResultBackendArray([1.0, 2.0]),
        ResultBackendArray([-1000.0, 0.0]),
    )
    @test lognormalizer(stable) ≈ LogExpFunctions.logsumexp([-1000.0, 0.0]) - log(2)
    all_zero = ImportanceSamplers._adopt_weighted_samples(
        ResultBackendArray([1.0, 2.0]),
        ResultBackendArray([-Inf, -Inf]),
    )
    @test_throws AllZeroWeightsError normalized_weights(all_zero)
    @test all_zero.diagnostics.transfers.count == 2
end

@testset "explicit result transfer" begin
    source = ImportanceSamplers._adopt_weighted_samples(
        ResultBackendArray([10.0, 20.0, 30.0]),
        ResultBackendArray([-3.0, -2.0, -1.0]);
        provenance=(proposal_id=ResultBackendArray([1, 2, 3]),),
        diagnostics=(method=:plain_is, trace=ResultBackendArray([3, 2, 1])),
    )
    result_backend_copies[] = 0
    result_backend_deepcopies[] = 0
    result_backend_materializations[] = 0
    destination = @inferred MLDataDevices.cpu_device()(source)
    @test result_backend_copies[] == 0
    @test result_backend_deepcopies[] == 0
    @test result_backend_materializations[] == 4
    for (actual, expected) in (
        (destination.samples, [10.0, 20.0, 30.0]),
        (destination.logweights, [-3.0, -2.0, -1.0]),
        (destination.provenance.proposal_id, [1, 2, 3]),
        (destination.diagnostics.trace, [3, 2, 1]),
    )
        @test actual == expected
        @test actual isa Vector
    end
    @test destination.diagnostics.method === :plain_is
    @test destination.diagnostics.transfers !== source.diagnostics.transfers
    @test (destination.diagnostics.transfers.count, destination.diagnostics.transfers.bytes) ==
          (1, sizeof(Bool))

    result_backend_copies[] = 0
    result_backend_deepcopies[] = 0
    result_backend_materializations[] = 0
    same_device = @inferred ResultTestDevice(1)(source)
    @test result_backend_copies[] == 4
    @test result_backend_deepcopies[] == 0
    @test result_backend_materializations[] == 0
    for (actual, original) in (
        (same_device.samples, source.samples),
        (same_device.logweights, source.logweights),
        (same_device.provenance.proposal_id, source.provenance.proposal_id),
        (same_device.diagnostics.trace, source.diagnostics.trace),
    )
        @test actual !== original
    end
    @test same_device.diagnostics.transfers !== source.diagnostics.transfers
    referent = ResultMutableNumber(1)
    shallow_source = ImportanceSamplers._adopt_weighted_samples(
        ResultBackendArray([referent]),
        ResultBackendArray([-1.0]),
    )
    shallow_destination = ResultTestDevice(1)(shallow_source)
    @test shallow_destination.samples !== shallow_source.samples
    @test shallow_destination.samples.storage[1] === referent

    same_eltype_source = WeightedSamples(
        Float32[1, 2],
        Float32[-2, -1];
        diagnostics=(trace=Float32[2, 1],),
    )
    same_eltype_destination = @inferred MLDataDevices.cpu_device(Float32)(same_eltype_source)
    same_eltype_destination.samples[1] = same_eltype_destination.logweights[1] = 99
    same_eltype_destination.diagnostics.trace[1] = 99
    @test same_eltype_source.samples == Float32[1, 2]
    @test same_eltype_source.logweights == Float32[-2, -1]
    @test same_eltype_source.diagnostics.trace == Float32[2, 1]

    destination.samples[1] = destination.logweights[1] = -10.0
    destination.provenance.proposal_id[1] = destination.diagnostics.trace[1] = -1
    @test Array(source.samples) == [10.0, 20.0, 30.0]
    @test Array(source.logweights) == [-3.0, -2.0, -1.0]
    @test Array(source.provenance.proposal_id) == [1, 2, 3]
    @test Array(source.diagnostics.trace) == [3, 2, 1]
    lognormalizer(source)
    @test (source.diagnostics.transfers.count, destination.diagnostics.transfers.count) ==
          (2, 1)

    cpu_source = WeightedSamples([1.0, 2.0], [-2.0, -1.0]; diagnostics=(trace=[2.0, 1.0],))
    cpu_destination = @inferred MLDataDevices.cpu_device(Float32)(cpu_source)
    @test cpu_destination.samples == Float32[1, 2]
    @test cpu_destination.logweights == Float32[-2, -1]
    @test cpu_destination.diagnostics.trace == Float32[2, 1]
    @test cpu_destination.samples !== cpu_source.samples
    @test cpu_destination.logweights !== cpu_source.logweights
    @test cpu_destination.diagnostics.trace !== cpu_source.diagnostics.trace
    @test cpu_destination.diagnostics.transfers !== cpu_source.diagnostics.transfers
end

@testset "explicit view transfer" begin
    source = ImportanceSamplers._adopt_weighted_samples(
        ResultBackendArray([10.0, 20.0, 30.0]),
        ResultBackendArray([-3.0, -2.0, -1.0]);
        provenance=(proposal_id=ResultBackendArray([1, 2, 3]),),
    )
    source_view = source[ResultBackendArray([3, 1])]
    destination = @inferred MLDataDevices.cpu_device()(source_view)

    @test destination isa WeightedSampleView
    @test destination.samples == [30.0, 10.0]
    @test destination.logweights == [-1.0, -3.0]
    @test destination.provenance.proposal_id == [3, 1]
    @test collect(destination) == [
        (sample=30.0, logweight=-1.0, provenance=(proposal_id=3,)),
        (sample=10.0, logweight=-3.0, provenance=(proposal_id=1,)),
    ]
    @test destination.transfers !== source_view.transfers
    @test (destination.transfers.count, destination.transfers.bytes) ==
          (source_view.transfers.count, source_view.transfers.bytes)
    @test destination.samples !== source_view.samples
    @test destination.logweights !== source_view.logweights
    @test destination.provenance.proposal_id !==
          source_view.provenance.proposal_id

    same_device = ResultTestDevice(1)(source_view)
    @test same_device.samples !== source_view.samples
    @test same_device.logweights !== source_view.logweights
    @test same_device.provenance.proposal_id !==
          source_view.provenance.proposal_id
    @test same_device.transfers !== source_view.transfers

    source.samples.storage[3] = 300.0
    source.logweights.storage[3] = -10.0
    source.provenance.proposal_id.storage[3] = 30
    @test destination.samples == [30.0, 10.0]
    @test destination.logweights == [-1.0, -3.0]
    @test destination.provenance.proposal_id == [3, 1]
    forbid_result_backend_scalar_access[] = false
    @test same_device.samples == [30.0, 10.0]
    @test same_device.logweights == [-1.0, -3.0]
    @test same_device.provenance.proposal_id == [3, 1]
    forbid_result_backend_scalar_access[] = true

    cpu_source = WeightedSamples(
        [10.0, 20.0, 30.0],
        [-3.0, -2.0, -1.0];
        provenance=(proposal_id=[1, 2, 3],),
    )
    cpu_view = cpu_source[2:3]
    cpu_destination = MLDataDevices.cpu_device()(cpu_view)
    @test cpu_destination.samples !== cpu_view.samples
    @test cpu_destination.logweights !== cpu_view.logweights
    @test cpu_destination.provenance.proposal_id !==
          cpu_view.provenance.proposal_id
    @test cpu_destination.transfers !== cpu_view.transfers
    cpu_destination.samples[1] = 200.0
    cpu_destination.logweights[1] = -20.0
    cpu_destination.provenance.proposal_id[1] = 20
    @test collect(cpu_view) == [
        (sample=20.0, logweight=-2.0, provenance=(proposal_id=2,)),
        (sample=30.0, logweight=-1.0, provenance=(proposal_id=3,)),
    ]
end
