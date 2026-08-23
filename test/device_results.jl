using Test
using ImportanceSamplers
import MLDataDevices
import Statistics
struct ResultTestDevice <: MLDataDevices.AbstractAcceleratorDevice end
struct ResultBackendArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    storage::A
end
ResultBackendArray(storage::AbstractArray{T,N}) where {T,N} =
    ResultBackendArray{T,N,typeof(storage)}(storage)
const result_backend_reductions = Ref(0)
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
    return mapreduce(f, op, array.storage; kwargs...)
end
function Base.reduce(op, array::ResultBackendArray; kwargs...)
    result_backend_reductions[] += 1
    return reduce(op, array.storage; kwargs...)
end
Base.copy(array::ResultBackendArray) = ResultBackendArray(copy(array.storage))
Base.deepcopy_internal(array::ResultBackendArray, stackdict::IdDict) =
    ResultBackendArray(Base.deepcopy_internal(array.storage, stackdict))
Base.view(array::ResultBackendArray, indices...) = ResultBackendArray(
    view(array.storage, map(_backend_selector, indices)...),
)
Base.Array(array::ResultBackendArray) = Array(array.storage)
_backend_selector(selector::ResultBackendArray) = selector.storage
_backend_selector(selector) = selector
function Base.Broadcast.broadcasted(f, array::ResultBackendArray, args...)
    return ResultBackendArray(broadcast(f, array.storage, map(_backend_selector, args)...))
end
MLDataDevices.get_device(::ResultBackendArray) = ResultTestDevice()
MLDataDevices.get_device_type(::ResultBackendArray) = ResultTestDevice
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
    )
        error = _caught() do
            ImportanceSamplers._adopt_weighted_samples(args[1], args[2]; provenance=args[3])
        end
        @test error isa ArgumentError
        @test occursin("same device", sprint(showerror, error))
    end
    result_backend_reductions[] = 0
    trusted = ImportanceSamplers._adopt_validated_weighted_samples(
        ResultBackendArray([1.0, 2.0]),
        ResultBackendArray([-2.0, -1.0]);
        diagnostics=(transfers=(count=1, bytes=2sizeof(UInt64)),),
    )
    @test result_backend_reductions[] == 0
    @test (trusted.diagnostics.transfers.count, trusted.diagnostics.transfers.bytes) ==
          (1, 2sizeof(UInt64))
    @test ImportanceSamplers._kernel_result_transfers(trusted.logweights) ==
          (count=1, bytes=2sizeof(UInt64))
    @test ImportanceSamplers._kernel_result_transfers([-2.0, -1.0]) ==
          (count=0, bytes=0)
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
    weights = @inferred normalized_weights(result)
    @test weights isa ResultBackendArray
    @test Array(weights) ≈ exp.([-2.0, -1.0, 0.0]) ./ sum(exp.([-2.0, -1.0, 0.0]))
    @test (result.diagnostics.transfers.count, result.diagnostics.transfers.bytes) ==
          (2, sizeof(Float64) + sizeof(Bool))
    @test (@inferred lognormalizer(result)) ≈
          -1 + log(exp(-2.0) + exp(-1.0) + 1) - log(3.0)
    @test (result.diagnostics.transfers.count, result.diagnostics.transfers.bytes) ==
          (3, 2sizeof(Float64) + sizeof(Bool))
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
    destination = @inferred MLDataDevices.cpu_device()(source)
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
    @test cpu_destination.diagnostics.transfers !== cpu_source.diagnostics.transfers
end
