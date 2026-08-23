function _draw_batch(rng::Random.AbstractRNG, algorithm::ImportanceSampling)
    return _draw_batch(rng, algorithm.proposal, algorithm.nsamples)
end

function _draw_batch(rng::Random.AbstractRNG, proposal, nsamples::Int)
    nsamples > 0 || throw(ArgumentError("batch sample count must be positive"))

    first_sample = rand(rng, proposal)
    batch = _allocate_batch(first_sample, nsamples)
    _store_sample!(batch, first_sample, 1)

    for sample_index in 2:nsamples
        sample = rand(rng, proposal)
        _store_sample!(batch, sample, sample_index)
    end

    return batch
end

_allocate_batch(sample::T, nsamples) where {T<:Number} = Vector{T}(undef, nsamples)

function _allocate_batch(sample::AbstractVector{T}, nsamples) where {T<:Number}
    isconcretetype(T) || throw(
        ArgumentError("dense vector sample element type $T must be concrete"),
    )
    return Matrix{T}(undef, length(sample), nsamples)
end

function _allocate_batch(sample::NamedTuple, nsamples)
    isempty(sample) && throw(ArgumentError("named-tuple samples must contain numeric leaves"))
    leaf_batches = map(leaf -> _allocate_batch(leaf, nsamples), values(sample))
    return NamedTuple{keys(sample)}(leaf_batches)
end

function _allocate_batch(sample, nsamples)
    throw(ArgumentError("unsupported sample structure $(typeof(sample))"))
end

function _store_sample!(batch::Vector{T}, sample, sample_index) where {T}
    typeof(sample) === T || _throw_leaf_mismatch(T, typeof(sample), sample_index)
    batch[sample_index] = sample
    return batch
end

function _store_sample!(batch::Matrix{T}, sample, sample_index) where {T}
    sample isa AbstractVector || _throw_structure_mismatch(sample_index)
    eltype(sample) === T || _throw_leaf_mismatch(T, eltype(sample), sample_index)
    length(sample) == size(batch, 1) || throw(
        ArgumentError("sample $sample_index has inconsistent vector length"),
    )
    copyto!(view(batch, :, sample_index), sample)
    return batch
end

function _store_sample!(batch::NamedTuple, sample, sample_index)
    sample isa NamedTuple || _throw_structure_mismatch(sample_index)
    keys(sample) == keys(batch) || _throw_structure_mismatch(sample_index)
    for field_name in keys(batch)
        _store_sample!(
            getproperty(batch, field_name),
            getproperty(sample, field_name),
            sample_index,
        )
    end
    return batch
end

function _throw_leaf_mismatch(expected, actual, sample_index)
    throw(
        ArgumentError(
            "sample $sample_index has leaf type $actual; expected leaf type $expected",
        ),
    )
end

function _throw_structure_mismatch(sample_index)
    throw(ArgumentError("sample $sample_index has inconsistent structure"))
end

_storage_device(storage::AbstractArray) = MLDataDevices.get_device(storage)

function _combine_storage_devices(devices::Tuple)
    device = first(devices)
    all(other -> typeof(other) === typeof(device), devices) || throw(
        ArgumentError("numeric storage leaves must use the same device"),
    )
    return device
end

_is_host_storage(storage::AbstractArray) =
    _storage_device(storage) isa MLDataDevices.AbstractCPUDevice

_sample_count(batch::AbstractVector) = length(batch)
_sample_count(batch::AbstractMatrix) = size(batch, 2)

function _sample_count(batch::NamedTuple)
    isempty(batch) && throw(ArgumentError("sample storage has no leaves"))
    counts = map(_sample_count, values(batch))
    all(==(first(counts)), counts) || throw(ArgumentError("sample storage is not aligned"))
    return first(counts)
end

@inline _sample_at(batch::AbstractVector, sample_index) = batch[sample_index]
@inline _sample_at(batch::AbstractMatrix, sample_index) = view(batch, :, sample_index)

function _sample_at(batch::NamedTuple, sample_index)
    sample_leaves = map(leaf -> _sample_at(leaf, sample_index), values(batch))
    return NamedTuple{keys(batch)}(sample_leaves)
end
