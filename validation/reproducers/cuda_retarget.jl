using CUDA
using ImportanceSamplers
using MLDataDevices
using Random
using Test

include(joinpath(@__DIR__, "..", "cuda_plain_is_support.jl"))

const RetargetIS = ImportanceSamplers
const RETARGET_CUDA_SEED = 0x7265746172676574

struct CUDARetargetTarget end
struct CUDARetargetGradient end

@inline function (::CUDARetargetTarget)(sample, context)
    T = eltype(sample)
    value = context.offset[1]
    @inbounds for coordinate in eachindex(sample)
        value -= T(0.5) * abs2(sample[coordinate] - context.location[coordinate])
    end
    return value
end

@inline function (::CUDARetargetGradient)(destination, sample, context)
    @inbounds for coordinate in eachindex(destination)
        destination[coordinate] = context.location[coordinate] - sample[coordinate]
    end
    return destination
end

cuda_retarget_context(::Type{T}, location, offset) where {T} = (
    location=T[location...],
    offset=T[offset],
)

cuda_retarget_bank(::Type{T}) where {T} = ProposalBank([
    FactorGaussian(T[-1, 0], T[1 0; 0.1 0.8]),
    FactorGaussian(T[1, 0], T[0.9 0; -0.2 1.1]),
])

cuda_retarget_value(value::Number) = value
cuda_retarget_value(value::AbstractArray) = copy(value)

cuda_retarget_snapshot(proposal) = (
    location=cuda_retarget_value(proposal.location),
    scale=cuda_retarget_value(getfield(proposal.scale, 1)),
    lognormalizer=proposal.lognormalizer,
)

cuda_retarget_snapshot(bank::ProposalBank) = (
    proposals=map(cuda_retarget_snapshot, bank.proposals),
    masses=copy(bank.masses),
)

function assert_cuda_retarget_residence(sampler, result, physical_device)
    @test RetargetIS._backend_state_resident(sampler.device, sampler.target)
    @test RetargetIS._backend_state_resident(sampler.device, sampler.method_state)
    @test RetargetIS._backend_state_resident(sampler.device, sampler.random_buffers)
    storage = (result.samples, result.logweights, values(result.provenance)...)
    @test all(array -> array isa CUDA.AnyCuArray, storage)
    @test all(array -> CUDA.device(array) == physical_device, storage)
    return nothing
end

function cuda_retarget_case(device, caller, target, algorithm, seed)
    T = Float32
    source = prepare_sampler(
        Random.Xoshiro(seed),
        target,
        cuda_retarget_context(T, (-0.25, 0.25), 0.0),
        algorithm;
        threaded=true,
    ) |> device
    importance_sample!(source)
    CUDA.synchronize()
    @test CUDA.device() == caller

    committed = cuda_retarget_snapshot(
        current_proposal(MLDataDevices.cpu_device(), source),
    )
    root_rng = Random.Xoshiro(seed + 1)
    rng_oracle = copy(root_rng)
    rand(rng_oracle, UInt64)
    sampler = @inferred retarget(
        root_rng,
        source,
        target,
        cuda_retarget_context(T, (0.75, -0.5), 1.0),
    )
    @test rand(root_rng, UInt64) == rand(rng_oracle, UInt64)
    @test CUDA.device() == caller
    @test sampler.device == device
    @test cuda_retarget_snapshot(
        current_proposal(MLDataDevices.cpu_device(), sampler),
    ) == committed

    result = @inferred importance_sample!(sampler)
    CUDA.synchronize()
    @test CUDA.device() == caller
    assert_cuda_retarget_residence(sampler, result, device.device)
    @test cuda_retarget_snapshot(
        current_proposal(MLDataDevices.cpu_device(), source),
    ) == committed
    return nothing
end

function main()
    physical_devices = collect(CUDA.devices())
    caller = CUDA.device()
    requested_index = findfirst(!=(caller), physical_devices)
    device = isnothing(requested_index) ? cuda_device() : cuda_device(requested_index)
    target = LogTarget(CUDARetargetTarget(); grad=CUDARetargetGradient())
    T = Float32

    cuda_retarget_case(
        device,
        caller,
        target,
        DeterministicMixturePMC(
            cuda_retarget_bank(T);
            rounds=1,
            round_size=8,
        ),
        RETARGET_CUDA_SEED,
    )
    cuda_retarget_case(
        device,
        caller,
        target,
        AMIS(
            FactorGaussian(T[0, 0], T[1 0; 0.2 0.9]);
            rounds=1,
            round_size=8,
        ),
        RETARGET_CUDA_SEED + 0x10,
    )
    cuda_retarget_case(
        device,
        caller,
        target,
        FirstOrderGRAMIS(
            cuda_retarget_bank(T);
            rounds=1,
            round_size=8,
            repulsion_strength=T(0.1),
        ),
        RETARGET_CUDA_SEED + 0x20,
    )
    @test CUDA.device() == caller
    return :passed
end

main()
