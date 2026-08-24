using Test
using ImportanceSamplers
import Adapt
import DensityInterface
import KernelAbstractions
import MLDataDevices
import Random
import Random: rand, randn

const IS = ImportanceSamplers

struct TransferProposal{A<:AbstractVector}
    location::A
end

function rand(rng::Random.AbstractRNG, proposal::TransferProposal)
    return only(proposal.location) + randn(rng, eltype(proposal.location))
end

function DensityInterface.logdensityof(proposal::TransferProposal, sample::Real)
    offset = sample - only(proposal.location)
    half = oftype(offset, 0.5)
    return -half * abs2(offset) - half * oftype(offset, log(2pi))
end

struct TransferTarget{A<:AbstractVector}
    offset::A
end

function (target::TransferTarget)(sample, p)
    return p.shift[1] + target.offset[1] - abs2(sample) / 2
end

struct UncopyableRNG <: Random.AbstractRNG end
Base.copy(rng::UncopyableRNG) = rng

struct FunctionalAccelerator <: MLDataDevices.AbstractAcceleratorDevice end
MLDataDevices.functional(::FunctionalAccelerator) = true

struct NonfunctionalAccelerator <: MLDataDevices.AbstractAcceleratorDevice end
MLDataDevices.functional(::NonfunctionalAccelerator) = false

struct KernelArgumentTestBackend <: KernelAbstractions.GPU end

struct KernelArgumentTestArray{T,N} <: AbstractArray{T,N}
    storage::Array{T,N}
end

Base.size(array::KernelArgumentTestArray) = size(array.storage)
Base.getindex(array::KernelArgumentTestArray, indices...) =
    getindex(array.storage, indices...)
Base.IndexStyle(::Type{<:KernelArgumentTestArray}) = IndexLinear()
Base.similar(
    ::KernelArgumentTestArray,
    ::Type{T},
    dimensions::Dims{N},
) where {T,N} = KernelArgumentTestArray(Array{T,N}(undef, dimensions))

KernelAbstractions.get_backend(::KernelArgumentTestArray) =
    KernelArgumentTestBackend()

struct KernelArgumentTestDeviceArray{T,N}
    pointer::Ptr{T}
    dimensions::NTuple{N,Int}
end

struct KernelArgumentTestAdaptor end

Adapt.adapt_storage(
    ::KernelArgumentTestAdaptor,
    array::KernelArgumentTestArray{T,N},
) where {T,N} = KernelArgumentTestDeviceArray{T,N}(
    pointer(array.storage),
    size(array.storage),
)

KernelAbstractions.argconvert(
    ::KernelAbstractions.Kernel{KernelArgumentTestBackend},
    argument,
) = Adapt.adapt(KernelArgumentTestAdaptor(), argument)

struct KernelArgumentTestAccelerator <: MLDataDevices.AbstractAcceleratorDevice end
MLDataDevices.functional(::KernelArgumentTestAccelerator) = true
Adapt.adapt_storage(::KernelArgumentTestAccelerator, array::Array) =
    KernelArgumentTestArray(copy(array))

@eval ImportanceSamplers begin
    _owned_backend_rng(::Main.KernelArgumentTestAccelerator, seed::UInt64) =
        Random.Xoshiro(seed)
end

struct UnadaptedKernelContext{A}
    shift::A
end

struct AdaptedKernelContext{A}
    shift::A
end

Adapt.@adapt_structure AdaptedKernelContext

function kernel_context_target(sample, context)::Float64
    return context.shift[1] - abs2(sample) / 2
end

struct UnsupportedTestDevice <: MLDataDevices.AbstractDevice end

struct NestedClosureDensityTarget{F}
    logdensity::F
end

struct FunctionCollectionTarget{F<:AbstractVector}
    functions::F
end

function (target::FunctionCollectionTarget)(sample, p)
    return first(target.functions)(sample, p)
end

DensityInterface.DensityKind(::NestedClosureDensityTarget) =
    DensityInterface.IsDensity()
DensityInterface.logdensityof(target::NestedClosureDensityTarget, sample) =
    target.logdensity(sample)

top_level_transfer_target(sample, p) = p.shift[1] - abs2(sample) / 2
product_transfer_target(sample) = 0.0

function prepared_parts(prepared)
    target = getfield(prepared, :target)
    return (
        rng=getfield(prepared, :rng),
        callable=getfield(target, :target),
        context=getfield(target, :context),
        proposal=getfield(getfield(prepared, :algorithm), :proposal),
        algorithm=getfield(prepared, :algorithm),
        device=getfield(prepared, :device),
    )
end

function make_transfer_sampler(
    rng::Random.AbstractRNG;
    threaded=true,
    target=TransferTarget([0.25]),
)
    proposal = TransferProposal([0.5])
    context = (shift=[1.5], nested=(scale=[2.0, 3.0],))
    algorithm = ImportanceSampling(proposal; nsamples=4)
    return prepare_sampler(
        rng,
        target,
        context,
        algorithm;
        threaded=threaded,
    )
end

make_transfer_sampler(seed::Integer; kwargs...) =
    make_transfer_sampler(Random.Xoshiro(seed); kwargs...)

function caught_device_error(f)
    try
        f()
    catch error
        return error
    end
    return nothing
end

@testset "explicit prepared device transfer" begin
    cpu = MLDataDevices.cpu_device()
    cpu32 = MLDataDevices.cpu_device(Float32)
    source = make_transfer_sampler(2101; threaded=false)
    source_parts = prepared_parts(source)

    @test source_parts.device isa MLDataDevices.CPUDevice
    @test getfield(source, :executed) === false
    @test_throws MethodError prepare_sampler(
        Random.Xoshiro(2102),
        x -> -abs2(x),
        ImportanceSampling(TestScalarProposal(0.0); nsamples=1);
        device=cpu,
    )
    @test_throws MethodError importance_sample(
        Random.Xoshiro(2103),
        x -> -abs2(x),
        ImportanceSampling(TestScalarProposal(0.0); nsamples=1);
        device=cpu,
    )

    destination = @inferred cpu32(source)
    destination_parts = prepared_parts(destination)

    @test destination !== source
    @test destination_parts.device === cpu32
    @test destination_parts.algorithm.nsamples == source_parts.algorithm.nsamples
    @test getfield(destination, :threaded) === getfield(source, :threaded)
    @test getfield(destination, :executed) === false
    @test destination_parts.callable isa TransferTarget
    @test destination_parts.callable.offset == source_parts.callable.offset
    @test destination_parts.context == source_parts.context
    @test destination_parts.proposal.location == source_parts.proposal.location
    @test eltype(destination_parts.callable.offset) === Float32
    @test eltype(destination_parts.context.shift) === Float32
    @test eltype(destination_parts.context.nested.scale) === Float32
    @test eltype(destination_parts.proposal.location) === Float32

    @test destination_parts.rng !== source_parts.rng
    @test destination_parts.callable !== source_parts.callable
    @test destination_parts.callable.offset !== source_parts.callable.offset
    @test destination_parts.context.shift !== source_parts.context.shift
    @test destination_parts.context.nested.scale !== source_parts.context.nested.scale
    @test destination_parts.proposal !== source_parts.proposal
    @test destination_parts.proposal.location !== source_parts.proposal.location
    @test rand(copy(destination_parts.rng)) == rand(copy(source_parts.rng))

    destination_parts.callable.offset[1] = 9.0
    destination_parts.context.shift[1] = 8.0
    destination_parts.context.nested.scale[1] = 7.0
    destination_parts.proposal.location[1] = 6.0
    @test source_parts.callable.offset == [0.25]
    @test source_parts.context.shift == [1.5]
    @test source_parts.context.nested.scale == [2.0, 3.0]
    @test source_parts.proposal.location == [0.5]

    default_rng_source = make_transfer_sampler(
        Random.default_rng();
        threaded=false,
    )
    default_rng_destination = @inferred cpu(default_rng_source)
    @test getfield(default_rng_destination, :rng) isa Random.Xoshiro
    @test getfield(default_rng_destination, :rng) !==
          getfield(default_rng_source, :rng)
    expected_source_rng = copy(getfield(default_rng_source, :rng))
    rand(getfield(default_rng_destination, :rng))
    @test rand(getfield(default_rng_source, :rng)) == rand(expected_source_rng)

    uncopyable_source = make_transfer_sampler(UncopyableRNG(); threaded=false)
    uncopyable_error = caught_device_error(() -> cpu(uncopyable_source))
    @test uncopyable_error isa SamplerDeviceError
    @test uncopyable_error.reason === :rng_not_cloneable

    direct = cpu(make_transfer_sampler(2104; threaded=false))
    piped = make_transfer_sampler(2104; threaded=false) |> cpu
    @test importance_sample!(direct).samples == importance_sample!(piped).samples

    ordinary = let captured = [0.75]
        (sample, p) -> p.shift[1] + captured[1] - abs2(sample) / 2
    end
    ordinary_source = make_transfer_sampler(2105; threaded=false, target=ordinary)
    ordinary_destination = cpu(ordinary_source)
    @test prepared_parts(ordinary_destination).callable === ordinary

    source_result = importance_sample!(source)
    destination_result = importance_sample!(
        cpu(make_transfer_sampler(2101; threaded=false)),
    )
    @test source_result.samples == destination_result.samples
    @test source_result.logweights == destination_result.logweights
end

@testset "device transfer lifecycle and accelerator limits" begin
    default_cuda = MLDataDevices.CUDADevice()
    @test Base.eltype(default_cuda) === Missing

    scalar_policy_source = prepare_sampler(
        Random.Xoshiro(0x2200),
        product_transfer_target,
        ImportanceSampling(SphericalGaussian(0.0, 1.0); nsamples=4);
        threaded=true,
    )
    expected_scalar_policy_rng = copy(getfield(scalar_policy_source, :rng))
    scalar_policy_error = caught_device_error(
        () -> default_cuda(scalar_policy_source),
    )
    @test scalar_policy_error isa SamplerDeviceError
    @test scalar_policy_error.reason === :scalar_policy_unspecified
    @test rand(getfield(scalar_policy_source, :rng), UInt64) ==
          rand(expected_scalar_policy_rng, UInt64)
    @test sprint(showerror, scalar_policy_error) ==
          "prepared-sampler device transfer failed for " *
          "$(typeof(default_cuda)): the accelerator scalar policy is " *
          "unspecified; construct a preserving device with " *
          "gpu_device(nothing; force=true) or " *
          "gpu_device(device_id, nothing; force=true)"

    cpu = MLDataDevices.cpu_device()
    executed = make_transfer_sampler(2201; threaded=false)
    importance_sample!(executed)
    @test getfield(executed, :executed) === true
    @test_throws SamplerAlreadyExecutedError cpu(executed)

    failed = make_transfer_sampler(
        2205;
        threaded=false,
        target=(sample, p) -> sample > -Inf ? error("expected failure") : p.shift[1],
    )
    @test_throws SamplerExecutionError importance_sample!(failed)
    @test getfield(failed, :executed) === true
    @test getfield(failed, :running) === false
    @test_throws SamplerAlreadyExecutedError cpu(failed)

    unavailable = NonfunctionalAccelerator()
    unavailable_error = caught_device_error(
        () -> unavailable(make_transfer_sampler(2202; threaded=true)),
    )
    @test unavailable_error isa SamplerDeviceError
    @test unavailable_error.reason === :backend_unavailable

    serial_error = caught_device_error(
        () -> unavailable(make_transfer_sampler(2203; threaded=false)),
    )
    @test serial_error isa SamplerDeviceError
    @test serial_error.reason === :serial_accelerator

    captured_target = let captured = [1.0]
        (sample, p) -> p.shift[1] + captured[1] - abs2(sample) / 2
    end
    closure_error = caught_device_error(
        () -> unavailable(
            make_transfer_sampler(2204; threaded=true, target=captured_target),
        ),
    )
    @test closure_error isa SamplerDeviceError
    @test closure_error.reason === :opaque_host_closure

    nested_closure = let captured = [1.0]
        sample -> captured[1] - abs2(sample) / 2
    end
    nested_target = NestedClosureDensityTarget(nested_closure)
    nested_sampler = prepare_sampler(
        Random.Xoshiro(2206),
        nested_target,
        ImportanceSampling(TestScalarProposal(0.0); nsamples=1);
        threaded=true,
    )
    nested_error = caught_device_error(
        () -> FunctionalAccelerator()(nested_sampler),
    )
    @test nested_error isa SamplerDeviceError
    @test nested_error.reason === :opaque_host_closure

    captured_values = [1.0]
    collection_closure =
        (sample, p) -> p.shift[1] + captured_values[1] - abs2(sample) / 2
    collection_target = FunctionCollectionTarget(Function[collection_closure])
    @test (@inferred IS._has_opaque_host_closure(collection_target)) === true
    collection_sampler = make_transfer_sampler(
        2209;
        threaded=true,
        target=collection_target,
    )
    collection_error = caught_device_error(
        () -> FunctionalAccelerator()(collection_sampler),
    )
    @test collection_error isa SamplerDeviceError
    @test collection_error.reason === :opaque_host_closure
    cpu_collection_error = caught_device_error(
        () -> MLDataDevices.cpu_device(Float32)(collection_sampler),
    )
    @test cpu_collection_error isa SamplerDeviceError
    @test cpu_collection_error.reason === :opaque_host_closure
    @test sprint(showerror, cpu_collection_error) ==
          "prepared-sampler device transfer failed for " *
          "$(typeof(MLDataDevices.cpu_device(Float32))): " *
          "an opaque target closure reachable through target state cannot be " *
          "transferred without inspecting or reconstructing its captures; " *
          "pass numerical state through p"
    @test first(prepared_parts(collection_sampler).callable.functions) ===
          collection_closure
    @test captured_values isa Vector{Float64}

    rng_limit_sampler = prepare_sampler(
        Random.Xoshiro(2207),
        sample -> -abs2(sample) / 2,
        ImportanceSampling(SphericalGaussian(0.0, 1.0); nsamples=4);
        threaded=true,
    )
    rng_limit_error = caught_device_error(
        () -> FunctionalAccelerator()(rng_limit_sampler),
    )
    @test rng_limit_error isa SamplerDeviceError
    @test rng_limit_error.reason === :accelerator_rng_unavailable

    product = ProductProposal((
        left=SphericalGaussian(0.0, 1.0),
        right=SphericalGaussian(0.0, 1.0),
    ))
    product_sampler = prepare_sampler(
        Random.Xoshiro(2210),
        product_transfer_target,
        ImportanceSampling(product; nsamples=1);
        threaded=true,
    )
    product_error = caught_device_error(
        () -> FunctionalAccelerator()(product_sampler),
    )
    @test product_error isa SamplerDeviceError
    @test product_error.reason === :product_proposal_cpu_only
    @test sprint(showerror, product_error) ==
          "prepared-sampler device transfer failed for FunctionalAccelerator: " *
          "ProductProposal is CPU-only"

    unsupported_error = caught_device_error(
        () -> UnsupportedTestDevice()(make_transfer_sampler(2208)),
    )
    @test unsupported_error isa SamplerDeviceError
    @test unsupported_error.reason === :unsupported_device
end

@testset "accelerator capability rejection precedes RNG consumption" begin
    device = KernelArgumentTestAccelerator()
    proposal = SphericalGaussian(0.0, 1.0)
    algorithm = ImportanceSampling(proposal; nsamples=16)
    context_sampler(seed, context) = prepare_sampler(
        Random.Xoshiro(seed),
        kernel_context_target,
        context,
        algorithm;
        threaded=true,
    )

    unadapted = context_sampler(
        0x2211,
        UnadaptedKernelContext([0.25]),
    )
    expected_unadapted_rng = copy(getfield(unadapted, :rng))
    unadapted_error = caught_device_error(() -> device(unadapted))
    @test unadapted_error isa SamplerDeviceError
    @test unadapted_error.reason === :kernel_argument_unsupported
    @test rand(getfield(unadapted, :rng), UInt64) ==
          rand(expected_unadapted_rng, UInt64)

    for (seed, context) in (
        (0x2212, AdaptedKernelContext([0.25])),
        (0x2213, (shift=[0.25],)),
    )
        destination = device(context_sampler(seed, context))
        @test getfield(destination, :device) === device
    end

    generic = prepare_sampler(
        Random.Xoshiro(0x2214),
        top_level_transfer_target,
        (shift=[0.25],),
        ImportanceSampling(TransferProposal([0.0]); nsamples=16);
        threaded=true,
    )
    expected_generic_rng = copy(getfield(generic, :rng))
    generic_error = caught_device_error(() -> device(generic))
    @test generic_error isa SamplerDeviceError
    @test generic_error.reason === :generic_proposal_cpu_only
    @test rand(getfield(generic, :rng), UInt64) == rand(expected_generic_rng, UInt64)

    accelerator = device(
        context_sampler(0x2215, (shift=[0.25],)),
    )
    for destination in (
        MLDataDevices.cpu_device(),
        device,
        UnsupportedTestDevice(),
    )
        expected_accelerator_rng = copy(getfield(accelerator, :rng))
        migration_error = caught_device_error(() -> destination(accelerator))
        @test migration_error isa SamplerDeviceError
        @test migration_error.reason === :prepared_migration_unsupported
        @test rand(getfield(accelerator, :rng), UInt64) ==
              rand(expected_accelerator_rng, UInt64)
    end
end
