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

struct AdaptableFunction{A<:AbstractVector} <: Function
    offset::A
end

Adapt.@adapt_structure AdaptableFunction

function (target::AdaptableFunction)(sample, p)
    return p.shift[1] + target.offset[1] - abs2(sample) / 2
end

struct UncopyableRNG <: Random.AbstractRNG end
Base.copy(rng::UncopyableRNG) = rng

struct FunctionalAccelerator <: MLDataDevices.AbstractAcceleratorDevice end
MLDataDevices.functional(::FunctionalAccelerator) = true

struct NonfunctionalAccelerator <: MLDataDevices.AbstractAcceleratorDevice end
MLDataDevices.functional(::NonfunctionalAccelerator) = false

struct BroadFunctionCPUDevice <: MLDataDevices.AbstractCPUDevice end
MLDataDevices.functional(::BroadFunctionCPUDevice) = true
Adapt.adapt_storage(::BroadFunctionCPUDevice, array::Array) = copy(array)
Adapt.adapt_structure(::BroadFunctionCPUDevice, target::F) where {F<:Function} =
    target

struct BroadFunctionAccelerator <: MLDataDevices.AbstractAcceleratorDevice end
Base.eltype(::BroadFunctionAccelerator) = Nothing
MLDataDevices.functional(::BroadFunctionAccelerator) = true
Adapt.adapt_structure(
    ::BroadFunctionAccelerator,
    target::F,
) where {F<:Function} = target

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

const KERNEL_ARGUMENT_TEST_ARGUMENTS = Tuple{DataType,DataType}[]

Adapt.adapt_storage(
    ::KernelArgumentTestAdaptor,
    array::KernelArgumentTestArray{T,N},
) where {T,N} = KernelArgumentTestDeviceArray{T,N}(
    pointer(array.storage),
    size(array.storage),
)

kernel_argument_test_adapt(argument) =
    Adapt.adapt(KernelArgumentTestAdaptor(), argument)

function kernel_argument_test_adapt(
    argument::SubArray{T,N,<:KernelArgumentTestArray},
) where {T,N}
    return KernelArgumentTestDeviceArray{T,N}(
        pointer(parent(argument).storage),
        size(argument),
    )
end

function KernelAbstractions.argconvert(
    kernel::KernelAbstractions.Kernel{KernelArgumentTestBackend},
    argument,
)
    push!(KERNEL_ARGUMENT_TEST_ARGUMENTS, (typeof(kernel), typeof(argument)))
    return kernel_argument_test_adapt(argument)
end

struct KernelArgumentTestAccelerator <: MLDataDevices.AbstractAcceleratorDevice end
MLDataDevices.functional(::KernelArgumentTestAccelerator) = true
const KERNEL_ARGUMENT_TEST_CURRENT = Ref(:caller)
const KERNEL_ARGUMENT_TEST_CPU_COPIES = Ref(0)

function Adapt.adapt_storage(::KernelArgumentTestAccelerator, array::Array)
    KERNEL_ARGUMENT_TEST_CURRENT[] === :selected || error("wrong active mock device")
    return KernelArgumentTestArray(copy(array))
end

function Base.Array(array::KernelArgumentTestArray)
    KERNEL_ARGUMENT_TEST_CURRENT[] === :selected || error("wrong active mock device")
    KERNEL_ARGUMENT_TEST_CPU_COPIES[] += 1
    return copy(array.storage)
end

struct LateFailAccelerator <: MLDataDevices.AbstractAcceleratorDevice end
MLDataDevices.functional(::LateFailAccelerator) = true
Adapt.adapt_storage(::LateFailAccelerator, array::Array) =
    KernelArgumentTestArray(copy(array))

@eval ImportanceSamplers begin
    function _with_backend_device(f, ::Main.KernelArgumentTestAccelerator)
        previous = Main.KERNEL_ARGUMENT_TEST_CURRENT[]
        Main.KERNEL_ARGUMENT_TEST_CURRENT[] = :selected
        try
            return f()
        finally
            Main.KERNEL_ARGUMENT_TEST_CURRENT[] = previous
        end
    end

    _owned_backend_rng(::Main.KernelArgumentTestAccelerator, seed::UInt64) =
        Random.Xoshiro(seed)
    _owned_backend_rng(::Main.LateFailAccelerator, seed::UInt64) =
        iszero(seed) ? Random.Xoshiro(seed) : error("late RNG construction failure")
end

const HOOK_ACCELERATOR_CURRENT = Ref(:caller)

struct HookAccelerator <: MLDataDevices.AbstractAcceleratorDevice
    resident::Bool
end

Base.eltype(::HookAccelerator) = Nothing

function Adapt.adapt_storage(::HookAccelerator, array::Array)
    HOOK_ACCELERATOR_CURRENT[] === :selected || error("wrong active mock device")
    return copy(array)
end

function hook_accelerator_target(sample, context)::Float64
    HOOK_ACCELERATOR_CURRENT[] === :selected || error("wrong active mock device")
    return context.shift[1] - abs2(sample) / 2
end

@eval ImportanceSamplers begin
    _backend_functional(::Main.HookAccelerator) = true

    function _with_backend_device(f, ::Main.HookAccelerator)
        previous = Main.HOOK_ACCELERATOR_CURRENT[]
        Main.HOOK_ACCELERATOR_CURRENT[] = :selected
        try
            return f()
        finally
            Main.HOOK_ACCELERATOR_CURRENT[] = previous
        end
    end

    _backend_state_resident(device::Main.HookAccelerator, state) = device.resident
    _owned_backend_rng(::Main.HookAccelerator, seed::UInt64) = Random.Xoshiro(seed)
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

function static_mis_device_target(sample, context)
    T = sample isa Number ? typeof(sample) : eltype(sample)
    radius = sample isa Number ? abs2(sample) : sum(abs2, sample)
    return context.shift[1] - T(0.5) * radius
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

@testset "broad device Function rules do not opt closures in" begin
    ordinary = let captured = [0.75]
        (sample, p) -> p.shift[1] + captured[1] - abs2(sample) / 2
    end
    cpu_source = make_transfer_sampler(
        0x2100;
        threaded=false,
        target=ordinary,
    )
    cpu_destination = BroadFunctionCPUDevice()(cpu_source)
    @test prepared_parts(cpu_destination).callable === ordinary

    accelerator_source = make_transfer_sampler(
        0x2101;
        threaded=true,
        target=ordinary,
    )
    accelerator_error = caught_device_error(
        () -> BroadFunctionAccelerator()(accelerator_source),
    )
    @test accelerator_error isa SamplerDeviceError
    @test accelerator_error.reason === :opaque_host_closure
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

    adaptable = AdaptableFunction([0.75])
    adaptable_source = make_transfer_sampler(
        2106;
        threaded=false,
        target=adaptable,
    )
    adaptable_destination = cpu32(adaptable_source)
    adaptable_callable = prepared_parts(adaptable_destination).callable
    @test adaptable_callable isa AdaptableFunction{Vector{Float32}}
    @test adaptable_callable !== adaptable
    @test adaptable_callable.offset !== adaptable.offset
    @test adaptable_callable.offset == Float32[0.75]
    adaptable.offset[1] = 9.0
    @test adaptable_callable.offset == Float32[0.75]

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
          "unspecified; construct a preserving device whose eltype policy is " *
          "Nothing, Float32, or Float64"

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

@testset "static MIS accelerator transfer and capability limits" begin
    device = KernelArgumentTestAccelerator()
    proposals = [
        SphericalGaussian([-1.0], 0.75),
        DiagonalGaussian([0.5], [1.25]),
        SphericalGaussian([2.0], 0.5),
    ]
    bank = ProposalBank(proposals, [1.0, 3.0, 0.0])
    algorithm = ImportanceSampling(
        bank;
        nsamples=17,
        mis_scheme=PartialDeterministicMixture(((1, 3), (2,))),
    )
    source = prepare_sampler(
        Random.Xoshiro(0x2218),
        static_mis_device_target,
        (shift=[0.25],),
        algorithm;
        threaded=true,
    )
    expected_source_rng = copy(getfield(source, :rng))

    @test IS._accelerator_proposal_limit(bank) === nothing
    destination = @inferred device(source)
    method_state = getfield(destination, :method_state)
    packed = getfield(method_state, :bank)
    denominator = getfield(getfield(method_state, :design), :denominator)
    buffers = getfield(destination, :random_buffers)
    @test packed isa IS._PackedDiagonalGaussianBank
    @test all(
        array -> array isa KernelArgumentTestArray,
        (
            packed.locations,
            packed.scales,
            packed.lognormalizers,
            packed.logmasses,
            packed.cdf,
            packed.proposal_ids,
            denominator.group_of_slot,
            denominator.offsets,
            denominator.members,
            denominator.logcoefficients,
            buffers.uniform,
            buffers.normal,
            buffers.assignments,
            buffers.failure_scratch.record.storage,
        ),
    )
    rand(expected_source_rng, UInt64)
    @test rand(getfield(source, :rng), UInt64) ==
          rand(expected_source_rng, UInt64)

    full_source = prepare_sampler(
        Random.Xoshiro(0x221d),
        static_mis_device_target,
        (shift=[0.25],),
        ImportanceSampling(bank; nsamples=17, mis_scheme=StratifiedMixture());
        threaded=true,
    )
    @test @inferred(device(full_source)) isa IS._PreparedImportanceSampler

    bigfloat_bank = ProposalBank(
        [
            SphericalGaussian([-1.0], 0.75),
            SphericalGaussian([1.0], 1.25),
        ],
        BigFloat[1, 3],
    )
    bigfloat_source = prepare_sampler(
        Random.Xoshiro(0x2221),
        static_mis_device_target,
        (shift=[0.25],),
        ImportanceSampling(bigfloat_bank; nsamples=17);
        threaded=true,
    )
    expected_bigfloat_rng = copy(getfield(bigfloat_source, :rng))
    @test bigfloat_source.method_state.bank isa IS._ActiveProposalBank
    bigfloat_error = caught_device_error(() -> device(bigfloat_source))
    @test bigfloat_error isa SamplerDeviceError
    @test bigfloat_error.reason === :generic_proposal_cpu_only
    @test rand(getfield(bigfloat_source, :rng), UInt64) ==
          rand(expected_bigfloat_rng, UInt64)

    late_source = prepare_sampler(
        Random.Xoshiro(0x221e),
        static_mis_device_target,
        (shift=[0.25],),
        algorithm;
        threaded=true,
    )
    expected_late_rng = copy(getfield(late_source, :rng))
    late_error = caught_device_error(() -> LateFailAccelerator()(late_source))
    @test late_error isa SamplerDeviceError
    @test late_error.reason === :accelerator_rng_unavailable
    @test rand(getfield(late_source, :rng), UInt64) ==
          rand(expected_late_rng, UInt64)

    generic = ProposalBank([TransferProposal([0.0]), TransferProposal([1.0])])
    factor = ProposalBank([
        FactorGaussian([0.0, 0.0], [1.0 0.0; 0.0 1.0]),
        FactorGaussian([1.0, 1.0], [1.0 0.0; 0.0 1.0]),
    ])
    transformed = ProposalBank([
        TransformedProposal(SphericalGaussian(0.0, 1.0), PositiveTransform()),
        TransformedProposal(SphericalGaussian(1.0, 1.0), PositiveTransform()),
    ])
    product = ProposalBank([
        ProductProposal((x=SphericalGaussian(0.0, 1.0),)),
        ProductProposal((x=SphericalGaussian(1.0, 1.0),)),
    ])
    limits = (
        generic => :generic_proposal_cpu_only,
        transformed => :transformed_proposal_cpu_only,
        product => :product_proposal_cpu_only,
    )
    @test IS._accelerator_proposal_limit(factor) === nothing
    for (rejected_bank, reason) in limits
        @test IS._accelerator_proposal_limit(rejected_bank) === reason
    end

    for (seed, rejected_bank, reason) in (
        (0x2219, generic, :generic_proposal_cpu_only),
        (0x221b, transformed, :transformed_proposal_cpu_only),
        (0x221c, product, :product_proposal_cpu_only),
    )
        rejected = prepare_sampler(
            Random.Xoshiro(seed),
            static_mis_device_target,
            (shift=[0.25],),
            ImportanceSampling(rejected_bank; nsamples=17);
            threaded=true,
        )
        expected_rng = copy(getfield(rejected, :rng))
        error = caught_device_error(() -> device(rejected))
        @test error isa SamplerDeviceError
        @test error.reason === reason
        @test rand(getfield(rejected, :rng), UInt64) == rand(expected_rng, UInt64)
    end

    factor_source = prepare_sampler(
        Random.Xoshiro(0x221a),
        static_mis_device_target,
        (shift=[0.25],),
        ImportanceSampling(factor; nsamples=17);
        threaded=true,
    )
    factor_destination = @inferred device(factor_source)
    factor_state = getfield(factor_destination, :method_state)
    factor_packed = getfield(factor_state, :bank)
    factor_buffers = getfield(factor_destination, :random_buffers)
    @test factor_packed isa IS._PackedFactorGaussianBank
    @test all(
        array -> array isa KernelArgumentTestArray,
        (
            factor_packed.locations,
            factor_packed.factors,
            factor_packed.lognormalizers,
            factor_packed.logmasses,
            factor_packed.cdf,
            factor_packed.proposal_ids,
            factor_buffers.uniform,
            factor_buffers.normal,
            factor_buffers.assignments,
            factor_buffers.solve_scratch,
            factor_buffers.failure_scratch.record.storage,
        ),
    )

    for (seed, rejected_bank, error_type, message) in (
        (
            0x221f,
            ProposalBank([
                SphericalGaussian([0.0], 1.0),
                SphericalGaussian([0.0, 1.0], 1.0),
            ]),
            DimensionMismatch,
            "one common dimension",
        ),
        (
            0x2220,
            ProposalBank(Any[
                SphericalGaussian(Float32[0], 1.0f0),
                SphericalGaussian(Float64[0], 1.0),
            ]),
            ArgumentError,
            "one floating type",
        ),
    )
        rng = Random.Xoshiro(seed)
        expected_rng = copy(rng)
        error = caught_device_error() do
            prepare_sampler(
                rng,
                static_mis_device_target,
                (shift=[0.25],),
                ImportanceSampling(rejected_bank; nsamples=17);
                threaded=true,
            )
        end
        @test error isa error_type
        @test occursin(message, sprint(showerror, error))
        @test rand(rng, UInt64) == rand(expected_rng, UInt64)
    end
end

@testset "DM-PMC accelerator transfer and preflight" begin
    device = KernelArgumentTestAccelerator()
    diagonal_bank = ProposalBank(
        [
            SphericalGaussian([-1.0, 0.5], 0.75),
            DiagonalGaussian([0.5, -0.25], [1.25, 0.5]),
            SphericalGaussian([2.0, 1.0], 0.5),
        ],
        [1.0, 3.0, 0.0],
    )
    factor_bank = ProposalBank([
        FactorGaussian([0.0, 0.0], [1.0 0.0; 0.25 0.75]),
        FactorGaussian([1.0, 1.0], [0.8 0.0; -0.1 1.2]),
    ])

    function assert_dm_pmc_transfer(source, bank_type)
        empty!(KERNEL_ARGUMENT_TEST_ARGUMENTS)
        destination = device(source)
        method_state = getfield(destination, :method_state)
        bank = getfield(method_state, :bank)
        plan = getfield(method_state, :plan)
        workspace = getfield(method_state, :workspace)
        buffers = getfield(destination, :random_buffers)

        @test bank isa bank_type
        bank_scale = bank isa IS._PackedDiagonalGaussianBank ? bank.scales : bank.factors
        solve_scratch = workspace.solve_scratch
        arrays = (
            bank.locations,
            bank_scale,
            bank.lognormalizers,
            bank.logmasses,
            bank.cdf,
            bank.proposal_ids,
            plan.counts,
            plan.assignments,
            plan.logcoefficients,
            workspace.round_samples,
            workspace.round_logweights,
            workspace.round_proposal_ids,
            workspace.resampling_cdf,
            workspace.ancestors,
            workspace.candidate_locations,
            buffers.normals,
            buffers.resampling_uniforms,
            buffers.failure_scratch.record.storage,
        )
        @test all(array -> array isa KernelArgumentTestArray, arrays)
        if bank isa IS._PackedFactorGaussianBank
            @test solve_scratch isa KernelArgumentTestArray
        else
            @test solve_scratch isa IS._NoMISSolveScratch
        end
        @test plan.schedule isa Tuple
        @test plan.offsets isa Tuple
        @test getfield(destination, :algorithm) !== getfield(source, :algorithm)
        @test getfield(destination, :algorithm).bank.proposals isa Vector
        @test getfield(destination, :algorithm).bank.masses isa Vector

        snapshot_error = try
            current_proposal(destination)
            nothing
        catch error
            error
        end
        @test snapshot_error isa ArgumentError
        @test occursin("current_proposal(cpu_device(), sampler)", snapshot_error.msg)
        @test bank.locations isa KernelArgumentTestArray

        round_size = maximum(plan.schedule)
        representative_arguments = (
            IS._sample_view(workspace.round_samples, 1:round_size),
            view(workspace.round_logweights, 1:round_size),
            view(workspace.round_proposal_ids, 1:round_size),
            view(plan.assignments, 1:round_size, 1),
            view(workspace.resampling_cdf, 1:round_size),
        )
        @test ndims.(representative_arguments) == (2, 1, 1, 1, 1)
        @test all(argument -> argument isa SubArray, representative_arguments)
        backend = KernelAbstractions.get_backend(buffers.normals)
        round_kernel = IS._mis_round_launch_kernel!(backend)
        finalize_kernel = IS._dm_pmc_finalize_cdf_kernel!(backend)
        select_kernel = IS._dm_pmc_select_ancestors_kernel!(backend)
        gather_kernel = IS._dm_pmc_gather_ancestors_kernel!(backend)
        for (kernel, argument) in (
            (round_kernel, representative_arguments[1]),
            (round_kernel, representative_arguments[2]),
            (round_kernel, representative_arguments[3]),
            (round_kernel, representative_arguments[4]),
            (finalize_kernel, representative_arguments[5]),
            (select_kernel, representative_arguments[5]),
            (gather_kernel, representative_arguments[1]),
        )
            @test (typeof(kernel), typeof(argument)) in
                  KERNEL_ARGUMENT_TEST_ARGUMENTS
        end
        return destination
    end

    destinations = Dict{Symbol,Any}()
    for (label, seed, bank, bank_type) in (
        (:diagonal, 0x2222, diagonal_bank, IS._PackedDiagonalGaussianBank),
        (:factor, 0x2223, factor_bank, IS._PackedFactorGaussianBank),
    )
        algorithm = DeterministicMixturePMC(
            bank;
            rounds=2,
            round_size=[5, 7],
        )
        source = prepare_sampler(
            Random.Xoshiro(seed),
            static_mis_device_target,
            (shift=[0.25],),
            algorithm;
            threaded=true,
        )
        expected_source_rng = copy(getfield(source, :rng))
        destination = assert_dm_pmc_transfer(source, bank_type)
        destinations[label] = destination
        @test getfield(destination, :device) === device
        rand(expected_source_rng, UInt64)
        @test rand(getfield(source, :rng), UInt64) == rand(expected_source_rng, UInt64)
    end

    destination = destinations[:diagonal]
    packed = getfield(destination, :method_state).bank
    packed.locations.storage .= [-3.0 5.0; 4.0 -6.0]
    retained_locations = copy(packed.locations.storage)
    retained_ids = copy(packed.proposal_ids.storage)
    KERNEL_ARGUMENT_TEST_CURRENT[] = :caller
    KERNEL_ARGUMENT_TEST_CPU_COPIES[] = 0

    snapshot = @inferred current_proposal(MLDataDevices.cpu_device(), destination)
    @test KERNEL_ARGUMENT_TEST_CURRENT[] === :caller
    @test KERNEL_ARGUMENT_TEST_CPU_COPIES[] == 2
    @test snapshot.proposals[1].location == [-3.0, 4.0]
    @test snapshot.proposals[2].location == [5.0, -6.0]
    @test snapshot.proposals[3].location == diagonal_bank.proposals[3].location
    @test snapshot.masses == diagonal_bank.masses

    snapshot.proposals[1].location[1] = 1.0e6
    snapshot.proposals[3].location[1] = -1.0e6
    snapshot.masses[1] = 0.0
    @test packed.locations.storage == retained_locations
    @test packed.proposal_ids.storage == retained_ids
    @test destination.algorithm.bank.proposals[3].location ==
          diagonal_bank.proposals[3].location
    @test destination.algorithm.bank.masses == diagonal_bank.masses

    unsupported_destination = try
        current_proposal(device, destination)
        nothing
    catch error
        error
    end
    @test unsupported_destination isa ArgumentError
    @test occursin("CPU destination", unsupported_destination.msg)
    @test KERNEL_ARGUMENT_TEST_CURRENT[] === :caller

    rejected_source = prepare_sampler(
        Random.Xoshiro(0x2224),
        static_mis_device_target,
        UnadaptedKernelContext([0.25]),
        DeterministicMixturePMC(
            diagonal_bank;
            rounds=2,
            round_size=6,
        );
        threaded=true,
    )
    expected_rejected_rng = copy(getfield(rejected_source, :rng))
    rejected_error = caught_device_error(() -> device(rejected_source))
    @test rejected_error isa SamplerDeviceError
    @test rejected_error.reason === :kernel_argument_unsupported
    @test rand(getfield(rejected_source, :rng), UInt64) ==
          rand(expected_rejected_rng, UInt64)
    @test length(importance_sample!(rejected_source)) == 12

    serial_source = prepare_sampler(
        Random.Xoshiro(0x2225),
        static_mis_device_target,
        (shift=[0.25],),
        DeterministicMixturePMC(
            diagonal_bank;
            rounds=2,
            round_size=6,
        );
        threaded=false,
    )
    expected_serial_rng = copy(getfield(serial_source, :rng))
    serial_error = caught_device_error(() -> device(serial_source))
    @test serial_error isa SamplerDeviceError
    @test serial_error.reason === :serial_accelerator
    @test rand(getfield(serial_source, :rng), UInt64) ==
          rand(expected_serial_rng, UInt64)
end

@testset "AMIS accelerator transfer and factorization preflight" begin
    device = KernelArgumentTestAccelerator()
    source = prepare_sampler(
        Random.Xoshiro(0x2226),
        static_mis_device_target,
        (shift=[0.25],),
        AMIS(
            FactorGaussian([0.0, 0.0], [1.0 0.0; 0.25 0.75]);
            rounds=3,
            round_size=[4, 5, 6],
        );
        threaded=true,
    )
    expected_rng = copy(getfield(source, :rng))

    transferred = IS._with_backend_device(device) do
        algorithm = IS._copy_accelerator_algorithm(
            device,
            source.algorithm,
            source.method_state,
        )
        state = IS._prepare_transferred_method_state(
            device,
            algorithm,
            source.method_state,
        )
        buffers = IS._allocate_random_buffers(
            device,
            IS._algorithm_proposal(algorithm),
            state,
            IS._algorithm_sample_budget(algorithm),
        )
        (algorithm=algorithm, state=state, buffers=buffers)
    end

    history = transferred.state.history
    workspace = transferred.state.workspace
    @test all(
        array -> array isa KernelArgumentTestArray,
        (
            transferred.state.logcounts,
            history.means,
            history.factors,
            history.lognormalizers,
            workspace.samples,
            workspace.logtargets,
            workspace.lognumerators,
            workspace.logweights,
            workspace.normalized_weights,
            workspace.centered_scaled,
            workspace.covariance,
            transferred.buffers.uniform,
            transferred.buffers.normal,
            transferred.buffers.failure_scratch.record.storage,
        ),
    )
    transferred_backend = IS._transferred_backend_state(
        transferred.algorithm,
        transferred.state,
        source.target,
        transferred.buffers,
    )
    prepared_backend = IS._prepared_backend_state(source, source.method_state)
    @test transferred_backend == (
        transferred.state,
        source.target,
        transferred.buffers,
    )
    @test prepared_backend == (
        source.method_state,
        source.target,
        source.random_buffers,
        source.rng,
    )

    error = caught_device_error(() -> device(source))
    @test error isa SamplerDeviceError
    @test error.reason === :accelerator_factorization_unavailable
    @test rand(getfield(source, :rng), UInt64) == rand(expected_rng, UInt64)
end

@testset "backend hooks own device scope and validate residency" begin
    algorithm = ImportanceSampling(SphericalGaussian(0.0, 1.0); nsamples=16)
    make_hook_sampler(seed) = prepare_sampler(
        Random.Xoshiro(seed),
        hook_accelerator_target,
        (shift=(0.25,),),
        algorithm;
        threaded=true,
    )

    HOOK_ACCELERATOR_CURRENT[] = :caller
    prepared = HookAccelerator(true)(make_hook_sampler(0x2216))
    @test HOOK_ACCELERATOR_CURRENT[] === :caller

    result = importance_sample!(prepared)
    @test HOOK_ACCELERATOR_CURRENT[] === :caller
    @test length(result) == 16

    source = make_hook_sampler(0x2217)
    expected_rng = copy(getfield(source, :rng))
    residency_error = caught_device_error(() -> HookAccelerator(false)(source))
    @test residency_error isa SamplerDeviceError
    @test residency_error.reason === :device_residency_mismatch
    @test HOOK_ACCELERATOR_CURRENT[] === :caller
    @test rand(getfield(source, :rng), UInt64) == rand(expected_rng, UInt64)
end
