using Test
using ImportanceSamplers
import ADTypes
import Adapt
import DensityInterface
import ForwardDiff
import KernelAbstractions
import LinearAlgebra
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

const DERIVATIVE_VALUE_TRANSFERS = Ref(0)
const DERIVATIVE_GRADIENT_TRANSFERS = Ref(0)
const DERIVATIVE_CONTEXT_TRANSFERS = Ref(0)

struct DerivativeTransferValue{A<:AbstractVector}
    offset::A
end

MLDataDevices.isleaf(::DerivativeTransferValue) = true

function (target::DerivativeTransferValue)(sample, p)
    return p.shift[1] + target.offset[1] - abs2(sample) / 2
end

function Adapt.adapt_structure(to, target::DerivativeTransferValue)
    to isa KernelArgumentTestAdaptor || (DERIVATIVE_VALUE_TRANSFERS[] += 1)
    return DerivativeTransferValue(Adapt.adapt(to, target.offset))
end

struct DerivativeTransferGradient{A<:AbstractVector}
    scale::A
end

MLDataDevices.isleaf(::DerivativeTransferGradient) = true

function (gradient::DerivativeTransferGradient)(destination, sample, p)
    destination .= -gradient.scale[1] .* sample
    return destination
end

function Adapt.adapt_structure(to, gradient::DerivativeTransferGradient)
    to isa KernelArgumentTestAdaptor || (DERIVATIVE_GRADIENT_TRANSFERS[] += 1)
    return DerivativeTransferGradient(Adapt.adapt(to, gradient.scale))
end

struct DerivativeTransferContext{A<:AbstractVector}
    shift::A
end

MLDataDevices.isleaf(::DerivativeTransferContext) = true

function Adapt.adapt_structure(to, context::DerivativeTransferContext)
    to isa KernelArgumentTestAdaptor || (DERIVATIVE_CONTEXT_TRANSFERS[] += 1)
    return DerivativeTransferContext(Adapt.adapt(to, context.shift))
end

struct GRAMISTransferValue{A<:AbstractVector}
    offset::A
end

MLDataDevices.isleaf(::GRAMISTransferValue) = true

function (target::GRAMISTransferValue)(sample, p)
    return p.shift[1] + target.offset[1] - sum(abs2, sample) / 2
end

function Adapt.adapt_structure(to, target::GRAMISTransferValue)
    to isa KernelArgumentTestAdaptor || (DERIVATIVE_VALUE_TRANSFERS[] += 1)
    return GRAMISTransferValue(Adapt.adapt(to, target.offset))
end

struct GRAMISTransferGradient{A<:AbstractVector}
    scale::A
end

MLDataDevices.isleaf(::GRAMISTransferGradient) = true

function (gradient::GRAMISTransferGradient)(destination, sample, p)
    destination .= -gradient.scale[1] .* sample
    return destination
end

function Adapt.adapt_structure(to, gradient::GRAMISTransferGradient)
    to isa KernelArgumentTestAdaptor || (DERIVATIVE_GRADIENT_TRANSFERS[] += 1)
    return GRAMISTransferGradient(Adapt.adapt(to, gradient.scale))
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
Base.copy(array::KernelArgumentTestArray) = KernelArgumentTestArray(copy(array.storage))
const KERNEL_ARGUMENT_TEST_INT_SIMILAR_LENGTHS = Int[]
Base.similar(
    ::KernelArgumentTestArray,
    ::Type{T},
    dimensions::Dims{N},
) where {T,N} = begin
    T === Int && push!(KERNEL_ARGUMENT_TEST_INT_SIMILAR_LENGTHS, prod(dimensions))
    KernelArgumentTestArray(Array{T,N}(undef, dimensions))
end

KernelAbstractions.get_backend(::KernelArgumentTestArray) =
    KernelArgumentTestBackend()

struct KernelArgumentTestDeviceArray{T,N} <: AbstractArray{T,N}
    pointer::Ptr{T}
    dimensions::NTuple{N,Int}
end

Base.size(array::KernelArgumentTestDeviceArray) = array.dimensions

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
    converted = kernel_argument_test_adapt(argument)
    push!(KERNEL_ARGUMENT_TEST_ARGUMENTS, (typeof(kernel), typeof(converted)))
    return converted
end

struct KernelArgumentTestAccelerator <: MLDataDevices.AbstractAcceleratorDevice end
MLDataDevices.functional(::KernelArgumentTestAccelerator) = true
MLDataDevices.get_device(::KernelArgumentTestArray) = KernelArgumentTestAccelerator()
MLDataDevices.get_device(
    array::SubArray{T,N,<:KernelArgumentTestArray},
) where {T,N} = MLDataDevices.get_device(parent(array))
const KERNEL_ARGUMENT_TEST_CURRENT = Ref(:caller)
const KERNEL_ARGUMENT_TEST_CPU_COPIES = Ref(0)
const KERNEL_ARGUMENT_TEST_CPU_ELEMENTS = Ref(0)

function Adapt.adapt_storage(::KernelArgumentTestAccelerator, array::Array)
    KERNEL_ARGUMENT_TEST_CURRENT[] === :selected || error("wrong active mock device")
    return KernelArgumentTestArray(copy(array))
end

struct GRAMISFailClosedAccelerator <: MLDataDevices.AbstractAcceleratorDevice end
MLDataDevices.functional(::GRAMISFailClosedAccelerator) = true

function Adapt.adapt_storage(::GRAMISFailClosedAccelerator, array::Array)
    return Adapt.adapt_storage(KernelArgumentTestAccelerator(), array)
end

function Base.Array(array::KernelArgumentTestArray)
    KERNEL_ARGUMENT_TEST_CURRENT[] === :selected || error("wrong active mock device")
    KERNEL_ARGUMENT_TEST_CPU_COPIES[] += 1
    KERNEL_ARGUMENT_TEST_CPU_ELEMENTS[] += length(array)
    return copy(array.storage)
end

function Base.Array(
    array::SubArray{T,N,<:KernelArgumentTestArray},
) where {T,N}
    KERNEL_ARGUMENT_TEST_CURRENT[] === :selected || error("wrong active mock device")
    KERNEL_ARGUMENT_TEST_CPU_COPIES[] += 1
    KERNEL_ARGUMENT_TEST_CPU_ELEMENTS[] += length(array)
    return Array(view(parent(array).storage, parentindices(array)...))
end

struct LateFailAccelerator <: MLDataDevices.AbstractAcceleratorDevice end
MLDataDevices.functional(::LateFailAccelerator) = true
Adapt.adapt_storage(::LateFailAccelerator, array::Array) =
    KernelArgumentTestArray(copy(array))

struct AMISExecutionTestAccelerator <: MLDataDevices.AbstractAcceleratorDevice end
MLDataDevices.functional(::AMISExecutionTestAccelerator) = true
const AMIS_EXECUTION_TEST_CURRENT = Ref(:caller)

struct AMISPublicationSyncBackend <: KernelAbstractions.GPU end

struct AMISPublicationSyncArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    storage::A
end

Base.size(array::AMISPublicationSyncArray) = size(array.storage)
Base.IndexStyle(::Type{<:AMISPublicationSyncArray}) = IndexLinear()
Base.getindex(array::AMISPublicationSyncArray, indices...) =
    getindex(array.storage, indices...)
Base.setindex!(array::AMISPublicationSyncArray, value, indices...) =
    setindex!(array.storage, value, indices...)
KernelAbstractions.get_backend(::AMISPublicationSyncArray) =
    AMISPublicationSyncBackend()
KernelAbstractions.synchronize(::AMISPublicationSyncBackend) =
    error("intentional AMIS publication synchronization failure")

mutable struct AMISExecutionTestRNG <: Random.AbstractRNG
    draws::Int
end

mutable struct AMISExecutionPrefilledRNG{T} <: Random.AbstractRNG
    batches::Vector{Vector{T}}
    index::Int
end

struct AMISZeroFloat32Target end
(::AMISZeroFloat32Target)(sample)::Float32 = 0.0f0

struct AMISZeroSampleFailureTarget{T} end

function (::AMISZeroSampleFailureTarget{T})(sample)::T where {T}
    value = sample isa Real ? sample : sample[1]
    return iszero(value) ? T(NaN) : zero(T)
end

function Random.randn!(rng::AMISExecutionTestRNG, destination::AbstractArray)
    fill!(destination, zero(eltype(destination)))
    rng.draws += 1
    return destination
end

function Random.randn!(rng::AMISExecutionPrefilledRNG, destination::AbstractArray)
    copyto!(destination, rng.batches[rng.index])
    rng.index += 1
    return destination
end

function Adapt.adapt_storage(::AMISExecutionTestAccelerator, array::Array)
    AMIS_EXECUTION_TEST_CURRENT[] === :selected || error("wrong active mock device")
    return copy(array)
end

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

    _preflight_first_order_gramis_factorization!(
        ::Main.KernelArgumentTestAccelerator,
        method_state::_PreparedFirstOrderGRAMIS,
    ) = nothing

    _execute_first_order_gramis_live_preflight!(
        ::Main.KernelArgumentTestAccelerator,
        method_state::_PreparedFirstOrderGRAMIS,
        target,
        random_buffers::_RandomBuffers,
    ) = nothing

    function _with_backend_device(f, ::Main.GRAMISFailClosedAccelerator)
        previous = Main.KERNEL_ARGUMENT_TEST_CURRENT[]
        Main.KERNEL_ARGUMENT_TEST_CURRENT[] = :selected
        try
            return f()
        finally
            Main.KERNEL_ARGUMENT_TEST_CURRENT[] = previous
        end
    end

    _owned_backend_rng(::Main.GRAMISFailClosedAccelerator, seed::UInt64) =
        Random.Xoshiro(seed)

    _owned_backend_rng(::Main.LateFailAccelerator, seed::UInt64) =
        iszero(seed) ? Random.Xoshiro(seed) : error("late RNG construction failure")

    function _with_backend_device(f, ::Main.AMISExecutionTestAccelerator)
        previous = Main.AMIS_EXECUTION_TEST_CURRENT[]
        Main.AMIS_EXECUTION_TEST_CURRENT[] = :selected
        try
            return f()
        finally
            Main.AMIS_EXECUTION_TEST_CURRENT[] = previous
        end
    end

    _backend_state_resident(::Main.AMISExecutionTestAccelerator, state) = true
    _owned_backend_rng(::Main.AMISExecutionTestAccelerator, seed::UInt64) =
        Main.AMISExecutionTestRNG(0)

    function _amis_potrf!(
        ::Main.AMISExecutionTestAccelerator,
        factor::StridedMatrix{T},
    ) where {T<:Union{Float32,Float64}}
        all(isfinite, factor) || return factor
        factorization = LinearAlgebra.cholesky!(LinearAlgebra.Symmetric(factor, :L))
        copyto!(factor, factorization.L)
        return factor
    end
    _preflight_accelerator_method(
        ::Main.AMISExecutionTestAccelerator,
        target,
        algorithm::AMIS,
        method_state::_PreparedAMIS,
        random_buffers::_RandomBuffers,
        factor_execution,
    ) = nothing

    function _amis_potrf!(
        ::Main.KernelArgumentTestAccelerator,
        factor::Main.KernelArgumentTestArray{T,2},
    ) where {T<:Union{Float32,Float64}}
        factorization = LinearAlgebra.cholesky!(
            LinearAlgebra.Symmetric(factor.storage, :L),
        )
        copyto!(factor.storage, factorization.L)
        return factor
    end
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
        callable=getfield(target, :logdensity),
        context=getfield(target, :context),
        proposal=getfield(getfield(prepared, :algorithm), :proposal),
        algorithm=getfield(prepared, :algorithm),
        device=getfield(prepared, :device),
    )
end

function make_transfer_sampler(
    rng::Random.AbstractRNG;
    factor_execution=FusedFactorExecution(),
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
        factor_execution,
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

function first_order_gramis_device_target(sample, context)
    return context.shift[1] - sum(abs2, sample) / 2
end

function first_order_gramis_device_gradient!(destination, sample, context)
    destination .= -sample
    return destination
end

function first_order_gramis_float16_target(sample, context)::Float16
    return context.shift[1] - sum(abs2, sample) / 2
end

function first_order_gramis_outofplace_gradient(sample, context)
    return -sample
end

function first_order_gramis_cpu_gradient!(
    destination::Vector,
    sample::Vector,
    context,
)
    destination .= -sample
    return destination
end

function first_order_gramis_cpu_gradient!(
    destination::Vector,
    sample::SubArray{T,1,<:Matrix},
    context,
) where {T}
    destination .= -sample
    return destination
end

function collect_nested_arrays(value)
    arrays = Any[]
    seen = Base.IdSet{Any}()

    function visit(value)
        if value isa AbstractArray
            push!(arrays, value)
            return
        end
        type = typeof(value)
        if value isa Union{Nothing,Number,AbstractString,Symbol,Type,Module} ||
           isprimitivetype(type)
            return
        end
        if Base.ismutabletype(type)
            value in seen && return
            push!(seen, value)
        end
        for field in 1:fieldcount(type)
            isdefined(value, field) && visit(getfield(value, field))
        end
        return
    end

    visit(value)
    return arrays
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

@testset "FirstOrderGRAMIS target ownership and accelerator transfer" begin
    bank = ProposalBank([
        SphericalGaussian([-2.0, 0.0], 0.75),
        DiagonalGaussian([0.0, 2.0], [1.25, 0.5]),
        FactorGaussian([2.0, 0.0], [1.0 0.0; 0.25 1.5]),
    ])
    algorithm = FirstOrderGRAMIS(
        bank;
        rounds=3,
        round_size=[15, 16, 17],
        repulsion_strength=[0.1, 0.2, 0.3],
    )
    value = GRAMISTransferValue([0.0])
    gradient = GRAMISTransferGradient([1.0])
    context = DerivativeTransferContext([0.25])
    source = prepare_sampler(
        Random.Xoshiro(0x4752414d4953),
        LogTarget(value; grad=gradient),
        context,
        algorithm;
        threaded=true,
    )
    @test source.target === source.method_state.serial_gradient.target ===
          source.method_state.threaded_gradient.target
    @test source.target.context === context

    device = KernelArgumentTestAccelerator()
    expected_rng = copy(source.rng)
    DERIVATIVE_VALUE_TRANSFERS[] = 0
    DERIVATIVE_GRADIENT_TRANSFERS[] = 0
    DERIVATIVE_CONTEXT_TRANSFERS[] = 0
    destination = device(source)
    state = destination.method_state
    committed = state.committed
    run = state.run
    candidate = state.candidate
    transferred_arrays = collect_nested_arrays((
        IS._first_order_gramis_resident_state(state),
        destination.target,
        destination.random_buffers,
        destination.rng,
    ))
    @test !isempty(transferred_arrays)
    @test all(array -> array isa KernelArgumentTestArray, transferred_arrays)
    for arrays in (
        (committed.locations, run.locations, candidate.locations),
        (committed.factors, run.factors, candidate.factors),
        (
            committed.lognormalizers,
            run.lognormalizers,
            candidate.lognormalizers,
        ),
    )
        @test arrays[1] !== arrays[2]
        @test arrays[1] !== arrays[3]
        @test arrays[2] !== arrays[3]
    end
    @test run.logmasses === committed.logmasses === candidate.logmasses
    @test run.cdf === committed.cdf === candidate.cdf
    @test run.proposal_ids === committed.proposal_ids === candidate.proposal_ids
    @test state.serial_gradient === state.threaded_gradient
    @test state.serial_gradient !== source.method_state.serial_gradient
    @test state.serial_gradient.target === destination.target
    @test destination.target.context === state.serial_gradient.target.context
    @test DERIVATIVE_VALUE_TRANSFERS[] == 1
    @test DERIVATIVE_GRADIENT_TRANSFERS[] == 1
    @test DERIVATIVE_CONTEXT_TRANSFERS[] == 1
    @test destination.device === device
    rand(expected_rng, UInt64)
    @test rand(source.rng, UInt64) == rand(expected_rng, UInt64)

    fail_closed_device = GRAMISFailClosedAccelerator()
    @test MLDataDevices.functional(fail_closed_device)
    fail_closed_source = prepare_sampler(
        Random.Xoshiro(0x4752414d4954),
        LogTarget(value; grad=gradient),
        context,
        algorithm;
        threaded=true,
    )
    expected_fail_closed_rng = copy(fail_closed_source.rng)
    fail_closed_error = caught_device_error(
        () -> fail_closed_device(fail_closed_source),
    )
    @test fail_closed_error isa SamplerDeviceError
    @test fail_closed_error.reason === :first_order_gramis_accelerator_unavailable
    @test rand(fail_closed_source.rng, UInt64) ==
          rand(expected_fail_closed_rng, UInt64)
end

@testset "FirstOrderGRAMIS accelerator derivative preflight" begin
    bank = ProposalBank([
        SphericalGaussian([-1.0, 0.0], 1.0),
        SphericalGaussian([1.0, 0.0], 1.0),
    ])
    algorithm = FirstOrderGRAMIS(
        bank;
        rounds=1,
        round_size=8,
        repulsion_strength=0.0,
    )
    context = DerivativeTransferContext([0.25])
    device = KernelArgumentTestAccelerator()

    missing_rng = Random.Xoshiro(0x4752414d495301)
    expected_missing_rng = copy(missing_rng)
    @test_throws ArgumentError prepare_sampler(
        missing_rng,
        LogTarget(GRAMISTransferValue([0.0])),
        context,
        algorithm;
        threaded=true,
    )
    @test rand(missing_rng, UInt64) == rand(expected_missing_rng, UInt64)

    wrong_scalar_rng = Random.Xoshiro(0x4752414d495306)
    expected_wrong_scalar_rng = copy(wrong_scalar_rng)
    wrong_scalar_error = caught_device_error() do
        prepare_sampler(
            wrong_scalar_rng,
            LogTarget(
                first_order_gramis_float16_target;
                grad=first_order_gramis_device_gradient!,
            ),
            context,
            algorithm;
            threaded=true,
        )
    end
    @test wrong_scalar_error isa SamplerExecutionError
    @test occursin(
        "target log-density return type must be provably limited to Float32 " *
        "and Float64; inferred Float16",
        sprint(showerror, wrong_scalar_error),
    )
    @test rand(wrong_scalar_rng, UInt64) == rand(expected_wrong_scalar_rng, UInt64)

    cases = (
        (
            0x4752414d495302,
            LogTarget(
                GRAMISTransferValue([0.0]),
                ADTypes.AutoForwardDiff(),
            ),
            :gradient_source_cpu_only,
        ),
        (
            0x4752414d495303,
            LogTarget(
                GRAMISTransferValue([0.0]);
                grad=first_order_gramis_outofplace_gradient,
            ),
            :out_of_place_gradient_cpu_only,
        ),
        (
            0x4752414d495304,
            LogTarget(
                GRAMISTransferValue([0.0]);
                grad=first_order_gramis_cpu_gradient!,
            ),
            :gradient_source_cpu_only,
        ),
    )
    for (seed, target, reason) in cases
        source = prepare_sampler(
            Random.Xoshiro(seed),
            target,
            context,
            algorithm;
            threaded=true,
        )
        expected_rng = copy(source.rng)
        error = caught_device_error(() -> device(source))
        @test error isa SamplerDeviceError
        @test error.reason === reason
        @test rand(source.rng, UInt64) == rand(expected_rng, UInt64)
    end

    opaque_gradient = let scale = [1.0]
        (destination, sample, p) -> (destination .= -scale[1] .* sample)
    end
    opaque_source = prepare_sampler(
        Random.Xoshiro(0x4752414d495305),
        LogTarget(GRAMISTransferValue([0.0]); grad=opaque_gradient),
        context,
        algorithm;
        threaded=true,
    )
    expected_opaque_rng = copy(opaque_source.rng)
    opaque_error = caught_device_error(() -> device(opaque_source))
    @test opaque_error isa SamplerDeviceError
    @test opaque_error.reason === :opaque_host_closure
    @test rand(opaque_source.rng, UInt64) == rand(expected_opaque_rng, UInt64)
end

@testset "explicit prepared device transfer" begin
    cpu = MLDataDevices.cpu_device()
    cpu32 = MLDataDevices.cpu_device(Float32)
    source = make_transfer_sampler(
        2101;
        factor_execution=BatchedFactorExecution(),
        threaded=false,
    )
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
    @test getfield(destination, :factor_execution) ===
          getfield(source, :factor_execution)
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

@testset "LogTarget derivative metadata transfers once" begin
    value = DerivativeTransferValue([0.25])
    gradient = DerivativeTransferGradient([2.0])
    context = DerivativeTransferContext([1.5])
    adtype = ADTypes.AutoForwardDiff(chunksize=2, tag=:transfer_test)
    source = prepare_sampler(
        Random.Xoshiro(0x2110),
        LogTarget(value, adtype; grad=gradient),
        context,
        ImportanceSampling(TransferProposal([0.5]); nsamples=1);
        threaded=false,
    )
    DERIVATIVE_VALUE_TRANSFERS[] = 0
    DERIVATIVE_GRADIENT_TRANSFERS[] = 0
    DERIVATIVE_CONTEXT_TRANSFERS[] = 0

    destination = @inferred MLDataDevices.cpu_device(Float32)(source)

    @test DERIVATIVE_VALUE_TRANSFERS[] == 1
    @test DERIVATIVE_GRADIENT_TRANSFERS[] == 1
    @test DERIVATIVE_CONTEXT_TRANSFERS[] == 1
    @test destination.target.logdensity isa
          DerivativeTransferValue{Vector{Float32}}
    @test destination.target.gradient isa
          DerivativeTransferGradient{Vector{Float32}}
    @test destination.target.context isa
          DerivativeTransferContext{Vector{Float32}}
    @test destination.target.adtype isa ADTypes.AutoForwardDiff
    @test typeof(destination.target.adtype) === typeof(source.target.adtype)
    @test destination.target.adtype == source.target.adtype

    bound = @inferred IS._bind_resolved_target(
        destination.target,
        Float32(0.25),
    )
    @test bound(Float32(0.25)) === Float32(1.71875)
end

@testset "LogTarget derivative closures reject accelerator transfer" begin
    context = DerivativeTransferContext([1.5])
    proposal = ImportanceSampling(TransferProposal([0.5]); nsamples=1)
    opaque_value = let captured = [0.25]
        (sample, p) -> p.shift[1] + captured[1] - abs2(sample) / 2
    end
    value_source = prepare_sampler(
        Random.Xoshiro(0x2111),
        LogTarget(opaque_value; grad=DerivativeTransferGradient([1.0])),
        context,
        proposal;
        threaded=true,
    )
    value_error = caught_device_error(() -> FunctionalAccelerator()(value_source))
    @test value_error isa SamplerDeviceError
    @test value_error.reason === :opaque_host_closure

    opaque_gradient = let captured = [1.0]
        (destination, sample, p) -> (destination .= -captured[1] .* sample)
    end
    gradient_source = prepare_sampler(
        Random.Xoshiro(0x2112),
        LogTarget(DerivativeTransferValue([0.25]); grad=opaque_gradient),
        context,
        proposal;
        threaded=true,
    )
    gradient_error = caught_device_error(
        () -> FunctionalAccelerator()(gradient_source),
    )
    @test gradient_error isa SamplerDeviceError
    @test gradient_error.reason === :opaque_host_closure
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
        run_bank = getfield(method_state, :run_bank)
        plan = getfield(method_state, :plan)
        workspace = getfield(method_state, :workspace)
        buffers = getfield(destination, :random_buffers)

        @test bank isa bank_type
        @test run_bank isa bank_type
        @test run_bank.locations isa KernelArgumentTestArray
        @test run_bank.locations !== bank.locations
        @test run_bank.lognormalizers === bank.lognormalizers
        @test run_bank.logmasses === bank.logmasses
        @test run_bank.cdf === bank.cdf
        @test run_bank.proposal_ids === bank.proposal_ids
        if bank isa IS._PackedDiagonalGaussianBank
            @test run_bank.scales === bank.scales
        else
            @test run_bank.factors === bank.factors
        end
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
            @test (typeof(kernel), typeof(kernel_argument_test_adapt(argument))) in
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

@testset "AMIS accelerator transfer, preflight, and snapshots" begin
    device = KernelArgumentTestAccelerator()
    schedule = [4, 5, 6]

    function make_amis_source(seed, proposal)
        return prepare_sampler(
            Random.Xoshiro(seed),
            static_mis_device_target,
            (shift=[0.25],),
            AMIS(proposal; rounds=3, round_size=schedule);
            threaded=true,
        )
    end

    for (seed, proposal, history_type) in (
        (0x2226, SphericalGaussian(0.0, 1.0), IS._AMISScalarHistory),
        (
            0x2227,
            FactorGaussian([0.0, 0.0], [1.0 0.0; 0.25 0.75]),
            IS._AMISFactorHistory,
        ),
    )
        empty!(KERNEL_ARGUMENT_TEST_ARGUMENTS)
        empty!(KERNEL_ARGUMENT_TEST_INT_SIMILAR_LENGTHS)
        source = make_amis_source(seed, proposal)
        expected_rng = copy(getfield(source, :rng))
        destination = device(source)
        state = destination.method_state
        history = state.history
        workspace = state.workspace
        buffers = destination.random_buffers
        scale_storage = history isa IS._AMISScalarHistory ?
                        history.scales : history.factors
        @test history isa history_type
        @test all(
            array -> array isa KernelArgumentTestArray,
            (
                state.logcounts,
                history.means,
                scale_storage,
                history.lognormalizers,
                workspace.samples,
                workspace.logtargets,
                workspace.lognumerators,
                workspace.logweights,
                workspace.normalized_weights,
                workspace.centered_scaled,
                workspace.covariance,
                workspace.candidate_mean,
                workspace.candidate_scale,
                workspace.candidate_lognormalizer,
                buffers.uniform,
                buffers.normal,
                buffers.failure_scratch.record.storage,
            ),
        )
        @test state.schedule isa Tuple
        @test state.offsets isa Tuple
        @test destination.device === device
        @test KERNEL_ARGUMENT_TEST_CURRENT[] === :caller
        @test KERNEL_ARGUMENT_TEST_INT_SIMILAR_LENGTHS == [1]
        rand(expected_rng, UInt64)
        @test rand(getfield(source, :rng), UInt64) == rand(expected_rng, UInt64)

        representative_round = findmax(state.schedule)[2]
        first_sample = state.offsets[representative_round]
        last_sample = state.offsets[representative_round + 1] - 1
        new_indices = first_sample:last_sample
        samples = IS._sample_view(workspace.samples, new_indices)
        binding_sample = IS._native_binding_sample(samples)
        bound_target = IS._bind_resolved_target(destination.target, binding_sample)
        log_type = eltype(workspace.logweights)
        target_argument = IS._NativeDeviceTarget{
            log_type,
            typeof(bound_target),
        }(bound_target)
        backend = KernelAbstractions.get_backend(buffers.normal)
        round_ids = similar(workspace.logweights, Int, length(new_indices))
        logtotal = log(eltype(state.logcounts)(last_sample))
        assignments = IS._FixedMISAssignments(
            representative_round,
            length(new_indices),
        )
        denominator = IS._AMISMixtureDenominator(
            state.logcounts,
            representative_round,
        )
        round_kernel = IS._amis_round_launch_kernel!(backend)
        append_kernel = IS._append_logmixture_kernel!(backend)
        weight_kernel = IS._form_amis_logweights_kernel!(backend)
        expected_preflight = Tuple{DataType,DataType}[]
        function expect_preflight!(kernel, arguments)
            append!(
                expected_preflight,
                (
                    (typeof(kernel), typeof(kernel_argument_test_adapt(argument))) for
                    argument in arguments
                ),
            )
        end
        expect_preflight!(
            round_kernel,
            (
                samples,
                view(workspace.logtargets, new_indices),
                view(workspace.lognumerators, new_indices),
                view(workspace.logweights, new_indices),
                round_ids,
                logtotal,
                representative_round,
                buffers.failure_scratch.record.storage,
                buffers.normal,
                target_argument,
                history,
                assignments,
                denominator,
                workspace.centered_scaled,
            ),
        )
        old_indices = 1:max(first_sample - 1, 1)
        expect_preflight!(
            append_kernel,
            (
                view(workspace.lognumerators, old_indices),
                IS._sample_view(workspace.samples, old_indices),
                history,
                representative_round,
                state.logcounts,
                buffers.failure_scratch.record.storage,
                workspace.centered_scaled,
            ),
        )
        current_indices = 1:last_sample
        expect_preflight!(
            weight_kernel,
            (
                view(workspace.logweights, current_indices),
                view(workspace.logtargets, current_indices),
                view(workspace.lognumerators, current_indices),
                logtotal,
                buffers.failure_scratch.record.storage,
            ),
        )
        if history isa IS._AMISScalarHistory
            ridge_kernel = IS._add_amis_scalar_ridge_kernel!(backend)
            finish_kernel = IS._finish_amis_scalar_candidate_kernel!(backend)
            expect_preflight!(
                ridge_kernel,
                (workspace.covariance, history.scales, representative_round),
            )
            expect_preflight!(
                finish_kernel,
                (
                    workspace.candidate_scale,
                    workspace.candidate_lognormalizer,
                    workspace.covariance,
                    buffers.failure_scratch.record.storage,
                    last_sample + 1,
                ),
            )
        else
            ridge_kernel = IS._add_amis_factor_ridge_kernel!(backend)
            finish_kernel = IS._finish_amis_factor_candidate_kernel!(backend)
            expect_preflight!(
                ridge_kernel,
                (
                    workspace.covariance,
                    history.factors,
                    representative_round,
                ),
            )
            expect_preflight!(
                finish_kernel,
                (
                    workspace.candidate_mean,
                    workspace.candidate_scale,
                    workspace.candidate_lognormalizer,
                    buffers.failure_scratch.record.storage,
                    last_sample + 1,
                ),
            )
        end
        @test KERNEL_ARGUMENT_TEST_ARGUMENTS == expected_preflight

        KERNEL_ARGUMENT_TEST_CPU_COPIES[] = 0
        KERNEL_ARGUMENT_TEST_CPU_ELEMENTS[] = 0
        implicit_error = caught_device_error(() -> current_proposal(destination))
        @test implicit_error isa ArgumentError
        @test occursin("current_proposal(cpu_device(), sampler)", implicit_error.msg)
        @test KERNEL_ARGUMENT_TEST_CPU_COPIES[] == 0
        @test KERNEL_ARGUMENT_TEST_CPU_ELEMENTS[] == 0

        preserving = MLDataDevices.cpu_device()
        snapshot = @inferred current_proposal(preserving, destination)
        @test KERNEL_ARGUMENT_TEST_CURRENT[] === :caller
        @test KERNEL_ARGUMENT_TEST_CPU_COPIES[] == 2
        expected_elements = history isa IS._AMISScalarHistory ? 2 : 6
        @test KERNEL_ARGUMENT_TEST_CPU_ELEMENTS[] == expected_elements
        @test snapshot.location == proposal.location
        if history isa IS._AMISScalarHistory
            @test snapshot.scale.scale == proposal.scale.scale
        else
            @test snapshot.scale.factor == proposal.scale.factor
        end

        KERNEL_ARGUMENT_TEST_CPU_COPIES[] = 0
        KERNEL_ARGUMENT_TEST_CPU_ELEMENTS[] = 0
        converting_error = caught_device_error(
            () -> current_proposal(MLDataDevices.cpu_device(Float32), destination),
        )
        @test converting_error isa ArgumentError
        @test occursin("preserving CPU destination", converting_error.msg)
        @test KERNEL_ARGUMENT_TEST_CPU_COPIES[] == 0
        @test KERNEL_ARGUMENT_TEST_CPU_ELEMENTS[] == 0
        noncpu_error = caught_device_error(
            () -> current_proposal(device, destination),
        )
        @test noncpu_error isa ArgumentError
        @test occursin("CPU destination", noncpu_error.msg)
        @test KERNEL_ARGUMENT_TEST_CPU_COPIES[] == 0
        @test KERNEL_ARGUMENT_TEST_CPU_ELEMENTS[] == 0
    end

    unsupported = make_amis_source(
        0x2228,
        FactorGaussian([0.0, 0.0], [1.0 0.0; 0.25 0.75]),
    )
    expected_unsupported_rng = copy(unsupported.rng)
    unsupported_error = caught_device_error(() -> LateFailAccelerator()(unsupported))
    @test unsupported_error isa SamplerDeviceError
    @test unsupported_error.reason === :accelerator_factorization_unavailable
    @test rand(unsupported.rng, UInt64) == rand(expected_unsupported_rng, UInt64)
end

@testset "AMIS normalization returns the reusable round summary" begin
    for T in (Float32, Float64)
        logweights = T[log(T(1)), log(T(3)), log(T(2)), T(100)]
        normalized = zeros(T, 4)
        transfers = IS._ResultTransferCounter(0, 0)

        summary = @inferred IS._normalize_amis_weights!(
            normalized,
            logweights,
            3,
            transfers,
        )

        @test normalized[1:3] ≈ T[1 / 6, 1 / 2, 1 / 3] rtol = 8eps(T)
        @test normalized[4] == zero(T)
        @test summary.ess ≈ T(18 / 7) rtol = 8eps(T)
        @test summary.lognormalizer ≈ log(T(2)) rtol = 8eps(T)
        @test iszero(transfers.count)
        @test iszero(transfers.bytes)
    end
end

@testset "AMIS covariance failure diagnostics transfer only two device scalars" begin
    covariance = KernelArgumentTestArray([2.0 -4.0; -4.0 3.0])
    transfers = IS._ResultTransferCounter(0, 0)
    KERNEL_ARGUMENT_TEST_CPU_COPIES[] = 0
    KERNEL_ARGUMENT_TEST_CPU_ELEMENTS[] = 0

    diagnostics = IS._amis_covariance_diagnostics(covariance, transfers)

    @test diagnostics == (
        minimum_diagonal=2.0,
        maximum_absolute_entry=4.0,
    )
    @test transfers.count == 2
    @test transfers.bytes == 2sizeof(Float64)
    @test transfers.reasons.covariance_diagnostic.count == 2
    @test transfers.reasons.covariance_diagnostic.bytes == 2sizeof(Float64)
    @test iszero(KERNEL_ARGUMENT_TEST_CPU_COPIES[])
    @test iszero(KERNEL_ARGUMENT_TEST_CPU_ELEMENTS[])
end

@testset "AMIS accelerator scalar covariance failure is transactional" begin
    T = Float32
    scale = nextfloat(zero(T))
    algorithm = AMIS(
        SphericalGaussian(zero(T), scale);
        rounds=1,
        round_size=1,
    )
    target = AMISZeroFloat32Target()

    cpu = prepare_sampler(
        AMISExecutionTestRNG(0),
        target,
        algorithm;
        threaded=false,
    )
    cpu_before = current_proposal(cpu)
    cpu_failure = caught_device_error(() -> importance_sample!(cpu))
    @test cpu_failure isa AMISRoundError
    @test cpu_failure.phase === :factorization
    @test cpu_failure.cause isa LinearAlgebra.PosDefException
    @test cpu_failure.cause.info == 1
    @test cpu_failure.diagnostics.covariance == (
        minimum_diagonal=zero(T),
        maximum_absolute_entry=zero(T),
    )
    @test iszero(cpu_failure.diagnostics.transfers.count)
    @test cpu.rng.draws == 1
    @test current_proposal(cpu) == cpu_before

    source = prepare_sampler(
        Random.Xoshiro(0x2229),
        target,
        algorithm;
        threaded=true,
    )
    device = AMISExecutionTestAccelerator()
    prepared = device(source)
    before = current_proposal(MLDataDevices.cpu_device(), prepared)
    failure = caught_device_error(() -> importance_sample!(prepared))
    @test failure isa AMISRoundError
    if failure isa AMISRoundError
        @test failure.round == 1
        @test failure.phase === :factorization
        @test failure.cause isa LinearAlgebra.PosDefException
        @test failure.diagnostics.covariance ==
              cpu_failure.diagnostics.covariance
        @test iszero(failure.diagnostics.transfers.count)
        if failure.cause isa LinearAlgebra.PosDefException
            @test failure.cause.info == cpu_failure.cause.info == 1
        end
    end
    @test prepared.rng.draws == 1
    failure_storage = prepared.random_buffers.failure_scratch.record.storage
    failure_snapshot = IS._decode_native_failure(
        failure_storage[1],
        failure_storage[2],
    )
    @test failure_snapshot.count == 1
    @test failure_snapshot.reason_bits == IS._AMIS_COVARIANCE_INVALID
    @test iszero(failure_storage[3])
    after = try
        current_proposal(MLDataDevices.cpu_device(), prepared)
    catch cause
        cause
    end
    @test after == before
    @test AMIS_EXECUTION_TEST_CURRENT[] === :caller
    @test !prepared.running
end

@testset "AMIS accelerator authority synchronization is transactional" begin
    T = Float64
    algorithm = AMIS(
        SphericalGaussian(zero(T), one(T));
        rounds=1,
        round_size=3,
    )
    device = AMISExecutionTestAccelerator()
    base = device(prepare_sampler(
        Random.Xoshiro(0x2231),
        AMISZeroFloat32Target(),
        algorithm;
        threaded=true,
    ))
    old_state = base.method_state
    old_history = old_state.history
    history = IS._AMISScalarHistory(
        AMISPublicationSyncArray(old_history.means),
        old_history.scales,
        old_history.lognormalizers,
    )
    state = IS._PreparedAMIS(
        old_state.schedule,
        old_state.offsets,
        old_state.logcounts,
        history,
        old_state.workspace,
        old_state.committed_in_workspace,
    )
    prepared = IS._PreparedImportanceSampler(
        AMISExecutionPrefilledRNG(fill(T[-1, 0, 1], 2), 1),
        base.random_buffers,
        base.target,
        base.algorithm,
        state,
        base.device,
        base.factor_execution,
        base.threaded,
        false,
        false,
    )
    @test importance_sample!(prepared) isa WeightedSamples
    @test prepared.method_state.committed_in_workspace
    before = current_proposal(MLDataDevices.cpu_device(), prepared)

    failure = caught_device_error(() -> importance_sample!(prepared))

    @test failure isa AMISRoundError
    if failure isa AMISRoundError
        @test failure.round == 1
        @test failure.phase === :result_construction
        @test failure.cause isa ErrorException
        @test failure.cause.msg ==
              "intentional AMIS publication synchronization failure"
        @test failure.diagnostics.completed_rounds == 0
    end
    @test current_proposal(MLDataDevices.cpu_device(), prepared) == before
    @test prepared.method_state.committed_in_workspace
    @test !prepared.running
    @test AMIS_EXECUTION_TEST_CURRENT[] === :caller
end

@testset "AMIS final workspace synchronization is transactional" begin
    T = Float64
    algorithm = AMIS(
        SphericalGaussian(zero(T), one(T));
        rounds=1,
        round_size=3,
    )
    device = AMISExecutionTestAccelerator()
    prepared = device(prepare_sampler(
        Random.Xoshiro(0x2232),
        AMISZeroFloat32Target(),
        algorithm;
        threaded=true,
    ))
    @test importance_sample!(prepared) isa WeightedSamples
    before = current_proposal(MLDataDevices.cpu_device(), prepared)

    old_state = prepared.method_state
    old_workspace = old_state.workspace
    workspace = IS._AMISWorkspace(
        old_workspace.samples,
        old_workspace.logtargets,
        old_workspace.lognumerators,
        old_workspace.logweights,
        old_workspace.normalized_weights,
        old_workspace.centered_scaled,
        old_workspace.covariance,
        AMISPublicationSyncArray(old_workspace.candidate_mean),
        old_workspace.candidate_scale,
        old_workspace.candidate_lognormalizer,
    )
    state = IS._PreparedAMIS(
        old_state.schedule,
        old_state.offsets,
        old_state.logcounts,
        old_state.history,
        workspace,
        old_state.committed_in_workspace,
    )
    failing = IS._PreparedImportanceSampler(
        AMISExecutionPrefilledRNG([T[-2, 0, 2]], 1),
        prepared.random_buffers,
        prepared.target,
        prepared.algorithm,
        state,
        prepared.device,
        prepared.factor_execution,
        prepared.threaded,
        false,
        false,
    )

    failure = caught_device_error(() -> importance_sample!(failing))

    @test failure isa AMISRoundError
    if failure isa AMISRoundError
        @test failure.round == 1
        @test failure.phase === :result_construction
        @test failure.cause isa ErrorException
        @test failure.cause.msg ==
              "intentional AMIS publication synchronization failure"
        @test failure.diagnostics.completed_rounds == 1
    end
    @test current_proposal(MLDataDevices.cpu_device(), failing) == before
    @test !failing.method_state.committed_in_workspace
    @test !failing.running
    @test AMIS_EXECUTION_TEST_CURRENT[] === :caller
end

@testset "AMIS packed sample failure precedes the later fit sentinel" begin
    T = Float64
    round_size = 3
    scale = sqrt(floatmax(T)) / T(4)
    algorithm = AMIS(
        FactorGaussian(T[0], reshape(T[scale], 1, 1));
        rounds=1,
        round_size,
    )
    device = AMISExecutionTestAccelerator()
    base = device(prepare_sampler(
        Random.Xoshiro(0x2230),
        AMISZeroSampleFailureTarget{T}(),
        algorithm;
        threaded=true,
    ))
    base_buffers = base.random_buffers
    failure_scratch = IS._NativeFailureScratch(
        base_buffers.failure_scratch.record,
        IS._NoNativeTargetFailures(),
    )
    buffers = IS._RandomBuffers(
        base_buffers.uniform,
        base_buffers.normal,
        failure_scratch,
    )
    rng = AMISExecutionPrefilledRNG(
        [T[-8, 0, 8], T[-0.25, 0.25, 0.5]],
        1,
    )
    prepared = IS._PreparedImportanceSampler(
        rng,
        buffers,
        base.target,
        base.algorithm,
        base.method_state,
        base.device,
        base.factor_execution,
        base.threaded,
        false,
        false,
    )
    before = current_proposal(MLDataDevices.cpu_device(), prepared)

    failure = caught_device_error(() -> importance_sample!(prepared))
    snapshot = IS._decode_native_failure(
        prepared.random_buffers.failure_scratch.record.storage[1],
        prepared.random_buffers.failure_scratch.record.storage[2],
    )

    @test failure isa AMISRoundError
    @test failure.phase === :target
    @test failure.cause isa SamplerExecutionError
    @test failure.cause.sample_index == 2
    @test snapshot.count == 2
    @test snapshot.first_logical_index == 2
    @test snapshot.reason_bits == IS._NATIVE_TARGET_NAN
    @test iszero(prepared.random_buffers.failure_scratch.record.storage[3])
    after = current_proposal(MLDataDevices.cpu_device(), prepared)
    @test after.location == before.location
    @test after.scale.factor == before.scale.factor
    @test after.lognormalizer == before.lognormalizer
    @test rng.index == 2
    @test !prepared.running

    @test importance_sample!(prepared) isa WeightedSamples
    @test rng.index == 3
    @test !prepared.running
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
