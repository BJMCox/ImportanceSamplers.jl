import ADTypes
import DensityInterface
import LogDensityProblems
import MLDataDevices
import Random

const ISD = ImportanceSamplers

struct FirstOrderLDP end

LogDensityProblems.capabilities(::Type{FirstOrderLDP}) =
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(::FirstOrderLDP) = 2
LogDensityProblems.logdensity(::FirstOrderLDP, x) = -sum(abs2, x) / 2
LogDensityProblems.logdensity_and_gradient(::FirstOrderLDP, x) =
    (-sum(abs2, x) / 2, -x)

struct CountingFirstOrderLDP
    value_calls::Base.RefValue{Int}
    ldp_gradient_calls::Base.RefValue{Int}
end

LogDensityProblems.capabilities(::Type{CountingFirstOrderLDP}) =
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(::CountingFirstOrderLDP) = 2

function (target::CountingFirstOrderLDP)(x)
    target.value_calls[] += 1
    return -sum(abs2, x) / 2
end

function LogDensityProblems.logdensity(target::CountingFirstOrderLDP, x)
    target.value_calls[] += 1
    return -sum(abs2, x) / 2
end

function LogDensityProblems.logdensity_and_gradient(
    target::CountingFirstOrderLDP,
    x,
)
    target.ldp_gradient_calls[] += 1
    return -sum(abs2, x) / 2, -x
end

struct CountingExplicitGradient
    calls::Base.RefValue{Int}
end

function (gradient::CountingExplicitGradient)(destination, x)
    gradient.calls[] += 1
    destination .= -x
    return destination
end

struct BothFormExplicitGradient
    inplace_calls::Base.RefValue{Int}
    outofplace_calls::Base.RefValue{Int}
end

function (gradient::BothFormExplicitGradient)(destination, x)
    gradient.inplace_calls[] += 1
    destination .= -x
    return destination
end

function (gradient::BothFormExplicitGradient)(x)
    gradient.outofplace_calls[] += 1
    return -x
end

explicit_inplace_gradient!(destination, x) = (destination .= -x; destination)
explicit_inplace_gradient!(destination, x, p) =
    (destination .= -p[1] .* x; destination)
explicit_inplace_affine_gradient!(destination, x, p) =
    (destination .= -p[1] .* x .+ p[2]; destination)

explicit_outofplace_gradient(x) = -x
explicit_outofplace_context_gradient(x, p) = -p[1] .* x
explicit_outofplace_affine_gradient(x, p) = -p[1] .* x .+ p[2]

cpu_vector_inplace_gradient!(destination::Vector, x::Vector) =
    (destination .= -x; destination)
cpu_vector_outofplace_gradient(x::Vector) = -x
invalid_explicit_gradient(::String) = nothing

context_free_logtarget(x) = -sum(abs2, x) / 2
contextual_logtarget(x, p) = -p[1] * sum(abs2, x) / 2 + p[end]

struct GradientTestAccelerator <: MLDataDevices.AbstractAcceleratorDevice end

struct DerivativeDensityTarget end

DensityInterface.DensityKind(::DerivativeDensityTarget) =
    DensityInterface.IsDensity()
DensityInterface.logdensityof(::DerivativeDensityTarget, x) =
    -sum(abs2, x) / 2

struct GradientTestDeviceVector{T} <: AbstractVector{T}
    data::Vector{T}
end

Base.size(vector::GradientTestDeviceVector) = size(vector.data)
Base.getindex(vector::GradientTestDeviceVector, index::Int) = vector.data[index]
Base.setindex!(vector::GradientTestDeviceVector, value, index::Int) =
    setindex!(vector.data, value, index)
Base.IndexStyle(::Type{<:GradientTestDeviceVector}) = IndexLinear()
Base.similar(vector::GradientTestDeviceVector) =
    GradientTestDeviceVector(similar(vector.data))
MLDataDevices.get_device(::GradientTestDeviceVector) = GradientTestAccelerator()

@testset "LogTarget derivative metadata" begin
    f = (x, p) -> p.shift - sum(abs2, x) / 2
    g! = (G, x, p) -> (G .= -x; G)

    plain = LogTarget(f)
    explicit = LogTarget(f; grad=g!)
    generated = LogTarget(f, ADTypes.AutoForwardDiff())

    @test plain.logdensity === f
    @test plain.adtype isa ADTypes.NoAutoDiff
    @test plain.grad === nothing
    @test explicit.grad === g!
    @test generated.adtype isa ADTypes.AutoForwardDiff
end

@testset "prepared LogTarget retains value binding metadata" begin
    f = (x, p) -> p.shift - sum(abs2, x) / 2
    g! = (G, x, p) -> (G .= -x; G)
    context = (shift=3.0,)
    proposal = TestVectorProposal(zeros(2))
    target = LogTarget(f, ADTypes.AutoForwardDiff(); grad=g!)

    prepared = ISD._prepare_target(target, context, proposal)
    @test prepared isa ISD._PreparedLogTarget
    @test prepared.logdensity === f
    @test prepared.context === context
    @test prepared.adtype === target.adtype
    @test prepared.gradient === g!

    bound = ISD._bind_resolved_target(prepared, [0.25, -0.5])
    @test bound isa ISD._BoundContextualTarget
    @test bound([0.25, -0.5]) == 2.84375

    sampler = prepare_sampler(
        Random.Xoshiro(0x101),
        target,
        context,
        ImportanceSampling(proposal; nsamples=1);
        threaded=false,
    )
    @test sampler.target.logdensity === f
    @test sampler.target.context === context
    @test sampler.target.adtype === target.adtype
    @test sampler.target.gradient === g!
end

@testset "context-free prepared LogTarget retains one derivative source" begin
    f = x -> 1.0 - sum(abs2, x) / 2
    g! = (G, x) -> (G .= -x; G)
    proposal = TestVectorProposal(zeros(2))
    sample = [0.25, -0.5]

    explicit = prepare_sampler(
        Random.Xoshiro(0x102),
        LogTarget(f; grad=g!),
        ImportanceSampling(proposal; nsamples=1);
        threaded=false,
    )
    @test explicit.target.context isa ISD._NoTargetContext
    @test explicit.target.adtype isa ADTypes.NoAutoDiff
    @test explicit.target.gradient === g!
    explicit_value = ISD._bind_resolved_target(explicit.target, sample)
    @test explicit_value isa ISD._BoundContextFreeTarget
    @test explicit_value(sample) == 0.84375

    generated = prepare_sampler(
        Random.Xoshiro(0x103),
        LogTarget(f, ADTypes.AutoForwardDiff()),
        ImportanceSampling(proposal; nsamples=1);
        threaded=false,
    )
    @test generated.target.context isa ISD._NoTargetContext
    @test generated.target.adtype isa ADTypes.AutoForwardDiff
    @test generated.target.gradient === nothing
    generated_value = ISD._bind_resolved_target(generated.target, sample)
    @test generated_value isa ISD._BoundContextFreeTarget
    @test generated_value(sample) == 0.84375
end

@testset "gradient source priority" begin
    f = x -> -sum(abs2, x) / 2
    g! = (G, x) -> (G .= -x; G)

    @test ISD._gradient_source(LogTarget(f; grad=g!)) === :explicit
    @test ISD._gradient_source(FirstOrderLDP()) === :logdensityproblems
    @test ISD._gradient_source(
        LogTarget(f, ADTypes.AutoForwardDiff()),
    ) === :ad
    @test_throws ArgumentError ISD._gradient_source(LogTarget(f))

    value_calls = Ref(0)
    ldp_gradient_calls = Ref(0)
    explicit_calls = Ref(0)
    payload = CountingFirstOrderLDP(value_calls, ldp_gradient_calls)
    prepared = ISD._prepare_target(
        LogTarget(payload; grad=CountingExplicitGradient(explicit_calls)),
        TestVectorProposal(zeros(2)),
    )
    sample = [0.25, -0.5]
    destination = similar(sample)

    bound_gradient = ISD._prepare_bound_gradient(prepared, sample, 1)
    @test bound_gradient isa ISD._BoundInPlaceGradient
    @test ISD._gradient!(destination, bound_gradient, sample) === destination
    @test destination == [-0.25, 0.5]
    @test explicit_calls[] == 1
    @test ldp_gradient_calls[] == 0

    bound_value = ISD._bind_resolved_target(prepared, sample)
    @test bound_value(sample) == -0.15625
    @test value_calls[] == 1
    @test explicit_calls[] == 1
    @test ldp_gradient_calls[] == 0

    inplace_calls = Ref(0)
    outofplace_calls = Ref(0)
    both_forms = ISD._prepare_bound_gradient(
        ISD._prepare_target(
            LogTarget(
                context_free_logtarget;
                grad=BothFormExplicitGradient(inplace_calls, outofplace_calls),
            ),
            TestVectorProposal(zeros(2)),
        ),
        sample,
        1,
    )
    ISD._gradient!(destination, both_forms, sample)
    @test both_forms isa ISD._BoundInPlaceGradient
    @test inplace_calls[] == 1
    @test outofplace_calls[] == 0
end

@testset "explicit gradient shape and positional context" begin
    for T in (Float32, Float64)
        sample = T[0.25, -0.5]
        proposal = TestVectorProposal(copy(sample))
        cases = (
            (
                explicit_inplace_gradient!,
                ISD._NoTargetContext(),
                T[-0.25, 0.5],
                ISD._BoundInPlaceGradient,
            ),
            (
                explicit_inplace_gradient!,
                (T(2),),
                T[-0.5, 1.0],
                ISD._BoundInPlaceGradient,
            ),
            (
                explicit_inplace_affine_gradient!,
                (T(2), T(0.25)),
                T[-0.25, 1.25],
                ISD._BoundInPlaceGradient,
            ),
            (
                explicit_outofplace_gradient,
                ISD._NoTargetContext(),
                T[-0.25, 0.5],
                ISD._BoundOutOfPlaceGradient,
            ),
            (
                explicit_outofplace_context_gradient,
                (T(2),),
                T[-0.5, 1.0],
                ISD._BoundOutOfPlaceGradient,
            ),
            (
                explicit_outofplace_affine_gradient,
                (T(2), T(0.25)),
                T[-0.25, 1.25],
                ISD._BoundOutOfPlaceGradient,
            ),
        )

        for (gradient, context, expected, bound_type) in cases
            target = if context isa ISD._NoTargetContext
                ISD._prepare_target(
                    LogTarget(context_free_logtarget; grad=gradient),
                    proposal,
                )
            else
                ISD._prepare_target(
                    LogTarget(contextual_logtarget; grad=gradient),
                    context,
                    proposal,
                )
            end
            bound_gradient = ISD._prepare_bound_gradient(target, sample, 1)
            destination = similar(sample)
            returned = @inferred ISD._gradient!(
                destination,
                bound_gradient,
                sample,
            )

            @test bound_gradient isa bound_type
            @test bound_gradient.target === target
            @test returned === destination
            @test size(destination) == size(sample)
            @test eltype(destination) === eltype(sample)
            @test destination == expected
        end
    end
end

@testset "CPU-only gradient sources reject accelerator samples" begin
    proposal = TestVectorProposal(zeros(2))
    sample = GradientTestDeviceVector([0.25, -0.5])

    outofplace_target = ISD._prepare_target(
        LogTarget(context_free_logtarget; grad=explicit_outofplace_gradient),
        proposal,
    )
    outofplace_error = try
        ISD._prepare_bound_gradient(outofplace_target, sample, 1)
        nothing
    catch error
        error
    end
    @test outofplace_error isa SamplerDeviceError
    @test outofplace_error.device isa GradientTestAccelerator
    @test outofplace_error.reason === :out_of_place_gradient_cpu_only

    cpu_inplace_target = ISD._prepare_target(
        LogTarget(context_free_logtarget; grad=cpu_vector_inplace_gradient!),
        proposal,
    )
    @test ISD._prepare_bound_gradient(
        cpu_inplace_target,
        [0.25, -0.5],
        1,
    ) isa ISD._BoundInPlaceGradient
    cpu_inplace_error = try
        ISD._prepare_bound_gradient(cpu_inplace_target, sample, 1)
        nothing
    catch error
        error
    end
    @test cpu_inplace_error isa SamplerDeviceError
    @test cpu_inplace_error.device isa GradientTestAccelerator
    @test cpu_inplace_error.reason === :gradient_source_cpu_only

    cpu_outofplace_target = ISD._prepare_target(
        LogTarget(
            context_free_logtarget;
            grad=cpu_vector_outofplace_gradient,
        ),
        proposal,
    )
    @test ISD._prepare_bound_gradient(
        cpu_outofplace_target,
        [0.25, -0.5],
        1,
    ) isa ISD._BoundOutOfPlaceGradient
    cpu_outofplace_error = try
        ISD._prepare_bound_gradient(cpu_outofplace_target, sample, 1)
        nothing
    catch error
        error
    end
    @test cpu_outofplace_error isa SamplerDeviceError
    @test cpu_outofplace_error.device isa GradientTestAccelerator
    @test cpu_outofplace_error.reason === :out_of_place_gradient_cpu_only

    invalid_target = ISD._prepare_target(
        LogTarget(context_free_logtarget; grad=invalid_explicit_gradient),
        proposal,
    )
    @test_throws ArgumentError ISD._prepare_bound_gradient(
        invalid_target,
        sample,
        1,
    )

    ldp_target = ISD._prepare_target(FirstOrderLDP(), proposal)
    ldp_error = try
        ISD._prepare_bound_gradient(ldp_target, sample, 1)
        nothing
    catch error
        error
    end
    @test ldp_error isa SamplerDeviceError
    @test ldp_error.device isa GradientTestAccelerator
    @test ldp_error.reason === :gradient_source_cpu_only

    ad_target = ISD._prepare_target(
        LogTarget(context_free_logtarget, ADTypes.AutoForwardDiff()),
        proposal,
    )
    ad_error = try
        ISD._prepare_bound_gradient(ad_target, sample, 1)
        nothing
    catch error
        error
    end
    @test ad_error isa SamplerDeviceError
    @test ad_error.device isa GradientTestAccelerator
    @test ad_error.reason === :gradient_source_cpu_only

    inplace_target = ISD._prepare_target(
        LogTarget(context_free_logtarget; grad=explicit_inplace_gradient!),
        proposal,
    )
    inplace = ISD._prepare_bound_gradient(inplace_target, sample, 1)
    @test inplace isa ISD._BoundInPlaceGradient
    @test inplace.target === inplace_target
end

@testset "prepared DensityInterface target reports a missing gradient" begin
    proposal = TestVectorProposal(zeros(2))
    prepared = ISD._prepare_target(DerivativeDensityTarget(), proposal)

    error = try
        ISD._prepare_bound_gradient(prepared, zeros(2), 1)
        nothing
    catch caught
        caught
    end

    @test prepared isa ISD._BoundDensityInterfaceTarget
    @test error isa ArgumentError
    @test occursin(
        "LogTarget(logdensity; grad=grad)",
        sprint(showerror, error),
    )
end

import ForwardDiff

@testset "gradient paths are inferred at native precision" begin
    proposal = TestVectorProposal(zeros(2))
    for T in (Float32, Float64)
        sample = T[0.25, -0.5]
        destination = similar(sample)

        explicit_target = ISD._prepare_target(
            LogTarget(context_free_logtarget; grad=explicit_inplace_gradient!),
            proposal,
        )
        explicit = ISD._prepare_bound_gradient(
            explicit_target,
            sample,
            1,
        )
        @test @inferred(ISD._gradient!(destination, explicit, sample)) ===
              destination
        @test destination == T[-0.25, 0.5]

        ldp_target = ISD._prepare_target(FirstOrderLDP(), proposal)
        ldp = ISD._prepare_bound_gradient(ldp_target, sample, 1)
        @test @inferred(ISD._gradient!(destination, ldp, sample)) === destination
        @test destination == T[-0.25, 0.5]

        ad_target = ISD._prepare_target(
            LogTarget(context_free_logtarget, ADTypes.AutoForwardDiff()),
            proposal,
        )
        ad = ISD._prepare_bound_gradient(ad_target, sample, 1)
        @test @inferred(ISD._gradient!(destination, ad, sample)) === destination
        @test destination == T[-0.25, 0.5]

        context = (T(2), T(0.25))
        contextual_ad_target = ISD._prepare_target(
            LogTarget(contextual_logtarget, ADTypes.AutoForwardDiff()),
            context,
            proposal,
        )
        contextual_ad = ISD._prepare_bound_gradient(
            contextual_ad_target,
            sample,
            1,
        )
        @test @inferred(
            ISD._gradient!(destination, contextual_ad, sample)
        ) === destination
        @test destination == T[-0.5, 1.0]
    end
end

@testset "generated gradients use one preparation per default worker" begin
    sample = [0.25, -0.5]
    proposal = TestVectorProposal(zeros(2))
    target = ISD._prepare_target(
        LogTarget(context_free_logtarget, ADTypes.AutoForwardDiff()),
        proposal,
    )
    worker_count = Threads.nthreads(:default)
    bound = ISD._prepare_bound_gradient(target, sample, worker_count)
    destinations = [similar(sample) for _ in 1:worker_count]

    Threads.@threads :static for worker in 1:worker_count
        ISD._gradient!(destinations[worker], bound, sample)
    end

    @test all(==([-0.25, 0.5]), destinations)
    @test isconcretetype(typeof(bound))
    @test all(isconcretetype, fieldtypes(typeof(bound)))
    @test bound.target === target
    @test length(bound.preparation.preparations) == worker_count
    @test all(
        left == right ||
        bound.preparation.preparations[left] !==
        bound.preparation.preparations[right]
        for left in 1:worker_count for right in 1:worker_count
    )
    @test bound.preparation.thread_slots[
        Threads.threadpooltids(:default)
    ] == collect(1:worker_count)
end
