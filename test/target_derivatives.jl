import ADTypes
import Random

const ISD = ImportanceSamplers

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
