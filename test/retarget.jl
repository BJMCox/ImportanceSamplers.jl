using Test
using ImportanceSamplers
import DensityInterface
import MLDataDevices
import Random

function retarget_logdensity(sample, p)
    displacement = sample - p.location
    radius = displacement isa Number ? abs2(displacement) : sum(abs2, displacement)
    return p.offset - radius / 2
end

function retarget_gradient!(destination, sample, p)
    destination .= p.location .- sample
    return destination
end

retarget_array_logdensity(sample, p) =
    p.offset[1] - abs2(sample - p.location[1]) / 2
retarget_origin_logdensity(sample) = -abs2(sample) / 2
retarget_shifted_logdensity(sample) = -abs2(sample - 0.5) / 2

mutable struct CountingRetargetControl
    calls::Int
end

mutable struct RetargetNoncopyableRNG{R<:Random.AbstractRNG} <: Random.AbstractRNG
    rng::R
end

Base.copy(rng::RetargetNoncopyableRNG) = rng
Random.rand!(
    rng::RetargetNoncopyableRNG,
    destination::AbstractArray{T},
) where {T} =
    Random.rand!(rng.rng, destination)
Random.randn!(
    rng::RetargetNoncopyableRNG,
    destination::AbstractArray{T},
) where {T} =
    Random.randn!(rng.rng, destination)

function (control::CountingRetargetControl)(round)
    control.calls += 1
    return 0.1round
end

function assert_same_proposal(actual, expected, points)
    @test actual.location == expected.location
    for point in points
        @test DensityInterface.logdensityof(actual, point) ==
              DensityInterface.logdensityof(expected, point)
    end
end

function assert_same_proposal(actual::ProposalBank, expected::ProposalBank, points)
    @test actual.masses == expected.masses
    @test length(actual.proposals) == length(expected.proposals)
    for (left, right) in zip(actual.proposals, expected.proposals)
        assert_same_proposal(left, right, points)
    end
end

function assert_same_result(actual, expected)
    @test actual.samples == expected.samples
    @test actual.logweights == expected.logweights
    @test actual.provenance == expected.provenance
    for name in propertynames(actual.diagnostics)
        name === :transfers && continue
        @test getproperty(actual.diagnostics, name) ==
              getproperty(expected.diagnostics, name)
    end
end

function assert_retarget_equivalent(
    algorithm,
    rebuild,
    old_target,
    old_context,
    new_target,
    new_context,
    points,
    seed,
)
    source = prepare_sampler(
        Random.Xoshiro(seed),
        old_target,
        old_context,
        algorithm;
        threaded=false,
    )
    control = prepare_sampler(
        Random.Xoshiro(seed),
        old_target,
        old_context,
        algorithm;
        threaded=false,
    )
    assert_same_result(importance_sample!(source), importance_sample!(control))
    committed = current_proposal(source)

    rng = Random.Xoshiro(seed + 1)
    expected = prepare_sampler(
        copy(rng),
        new_target,
        new_context,
        rebuild(committed);
        threaded=false,
    )
    retargeted = @inferred retarget(rng, source, new_target, new_context)

    assert_same_proposal(current_proposal(retargeted), committed, points)
    assert_same_result(importance_sample!(source), importance_sample!(control))
    assert_same_proposal(current_proposal(source), current_proposal(control), points)
    assert_same_result(importance_sample!(retargeted), importance_sample!(expected))
    assert_same_proposal(
        current_proposal(retargeted),
        current_proposal(expected),
        points,
    )
end

@testset "retarget reuses committed adaptive proposal state" begin
    dm_algorithm = DeterministicMixturePMC(
        ProposalBank([
            SphericalGaussian(-2.0, 1.0),
            SphericalGaussian(2.0, 1.0),
        ]);
        rounds=2,
        round_size=8,
        resampling=LocalResampling(),
    )
    assert_retarget_equivalent(
        dm_algorithm,
        bank -> DeterministicMixturePMC(
            bank;
            rounds=2,
            round_size=8,
            resampling=LocalResampling(),
        ),
        retarget_logdensity,
        (offset=0.0, location=-0.5),
        retarget_logdensity,
        (offset=1.25, location=0.75),
        (-1.0, 1.0),
        0x7910,
    )

    amis_algorithm = AMIS(SphericalGaussian(0.0, 2.0); rounds=2, round_size=8)
    assert_retarget_equivalent(
        amis_algorithm,
        proposal -> AMIS(proposal; rounds=2, round_size=8),
        retarget_logdensity,
        (offset=0.0, location=-1.0),
        retarget_logdensity,
        (offset=1.25, location=0.75),
        (-1.0, 1.0),
        0x7920,
    )

    gramis_algorithm = FirstOrderGRAMIS(
        ProposalBank([
            SphericalGaussian([-2.0], 1.0),
            SphericalGaussian([2.0], 1.0),
        ]);
        rounds=1,
        round_size=8,
        repulsion_strength=0.1,
        covariance_ess_threshold=3,
        covariance_rate=0.6,
        covariance_regularization=1e-5,
        tempering_tolerance=1e-3,
        tempering_max_iterations=8,
        repulsion_softening=0.7,
        max_backtracking_trials=10,
    )
    gramis_target = LogTarget(retarget_logdensity; grad=retarget_gradient!)
    assert_retarget_equivalent(
        gramis_algorithm,
        bank -> FirstOrderGRAMIS(
            bank;
            rounds=1,
            round_size=8,
            repulsion_strength=0.1,
            covariance_ess_threshold=3,
            covariance_rate=0.6,
            covariance_regularization=1e-5,
            tempering_tolerance=1e-3,
            tempering_max_iterations=8,
            repulsion_softening=0.7,
            max_backtracking_trials=10,
        ),
        gramis_target,
        (offset=0.0, location=[-0.5]),
        gramis_target,
        (offset=1.25, location=[0.75]),
        ([-1.0], [1.0]),
        0x7930,
    )
end


@testset "retarget preserves source ownership" begin
    bank = ProposalBank([
        SphericalGaussian([-1.0], 1.0),
        SphericalGaussian([1.0], 1.0),
    ])
    control = CountingRetargetControl(0)
    algorithm = FirstOrderGRAMIS(
        bank;
        rounds=2,
        round_size=8,
        repulsion_strength=control,
    )
    source = prepare_sampler(
        Random.Xoshiro(0x7950),
        LogTarget(retarget_logdensity; grad=retarget_gradient!),
        (offset=0.0, location=[0.0]),
        algorithm;
        threaded=false,
    )
    @test control.calls == 2
    retarget(
        Random.Xoshiro(0x7951),
        source,
        LogTarget(retarget_logdensity; grad=retarget_gradient!),
        (offset=1.0, location=[0.5]),
    )
    @test control.calls == 2

    rng = Random.Xoshiro(0x7952)
    dm_source = prepare_sampler(
        rng,
        retarget_logdensity,
        (offset=0.0, location=0.0),
        DeterministicMixturePMC(
            ProposalBank([
                SphericalGaussian(-1.0, 1.0),
                SphericalGaussian(1.0, 1.0),
            ]);
            rounds=1,
            round_size=8,
        );
        threaded=false,
    )
    @test_throws ArgumentError retarget(
        rng,
        dm_source,
        retarget_logdensity,
        (offset=1.0, location=0.5),
    )

    context_free_source = prepare_sampler(
        Random.Xoshiro(0x7953),
        retarget_origin_logdensity,
        AMIS(SphericalGaussian(0.0, 1.0); rounds=1, round_size=8);
        threaded=false,
    )
    importance_sample!(context_free_source)
    context_free = @inferred retarget(
        Random.Xoshiro(0x7954),
        context_free_source,
        retarget_shifted_logdensity,
    )
    @test length(importance_sample!(context_free)) == 8
end

@testset "retarget preserves the prepared device policy" begin
    device = MLDataDevices.cpu_device(Float32)
    algorithm = DeterministicMixturePMC(
        ProposalBank([
            SphericalGaussian(-1.0, 1.0),
            SphericalGaussian(1.0, 1.0),
        ]);
        rounds=1,
        round_size=8,
    )
    source = prepare_sampler(
        Random.Xoshiro(0x7940),
        retarget_array_logdensity,
        (offset=[0.0], location=[0.0]),
        algorithm;
        threaded=false,
    ) |> device
    importance_sample!(source)
    sampler = @inferred retarget(
        RetargetNoncopyableRNG(Random.Xoshiro(0x7941)),
        source,
        retarget_array_logdensity,
        (offset=[1.0], location=[0.5]),
    )
    result = @inferred importance_sample!(sampler)

    @test eltype(result.samples) === Float32
    @test eltype(result.logweights) === Float32
end
