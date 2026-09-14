using Test
using ImportanceSamplers
import DensityInterface
import LinearAlgebra
import Random

const IS = ImportanceSamplers

struct FixedProductBlock{T}
    coordinate::T
    logdensity::T
end

Random.rand(::Random.AbstractRNG, proposal::FixedProductBlock) = proposal.coordinate
DensityInterface.logdensityof(proposal::FixedProductBlock, sample) = proposal.logdensity

struct SplitDensityTransformBase{C,D}
    coordinate::C
    logdensity::D
end

Random.rand(::Random.AbstractRNG, proposal::SplitDensityTransformBase) =
    proposal.coordinate
DensityInterface.logdensityof(proposal::SplitDensityTransformBase, sample) =
    proposal.logdensity

function _standard_simplex_logdensity(weights::AbstractVector{T}) where {T}
    logweights = log.(weights)
    mean_logweight = sum(logweights) / T(3)
    inverse_root_dimension = inv(sqrt(T(3)))
    transpose_coefficient = inverse_root_dimension / (one(T) - inverse_root_dimension)
    last_centered = logweights[3] - mean_logweight
    z1 = logweights[1] - mean_logweight + transpose_coefficient * last_centered
    z2 = logweights[2] - mean_logweight + transpose_coefficient * last_centered
    base_logdensity = -log(T(2) * T(pi)) - T(0.5) * (abs2(z1) + abs2(z2))
    logabsjac = T(0.5) * log(T(3)) + sum(logweights)
    return base_logdensity - logabsjac
end

function _independent_structured_logdensity(sample::NamedTuple)
    T = typeof(sample.rate)
    return _standard_simplex_logdensity(sample.weights) -
           T(0.5) * abs2(log(sample.rate)) - log(sample.rate) -
           T(0.5) * log(T(2) * T(pi)) -
           T(0.5) * abs2(sample.offset) - T(0.5) * log(T(2) * T(pi))
end

function _dependent_structured_logdensity(sample::NamedTuple, factor)
    T = eltype(factor)
    logweights = log.(sample.weights)
    mean_logweight = sum(logweights) / T(3)
    inverse_root_dimension = inv(sqrt(T(3)))
    transpose_coefficient = inverse_root_dimension / (one(T) - inverse_root_dimension)
    last_centered = logweights[3] - mean_logweight
    coordinates = T[
        logweights[1] - mean_logweight + transpose_coefficient * last_centered,
        logweights[2] - mean_logweight + transpose_coefficient * last_centered,
        log(sample.rate),
        sample.offset,
    ]
    standardized = LinearAlgebra.LowerTriangular(factor) \ coordinates
    base_logdensity =
        -T(2) * log(T(2) * T(pi)) -
        sum(log, LinearAlgebra.diag(factor)) -
        T(0.5) * sum(abs2, standardized)
    logabsjac = T(0.5) * log(T(3)) + sum(logweights) + log(sample.rate)
    return base_logdensity - logabsjac
end

@testset "native proposals implement DensityInterface" begin
    gaussian = SphericalGaussian(0.0, 1.0)
    product = ProductProposal((left=gaussian, right=SphericalGaussian(1.0, 2.0)))
    transformed = TransformedProposal(gaussian, IdentityTransform())
    cases = (
        (gaussian, 0.25),
        (product, (left=0.25, right=-0.5)),
        (transformed, 0.25),
    )

    for (index, (proposal, sample)) in enumerate(cases)
        expected = DensityInterface.logdensityof(proposal, sample)
        @test DensityInterface.DensityKind(proposal) isa DensityInterface.HasDensity
        @test DensityInterface.logdensityof(proposal)(sample) == expected
        DensityInterface.test_density_interface(proposal, sample, expected)

        result = importance_sample(
            Random.Xoshiro(0x7400 + index),
            proposal,
            ImportanceSampling(proposal; nsamples=8);
            threaded=false,
        )
        @test result.logweights == zeros(eltype(result.logweights), 8)
        @test lognormalizer(result) == zero(eltype(result.logweights))
    end
end

@testset "product proposal CPU semantics" begin
    blocks = (
        first=SphericalGaussian(1.0, 0.5),
        second=SphericalGaussian(-2.0, 1.25),
        third=SphericalGaussian(0.25, 2.0),
    )
    proposal = @inferred ProductProposal(blocks)
    @test isconcretetype(typeof(proposal))
    @test_throws ArgumentError ProductProposal(NamedTuple())
    abstract_blocks = NamedTuple{(:only,),Tuple{Any}}((blocks.first,))
    @test_throws ArgumentError ProductProposal(abstract_blocks)

    actual_rng = Random.Xoshiro(0x7201)
    expected_rng = copy(actual_rng)
    actual = @inferred rand(actual_rng, proposal)
    expected = (
        first=rand(expected_rng, blocks.first),
        second=rand(expected_rng, blocks.second),
        third=rand(expected_rng, blocks.third),
    )
    @test actual == expected
    @test rand(actual_rng) == rand(expected_rng)

    logdensity = @inferred DensityInterface.logdensityof(proposal, actual)
    expected_logdensity = sum(
        DensityInterface.logdensityof(block, value) for
        (block, value) in zip(values(blocks), values(actual))
    )
    @test logdensity == expected_logdensity
    @test_throws ArgumentError DensityInterface.logdensityof(
        proposal,
        (first=actual.first, third=actual.third, second=actual.second),
    )

    rand(actual_rng, proposal)
    DensityInterface.logdensityof(proposal, actual)
    @test (@allocated rand(actual_rng, proposal)) == 0
    @test (@allocated DensityInterface.logdensityof(proposal, actual)) == 0
end

@testset "transformed support density type stability" begin
    split_base = SplitDensityTransformBase(0.0f0, 0.0)
    split = @inferred TransformedProposal(split_base, PositiveTransform())
    split_valid = @inferred DensityInterface.logdensityof(split, 1.0f0)
    split_outside = @inferred DensityInterface.logdensityof(split, 0.0f0)
    @test split_valid isa Float64
    @test split_outside isa Float64
    @test split_outside === -Inf

    mixed_base = ProductProposal((
        weights=SphericalGaussian(zeros(Float32, 2), 1.0f0),
        rate=SphericalGaussian(0.0, 1.0),
        offset=SphericalGaussian(0.0f0, 1.0f0),
    ))
    mixed = @inferred TransformedProposal(
        mixed_base,
        (weights=SimplexTransform(3), rate=PositiveTransform()),
    )
    mixed_valid_value = (
        weights=fill(inv(Float32(3)), 3),
        rate=1.0,
        offset=0.0f0,
    )
    mixed_outside_value = merge(mixed_valid_value, (rate=0.0,))
    mixed_valid = @inferred DensityInterface.logdensityof(mixed, mixed_valid_value)
    mixed_outside = @inferred DensityInterface.logdensityof(mixed, mixed_outside_value)
    @test mixed_valid isa Float64
    @test mixed_outside isa Float64
    @test mixed_outside === -Inf
end

@testset "named partial transformed proposal" begin
    base = ProductProposal((
        weights=SphericalGaussian(zeros(2), 1.0),
        rate=SphericalGaussian(0.0, 1.0),
        offset=SphericalGaussian(0.0, 1.0),
    ))
    proposal = @inferred TransformedProposal(
        base,
        (weights=SimplexTransform(3), rate=PositiveTransform()),
    )
    @test isconcretetype(typeof(getfield(proposal, :transform)))
    @test keys(getfield(getfield(proposal, :transform), :blocks)) == keys(base.blocks)
    @test getfield(proposal, :transform).blocks.offset.transform isa IdentityTransform

    actual_rng = Random.Xoshiro(0x7202)
    expected_rng = copy(actual_rng)
    raw = rand(expected_rng, base)
    expected = (
        weights=IS._transform_with_logjac(SimplexTransform(3), raw.weights)[1],
        rate=IS._transform_with_logjac(PositiveTransform(), raw.rate)[1],
        offset=raw.offset,
    )
    actual = @inferred rand(actual_rng, proposal)
    @test actual == expected
    @test keys(actual) == (:weights, :rate, :offset)
    @test rand(actual_rng) == rand(expected_rng)

    uniform = fill(1 / 3, 3)
    logical_origin = (weights=uniform, rate=1.0, offset=0.0)
    at_origin = @inferred DensityInterface.logdensityof(proposal, logical_origin)
    expected_origin = -2log(2pi) + 2.5log(3)
    @test at_origin ≈ expected_origin atol = 16eps()
    @test at_origin ≈ _independent_structured_logdensity(logical_origin) atol = 16eps()

    outside = (weights=[0.0, 0.5, 0.5], rate=1.0, offset=0.0)
    @test DensityInterface.logdensityof(proposal, outside) === -Inf
    @test_throws ArgumentError TransformedProposal(
        base,
        (weights=SimplexTransform(3), unknown=PositiveTransform()),
    )
    @test_throws ArgumentError TransformedProposal(
        SphericalGaussian(zeros(3), 1.0),
        (rate=PositiveTransform(),),
    )

    failing = TransformedProposal(
        ProductProposal((
            weights=SphericalGaussian(zeros(2), 1.0),
            rate=FixedProductBlock(floatmax(Float64), 0.0),
            offset=SphericalGaussian(0.0, 1.0),
        )),
        (rate=PositiveTransform(),),
    )
    generated_error = try
        rand(Random.Xoshiro(0x7206), failing)
        nothing
    catch error
        error
    end
    @test generated_error isa InvalidTransformError
    @test generated_error.reason === :nonfinite_output
    @test generated_error.location === :rate
end

@testset "dependent flat selector transformed proposal" begin
    factor = [
        1.0 0.0 0.0 0.0
        0.3 1.2 0.0 0.0
        -0.2 0.4 0.8 0.0
        0.1 -0.3 0.25 1.5
    ]
    base = FactorGaussian(zeros(4), factor)
    specification = (
        weights=(1:2 => SimplexTransform(3)),
        rate=(3 => PositiveTransform()),
        offset=(4 => IdentityTransform()),
    )
    proposal = @inferred TransformedProposal(base, specification)
    layout = getfield(proposal, :transform)
    @test isconcretetype(typeof(layout))
    @test keys(getfield(layout, :blocks)) == (:weights, :rate, :offset)

    actual_rng = Random.Xoshiro(0x7203)
    expected_rng = copy(actual_rng)
    raw = rand(expected_rng, base)
    expected = (
        weights=IS._transform_with_logjac(SimplexTransform(3), raw[1:2])[1],
        rate=IS._transform_with_logjac(PositiveTransform(), raw[3])[1],
        offset=raw[4],
    )
    actual = @inferred rand(actual_rng, proposal)
    @test actual == expected
    @test actual.weights isa Vector{Float64}
    @test actual.rate isa Float64
    @test actual.offset isa Float64
    @test rand(actual_rng) == rand(expected_rng)

    logical_origin = (weights=fill(1 / 3, 3), rate=1.0, offset=0.0)
    at_origin = @inferred DensityInterface.logdensityof(proposal, logical_origin)
    @test at_origin ≈ _dependent_structured_logdensity(logical_origin, factor) atol = 32eps()
    @test DensityInterface.logdensityof(
        proposal,
        (weights=[0.2, 0.3, 0.6], rate=1.0, offset=0.0),
    ) === -Inf

    @test_throws ArgumentError TransformedProposal(
        base,
        (left=(1:2 => IdentityTransform()), right=(2:4 => IdentityTransform())),
    )
    @test_throws ArgumentError TransformedProposal(
        base,
        (left=(1:2 => IdentityTransform()), right=(4 => IdentityTransform())),
    )
    @test_throws ArgumentError TransformedProposal(
        base,
        (all=(1:5 => IdentityTransform()),),
    )
    @test_throws ArgumentError TransformedProposal(
        base,
        (odd=(1:2:3 => IdentityTransform()), even=(2:2:4 => IdentityTransform())),
    )
    @test_throws ArgumentError TransformedProposal(
        base,
        (bad=(true => IdentityTransform()), rest=(2:4 => IdentityTransform())),
    )
end

@testset "known transform shape and precision are rejected before drawing" begin
    flat_base = FactorGaussian(zeros(4), Matrix{Float64}(LinearAlgebra.I, 4, 4))
    direct_vector_base = SphericalGaussian(zeros(2), 1.0)
    scalar_transform_builders = map(
        transform -> () -> TransformedProposal(
            flat_base,
            (
                invalid=(1:2 => transform),
                remainder=(3:4 => IdentityTransform()),
            ),
        ),
        (
            PositiveTransform(),
            SoftplusTransform(),
            IntervalTransform(0.0, nothing),
        ),
    )
    for transform in (
        PositiveTransform(),
        SoftplusTransform(),
        IntervalTransform(0.0, nothing),
    )
        @test_throws DimensionMismatch TransformedProposal(direct_vector_base, transform)
    end
    @test_throws DimensionMismatch TransformedProposal(
        SphericalGaussian(zeros(3), 1.0),
        SimplexTransform(3),
    )

    flat_simplex_builder = () -> TransformedProposal(
        flat_base,
        (
            invalid=(1 => SimplexTransform(3)),
            remainder=(2:4 => IdentityTransform()),
        ),
    )
    named_simplex_builder = () -> TransformedProposal(
        ProductProposal((
            invalid=SphericalGaussian(zeros(3), 1.0),
            remainder=SphericalGaussian(0.0, 1.0),
        )),
        (invalid=SimplexTransform(3),),
    )

    for build_invalid in (
        scalar_transform_builders...,
        flat_simplex_builder,
        named_simplex_builder,
    )
        @test_throws DimensionMismatch build_invalid()

        actual_rng = Random.Xoshiro(0x7207)
        expected_rng = copy(actual_rng)
        error = try
            rand(actual_rng, build_invalid())
            nothing
        catch caught
            caught
        end
        @test error isa DimensionMismatch
        @test rand(actual_rng) == rand(expected_rng)
    end

    mixed_interval_builders = (
        () -> TransformedProposal(
            SphericalGaussian(0.0f0, 1.0f0),
            IntervalTransform(0.0, 1.0),
        ),
        () -> TransformedProposal(
            SphericalGaussian(0.0, 1.0),
            IntervalTransform(0.0f0, 1.0f0),
        ),
    )
    for build_invalid in mixed_interval_builders
        actual_rng = Random.Xoshiro(0x7208)
        expected_rng = copy(actual_rng)
        error = try
            rand(actual_rng, build_invalid())
            nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
        @test occursin("interval endpoint type must match", sprint(showerror, error))
        @test rand(actual_rng) == rand(expected_rng)
    end
end

@testset "structured transformed normalizer identities" begin
    named = TransformedProposal(
        ProductProposal((
            weights=SphericalGaussian(zeros(2), 1.0),
            rate=SphericalGaussian(0.0, 1.0),
            offset=SphericalGaussian(0.0, 1.0),
        )),
        (weights=SimplexTransform(3), rate=PositiveTransform()),
    )
    named_result = importance_sample(
        Random.Xoshiro(0x7204),
        _independent_structured_logdensity,
        ImportanceSampling(named; nsamples=128);
        threaded=false,
    )
    @test maximum(abs, named_result.logweights) <= 64eps()
    @test abs(lognormalizer(named_result)) <= 64eps()

    factor = [
        1.0 0.0 0.0 0.0
        0.3 1.2 0.0 0.0
        -0.2 0.4 0.8 0.0
        0.1 -0.3 0.25 1.5
    ]
    flat = TransformedProposal(
        FactorGaussian(zeros(4), factor),
        (
            weights=(1:2 => SimplexTransform(3)),
            rate=(3 => PositiveTransform()),
            offset=(4 => IdentityTransform()),
        ),
    )
    flat_target(sample) = _dependent_structured_logdensity(sample, factor)
    flat_result = importance_sample(
        Random.Xoshiro(0x7205),
        flat_target,
        ImportanceSampling(flat; nsamples=128);
        threaded=false,
    )
    @test maximum(abs, flat_result.logweights) <= 256eps()
    @test abs(lognormalizer(flat_result)) <= 256eps()
end

@testset "named target transform preserves the normalized chart measure" begin
    factor = [
        1.0 0.0 0.0 0.0
        0.3 1.2 0.0 0.0
        -0.2 0.4 0.8 0.0
        0.1 -0.3 0.25 1.5
    ]
    proposal = FactorGaussian(zeros(4), factor)
    layout = (
        weights=(1:2 => SimplexTransform(3)),
        rate=(3 => PositiveTransform()),
        offset=(4 => IdentityTransform()),
    )
    target(sample) = _dependent_structured_logdensity(sample, factor)
    result = importance_sample(
        Random.Xoshiro(0x7209),
        target,
        ImportanceSampling(proposal; nsamples=128);
        transform=layout,
        threaded=false,
    )

    @test size(result.samples.weights) == (3, 128)
    @test length(result.samples.rate) == 128
    @test length(result.samples.offset) == 128
    @test maximum(abs, result.logweights) <= 256eps()
    @test abs(lognormalizer(result)) <= 256eps()
end
