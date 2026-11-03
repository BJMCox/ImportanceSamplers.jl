using Test
using ImportanceSamplers
import DensityInterface
import LinearAlgebra
import Random
import LogDensityProblems

const IS = ImportanceSamplers

struct NamedProductDensity end
LogDensityProblems.capabilities(::Type{NamedProductDensity}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.dimension(::NamedProductDensity) = 2
LogDensityProblems.logdensity(::NamedProductDensity, x) = -(abs2(x.a) + abs2(x.b)) / 2

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
        @test maximum(abs, result.logweights) <= 16eps(eltype(result.logweights))
        @test abs(lognormalizer(result)) <= 16eps(eltype(result.logweights))
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
end

@testset "native products preserve joint laws and named shapes" begin
    scalar = SphericalGaussian(0.5f0, 1.25f0)
    singleton = DiagonalGaussian(Float32[1], Float32[0.75])
    products = map((false, true)) do correlated
        vector = correlated ?
            FactorGaussian(Float32[-1, 2], Float32[1 0; 0.4 2]) :
            DiagonalGaussian(Float32[-1, 2], Float32[1, 2])
        product = ProductProposal((; scalar, vector, singleton))
        joint = correlated ?
            FactorGaussian(Float32[0.5, -1, 2, 1],
                Float32[1.25 0 0 0; 0 1 0 0; 0 0.4 2 0; 0 0 0 0.75]) :
            DiagonalGaussian(Float32[0.5, -1, 2, 1], Float32[1.25, 1, 2, 0.75])
        target(x) = DensityInterface.logdensityof(product, x)
        sampler = prepare_sampler(Random.Xoshiro(81), target,
            ImportanceSampling(product; nsamples=64); threaded=false)
        actual = importance_sample!(sampler)
        expected = importance_sample(Random.Xoshiro(81), target,
            ImportanceSampling(joint; nsamples=64);
            transform=(scalar=1, vector=2:3, singleton=4:4), threaded=false)
        correlated || @test actual.samples == expected.samples
        @test map(size, actual.samples) == (scalar=(64,), vector=(2, 64), singleton=(1, 64))
        @test maximum(abs, actual.logweights) <= 32eps(Float32)
        @test eltype(actual.logweights) === Float32
        saved = deepcopy(actual.samples)
        importance_sample!(sampler)
        @test actual.samples == saved
        product
    end

    # The unused first slot has no matching schema and must not renumber active IDs.
    bank = ProposalBank([ProductProposal((unused=FixedProductBlock(0f0, 0f0),)),
        products...], Float32[0, 3, 7])
    target(x) = -(abs2(x.scalar) + sum(abs2, x.vector) + sum(abs2, x.singleton)) / 2f0
    result = importance_sample(Random.Xoshiro(82), target,
        ImportanceSampling(bank; nsamples=128, mis_scheme=RandomMixture()); threaded=false)
    expected = map(1:length(result)) do index
        x = (scalar=result.samples.scalar[index], vector=result.samples.vector[:, index],
            singleton=result.samples.singleton[:, index])
        density = sum(bank.masses[j] * exp(DensityInterface.logdensityof(bank.proposals[j], x))
            for j in 2:3)
        target(x) - log(density)
    end
    @test result.logweights ≈ expected rtol=32eps(Float32)
    @test Set(result.provenance.proposal_id) == Set((2, 3))

    named_proposal = ProductProposal((a=SphericalGaussian(0.0, 1.0), b=SphericalGaussian(0.0, 1.0)))
    named_result = importance_sample(Random.Xoshiro(83), NamedProductDensity(),
        ImportanceSampling(named_proposal; nsamples=16); threaded=false)
    @test all(isapprox(log(2pi); atol=8eps()), named_result.logweights)
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
        offset=4,
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

    identity = TransformedProposal(base, (scalar=1, singleton=2:2, tail=3:4))
    explicit = TransformedProposal(base, (scalar=1=>IdentityTransform(),
        singleton=2:2=>IdentityTransform(), tail=3:4=>IdentityTransform()))
    sample = rand(Random.Xoshiro(27), identity)
    @test sample == rand(Random.Xoshiro(27), explicit)
    @test DensityInterface.logdensityof(identity, sample) ≈
          DensityInterface.logdensityof(base, [sample.scalar; sample.singleton; sample.tail])

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
        offset=4,
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
