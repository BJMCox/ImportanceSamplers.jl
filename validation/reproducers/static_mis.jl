using DensityInterface
using ImportanceSamplers
using Pkg
using Random
using Test

const STATIC_MIS_CPU_SEED = 0x7374617469636370
const STATIC_MIS_CPU_COMMAND =
    "julia --project=validation validation/reproducers/static_mis.jl"
const STATIC_MIS_DIRECT_SEED = 0x7374617469636d69

include(joinpath(@__DIR__, "..", "static_mis_capabilities.jl"))

@inline function logaddexp(left, right)
    left == -Inf && return right
    right == -Inf && return left
    largest = max(left, right)
    return largest + log1p(exp(min(left, right) - largest))
end

function bank_logdensity(bank, sample; proposal_ids=eachindex(bank.proposals))
    total_mass = sum(bank.masses[proposal_ids])
    value = -Inf
    for proposal_id in proposal_ids
        iszero(bank.masses[proposal_id]) && continue
        term = log(bank.masses[proposal_id] / total_mass) +
               DensityInterface.logdensityof(bank.proposals[proposal_id], sample)
        value = logaddexp(value, term)
    end
    return value
end

function denominator_logdensity(bank, ::Union{StratifiedMixture,RandomMixture}, sample, id)
    return bank_logdensity(bank, sample)
end

function denominator_logdensity(bank, ::StandardMIS, sample, id)
    return DensityInterface.logdensityof(bank.proposals[id], sample)
end

function denominator_logdensity(
    bank,
    scheme::PartialDeterministicMixture,
    sample,
    id,
)
    group = only(filter(group -> id in group, scheme.groups))
    return bank_logdensity(bank, sample; proposal_ids=collect(group))
end

function run_case(bank, scheme, nsamples; seed=STATIC_MIS_CPU_SEED)
    logtarget(sample) = bank_logdensity(bank, sample)
    result = importance_sample(
        Xoshiro(seed),
        logtarget,
        ImportanceSampling(bank; nsamples, mis_scheme=scheme);
        threaded=false,
    )
    expected = [
        logtarget(sample) - denominator_logdensity(bank, scheme, sample, id) for
        (sample, id) in zip(result.samples, result.provenance.proposal_id)
    ]
    tolerance = 512eps(Float64)
    @test length(result) == nsamples
    @test length(result.provenance.proposal_id) == nsamples
    @test all(id -> 1 <= id <= length(bank.proposals), result.provenance.proposal_id)
    @test result.logweights ≈ expected atol = tolerance rtol = 0
    return result
end

function check_full_mixture_identities()
    equal_bank = ProposalBank([
        SphericalGaussian(-1.0, 0.8),
        SphericalGaussian(1.0, 1.2),
    ])
    equal = run_case(equal_bank, StratifiedMixture(), 8_192)
    @test count(==(1), equal.provenance.proposal_id) == 4_096
    @test count(==(2), equal.provenance.proposal_id) == 4_096
    @test maximum(abs, equal.logweights) <= 512eps(Float64)
    @test abs(lognormalizer(equal)) <= 512eps(Float64)

    unequal_bank = ProposalBank([
        SphericalGaussian(-2.0, 0.7),
        SphericalGaussian(0.0, 1.0),
        SphericalGaussian(2.0, 1.3),
        SphericalGaussian(100.0, 1.0),
    ], [1, 3, 2, 0])
    integral = run_case(unequal_bank, StratifiedMixture(), 12_000; seed=STATIC_MIS_CPU_SEED + 1)
    @test [count(==(id), integral.provenance.proposal_id) for id in 1:4] ==
          [2_000, 6_000, 4_000, 0]

    nonintegral = run_case(
        unequal_bank,
        StratifiedMixture(),
        10_003;
        seed=STATIC_MIS_CPU_SEED + 2,
    )
    counts = [count(==(id), nonintegral.provenance.proposal_id) for id in 1:4]
    @test static_mis_stratified_counts_within_bound(counts, unequal_bank.masses, 10_003)
    random = run_case(
        unequal_bank,
        RandomMixture(),
        10_003;
        seed=STATIC_MIS_CPU_SEED + 3,
    )
    @test maximum(abs, random.logweights) <= 512eps(Float64)
    @test 4 ∉ random.provenance.proposal_id
    return nothing
end

function check_stratified_count_bound()
    counts = [1_668, 5_002, 0, 3_333]
    masses = Float32[1, 3, 0, 2]
    masses ./= sum(masses)
    deviations = abs.(counts .- 10_003 .* masses)
    @test maximum(deviations) > 1
    @test static_mis_stratified_counts_within_bound(counts, masses, 10_003)
    return nothing
end

function check_generating_and_partial_identities()
    identical = ProposalBank([
        SphericalGaussian(0.25, 1.1),
        SphericalGaussian(0.25, 1.1),
        SphericalGaussian(0.25, 1.1),
    ], [1, 2, 3])
    standard = run_case(
        identical,
        StandardMIS(),
        10_003;
        seed=STATIC_MIS_CPU_SEED + 4,
    )
    @test maximum(abs, standard.logweights) <= 512eps(Float64)
    @test abs(lognormalizer(standard)) <= 512eps(Float64)

    partial_bank = ProposalBank([
        SphericalGaussian(-2.0, 0.7),
        SphericalGaussian(-0.5, 1.0),
        SphericalGaussian(1.0, 0.8),
        SphericalGaussian(2.5, 1.2),
        SphericalGaussian(100.0, 1.0),
    ], [1, 3, 2, 4, 0])
    scheme = PartialDeterministicMixture(((1, 2), (3, 4), (5,)))
    partial = run_case(
        partial_bank,
        scheme,
        10_003;
        seed=STATIC_MIS_CPU_SEED + 5,
    )
    @test 5 ∉ partial.provenance.proposal_id
    @test all(isfinite, partial.logweights)
    return nothing
end

function check_scalar_direct_fixture()
    row = only(filter(
        row -> !isnothing(row.direct) && row.direct.sample_layout === :scalar,
        STATIC_MIS_CAPABILITY_ROWS,
    ))
    for (case_index, scheme) in enumerate(STATIC_MIS_COMPLETE_SCHEMES)
        bank = row.factory(Float32)
        result = importance_sample(
            Xoshiro(STATIC_MIS_DIRECT_SEED + UInt(case_index)),
            sample -> bank_logdensity(bank, sample),
            ImportanceSampling(bank; nsamples=10_003, mis_scheme=scheme.value);
            threaded=false,
        )
        weights = exp.(result.logweights)
        normalizer = sum(weights) / length(weights)
        variance = sum(weight -> abs2(weight - normalizer), weights) / length(weights)
        lognormalizer_se = sqrt(variance / length(weights)) / normalizer
        @test abs(lognormalizer(result)) <= 7lognormalizer_se
    end
    return nothing
end

function check_direct_sample_shapes()
    rows = filter(row -> !isnothing(row.direct), STATIC_MIS_CAPABILITY_ROWS)
    for (case_index, row) in enumerate(rows)
        bank = row.factory(Float32)
        location = getfield(first(bank.proposals), :location)
        expected_mean = zeros(Float32, location isa Real ? 1 : length(location))
        result = importance_sample(
            Xoshiro(STATIC_MIS_DIRECT_SEED + UInt(100 + case_index)),
            sample -> bank_logdensity(bank, sample),
            ImportanceSampling(bank; nsamples=17, mis_scheme=StratifiedMixture());
            threaded=false,
        )
        @test size(result.samples) == static_mis_expected_sample_size(row, expected_mean, 17)
    end
    return nothing
end

function package_versions()
    wanted = Set(("DensityInterface", "ImportanceSamplers", "LogExpFunctions"))
    return sort!(
        [
            (dependency.name, something(dependency.version, "unversioned"))
            for dependency in values(Pkg.dependencies())
            if dependency.name in wanted
        ];
        by=first,
    )
end

function main()
    @testset "static MIS analytic CPU reproducer" begin
        check_full_mixture_identities()
        check_stratified_count_bound()
        check_generating_and_partial_identities()
        check_scalar_direct_fixture()
        check_direct_sample_shapes()
    end
    return (
        command=STATIC_MIS_CPU_COMMAND,
        seed=STATIC_MIS_CPU_SEED,
        julia=VERSION,
        packages=package_versions(),
        status=:passed,
    )
end

main()
