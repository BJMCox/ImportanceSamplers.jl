using Test
using ImportanceSamplers
import DensityInterface
import Random

include("support/static_mis.jl")

const IS = ImportanceSamplers

function caught_static_mis_failure(f)
    try
        f()
    catch error
        return error
    end
    return nothing
end

@testset "proposal-bank configuration" begin
    proposals = [
        SphericalGaussian(-1.0, 1.0),
        SphericalGaussian(1.0, 1.0),
    ]

    equal_bank = @inferred ProposalBank(proposals)
    weighted_bank = @inferred ProposalBank(proposals, [1, 3])

    @test ProposalBank <: AbstractProposalPopulation
    @test equal_bank.proposals !== proposals
    @test equal_bank.proposals == proposals
    @test equal_bank.masses == [0.5, 0.5]
    @test weighted_bank.masses == [0.25, 0.75]

    largest = floatmax(Float64)
    near_overflow = ProposalBank(proposals, [largest, largest / 2])
    @test near_overflow.masses ≈ [2 / 3, 1 / 3]

    smallest = nextfloat(0.0)
    subnormal = ProposalBank(proposals, [smallest, 2smallest])
    @test subnormal.masses ≈ [1 / 3, 2 / 3]
    @test all(>(0), subnormal.masses)

    integer_overflow = ProposalBank(proposals, fill(typemax(Int), 2))
    @test integer_overflow.masses == [0.5, 0.5]

    low_precision_count = 70_000
    low_precision = ProposalBank(
        fill(first(proposals), low_precision_count),
        fill(Float16(1), low_precision_count),
    )
    @test eltype(low_precision.masses) === Float32
    @test all(>(0), low_precision.masses)
    @test isapprox(sum(low_precision.masses), 1; rtol=4eps(Float32))
    @test eltype(ProposalBank(proposals, Float64[1, 1]).masses) === Float64
    @test eltype(ProposalBank(proposals, BigFloat[1, 1]).masses) === BigFloat

    @test_throws ArgumentError ProposalBank(typeof(proposals)())
    @test_throws ArgumentError ProposalBank(proposals, [1])
    @test_throws ArgumentError ProposalBank(proposals, Bool[true, false])
    @test_throws ArgumentError ProposalBank(proposals, [1, -1])
    @test_throws ArgumentError ProposalBank(proposals, [1.0, NaN])
    @test_throws ArgumentError ProposalBank(proposals, [1.0, Inf])
    @test_throws ArgumentError ProposalBank(proposals, [1.0, -Inf])
    @test_throws ArgumentError ProposalBank(proposals, [0, 0])
    @test_throws ArgumentError ProposalBank(proposals, Real[1, 3])
    @test_throws ArgumentError ProposalBank(proposals, Any[1, 3])
    @test_throws ArgumentError ProposalBank(proposals, [1 + 0im, 3 + 0im])
    @test_throws ArgumentError ProposalBank(proposals, ["1", "3"])
    @test_throws MethodError ProposalBank(Tuple(proposals))
    @test_throws MethodError ProposalBank(proposals, (1, 3))
end

@testset "stratified static-MIS identity and preparation" begin
    first_proposal = StaticMISGaussian(-1.0)
    zero_mass_proposal = StaticMISGaussian(50.0)
    third_proposal = StaticMISGaussian(1.0)
    bank = ProposalBank(
        [first_proposal, zero_mass_proposal, third_proposal],
        [1, 0, 3],
    )
    algorithm = ImportanceSampling(bank; nsamples=10_001)
    mixture_logtarget = sample -> log(
        0.25 * exp(static_mis_gaussian_logdensity(-1.0, sample)) +
        0.75 * exp(static_mis_gaussian_logdensity(1.0, sample)),
    )

    sampler = @inferred prepare_sampler(
        Random.Xoshiro(42),
        mixture_logtarget,
        algorithm;
        threaded=false,
    )
    result = @inferred importance_sample!(sampler)

    @test length(result) == 10_001
    @test propertynames(result.provenance) == (:proposal_id,)
    @test all(id -> id in (1, 3), result.provenance.proposal_id)
    @test maximum(abs, result.logweights) <= 4096eps(Float64)
    @test abs(lognormalizer(result)) <= 4096eps(Float64)
    @test first_proposal.draw_count[] + third_proposal.draw_count[] == 10_001
    @test first_proposal.density_count[] == 10_001
    @test third_proposal.density_count[] == 10_001
    @test zero_mass_proposal.draw_count[] == 0
    @test zero_mass_proposal.density_count[] == 0
    @test result.diagnostics.method === :importance_sampling
    @test result.diagnostics.mis_scheme === :stratified_mixture
    @test sampler.method_state isa IS._PreparedStaticMIS
    @test sampler.method_state.design.assignment isa IS._StratifiedAssignment
    @test sampler.method_state.design.denominator isa IS._FullMixtureDenominator

    assignment_storage = sampler.random_buffers.assignments
    first_samples = copy(result.samples)
    first_weights = copy(result.logweights)
    first_ids = copy(result.provenance.proposal_id)
    second_result = @inferred importance_sample!(sampler)
    @test first_proposal.draw_count[] + third_proposal.draw_count[] == 20_002
    @test result.samples !== second_result.samples
    @test result.logweights !== second_result.logweights
    @test result.provenance.proposal_id !== second_result.provenance.proposal_id
    @test sampler.random_buffers.assignments === assignment_storage
    @test result.samples == first_samples
    @test result.logweights == first_weights
    @test result.provenance.proposal_id == first_ids

    draws_before = first_proposal.draw_count[] + third_proposal.draw_count[]
    abstract_bank = ProposalBank(Any[first_proposal, third_proposal])
    @test_throws ArgumentError prepare_sampler(
        Random.Xoshiro(44),
        mixture_logtarget,
        ImportanceSampling(abstract_bank; nsamples=3);
        threaded=false,
    )
    @test first_proposal.draw_count[] + third_proposal.draw_count[] == draws_before
end

@testset "random-mixture assignment and identity" begin
    first_proposal = StaticMISGaussian(-1.0)
    inert_proposal = StaticMISGaussian(50.0)
    third_proposal = StaticMISGaussian(1.0)
    bank = ProposalBank(
        [first_proposal, inert_proposal, third_proposal],
        [1, 0, 3],
    )
    mixture_logtarget = sample -> log(
        0.25 * exp(static_mis_gaussian_logdensity(-1.0, sample)) +
        0.75 * exp(static_mis_gaussian_logdensity(1.0, sample)),
    )
    nsamples = 20_000
    sampler = @inferred prepare_sampler(
        Random.Xoshiro(0x5301),
        mixture_logtarget,
        ImportanceSampling(
            bank;
            nsamples,
            mis_scheme=RandomMixture(),
        );
        threaded=false,
    )
    result = @inferred importance_sample!(sampler)

    first_frequency = count(==(1), result.provenance.proposal_id) / nsamples
    @test length(result) == nsamples
    @test abs(first_frequency - 0.25) < 0.02
    @test all(id -> id in (1, 3), result.provenance.proposal_id)
    @test maximum(abs, result.logweights) <= 4096eps(Float64)
    @test abs(lognormalizer(result)) <= 4096eps(Float64)
    @test first_proposal.draw_count[] + third_proposal.draw_count[] == nsamples
    @test first_proposal.density_count[] == nsamples
    @test third_proposal.density_count[] == nsamples
    @test inert_proposal.draw_count[] == 0
    @test inert_proposal.density_count[] == 0
    @test sampler.method_state.design.assignment isa IS._RandomAssignment
    @test sampler.method_state.design.denominator isa IS._FullMixtureDenominator
    @test result.diagnostics.mis_scheme === :random_mixture
end

@testset "standard MIS generating denominator" begin
    first_proposal = StaticMISGaussian(0.0)
    inert_proposal = StaticMISGaussian(50.0)
    third_proposal = StaticMISGaussian(0.0)
    bank = ProposalBank(
        [first_proposal, inert_proposal, third_proposal],
        [1, 0, 3],
    )
    target = sample -> static_mis_gaussian_logdensity(0.0, sample)
    nsamples = 1_001
    sampler = @inferred prepare_sampler(
        Random.Xoshiro(0x5302),
        target,
        ImportanceSampling(
            bank;
            nsamples,
            mis_scheme=StandardMIS(),
        );
        threaded=false,
    )
    result = @inferred importance_sample!(sampler)

    @test length(result) == nsamples
    @test all(id -> id in (1, 3), result.provenance.proposal_id)
    @test maximum(abs, result.logweights) <= 16eps(Float64)
    @test first_proposal.density_count[] + third_proposal.density_count[] == nsamples
    @test first_proposal.density_count[] == count(==(1), result.provenance.proposal_id)
    @test third_proposal.density_count[] == count(==(3), result.provenance.proposal_id)
    @test inert_proposal.draw_count[] == 0
    @test inert_proposal.density_count[] == 0
    @test sampler.method_state.design.assignment isa IS._StratifiedAssignment
    @test sampler.method_state.design.denominator isa IS._GeneratingDenominator
    @test result.diagnostics.mis_scheme === :standard_mis
end

@testset "partial deterministic-mixture validation" begin
    proposals = [StaticMISGaussian(Float64(id)) for id in 1:5]
    bank = ProposalBank(proposals, [1, 2, 3, 4, 0])
    target = _ -> 0.0
    invalid_groups = (
        (),
        ((1, 2), (), (3, 4, 5)),
        ((1, 2), (2, 3, 4, 5)),
        ((1, 2), (3, 4)),
        ((1, 2), (3, 4, 5, 6)),
    )

    for groups in invalid_groups
        events = Symbol[]
        rng = StaticMISRecordingRNG(Random.Xoshiro(0x5303), events)
        algorithm = ImportanceSampling(
            bank;
            nsamples=8,
            mis_scheme=PartialDeterministicMixture(groups),
        )
        @test_throws ArgumentError prepare_sampler(
            rng,
            target,
            algorithm;
            threaded=false,
        )
        @test isempty(events)
    end

    sampler = @inferred prepare_sampler(
        Random.Xoshiro(0x5304),
        target,
        ImportanceSampling(
            bank;
            nsamples=16,
            mis_scheme=PartialDeterministicMixture(
                ((1, 2), (5,), (3, 4)),
            ),
        );
        threaded=false,
    )
    denominator = sampler.method_state.design.denominator
    result = @inferred importance_sample!(sampler)

    @test denominator isa IS._PartialMixtureDenominator
    @test denominator.group_of_slot == [1, 1, 2, 2]
    @test denominator.offsets == [1, 3, 5]
    @test denominator.members == [1, 2, 3, 4]
    @test exp.(denominator.logcoefficients) ≈ [1 / 3, 2 / 3, 3 / 7, 4 / 7]
    @test all(id -> id in (1, 2, 3, 4), result.provenance.proposal_id)
    @test proposals[5].draw_count[] == 0
    @test proposals[5].density_count[] == 0
end

@testset "partial deterministic-mixture brute-force denominator" begin
    proposal_logs = [
        -0.1 -0.7 -1.2 -1.8
        -0.4 -0.2 -1.5 -1.1
        -1.4 -1.0 -0.3 -0.8
        -1.7 -1.3 -0.6 -0.05
    ]
    proposals = [
        StaticMISTableProposal(id, Float64(id), proposal_logs) for id in 1:4
    ]
    nominal_masses = [1.0, 2.0, 3.0, 4.0]
    groups = ((1, 2), (3, 4))
    nsamples = 101
    sampler = @inferred prepare_sampler(
        Random.Xoshiro(0x5305),
        StaticMISTableTarget(zeros(4)),
        ImportanceSampling(
            ProposalBank(proposals, nominal_masses);
            nsamples,
            mis_scheme=PartialDeterministicMixture(groups),
        );
        threaded=false,
    )
    result = @inferred importance_sample!(sampler)

    for sample_index in eachindex(result.logweights)
        proposal_id = result.provenance.proposal_id[sample_index]
        group = proposal_id <= 2 ? groups[1] : groups[2]
        sample = Int(result.samples[sample_index])
        group_masses = nominal_masses[collect(group)]
        coefficients = group_masses ./ sum(group_masses)
        expected_denominator = log(sum(
            coefficients[group_index] * exp(proposal_logs[member, sample])
            for (group_index, member) in enumerate(group)
        ))
        @test -result.logweights[sample_index] ≈ expected_denominator
    end
    first_group_count = count(<=(2), result.provenance.proposal_id)
    @test proposals[1].density_count[] == first_group_count
    @test proposals[2].density_count[] == first_group_count
    @test proposals[3].density_count[] == nsamples - first_group_count
    @test proposals[4].density_count[] == nsamples - first_group_count
    @test result.diagnostics.mis_scheme === :partial_deterministic_mixture
end

@testset "static-MIS active proposal dimension" begin
    bank = ProposalBank(
        [
            SphericalGaussian(zeros(2), 1.0),
            SphericalGaussian(ones(2), 1.0),
            SphericalGaussian(zeros(7), 1.0),
        ],
        [1, 1, 0],
    )
    algorithm = ImportanceSampling(bank; nsamples=8)

    @test_throws DimensionMismatch prepare_sampler(
        Random.Xoshiro(0x5200),
        StaticMISDimensionTarget{3}(),
        algorithm;
        threaded=false,
    )

    sampler = @inferred prepare_sampler(
        Random.Xoshiro(0x5201),
        StaticMISDimensionTarget{2}(),
        algorithm;
        threaded=false,
    )
    result = @inferred importance_sample!(sampler)
    @test size(result.samples) == (2, 8)
    @test all(id -> id in (1, 2), result.provenance.proposal_id)

    external_bank = ProposalBank(
        [StaticMISGaussian(-1.0), StaticMISGaussian(1.0)],
    )
    unknown_dimension = @inferred prepare_sampler(
        Random.Xoshiro(0x5202),
        StaticMISDimensionTarget{3}(),
        ImportanceSampling(external_bank; nsamples=4);
        threaded=false,
    )
    @test length(importance_sample!(unknown_dimension)) == 4
end

@testset "stratified assignment failure location" begin
    proposals = [StaticMISGaussian(-1.0), StaticMISGaussian(1.0)]
    rng = StaticMISFailingAssignmentRNG(Random.Xoshiro(0x5207), 0, 3)
    sampler = prepare_sampler(
        rng,
        _ -> 0.0,
        ImportanceSampling(ProposalBank(proposals); nsamples=8);
        threaded=false,
    )
    failure = caught_static_mis_failure(() -> importance_sample!(sampler))

    @test failure isa SamplerExecutionError
    @test (failure.phase, failure.sample_index) == (:proposal_draw, 3)
    @test failure.captured.ex isa StaticMISAssignmentFailure
    @test failure.captured.ex.sample_index == 3
    @test rng.uniform_count == 3
    @test all(iszero(proposal.draw_count[]) for proposal in proposals)
end

@testset "static-MIS assignments precede proposal draws" begin
    nsamples = 17
    for scheme in (StratifiedMixture(), RandomMixture())
        events = Symbol[]
        proposals = [
            StaticMISUniformProposal(-1.0, Ref(0), events),
            StaticMISUniformProposal(1.0, Ref(0), events),
        ]
        bank = ProposalBank(proposals, [1, 3])
        rng = StaticMISRecordingRNG(Random.Xoshiro(0x5202), events)
        result = importance_sample(
            rng,
            _ -> 0.0,
            ImportanceSampling(bank; nsamples, mis_scheme=scheme);
            threaded=false,
        )

        @test length(result) == nsamples
        @test events[1:nsamples] == fill(:random, nsamples)
        @test events[(nsamples + 1):end] == repeat([:draw, :random], nsamples)
        @test sum(proposal.draw_count[] for proposal in proposals) == nsamples

        expected_rng = Random.Xoshiro(0x5202)
        cdf = cumsum(bank.masses)
        cdf[end] = 1.0
        expected_slots = [
            searchsortedfirst(
                cdf,
                scheme isa RandomMixture ? rand(expected_rng) :
                ((sample_index - 1) + rand(expected_rng)) / nsamples,
            ) for sample_index in 1:nsamples
        ]
        expected_samples = [
            proposals[slot].location + rand(expected_rng) for slot in expected_slots
        ]
        @test result.provenance.proposal_id == expected_slots
        @test result.samples == expected_samples
    end
end

function static_mis_table_result(
    proposal_logs::Matrix{T},
    target_logs::Vector{T};
    fail_draw_at=0,
    threaded=false,
) where {T<:AbstractFloat}
    proposals = [
        StaticMISTableProposal(
            proposal_id,
            T(proposal_id),
            proposal_logs;
            fail_draw=proposal_id == fail_draw_at,
        ) for proposal_id in 1:2
    ]
    bank = ProposalBank(proposals)
    sampler = prepare_sampler(
        Random.Xoshiro(0x5203),
        StaticMISTableTarget(target_logs),
        ImportanceSampling(bank; nsamples=2);
        threaded,
    )
    return sampler, proposals
end

function static_mis_scheme_table_result(
    proposal_logs::Matrix{T},
    target_logs::Vector{T},
    scheme;
    threaded=false,
) where {T<:AbstractFloat}
    nproposals = size(proposal_logs, 1)
    proposals = [
        StaticMISTableProposal(
            proposal_id,
            T(proposal_id),
            proposal_logs,
        ) for proposal_id in 1:nproposals
    ]
    return importance_sample(
        Random.Xoshiro(0x5306),
        StaticMISTableTarget(target_logs),
        ImportanceSampling(
            ProposalBank(proposals);
            nsamples=nproposals,
            mis_scheme=scheme,
        );
        threaded,
    )
end

@testset "static-MIS support conditions" begin
    nproposals = 4
    target_logs = fill(-log(nproposals), nproposals)
    disjoint_logs = fill(-Inf, nproposals, nproposals)
    for proposal_id in 1:nproposals
        disjoint_logs[proposal_id, proposal_id] = 0.0
    end

    aggregate = static_mis_scheme_table_result(
        disjoint_logs,
        target_logs,
        StratifiedMixture(),
    )
    partial_without_group_support = static_mis_scheme_table_result(
        disjoint_logs,
        target_logs,
        PartialDeterministicMixture(((1, 2), (3, 4))),
    )
    standard_without_generator_support = static_mis_scheme_table_result(
        disjoint_logs,
        target_logs,
        StandardMIS(),
    )

    @test all(isfinite, aggregate.logweights)
    @test maximum(abs, aggregate.logweights) <= 16eps(Float64)
    @test abs(lognormalizer(aggregate)) <= 16eps(Float64)
    @test partial_without_group_support.logweights ≈ fill(-log(2), nproposals)
    @test lognormalizer(partial_without_group_support) ≈ -log(2)
    @test standard_without_generator_support.logweights ≈
          fill(-log(nproposals), nproposals)
    @test lognormalizer(standard_without_generator_support) ≈ -log(nproposals)

    common_logs = fill(-log(nproposals), nproposals, nproposals)
    for scheme in (
        StandardMIS(),
        PartialDeterministicMixture(((1, 2), (3, 4))),
    )
        supported = static_mis_scheme_table_result(
            common_logs,
            target_logs,
            scheme,
        )
        @test all(isfinite, supported.logweights)
        @test maximum(abs, supported.logweights) <= 16eps(Float64)
        @test abs(lognormalizer(supported)) <= 16eps(Float64)
    end
end

@testset "remaining static-MIS denominator truth values" begin
    generating_plus_inf = [Inf 0.0; 0.0 Inf]
    for scheme in (
        StandardMIS(),
        PartialDeterministicMixture(((1, 2),)),
    )
        zero_weight = static_mis_scheme_table_result(
            generating_plus_inf,
            zeros(2),
            scheme,
        )
        @test zero_weight.logweights == fill(-Inf, 2)
        @test lognormalizer(zero_weight) == -Inf

        for invalid in ([-Inf 0.0; 0.0 0.0], [NaN 0.0; 0.0 0.0])
            failure = caught_static_mis_failure() do
                static_mis_scheme_table_result(invalid, zeros(2), scheme)
            end
            @test failure isa SamplerExecutionError
            @test (failure.phase, failure.sample_index) ==
                  (:proposal_logdensity, 1)
            @test failure.captured.ex isa DomainError
        end
    end
end

@testset "stratified static-MIS log-value truth table" begin
    float32_logs = zeros(Float32, 2, 2)
    float32_proposals = [
        StaticMISTableProposal(i, Float32(i), float32_logs) for i in 1:2
    ]
    float32_result = @inferred importance_sample(
        Random.Xoshiro(44),
        StaticMISTableTarget(zeros(Float32, 2)),
        ImportanceSampling(
            ProposalBank(float32_proposals, Float32[1, 3]);
            nsamples=2,
        );
        threaded=false,
    )
    @test eltype(float32_result.logweights) === Float32
    @test maximum(abs, float32_result.logweights) <= 8eps(Float32)

    _, big_mass_proposals = static_mis_table_result(
        zeros(2, 2),
        zeros(2),
    )
    big_mass_bank = ProposalBank(big_mass_proposals, BigFloat[1, 3])
    big_mass_result = @inferred importance_sample(
        Random.Xoshiro(45),
        StaticMISTableTarget(zeros(2)),
        ImportanceSampling(big_mass_bank; nsamples=2);
        threaded=false,
    )
    @test eltype(big_mass_result.logweights) === BigFloat
    @test maximum(abs, big_mass_result.logweights) <= 8eps(BigFloat)

    finite_sampler, _ = static_mis_table_result(zeros(2, 2), [0.0, -Inf])
    @test importance_sample!(finite_sampler).logweights == [0.0, -Inf]

    non_generating_minus_inf = [0.0 -Inf; -Inf 0.0]
    minus_sampler, _ = static_mis_table_result(
        non_generating_minus_inf,
        fill(-log(2), 2),
    )
    @test importance_sample!(minus_sampler).logweights == [0.0, 0.0]

    non_generating_plus_inf = [0.0 Inf; Inf 0.0]
    plus_sampler, _ = static_mis_table_result(non_generating_plus_inf, zeros(2))
    @test importance_sample!(plus_sampler).logweights == fill(-Inf, 2)

    generating_plus_inf = [Inf -Inf; -Inf Inf]
    generating_plus_sampler, _ = static_mis_table_result(generating_plus_inf, zeros(2))
    @test importance_sample!(generating_plus_sampler).logweights == fill(-Inf, 2)

    for bad_target in (NaN, Inf)
        sampler, _ = static_mis_table_result(zeros(2, 2), [bad_target, 0.0])
        failure = caught_static_mis_failure(() -> importance_sample!(sampler))
        @test failure isa SamplerExecutionError
        @test (failure.phase, failure.sample_index) == (:target, 1)
        @test failure.captured.ex isa DomainError
    end

    nan_term = [0.0 0.0; NaN 0.0]
    nan_sampler, _ = static_mis_table_result(nan_term, zeros(2))
    nan_failure = caught_static_mis_failure(() -> importance_sample!(nan_sampler))
    @test nan_failure isa SamplerExecutionError
    @test (nan_failure.phase, nan_failure.sample_index) == (:proposal_logdensity, 1)
    @test nan_failure.captured.ex isa DomainError

    generating_minus_inf = [-Inf 0.0; Inf 0.0]
    generating_minus_sampler, _ = static_mis_table_result(
        generating_minus_inf,
        zeros(2),
    )
    generating_failure = caught_static_mis_failure(
        () -> importance_sample!(generating_minus_sampler),
    )
    @test generating_failure isa SamplerExecutionError
    @test (generating_failure.phase, generating_failure.sample_index) ==
          (:proposal_logdensity, 1)
    @test generating_failure.captured.ex isa DomainError

    unsupported = fill(-Inf, 2, 2)
    unsupported_sampler, _ = static_mis_table_result(unsupported, zeros(2))
    support_failure = caught_static_mis_failure(
        () -> importance_sample!(unsupported_sampler),
    )
    @test support_failure isa SamplerExecutionError
    @test (support_failure.phase, support_failure.sample_index) ==
          (:proposal_logdensity, 1)
    @test support_failure.captured.ex isa DomainError

    draw_sampler, draw_proposals = static_mis_table_result(
        zeros(2, 2),
        zeros(2);
        fail_draw_at=2,
    )
    draw_failure = caught_static_mis_failure(() -> importance_sample!(draw_sampler))
    @test draw_failure isa SamplerExecutionError
    @test (draw_failure.phase, draw_failure.sample_index) == (:proposal_draw, 2)
    @test sum(proposal.draw_count[] for proposal in draw_proposals) == 2
    @test all(iszero(proposal.density_count[]) for proposal in draw_proposals)
end

@testset "stratified static-MIS serial and threaded execution" begin
    nsamples = 256
    serial_state = StaticMISThreadState(nsamples, 2)
    threaded_state = StaticMISThreadState(nsamples, 2)
    serial_bank = ProposalBank(
        [
            StaticMISThreadProposal(1, serial_state),
            StaticMISThreadProposal(2, serial_state),
        ],
        [1, 3],
    )
    threaded_bank = ProposalBank(
        [
            StaticMISThreadProposal(1, threaded_state),
            StaticMISThreadProposal(2, threaded_state),
        ],
        [1, 3],
    )
    caller_task = current_task()
    serial_result = importance_sample(
        Random.Xoshiro(0x5204),
        StaticMISThreadTarget(serial_state),
        ImportanceSampling(serial_bank; nsamples=nsamples);
        threaded=false,
    )
    threaded_result = @inferred importance_sample(
        Random.Xoshiro(0x5204),
        StaticMISThreadTarget(threaded_state),
        ImportanceSampling(threaded_bank; nsamples=nsamples);
        threaded=true,
    )

    @test serial_result.samples == threaded_result.samples
    @test serial_result.logweights == threaded_result.logweights
    @test serial_result.provenance == threaded_result.provenance
    @test all(==(caller_task), serial_state.draw_tasks)
    @test all(==(caller_task), threaded_state.draw_tasks)
    @test serial_state.density_draw_counts == fill(nsamples, 2, nsamples)
    @test threaded_state.density_draw_counts == fill(nsamples, 2, nsamples)
    @test serial_result.diagnostics.execution === :serial
    expected_execution = Threads.nthreads(:default) > 1 ? :threaded : :serial
    @test threaded_result.diagnostics.execution === expected_execution

    if Threads.nthreads(:default) > 1
        @test all(!=(caller_task), threaded_state.target_tasks)
        @test all(!=(caller_task), threaded_state.density_tasks)
        @test length(unique(threaded_state.target_tasks)) > 1
        @test length(unique(threaded_state.density_tasks)) > 1
    else
        @test all(==(caller_task), threaded_state.target_tasks)
        @test all(==(caller_task), threaded_state.density_tasks)
    end
end

@testset "remaining static-MIS schemes serial and threaded" begin
    proposals = [
        SphericalGaussian(-3.0, 1.0),
        SphericalGaussian(-1.0, 1.0),
        SphericalGaussian(1.0, 1.0),
        SphericalGaussian(3.0, 1.0),
    ]
    bank = ProposalBank(proposals, [1, 2, 3, 4])
    target = sample -> -abs2(sample) / 2
    nsamples = 257

    for scheme in (
        RandomMixture(),
        StandardMIS(),
        PartialDeterministicMixture(((1, 2), (3, 4))),
    )
        algorithm = ImportanceSampling(bank; nsamples, mis_scheme=scheme)
        serial = @inferred importance_sample(
            Random.Xoshiro(0x5307),
            target,
            algorithm;
            threaded=false,
        )
        threaded = @inferred importance_sample(
            Random.Xoshiro(0x5307),
            target,
            algorithm;
            threaded=true,
        )

        @test serial.samples == threaded.samples
        @test serial.logweights == threaded.logweights
        @test serial.provenance == threaded.provenance
        @test serial.diagnostics.execution === :serial
        expected_execution = Threads.nthreads(:default) > 1 ? :threaded : :serial
        @test threaded.diagnostics.execution === expected_execution
    end
end

@testset "static-MIS scheme and algorithm configuration" begin
    proposals = [
        SphericalGaussian(-1.0, 1.0),
        SphericalGaussian(1.0, 1.0),
    ]
    bank = ProposalBank(proposals, [1, 3])

    stratified = @inferred StratifiedMixture()
    random = @inferred RandomMixture()
    standard = @inferred StandardMIS()
    groups = ((1, 2),)
    partial = @inferred PartialDeterministicMixture(groups)

    @test stratified isa AbstractMISScheme
    @test random isa AbstractMISScheme
    @test standard isa AbstractMISScheme
    @test partial isa AbstractMISScheme
    @test partial.groups === groups

    default_algorithm = @inferred ImportanceSampling(bank; nsamples=9)
    standard_algorithm = @inferred ImportanceSampling(
        bank;
        nsamples=9,
        mis_scheme=standard,
    )

    @test default_algorithm.proposal === bank
    @test default_algorithm.nsamples === 9
    @test default_algorithm.mis_scheme isa StratifiedMixture
    @test standard_algorithm.mis_scheme === standard
    @test_throws UndefKeywordError ImportanceSampling(bank)
    @test_throws ArgumentError ImportanceSampling(bank; nsamples=true)
    @test_throws ArgumentError ImportanceSampling(bank; nsamples=9.0)
    @test_throws MethodError ImportanceSampling(
        first(proposals);
        nsamples=9,
        mis_scheme=standard,
    )
end
