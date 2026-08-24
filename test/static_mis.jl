using Test
using ImportanceSamplers
import DensityInterface
import Random

include("support/static_mis.jl")

const IS = ImportanceSamplers

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
    failure_storage = sampler.random_buffers.failures
    first_samples = copy(result.samples)
    first_weights = copy(result.logweights)
    first_ids = copy(result.provenance.proposal_id)
    second_result = @inferred importance_sample!(sampler)
    @test first_proposal.draw_count[] + third_proposal.draw_count[] == 20_002
    @test result.samples !== second_result.samples
    @test result.logweights !== second_result.logweights
    @test result.provenance.proposal_id !== second_result.provenance.proposal_id
    @test sampler.random_buffers.assignments === assignment_storage
    @test sampler.random_buffers.failures === failure_storage
    @test result.samples == first_samples
    @test result.logweights == first_weights
    @test result.provenance.proposal_id == first_ids

    draws_before = first_proposal.draw_count[] + third_proposal.draw_count[]
    for scheme in (
        RandomMixture(),
        StandardMIS(),
        PartialDeterministicMixture(((1, 2, 3),)),
    )
        unsupported = ImportanceSampling(bank; nsamples=3, mis_scheme=scheme)
        @test_throws ArgumentError prepare_sampler(
            Random.Xoshiro(43),
            mixture_logtarget,
            unsupported;
            threaded=false,
        )
    end
    @test first_proposal.draw_count[] + third_proposal.draw_count[] == draws_before

    abstract_bank = ProposalBank(Any[first_proposal, third_proposal])
    @test_throws ArgumentError prepare_sampler(
        Random.Xoshiro(44),
        mixture_logtarget,
        ImportanceSampling(abstract_bank; nsamples=3);
        threaded=false,
    )
    @test first_proposal.draw_count[] + third_proposal.draw_count[] == draws_before
end

@testset "stratified assignments precede proposal draws" begin
    nsamples = 17
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
        ImportanceSampling(bank; nsamples=nsamples);
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
        searchsortedfirst(cdf, ((sample_index - 1) + rand(expected_rng)) / nsamples)
        for sample_index in 1:nsamples
    ]
    expected_samples = [
        proposals[slot].location + rand(expected_rng) for slot in expected_slots
    ]
    @test result.provenance.proposal_id == expected_slots
    @test result.samples == expected_samples
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

function caught_static_mis_failure(f)
    try
        f()
    catch error
        return error
    end
    return nothing
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
