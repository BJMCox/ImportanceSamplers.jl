using ImportanceSamplers
using Random

function correlated_logtarget(sample)::Float64
    centered_1 = sample[1] - 1.0
    centered_2 = sample[2] + 0.5
    return -0.5 * (
        1.5625abs2(centered_1) -
        1.25centered_1 * centered_2 +
        1.25abs2(centered_2)
    )
end

function correlated_gradient!(destination, sample)
    centered_1 = sample[1] - 1.0
    centered_2 = sample[2] + 0.5
    destination[1] = -1.5625centered_1 + 0.625centered_2
    destination[2] = 0.625centered_1 - 1.25centered_2
    return destination
end

function main(; rounds=4, round_size=3_000)
    bank = ProposalBank([
        FactorGaussian([-3.0, 1.0], [1.6 0.0; 0.2 1.3]),
        DiagonalGaussian([0.0, -2.5], [1.5, 1.1]),
        SphericalGaussian([3.0, 1.0], 1.8),
    ])
    algorithm = FirstOrderGRAMIS(
        bank;
        rounds,
        round_size,
        repulsion_strength=0.05,
    )
    sampler = prepare_sampler(
        Xoshiro(0x4752414d49534558),
        LogTarget(correlated_logtarget; grad=correlated_gradient!),
        algorithm;
        threaded=false,
    )

    first_result = importance_sample!(sampler)
    first_proposal = current_proposal(sampler)
    second_result = importance_sample!(sampler)
    second_proposal = current_proposal(sampler)

    first_weights = normalized_weights(first_result)
    second_weights = normalized_weights(second_result)
    summary = (
        first_log_normalizer=lognormalizer(first_result),
        second_log_normalizer=lognormalizer(second_result),
        first_mean=first_result.samples * first_weights,
        second_mean=second_result.samples * second_weights,
        learned_locations=(
            after_first=[
                copy(proposal.location) for proposal in first_proposal.proposals
            ],
            after_second=[
                copy(proposal.location) for proposal in second_proposal.proposals
            ],
        ),
    )
    println(summary)
    return (; sampler, first_result, second_result, summary)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
