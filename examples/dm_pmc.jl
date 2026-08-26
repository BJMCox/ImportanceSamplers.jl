using ImportanceSamplers
using Random

function gaussian_component_logdensity(
    sample,
    location::NTuple{2,Float64},
)::Float64
    squared_radius = abs2(sample[1] - location[1]) +
                     abs2(sample[2] - location[2])
    return -log(2pi) - 0.5 * squared_radius
end

function bimodal_logtarget(sample)::Float64
    # The target is an equally weighted mixture with modes at (-2, 0) and (2, 0).
    left = gaussian_component_logdensity(sample, (-2.0, 0.0))
    right = gaussian_component_logdensity(sample, (2.0, 0.0))
    largest = max(left, right)
    return largest + log(exp(left - largest) + exp(right - largest)) - log(2)
end

function main(; rounds=6, round_size=10_000)
    # Four native Gaussian proposals give the initial population broad mode coverage.
    bank = ProposalBank([
        FactorGaussian([-4.0, 0.0], [1.2 0.0; 0.2 1.0]),
        FactorGaussian([-1.0, 0.5], [0.9 0.0; -0.2 0.8]),
        FactorGaussian([1.0, -0.5], [0.9 0.0; 0.2 0.8]),
        FactorGaussian([4.0, 0.0], [1.2 0.0; -0.2 1.0]),
    ])

    # round_size is the number of samples in each adaptive round.
    algorithm = DeterministicMixturePMC(bank; rounds, round_size)
    sampler = prepare_sampler(
        Xoshiro(0x444d504d43455841),
        bimodal_logtarget,
        algorithm;
        threaded=false,
    )

    # One call returns every round; the sampler retains the final resampled population.
    result = importance_sample!(sampler)
    weights = normalized_weights(result)
    learned = current_proposal(sampler)
    summary = (
        sample_count=length(result),
        lognormalizer=lognormalizer(result),
        concentration_ess=inv(sum(abs2, weights)),
        final_locations=[copy(proposal.location) for proposal in learned.proposals],
        round_concentration_ess=result.diagnostics.round_ess,
    )

    # These ESS values describe normalized-weight concentration, not variance-equivalent counts.
    println("total sample count: ", summary.sample_count)
    println("flattened log normalizer: ", summary.lognormalizer)
    println("flattened normalized-weight concentration ESS: ", summary.concentration_ess)
    println("final learned locations: ", summary.final_locations)
    println("per-round concentration ESS: ", summary.round_concentration_ess)
    return (; sampler, result, summary)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
