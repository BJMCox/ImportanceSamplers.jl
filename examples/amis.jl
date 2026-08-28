using ImportanceSamplers
using Random

function correlated_gaussian_logtarget(sample)::Float64
    # Solve the lower-triangular factor [1 0; 0.8 0.6] without an inverse.
    standardized_1 = sample[1] - 1.0
    standardized_2 = (sample[2] + 1.0 - 0.8standardized_1) / 0.6
    return -log(2pi) - log(0.6) -
           0.5 * (abs2(standardized_1) + abs2(standardized_2))
end

function main(; rounds=4, round_size=1_000)
    # Start well away from the correlated target with a broad spherical proposal.
    initial = SphericalGaussian([-3.0, 3.0], 3.5)
    algorithm = AMIS(initial; rounds, round_size)

    # Preparation owns the copied schedule, RNG, reusable workspaces, and learned state.
    sampler = prepare_sampler(
        Xoshiro(0x414d49534558414d),
        correlated_gaussian_logtarget,
        algorithm;
        threaded=false,
    )

    # One call returns every round with final all-history retrospective weights.
    result = importance_sample!(sampler)
    weights = normalized_weights(result)
    weighted_mean = result.samples * weights
    centered = result.samples .- reshape(weighted_mean, :, 1)
    weighted_covariance = (centered .* reshape(weights, 1, :)) * centered'

    # Provenance counts recover the configured schedule; this snapshot seeds the next call.
    round_counts = [
        count(==(round), result.provenance.round) for round in 1:rounds
    ]
    learned = current_proposal(sampler)
    summary = (
        sample_count=length(result),
        concentration_ess=inv(sum(abs2, weights)),
        weighted_mean,
        weighted_covariance,
        round_counts,
        current_proposal=learned,
    )
    println(summary)
    return (; sampler, result, summary)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
