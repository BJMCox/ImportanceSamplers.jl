using ImportanceSamplers
using Random

function contextual_gaussian_logtarget(sample, context)::Float64
    value = -length(sample) * (log(context.scale) + 0.5 * log(2pi))
    for coordinate in eachindex(sample, context.location)
        value -= 0.5 * abs2(
            (sample[coordinate] - context.location[coordinate]) / context.scale,
        )
    end
    return value
end

function main(; nsamples=20_003)
    # The bank is an explicit proposal population; masses need not be equal.
    bank = ProposalBank([
        SphericalGaussian([-2.0, 0.0], 0.8),
        SphericalGaussian([0.0, 0.0], 1.1),
        SphericalGaussian([2.0, 0.5], 1.4),
    ], [1, 4, 2])

    # Target constants live in a two-field context passed separately from code.
    context = (location=[0.4, -0.2], scale=0.9)
    algorithm = ImportanceSampling(
        bank;
        nsamples,
        mis_scheme=StratifiedMixture(),
    )
    result = importance_sample(
        Xoshiro(0x5354415449434d49),
        contextual_gaussian_logtarget,
        context,
        algorithm;
        threaded=false,
    )

    # Results retain the exact count, stable one-based proposal IDs, and raw weights.
    @assert length(result) == nsamples
    proposal_counts = [
        count(==(proposal_id), result.provenance.proposal_id) for
        proposal_id in eachindex(bank.proposals)
    ]
    weights = normalized_weights(result)
    summary = (
        sample_count=length(result),
        proposal_counts,
        effective_sample_size=inv(sum(abs2, weights)),
        lognormalizer=lognormalizer(result),
    )
    println(summary)
    return (; bank, context, result, summary)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
