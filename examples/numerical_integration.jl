using ImportanceSamplers
using Random

logintegrand(x)::Float64 = -abs2(x)

function main(; nsamples=100_000)
    # The package accepts the log-integrand to keep ratios stable over wide ranges.
    proposal = SphericalGaussian(0.0, 1.0)
    algorithm = ImportanceSampling(proposal; nsamples)
    samples = importance_sample(
        Xoshiro(0x494e54454752414c),
        logintegrand,
        algorithm;
        threaded=false,
    )

    # Exponentiating the estimated log normalizer gives the requested integral.
    estimate = exp(lognormalizer(samples))
    exact = sqrt(pi)
    summary = (
        estimate,
        exact,
        absolute_error=abs(estimate - exact),
        sample_count=length(samples),
    )
    println("integral estimate: ", summary.estimate)
    println("exact sqrt(pi): ", summary.exact)
    println("absolute error: ", summary.absolute_error)
    println("sample count: ", summary.sample_count)
    return (; samples, summary)
end

# A different normalized proposal changes numerical variance, not the integral.
# This direct log-weight path requires a nonnegative integrand. For a signed
# integrand, split positive and negative parts or calculate a proposal
# expectation outside the log-target interface.

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
