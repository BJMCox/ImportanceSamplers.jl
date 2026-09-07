using ImportanceSamplers, Random, Statistics

function npmc_logtarget(x)
    # The target mean is [1, -1], with correlation 0.8 and unit marginal variances.
    first = x[1] - 1
    second = (x[2] + 1 - 0.8first) / 0.6
    return -0.5 * (abs2(first) + abs2(second))
end

function main(; rounds=4, round_size=10_000)
    # A broad initial proposal covers the target before adaptation learns its shape.
    proposal = SphericalGaussian([-2.0, 2.0], 3.0)
    prepared = prepare_sampler(Xoshiro(42), npmc_logtarget,
        NPMC(proposal; rounds, round_size))

    # Every round contributes samples. Clipped weights affect only the next proposal.
    samples = importance_sample!(prepared)
    estimate = mean(samples)
    covariance = cov(samples)

    # Raw weights estimate the omitted constant: integral(exp(logtarget)) = 2pi*0.6.
    integral = exp(lognormalizer(samples))
    learned = current_proposal(prepared)
    return (; samples, estimate, covariance, integral, expected_integral=2pi * 0.6, learned)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = main()
    @show result.estimate result.covariance result.integral result.expected_integral
end
