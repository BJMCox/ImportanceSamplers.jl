using ImportanceSamplers
using LinearAlgebra
using Random

const TRUE_COEFFICIENTS = [-0.4, 1.1, -0.8]
const COEFFICIENT_NAMES = ("intercept", "x1", "x2")

@inline logistic(x) = inv(1 + exp(-x))
@inline softplus(x) = max(x, zero(x)) + log1p(exp(-abs(x)))

function simulate_logistic_data(rng, coefficients; observations=50)
    predictors = Matrix{Float64}(undef, observations, length(coefficients))
    outcomes = Vector{Int}(undef, observations)

    for observation in 1:observations
        predictors[observation, 1] = 1.0
        for coefficient in 2:length(coefficients)
            predictors[observation, coefficient] = randn(rng)
        end

        linear_predictor = 0.0
        for coefficient in eachindex(coefficients)
            linear_predictor +=
                predictors[observation, coefficient] * coefficients[coefficient]
        end
        outcomes[observation] = rand(rng) < logistic(linear_predictor)
    end

    return (; predictors, outcomes)
end

function logistic_logtarget(coefficients, context)::Float64
    prior_scale = context.prior_scale
    value = -length(coefficients) * (log(prior_scale) + 0.5 * log(2pi))

    for coefficient in coefficients
        value -= 0.5 * abs2(coefficient / prior_scale)
    end

    for observation in eachindex(context.outcomes)
        linear_predictor = 0.0
        for coefficient in eachindex(coefficients)
            linear_predictor += context.predictors[observation, coefficient] *
                                coefficients[coefficient]
        end
        value -= context.outcomes[observation] == 1 ?
                 softplus(-linear_predictor) : softplus(linear_predictor)
    end

    return value
end

function weighted_quantile(values, weights, probability)
    order = sortperm(values)
    cumulative_weight = 0.0
    for index in order
        cumulative_weight += weights[index]
        cumulative_weight >= probability && return values[index]
    end
    return values[last(order)]
end

function summarize_coefficients(result)
    weights = normalized_weights(result)
    samples = result.samples
    coefficient_count = size(samples, 1)

    means = zeros(coefficient_count)
    standard_deviations = zeros(coefficient_count)
    intervals = Vector{Tuple{Float64,Float64}}(undef, coefficient_count)

    for coefficient in 1:coefficient_count
        values = view(samples, coefficient, :)
        means[coefficient] = sum(weights .* values)
        standard_deviations[coefficient] = sqrt(
            sum(weights .* abs2.(values .- means[coefficient])),
        )
        intervals[coefficient] = (
            weighted_quantile(values, weights, 0.025),
            weighted_quantile(values, weights, 0.975),
        )
    end

    return (
        means,
        standard_deviations,
        intervals,
        effective_sample_size=inv(sum(abs2, weights)),
        log_evidence=lognormalizer(result),
    )
end

function fit_gaussian_proposal(result; covariance_inflation=1.5)
    weights = normalized_weights(result)
    samples = result.samples
    coefficient_count = size(samples, 1)
    means = zeros(coefficient_count)

    for coefficient in 1:coefficient_count
        means[coefficient] = sum(weights .* view(samples, coefficient, :))
    end

    covariance = zeros(coefficient_count, coefficient_count)
    for sample in axes(samples, 2)
        for column in 1:coefficient_count
            column_difference = samples[column, sample] - means[column]
            for row in column:coefficient_count
                covariance[row, column] += weights[sample] *
                                           (samples[row, sample] - means[row]) *
                                           column_difference
            end
        end
    end
    for column in 1:coefficient_count
        for row in 1:(column - 1)
            covariance[row, column] = covariance[column, row]
        end
    end

    covariance_correction = 1 - sum(abs2, weights)
    covariance_correction > 0 || error("the pilot weights collapsed")
    covariance ./= covariance_correction
    # Broaden the fitted proposal to reduce sensitivity to pilot tail coverage.
    covariance .*= covariance_inflation
    covariance += 1e-8I
    factor = cholesky(Symmetric(covariance)).L
    return FactorGaussian(means, factor)
end

function main(; pilot_samples=20_000, nsamples=100_000)
    data = simulate_logistic_data(Xoshiro(0x4c4f474953544943), TRUE_COEFFICIENTS)
    prior_scale = 1.5
    context = (; data..., prior_scale)
    prior = SphericalGaussian(zeros(length(TRUE_COEFFICIENTS)), prior_scale)

    # Use the normalized prior as a simple pilot proposal.
    pilot = importance_sample(
        Xoshiro(0x50494c4f54495331),
        logistic_logtarget,
        context,
        ImportanceSampling(prior; nsamples=pilot_samples),
    )
    proposal = fit_gaussian_proposal(pilot)

    # Draw a fresh, independent final sample from the fitted proposal.
    result = importance_sample(
        Xoshiro(0x49534c4f47495431),
        logistic_logtarget,
        context,
        ImportanceSampling(proposal; nsamples),
    )
    summary = summarize_coefficients(result)
    pilot_ess = inv(sum(abs2, normalized_weights(pilot)))

    println("Logistic regression posterior")
    for coefficient in eachindex(COEFFICIENT_NAMES)
        interval = summary.intervals[coefficient]
        println(
            "  ", COEFFICIENT_NAMES[coefficient],
            ": true=", TRUE_COEFFICIENTS[coefficient],
            ", mean=", round(summary.means[coefficient]; digits=3),
            ", sd=", round(summary.standard_deviations[coefficient]; digits=3),
            ", 95% interval=", round.(interval; digits=3),
        )
    end
    println(
        "  pilot effective sample size: ",
        round(pilot_ess; digits=1),
        " / ",
        pilot_samples,
    )
    println(
        "  final effective sample size: ",
        round(summary.effective_sample_size; digits=1),
        " / ",
        nsamples,
    )
    println("  estimated log evidence: ", round(summary.log_evidence; digits=3))

    return (; pilot, proposal, result, summary, context)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
