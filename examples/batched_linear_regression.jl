using ImportanceSamplers, LinearAlgebra, Random, Statistics
import MLDataDevices

# Unit observation noise and independent Normal(0, 2) coefficient priors.
# Constants independent of beta can be omitted from this log target.
function regression_logtarget(beta, p)
    value = -sum(abs2, beta) / 8
    for row in axes(p.design, 1)
        residual = -p.observed[row]
        for column in eachindex(beta)
            residual += p.design[row, column] * beta[column]
        end
        value -= abs2(residual) / 2
    end
    return value
end

function regression_batch!(values, samples, p)
    # One matrix multiplication evaluates every proposed coefficient vector.
    # Samples are columns. Neither the samples nor the data are modified.
    residuals = p.design * samples .- p.observed
    values .= .-vec(sum(abs2, residuals; dims=1)) ./ 2 .-
              vec(sum(abs2, samples; dims=1)) ./ 8
    return nothing
end

function batched_regression_example(device=MLDataDevices.CPUDevice(); T=Float64)
    rng = Xoshiro(42)
    truth = T[-0.4, 1.1, -0.8]
    design = randn(rng, T, 512, length(truth))
    observed = design * truth + randn(rng, T, size(design, 1))
    p = (; design, observed)

    # Least squares gives a cheap proposal centre. A deliberately broader
    # proposal covers posterior uncertainty. Importance weights correct it.
    proposal = SphericalGaussian(design \ observed, T(2) / sqrt(T(size(design, 1))))
    target = LogTarget(regression_logtarget; batch=regression_batch!)
    prepared = prepare_sampler(rng, target, p,
        ImportanceSampling(proposal; nsamples=16_384))

    # This transfers the sampler, target data, proposal and random buffers.
    # With an accelerator, the callback's matrix operations stay on that device.
    sampler = device(prepared)
    return importance_sample!(sampler)
end

if abspath(PROGRAM_FILE) == @__FILE__
    samples = batched_regression_example()
    println("Posterior coefficient means: ", mean(samples))
    println("Weight ESS: ", inv(sum(abs2, normalized_weights(samples))))
end
