# Run from the package root with:
# julia --project=validation examples/cuda_logistic_regression.jl

include(joinpath(@__DIR__, "logistic_regression.jl"))

using CUDA
using MLDataDevices

function cuda_device()
    CUDA.functional() || error("CUDA is not functional")
    physical = CUDA.device()
    return MLDataDevices.CUDADevice{typeof(physical),Nothing}(physical)
end

function sample_on_cuda(rng, target, context, algorithm, device)
    prepared = prepare_sampler(
        rng,
        target,
        context,
        algorithm;
        threaded=true,
    )
    return importance_sample!(prepared |> device)
end

function main_cuda(; pilot_samples=20_000, nsamples=100_000)
    CUDA.allowscalar(false)
    device = cuda_device()
    data = simulate_logistic_data(Xoshiro(0x4c4f474953544943), TRUE_COEFFICIENTS)
    prior_scale = 1.5
    context = (; data..., prior_scale)
    prior = SphericalGaussian(zeros(length(TRUE_COEFFICIENTS)), prior_scale)

    # Each round transfers its complete prepared sampler, including the data.
    pilot = sample_on_cuda(
        Xoshiro(0x50494c4f54495331),
        logistic_logtarget,
        context,
        ImportanceSampling(prior; nsamples=pilot_samples),
        device,
    )

    # Adapt between rounds on CPU, then send the fitted proposal back to CUDA.
    host_pilot = pilot |> cpu_device()
    proposal = fit_gaussian_proposal(host_pilot)
    result = sample_on_cuda(
        Xoshiro(0x49534c4f47495431),
        logistic_logtarget,
        context,
        ImportanceSampling(proposal; nsamples),
        device,
    )

    # Samples and log weights stay on CUDA until this explicit transfer.
    host_result = result |> cpu_device()
    summary = summarize_coefficients(host_result)
    pilot_ess = inv(sum(abs2, normalized_weights(host_pilot)))
    print_logistic_summary(
        "CUDA logistic regression posterior",
        summary,
        pilot_ess,
        pilot_samples,
        nsamples,
    )

    return (; pilot, proposal, result, host_result, summary, context, device)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_cuda()
end
