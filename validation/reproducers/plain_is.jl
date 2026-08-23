using DensityInterface
using ImportanceSamplers
using Pkg
using Random

const VALIDATION_SEED = 0x76c4f2a19b038d45
const SCALAR_TYPE = Float64
const EQUAL_SAMPLE_COUNT = 8_192
const GAUSSIAN_SAMPLE_COUNT = 100_000
const EQUAL_LOGWEIGHT_TOLERANCE = 32eps(SCALAR_TYPE)
const EQUAL_MEAN_TOLERANCE = 0.04
const GAUSSIAN_MEAN_TOLERANCE = 0.02
const GAUSSIAN_LOGNORMALIZER_TOLERANCE = 0.015
const VALIDATION_COMMAND =
    "julia --project=validation validation/reproducers/plain_is.jl"

struct GaussianProposal{T<:AbstractFloat}
    mean::T
    scale::T

    function GaussianProposal(mean::T, scale::T) where {T<:AbstractFloat}
        scale > zero(T) || throw(ArgumentError("scale must be positive"))
        return new{T}(mean, scale)
    end
end

function Random.rand(
    rng::Random.AbstractRNG,
    proposal::GaussianProposal{T},
) where {T}
    return proposal.mean + proposal.scale * randn(rng, T)
end

function gaussian_logdensity(x, mean, scale)
    standardized = (x - mean) / scale
    return -oftype(standardized, 0.5) * abs2(standardized) - log(scale) -
           oftype(standardized, 0.5 * log(2pi))
end

function DensityInterface.logdensityof(proposal::GaussianProposal, x::Real)
    return gaussian_logdensity(x, proposal.mean, proposal.scale)
end

function require_identity(label, observed, expected, tolerance)
    isapprox(observed, expected; atol=tolerance, rtol=zero(tolerance)) || error(
        "$label failed: observed=$observed, expected=$expected, " *
        "absolute_tolerance=$tolerance",
    )
    return nothing
end

function print_versions()
    wanted = Set((
        "DensityInterface",
        "ImportanceSamplers",
        "LogDensityProblems",
        "LogExpFunctions",
        "MLDataDevices",
    ))
    versions = sort!(
        [
            (dependency.name, dependency.version)
            for dependency in values(Pkg.dependencies())
            if dependency.name in wanted
        ];
        by=first,
    )
    println("Julia version: ", VERSION)
    for (name, version) in versions
        println(name, " version: ", something(version, "unversioned"))
    end
    return nothing
end

function validate_equal_normalized_target()
    proposal = GaussianProposal(0.0, 1.0)
    logtarget(x) = gaussian_logdensity(x, 0.0, 1.0)
    algorithm = ImportanceSampling(proposal; nsamples=EQUAL_SAMPLE_COUNT)
    result = importance_sample(
        Xoshiro(VALIDATION_SEED),
        logtarget,
        algorithm;
        threaded=false,
    )

    maximum(abs, result.logweights) <= EQUAL_LOGWEIGHT_TOLERANCE || error(
        "proposal-equals-target identity failed: maximum absolute raw log " *
        "weight was $(maximum(abs, result.logweights))",
    )
    require_identity(
        "proposal-equals-target log normalizer",
        lognormalizer(result),
        0.0,
        EQUAL_LOGWEIGHT_TOLERANCE,
    )
    require_identity(
        "proposal-equals-target mean",
        sum(result.samples .* normalized_weights(result)),
        0.0,
        EQUAL_MEAN_TOLERANCE,
    )
    println("PASS analytic identity: normalized proposal equals normalized target")
    return nothing
end

function validate_gaussian_mean_and_normalizer()
    target_mean = 0.75
    target_scale = 1.25
    target_lognormalizer = 0.7
    proposal = GaussianProposal(0.0, 2.0)
    logtarget(x) = target_lognormalizer +
                   gaussian_logdensity(x, target_mean, target_scale)
    algorithm = ImportanceSampling(proposal; nsamples=GAUSSIAN_SAMPLE_COUNT)
    result = importance_sample(
        Xoshiro(VALIDATION_SEED),
        logtarget,
        algorithm;
        threaded=false,
    )

    estimated_mean = sum(result.samples .* normalized_weights(result))
    estimated_lognormalizer = lognormalizer(result)
    require_identity(
        "Gaussian weighted mean",
        estimated_mean,
        target_mean,
        GAUSSIAN_MEAN_TOLERANCE,
    )
    require_identity(
        "Gaussian log normalizer",
        estimated_lognormalizer,
        target_lognormalizer,
        GAUSSIAN_LOGNORMALIZER_TOLERANCE,
    )
    println("PASS analytic identity: Gaussian weighted mean")
    println("PASS analytic identity: Gaussian log normalizer")
    println("  observed weighted mean: ", estimated_mean)
    println("  observed log normalizer: ", estimated_lognormalizer)
    return nothing
end

function main()
    println("ImportanceSamplers plain-IS validation")
    println("classification: analytic identities")
    println("command: ", VALIDATION_COMMAND)
    println("seed: 0x", string(VALIDATION_SEED; base=16, pad=16))
    println("scalar type: ", SCALAR_TYPE)
    println("proposal-equals-target sample count: ", EQUAL_SAMPLE_COUNT)
    println("Gaussian sample count: ", GAUSSIAN_SAMPLE_COUNT)
    println("proposal-equals-target raw-logweight tolerance: ", EQUAL_LOGWEIGHT_TOLERANCE)
    println("proposal-equals-target mean tolerance: ", EQUAL_MEAN_TOLERANCE)
    println("Gaussian mean tolerance: ", GAUSSIAN_MEAN_TOLERANCE)
    println(
        "Gaussian log-normalizer tolerance: ",
        GAUSSIAN_LOGNORMALIZER_TOLERANCE,
    )
    println(
        "tolerance rationale: deterministic absolute tolerances cover the " *
        "fixed Monte Carlo budgets while remaining small relative to unit scale",
    )
    print_versions()
    validate_equal_normalized_target()
    validate_gaussian_mean_and_normalizer()
    println("All analytic identities passed.")
    return nothing
end

main()
