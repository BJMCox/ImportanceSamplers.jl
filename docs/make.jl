using DensityInterface
using Documenter
using ImportanceSamplers
using Markdown
using Random
import MLDataDevices

include(joinpath(@__DIR__, "..", "validation", "cuda_plain_is_capabilities.jl"))

struct CapabilityGaussian end

Random.rand(rng::Random.AbstractRNG, ::CapabilityGaussian) = randn(rng)
DensityInterface.logdensityof(::CapabilityGaussian, x::Real) =
    -0.5 * abs2(x) - 0.5 * log(2pi)

struct CapabilityAccelerator <: MLDataDevices.AbstractAcceleratorDevice end
MLDataDevices.functional(::CapabilityAccelerator) = true
capability_product_target(sample)::Float64 = 0.0

(::CapabilityAccelerator)(proposal::CapabilityGaussian) = deepcopy(proposal)

function native_capability_proposals()
    factor = Float64[
        1.0 0.0 0.0 0.0
        0.2 1.1 0.0 0.0
        0.0 0.1 0.9 0.0
        0.0 0.0 0.2 1.2
    ]
    return (
        scalar=SphericalGaussian(0.0, 1.0),
        spherical_vector=SphericalGaussian(zeros(2), 1.0),
        vector=DiagonalGaussian(zeros(2), [0.5, 1.5]),
        factor_vector=FactorGaussian(zeros(2), [1.0 0.0; 0.25 1.2]),
        identity=TransformedProposal(
            SphericalGaussian(0.0, 1.0),
            IdentityTransform(),
        ),
        positive=TransformedProposal(
            SphericalGaussian(0.0, 1.0),
            PositiveTransform(),
        ),
        softplus=TransformedProposal(
            SphericalGaussian(0.0, 1.0),
            SoftplusTransform(),
        ),
        lower_interval=TransformedProposal(
            SphericalGaussian(0.0, 1.0),
            IntervalTransform(0.0, nothing),
        ),
        upper_interval=TransformedProposal(
            SphericalGaussian(0.0, 1.0),
            IntervalTransform(nothing, 1.0),
        ),
        interval=TransformedProposal(
            SphericalGaussian(0.0, 1.0),
            IntervalTransform(-1.0, 2.0),
        ),
        simplex=TransformedProposal(
            SphericalGaussian(zeros(2), 1.0),
            SimplexTransform(3),
        ),
        complete_flat=TransformedProposal(
            FactorGaussian(zeros(4), factor),
            (
                weights=(1:2 => SimplexTransform(3)),
                rate=(3 => PositiveTransform()),
                offset=(4 => IdentityTransform()),
            ),
        ),
    )
end

function checked_native_cpu_proposals()
    proposals = native_capability_proposals()
    for (index, proposal) in enumerate(values(proposals))
        target = let proposal = proposal
            sample -> DensityInterface.logdensityof(proposal, sample)
        end
        result = importance_sample(
            Xoshiro(index),
            target,
            ImportanceSampling(proposal; nsamples=4);
            threaded=false,
        )
        length(result) == 4 || error("native CPU capability check returned wrong count")
        maximum(abs, result.logweights) <= 8192eps(Float64) || error(
            "native CPU capability check did not preserve proposal identity",
        )
    end
    return keys(proposals)
end

const NATIVE_CAPABILITY_NAMES = (
    scalar="spherical scalar Gaussian",
    spherical_vector="spherical vector Gaussian",
    vector="diagonal vector Gaussian",
    factor_vector="factor Gaussian",
    identity="identity transform",
    positive="positive transform",
    softplus="softplus transform",
    lower_interval="lower-bounded interval",
    upper_interval="upper-bounded interval",
    interval="bounded interval",
    simplex="simplex transform",
    complete_flat="complete flat layout",
)

function capability_names(labels)
    all(label -> hasproperty(NATIVE_CAPABILITY_NAMES, label), labels) ||
        error("capability metadata contains an unknown label")
    return join((getproperty(NATIVE_CAPABILITY_NAMES, label) for label in labels), ", ")
end

function checked_plain_is_capability_table()
    proposal = CapabilityGaussian()
    algorithm = ImportanceSampling(proposal; nsamples=8)
    context_free(x) = DensityInterface.logdensityof(proposal, x)
    contextual(x, p) = DensityInterface.logdensityof(proposal, x) + p.shift

    serial = prepare_sampler(Xoshiro(0x1), context_free, algorithm; threaded=false)
    threaded = prepare_sampler(
        Xoshiro(0x2), contextual, (shift=0.0,), algorithm; threaded=true,
    )
    serial_result = importance_sample!(serial)
    threaded_result = importance_sample!(threaded)
    length(serial_result) == 8 || error("serial capability check returned wrong count")
    serial_result.diagnostics.threaded === false || error(
        "serial capability check did not retain threaded=false",
    )
    serial_result.diagnostics.execution === :serial || error(
        "threaded=false capability check did not execute serially",
    )
    length(threaded_result) == 8 || error(
        "thread-requested capability check returned wrong count",
    )
    threaded_result.diagnostics.threaded === true || error(
        "thread-requested capability check did not retain threaded=true",
    )
    expected_threaded_execution =
        Threads.nthreads(:default) > 1 ? :threaded : :serial
    threaded_result.diagnostics.execution === expected_threaded_execution || error(
        "thread-requested capability check used unexpected execution mode",
    )
    threaded_detail = expected_threaded_execution === :threaded ?
                      "threaded execution verified" :
                      "one-thread serial fallback verified"

    product = ProductProposal((left=CapabilityGaussian(), right=CapabilityGaussian()))
    product_cases = (
        product=product,
        named_layout=TransformedProposal(product, (left=PositiveTransform(),)),
    )
    for (index, product_case) in enumerate(values(product_cases))
        product_cpu = importance_sample(
            Xoshiro(index + 2),
            capability_product_target,
            ImportanceSampling(product_case; nsamples=2);
            threaded=false,
        )
        length(product_cpu) == 2 || error(
            "product CPU capability check returned wrong count",
        )
        product_sampler = prepare_sampler(
            Xoshiro(index + 2),
            capability_product_target,
            ImportanceSampling(product_case; nsamples=1);
            threaded=true,
        )
        product_error = try
            CapabilityAccelerator()(product_sampler)
            nothing
        catch error
            error
        end
        product_error isa SamplerDeviceError || error(
            "product accelerator capability check did not return SamplerDeviceError",
        )
        product_error.reason === :product_proposal_cpu_only || error(
            "product accelerator capability check returned the wrong reason",
        )
    end

    generic_source = prepare_sampler(Xoshiro(0x4), context_free, algorithm; threaded=true)
    generic_error = try
        CapabilityAccelerator()(generic_source)
        nothing
    catch error
        error
    end
    generic_error isa SamplerDeviceError || error(
        "generic accelerator capability check did not return SamplerDeviceError",
    )
    generic_error.reason === :generic_proposal_cpu_only || error(
        "generic accelerator capability check returned the wrong reason",
    )

    cpu_native = checked_native_cpu_proposals()
    all(label -> label in cpu_native, CUDA_PLAIN_IS_A100_CASES) ||
        error("A100 metadata contains a case missing from the CPU check")
    other_native = filter(label -> !(label in CUDA_PLAIN_IS_A100_CASES), cpu_native)
    a100_names = capability_names(CUDA_PLAIN_IS_A100_CASES)
    other_names = capability_names(other_native)
    a100_types = join(string.(CUDA_PLAIN_IS_A100_TYPES), " and ")

    return Markdown.parse(
        "| Proposal or transform | Docs-build CPU check | CUDA status/evidence |\n" *
        "|:--|:--|:--|\n" *
        "| Generic normalized proposal | serial and threaded (`$threaded_detail`) " *
        "| rejected: generic proposal is CPU-only |\n" *
        "| $a100_names | serial execution | A100 execution with $a100_types |\n" *
        "| $other_names | serial execution | not A100-validated |\n" *
        "| `ProductProposal` and named product layout | serial execution " *
        "| rejected: CPU-only proposal |",
    )
end

const NATIVE_PLAIN_IS_CAPABILITY_TABLE = checked_plain_is_capability_table()

makedocs(
    modules=[ImportanceSamplers],
    sitename="ImportanceSamplers.jl",
    format=Documenter.HTML(
        prettyurls=false,
        canonical=nothing,
        edit_link=nothing,
        repolink=nothing,
    ),
    build=mktempdir(),
    remotes=nothing,
    pages=[
        "Home" => "index.md",
        "Methods" => [
            "Plain importance sampling" => "methods/importance_sampling.md",
        ],
        "Guides" => [
            "Native proposals" => "guide/native_proposals.md",
            "Transforms" => "guide/transforms.md",
            "Accelerators" => "guide/accelerators.md",
        ],
        "Reference" => "reference.md",
    ],
    doctest=true,
    checkdocs=:exports,
    linkcheck=true,
    warnonly=false,
)
