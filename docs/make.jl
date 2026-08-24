using DensityInterface
using Documenter
using ImportanceSamplers
using Markdown
using Random

const MLDataDevices = ImportanceSamplers.MLDataDevices

struct CapabilityGaussian end

Random.rand(rng::Random.AbstractRNG, ::CapabilityGaussian) = randn(rng)
DensityInterface.logdensityof(::CapabilityGaussian, x::Real) =
    -0.5 * abs2(x) - 0.5 * log(2pi)

struct CapabilityAccelerator <: MLDataDevices.AbstractAcceleratorDevice end
MLDataDevices.functional(::CapabilityAccelerator) = true
capability_product_target(sample)::Float64 = 0.0

(::CapabilityAccelerator)(proposal::CapabilityGaussian) = deepcopy(proposal)

function checked_native_cpu_proposals()
    factor = Float64[
        1.0 0.0 0.0 0.0
        0.2 1.1 0.0 0.0
        0.0 0.1 0.9 0.0
        0.0 0.0 0.2 1.2
    ]
    proposals = (
        SphericalGaussian(0.0, 1.0),
        SphericalGaussian(zeros(2), 1.0),
        DiagonalGaussian(zeros(2), [0.5, 1.5]),
        FactorGaussian(zeros(2), [1.0 0.0; 0.25 1.2]),
        TransformedProposal(SphericalGaussian(0.0, 1.0), PositiveTransform()),
        TransformedProposal(SphericalGaussian(0.0, 1.0), SoftplusTransform()),
        TransformedProposal(
            SphericalGaussian(0.0, 1.0),
            IntervalTransform(0.0, nothing),
        ),
        TransformedProposal(
            SphericalGaussian(0.0, 1.0),
            IntervalTransform(nothing, 1.0),
        ),
        TransformedProposal(
            SphericalGaussian(0.0, 1.0),
            IntervalTransform(-1.0, 2.0),
        ),
        TransformedProposal(
            SphericalGaussian(zeros(2), 1.0),
            SimplexTransform(3),
        ),
        TransformedProposal(
            FactorGaussian(zeros(4), factor),
            (
                weights=(1:2 => SimplexTransform(3)),
                rate=(3 => PositiveTransform()),
                offset=(4 => IdentityTransform()),
            ),
        ),
    )
    for (index, proposal) in pairs(proposals)
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
    return nothing
end

function checked_plain_is_capability_table()
    proposal = CapabilityGaussian()
    algorithm = ImportanceSampling(proposal; nsamples=8)
    context_free(x) = DensityInterface.logdensityof(proposal, x)
    contextual(x, p) = DensityInterface.logdensityof(proposal, x) + p.shift

    serial = prepare_sampler(
        Xoshiro(0x1),
        context_free,
        algorithm;
        threaded=false,
    )
    threaded = prepare_sampler(
        Xoshiro(0x2),
        contextual,
        (shift=0.0,),
        algorithm;
        threaded=true,
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
    product_sampler = prepare_sampler(
        Xoshiro(0x3),
        capability_product_target,
        ImportanceSampling(product; nsamples=1);
        threaded=true,
    )
    product_cpu = importance_sample(
        Xoshiro(0x3),
        capability_product_target,
        ImportanceSampling(product; nsamples=2);
        threaded=false,
    )
    length(product_cpu) == 2 || error(
        "ProductProposal CPU capability check returned wrong count",
    )
    product_error = try
        CapabilityAccelerator()(product_sampler)
        nothing
    catch error
        error
    end
    product_error isa SamplerDeviceError || error(
        "ProductProposal accelerator capability check did not return SamplerDeviceError",
    )
    product_error.reason === :product_proposal_cpu_only || error(
        "ProductProposal accelerator capability check returned the wrong reason",
    )

    generic_source = prepare_sampler(
        Xoshiro(0x4),
        context_free,
        algorithm;
        threaded=true,
    )
    generic_error = try
        CapabilityAccelerator()(generic_source)
        nothing
    catch error
        error
    end
    generic_error isa SamplerDeviceError || error(
        "generic accelerator capability check did not return SamplerDeviceError",
    )
    generic_error.reason === :accelerator_rng_unavailable || error(
        "generic accelerator capability check returned the wrong reason",
    )

    checked_native_cpu_proposals()

    cuda_reproducer = joinpath(
        @__DIR__,
        "..",
        "validation",
        "reproducers",
        "cuda_plain_is.jl",
    )
    isfile(cuda_reproducer) || error("CUDA capability reproducer is missing")

    return Markdown.parse(
        "| Proposal and layout | CPU | CUDA | Other accelerators | Evidence |\n" *
        "|:--|:--|:--|:--|:--|\n" *
        "| Generic normalized proposal | serial and threaded (`$threaded_detail`) " *
        "| no native random-buffer contract | unclaimed | docs-build execution and typed rejection |\n" *
        "| Native spherical, diagonal, and factor Gaussian | serial and threaded " *
        "| supported with a device-compatible target | unclaimed | docs-build CPU execution; A100 reproducer |\n" *
        "| Native identity, positive, softplus, interval, simplex, and complete flat layout " *
        "| serial and threaded | supported with a device-compatible target | unclaimed " *
        "| docs-build CPU execution; A100 reproducer |\n" *
        "| `ProductProposal` and named product layout | coordinator draws blocks in field order " *
        "| rejected | unclaimed | docs-build execution and typed rejection |",
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
