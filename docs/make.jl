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

    return Markdown.parse(
        "| Method | Target forms | Device | Execution policies | Status |\n" *
        "|:--|:--|:--|:--|:--|\n" *
        "| Plain importance sampling | `logtarget(x)` and `logtarget(x, p)` " *
        "| CPU | `threaded=false` serial; `threaded=true` accepted " *
        "($threaded_detail) | **supported** |\n" *
        "| `ProductProposal` | named independent blocks | CPU only | " *
        "coordinator draws in block order | **supported on CPU; typed accelerator rejection verified** |",
    )
end

const PLAIN_IS_CAPABILITY_TABLE = checked_plain_is_capability_table()

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
    ],
    doctest=true,
    checkdocs=:exports,
    linkcheck=true,
    warnonly=false,
)
