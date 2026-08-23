using DensityInterface
using Documenter
using ImportanceSamplers
using Markdown
using Random

struct CapabilityGaussian end

Random.rand(rng::Random.AbstractRNG, ::CapabilityGaussian) = randn(rng)
DensityInterface.logdensityof(::CapabilityGaussian, x::Real) =
    -0.5 * abs2(x) - 0.5 * log(2pi)

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

    return Markdown.parse(
        "| Method | Target forms | Device | Execution policies | Status |\n" *
        "|:--|:--|:--|:--|:--|\n" *
        "| Plain importance sampling | `logtarget(x)` and `logtarget(x, p)` " *
        "| CPU | `threaded=false` serial; `threaded=true` accepted " *
        "($threaded_detail) | **supported** |",
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
