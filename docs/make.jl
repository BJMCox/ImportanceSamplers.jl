using DensityInterface
using Documenter
using ImportanceSamplers
using Markdown
using Random
import MLDataDevices

include(joinpath(@__DIR__, "..", "validation", "cuda_plain_is_capabilities.jl"))
include(joinpath(@__DIR__, "..", "validation", "static_mis_capabilities.jl"))
include(joinpath(@__DIR__, "..", "validation", "dm_pmc_capabilities.jl"))

struct CapabilityGaussian end

Random.rand(rng::Random.AbstractRNG, ::CapabilityGaussian) = randn(rng)
DensityInterface.logdensityof(::CapabilityGaussian, x::Real) =
    -0.5 * abs2(x) - 0.5 * log(2pi)

mutable struct CapabilityRNGSentinel <: Random.AbstractRNG; consumed::Bool; end
mark_rng_consumed(rng::CapabilityRNGSentinel) =
    (rng.consumed = true; error("capability preparation consumed its RNG"))
Random.rand(rng::CapabilityRNGSentinel, args...) = mark_rng_consumed(rng)
Random.rand!(rng::CapabilityRNGSentinel, args...) = mark_rng_consumed(rng)
Random.randn!(rng::CapabilityRNGSentinel, args...) = mark_rng_consumed(rng)

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

function checked_static_mis_capability_table()
    for (bank_index, row) in enumerate(STATIC_MIS_CAPABILITY_ROWS)
        bank = row.factory(Float64)
        proposal = first(bank.proposals)
        target = sample -> DensityInterface.logdensityof(proposal, sample)
        for (scheme_index, scheme) in enumerate(STATIC_MIS_COMPLETE_SCHEMES)
            result = importance_sample(
                Xoshiro(10bank_index + scheme_index), target,
                ImportanceSampling(bank; nsamples=7, mis_scheme=scheme.value);
                threaded=false,
            )
            length(result) == 7 || error("static-MIS CPU capability check failed")
        end
        if row.device !== :supported
            sampler = prepare_sampler(
                Xoshiro(0x53544154), capability_product_target,
                ImportanceSampling(bank; nsamples=4); threaded=true,
            )
            caught = try CapabilityAccelerator()(sampler); nothing catch error; error end
            caught isa SamplerDeviceError && caught.reason === row.device ||
                error("static-MIS accelerator rejection check failed")
        end
    end
    for rejection in STATIC_MIS_PREPARATION_REJECTIONS
        rng = CapabilityRNGSentinel(false)
        caught = try prepare_sampler(
            rng, capability_product_target,
            ImportanceSampling(rejection.factory(); nsamples=4); threaded=false,
        ); nothing catch error; error end
        caught isa rejection.error || error("static-MIS preparation rejection check failed")
        rng.consumed && error("static-MIS preparation rejection consumed its RNG")
    end
    scheme_names = join((scheme.name for scheme in STATIC_MIS_COMPLETE_SCHEMES), ", ")
    function accelerator(row)
        row.device !== :supported && return "rejected: `$(row.device)`"
        if isnothing(row.direct)
            hasproperty(row, :evidence) ||
                return "CUDA execution; not directly hardware-validated"
            evidence = row.evidence
            types = join(string.(evidence.types), " and ")
            schemes = join(string.(evidence.schemes), ", ")
            return "$(evidence.hardware) execution with $types for $schemes; " *
                   "other schemes not directly hardware-validated"
        end
        direct = row.direct
        types = join(string.(direct.types), " and ")
        return "$(direct.hardware) execution with $types across $(length(direct.schemes)) schemes"
    end
    rows = join(
        ("| $(r.bank) | $(r.cpu) | $(accelerator(r)) |" for r in STATIC_MIS_CAPABILITY_ROWS),
        '\n',
    )
    rejections = join(
        (
            "| $(r.input) | `$(nameof(r.error))` before RNG use |" for
            r in STATIC_MIS_PREPARATION_REJECTIONS
        ),
        '\n',
    )
    return Markdown.parse(
        "All rows support the four complete schemes: $scheme_names.\n\n" *
        "| Proposal bank | CPU | CUDA status/evidence |\n" *
        "|:--|:--|:--|\n" * rows * "\n\n" *
        "| Invalid positive-mass bank | Preparation result |\n" *
        "|:--|:--|\n" * rejections,
    )
end

const STATIC_MIS_CAPABILITY_TABLE = checked_static_mis_capability_table()

function dm_pmc_capability_target(sample::AbstractVector{T})::T where {T}
    squared_radius = zero(T)
    for coordinate in eachindex(sample)
        squared_radius += abs2(sample[coordinate])
    end
    return -T(0.5) * squared_radius
end

function checked_dm_pmc_capability_table()
    core_labels = (
        :float32_diagonal,
        :float64_diagonal,
        :float32_factor,
        :float64_factor,
    )
    rows = filter(row -> row.label in core_labels, DM_PMC_CUDA_CAPABILITY_ROWS)
    length(rows) == length(core_labels) || error(
        "DM-PMC capability metadata is missing a documented CPU/CUDA row",
    )
    table_rows = map(rows) do row
        schedule = [20, 24]
        bank = dm_pmc_validation_bank(row.type, Val(row.bank))
        sampler = prepare_sampler(
            Xoshiro(0x444f4353444d504d),
            dm_pmc_capability_target,
            DeterministicMixturePMC(
                bank;
                rounds=length(schedule),
                round_size=schedule,
            );
            threaded=true,
        )
        result = importance_sample!(sampler)
        length(result) == sum(schedule) || error(
            "DM-PMC docs-build CPU check returned the wrong count",
        )
        result.diagnostics.round_sizes == schedule || error(
            "DM-PMC docs-build CPU check returned the wrong schedule",
        )
        length(current_proposal(sampler).proposals) == length(bank.proposals) || error(
            "DM-PMC docs-build CPU check returned the wrong proposal snapshot",
        )
        bank_name = row.bank === :diagonal ? "diagonal Gaussian" : "factor Gaussian"
        "| `$(row.type)` | $bank_name | public execution during docs build | " *
        "$(DM_PMC_CUDA_HARDWARE) reproducer |"
    end
    return Markdown.parse(
        "| Scalar type | Proposal bank | CPU evidence | CUDA evidence |\n" *
        "|:--|:--|:--|:--|\n" * join(table_rows, '\n'),
    )
end

const DM_PMC_CAPABILITY_TABLE = checked_dm_pmc_capability_table()

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
            "Static multiple importance sampling" => "methods/static_mis.md",
            "Deterministic-mixture population Monte Carlo" => "methods/dm_pmc.md",
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
    linkcheck_ignore=[
        r"^https://github\.com/BJMCox/ImportanceSamplers\.jl/blob/main/validation/reproducers/(cuda_)?static_mis\.jl$",
        r"^https://github\.com/BJMCox/ImportanceSamplers\.jl/blob/main/(benchmark/dm_pmc|examples/(dm_pmc|numerical_integration)|validation/reproducers/(cuda_dm_pmc|dm_pmc_global))\.jl$",
    ],
    warnonly=false,
)
