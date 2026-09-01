const FIRST_ORDER_GRAMIS_CPU_EVIDENCE =
    "validation/reproducers/first_order_gramis.jl"
const FIRST_ORDER_GRAMIS_CUDA_EVIDENCE =
    "validation/reproducers/cuda_first_order_gramis.jl"
const FIRST_ORDER_GRAMIS_CUDA_HARDWARE = "NVIDIA A100-PCIE-40GB"

const FIRST_ORDER_GRAMIS_CAPABILITY_ROWS = (
    (
        label=:float32_explicit_inplace,
        type=Float32,
        gradient=:explicit_inplace,
        cpu=(status=:supported, evidence=FIRST_ORDER_GRAMIS_CPU_EVIDENCE),
        cuda=(
            status=:supported,
            reason=nothing,
            hardware=FIRST_ORDER_GRAMIS_CUDA_HARDWARE,
            context=:context_free,
            evidence=FIRST_ORDER_GRAMIS_CUDA_EVIDENCE,
        ),
    ),
    (
        label=:float32_explicit_outofplace,
        type=Float32,
        gradient=:explicit_outofplace,
        cpu=(status=:supported, evidence=FIRST_ORDER_GRAMIS_CPU_EVIDENCE),
        cuda=(
            status=:rejected,
            reason=:out_of_place_gradient_cpu_only,
            hardware=FIRST_ORDER_GRAMIS_CUDA_HARDWARE,
            context=:context_free,
            evidence=FIRST_ORDER_GRAMIS_CUDA_EVIDENCE,
        ),
    ),
    (
        label=:float32_logdensityproblems,
        type=Float32,
        gradient=:logdensityproblems,
        cpu=(status=:supported, evidence=FIRST_ORDER_GRAMIS_CPU_EVIDENCE),
        cuda=(
            status=:rejected,
            reason=:gradient_source_cpu_only,
            hardware=FIRST_ORDER_GRAMIS_CUDA_HARDWARE,
            context=:context_free,
            evidence=FIRST_ORDER_GRAMIS_CUDA_EVIDENCE,
        ),
    ),
    (
        label=:float64_explicit_inplace,
        type=Float64,
        gradient=:explicit_inplace,
        cpu=(status=:supported, evidence=FIRST_ORDER_GRAMIS_CPU_EVIDENCE),
        cuda=(
            status=:supported,
            reason=nothing,
            hardware=FIRST_ORDER_GRAMIS_CUDA_HARDWARE,
            context=:context_free,
            evidence=FIRST_ORDER_GRAMIS_CUDA_EVIDENCE,
        ),
    ),
    (
        label=:float64_explicit_outofplace,
        type=Float64,
        gradient=:explicit_outofplace,
        cpu=(status=:supported, evidence=FIRST_ORDER_GRAMIS_CPU_EVIDENCE),
        cuda=(
            status=:rejected,
            reason=:out_of_place_gradient_cpu_only,
            hardware=FIRST_ORDER_GRAMIS_CUDA_HARDWARE,
            context=:context_free,
            evidence=FIRST_ORDER_GRAMIS_CUDA_EVIDENCE,
        ),
    ),
    (
        label=:float64_logdensityproblems,
        type=Float64,
        gradient=:logdensityproblems,
        cpu=(status=:supported, evidence=FIRST_ORDER_GRAMIS_CPU_EVIDENCE),
        cuda=(
            status=:rejected,
            reason=:gradient_source_cpu_only,
            hardware=FIRST_ORDER_GRAMIS_CUDA_HARDWARE,
            context=:context_free,
            evidence=FIRST_ORDER_GRAMIS_CUDA_EVIDENCE,
        ),
    ),
)

function checked_first_order_gramis_capability_rows(
    repository_root=normpath(joinpath(@__DIR__, "..")),
)
    rows = FIRST_ORDER_GRAMIS_CAPABILITY_ROWS
    expected = (
        (:float32_explicit_inplace, Float32, :explicit_inplace, :supported, nothing),
        (
            :float32_explicit_outofplace,
            Float32,
            :explicit_outofplace,
            :rejected,
            :out_of_place_gradient_cpu_only,
        ),
        (
            :float32_logdensityproblems,
            Float32,
            :logdensityproblems,
            :rejected,
            :gradient_source_cpu_only,
        ),
        (:float64_explicit_inplace, Float64, :explicit_inplace, :supported, nothing),
        (
            :float64_explicit_outofplace,
            Float64,
            :explicit_outofplace,
            :rejected,
            :out_of_place_gradient_cpu_only,
        ),
        (
            :float64_logdensityproblems,
            Float64,
            :logdensityproblems,
            :rejected,
            :gradient_source_cpu_only,
        ),
    )
    observed = Tuple(
        (row.label, row.type, row.gradient, row.cuda.status, row.cuda.reason)
        for row in rows
    )
    observed == expected || error(
        "FirstOrderGRAMIS capability rows do not match the documented contract",
    )
    for row in rows
        propertynames(row) == (:label, :type, :gradient, :cpu, :cuda) || error(
            "FirstOrderGRAMIS capability row $(row.label) has invalid fields",
        )
        row.cpu == (status=:supported, evidence=FIRST_ORDER_GRAMIS_CPU_EVIDENCE) ||
            error("FirstOrderGRAMIS CPU metadata mismatch for $(row.label)")
        propertynames(row.cuda) ==
        (:status, :reason, :hardware, :context, :evidence) || error(
            "FirstOrderGRAMIS CUDA metadata is incomplete for $(row.label)",
        )
        row.cuda.hardware == FIRST_ORDER_GRAMIS_CUDA_HARDWARE || error(
            "FirstOrderGRAMIS CUDA hardware mismatch for $(row.label)",
        )
        row.cuda.context === :context_free || error(
            "FirstOrderGRAMIS CUDA context mismatch for $(row.label)",
        )
        row.cuda.evidence == FIRST_ORDER_GRAMIS_CUDA_EVIDENCE || error(
            "FirstOrderGRAMIS CUDA evidence mismatch for $(row.label)",
        )
        for evidence in (row.cpu.evidence, row.cuda.evidence)
            isfile(joinpath(repository_root, evidence)) || error(
                "FirstOrderGRAMIS evidence path does not exist: $evidence",
            )
        end
    end
    return rows
end

struct FirstOrderGRAMISCapabilityTarget{T<:AbstractFloat} end
struct FirstOrderGRAMISCapabilityInPlaceGradient end
struct FirstOrderGRAMISCapabilityOutOfPlaceGradient end
struct FirstOrderGRAMISCapabilityLDPTarget{T<:AbstractFloat} end

function (::FirstOrderGRAMISCapabilityTarget{T})(sample)::T where {T}
    squared_radius = zero(T)
    @inbounds for row in eachindex(sample)
        squared_radius += abs2(sample[row])
    end
    return -T(0.5) * squared_radius
end

function (::FirstOrderGRAMISCapabilityInPlaceGradient)(destination, sample)
    @inbounds for row in eachindex(destination)
        destination[row] = -sample[row]
    end
    return destination
end

(::FirstOrderGRAMISCapabilityOutOfPlaceGradient)(sample) = -sample

const FirstOrderGRAMISCapabilityLDP = ImportanceSamplers.LogDensityProblems
FirstOrderGRAMISCapabilityLDP.capabilities(
    ::Type{<:FirstOrderGRAMISCapabilityLDPTarget},
) = FirstOrderGRAMISCapabilityLDP.LogDensityOrder{1}()
FirstOrderGRAMISCapabilityLDP.dimension(::FirstOrderGRAMISCapabilityLDPTarget) = 2
FirstOrderGRAMISCapabilityLDP.logdensity(
    ::FirstOrderGRAMISCapabilityLDPTarget{T},
    sample,
) where {T} = -T(0.5) * sum(abs2, sample)
FirstOrderGRAMISCapabilityLDP.logdensity_and_gradient(
    ::FirstOrderGRAMISCapabilityLDPTarget{T},
    sample,
) where {T} = (-T(0.5) * sum(abs2, sample), -sample)

function first_order_gramis_capability_target(row)
    value = FirstOrderGRAMISCapabilityTarget{row.type}()
    row.gradient === :explicit_inplace && return LogTarget(
        value;
        grad=FirstOrderGRAMISCapabilityInPlaceGradient(),
    )
    row.gradient === :explicit_outofplace && return LogTarget(
        value;
        grad=FirstOrderGRAMISCapabilityOutOfPlaceGradient(),
    )
    row.gradient === :logdensityproblems && return (
        FirstOrderGRAMISCapabilityLDPTarget{row.type}()
    )
    error("unknown FirstOrderGRAMIS gradient form $(row.gradient)")
end

function first_order_gramis_capability_bank(::Type{T}) where {T}
    return ProposalBank([
        FactorGaussian(T[-1, 0], T[1 0; 0.1 0.8]),
        FactorGaussian(T[1, 0], T[0.9 0; -0.2 1.1]),
    ])
end
