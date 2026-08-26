const DM_PMC_CUDA_HARDWARE = "NVIDIA A100-PCIE-40GB"
const DM_PMC_CUDA_DIMENSION = 4
const DM_PMC_CUDA_PROPOSALS = 4

const DM_PMC_CUDA_CAPABILITY_ROWS = (
    (label=:float32_diagonal, type=Float32, bank=:diagonal),
    (label=:float64_diagonal, type=Float64, bank=:diagonal),
    (label=:float32_factor, type=Float32, bank=:factor),
    (label=:float64_factor, type=Float64, bank=:factor),
    (label=:unequal_masses_with_zero, type=Float64, bank=:diagonal),
    (label=:unequal_round_sizes, type=Float64, bank=:diagonal),
    (label=:duplicate_resampled_ancestors, type=Float64, bank=:diagonal),
    (label=:repeated_prepared_execution, type=Float64, bank=:factor),
)

dm_pmc_diagonal_tolerance(::Type{T}) where {T} = T(4096) * eps(T)

# A factor path performs one forward solve per proposal-density term. Scale the
# diagonal tolerance by dimension to cover that documented operation count.
dm_pmc_factor_tolerance(::Type{T}, dimension) where {T} =
    T(4096 * dimension) * eps(T)

function dm_pmc_validation_bank(::Type{T}, ::Val{:diagonal}; zero_mass=false) where {T}
    proposals = [
        DiagonalGaussian(
            T[-0.9, -0.4, 0.1, 0.6] .+ T(0.35 * (slot - 1)),
            T[0.65, 0.8, 1.05, 1.2] .+ T(0.03 * slot),
        ) for slot in 1:DM_PMC_CUDA_PROPOSALS
    ]
    masses = zero_mass ? T[1, 0, 3, 2] : T[1, 2, 3, 4]
    return ProposalBank(proposals, masses)
end

function dm_pmc_validation_bank(::Type{T}, ::Val{:factor}; zero_mass=false) where {T}
    proposals = [
        FactorGaussian(
            T[-0.9, -0.4, 0.1, 0.6] .+ T(0.35 * (slot - 1)),
            T[
                0.70 + 0.02slot 0 0 0
                0.08 0.85 + 0.03slot 0 0
                -0.04 0.12 1.00 + 0.02slot 0
                0.06 -0.05 0.10 1.15 + 0.01slot
            ],
        ) for slot in 1:DM_PMC_CUDA_PROPOSALS
    ]
    masses = zero_mass ? T[1, 0, 3, 2] : T[1, 2, 3, 4]
    return ProposalBank(proposals, masses)
end
