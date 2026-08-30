using ImportanceSamplers
using LinearAlgebra
using Test

const IS = ImportanceSamplers
const FIRST_ORDER_GRAMIS_REPRODUCER_COMMAND =
    "julia --project=validation validation/reproducers/first_order_gramis.jl"

"""
    canonical_gramis_equation_10(means, masses, strength)

Literal GRAMIS equation 10 for a column-major proposal population. It uses
proposal-mass products, singular raw Euclidean distance, and a peer sum.
"""
function canonical_gramis_equation_10(means, masses, strength)
    T = eltype(means)
    dimension, proposal_count = size(means)
    force = zeros(T, size(means))
    for proposal_slot in 1:proposal_count
        for peer_slot in 1:proposal_count
            peer_slot == proposal_slot && continue
            distance2 = zero(T)
            for row in 1:dimension
                distance2 += abs2(
                    means[row, proposal_slot] - means[row, peer_slot],
                )
            end
            denominator = distance2 ^ (T(dimension) / T(2))
            coefficient =
                strength * masses[proposal_slot] * masses[peer_slot] /
                denominator
            for row in 1:dimension
                force[row, proposal_slot] += coefficient *
                    (means[row, proposal_slot] - means[row, peer_slot])
            end
        end
    end
    return force
end

"""
    pooled_whitened_package_force(means, covariances, strength, softening)

Independent package-force oracle. Relative to literal GRAMIS equation 10, it
uses equal-weight pooled-covariance whitening for distance, adds positive
softening, fixes repulsion masses to one, and averages over `M - 1` peers. The
force direction remains the original-coordinate mean difference.
"""
function pooled_whitened_package_force(
    means,
    covariances,
    strength,
    softening,
)
    T = eltype(means)
    dimension, proposal_count = size(means)
    pooled_covariance = dropdims(
        sum(covariances; dims=3);
        dims=3,
    ) / T(proposal_count)
    pooled_factor = cholesky(Hermitian(pooled_covariance)).L
    whitened_means = pooled_factor \ means
    force = zeros(T, size(means))
    softening2 = abs2(softening)
    for proposal_slot in 1:proposal_count
        for peer_slot in 1:proposal_count
            peer_slot == proposal_slot && continue
            distance2 = zero(T)
            for row in 1:dimension
                distance2 += abs2(
                    whitened_means[row, proposal_slot] -
                    whitened_means[row, peer_slot],
                )
            end
            denominator =
                (distance2 + softening2) ^ (T(dimension) / T(2))
            for row in 1:dimension
                force[row, proposal_slot] +=
                    (means[row, proposal_slot] - means[row, peer_slot]) /
                    denominator
            end
        end
    end
    force .*= strength / T(proposal_count - 1)
    return force
end

function package_repulsion(means, factors, strength, softening)
    T = eltype(means)
    dimension, proposal_count = size(means)
    force = similar(means)
    collision_counts = Vector{Int}(undef, proposal_count)
    pooled_covariance = zeros(T, dimension, dimension)
    whitened_means = similar(means)
    IS._repulsion!(
        force,
        collision_counts,
        pooled_covariance,
        whitened_means,
        means,
        factors,
        strength,
        softening,
        IS._SerialCPUExecution(),
    )
    return (; force, collision_counts)
end

function reproducer_fixture(::Type{T}) where {T}
    means = T[-2 0.5 3; 1 -1 2]
    factors = zeros(T, 2, 2, 3)
    factors[:, :, 1] .= T[1.5 0; 0.2 0.7]
    factors[:, :, 2] .= T[0.8 0; -0.1 1.4]
    factors[:, :, 3] .= T[1.2 0; 0.5 0.9]
    covariances = similar(factors)
    for proposal_slot in axes(factors, 3)
        factor = factors[:, :, proposal_slot]
        covariances[:, :, proposal_slot] .= factor * transpose(factor)
    end
    return means, factors, covariances
end

function validate_first_order_gramis_repulsion()
    canonical_terms = (
        distance=:raw_euclidean,
        denominator=:singular_power_d,
        repulsion_masses=:proposal_mass_products,
        peer_reduction=:sum,
        direction=:original_coordinates,
    )
    package_terms = (
        distance=:pooled_whitened,
        denominator=:softened_power_d,
        repulsion_masses=:fixed_one,
        peer_reduction=:average,
        direction=:original_coordinates,
    )
    @test canonical_terms.distance != package_terms.distance
    @test canonical_terms.denominator != package_terms.denominator
    @test canonical_terms.repulsion_masses != package_terms.repulsion_masses
    @test canonical_terms.peer_reduction != package_terms.peer_reduction
    @test canonical_terms.direction == package_terms.direction

    for T in (Float32, Float64)
        means, factors, covariances = reproducer_fixture(T)
        strength = T(0.35)
        softening = T(0.6)
        masses = T[1, 2, 1]
        canonical = canonical_gramis_equation_10(
            means,
            masses,
            strength,
        )
        oracle = pooled_whitened_package_force(
            means,
            covariances,
            strength,
            softening,
        )
        observed = package_repulsion(
            means,
            factors,
            strength,
            softening,
        )
        tolerance = T === Float32 ? T(128) * eps(T) : T(64) * eps(T)
        @test observed.force ≈ oracle rtol = tolerance atol = zero(T)
        @test canonical != oracle
        @test observed.collision_counts == zeros(Int, 3)

        exact_means = repeat(view(means, :, 1), 1, 2)
        exact_factors = factors[:, :, 1:2]
        exact_covariances = covariances[:, :, 1:2]
        exact_oracle = pooled_whitened_package_force(
            exact_means,
            exact_covariances,
            strength,
            softening,
        )
        exact_observed = package_repulsion(
            exact_means,
            exact_factors,
            strength,
            softening,
        )
        @test exact_oracle == zeros(T, 2, 2)
        @test exact_observed.force == exact_oracle
        @test exact_observed.collision_counts == [1, 1]
    end
    return (
        canonical=:literal_gramis_equation_10,
        package=:pooled_whitened_softened_peer_average,
        scalar_types=(Float32, Float64),
        status=:passed,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    display(validate_first_order_gramis_repulsion())
end
