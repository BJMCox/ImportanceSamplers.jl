using ImportanceSamplers
using LinearAlgebra
using Random
using Test
import DensityInterface

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

function covariance_reproducer_state(::Type{T}) where {T}
    bank = ProposalBank([
        FactorGaussian(T[10], reshape(T[2], 1, 1)),
        FactorGaussian(T[20], reshape(T[3], 1, 1)),
        FactorGaussian(T[100], reshape(T[4], 1, 1)),
    ])
    sampler = prepare_sampler(
        Xoshiro(0x434149534f524143),
        LogTarget(CausalTarget(); grad=causal_gradient!),
        FirstOrderGRAMIS(
            bank;
            rounds=1,
            round_size=12,
            repulsion_strength=zero(T),
            covariance_ess_threshold=3,
        );
        threaded=false,
    )
    state = sampler.method_state
    state.workspace.samples .= reshape(
        T[0, 2, 4, 6, 0, 2, 4, 6, 0, 1, 2, 4],
        1,
        :,
    )
    state.workspace.local_logweights .= T[
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -Inf,
        log(T(8)),
        0,
        0,
        0,
    ]
    return state
end

function literal_weighted_variance(samples, logweights, power, center)
    shifted = exp.(power .* (logweights .- maximum(logweights)))
    weights = shifted ./ sum(shifted)
    return sum(weights .* abs2.(samples .- center)), weights
end

function validate_first_order_gramis_covariance_rules()
    for T in (Float32, Float64)
        state = covariance_reproducer_state(T)
        IS._fit_local_covariances!(state, 1, IS._SerialCPUExecution())
        workspace = state.workspace

        raw_samples = T[0, 2, 4, 6]
        raw_one, _ = literal_weighted_variance(
            raw_samples,
            zeros(T, 4),
            one(T),
            T(10),
        )
        raw_two, _ = literal_weighted_variance(
            raw_samples,
            T[0, 0, 0, -Inf],
            one(T),
            T(20),
        )
        @test workspace.covariances[1, 1, 1] ≈ raw_one rtol = 8eps(T)
        @test workspace.covariances[1, 1, 2] ≈ raw_two rtol = 8eps(T)

        power = workspace.tempering_powers[3]
        tempered_samples = T[0, 1, 2, 4]
        tempered_logs = T[log(T(8)), 0, 0, 0]
        _, tempered_weights = literal_weighted_variance(
            tempered_samples,
            tempered_logs,
            power,
            zero(T),
        )
        tempered_center = sum(tempered_weights .* tempered_samples)
        tempered_variance, _ = literal_weighted_variance(
            tempered_samples,
            tempered_logs,
            power,
            tempered_center,
        )
        @test zero(T) < power < one(T)
        @test inv(sum(abs2, tempered_weights)) >= T(3)
        @test workspace.covariances[1, 1, 3] ≈ tempered_variance rtol = 64eps(T)
    end
    return (
        raw_center=:frozen_proposal_mean,
        tempered_center=:tempered_weighted_mean,
        scalar_types=(Float32, Float64),
        status=:passed,
    )
end

mutable struct CausalPrefilledRNG{T} <: Random.AbstractRNG
    batches::Vector{Vector{T}}
    next_batch::Int
end

function Random.randn!(rng::CausalPrefilledRNG, destination::AbstractArray)
    batch = rng.batches[rng.next_batch]
    length(batch) == length(destination) || throw(
        DimensionMismatch("causal normal batch has the wrong length"),
    )
    copyto!(destination, 1, batch, 1, length(destination))
    rng.next_batch += 1
    return destination
end

struct CausalTarget end

(::CausalTarget)(sample) = -0.5 * abs2(only(sample))

function causal_gradient!(destination, sample)
    destination[1] = -only(sample)
    return destination
end

function causal_bank(::Type{T}) where {T}
    return ProposalBank([
        FactorGaussian(T[-0.2], reshape(T[1], 1, 1)),
        FactorGaussian(T[0.2], reshape(T[1], 1, 1)),
    ])
end

function causal_logaddexp(left, right)
    maximum_value = max(left, right)
    return maximum_value + log(
        exp(left - maximum_value) + exp(right - maximum_value),
    )
end

function causal_gaussian_logdensity(mean, factor, sample)
    return -0.5log(2pi) - log(factor) - 0.5abs2((sample - mean) / factor)
end

function causal_mixture_logdensity(population, sample, counts)
    total = sum(counts)
    terms = map(eachindex(counts)) do slot
        log(counts[slot] / total) + causal_gaussian_logdensity(
            population.locations[slot],
            population.factors[slot],
            sample,
        )
    end
    return causal_logaddexp(terms...)
end

function literal_causal_transition(population, normals, assignments)
    T = eltype(normals)
    samples = map(eachindex(normals)) do index
        slot = assignments[index]
        population.locations[slot] + population.factors[slot] * normals[index]
    end
    counts = [count(==(slot), assignments) for slot in 1:2]
    denominators = map(samples) do sample
        causal_mixture_logdensity(population, sample, counts)
    end
    target_values = map(sample -> -T(0.5) * abs2(sample), samples)
    returned_logweights = target_values .- denominators
    local_logweights = map(eachindex(samples)) do index
        slot = assignments[index]
        target_values[index] - causal_gaussian_logdensity(
            population.locations[slot],
            population.factors[slot],
            samples[index],
        )
    end

    covariance_rate = T(0.4)
    regularization = T(0.01)
    candidate_factors = similar(population.factors)
    for slot in 1:2
        indices = findall(==(slot), assignments)
        shifted = exp.(local_logweights[indices] .-
                       maximum(local_logweights[indices]))
        weights = shifted ./ sum(shifted)
        ess = inv(sum(abs2, weights))
        @test ess >= T(2)
        estimate = sum(weights .* abs2.(samples[indices] .-
                                       population.locations[slot]))
        old_variance = abs2(population.factors[slot])
        blended = (one(T) - covariance_rate) * old_variance +
                  covariance_rate * estimate + regularization * old_variance
        candidate_factors[slot] = sqrt(blended)
    end

    gradient_locations = similar(population.locations)
    for slot in 1:2
        frozen = population.locations[slot]
        move = abs2(population.factors[slot]) * (-frozen)
        gradient_locations[slot] = frozen
        for trial in 1:4
            candidate = frozen + ldexp(one(T), 1 - trial) * move
            if -T(0.5) * abs2(candidate) >= -T(0.5) * abs2(frozen)
                gradient_locations[slot] = candidate
                break
            end
        end
    end

    pooled_factor = sqrt(sum(abs2, population.factors) / T(2))
    whitened = population.locations ./ pooled_factor
    repulsion_strength = T(0.05)
    softening = T(0.5)
    candidate_locations = similar(population.locations)
    for slot in 1:2
        peer = 3 - slot
        denominator = hypot(
            softening,
            whitened[slot] - whitened[peer],
        )
        repulsion = repulsion_strength *
                    (population.locations[slot] - population.locations[peer]) /
                    denominator
        candidate_locations[slot] = gradient_locations[slot] + repulsion
    end
    candidate = (
        locations=candidate_locations,
        factors=candidate_factors,
        lognormalizers=-T(0.5) * log(T(2pi)) .- log.(candidate_factors),
    )
    return (; candidate, samples, denominators, returned_logweights)
end

function test_causal_population(actual, expected, tolerance)
    for slot in eachindex(actual.proposals)
        proposal = actual.proposals[slot]
        @test only(proposal.location) ≈ expected.locations[slot] rtol = tolerance
        @test only(proposal.scale.factor) ≈ expected.factors[slot] rtol = tolerance
        @test proposal.lognormalizer ≈ expected.lognormalizers[slot] rtol = tolerance
    end
end

function validate_first_order_gramis_causal_rounds()
    T = Float64
    target_value = CausalTarget()
    target = LogTarget(target_value; grad=causal_gradient!)
    bank = causal_bank(T)
    first_normals = T[-0.5, 0, 0.5, -0.5, 0, 0.5]
    second_normals = T[-0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75]
    q1 = (
        locations=T[-0.2, 0.2],
        factors=ones(T, 2),
        lognormalizers=fill(-T(0.5) * log(T(2pi)), 2),
    )
    round1 = literal_causal_transition(q1, first_normals, [1, 1, 1, 2, 2, 2])
    q2 = round1.candidate
    round2 = literal_causal_transition(
        q2,
        second_normals,
        [1, 1, 1, 1, 2, 2, 2, 2],
    )
    q3 = round2.candidate
    one_round = prepare_sampler(
        CausalPrefilledRNG([first_normals], 1),
        target,
        FirstOrderGRAMIS(
            bank;
            rounds=1,
            round_size=6,
            repulsion_strength=T(0.05),
            covariance_ess_threshold=2,
            covariance_rate=T(0.4),
            covariance_regularization=T(0.01),
            repulsion_softening=T(0.5),
            max_backtracking_trials=4,
        );
        factor_execution=BatchedFactorExecution(),
        threaded=false,
    )
    importance_sample!(one_round)
    observed_q2 = current_proposal(one_round)
    test_causal_population(observed_q2, q2, T(64) * eps(T))

    two_round = prepare_sampler(
        CausalPrefilledRNG([
            first_normals,
            second_normals,
        ], 1),
        target,
        FirstOrderGRAMIS(
            bank;
            rounds=2,
            round_size=[6, 8],
            repulsion_strength=T(0.05),
            covariance_ess_threshold=2,
            covariance_rate=T(0.4),
            covariance_regularization=T(0.01),
            repulsion_softening=T(0.5),
            max_backtracking_trials=4,
        );
        factor_execution=BatchedFactorExecution(),
        threaded=false,
    )
    result = importance_sample!(two_round)
    observed_q3 = current_proposal(two_round)
    test_causal_population(observed_q3, q3, T(128) * eps(T))
    expected_samples = vcat(round1.samples, round2.samples)
    expected_logweights = vcat(
        round1.returned_logweights,
        round2.returned_logweights,
    )
    @test vec(result.samples) ≈ expected_samples rtol = T(64) * eps(T)
    @test result.logweights ≈ expected_logweights rtol = T(64) * eps(T)
    q3_denominators = map(round2.samples) do sample
        causal_mixture_logdensity(q3, sample, [4, 4])
    end
    returned_denominators = round2.denominators
    @test any(!isapprox(left, right; rtol=64eps(T), atol=0) for
              (left, right) in zip(q3_denominators, returned_denominators))
    @test result.provenance.round == vcat(fill(1, 6), fill(2, 8))
    return (
        sampled_populations=(:q1, :q2),
        retained_population=:q3,
        denominator_populations=(:q1, :q2),
        status=:passed,
    )
end

function validate_first_order_gramis_estimator_identity()
    counts = [4, 3, 3]
    total = sum(counts)
    proposals = Rational{Int}[
        1//2 1//4 1//4
        1//4 1//2 1//4
        1//4 1//4 1//2
    ]
    pi_h = Rational{Int}[2//5, -1//7, 3//11]
    psi = [
        sum((counts[slot] // total) * proposals[slot, point] for slot in 1:3) for
        point in 1:3
    ]
    pointwise = [
        sum(
            (counts[slot] // total) * proposals[slot, point] *
            pi_h[point] / psi[point] for slot in 1:3
        ) for point in 1:3
    ]
    @test vec(sum(proposals; dims=2)) == fill(1//1, 3)
    @test pointwise == pi_h
    @test sum(pointwise) == 204//385

    round_sizes = [10, 13]
    round_sums = Rational{Int}[17//6, -5//4]
    combined = sum(
        (round_sizes[round] // sum(round_sizes)) *
        (round_sums[round] / round_sizes[round]) for round in 1:2
    )
    @test combined == sum(round_sums) / sum(round_sizes) == 19//276
    return (
        conditional_unbiasedness=:exact_finite_law,
        unequal_round_combination=:all_sample_average,
        status=:passed,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    display((
        repulsion=validate_first_order_gramis_repulsion(),
        covariance=validate_first_order_gramis_covariance_rules(),
        causal_rounds=validate_first_order_gramis_causal_rounds(),
        estimator_identity=validate_first_order_gramis_estimator_identity(),
    ))
end
