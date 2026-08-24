using ImportanceSamplers
using Random

const IS = ImportanceSamplers

function _logfactorial(::Type{T}, value::Int) where {T}
    total = zero(T)
    for factor in 2:value
        total += log(T(factor))
    end
    return total
end

"""
    normalized_dirichlet_evidence(; samples=1_000_000, concentration=2,
                                  proposal_rate=1.5, seed=0x6136)

Estimate the evidence of a normalized, symmetric three-component Dirichlet
density after `SimplexTransform(3)`. The independently coded proposal has two
independent Laplace coordinates with density
`proposal_rate * exp(-proposal_rate * abs(z)) / 2`. The Dirichlet normalizer is
computed from integer factorials, so this reproducer needs no distribution or
special-function package.

For `y = Bz`, `sum(y) = 0`, and the transformed symmetric Dirichlet density is
proportional to `exp(-3 * concentration * logsumexp(y))`. Orthonormality gives
`max(y) >= norm(z) / sqrt(6)`, so its square decays at least as
`exp(-2 * concentration * sqrt(3 / 2) * norm(z))`. The reciprocal product
Laplace density grows no faster than
`exp(proposal_rate * sqrt(2) * norm(z))`. Thus the importance weights have a
finite second moment when `proposal_rate < concentration * sqrt(3)`, as
required below.
"""
function normalized_dirichlet_evidence(;
    samples=1_000_000,
    concentration=2,
    proposal_rate=1.5,
    seed=0x6136,
)
    T = Float64
    dimension = 3
    coordinate_dimension = dimension - 1
    samples > 1 || throw(ArgumentError("samples must be greater than one"))
    concentration >= 1 || throw(ArgumentError("concentration must be positive"))
    zero(T) < proposal_rate < T(concentration) * sqrt(T(3)) || throw(
        ArgumentError("proposal rate does not guarantee a finite second moment"),
    )
    transform = SimplexTransform(dimension)
    rng = Xoshiro(seed)

    log_dirichlet_normalizer =
        _logfactorial(T, dimension * concentration - 1) -
        T(dimension) * _logfactorial(T, concentration - 1)
    laplace_log_normalizer =
        T(coordinate_dimension) * (log(T(proposal_rate)) - log(T(2)))

    evidence = zero(T)
    squared_difference_sum = zero(T)
    coordinates = Vector{T}(undef, coordinate_dimension)
    for sample_index in 1:samples
        absolute_coordinate_sum = zero(T)
        for index in eachindex(coordinates)
            coordinate =
                (randexp(rng, T) - randexp(rng, T)) / T(proposal_rate)
            coordinates[index] = coordinate
            absolute_coordinate_sum += abs(coordinate)
        end
        log_proposal =
            laplace_log_normalizer - T(proposal_rate) * absolute_coordinate_sum
        simplex, logabsjac = IS._transform_with_logjac(transform, coordinates)
        log_target =
            log_dirichlet_normalizer + T(concentration - 1) * sum(log, simplex)
        weight = exp(log_target + logabsjac - log_proposal)
        difference = weight - evidence
        evidence += difference / T(sample_index)
        squared_difference_sum += difference * (weight - evidence)
    end

    sample_variance = squared_difference_sum / T(samples - 1)
    standard_error = sqrt(sample_variance / T(samples))
    standard_score = abs(evidence - one(T)) / standard_error
    return (;
        evidence,
        standard_error,
        standard_score,
        samples,
        concentration,
        proposal_rate,
        seed,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = normalized_dirichlet_evidence()
    display(result)
    @assert isfinite(result.standard_error) && result.standard_error > 0
    @assert result.standard_score <= 6
end
