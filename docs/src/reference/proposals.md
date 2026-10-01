# [Proposals and transforms](@id proposal-reference)

[Proposals](@ref proposals-guide) explains scale conventions.
[Constraints and named parameters](@ref transforms-guide) explains coordinate and Jacobian ownership.

## Native families

```@docs
AbstractProposalFamily
AbstractRadialProposalFamily
SphericalGaussian
DiagonalGaussian
FactorGaussian
SphericalStudentT
DiagonalStudentT
FactorStudentT
```

## Composition and populations

```@docs
AbstractProposalPopulation
ProposalBank
ProductProposal
TransformedProposal
```

## Coordinate transforms

```@docs
AbstractSampleTransform
IdentityTransform
PositiveTransform
SoftplusTransform
IntervalTransform
SimplexTransform
```
