# Public API

## Algorithms and execution

```@docs
AbstractImportanceSampler
ImportanceSampling
DeterministicMixturePMC
prepare_sampler
importance_sample
importance_sample!
current_proposal
```

## Proposal populations and MIS schemes

```@docs
AbstractProposalPopulation
ProposalBank
AbstractMISScheme
StratifiedMixture
RandomMixture
StandardMIS
PartialDeterministicMixture
```

## Native proposals and transforms

```@docs
AbstractProposalFamily
AbstractRadialProposalFamily
SphericalGaussian
DiagonalGaussian
FactorGaussian
ProductProposal
TransformedProposal
AbstractSampleTransform
IdentityTransform
PositiveTransform
SoftplusTransform
IntervalTransform
SimplexTransform
LogTarget
```

## Results

```@docs
WeightedSamples
WeightedSampleView
normalized_weights
lognormalizer
```

## Errors

```@docs
AllZeroWeightsError
InvalidTransformError
SamplerBusyError
SamplerAlreadyExecutedError
SamplerDeviceError
SamplerExecutionError
DMPMCRoundError
```
