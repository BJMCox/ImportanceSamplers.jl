# Public API

## Algorithms and execution

```@docs
AbstractImportanceSampler
ImportanceSampling
AMIS
DeterministicMixturePMC
FusedFactorExecution
BatchedFactorExecution
prepare_sampler
retarget
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
UnweightedSamples
normalized_weights
lognormalizer
AbstractResamplingMethod
MultinomialResampling
resample
Statistics.mean
Statistics.var
Statistics.std
Statistics.cov
Statistics.quantile
Statistics.median
```

## Errors

```@docs
AllZeroWeightsError
InvalidTransformError
SamplerBusyError
SamplerAlreadyExecutedError
SamplerDeviceError
SamplerExecutionError
AMISRoundError
DMPMCRoundError
```
