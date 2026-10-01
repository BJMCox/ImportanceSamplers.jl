# [Samplers and transitions](@id sampler-reference)

See [Choosing a method](@ref choosing-method) for the method comparison and allocation requirements.

## Algorithm configurations

```@docs
AbstractImportanceSampler
ImportanceSampling
AMIS
NPMC
DeterministicMixturePMC
APIS
CAIS
LAIS
FirstOrderGRAMIS
```

## Static MIS schemes

```@docs
AbstractMISScheme
StratifiedMixture
RandomMixture
StandardMIS
PartialDeterministicMixture
```

## Population resampling

```@docs
AbstractPMCResamplingPolicy
GlobalResampling
LocalResampling
```

## LAIS transitions and tuning

```@docs
AbstractMCMCTransition
RandomWalkMetropolis
RAM
WarmupTuning
ContinuousTuning
SampleMetropolisHastings
```

## Factor execution

```@docs
FusedFactorExecution
BatchedFactorExecution
```
