# [Errors](@id error-reference)

[Troubleshooting](@ref troubleshooting-guide) explains common causes and remedies.
Adaptive round errors preserve the pre-call learned proposal but do not restore the RNG.

## Targets, weights, and transforms

```@docs
AllZeroWeightsError
InvalidTransformError
SamplerExecutionError
```

## Sampler lifecycle and devices

```@docs
SamplerBusyError
SamplerAlreadyExecutedError
SamplerDeviceError
```

## Adaptive rounds

```@docs
AMISRoundError
NPMCRoundError
DMPMCRoundError
APISRoundError
CAISRoundError
LAISRoundError
FirstOrderGRAMISRoundError
```
