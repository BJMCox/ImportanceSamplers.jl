# Scalar and batch regression targets

Provisional shared-host measurements. CPU contention affects GPU setup too. Do not replace published CPU timings.

| Model | Method | Device | Mode | Seconds [range] | Weight ESS/s | Host MiB | Host allocations | Max mean error / SD | Max variance error | Accuracy |
|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|:--|
| linear | lais | CPU | batch | 0.9326 [0.8248, 1.109] | 4.269e+04 | 242 | 10835 | 0.0132 | 0.0197 | pass |
| linear | lais | CPU | scalar | 3.309 [3.151, 3.485] | 1.203e+04 | 175 | 7030 | 0.0132 | 0.0197 | pass |
| linear | lais | CUDA | batch | 0.1341 [0.1276, 0.1416] | 3.17e+05 | 101 | 38129 | 0.0133 | 0.016 | pass |
| linear | lais | CUDA | scalar | 0.3123 [0.307, 0.3223] | 1.361e+05 | 101 | 23094 | 0.0133 | 0.016 | pass |

Time includes fitting, pilot, scratch setup, transfers, sampling and posterior mean.
Each seed interleaves 4 executions per mode in balanced ABBA/BAAB blocks. Diagnostics follow both modes. Full GC precedes each execution outside timing.

| Model | Method | Device | Batch/scalar paired time ratio: median [range] |
|:--|:--|:--|--:|
| linear | lais | CPU | 0.285 [0.257, 0.314] |
| linear | lais | CUDA | 0.428 [0.423, 0.441] |

A ratio above one means batching took longer. Interleaving reduces drift but cannot eliminate shared-host contention.
Host allocations exclude device allocations. CUDA pools are warm. Raw times, GC times, load and BLAS threads remain in TOML.
Accuracy warns at >0.2 posterior-SD mean error or >30% marginal variance error.
Julia 1.13.0, threads 16; AMD EPYC 7702P 64-Core Processor. Samples: 262144; seeds: 9301, 9302, 9303.

Manifest SHA-256: `19b44b29f3a31074ce0faa2bdc57d8948e9739c9f33637bcfe0fa50c88eb8bfa`.

| Package | Version |
|:--|:--|
| AbstractMCMC | 5.16.0 |
| AdvancedHMC | 0.8.7 |
| AdvancedMH | 0.8.10 |
| BenchmarkTools | 1.8.0 |
| CUDA | 6.4.0 |
| Distributions | 0.25.131 |
| EnsembleMCMC | 0.0.2 |
| FFTW | 1.10.0 |
| ForwardDiff | 1.4.6 |
| ImportanceSamplers | 0.0.1 |
| LinearAlgebra | 1.13.0 |
| LogDensityProblems | 2.2.0 |
| MCMCDiagnosticTools | 0.3.19 |
| MLDataDevices | 1.17.10 |
| Optim | 2.3.2 |
| Pkg | 1.13.0 |
| Printf | 1.11.0 |
| Random | 1.11.0 |
| Random123 | 1.7.1 |
| SHA | 1.0.0 |
| SliceSampling | 0.7.12 |
| Statistics | 1.11.5 |
| TOML | 1.0.3 |
