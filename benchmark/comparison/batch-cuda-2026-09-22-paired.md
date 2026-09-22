# Scalar and batch regression targets

Provisional shared-host measurements. CPU contention affects GPU setup too. Do not replace published CPU timings.

| Model | Method | Device | Mode | Seconds [range] | Weight ESS/s | Host MiB | Host allocations | Max mean error / SD | Max variance error | Accuracy |
|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|:--|
| linear | amis | CUDA | batch | 0.2742 [0.2501, 0.322] | 4.958e+05 | 426 | 12957 | 0.00698 | 0.0106 | pass |
| linear | amis | CUDA | scalar | 0.2967 [0.2763, 0.3649] | 4.582e+05 | 297 | 6659 | 0.00698 | 0.0106 | pass |
| linear | lais | CUDA | batch | 0.8643 [0.807, 0.9324] | 55.54 | 315 | 429282 | 0.367 | 0.645 | warn |
| linear | lais | CUDA | scalar | 4.113 [4.099, 4.129] | 11.67 | 101 | 23598 | 0.367 | 0.645 | warn |
| logistic | amis | CUDA | batch | 0.2147 [0.201, 0.2286] | 9.003e+05 | 255 | 12800 | 0.00653 | 0.00699 | pass |
| logistic | amis | CUDA | scalar | 0.1267 [0.1208, 0.1357] | 1.526e+06 | 127 | 6505 | 0.00653 | 0.00699 | pass |
| logistic | lais | CUDA | batch | 0.5998 [0.5674, 0.6392] | 1.489e+04 | 262 | 428993 | 0.0221 | 0.0346 | pass |
| logistic | lais | CUDA | scalar | 2.196 [2.182, 2.245] | 4066 | 47.6 | 23270 | 0.0221 | 0.0346 | pass |
| poisson | amis | CUDA | batch | 0.1875 [0.1716, 0.1987] | 1.02e+06 | 255 | 12901 | 0.0072 | 0.00777 | pass |
| poisson | amis | CUDA | scalar | 0.102 [0.09311, 0.1178] | 1.874e+06 | 127 | 6604 | 0.0072 | 0.00777 | pass |
| poisson | lais | CUDA | batch | 0.6028 [0.5873, 0.6659] | 1.971e+04 | 262 | 429358 | 0.03 | 0.0261 | pass |
| poisson | lais | CUDA | scalar | 1.161 [1.15, 1.231] | 1.024e+04 | 47.6 | 23631 | 0.03 | 0.0261 | pass |
| robust | amis | CUDA | batch | 0.06004 [0.05554, 0.07208] | 3.117e+06 | 200 | 14061 | 0.00604 | 0.0113 | pass |
| robust | amis | CUDA | scalar | 0.04631 [0.04185, 0.07474] | 4.042e+06 | 135 | 6614 | 0.00604 | 0.0113 | pass |
| robust | lais | CUDA | batch | 0.3937 [0.3739, 0.4128] | 1.675e+04 | 171 | 409711 | 0.0244 | 0.0419 | pass |
| robust | lais | CUDA | scalar | 0.9644 [0.9492, 1.003] | 6838 | 50.1 | 23379 | 0.0244 | 0.0419 | pass |

Time includes fitting, pilot, scratch setup, transfers, sampling and posterior mean.
Each seed interleaves 4 executions per mode in balanced ABBA/BAAB blocks. Diagnostics follow both modes. Full GC precedes each execution outside timing.

| Model | Method | Device | Batch/scalar paired time ratio: median [range] |
|:--|:--|:--|--:|
| linear | amis | CUDA | 0.915 [0.889, 0.963] |
| linear | lais | CUDA | 0.211 [0.201, 0.224] |
| logistic | amis | CUDA | 1.69 [1.61, 1.76] |
| logistic | lais | CUDA | 0.274 [0.257, 0.287] |
| poisson | amis | CUDA | 1.83 [1.65, 1.93] |
| poisson | lais | CUDA | 0.515 [0.503, 0.546] |
| robust | amis | CUDA | 1.28 [1.06, 1.45] |
| robust | lais | CUDA | 0.404 [0.395, 0.424] |

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
