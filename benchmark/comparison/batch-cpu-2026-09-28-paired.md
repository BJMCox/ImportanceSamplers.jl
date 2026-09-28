# Scalar and batch regression targets

Provisional shared-host measurements. CPU contention affects GPU setup too. Do not replace published CPU timings.

| Model | Method | Device | Mode | Seconds [range] | Weight ESS/s | Host MiB | Host allocations | Max mean error / SD | Max variance error | Accuracy |
|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|:--|
| linear | amis | CPU | batch | 0.7667 [0.6996, 0.7982] | 1.772e+05 | 295 | 5558 | 0.00782 | 0.00835 | pass |
| linear | amis | CPU | scalar | 2.9 [2.743, 2.982] | 4.684e+04 | 231 | 2142 | 0.00782 | 0.00835 | pass |
| linear | lais | CPU | batch | 1.144 [1.106, 1.172] | 29.24 | 251 | 113651 | 0.522 | 0.444 | warn |
| linear | lais | CPU | scalar | 3.395 [2.915, 3.488] | 9.858 | 184 | 98758 | 0.522 | 0.444 | warn |
| logistic | amis | CPU | batch | 1.181 [1.135, 1.225] | 1.637e+05 | 165 | 5462 | 0.0056 | 0.00885 | pass |
| logistic | amis | CPU | scalar | 2.403 [2.31, 2.44] | 8.041e+04 | 101 | 2046 | 0.0056 | 0.00885 | pass |
| logistic | lais | CPU | batch | 1.483 [1.441, 1.527] | 9733 | 158 | 113475 | 0.0134 | 0.0226 | pass |
| logistic | lais | CPU | scalar | 2.815 [2.784, 2.92] | 5128 | 90.9 | 98582 | 0.0134 | 0.0226 | pass |
| poisson | amis | CPU | batch | 0.6825 [0.662, 0.7063] | 2.8e+05 | 165 | 5563 | 0.00618 | 0.00749 | pass |
| poisson | amis | CPU | scalar | 0.9566 [0.8911, 1] | 1.998e+05 | 101 | 2147 | 0.00618 | 0.00749 | pass |
| poisson | lais | CPU | batch | 0.949 [0.8878, 1.003] | 1.591e+04 | 158 | 113576 | 0.0212 | 0.0229 | pass |
| poisson | lais | CPU | scalar | 1.114 [1.059, 1.161] | 1.356e+04 | 90.9 | 98683 | 0.0212 | 0.0229 | pass |
| robust | amis | CPU | batch | 0.6954 [0.6579, 0.7473] | 2.694e+05 | 140 | 5571 | 0.00742 | 0.00862 | pass |
| robust | amis | CPU | scalar | 0.9415 [0.9002, 0.9986] | 1.99e+05 | 107 | 2155 | 0.00742 | 0.00862 | pass |
| robust | lais | CPU | batch | 0.9133 [0.8829, 0.9459] | 1.414e+04 | 131 | 113584 | 0.0149 | 0.0329 | pass |
| robust | lais | CPU | scalar | 1.126 [1.107, 1.14] | 1.148e+04 | 95.5 | 98691 | 0.0149 | 0.0329 | pass |

Time includes fitting, pilot, scratch setup, transfers, sampling and posterior mean.
Each seed interleaves 4 executions per mode in balanced ABBA/BAAB blocks. Diagnostics follow both modes. Full GC precedes each execution outside timing.

| Model | Method | Device | Batch/scalar paired time ratio: median [range] |
|:--|:--|:--|--:|
| linear | amis | CPU | 0.264 [0.252, 0.276] |
| linear | lais | CPU | 0.34 [0.331, 0.359] |
| logistic | amis | CPU | 0.493 [0.48, 0.499] |
| logistic | lais | CPU | 0.527 [0.508, 0.533] |
| poisson | amis | CPU | 0.715 [0.698, 0.749] |
| poisson | lais | CPU | 0.86 [0.811, 0.881] |
| robust | amis | CPU | 0.735 [0.702, 0.77] |
| robust | lais | CPU | 0.809 [0.8, 0.835] |

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
