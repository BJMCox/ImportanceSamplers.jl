# Scalar and batch regression targets

Provisional shared-host measurements. CPU contention affects GPU setup too. Do not replace published CPU timings.

| Model | Method | Device | Mode | Seconds [range] | Weight ESS/s | Host MiB | Host allocations | Max mean error / SD | Max variance error | Accuracy |
|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|:--|
| linear | amis | CPU | batch | 1.29 [1.095, 1.432] | 1.053e+05 | 299 | 3345 | 0.00782 | 0.00835 | pass |
| linear | amis | CPU | scalar | 2.731 [2.415, 2.865] | 4.973e+04 | 231 | 2142 | 0.00782 | 0.00835 | pass |
| linear | lais | CPU | batch | 1.683 [1.572, 1.865] | 19.89 | 247 | 40370 | 0.522 | 0.444 | warn |
| linear | lais | CPU | scalar | 3.309 [3.099, 3.713] | 10.12 | 184 | 98758 | 0.522 | 0.444 | warn |
| logistic | amis | CPU | batch | 22.6 [22.36, 22.88] | 8552 | 169 | 3249 | 0.0056 | 0.00885 | pass |
| logistic | amis | CPU | scalar | 2.334 [2.259, 2.435] | 8.281e+04 | 101 | 2046 | 0.0056 | 0.00885 | pass |
| logistic | lais | CPU | batch | 23.36 [19.91, 25.23] | 618.1 | 154 | 38134 | 0.0134 | 0.0226 | pass |
| logistic | lais | CPU | scalar | 2.696 [2.129, 3.046] | 5355 | 90.9 | 98582 | 0.0134 | 0.0226 | pass |
| poisson | amis | CPU | batch | 2.334 [2.221, 2.807] | 8.188e+04 | 169 | 3350 | 0.00618 | 0.00749 | pass |
| poisson | amis | CPU | scalar | 0.7607 [0.6858, 0.8585] | 2.513e+05 | 101 | 2147 | 0.00618 | 0.00749 | pass |
| poisson | lais | CPU | batch | 2.636 [2.464, 3.25] | 5727 | 154 | 38235 | 0.0212 | 0.0229 | pass |
| poisson | lais | CPU | scalar | 0.9159 [0.8047, 1.048] | 1.648e+04 | 90.9 | 98683 | 0.0212 | 0.0229 | pass |
| robust | amis | CPU | batch | 8.238 [7.561, 9.335] | 2.274e+04 | 143 | 3358 | 0.00742 | 0.00862 | pass |
| robust | amis | CPU | scalar | 0.9798 [0.8591, 1.176] | 1.912e+05 | 107 | 2155 | 0.00742 | 0.00862 | pass |
| robust | lais | CPU | batch | 8.238 [6.474, 9.979] | 1568 | 127 | 38243 | 0.0149 | 0.0329 | pass |
| robust | lais | CPU | scalar | 1.14 [0.8873, 1.508] | 1.133e+04 | 95.5 | 98691 | 0.0149 | 0.0329 | pass |

Time includes fitting, pilot, scratch setup, transfers, sampling and posterior mean.
Each seed interleaves 4 executions per mode in balanced ABBA/BAAB blocks. Diagnostics follow both modes. Full GC precedes each execution outside timing.

| Model | Method | Device | Batch/scalar paired time ratio: median [range] |
|:--|:--|:--|--:|
| linear | amis | CPU | 0.479 [0.459, 0.491] |
| linear | lais | CPU | 0.507 [0.47, 0.526] |
| logistic | amis | CPU | 9.74 [9.33, 9.91] |
| logistic | lais | CPU | 8.67 [8.08, 9.64] |
| poisson | amis | CPU | 3.18 [2.75, 3.39] |
| poisson | lais | CPU | 2.9 [2.59, 3.25] |
| robust | amis | CPU | 8.77 [7.44, 9.23] |
| robust | lais | CPU | 7.23 [6.34, 8.3] |

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
