# Scalar and batch regression targets

Provisional shared-host measurements. CPU contention affects GPU setup too. Do not replace published CPU timings.

| Model | Method | Device | Mode | Seconds [range] | Weight ESS/s | Host MiB | Host allocations | Max mean error / SD | Max variance error | Accuracy |
|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|:--|
| linear | amis | CUDA | batch | 0.2375 [0.2228, 0.2587] | 5.722e+05 | 298 | 13021 | 0.00698 | 0.0106 | pass |
| linear | amis | CUDA | scalar | 0.3493 [0.3224, 0.4611] | 3.891e+05 | 297 | 6660 | 0.00698 | 0.0106 | pass |
| linear | lais | CUDA | batch | 0.7769 [0.7008, 0.8245] | 61.79 | 123 | 438617 | 0.367 | 0.645 | warn |
| linear | lais | CUDA | scalar | 4.113 [4.084, 4.134] | 11.67 | 101 | 23598 | 0.367 | 0.645 | warn |
| logistic | amis | CUDA | batch | 0.06224 [0.05491, 0.06919] | 3.106e+06 | 127 | 12863 | 0.00653 | 0.00699 | pass |
| logistic | amis | CUDA | scalar | 0.07481 [0.06377, 0.08281] | 2.584e+06 | 127 | 6505 | 0.00653 | 0.00699 | pass |
| logistic | lais | CUDA | batch | 0.462 [0.4188, 0.5123] | 1.933e+04 | 70.3 | 438328 | 0.0221 | 0.0346 | pass |
| logistic | lais | CUDA | scalar | 2.184 [2.177, 2.214] | 4089 | 47.6 | 23270 | 0.0221 | 0.0346 | pass |
| poisson | amis | CUDA | batch | 0.09684 [0.08232, 0.1046] | 1.974e+06 | 127 | 12964 | 0.0072 | 0.00777 | pass |
| poisson | amis | CUDA | scalar | 0.09918 [0.08842, 0.1117] | 1.928e+06 | 127 | 6604 | 0.0072 | 0.00777 | pass |
| poisson | lais | CUDA | batch | 0.4653 [0.4075, 0.4966] | 2.554e+04 | 70.3 | 438694 | 0.03 | 0.0261 | pass |
| poisson | lais | CUDA | scalar | 1.157 [1.149, 1.175] | 1.028e+04 | 47.6 | 23631 | 0.03 | 0.0261 | pass |
| robust | amis | CUDA | batch | 0.08767 [0.08262, 0.1074] | 2.135e+06 | 136 | 14188 | 0.00604 | 0.0113 | pass |
| robust | amis | CUDA | scalar | 0.08912 [0.08044, 0.09493] | 2.1e+06 | 135 | 6614 | 0.00604 | 0.0113 | pass |
| robust | lais | CUDA | batch | 0.3817 [0.3484, 0.4207] | 1.728e+04 | 75.4 | 421169 | 0.0244 | 0.0419 | pass |
| robust | lais | CUDA | scalar | 0.9555 [0.9519, 1.007] | 6902 | 50.1 | 23379 | 0.0244 | 0.0419 | pass |

Time includes fitting, pilot, scratch setup, transfers, sampling and posterior mean.
Each seed interleaves 4 executions per mode in balanced ABBA/BAAB blocks. Diagnostics follow both modes. Full GC precedes each execution outside timing.

| Model | Method | Device | Batch/scalar paired time ratio: median [range] |
|:--|:--|:--|--:|
| linear | amis | CUDA | 0.675 [0.615, 0.725] |
| linear | lais | CUDA | 0.19 [0.18, 0.197] |
| logistic | amis | CUDA | 0.853 [0.782, 0.878] |
| logistic | lais | CUDA | 0.214 [0.197, 0.224] |
| poisson | amis | CUDA | 0.956 [0.927, 1.04] |
| poisson | lais | CUDA | 0.396 [0.389, 0.415] |
| robust | amis | CUDA | 1.04 [0.929, 1.1] |
| robust | lais | CUDA | 0.397 [0.373, 0.433] |

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
