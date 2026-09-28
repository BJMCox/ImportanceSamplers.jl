Effective sample size per second (ESS/s)\*.

### CPU

| Sampler | Linear (32) |
|:--|--:|
| DM-PMC | 1e+02 |
| LAIS-RAM | **1.5e+04** |

### CUDA

| Sampler | Linear (32) |
|:--|--:|
| DM-PMC | 9.9e+02 |
| LAIS-RAM | 1.2e+05 |


\* ESS denotes weight ESS for importance sampling, minimum bulk ESS for independent-chain MCMC, and minimum mean ESS for EnsembleMCMC. Ensemble mean ESS is marginal variance divided by the squared MCSE of the sweep-mean process. These diagnostics do not define an equal-accuracy comparison.

Bold denotes the highest measured CPU rate per model. GPU results appear separately.

† At least one retained NUTS transition diverged. ‡ At least one run had maximum R-hat above 1.01. § Weight ESS varied by more than a factor of ten across seeds.

¶ At least one run exceeded a mean error of 0.2 posterior standard deviations or a marginal variance error of 30%.

Ensemble R-hat splits the sweep-mean time series, not the walkers. Settings below apply when recorded; archived ensemble runs used 4d walkers and rounded their pooled draw budget to complete sweeps.

Elapsed time includes initialization, warmup or adaptation, sampling, and posterior-mean estimation. Compilation and post-run diagnostics are excluded.

Times use the median of up to 3 executions per seed under a five-second BenchmarkTools budget. A complete execution may exceed that budget. Raw times and GC times are retained.

DM-PMC and LAIS use an independent 4096-draw width pilot. Its full cost is timed. Only proposal widths pass to the main run; its settings and retained sample count are unchanged.

| Model | Sampler / device | Mean pilot seconds | Width multiplier range |
|:--|:--|--:|--:|
| linear | DM-PMC / CPU | 1.51 | 0.795–0.795 |
| linear | DM-PMC / CUDA | 0.093 | 0.793–0.796 |
| linear | LAIS-RAM / CPU | 0.244 | 0.806–0.808 |
| linear | LAIS-RAM / CUDA | 0.0298 | 0.805–0.808 |

Julia 1.13.0. AMD EPYC 7702P 64-Core Processor. Julia threads: 16. BLAS threads: 1.
GPU: NVIDIA A100-PCIE-40GB.

3 independent seeds. IS samples: 262144. MCMC samples: 262144. MCMC chains: 16. MCMC warmup: 1024 per chain. Ensemble budgets and warmup are recorded per row.

Population settings supplied to this run. Per-run initialization and adaptation remain timed.

| Model | Sampler / device | Family | Initial scale / Laplace factor | Proposals | Rounds |
|:--|:--|:--|--:|--:|--:|
| linear | DM-PMC / CPU | student_t | 1.2 | 256 | 2 |
| linear | DM-PMC / CUDA | student_t | 1.2 | 256 | 2 |
| linear | LAIS-RAM / CPU | student_t | 1.2 | 16 | 4 |
| linear | LAIS-RAM / CUDA | student_t | 1.2 | 16 | 4 |

| Model | Sampler / device | Mean seconds | ESS/s range across seeds | Max R-hat | Max pooled-mean error / posterior SD | Max per-run variance error |
|:--|:--|--:|--:|--:|--:|--:|
| linear | DM-PMC / CPU | 8.67 | 49–1.3e+02 | — | 0.0486 | 0.125 |
| linear | DM-PMC / CUDA | 0.774 | 7.4e+02–1.2e+03 | — | 0.0389 | 0.156 |
| linear | LAIS-RAM / CPU | 3.57 | 1.1e+04–2e+04 | — | 0.00432 | 0.0187 |
| linear | LAIS-RAM / CUDA | 0.365 | 7.1e+04–2.2e+05 | — | 0.00724 | 0.0238 |

| Model | Sampler / device | Timed executions | Raw seconds range |
|:--|:--|--:|--:|
| linear | DM-PMC / CPU | 3 | 8.56–8.76 |
| linear | DM-PMC / CUDA | 9 | 0.626–0.922 |
| linear | LAIS-RAM / CPU | 6 | 3.49–3.67 |
| linear | LAIS-RAM / CUDA | 9 | 0.302–0.503 |

| Package | Version | Source revision |
|:--|:--|:--|
| AbstractMCMC | 5.16.0 | — |
| AdvancedHMC | 0.8.7 | — |
| AdvancedMH | 0.8.10 | — |
| BenchmarkTools | 1.8.0 | — |
| CUDA | 6.4.0 | — |
| Distributions | 0.25.131 | — |
| EnsembleMCMC | 0.0.2 | 5fcb74c1a7c19bedf95795644fb6296901d25d0e |
| FFTW | 1.10.0 | — |
| ForwardDiff | 1.4.6 | — |
| ImportanceSamplers | 0.0.1 | — |
| LinearAlgebra | 1.13.0 | — |
| LogDensityProblems | 2.2.0 | — |
| MCMCDiagnosticTools | 0.3.19 | — |
| MLDataDevices | 1.17.10 | — |
| Optim | 2.3.2 | — |
| Pkg | 1.13.0 | — |
| Printf | 1.11.0 | — |
| Random | 1.11.0 | — |
| Random123 | 1.7.1 | — |
| SHA | 1.0.0 | — |
| SliceSampling | 0.7.12 | — |
| Statistics | 1.11.5 | — |
| TOML | 1.0.3 | — |

Source: `9771b339aeb9d72e203fcd5ea7f2f57a2e7294f6`. Manifest SHA-256: `19b44b29f3a31074ce0faa2bdc57d8948e9739c9f33637bcfe0fa50c88eb8bfa`.
