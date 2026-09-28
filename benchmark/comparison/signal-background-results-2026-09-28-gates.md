Effective sample size per second (ESS/s)\*.

### CPU

| Sampler | Signal-background (9) |
|:--|--:|
| IS | 5.4e+04 |
| AMIS | **1.5e+05** |
| DM-PMC | 4.1e+04 |
| CAIS | 1.3e+04§¶ |
| LAIS-RAM | 4.4e+04 |
| First-order GRAMIS-CAIS | 7.1e+04 |

### CUDA

| Sampler | Signal-background (9) |
|:--|--:|
| IS | 7.3e+05 |
| AMIS | 6e+05 |
| DM-PMC | 1.4e+05 |
| CAIS | 8.7e+05 |
| LAIS-RAM | 5.7e+04 |
| First-order GRAMIS-CAIS | 3e+05 |


\* ESS denotes weight ESS for importance sampling, minimum bulk ESS for independent-chain MCMC, and minimum mean ESS for EnsembleMCMC. Ensemble mean ESS is marginal variance divided by the squared MCSE of the sweep-mean process. These diagnostics do not define an equal-accuracy comparison.

Bold denotes the highest measured CPU rate per model. GPU results appear separately.

† At least one retained NUTS transition diverged. ‡ At least one run had maximum R-hat above 1.01. § Weight ESS varied by more than a factor of ten across seeds.

¶ At least one run exceeded a mean error of 0.2 posterior standard deviations, or a marginal variance error of both 30% and three standard errors. Importance rows use a delta-method variance standard error, which is low when one weight dominates. Rows without it keep the 30% rule. This is a per-run check, not a tail-accuracy guarantee.

Ensemble R-hat splits the sweep-mean time series, not the walkers. Settings below apply when recorded; archived ensemble runs used 4d walkers and rounded their pooled draw budget to complete sweeps.

Elapsed time includes initialization, warmup or adaptation, sampling, and posterior-mean estimation. Compilation and post-run diagnostics are excluded.

Times use the median of up to 3 executions per seed under a five-second BenchmarkTools budget. A complete execution may exceed that budget. Raw times and GC times are retained.

DM-PMC and LAIS use an independent 4096-draw width pilot. Its full cost is timed. Only proposal widths pass to the main run; its settings and retained sample count are unchanged.

| Model | Sampler / device | Mean pilot seconds | Width multiplier range |
|:--|:--|--:|--:|
| signal-background | DM-PMC / CPU | 0.0424 | 1.25–1.31 |
| signal-background | DM-PMC / CUDA | 0.00544 | 1.26–1.67 |
| signal-background | LAIS-RAM / CPU | 0.0499 | 1.25–1.31 |
| signal-background | LAIS-RAM / CUDA | 0.00523 | 1.26–1.67 |

Julia 1.13.0. AMD EPYC 7702P 64-Core Processor. Julia threads: 16. BLAS threads: 1.
GPU: NVIDIA A100-PCIE-40GB.

3 independent seeds. IS samples: 262144. MCMC samples: 262144. MCMC chains: 16. MCMC warmup: 1024 per chain. Ensemble budgets and warmup are recorded per row.

Population settings supplied to this run. Per-run initialization and adaptation remain timed.

| Model | Sampler / device | Family | Initial scale / Laplace factor | Proposals | Rounds |
|:--|:--|:--|--:|--:|--:|

| Model | Sampler / device | Mean seconds | ESS/s range across seeds | Max R-hat | Max pooled-mean error / posterior SD | Max per-run variance error | Max variance z |
|:--|:--|--:|--:|--:|--:|--:|--:|
| signal-background | IS / CPU | 0.204 | 4.5e+04–6.5e+04 | — | 0.0232 | 0.143 | 7.14 |
| signal-background | IS / CUDA | 0.0155 | 4.1e+05–3.2e+06 | — | 0.0132 | 0.198 | 9.93 |
| signal-background | AMIS / CPU | 0.275 | 1.3e+05–1.9e+05 | — | 0.0129 | 0.177 | 9.98 |
| signal-background | AMIS / CUDA | 0.107 | 5e+05–9.6e+05 | — | 0.0184 | 0.207 | 17.8 |
| signal-background | DM-PMC / CPU | 0.295 | 3e+04–6.6e+04 | — | 0.0212 | 0.169 | 3.15 |
| signal-background | DM-PMC / CUDA | 0.0902 | 8e+04–3.7e+05 | — | 0.00763 | 0.0627 | 2.34 |
| signal-background | CAIS / CPU | 0.206 | 26–3.9e+04 | — | 1.15 | 11.4 | 20 |
| signal-background | CAIS / CUDA | 0.0232 | 6e+05–1.2e+06 | — | 0.0281 | 0.215 | 7.64 |
| signal-background | LAIS-RAM / CPU | 0.374 | 2.4e+04–6.6e+04 | — | 0.0121 | 0.124 | 4.74 |
| signal-background | LAIS-RAM / CUDA | 0.126 | 3.3e+04–1e+05 | — | 0.0312 | 0.44 | 2.6 |
| signal-background | First-order GRAMIS-CAIS / CPU | 0.266 | 5.1e+04–8.4e+04 | — | 0.0236 | 0.178 | 7.58 |
| signal-background | First-order GRAMIS-CAIS / CUDA | 0.0515 | 2.1e+05–4.6e+05 | — | 0.0181 | 0.181 | 7.21 |

| Model | Sampler / device | Timed executions | Raw seconds range |
|:--|:--|--:|--:|
| signal-background | IS / CPU | 9 | 0.153–0.317 |
| signal-background | IS / CUDA | 9 | 0.00341–0.21 |
| signal-background | AMIS / CPU | 9 | 0.223–0.383 |
| signal-background | AMIS / CUDA | 9 | 0.0291–0.189 |
| signal-background | DM-PMC / CPU | 9 | 0.267–0.412 |
| signal-background | DM-PMC / CUDA | 9 | 0.0319–2.19 |
| signal-background | CAIS / CPU | 9 | 0.173–0.36 |
| signal-background | CAIS / CUDA | 9 | 0.0149–0.0571 |
| signal-background | LAIS-RAM / CPU | 9 | 0.338–0.556 |
| signal-background | LAIS-RAM / CUDA | 9 | 0.102–0.246 |
| signal-background | First-order GRAMIS-CAIS / CPU | 9 | 0.243–0.397 |
| signal-background | First-order GRAMIS-CAIS / CUDA | 9 | 0.0369–0.215 |

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

Source: `990b56ee40a8832e451dadbfd9d856da3979890c`. Manifest SHA-256: `19b44b29f3a31074ce0faa2bdc57d8948e9739c9f33637bcfe0fa50c88eb8bfa`.
