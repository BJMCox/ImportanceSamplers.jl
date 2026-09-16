Effective sample size per second (ESS/s)\*.

### CUDA

| Sampler | Linear (32) | Logistic (12) | Poisson (12) | Robust (13) | Eight schools (10) |
|:--|--:|--:|--:|--:|--:|
| IS | 6.3e+05 | 1.8e+06 | 1.9e+06 | 4.1e+06 | 4.2e+04§¶ |
| AMIS | 3.1e+05 | 1e+06 | 1.4e+06 | 1.1e+06 | 1.1e+06 |
| DM-PMC | 2e+02 | 2.8e+05 | 3.2e+05 | 2.1e+05 | 2.2e+05 |
| CAIS | 2e+05 | 2.5e+06 | 3.2e+06 | 2.7e+06 | 1e+05 |
| LAIS-RAM | 1.3e+02 | 2.4e+04 | 3.8e+04 | 3.9e+04 | 4.1e+05 |


\* ESS denotes weight ESS for importance sampling and minimum bulk ESS across parameters for MCMC. These diagnostics do not define an equal-accuracy comparison.

Bold denotes the highest measured CPU rate per model. GPU results appear separately.

† At least one retained NUTS transition diverged. ‡ At least one run had maximum R-hat above 1.01. § Weight ESS varied by more than a factor of ten across seeds.

¶ At least one run exceeded a mean error of 0.2 posterior standard deviations or a marginal variance error of 30%.

Host conditions: GPU idle before measurement. CPU host shared with other users.  15:44:29 up 69 days,  3:52, 13 users,  load average: 56.59, 57.47, 57.38;  15:47:56 up 69 days,  3:56, 13 users,  load average: 53.41, 55.08, 56.40.

Elapsed time includes initialization, warmup or adaptation, sampling, and posterior-mean estimation. Compilation and post-run diagnostics are excluded.

Julia 1.13.0. AMD EPYC 7702P 64-Core Processor. Julia threads: 16. BLAS threads: 1.
GPU: NVIDIA A100-PCIE-40GB.

3 independent seeds. IS samples: 262144. MCMC samples: 262144. MCMC chains: 16. MCMC warmup: 1024 per chain. Ensemble warmup: 1024 sweeps.

Population settings were selected on separate pilot seeds. Per-run initialization and adaptation remain timed.

| Model | Sampler / device | Scale / Laplace factor | Proposals | Rounds |
|:--|:--|--:|--:|--:|
| linear | DM-PMC / CUDA | 1.2 | 256 | 4 |
| linear | LAIS-RAM / CUDA | 0.8 | 1024 | 4 |
| logistic | DM-PMC / CUDA | 0.8 | 64 | 4 |
| logistic | LAIS-RAM / CUDA | 0.8 | 1024 | 4 |
| poisson | DM-PMC / CUDA | 0.8 | 256 | 4 |
| poisson | LAIS-RAM / CUDA | 0.8 | 256 | 4 |
| robust | DM-PMC / CUDA | 0.8 | 64 | 4 |
| robust | LAIS-RAM / CUDA | 0.8 | 1024 | 4 |
| eight_schools | DM-PMC / CUDA | 1.2 | 64 | 16 |
| eight_schools | LAIS-RAM / CUDA | 0.8 | 256 | 4 |

| Model | Sampler / device | Mean seconds | ESS/s range across seeds | Max R-hat | Max pooled-mean error / posterior SD | Max per-run variance error |
|:--|:--|--:|--:|--:|--:|--:|
| linear | IS / CUDA | 0.181 | 4.5e+05–1.1e+06 | — | 0.00311 | 0.0123 |
| linear | AMIS / CUDA | 0.445 | 2.9e+05–3.2e+05 | — | 0.00521 | 0.00919 |
| linear | DM-PMC / CUDA | 0.958 | 1.1e+02–4.7e+02 | — | 0.0971 | 0.234 |
| linear | CAIS / CUDA | 0.246 | 1.8e+05–2.1e+05 | — | 0.00612 | 0.0178 |
| linear | LAIS-RAM / CUDA | 5.48 | 76–1.7e+02 | — | 0.03 | 0.102 |
| logistic | IS / CUDA | 0.0888 | 1e+06–3e+06 | — | 0.00276 | 0.0104 |
| logistic | AMIS / CUDA | 0.188 | 8.3e+05–1.6e+06 | — | 0.00239 | 0.00714 |
| logistic | DM-PMC / CUDA | 0.127 | 1.8e+05–3.8e+05 | — | 0.00497 | 0.021 |
| logistic | CAIS / CUDA | 0.0579 | 1.9e+06–3.2e+06 | — | 0.00377 | 0.0099 |
| logistic | LAIS-RAM / CUDA | 2.91 | 2.4e+04–2.5e+04 | — | 0.0052 | 0.0144 |
| poisson | IS / CUDA | 0.0807 | 8.9e+05–5.2e+06 | — | 0.00199 | 0.00926 |
| poisson | AMIS / CUDA | 0.139 | 8.7e+05–2.1e+06 | — | 0.00274 | 0.0083 |
| poisson | DM-PMC / CUDA | 0.201 | 2e+05–4.7e+05 | — | 0.00226 | 0.0101 |
| poisson | CAIS / CUDA | 0.0441 | 2.7e+06–4e+06 | — | 0.0046 | 0.008 |
| poisson | LAIS-RAM / CUDA | 1.29 | 3.7e+04–3.9e+04 | — | 0.00569 | 0.0108 |
| robust | IS / CUDA | 0.0359 | 3.5e+06–4.9e+06 | — | 0.00439 | 0.00817 |
| robust | AMIS / CUDA | 0.175 | 8.6e+05–1.7e+06 | — | 0.00272 | 0.00736 |
| robust | DM-PMC / CUDA | 0.133 | 1.5e+05–4.1e+05 | — | 0.00758 | 0.0163 |
| robust | CAIS / CUDA | 0.051 | 2.2e+06–3.1e+06 | — | 0.00314 | 0.0119 |
| robust | LAIS-RAM / CUDA | 1.51 | 3.8e+04–4.1e+04 | — | 0.00191 | 0.00999 |
| eight_schools | IS / CUDA | 0.0101 | 9.4e+02–5.3e+05 | — | 0.288 | 1.63 |
| eight_schools | AMIS / CUDA | 0.105 | 6.8e+05–2.2e+06 | — | 0.00741 | 0.0474 |
| eight_schools | DM-PMC / CUDA | 0.0679 | 6.8e+04–3.9e+05 | — | 0.0163 | 0.0988 |
| eight_schools | CAIS / CUDA | 0.0783 | 2.5e+04–3.8e+05 | — | 0.0139 | 0.166 |
| eight_schools | LAIS-RAM / CUDA | 0.138 | 3.3e+05–4.9e+05 | — | 0.00869 | 0.043 |

| Package | Version |
|:--|:--|
| AbstractMCMC | 5.16.0 |
| AdvancedHMC | 0.8.7 |
| AdvancedMH | 0.8.10 |
| BenchmarkTools | 1.8.0 |
| CUDA | 6.4.0 |
| Distributions | 0.25.131 |
| EnsembleMCMC | 0.0.1 |
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

Source: `a66bb278792124f694ae1bdaa6d0b45b4b82ae8d`. Manifest SHA-256: `e48da7e1ce3825329d34ee837e045c7e262c5a6116c082b56e3143d6870fc663`.
