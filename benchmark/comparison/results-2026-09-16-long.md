# Recorded ESS/s comparison

Effective sample size per second (ESS/s)\*.

| Sampler / device | Linear (32) | Logistic (12) | Poisson (12) | Robust (13) | Eight schools (10) |
|:--|--:|--:|--:|--:|--:|
| IS / CPU | 5.1e+04 | 7.4e+04 | 1.8e+05 | 1.6e+05 | 2.9e+04§ |
| AMIS / CPU | 4.9e+04 | 8.6e+04 | 1.8e+05 | 1.9e+05 | 7.6e+05 |
| DM-PMC / CPU | 9.7 | 5.9e+03 | 1.4e+04 | 1.1e+04 | 4.8e+04§ |
| CAIS / CPU | 2e+04 | 6.7e+04 | 1.5e+05 | 1.4e+05 | 1.8e+05§ |
| LAIS-RAM / CPU | 15 | 4e+03 | 1.1e+04 | 7.3e+03 | 6.2e+04 |
| AdvancedHMC NUTS / CPU | 1.5e+04 | 9e+03 | 2.1e+04 | 5e+04 | 1.3e+05† |
| AdvancedMH RWMH / CPU | 8.5e+02‡ | 2.6e+03 | 5.2e+03 | 4.5e+03 | 1.8e+04 |
| SliceSampling / CPU | 7.7e+02 | 1.9e+03 | 5.6e+03 | 4.4e+03 | 1.6e+05 |
| IS / CUDA | **5.4e+05** | **2.6e+06** | **3.3e+06** | **3.7e+06** | 4.8e+04 |
| AMIS / CUDA | 3.5e+05 | 9.4e+05 | 1.4e+06 | 1.3e+06 | **1.6e+06** |
| DM-PMC / CUDA | 33 | 1.1e+05 | 1.3e+05 | 1.1e+05 | 2.2e+04 |
| CAIS / CUDA | 1.9e+05 | 2.1e+06 | 2.4e+06 | 2.9e+06 | 3.5e+05 |
| LAIS-RAM / CUDA | 28 | 5.4e+03 | 1e+04 | 1.1e+04 | 1.8e+05 |

\* ESS denotes weight ESS for importance sampling and minimum bulk ESS across parameters for MCMC. These diagnostics do not define an equal-accuracy comparison.

Bold denotes the highest measured rate in each model column.

† At least one retained NUTS transition diverged. ‡ At least one run had maximum R-hat above 1.01. § Weight ESS varied by more than a factor of ten across seeds.

Elapsed time includes initialization, warmup or adaptation, sampling, and posterior-mean estimation. Compilation and post-run diagnostics are excluded.

Julia 1.13.0. AMD EPYC 7702P 64-Core Processor. Julia threads: 16. BLAS threads: 1.
GPU: NVIDIA A100-PCIE-40GB.

3 independent seeds. IS samples: 262144. MCMC samples: 262144. MCMC chains: 16. MCMC warmup: 1024 per chain. Ensemble warmup: 1024 sweeps. Adaptive IS: four rounds.

| Model | Sampler / device | Mean seconds | ESS/s range across seeds | Max R-hat | Max pooled-mean error / posterior SD |
|:--|:--|--:|--:|--:|--:|
| linear | IS / CPU | 2.23 | 4.8e+04–5.6e+04 | — | 0.00389 |
| linear | AMIS / CPU | 2.79 | 4.7e+04–5e+04 | — | 0.00491 |
| linear | DM-PMC / CPU | 2.78 | 3.6–19 | — | 0.193 |
| linear | CAIS / CPU | 2.52 | 1.7e+04–2.3e+04 | — | 0.0053 |
| linear | LAIS-RAM / CPU | 2.97 | 10–18 | — | 0.184 |
| linear | AdvancedHMC NUTS / CPU | 36.1 | 1.4e+04–1.5e+04 | 1.000 | 0.00157 |
| linear | AdvancedMH RWMH / CPU | 2.88 | 8.3e+02–8.8e+02 | 1.010 | 0.0218 |
| linear | SliceSampling / CPU | 312 | 7.6e+02–7.7e+02 | 1.000 | 0.00248 |
| linear | IS / CUDA | 0.211 | 3.7e+05–9.9e+05 | — | 0.00362 |
| linear | AMIS / CUDA | 0.392 | 2.7e+05–4.2e+05 | — | 0.00365 |
| linear | DM-PMC / CUDA | 0.81 | 20–92 | — | 0.219 |
| linear | CAIS / CUDA | 0.243 | 1.7e+05–2.2e+05 | — | 0.00534 |
| linear | LAIS-RAM / CUDA | 4.13 | 13–40 | — | 0.129 |
| logistic | IS / CPU | 2.14 | 7.1e+04–7.6e+04 | — | 0.00335 |
| logistic | AMIS / CPU | 2.25 | 8.4e+04–8.7e+04 | — | 0.00303 |
| logistic | DM-PMC / CPU | 2.19 | 4.8e+03–6.9e+03 | — | 0.00629 |
| logistic | CAIS / CPU | 2.15 | 6.6e+04–7e+04 | — | 0.00415 |
| logistic | LAIS-RAM / CPU | 2.35 | 3.4e+03–4.4e+03 | — | 0.00897 |
| logistic | AdvancedHMC NUTS / CPU | 59.8 | 4.3e+03–2.3e+04 | 1.000 | 0.00286 |
| logistic | AdvancedMH RWMH / CPU | 2.47 | 2.6e+03–2.7e+03 | 1.004 | 0.0106 |
| logistic | SliceSampling / CPU | 125 | 1.8e+03–2e+03 | 1.000 | 0.00293 |
| logistic | IS / CUDA | 0.0612 | 2.4e+06–2.7e+06 | — | 0.00457 |
| logistic | AMIS / CUDA | 0.206 | 7.7e+05–1.5e+06 | — | 0.00233 |
| logistic | DM-PMC / CUDA | 0.119 | 6.5e+04–1.8e+05 | — | 0.00611 |
| logistic | CAIS / CUDA | 0.0701 | 1.8e+06–2.5e+06 | — | 0.00426 |
| logistic | LAIS-RAM / CUDA | 2.03 | 3.4e+03–9.3e+03 | — | 0.0145 |
| poisson | IS / CPU | 0.837 | 1.8e+05–1.9e+05 | — | 0.00431 |
| poisson | AMIS / CPU | 1.04 | 1.7e+05–2e+05 | — | 0.00228 |
| poisson | DM-PMC / CPU | 0.917 | 1.3e+04–1.5e+04 | — | 0.00453 |
| poisson | CAIS / CPU | 0.92 | 1.4e+05–1.7e+05 | — | 0.00348 |
| poisson | LAIS-RAM / CPU | 1.04 | 8.4e+03–1.5e+04 | — | 0.012 |
| poisson | AdvancedHMC NUTS / CPU | 25.6 | 1.1e+04–4.1e+04 | 1.000 | 0.00249 |
| poisson | AdvancedMH RWMH / CPU | 1.28 | 3.8e+03–6.3e+03 | 1.004 | 0.0205 |
| poisson | SliceSampling / CPU | 42.3 | 4.9e+03–6.2e+03 | 1.000 | 0.00236 |
| poisson | IS / CUDA | 0.0462 | 2.5e+06–6.2e+06 | — | 0.00257 |
| poisson | AMIS / CUDA | 0.139 | 9.7e+05–2.4e+06 | — | 0.00305 |
| poisson | DM-PMC / CUDA | 0.102 | 7.7e+04–3.1e+05 | — | 0.0137 |
| poisson | CAIS / CUDA | 0.0599 | 2e+06–3.2e+06 | — | 0.00359 |
| poisson | LAIS-RAM / CUDA | 1.16 | 7.2e+03–1.3e+04 | — | 0.0118 |
| robust | IS / CPU | 0.915 | 1.5e+05–1.7e+05 | — | 0.00146 |
| robust | AMIS / CPU | 0.998 | 1.8e+05–1.9e+05 | — | 0.00356 |
| robust | DM-PMC / CPU | 0.972 | 8.9e+03–1.3e+04 | — | 0.0135 |
| robust | CAIS / CPU | 0.958 | 1.2e+05–1.6e+05 | — | 0.00292 |
| robust | LAIS-RAM / CPU | 1.1 | 3.6e+03–1.4e+04 | — | 0.00866 |
| robust | AdvancedHMC NUTS / CPU | 11.1 | 4.6e+04–5.3e+04 | 1.000 | 0.00231 |
| robust | AdvancedMH RWMH / CPU | 1.35 | 3.3e+03–5.8e+03 | 1.004 | 0.0164 |
| robust | SliceSampling / CPU | 52.7 | 4.2e+03–4.6e+03 | 1.000 | 0.00229 |
| robust | IS / CUDA | 0.0403 | 2.4e+06–6.8e+06 | — | 0.00232 |
| robust | AMIS / CUDA | 0.148 | 9.3e+05–2e+06 | — | 0.00233 |
| robust | DM-PMC / CUDA | 0.0978 | 6.6e+04–3e+05 | — | 0.00846 |
| robust | CAIS / CUDA | 0.0473 | 2.3e+06–4.1e+06 | — | 0.00323 |
| robust | LAIS-RAM / CUDA | 0.967 | 6.6e+03–1.6e+04 | — | 0.011 |
| eight_schools | IS / CPU | 0.112 | 1.3e+03–1.2e+05 | — | 0.0292 |
| eight_schools | AMIS / CPU | 0.156 | 5.9e+05–9.3e+05 | — | 0.00516 |
| eight_schools | DM-PMC / CPU | 0.191 | 1.9e+03–8.5e+04 | — | 0.0365 |
| eight_schools | CAIS / CPU | 0.0911 | 2.3e+03–2.7e+05 | — | 0.049 |
| eight_schools | LAIS-RAM / CPU | 0.201 | 1e+04–1.4e+05 | — | 0.033 |
| eight_schools | AdvancedHMC NUTS / CPU | 1.16 | 1.1e+05–1.9e+05 | 1.000 | 0.00167 |
| eight_schools | AdvancedMH RWMH / CPU | 0.152 | 1.5e+04–2e+04 | 1.006 | 0.0191 |
| eight_schools | SliceSampling / CPU | 0.853 | 1.1e+05–2.1e+05 | 1.000 | 0.00267 |
| eight_schools | IS / CUDA | 0.0332 | 3.5e+04–7.9e+05 | — | 0.0403 |
| eight_schools | AMIS / CUDA | 0.0788 | 1.2e+06–3.2e+06 | — | 0.00659 |
| eight_schools | DM-PMC / CUDA | 0.303 | 6.1e+03–6.3e+05 | — | 0.0158 |
| eight_schools | CAIS / CUDA | 0.0279 | 1.3e+05–1.2e+06 | — | 0.0168 |
| eight_schools | LAIS-RAM / CUDA | 0.0692 | 8.8e+04–2.5e+05 | — | 0.0157 |

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

Source: `ec5aeec3c59b48a2e6237a312b5bd9c909b4c107`. Manifest SHA-256: `e48da7e1ce3825329d34ee837e045c7e262c5a6116c082b56e3143d6870fc663`.
