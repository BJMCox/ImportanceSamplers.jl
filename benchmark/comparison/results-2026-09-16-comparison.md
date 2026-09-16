Effective sample size per second (ESS/s)\*.

### CPU

| Sampler | Linear (32) | Logistic (12) | Poisson (12) | Robust (13) | Eight schools (10) |
|:--|--:|--:|--:|--:|--:|
| IS | **5.1e+04** | 7.4e+04 | 1.8e+05 | 1.6e+05 | 2.9e+04§ |
| AMIS | 4.9e+04 | **8.6e+04** | 1.8e+05 | **1.9e+05** | **7.6e+05** |
| DM-PMC | 49¶ | 7.7e+03 | 1.6e+04 | 1.4e+04 | 4.8e+04 |
| CAIS | 2e+04 | 6.7e+04 | 1.5e+05 | 1.4e+05 | 1.8e+05§ |
| LAIS-RAM | 12¶ | 4.9e+03 | 1.2e+04 | 8.3e+03 | 7e+04 |
| AdvancedHMC NUTS | 1.5e+04 | 9e+03 | 2.1e+04 | 5e+04 | 1.3e+05† |
| AdvancedMH RWMH | 8.5e+02‡ | 2.6e+03 | 5.2e+03 | 4.5e+03 | 1.8e+04 |
| SliceSampling | 7.7e+02 | 1.9e+03 | 5.6e+03 | 4.4e+03 | 1.6e+05 |
| EnsembleMCMC DE | 67‡ | 8.5e+02‡ | 1.3e+03‡ | 1.2e+03‡ | 5e+03‡ |
| First-order GRAMIS-CAIS | 4.4e+04 | 8.3e+04 | **1.9e+05** | 1.6e+05 | 1.9e+04 |
| EnsembleMCMC Stretch | 23‡ | 2.1e+02‡ | 4.7e+02‡ | 2.8e+02‡ | 8.6e+02‡ |
| EnsembleMCMC snooker | 29‡ | 1.2e+02‡ | 1.5e+02‡ | 2.3e+02‡ | 4.2e+02‡ |

### CUDA

| Sampler | Linear (32) | Logistic (12) | Poisson (12) | Robust (13) | Eight schools (10) |
|:--|--:|--:|--:|--:|--:|
| IS | 5.4e+05 | 2.6e+06 | 3.3e+06 | 3.7e+06 | 4.8e+04 |
| AMIS | 3.5e+05 | 9.4e+05 | 1.4e+06 | 1.3e+06 | 1.6e+06 |
| DM-PMC | 1.8e+02¶ | 9.2e+04 | 2.2e+05 | 1.8e+05 | 3.3e+05 |
| CAIS | 1.9e+05 | 2.1e+06 | 2.4e+06 | 2.9e+06 | 3.5e+05 |
| LAIS-RAM | 4.8¶ | 7.5e+03 | 1.2e+04 | 1.4e+04 | 2.3e+05 |
| First-order GRAMIS-CAIS | 3.6e+05 | 1.4e+06 | 2e+06 | 2.1e+06 | 1.5e+05 |


\* ESS denotes weight ESS for importance sampling, minimum bulk ESS for independent-chain MCMC, and minimum mean ESS for EnsembleMCMC. Ensemble mean ESS is marginal variance divided by the squared MCSE of the sweep-mean process. These diagnostics do not define an equal-accuracy comparison.

Bold denotes the highest measured CPU rate per model. GPU results appear separately.

† At least one retained NUTS transition diverged. ‡ At least one run had maximum R-hat above 1.01. § Weight ESS varied by more than a factor of ten across seeds.

¶ At least one run exceeded a mean error of 0.2 posterior standard deviations or a marginal variance error of 30%.

EnsembleMCMC uses 4d walkers and rounds the retained count up to a complete sweep. Its R-hat splits the sweep-mean time series, not the walkers.

Host conditions: Shared host, 128 logical CPUs. Cpus_allowed_list:	0-3,16-21,23,28-31,62. Load averages: [57.86, 56.04, 60.51]; [70.96, 65.82, 63.15].

Elapsed time includes initialization, warmup or adaptation, sampling, and posterior-mean estimation. Compilation and post-run diagnostics are excluded.

Refreshed rows use the median of up to 3 executions per seed under a five-second BenchmarkTools budget. A complete execution may exceed that budget. Raw times and GC times are retained.

Archived rows retain one timed execution per seed and mean checks. They did not record per-run variances. A dash denotes an unavailable variance check, not a pass.

DM-PMC and LAIS use an independent 4096-draw width pilot. Its full cost is timed. Only proposal widths pass to the main run; its settings and retained sample count are unchanged.

| Model | Sampler / device | Mean pilot seconds | Width multiplier range |
|:--|:--|--:|--:|
| linear | DM-PMC / CPU | 0.165 | 0.807–0.81 |
| linear | LAIS-RAM / CPU | 0.162 | 0.807–0.81 |
| linear | DM-PMC / CUDA | 0.0297 | 0.807–0.81 |
| linear | LAIS-RAM / CUDA | 0.028 | 0.807–0.81 |
| logistic | DM-PMC / CPU | 0.162 | 0.79–0.792 |
| logistic | LAIS-RAM / CPU | 0.138 | 0.79–0.792 |
| logistic | DM-PMC / CUDA | 0.0107 | 0.789–0.792 |
| logistic | LAIS-RAM / CUDA | 0.00857 | 0.789–0.792 |
| poisson | DM-PMC / CPU | 0.0686 | 0.787–0.789 |
| poisson | LAIS-RAM / CPU | 0.109 | 0.787–0.789 |
| poisson | DM-PMC / CUDA | 0.00927 | 0.785–0.789 |
| poisson | LAIS-RAM / CUDA | 0.00617 | 0.785–0.789 |
| robust | DM-PMC / CPU | 0.0685 | 0.803–0.815 |
| robust | LAIS-RAM / CPU | 0.0764 | 0.803–0.815 |
| robust | DM-PMC / CUDA | 0.00725 | 0.803–0.808 |
| robust | LAIS-RAM / CUDA | 0.00571 | 0.803–0.808 |
| eight_schools | DM-PMC / CPU | 0.0669 | 1.11–1.2 |
| eight_schools | LAIS-RAM / CPU | 0.0326 | 1.11–1.2 |
| eight_schools | DM-PMC / CUDA | 0.0039 | 1.2–1.75 |
| eight_schools | LAIS-RAM / CUDA | 0.00401 | 1.2–1.75 |

Julia 1.13.0. AMD EPYC 7702P 64-Core Processor. Julia threads: 16. BLAS threads: 1.
GPU: NVIDIA A100-PCIE-40GB.

3 independent seeds. IS samples: 262144. MCMC samples: 262144. MCMC chains: 16. MCMC warmup: 1024 per chain. Ensemble warmup: 1024 sweeps.

| Model | Sampler / device | Mean seconds | ESS/s range across seeds | Max R-hat | Max pooled-mean error / posterior SD | Max per-run variance error |
|:--|:--|--:|--:|--:|--:|--:|
| linear | IS / CPU | 2.23 | 4.8e+04–5.6e+04 | — | 0.00389 | — |
| linear | AMIS / CPU | 2.79 | 4.7e+04–5e+04 | — | 0.00491 | — |
| linear | DM-PMC / CPU | 2.63 | 30–88 | — | 0.132 | 0.675 |
| linear | CAIS / CPU | 2.52 | 1.7e+04–2.3e+04 | — | 0.0053 | — |
| linear | LAIS-RAM / CPU | 3 | 6.3–15 | — | 0.173 | 0.462 |
| linear | AdvancedHMC NUTS / CPU | 36.1 | 1.4e+04–1.5e+04 | 1.000 | 0.00157 | — |
| linear | AdvancedMH RWMH / CPU | 2.88 | 8.3e+02–8.8e+02 | 1.010 | 0.0218 | — |
| linear | SliceSampling / CPU | 312 | 7.6e+02–7.7e+02 | 1.000 | 0.00248 | — |
| linear | EnsembleMCMC DE / CPU | 5.97 | 30–1.2e+02 | 1.390 | 0.029 | 0.0599 |
| linear | IS / CUDA | 0.211 | 3.7e+05–9.9e+05 | — | 0.00362 | — |
| linear | AMIS / CUDA | 0.392 | 2.7e+05–4.2e+05 | — | 0.00365 | — |
| linear | DM-PMC / CUDA | 0.269 | 44–2.8e+02 | — | 0.2 | 0.539 |
| linear | CAIS / CUDA | 0.243 | 1.7e+05–2.2e+05 | — | 0.00534 | — |
| linear | LAIS-RAM / CUDA | 4.15 | 1.8–10 | — | 0.369 | 1.15 |
| linear | First-order GRAMIS-CAIS / CPU | 2.56 | 4.4e+04–4.5e+04 | — | 0.00614 | 0.0117 |
| linear | First-order GRAMIS-CAIS / CUDA | 0.315 | 3e+05–4.3e+05 | — | 0.00565 | 0.0106 |
| linear | EnsembleMCMC Stretch / CPU | 5.93 | 21–24 | 2.076 | 0.0704 | 0.128 |
| linear | EnsembleMCMC snooker / CPU | 6.65 | 20–35 | 1.827 | 0.0357 | 0.161 |
| logistic | IS / CPU | 2.14 | 7.1e+04–7.6e+04 | — | 0.00335 | — |
| logistic | AMIS / CPU | 2.25 | 8.4e+04–8.7e+04 | — | 0.00303 | — |
| logistic | DM-PMC / CPU | 2.2 | 6.6e+03–9.3e+03 | — | 0.00784 | 0.0222 |
| logistic | CAIS / CPU | 2.15 | 6.6e+04–7e+04 | — | 0.00415 | — |
| logistic | LAIS-RAM / CPU | 2.63 | 4.2e+03–6e+03 | — | 0.00546 | 0.0243 |
| logistic | AdvancedHMC NUTS / CPU | 59.8 | 4.3e+03–2.3e+04 | 1.000 | 0.00286 | — |
| logistic | AdvancedMH RWMH / CPU | 2.47 | 2.6e+03–2.7e+03 | 1.004 | 0.0106 | — |
| logistic | SliceSampling / CPU | 125 | 1.8e+03–2e+03 | 1.000 | 0.00293 | — |
| logistic | EnsembleMCMC DE / CPU | 5.1 | 7.1e+02–1.1e+03 | 1.031 | 0.0146 | 0.029 |
| logistic | IS / CUDA | 0.0612 | 2.4e+06–2.7e+06 | — | 0.00457 | — |
| logistic | AMIS / CUDA | 0.206 | 7.7e+05–1.5e+06 | — | 0.00233 | — |
| logistic | DM-PMC / CUDA | 0.13 | 3.9e+04–2.1e+05 | — | 0.013 | 0.0191 |
| logistic | CAIS / CUDA | 0.0701 | 1.8e+06–2.5e+06 | — | 0.00426 | — |
| logistic | LAIS-RAM / CUDA | 2.04 | 4.7e+03–8.8e+03 | — | 0.00757 | 0.0262 |
| logistic | First-order GRAMIS-CAIS / CPU | 2.13 | 8.1e+04–8.5e+04 | — | 0.00333 | 0.00914 |
| logistic | First-order GRAMIS-CAIS / CUDA | 0.128 | 1e+06–1.8e+06 | — | 0.00274 | 0.00721 |
| logistic | EnsembleMCMC Stretch / CPU | 5.13 | 1.7e+02–2.7e+02 | 1.086 | 0.0156 | 0.0555 |
| logistic | EnsembleMCMC snooker / CPU | 5.38 | 37–1.9e+02 | 1.162 | 0.0304 | 0.053 |
| poisson | IS / CPU | 0.837 | 1.8e+05–1.9e+05 | — | 0.00431 | — |
| poisson | AMIS / CPU | 1.04 | 1.7e+05–2e+05 | — | 0.00228 | — |
| poisson | DM-PMC / CPU | 0.894 | 1.2e+04–2.3e+04 | — | 0.0134 | 0.0178 |
| poisson | CAIS / CPU | 0.92 | 1.4e+05–1.7e+05 | — | 0.00348 | — |
| poisson | LAIS-RAM / CPU | 1.27 | 7.3e+03–1.7e+04 | — | 0.0119 | 0.0193 |
| poisson | AdvancedHMC NUTS / CPU | 25.6 | 1.1e+04–4.1e+04 | 1.000 | 0.00249 | — |
| poisson | AdvancedMH RWMH / CPU | 1.28 | 3.8e+03–6.3e+03 | 1.004 | 0.0205 | — |
| poisson | SliceSampling / CPU | 42.3 | 4.9e+03–6.2e+03 | 1.000 | 0.00236 | — |
| poisson | EnsembleMCMC DE / CPU | 2.3 | 9.3e+02–1.9e+03 | 1.057 | 0.0152 | 0.0301 |
| poisson | IS / CUDA | 0.0462 | 2.5e+06–6.2e+06 | — | 0.00257 | — |
| poisson | AMIS / CUDA | 0.139 | 9.7e+05–2.4e+06 | — | 0.00305 | — |
| poisson | DM-PMC / CUDA | 0.0695 | 1.3e+05–3.5e+05 | — | 0.00691 | 0.019 |
| poisson | CAIS / CUDA | 0.0599 | 2e+06–3.2e+06 | — | 0.00359 | — |
| poisson | LAIS-RAM / CUDA | 1.16 | 8.7e+03–1.4e+04 | — | 0.0076 | 0.021 |
| poisson | First-order GRAMIS-CAIS / CPU | 0.897 | 1.9e+05–1.9e+05 | — | 0.00191 | 0.0109 |
| poisson | First-order GRAMIS-CAIS / CUDA | 0.0871 | 1.7e+06–2.1e+06 | — | 0.00188 | 0.0083 |
| poisson | EnsembleMCMC Stretch / CPU | 2.24 | 4.6e+02–4.8e+02 | 1.082 | 0.0197 | 0.0655 |
| poisson | EnsembleMCMC snooker / CPU | 2.73 | 64–2.3e+02 | 1.185 | 0.0177 | 0.0571 |
| robust | IS / CPU | 0.915 | 1.5e+05–1.7e+05 | — | 0.00146 | — |
| robust | AMIS / CPU | 0.998 | 1.8e+05–1.9e+05 | — | 0.00356 | — |
| robust | DM-PMC / CPU | 0.981 | 1.3e+04–1.4e+04 | — | 0.00682 | 0.0213 |
| robust | CAIS / CPU | 0.958 | 1.2e+05–1.6e+05 | — | 0.00292 | — |
| robust | LAIS-RAM / CPU | 1.26 | 4.2e+03–1.1e+04 | — | 0.0127 | 0.0247 |
| robust | AdvancedHMC NUTS / CPU | 11.1 | 4.6e+04–5.3e+04 | 1.000 | 0.00231 | — |
| robust | AdvancedMH RWMH / CPU | 1.35 | 3.3e+03–5.8e+03 | 1.004 | 0.0164 | — |
| robust | SliceSampling / CPU | 52.7 | 4.2e+03–4.6e+03 | 1.000 | 0.00229 | — |
| robust | EnsembleMCMC DE / CPU | 2.58 | 7e+02–1.7e+03 | 1.047 | 0.0161 | 0.0318 |
| robust | IS / CUDA | 0.0403 | 2.4e+06–6.8e+06 | — | 0.00232 | — |
| robust | AMIS / CUDA | 0.148 | 9.3e+05–2e+06 | — | 0.00233 | — |
| robust | DM-PMC / CUDA | 0.0575 | 1.6e+05–2.2e+05 | — | 0.00774 | 0.0183 |
| robust | CAIS / CUDA | 0.0473 | 2.3e+06–4.1e+06 | — | 0.00323 | — |
| robust | LAIS-RAM / CUDA | 0.966 | 9.7e+03–1.8e+04 | — | 0.00814 | 0.0172 |
| robust | First-order GRAMIS-CAIS / CPU | 0.991 | 1.5e+05–1.6e+05 | — | 0.00184 | 0.0122 |
| robust | First-order GRAMIS-CAIS / CUDA | 0.0743 | 1.7e+06–2.7e+06 | — | 0.0026 | 0.0107 |
| robust | EnsembleMCMC Stretch / CPU | 2.26 | 1.3e+02–4.1e+02 | 1.222 | 0.019 | 0.0584 |
| robust | EnsembleMCMC snooker / CPU | 2.75 | 63–4.3e+02 | 1.210 | 0.0201 | 0.0436 |
| eight_schools | IS / CPU | 0.112 | 1.3e+03–1.2e+05 | — | 0.0292 | — |
| eight_schools | AMIS / CPU | 0.156 | 5.9e+05–9.3e+05 | — | 0.00516 | — |
| eight_schools | DM-PMC / CPU | 0.157 | 2.1e+04–8.6e+04 | — | 0.018 | 0.177 |
| eight_schools | CAIS / CPU | 0.0911 | 2.3e+03–2.7e+05 | — | 0.049 | — |
| eight_schools | LAIS-RAM / CPU | 0.25 | 6.7e+04–7.7e+04 | — | 0.00945 | 0.119 |
| eight_schools | AdvancedHMC NUTS / CPU | 1.16 | 1.1e+05–1.9e+05 | 1.000 | 0.00167 | — |
| eight_schools | AdvancedMH RWMH / CPU | 0.152 | 1.5e+04–2e+04 | 1.006 | 0.0191 | — |
| eight_schools | SliceSampling / CPU | 0.853 | 1.1e+05–2.1e+05 | 1.000 | 0.00267 | — |
| eight_schools | EnsembleMCMC DE / CPU | 0.8 | 4.2e+03–5.7e+03 | 1.022 | 0.0134 | 0.0706 |
| eight_schools | IS / CUDA | 0.0332 | 3.5e+04–7.9e+05 | — | 0.0403 | — |
| eight_schools | AMIS / CUDA | 0.0788 | 1.2e+06–3.2e+06 | — | 0.00659 | — |
| eight_schools | DM-PMC / CUDA | 0.0268 | 2.4e+05–4e+05 | — | 0.0166 | 0.149 |
| eight_schools | CAIS / CUDA | 0.0279 | 1.3e+05–1.2e+06 | — | 0.0168 | — |
| eight_schools | LAIS-RAM / CUDA | 0.0524 | 9.1e+04–4.1e+05 | — | 0.0132 | 0.117 |
| eight_schools | First-order GRAMIS-CAIS / CPU | 0.161 | 5.7e+03–3.7e+04 | — | 0.0325 | 0.268 |
| eight_schools | First-order GRAMIS-CAIS / CUDA | 0.0291 | 1.1e+05–2.2e+05 | — | 0.0547 | 0.247 |
| eight_schools | EnsembleMCMC Stretch / CPU | 0.724 | 1.7e+02–2.1e+03 | 1.221 | 0.0094 | 0.0618 |
| eight_schools | EnsembleMCMC snooker / CPU | 1.33 | 1.4e+02–7.9e+02 | 1.200 | 0.0355 | 0.0939 |

| Model | Sampler / device | Timed executions | Raw seconds range |
|:--|:--|--:|--:|
| linear | IS / CPU | 3 | 2.03–2.39 |
| linear | AMIS / CPU | 3 | 2.73–2.87 |
| linear | DM-PMC / CPU | 6 | 2.47–2.86 |
| linear | CAIS / CPU | 3 | 2.36–2.65 |
| linear | LAIS-RAM / CPU | 6 | 2.96–3.08 |
| linear | AdvancedHMC NUTS / CPU | 3 | 34.9–37.8 |
| linear | AdvancedMH RWMH / CPU | 3 | 2.82–2.93 |
| linear | SliceSampling / CPU | 3 | 309–317 |
| linear | EnsembleMCMC DE / CPU | 3 | 5.93–6.04 |
| linear | IS / CUDA | 3 | 0.115–0.31 |
| linear | AMIS / CUDA | 3 | 0.325–0.495 |
| linear | DM-PMC / CUDA | 9 | 0.208–1.93 |
| linear | CAIS / CUDA | 3 | 0.207–0.296 |
| linear | LAIS-RAM / CUDA | 6 | 4.09–4.22 |
| linear | First-order GRAMIS-CAIS / CPU | 6 | 2.48–2.67 |
| linear | First-order GRAMIS-CAIS / CUDA | 9 | 0.259–0.402 |
| linear | EnsembleMCMC Stretch / CPU | 3 | 5.89–5.96 |
| linear | EnsembleMCMC snooker / CPU | 3 | 6.6–6.73 |
| logistic | IS / CPU | 3 | 2.09–2.24 |
| logistic | AMIS / CPU | 3 | 2.22–2.31 |
| logistic | DM-PMC / CPU | 9 | 2.14–2.31 |
| logistic | CAIS / CPU | 3 | 2.1–2.2 |
| logistic | LAIS-RAM / CPU | 6 | 2.51–2.73 |
| logistic | AdvancedHMC NUTS / CPU | 3 | 24.6–121 |
| logistic | AdvancedMH RWMH / CPU | 3 | 2.44–2.53 |
| logistic | SliceSampling / CPU | 3 | 120–130 |
| logistic | EnsembleMCMC DE / CPU | 5 | 4.84–5.39 |
| logistic | IS / CUDA | 3 | 0.0579–0.0649 |
| logistic | AMIS / CUDA | 3 | 0.125–0.25 |
| logistic | DM-PMC / CUDA | 9 | 0.066–1.75 |
| logistic | CAIS / CUDA | 3 | 0.0554–0.0818 |
| logistic | LAIS-RAM / CUDA | 9 | 2.03–2.14 |
| logistic | First-order GRAMIS-CAIS / CPU | 9 | 2.04–2.34 |
| logistic | First-order GRAMIS-CAIS / CUDA | 9 | 0.0943–0.271 |
| logistic | EnsembleMCMC Stretch / CPU | 4 | 4.77–5.36 |
| logistic | EnsembleMCMC snooker / CPU | 3 | 5.27–5.55 |
| poisson | IS / CPU | 3 | 0.809–0.856 |
| poisson | AMIS / CPU | 3 | 0.944–1.09 |
| poisson | DM-PMC / CPU | 9 | 0.865–1.03 |
| poisson | CAIS / CPU | 3 | 0.856–1.01 |
| poisson | LAIS-RAM / CPU | 9 | 1.05–1.36 |
| poisson | AdvancedHMC NUTS / CPU | 3 | 13.2–47.9 |
| poisson | AdvancedMH RWMH / CPU | 3 | 1.08–1.68 |
| poisson | SliceSampling / CPU | 3 | 39–47.8 |
| poisson | EnsembleMCMC DE / CPU | 9 | 2.01–2.51 |
| poisson | IS / CUDA | 3 | 0.025–0.0623 |
| poisson | AMIS / CUDA | 3 | 0.0784–0.197 |
| poisson | DM-PMC / CUDA | 9 | 0.0508–0.191 |
| poisson | CAIS / CUDA | 3 | 0.0455–0.0708 |
| poisson | LAIS-RAM / CUDA | 9 | 1.15–1.28 |
| poisson | First-order GRAMIS-CAIS / CPU | 9 | 0.813–1.03 |
| poisson | First-order GRAMIS-CAIS / CUDA | 9 | 0.0643–0.235 |
| poisson | EnsembleMCMC Stretch / CPU | 9 | 2.03–2.77 |
| poisson | EnsembleMCMC snooker / CPU | 6 | 2.51–3.08 |
| robust | IS / CPU | 3 | 0.888–0.959 |
| robust | AMIS / CPU | 3 | 0.965–1.02 |
| robust | DM-PMC / CPU | 9 | 0.923–1.05 |
| robust | CAIS / CPU | 3 | 0.894–1.07 |
| robust | LAIS-RAM / CPU | 9 | 1.12–1.44 |
| robust | AdvancedHMC NUTS / CPU | 3 | 10.5–12.2 |
| robust | AdvancedMH RWMH / CPU | 3 | 1.05–1.83 |
| robust | SliceSampling / CPU | 3 | 51.2–55.1 |
| robust | EnsembleMCMC DE / CPU | 7 | 2.13–2.99 |
| robust | IS / CUDA | 3 | 0.0217–0.0627 |
| robust | AMIS / CUDA | 3 | 0.0924–0.201 |
| robust | DM-PMC / CUDA | 9 | 0.0424–0.189 |
| robust | CAIS / CUDA | 3 | 0.0323–0.0587 |
| robust | LAIS-RAM / CUDA | 9 | 0.953–1.09 |
| robust | First-order GRAMIS-CAIS / CPU | 9 | 0.881–1.1 |
| robust | First-order GRAMIS-CAIS / CUDA | 9 | 0.0522–0.208 |
| robust | EnsembleMCMC Stretch / CPU | 9 | 2.17–2.54 |
| robust | EnsembleMCMC snooker / CPU | 6 | 2.52–3.05 |
| eight_schools | IS / CPU | 3 | 0.0589–0.185 |
| eight_schools | AMIS / CPU | 3 | 0.137–0.185 |
| eight_schools | DM-PMC / CPU | 9 | 0.137–0.302 |
| eight_schools | CAIS / CPU | 3 | 0.0883–0.0949 |
| eight_schools | LAIS-RAM / CPU | 9 | 0.22–0.352 |
| eight_schools | AdvancedHMC NUTS / CPU | 3 | 0.824–1.35 |
| eight_schools | AdvancedMH RWMH / CPU | 3 | 0.13–0.175 |
| eight_schools | SliceSampling / CPU | 3 | 0.636–1.18 |
| eight_schools | EnsembleMCMC DE / CPU | 9 | 0.646–1.07 |
| eight_schools | IS / CUDA | 3 | 0.00158–0.0636 |
| eight_schools | AMIS / CUDA | 3 | 0.0354–0.12 |
| eight_schools | DM-PMC / CUDA | 9 | 0.0178–0.151 |
| eight_schools | CAIS / CUDA | 3 | 0.0136–0.0367 |
| eight_schools | LAIS-RAM / CUDA | 9 | 0.0424–0.179 |
| eight_schools | First-order GRAMIS-CAIS / CPU | 9 | 0.142–0.27 |
| eight_schools | First-order GRAMIS-CAIS / CUDA | 9 | 0.0232–0.153 |
| eight_schools | EnsembleMCMC Stretch / CPU | 9 | 0.631–0.894 |
| eight_schools | EnsembleMCMC snooker / CPU | 9 | 1.08–1.94 |

## Measurement sources

Unchanged rows reuse archived measurements. Source files retain their original seeds, timing protocol, and hashes.

| Model | Sampler / device | Source |
|:--|:--|:--|
| linear | IS / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| linear | AMIS / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| linear | DM-PMC / CPU | [results-2026-09-16-partial.toml](results-2026-09-16-partial.toml) |
| linear | CAIS / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| linear | LAIS-RAM / CPU | [results-2026-09-16-partial.toml](results-2026-09-16-partial.toml) |
| linear | AdvancedHMC NUTS / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| linear | AdvancedMH RWMH / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| linear | SliceSampling / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| linear | EnsembleMCMC DE / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| linear | IS / CUDA | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| linear | AMIS / CUDA | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| linear | DM-PMC / CUDA | [results-2026-09-16-partial.toml](results-2026-09-16-partial.toml) |
| linear | CAIS / CUDA | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| linear | LAIS-RAM / CUDA | [results-2026-09-16-partial.toml](results-2026-09-16-partial.toml) |
| linear | First-order GRAMIS-CAIS / CPU | [results-2026-09-16-partial.toml](results-2026-09-16-partial.toml) |
| linear | First-order GRAMIS-CAIS / CUDA | [results-2026-09-16-partial.toml](results-2026-09-16-partial.toml) |
| linear | EnsembleMCMC Stretch / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| linear | EnsembleMCMC snooker / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| logistic | IS / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| logistic | AMIS / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| logistic | DM-PMC / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| logistic | CAIS / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| logistic | LAIS-RAM / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| logistic | AdvancedHMC NUTS / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| logistic | AdvancedMH RWMH / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| logistic | SliceSampling / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| logistic | EnsembleMCMC DE / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| logistic | IS / CUDA | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| logistic | AMIS / CUDA | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| logistic | DM-PMC / CUDA | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| logistic | CAIS / CUDA | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| logistic | LAIS-RAM / CUDA | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| logistic | First-order GRAMIS-CAIS / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| logistic | First-order GRAMIS-CAIS / CUDA | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| logistic | EnsembleMCMC Stretch / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| logistic | EnsembleMCMC snooker / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| poisson | IS / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| poisson | AMIS / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| poisson | DM-PMC / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| poisson | CAIS / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| poisson | LAIS-RAM / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| poisson | AdvancedHMC NUTS / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| poisson | AdvancedMH RWMH / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| poisson | SliceSampling / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| poisson | EnsembleMCMC DE / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| poisson | IS / CUDA | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| poisson | AMIS / CUDA | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| poisson | DM-PMC / CUDA | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| poisson | CAIS / CUDA | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| poisson | LAIS-RAM / CUDA | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| poisson | First-order GRAMIS-CAIS / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| poisson | First-order GRAMIS-CAIS / CUDA | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| poisson | EnsembleMCMC Stretch / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| poisson | EnsembleMCMC snooker / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| robust | IS / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| robust | AMIS / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| robust | DM-PMC / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| robust | CAIS / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| robust | LAIS-RAM / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| robust | AdvancedHMC NUTS / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| robust | AdvancedMH RWMH / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| robust | SliceSampling / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| robust | EnsembleMCMC DE / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| robust | IS / CUDA | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| robust | AMIS / CUDA | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| robust | DM-PMC / CUDA | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| robust | CAIS / CUDA | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| robust | LAIS-RAM / CUDA | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| robust | First-order GRAMIS-CAIS / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| robust | First-order GRAMIS-CAIS / CUDA | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| robust | EnsembleMCMC Stretch / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| robust | EnsembleMCMC snooker / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| eight_schools | IS / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| eight_schools | AMIS / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| eight_schools | DM-PMC / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| eight_schools | CAIS / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| eight_schools | LAIS-RAM / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| eight_schools | AdvancedHMC NUTS / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| eight_schools | AdvancedMH RWMH / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| eight_schools | SliceSampling / CPU | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| eight_schools | EnsembleMCMC DE / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| eight_schools | IS / CUDA | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| eight_schools | AMIS / CUDA | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| eight_schools | DM-PMC / CUDA | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| eight_schools | CAIS / CUDA | [results-2026-09-16-long.toml](results-2026-09-16-long.toml) |
| eight_schools | LAIS-RAM / CUDA | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| eight_schools | First-order GRAMIS-CAIS / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| eight_schools | First-order GRAMIS-CAIS / CUDA | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| eight_schools | EnsembleMCMC Stretch / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |
| eight_schools | EnsembleMCMC snooker / CPU | [results-2026-09-16-refresh.toml](results-2026-09-16-refresh.toml) |

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

Archived baseline source: `ec5aeec3c59b48a2e6237a312b5bd9c909b4c107`. Manifest SHA-256: `e48da7e1ce3825329d34ee837e045c7e262c5a6116c082b56e3143d6870fc663`.
