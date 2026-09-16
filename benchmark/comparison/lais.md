# LAIS with independently fitted proposals

The short fitted LAIS-RWM configuration improves throughput, but static IS remains
faster on these targets. Longer LAIS runs degrade the fitted proposal.
This study changes benchmark settings, not package defaults or sampler laws.

## Reproduce

Use Julia 1.13 and the comparison environment's committed manifest. From the
repository root, print the saved tables, package versions, and README rows:

```sh
julia --project=benchmark/comparison benchmark/comparison/lais.jl --report \
  benchmark/comparison/results-2026-09-16-lais-linear.toml \
  benchmark/comparison/results-2026-09-16-lais-robust.toml
```

Repeat the validation on CUDA:

```sh
julia --threads=128 --project=benchmark/comparison benchmark/comparison/lais.jl --cuda
```

Omit `--cuda` for CPU. Add `--screen` to repeat the initial linear configuration
screen. New runs use separate output names and refuse to overwrite existing files.
The script prints the table directly. Its `study` function accepts another output
path, seed list, or configuration list.

## Protocol

- Use the same 32-parameter linear and 13-parameter robust regression targets as
  the [main comparison](README.md). Retain 262,144 production draws.
- Run an independent AMIS pilot with eight rounds and 262,144 draws. Start from
  the timed Laplace fit, with Student-t proposals, eight degrees of freedom,
  and initial scale factor `1.2L`.
- Reuse the pilot's fitted parameters through `current_proposal`. Pilot and
  production RNG seeds differ. Pilot settings do not change production budgets.
- Initialize each lower proposal at the fitted mean. Keep its covariance fixed.
  Multiply the fitted Student-t factor by `sqrt(8/6)` for Gaussian proposals,
  so both families have equal covariance.
- Set upper RWM covariance to `(2.38^2/d)` times the fitted covariance.
  Four-round RWM uses no extra MCMC warmup. Upper moves remain active.
- Compare matched static Gaussian IS, RAM with 1,024 warmup steps, different
  proposal counts, and a 64-round RWM case with the same total retained count.
- Use seed 12101 for the screen. Freeze configurations before linear seeds
  12102–12103 and robust seeds 12201–12202.
- Compile each full configuration before three BenchmarkTools executions.
  Use their median time. Charge every workflow for Laplace fitting, the full
  AMIS pilot, preparation, transfers, upper moves, sampling, and the final mean.
  Compilation and post-run diagnostics are excluded.
- Check means and marginal variances against analytic linear moments or the
  [independent robust reference](accuracy-2026-09-16.toml). The gates are 0.2
  posterior SD for mean error and 30% relative variance error.

All validation rows pass those moment gates. They do not establish tail accuracy.
ESS/s in the README is total ESS divided by total elapsed time across both seeds.
The ranges below show each seed's ESS/s, including pilot cost.

## Results

| Linear method | Proposals | Rounds | ESS range | ESS/s range |
|:--|--:|--:|--:|--:|
| Static Gaussian IS | 1 | 1 | 260,905–260,995 | 342,526–359,612 |
| Student-t LAIS-RAM, 1,024 warmup | 256 | 4 | 353–392 | 69–78 |
| Gaussian LAIS-RWM, no warmup | 16 | 4 | 242,598–251,022 | 254,659–285,286 |
| Gaussian LAIS-RWM, no warmup | 256 | 4 | 251,595–253,214 | 220,431–233,192 |
| Gaussian LAIS-RWM, no warmup | 16 | 64 | 15,042–72,117 | 5,632–26,909 |

| Robust regression method | Proposals | ESS range | ESS/s range |
|:--|--:|--:|--:|
| Static Gaussian IS | 1 | 256,473–256,507 | 1,600,111–1,699,569 |
| Student-t LAIS-RAM, 1,024 warmup | 256 | 39,895–41,292 | 33,537–34,552 |
| Gaussian LAIS-RWM, no warmup | 16 | 225,202–241,319 | 1,134,860–1,136,437 |
| Student-t LAIS-RWM, no warmup | 16 | 174,995–180,814 | 586,731–897,775 |

Robust LAIS uses four production rounds. The static control uses one proposal.
Full raw records preserve timings, allocations, errors, upper evaluations,
acceptance counts, centre spread, and round ESS:

- [Configuration screen](results-2026-09-16-lais-screen.toml)
- [Linear validation](results-2026-09-16-lais-linear.toml)
- [Robust validation](results-2026-09-16-lais-robust.toml)

## Interpretation

The linear posterior is Gaussian. After fitting, static IS achieves 99.5–99.6%
weight efficiency. The robust control achieves about 97.8%. These targets leave
little room for adaptation to improve weight ESS.

The 16-centre, four-round Gaussian LAIS runs accept only three and five of 64
upper moves on linear regression. Their gain mostly comes from retaining the
pilot's good fit and avoiding costly extra warmup. This is not evidence that
LAIS adaptation beats static IS.

At 64 rounds, the same chains accept 126 and 149 moves. Their centres spread,
and ESS/s falls substantially. The lower covariance stays fixed while centres
move. RAM adapts the upper transition covariance, not the lower proposal width.

Short fitted LAIS-RWM is therefore a useful configuration, not a general fix.
The archived LAIS-RAM rows use a different pilot, proposal family, and transition.
Keep those labels distinct. The [main comparison's ESS caveats](README.md#ensemble-ess)
still apply: weight ESS is not a common accuracy metric across sampler classes.

## Provenance

Measurements used Julia 1.13.0, an A100-PCIE-40GB, 128 Julia host threads, and one
BLAS thread on an AMD EPYC 7702P. Host load was approximately 54–61 during this
shared-host study. These are not isolated-hardware timings.

The raw files are unchanged copies from the original study. Their historical
source hashes name the original research scripts. `lais.jl` consolidates the
measured workflow into a standalone reproducer. Each new run records its own
source hashes, revision, versions, manifest hash, and host load.

The recorded manifest SHA-256 is
`e48da7e1ce3825329d34ee837e045c7e262c5a6116c082b56e3143d6870fc663`.
The committed comparison manifest matches it. The robust reference SHA-256 is
`b2ad781d7c9f9c71d116f94d1e02f1c69811cae9b753c5e5d2d0cca0150123f3`.
