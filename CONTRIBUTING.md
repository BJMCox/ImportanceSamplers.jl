# Contributing

Keep sampler changes narrow. Preserve the sampling law, log weights, precision,
sample structure, RNG ownership, and device residency. Use analytic oracles for
numerical changes and interleaved BenchmarkTools runs for performance claims.

## Tests

From the repository root:

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

CI tests Julia 1.10 on Linux and the latest stable Julia on Linux, macOS, and Windows.
It also runs the Reactant CPU checks. Codecov upload failures do not block CI.
GPU execution requires separate hardware checks; passing CPU CI does not prove
GPU support. See the [validation guide](https://juliabayes.github.io/ImportanceSamplers.jl/dev/guide/validation/).

## Documentation

Build the complete site, examples, and doctests:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Serve `docs/build` with an HTTP server to preview the directory-style URLs.
The build is strict: broken internal links, undocumented exports, and failing
examples or doctests fail it. External links have a separate, non-gating CI check.

Write runnable examples before describing options. Define each term before
using it. Keep guides separate from the API reference. `DocumenterCodeBlocks`
provides the code highlighting and reference links used by Wren and EnsembleMCMC.

### Versioned deployment

`docs/make.jl` only builds the site. `docs/deploy.jl` retains versions on
`gh-pages`. After tests and docs pass, CI publishes that full branch through
GitHub Pages Actions:

| Source | Documentation |
| --- | --- |
| `main` | `/dev/` |
| A release tag such as `v0.1.0` | `/v0.1.0/` |
| Latest release | `/stable/` |

The root URL redirects to the latest release, or to `dev` before any release.
Pull requests build an artifact but do not deploy. Keep Pages configured to use
GitHub Actions. Only the deployment job receives repository write permission.

Push a release tag with credentials that trigger workflows, or dispatch `CI.yml`
on that tag. Tags created using only `GITHUB_TOKEN` do not trigger another
workflow automatically. If TagBot is added, configure its SSH key accordingly.
See [Documenter's hosting guide](https://documenter.juliadocs.org/stable/man/hosting/).
