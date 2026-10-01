using Documenter
using ImportanceSamplers
using Statistics

# Check repository source links locally instead of repeatedly requesting GitHub.
for (directory, _, files) in walkdir(joinpath(@__DIR__, "src")), file in files
    endswith(file, ".md") || continue
    for link in eachmatch(r"https://github\.com/BJMCox/ImportanceSamplers\.jl/blob/main/([^\s#)]+)",
                          read(joinpath(directory, file), String))
        isfile(joinpath(@__DIR__, "..", link.captures[1])) ||
            error("Missing repository link: $(link.match)")
    end
end

makedocs(
    modules=[ImportanceSamplers],
    sitename="ImportanceSamplers.jl",
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://bjmcox.github.io/ImportanceSamplers.jl/",
        edit_link="main",
        repolink="https://github.com/BJMCox/ImportanceSamplers.jl",
    ),
    build=joinpath(@__DIR__, "build"),
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "First weighted estimate" => "tutorials/first_estimate.md",
            "Logistic regression" => "tutorials/logistic_regression.md",
            "Numerical integration" => "tutorials/integration.md",
        ],
        "User guide" => [
            "Targets and data" => "guide/targets.md",
            "Proposals" => "guide/native_proposals.md",
            "Constraints and named parameters" => "guide/transforms.md",
            "Working with results" => "guide/results.md",
            "Adaptation and reuse" => "guide/reuse.md",
        ],
        "Methods" => [
            "Choosing a method" => "methods/index.md",
            "Plain IS" => "methods/importance_sampling.md",
            "Static MIS" => "methods/static_mis.md",
            "AMIS" => "methods/amis.md",
            "N-PMC" => "methods/npmc.md",
            "DM-PMC, GR-PMC, LR-PMC" => "methods/dm_pmc.md",
            "APIS" => "methods/apis.md",
            "CAIS" => "methods/cais.md",
            "LAIS" => "methods/lais.md",
            "First-order GRAMIS-CAIS" => "methods/first_order_gramis.md",
        ],
        "Execution" => [
            "Batch targets" => "guide/batch_targets.md",
            "Gradients" => "guide/gradients.md",
            "Devices" => "guide/accelerators.md",
            "Performance" => "guide/performance.md",
        ],
        "Reference" => [
            "Targets and execution" => "reference.md",
            "Samplers and transitions" => "reference/samplers.md",
            "Proposals and transforms" => "reference/proposals.md",
            "Results and statistics" => "reference/results.md",
            "Errors" => "reference/errors.md",
        ],
        "Further information" => [
            "Custom proposals and transitions" => "guide/extensions.md",
            "Troubleshooting" => "guide/troubleshooting.md",
            "Validation" => "guide/validation.md",
            "Benchmarks" => "guide/benchmarks.md",
        ],
    ],
    doctest=true,
    checkdocs=:exports,
    linkcheck=true,
    linkcheck_ignore=[
        r"^https://github\.com/BJMCox/ImportanceSamplers\.jl/blob/main/",
    ],
    warnonly=false,
)
