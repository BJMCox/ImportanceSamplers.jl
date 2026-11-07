using Documenter

deploydocs(
    root=@__DIR__,
    repo="github.com/JuliaBayes/ImportanceSamplers.jl.git",
    devbranch="main",
    versions=["stable" => "v^", "v#.#.#", "dev" => "dev"],
)
