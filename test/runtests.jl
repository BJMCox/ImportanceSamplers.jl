using Test
using ImportanceSamplers

include("support/proposals.jl")

test_files = isempty(ARGS) ? ["algorithm", "plain_is", "threading", "results", "failures"] : ARGS
for test_file in test_files
    include("$(test_file).jl")
end
