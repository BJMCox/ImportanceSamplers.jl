using Test
using ImportanceSamplers

include("support/proposals.jl")

test_files = if isempty(ARGS)
    [
        "algorithm",
        "static_mis",
        "device_api",
        "plain_is",
        "threading",
        "results",
        "device_results",
        "failures",
        "kernel_execution",
        "native_proposals",
        "transforms",
        "product_proposal",
    ]
else
    ARGS
end
for test_file in test_files
    include("$(test_file).jl")
end
