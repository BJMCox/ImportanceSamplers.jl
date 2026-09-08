using Test
using ImportanceSamplers

include("support/proposals.jl")

test_files = if isempty(ARGS)
    [
        "algorithm",
        "static_mis",
        "static_mis_kernel",
        "adaptive_schedule",
        "log_mixture_accumulator",
        "amis",
        "amis_kernel",
        "npmc",
        "dm_pmc",
        "dm_pmc_kernel",
        "apis",
        "lais",
        "cais",
        "first_order_gramis",
        "first_order_gramis_kernel",
        "device_api",
        "target_derivatives",
        "plain_is",
        "retarget",
        "threading",
        "results",
        "resampling",
        "device_results",
        "failures",
        "kernel_execution",
        "native_proposals",
        "transforms",
        "product_proposal",
        "distributions_ext",
    ]
else
    ARGS
end
for test_file in test_files
    include("$(test_file).jl")
end
