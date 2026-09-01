const GRAMIS_SOURCE_COMMIT = get(ENV, "IMPORTANCE_SAMPLERS_SOURCE_COMMIT", "")
occursin(r"^[0-9a-f]{40}$", GRAMIS_SOURCE_COMMIT) || error("set exact source commit")
include(joinpath(@__DIR__, "..", "first_order_gramis_capabilities.jl"))
show(stdout, MIME("text/plain"), GRAMIS_CUDA_CAPABILITY_RESULT); println()
GRAMIS_CUDA_CAPABILITY_RESULT
