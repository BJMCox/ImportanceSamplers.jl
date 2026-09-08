module CAISBenchmark

using BenchmarkTools
using ImportanceSamplers
using MLDataDevices
using Random

struct CorrelatedGaussianTarget{T} end
function (::CorrelatedGaussianTarget{T})(x) where {T}
    first = x[1] - T(0.5)
    total = abs2(first)
    @inbounds for coordinate in 2:length(x)
        total += abs2(x[coordinate] + T(0.2) * first)
    end
    return -total / T(2)
end

function measure(
    device;
    dimension=4,
    proposal_count=16,
    round_size=262_144,
    rounds=3,
    T=Float32,
    threaded=true,
    factor_execution=FusedFactorExecution(),
    shared_load_status=:not_measured,
)
    round_size % proposal_count == 0 || error("round size must divide equally")
    proposals = [
        SphericalGaussian(
            fill(T(location), dimension),
            T(2),
        ) for location in range(T(-4), T(4); length=proposal_count)
    ]
    bank = ProposalBank(proposals, ones(T, proposal_count))
    source = prepare_sampler(
        Xoshiro(42),
        CorrelatedGaussianTarget{T}(),
        CAIS(bank; rounds, round_size);
        threaded,
        factor_execution,
    )

    # Device transfer and prepared-state construction are setup, not timed.
    # Run this harness only when the machine is known to be idle. Report its
    # shared-load status with the result; do not infer comparative speed from a
    # shared or otherwise loaded host.
    trial = @benchmark importance_sample!(prepared) setup=(
        prepared=$device($source)
    ) evals=1 samples=20 seconds=10
    estimate = median(trial)
    return (
        dimension,
        proposal_count,
        round_size,
        rounds,
        scalar_type=T,
        threaded,
        factor_execution=typeof(factor_execution),
        julia_threads=Threads.nthreads(:default),
        milliseconds=estimate.time / 1e6,
        host_bytes=estimate.memory,
        host_allocations=estimate.allocs,
        samples_per_second=rounds * round_size * 1e9 / estimate.time,
        shared_load_status,
    )
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    @show CAISBenchmark.measure(MLDataDevices.cpu_device())
end
