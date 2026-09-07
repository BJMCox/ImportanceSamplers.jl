module APISBenchmark

using BenchmarkTools, ImportanceSamplers, MLDataDevices, Random

struct GaussianTarget{T} end
function (::GaussianTarget{T})(x) where {T}
    total = zero(T)
    @inbounds for coordinate in eachindex(x)
        total += abs2(x[coordinate] - T(0.5))
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
    shared_load_status=:not_measured,
)
    round_size % proposal_count == 0 || error("round size must divide equally")
    proposals = [
        SphericalGaussian(fill(T(location), dimension), T(2)) for
        location in range(T(-4), T(4); length=proposal_count)
    ]
    source = prepare_sampler(
        Xoshiro(42),
        GaussianTarget{T}(),
        APIS(ProposalBank(proposals); rounds, round_size);
        threaded,
    )
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
    @show APISBenchmark.measure(MLDataDevices.cpu_device())
end
