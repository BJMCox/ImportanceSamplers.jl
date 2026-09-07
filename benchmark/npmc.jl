module NPMCBenchmark

using BenchmarkTools, ImportanceSamplers, MLDataDevices, Random

struct GaussianTarget{T} end
function (::GaussianTarget{T})(x) where {T}
    total = zero(T)
    @inbounds for i in eachindex(x)
        total += abs2(x[i] - T(0.5))
    end
    return -total / T(2)
end

function measure(device; method=NPMC, dimension=4, round_size=262_144, rounds=3,
                 T=Float32, threaded=true)
    source = prepare_sampler(Xoshiro(42), GaussianTarget{T}(),
        method(SphericalGaussian(zeros(T, dimension), T(2)); rounds, round_size);
        threaded)
    # Setup is outside timing. Every trial starts from the same proposal and RNG state.
    trial = @benchmark importance_sample!(prepared) setup=(prepared=$device($source)) evals=1 samples=200 seconds=5
    estimate = median(trial)
    return (; method, dimension, round_size, rounds, scalar_type=T, threaded,
        julia_threads=Threads.nthreads(:default), milliseconds=estimate.time / 1e6,
        host_bytes=estimate.memory, host_allocations=estimate.allocs,
        samples_per_second=rounds * round_size * 1e9 / estimate.time)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    @show NPMCBenchmark.measure(NPMCBenchmark.cpu_device())
end
