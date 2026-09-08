module LAISBenchmark

using BenchmarkTools, ImportanceSamplers, LinearAlgebra, MLDataDevices, Random, Statistics

const KA = ImportanceSamplers.KernelAbstractions

# AR(1) whitening gives correlated targets without a dense target-side solve.
struct Target{Family} end

function radius(x, shift, correlation)
    total = abs2(x[1] - shift)
    scale = one(correlation) - abs2(correlation)
    @inbounds for i in 2:length(x)
        total += abs2(x[i] - correlation * (x[i-1] - (i == 2 ? shift : zero(shift)))) / scale
    end
    return total
end

(::Target{:gaussian})(x) = -radius(x, zero(eltype(x)), eltype(x)(0.7)) / 2
function (target::Target{:student})(x)
    T = eltype(x)
    return -(T(5) + T(length(x))) / T(2) *
        log1p(radius(x, zero(T), T(0.7)) / T(5))
end
function (target::Target{:mixture})(x)
    T = eltype(x)
    a = log(T(0.3)) - radius(x, T(-3), T(0.7)) / T(2)
    b = log(T(0.7)) - radius(x, T(2), T(0.7)) / T(2)
    larger = max(a, b)
    return larger + log1p(exp(min(a, b) - larger))
end

function configuration(; dimension=4, proposal_count=8, round_size=65_536,
    rounds=10, tuning=ContinuousTuning(), T=Float32, threaded=true,
    family=:gaussian, transition_scale=1, seed=42)
    return (; dimension, proposal_count, round_size, rounds, tuning, T, threaded,
        family, transition_scale, seed)
end

function fresh(device; kwargs...)
    (; dimension, proposal_count, round_size, rounds, tuning, T, threaded,
        family, transition_scale, seed) = configuration(; kwargs...)
    factor = Matrix{T}(I, dimension, dimension)
    bank = ProposalBank([
        FactorGaussian(fill(T(location), dimension), factor) for
        location in range(-3, 3; length=proposal_count)
    ], ones(T, proposal_count))
    covariance = T(transition_scale)^2
    transition = isnothing(tuning) ? RandomWalkMetropolis(covariance) : RAM(covariance; tuning)
    algorithm = LAIS(bank; transition, rounds, round_size)
    prepared = device(prepare_sampler(Xoshiro(seed), Target{family}(), algorithm; threaded))
    KA.synchronize(KA.get_backend(prepared.method_state.bank.locations))
    return prepared
end

function execute(prepared, backend)
    result = importance_sample!(prepared)
    KA.synchronize(backend)
    return result
end

function measure(device; samples=20, seconds=10, kwargs...)
    make_fresh = () -> fresh(device; kwargs...)
    prepared = make_fresh()
    backend = KA.get_backend(prepared.method_state.bank.locations)
    execute(prepared, backend) # Compile; each timed evaluation starts unexecuted.
    trial = @benchmark execute(prepared, $backend) setup=(prepared=$make_fresh()) evals=1 samples=samples seconds=seconds
    result = execute(make_fresh(), backend)
    estimate = median(trial)
    elapsed = estimate.time / 1e9
    return (
        configuration=configuration(; kwargs...), threads=Threads.nthreads(:default),
        blas_threads=BLAS.get_num_threads(), seconds=elapsed,
        time_range_seconds=extrema(trial.times) ./ 1e9,
        host_bytes=estimate.memory, host_allocations=estimate.allocs,
        samples_per_second=length(result.logweights) / elapsed,
        ess_per_second=inv(sum(abs2, normalized_weights(result))) / elapsed,
        diagnostics=result.diagnostics,
    )
end

# Independent runs measure error, not just the ESS proxy. Sampling time includes
# all upper warmup; preparation and the final scalar summary stay outside it.
function accuracy(device; seeds=1001:1020, kwargs...)
    compiled = fresh(device; kwargs...)
    execute(compiled, KA.get_backend(compiled.method_state.bank.locations))
    rows = NamedTuple[]
    truth = get(kwargs, :family, :gaussian) === :mixture ? 0.5 : 0.0
    for seed in seeds
        prepared = fresh(device; seed, kwargs...)
        backend = KA.get_backend(prepared.method_state.bank.locations)
        elapsed = @elapsed result = execute(prepared, backend)
        estimate = mean(first, result)
        push!(rows, (; seed, seconds=elapsed, estimate,
            squared_error=abs2(estimate - truth), count=length(result.logweights)))
    end
    return rows
end

end
