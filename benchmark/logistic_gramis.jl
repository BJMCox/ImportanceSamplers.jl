module LogisticGRAMISBenchmark

using BenchmarkTools, CUDA, ImportanceSamplers, Profile, Random, Reactant

module Example
include(joinpath(@__DIR__, "..", "examples", "logistic_regression.jl"))
end

function gradient!(g, x, p)
    for j in eachindex(x)
        g[j] = -x[j] / p.prior_scale^2
    end
    for i in eachindex(p.outcomes)
        eta = zero(eltype(x))
        for j in eachindex(x)
            eta += p.predictors[i, j] * x[j]
        end
        residual = p.outcomes[i] - Example.logistic(eta)
        for j in eachindex(x)
            g[j] += p.predictors[i, j] * residual
        end
    end
    return nothing
end

function run!(sampler)
    result = importance_sample!(sampler)
    if result.logweights isa Reactant.AnyConcreteRArray
        Reactant.synchronize((result.samples, result.logweights))
    else
        CUDA.synchronize()
    end
    return result
end

function device_profile(sampler)
    trace = CUDA.@profile run!(sampler)
    kernels = Dict{String,Tuple{Int,Float64}}()
    for i in eachindex(trace.device.name)
        name = first(split(trace.device.name[i], '('))
        count, seconds = get(kernels, name, (0, 0.0))
        kernels[name] = (count + 1, seconds + trace.device.stop[i] - trace.device.start[i])
    end
    host = Dict{String,Int}()
    for name in trace.host.name
        host[name] = get(host, name, 0) + 1
    end
    return (; kernels=sort!([(; name, calls=v[1], ms=1000v[2]) for (name, v) in kernels];
                            by=x -> -x.ms), host)
end

function host_profile(sampler, repetitions)
    Profile.init(n=2_000_000, delay=0.001)
    Profile.clear()
    Profile.@profile for _ in 1:repetitions
        run!(sampler)
    end
    data, frames = Profile.retrieve()
    compiler_ips = Set(ip for (ip, stack) in frames if any(stack) do frame
        occursin("/Reactant/", String(frame.file)) && occursin("compile", String(frame.func))
    end)
    callers = Dict{String,Int}()
    for ip in data, frame in get(frames, ip, [])
        path = String(frame.file)
        occursin("ImportanceSamplers", path) || continue
        key = string(frame.func, " @ ", basename(path), ':', frame.line)
        callers[key] = get(callers, key, 0) + 1
    end
    return (; repetitions, compiler_frame_hits=count(in(compiler_ips), data),
            callers=first(sort!(collect(callers); by=last, rev=true), min(12, length(callers))))
end

"""
    measure(device; observations=1000, round_size=65536, proposals=8, trials=20)

Profile the example's Float64 logistic posterior with a supplied analytic gradient.
Pass an explicit CUDA or GPU-backed Reactant device preserving Float64. Predictors
and outcomes transfer during preparation, not during sample or gradient evaluation.
Warm timings reuse adaptation. They do not measure fresh-run estimator accuracy.
Device and host profiles run separately from BenchmarkTools timing.
"""
function measure(device; observations=1000, round_size=65536, proposals=8, trials=20)
    CUDA.allowscalar(false)
    data = Example.simulate_logistic_data(Xoshiro(0x4c4f474953544943),
                                        Example.TRUE_COEFFICIENTS; observations)
    context = (; data..., prior_scale=1.5)
    rng = Xoshiro(0x4752414d49534c52)
    bank = ProposalBank([SphericalGaussian(0.5randn(rng, 3), 1.5) for _ in 1:proposals])
    algorithm = FirstOrderGRAMIS(bank; rounds=4, round_size, repulsion_strength=0.0)
    preparation = @timed device(prepare_sampler(rng,
        LogTarget(Example.logistic_logtarget; grad=gradient!), context, algorithm))
    sampler = preparation.value
    first_call = @timed run!(sampler)
    first_result = first_call.value
    first_summary = (; mean=Array(first_result.samples) * Array(normalized_weights(first_result)),
                     lognormalizer=lognormalizer(first_result))
    for _ in 1:3
        run!(sampler)
    end
    trial = @benchmark run!($sampler) samples=trials evals=1 seconds=120
    estimate = median(trial)
    return (; observations, round_size, proposals, precision=Float64,
        preparation_ms=1000preparation.time, preparation_host_bytes=preparation.bytes,
        first_call_ms=1000first_call.time, first_summary,
        warm=(; median_ms=estimate.time/1e6, min_ms=minimum(trial).time/1e6,
                max_ms=maximum(trial).time/1e6, host_bytes=estimate.memory,
                host_allocations=estimate.allocs, trials=length(trial)),
        device_profile=device_profile(sampler), host_profile=host_profile(sampler, 20),
        factor_execution=first_result.diagnostics.factor_execution_policy,
        target_evaluations=first_result.diagnostics.target_evaluations,
        gradient_evaluations=first_result.diagnostics.gradient_evaluations,
        transfers=first_result.diagnostics.transfers)
end

end
