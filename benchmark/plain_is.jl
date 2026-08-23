using BenchmarkTools
using DensityInterface
using ImportanceSamplers
using Random

const SMOKE_MODE = "--smoke" in ARGS
const BENCHMARK_SEED = 0x2d861e0f73a9bc45
const SAMPLE_COUNT = SMOKE_MODE ? 256 : 10_000
const TRIAL_SAMPLES = SMOKE_MODE ? 3 : 30
const TRIAL_SECONDS = SMOKE_MODE ? 0.2 : 5.0
const BENCHMARK_COMMAND =
    "julia --project=benchmark benchmark/plain_is.jl" *
    (SMOKE_MODE ? " --smoke" : "")

struct GaussianProposal{T<:AbstractFloat}
    mean::T
    scale::T
end

function Random.rand(
    rng::Random.AbstractRNG,
    proposal::GaussianProposal{T},
) where {T}
    return proposal.mean + proposal.scale * randn(rng, T)
end

function DensityInterface.logdensityof(proposal::GaussianProposal, x::Real)
    standardized = (x - proposal.mean) / proposal.scale
    return -oftype(standardized, 0.5) * abs2(standardized) -
           log(proposal.scale) - oftype(standardized, 0.5 * log(2pi))
end

const PROPOSAL = GaussianProposal(0.0, 2.0)
const ALGORITHM = ImportanceSampling(PROPOSAL; nsamples=SAMPLE_COUNT)

logtarget(x) = -0.5 * abs2(x - 0.75) - 0.5 * log(2pi)

function run_trial(benchmark)
    return run(
        benchmark;
        samples=TRIAL_SAMPLES,
        seconds=TRIAL_SECONDS,
        evals=1,
    )
end

function report_trial(label, trial; work_items=nothing)
    estimate = median(trial)
    println(label)
    println("  median time: ", BenchmarkTools.prettytime(estimate.time))
    println("  allocations: ", estimate.allocs)
    println("  allocated memory: ", BenchmarkTools.prettymemory(estimate.memory))
    if work_items !== nothing
        seconds = estimate.time / 1.0e9
        println("  throughput: ", round(work_items / seconds; sigdigits=6), " samples/s")
    end
    return nothing
end

function main()
    println("ImportanceSamplers plain-IS benchmark")
    println("command: ", BENCHMARK_COMMAND)
    println("Julia version: ", VERSION)
    println("default threads: ", Threads.nthreads(:default))
    println("smoke mode: ", SMOKE_MODE)
    println("sample count per execution: ", SAMPLE_COUNT)
    println("trial samples: ", TRIAL_SAMPLES)
    println("No timing threshold is asserted; compare trials on controlled hardware.")

    preparation_rng = Xoshiro(BENCHMARK_SEED)
    preparation_sink = Ref{Any}()
    preparation = @benchmarkable $preparation_sink[] = Base.inferencebarrier(
        prepare_sampler(
            $preparation_rng,
            logtarget,
            ALGORITHM;
            threaded=false,
        ),
    )
    preparation_trial = run_trial(preparation)
    report_trial("preparation", preparation_trial)

    sampler = prepare_sampler(
        Xoshiro(BENCHMARK_SEED),
        logtarget,
        ALGORITHM;
        threaded=false,
    )
    importance_sample!(sampler) # compile and warm the prepared path
    execution = @benchmarkable importance_sample!($sampler)
    execution_trial = run_trial(execution)
    report_trial(
        "warm prepared execution",
        execution_trial;
        work_items=SAMPLE_COUNT,
    )
    return nothing
end

main()
