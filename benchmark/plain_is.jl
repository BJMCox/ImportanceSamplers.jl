using BenchmarkTools
using DensityInterface
using ImportanceSamplers
using Random

const SMOKE_MODE = "--smoke" in ARGS
const BENCHMARK_SEED = 0x2d861e0f73a9bc45
const SAMPLE_COUNT = SMOKE_MODE ? 256 : 10_000
const TRIAL_SAMPLES = SMOKE_MODE ? 3 : 30
const TRIAL_SECONDS = SMOKE_MODE ? 0.2 : 5.0
const PREPARATION_BATCH_SIZE = SMOKE_MODE ? 16 : 64
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

function report_trial(label, trial; work_items=nothing, divisor=1)
    return report_trial(stdout, label, trial; work_items, divisor)
end

function report_trial(io::IO, label, trial; work_items=nothing, divisor=1)
    estimate = median(trial)
    println(io, label)
    if divisor != 1
        println(io, "  raw batch median time (ns): ", estimate.time)
        println(io, "  raw batch allocations: ", estimate.allocs)
        println(io, "  raw batch allocated bytes: ", estimate.memory)
    end
    println(io, "  median time: ", BenchmarkTools.prettytime(estimate.time / divisor))
    println(io, "  allocations: ", estimate.allocs / divisor)
    println(io, "  allocated memory: ", BenchmarkTools.prettymemory(estimate.memory / divisor))
    if work_items !== nothing
        seconds = estimate.time / 1.0e9
        println(io, "  throughput: ", round(work_items / seconds; sigdigits=6), " samples/s")
    end
    return nothing
end

function prepare_batch(rng, target, algorithm, threaded)
    sink = nothing
    for _ in 1:PREPARATION_BATCH_SIZE
        sink = Base.inferencebarrier(
            prepare_sampler(rng, target, algorithm; threaded),
        )
    end
    return sink
end

function main()
    println("ImportanceSamplers plain-IS benchmark")
    println("command: ", BENCHMARK_COMMAND)
    println("Julia version: ", VERSION)
    println("default threads: ", Threads.nthreads(:default))
    println("smoke mode: ", SMOKE_MODE)
    println("sample count per execution: ", SAMPLE_COUNT)
    println("trial samples: ", TRIAL_SAMPLES)
    println("preparations per timed evaluation: ", PREPARATION_BATCH_SIZE)
    println("No timing threshold is asserted; compare trials on controlled hardware.")

    preparation_rng = Xoshiro(BENCHMARK_SEED)
    preparation = @benchmarkable prepare_batch(
        $preparation_rng,
        logtarget,
        ALGORITHM,
        false,
    )
    preparation_trial = run_trial(preparation)
    report_trial(
        "preparation per prepared sampler",
        preparation_trial;
        divisor=PREPARATION_BATCH_SIZE,
    )

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
