module LAISComparison

using ImportanceSamplers, BenchmarkTools, Distributions, ForwardDiff, LinearAlgebra,
      Optim, Random, Statistics, TOML, SHA, Pkg, Printf
import CUDA, MLDataDevices, LogDensityProblems
include("models.jl")

const DRAWS = 2^18
const PILOT_ROUNDS = 8
const DOF = 8.0

config(; family=:student_t, upper=:ram, warmup=1024, count=256, rounds=4) =
    (; family, upper, warmup, count, rounds)

function configurations(model_index; screen=false)
    static = config(family=:gaussian, upper=:static, warmup=0, count=1, rounds=1)
    rwm = config(family=:gaussian, upper=:rwm, warmup=0, count=16)
    screen && return [static, config(), config(warmup=0), config(upper=:rwm,warmup=0),
        config(family=:gaussian), config(family=:gaussian,warmup=64),
        config(family=:gaussian,upper=:rwm,warmup=0), rwm,
        config(;rwm...,rounds=64)]
    return model_index == 1 ? [static,config(),rwm,config(;rwm...,count=256),
        config(;rwm...,rounds=64)] : [static,config(),rwm,config(;rwm...,family=:student_t)]
end

function measure(f; repeats=3)
    f() # Compile the complete shape before measuring it.
    value = Ref{Any}()
    bench = @benchmarkable $value[] = $f() samples=1 evals=1 gctrial=false
    trial = run(bench; samples=repeats, seconds=300, warmup=false)
    timing = Dict("seconds"=>median(trial).time/1e9, "times"=>trial.times./1e9,
        "host_bytes"=>trial.memory, "host_allocations"=>trial.allocs)
    return value[], timing
end

function pilot(model, fit, device, seed; draws=DRAWS)
    proposal = FactorStudentT(DOF,fit.location,1.2fit.factor)
    prepared = prepare_sampler(Xoshiro(seed+100_000),logtarget,model.data,
        AMIS(proposal; rounds=PILOT_ROUNDS, round_size=draws÷PILOT_ROUNDS))
    device === nothing || (prepared = device(prepared))
    importance_sample!(prepared)
    # Copy fitted parameters, not the pilot samples.
    return current_proposal(MLDataDevices.cpu_device(),prepared)
end

function produce(model, tuned, device, seed, c; draws=DRAWS)
    # Student-t scale is not covariance. Match covariances across both families.
    factor = sqrt(DOF/(DOF-2))*tuned.scale.factor
    proposal = c.family === :gaussian ? FactorGaussian(tuned.location,factor) :
        FactorStudentT(DOF,tuned.location,tuned.scale.factor)
    algorithm = if c.upper === :static
        ImportanceSampling(proposal; nsamples=draws)
    else
        bank = ProposalBank([deepcopy(proposal) for _ in 1:c.count],ones(c.count))
        covariance = (2.38^2/model.dimension)*Symmetric(factor*factor')
        transition = c.upper === :ram ? RAM(covariance;tuning=WarmupTuning(c.warmup)) :
            RandomWalkMetropolis(covariance)
        LAIS(bank; transition, rounds=c.rounds, round_size=draws÷c.rounds)
    end
    prepared = prepare_sampler(Xoshiro(seed+200_000),logtarget,model.data,algorithm)
    device === nothing || (prepared = device(prepared))
    samples = importance_sample!(prepared)
    return (; samples, estimate=Array(mean(samples)), prepared)
end

function fixture(index)
    model = models()[index]
    fit, timing = measure(() -> laplace(model))
    reference = if index == 1
        X, y = model.data.X, model.data.y
        precision = cholesky(Symmetric(X'X+I/4))
        covariance = precision \ Matrix{Float64}(I,model.dimension,model.dimension)
        (; location=precision\(X'y), covariance)
    else
        saved = TOML.parsefile(joinpath(@__DIR__,"accuracy-2026-09-16.toml"))
        ref = saved["models"][model.name]["reference"]
        (; location=ref["mean"], covariance=Diagonal(ref["variance"]))
    end
    return (; model, fit, timing, reference)
end

function summarize(value, data, tuned, c)
    samples = value.samples
    weights = Array(normalized_weights(samples))
    variance = Array(var(samples;corrected=false))
    sd = sqrt.(diag(data.reference.covariance))
    mean_error = maximum(abs.((value.estimate-data.reference.location)./sd))
    variance_error = maximum(abs.(variance./abs2.(sd).-1))
    row = Dict{String,Any}("ess"=>inv(sum(abs2,weights)), "max_weight"=>maximum(weights),
        "draws"=>length(samples), "mean_error"=>mean_error, "variance_error"=>variance_error,
        "accurate"=>mean_error<=0.2 && variance_error<=0.3,
        "estimate"=>value.estimate, "variance"=>variance,
        "round_ess"=>hasproperty(samples.diagnostics,:round_ess) ?
            Array(samples.diagnostics.round_ess) : Float64[])
    c.upper === :static && return row
    counts = samples.diagnostics.transition
    row["transition"] = Dict(String(k)=>v for (k,v) in pairs(counts))
    row["target_evaluations"] = samples.diagnostics.target_evaluations
    row["acceptance"] = counts.accepted/(counts.warmup_proposals+counts.production_proposals)
    bank = current_proposal(MLDataDevices.cpu_device(),value.prepared)
    factor = sqrt(DOF/(DOF-2))*tuned.scale.factor
    coordinates = LowerTriangular(factor) \ (stack(q.location for q in bank.proposals) .- tuned.location)
    row["centre_variance"] = mean(var(coordinates;dims=2,corrected=false))
    row["centre_mean_square"] = mean(abs2,coordinates)
    return row
end

function metadata(device)
    root = normpath(joinpath(@__DIR__,"../.."))
    sources = ["src/methods/amis.jl", "src/methods/lais.jl", "src/mcmc_transitions.jl",
               "benchmark/comparison/models.jl", "benchmark/comparison/lais.jl"]
    return Dict("julia"=>string(VERSION), "threads"=>Threads.nthreads(),
        "blas_threads"=>BLAS.get_num_threads(), "cpu"=>Sys.cpu_info()[1].model,
        "device"=>device === nothing ? "CPU" : CUDA.name(CUDA.device()),
        "revision"=>readchomp(`git -C $root rev-parse HEAD`),
        "source_sha256"=>Dict(path=>bytes2hex(sha256(read(joinpath(root,path)))) for path in sources),
        "manifest_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"Manifest.toml")))),
        "versions"=>Dict(info.name=>string(info.version) for info in values(Pkg.dependencies())
                         if info.version !== nothing))
end

function study(output; device=nothing, model_index=1, seeds=[12102,12103],
               configs=configurations(model_index), repeats=3,
               progress=println, cancelled=(() -> false))
    ispath(output) && error("Result already exists: $output")
    BLAS.set_num_threads(1)
    data = fixture(model_index)
    report = Dict{String,Any}("metadata"=>metadata(device), "model"=>data.model.name,
        "draws"=>DRAWS, "pilot_draws"=>DRAWS, "pilot_rounds"=>PILOT_ROUNDS,
        "setup_timing"=>data.timing, "pilots"=>Dict{String,Any}[], "runs"=>Dict{String,Any}[],
        "reference_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"accuracy-2026-09-16.toml")))),
        "load_before"=>collect(Sys.loadavg()))
    save() = open(io -> TOML.print(io,report),output,"w")
    save()
    for seed in seeds
        cancelled() && break
        progress("$(data.model.name) AMIS pilot seed=$seed")
        tuned, pilot_timing = measure(() -> pilot(data.model,data.fit,device,seed);repeats)
        push!(report["pilots"],Dict("seed"=>seed,"timing"=>pilot_timing))
        save()
        for c in configs
            cancelled() && break
            progress("LAIS seed=$seed $c")
            value, timing = measure(() -> produce(data.model,tuned,device,seed,c);repeats)
            row = summarize(value,data,tuned,c)
            total = data.timing["seconds"]+pilot_timing["seconds"]+timing["seconds"]
            merge!(row,Dict("seed"=>seed,
                "configuration"=>Dict(String(k)=>v isa Symbol ? String(v) : v for (k,v) in pairs(c)),
                "main_timing"=>timing, "total_seconds"=>total, "ess_per_second"=>row["ess"]/total))
            push!(report["runs"],row)
            save()
        end
    end
    report["load_after"] = collect(Sys.loadavg())
    save()
    return report
end

function print_table(report; io=stdout)
    println(io,"### ",report["model"],"\n\n| Method | Family | Proposals | Rounds | Warmup | ESS/s | ESS/s range | Max mean error / SD | Max variance error |\n|:--|:--|--:|--:|--:|--:|--:|--:|--:|")
    for c in unique(row["configuration"] for row in report["runs"])
        rows = filter(row->row["configuration"]==c,report["runs"])
        method = c["upper"] == "static" ? "IS" : "LAIS-$(uppercase(c["upper"]))"
        score = sum(r["ess"] for r in rows)/sum(r["total_seconds"] for r in rows)
        lo, hi = extrema(r["ess_per_second"] for r in rows)
        @printf(io,"| %s | %s | %d | %d | %d | %.2g | %.3g–%.3g | %.3g | %.3g |\n",
            method,c["family"],c["count"],c["rounds"],c["warmup"],score,lo,hi,
            maximum(r["mean_error"] for r in rows),maximum(r["variance_error"] for r in rows))
    end
    println(io,"\nMetadata and package versions:\n\n```toml")
    TOML.print(io,report["metadata"])
    println(io,"```\n")
end

function print_readme(reports; io=stdout)
    for (upper,label) in (("static","IS, AMIS-fitted‖"),("rwm","LAIS-RWM, AMIS-fitted‖"))
        cells = map(("linear","logistic","poisson","robust","eight_schools")) do name
            index = findfirst(r->r["model"]==name,reports)
            index === nothing && return "—"
            rows = filter(reports[index]["runs"]) do row
                c = row["configuration"]
                c["family"]=="gaussian" && c["upper"]==upper &&
                    c["count"]==(upper=="static" ? 1 : 16) &&
                    c["rounds"]==(upper=="static" ? 1 : 4) && c["warmup"]==0
            end
            isempty(rows) && return "—"
            return @sprintf("%.2g",sum(r["ess"] for r in rows)/sum(r["total_seconds"] for r in rows))
        end
        println(io,"| ",label," | ",join(cells," | ")," |")
    end
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    L = LAISComparison
    reports = if !isempty(ARGS) && first(ARGS)=="--report"
        length(ARGS)>1 || error("Pass saved TOML files after --report")
        L.TOML.parsefile.(ARGS[2:end])
    else
        isempty(setdiff(ARGS,["--cuda","--screen"])) || error("Use [--cuda] [--screen] or --report files...")
        gpu, screen = "--cuda" in ARGS, "--screen" in ARGS
        device = gpu ? L.MLDataDevices.with_eltype(L.MLDataDevices.CUDADevice(),nothing) : nothing
        gpu && L.CUDA.allowscalar(false)
        map(screen ? [1] : [1,4]) do index
            model = L.models()[index]
            suffix = screen ? "screen" : "validation"
            output = joinpath(@__DIR__,"lais-$(model.name)-$(gpu ? "cuda" : "cpu")-$suffix.toml")
            seeds = screen ? [12101] : index==1 ? [12102,12103] : [12201,12202]
            L.study(output;device,model_index=index,seeds,configs=L.configurations(index;screen))
        end
    end
    foreach(L.print_table,reports)
    println("\nREADME rows (same device, full pilot cost included):\n")
    L.print_readme(reports)
end
