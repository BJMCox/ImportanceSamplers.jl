module PopulationTuning

include("compare.jl")
using .SamplerComparison, LinearAlgebra, Statistics, TOML, SHA
const SC = SamplerComparison

configuration(scale, count, rounds) = Dict("scale"=>scale,"count"=>count,"rounds"=>rounds)
parameters(p) = (scale=p["scale"],count=p["count"],rounds=p["rounds"])

eligible(rows, reference) = all(row->SC.accurate_moments(row,reference),rows)
score(rows) = exp(mean(log(row["ess"]/row["seconds"]) for row in rows))

"""
    tune(; cuda=false, output, resume=false)

Search a fixed grid on seed 2101. Check its three best eligible settings on
2102 and 2103, then freeze the best setting that passes all moment checks.
All trials use the full retained budget. No held-out seed enters this search.
Offline search costs remain separate from each timed, freshly prepared run.
"""
function tune(; cuda=false, output=joinpath(@__DIR__,"population-pilots.toml"), resume=false)
    ispath(output) && !resume && error("Output exists: $output")
    BLAS.set_num_threads(1)
    SC.FFTW.set_num_threads(1)
    device = cuda ? SC.MLDataDevices.with_eltype(SC.MLDataDevices.CUDADevice(),nothing) : nothing
    cuda && SC.CUDA.allowscalar(false)
    backend = cuda ? "CUDA" : "CPU"
    references = TOML.parsefile(joinpath(@__DIR__,"accuracy-2026-09-16.toml"))
    grid = [configuration(s,k,r) for s in (0.5,0.8,1.2), k in (16,64,256,1024), r in (4,16,64)]
    report = Dict{String,Any}("metadata"=>SC.metadata(),"backend"=>backend,
        "pilot_sha256"=>bytes2hex(sha256(read(@__FILE__))),"seeds"=>[2101,2102,2103],
        "held_out_seeds"=>[5001,5002,5003],"nsamples"=>2^18,
        "mean_error_limit"=>0.2,"variance_error_limit"=>0.3,
        "models"=>Dict{String,Any}(),"selected"=>Dict{String,Any}())
    if resume
        saved = TOML.parsefile(output)
        for key in ("metadata","backend","pilot_sha256","seeds","held_out_seeds","nsamples",
                    "mean_error_limit","variance_error_limit")
            saved[key] == report[key] || error("Pilot resume differs: $key")
        end
        report = saved
    end
    save() = open(io->TOML.print(io,report),output,"w")
    for model in SC.models(), (method,name) in ((:dmpmc,"DM-PMC"),(:lais,"LAIS-RAM"))
        label = "$name / $backend"
        record = get!(report["models"],model.name,Dict{String,Any}())
        trials = get!(record,label) do
            [Dict("configuration"=>p,"runs"=>Any[]) for p in grid[:]]
        end
        selected = get!(report["selected"],model.name,Dict{String,Any}())
        haskey(selected,label) && continue
        reference = references["models"][model.name]["reference"]
        # Compilation uses a small valid population. Selection always measures
        # the full budget, including warmup and the same per-run Laplace fit.
        SC.run_method(method,device,model,900_001;nsamples=4096,warmup=128)
        SC.measure_run(method,device,model,900_002;nsamples=4096,warmup=128)
        function measure!(trial, seed)
            any(row->row["seed"]==seed,trial["runs"]) && return
            p = parameters(trial["configuration"])
            @info "Population pilot" model=model.name sampler=label seed settings=p
            row = SC.measure_run(method,device,model,seed;nsamples=report["nsamples"],population=p)
            push!(trial["runs"],row)
            save()
        end
        for trial in trials
            measure!(trial,2101)
        end
        candidates = filter(trial->eligible(trial["runs"][1:1],reference),trials)
        sort!(candidates;by=trial->score(trial["runs"][1:1]),rev=true)
        finalists = first(candidates,min(3,length(candidates)))
        for trial in finalists, seed in (2102,2103)
            measure!(trial,seed)
        end
        valid = filter(trial->eligible(trial["runs"],reference),finalists)
        if isempty(valid)
            @warn "No eligible population setting" model=model.name sampler=label
            # A failed search is a result. Do not relax thresholds or try
            # held-out seeds to find an attractive replacement.
            record["$label selection"] = "no configuration passed all pilot moment checks"
        else
            winner = valid[argmax(score(trial["runs"]) for trial in valid)]
            selected[label] = winner["configuration"]
        end
        save()
    end
    report["search_seconds"] = sum(row["seconds"] for record in values(report["models"])
        for trials in values(record) if trials isa AbstractVector
        for trial in trials for row in trial["runs"])
    report["frozen"] = true
    save()
    return report["selected"]
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    unknown = setdiff(ARGS,["--cuda","--resume"])
    isempty(unknown) || error("Usage: tune.jl [--cuda] [--resume]")
    PopulationTuning.tune(;cuda="--cuda" in ARGS,resume="--resume" in ARGS)
end
