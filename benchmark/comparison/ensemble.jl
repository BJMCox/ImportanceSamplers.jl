module EnsembleComparison

include("compare.jl")
using .SamplerComparison, Statistics, TOML
const SC = SamplerComparison
const LABELS = (ensemble="EnsembleMCMC DE / CPU", stretch="EnsembleMCMC Stretch / CPU",
    snooker="EnsembleMCMC snooker / CPU", gaussian="EnsembleMCMC Gaussian replacement / CPU")

"""
    screen(; output, repeats=3, sweeps=4096)

Compare model-specific widths on seeds 8301 onwards. Keep DE's reviewed default
scaling. Rank eligible settings by posterior-mean squared error times elapsed
time, not the noisiest minimum ESS estimate. Save every trial and the references.
This offline search is separate from the timed fitting and warmup in every run.
"""
function screen(; output=joinpath(@__DIR__,"ensemble-screen.toml"), repeats=3, sweeps=4096)
    ispath(output) && error("Screen output already exists: $output")
    SC.BLAS.set_num_threads(1)
    SC.FFTW.set_num_threads(1)
    archive = TOML.parsefile(joinpath(@__DIR__,"accuracy-2026-09-16.toml"))
    report = Dict{String,Any}("metadata"=>SC.metadata(), "seeds"=>collect(8301:8300+repeats),
        "sweeps"=>sweeps, "models"=>Dict{String,Any}(), "settings"=>Dict{String,Any}(),
        "script_sha256"=>bytes2hex(SC.sha256(read(@__FILE__))),
        "selection"=>"minimum mean(seconds * standardized mean squared error), subject to moment checks")
    save() = open(io->TOML.print(io,report),output,"w")
    for model in SC.models()
        reference = haskey(archive["models"],model.name) ? archive["models"][model.name]["reference"] : SC.reference(model; nchains=8)
        record = Dict{String,Any}("reference"=>reference, "trials"=>Dict{String,Any}())
        report["models"][model.name] = record
        settings = report["settings"][model.name] = Dict{String,Any}()
        for method in keys(LABELS)
            label = LABELS[method]
            base = SC.ensemble_options(Dict(),model,label,method; sweeps)
            candidates = method === :ensemble ? [base] : method === :gaussian ?
                [merge(base,(;shrinkage)) for shrinkage in (0.5,0.0,1.0)] :
                [merge(base,(;scale)) for scale in (method === :stretch ?
                    unique([base.scale,1+2.151/sqrt(model.dimension),1.5]) : [1.7,1.2,0.85])]
            trials = record["trials"][label] = Dict{String,Any}[]
            SC.run_method(method,nothing,model,900_001;
                ensemble=merge(base,(;sweeps=32,warmup=32)))
            for options in candidates
                rows = [SC.measure_run(method,nothing,model,seed; ensemble=options)
                    for seed in report["seeds"]]
                score = mean(row["seconds"] * mean(abs2,
                    (row["estimate"] .- reference["mean"]) ./ sqrt.(reference["variance"])) for row in rows)
                push!(trials,Dict("options"=>Dict(string(k)=>v for (k,v) in pairs(options)),
                    "runs"=>rows, "score"=>score,
                    "eligible"=>all(row->SC.accurate_moments(row,reference),rows)))
                save()
            end
            eligible = findall(trial->trial["eligible"],trials)
            # Never silently select a failed candidate. Retain the baseline and
            # record that none passed separately from held-out diagnostics.
            chosen = isempty(eligible) ? 1 : eligible[argmin([trials[i]["score"] for i in eligible])]
            settings[label] = copy(trials[chosen]["options"])
            delete!(settings[label],"sweeps")
            record["trials"][label][chosen]["selected"] = true
            save()
            @info "Ensemble screen" model=model.name method options=settings[label]
        end
    end
    return report
end

"""Select one eligible move per model using only the independent screening runs."""
function selected_moves(pilot)
    result = Dict{String,String}()
    for (name,record) in pilot["models"]
        candidates = Pair{String,Float64}[]
        for (label,trials) in record["trials"], trial in trials
            get(trial,"selected",false) && trial["eligible"] || continue
            rows = trial["runs"]
            isempty(rows) && continue
            rate = mean(r["ess"] for r in rows)/mean(r["seconds"] for r in rows)
            isfinite(rate) && rate > 0 && push!(candidates,label=>rate)
        end
        result[name] = isempty(candidates) ? "" : first(candidates[argmax(last.(candidates))])
    end
    return result
end

function screen_provenance(screenfile, pilot)
    failed = ["$name / $label" for (name,record) in pilot["models"]
        for (label,trials) in record["trials"] if all(!t["eligible"] for t in trials)]
    return Dict("source"=>basename(screenfile), "sha256"=>bytes2hex(SC.sha256(read(screenfile))),
        "script_sha256"=>get(pilot,"script_sha256","unrecorded"), "seeds"=>pilot["seeds"],
        "selection"=>pilot["selection"], "no_eligible_candidate"=>failed,
        "move_selection"=>"maximum mean(ESS) / mean(seconds) among eligible, selected screening settings")
end

function compare(; screenfile=joinpath(@__DIR__,"ensemble-screen.toml"),
                   only_methods=keys(LABELS), selected=eachindex(SC.models()), kwargs...)
    pilot = TOML.parsefile(screenfile)
    for model in SC.models()[collect(selected)], method in only_methods
        label = LABELS[method]
        trials = get(get(pilot["models"],model.name,Dict()),"trials",Dict())
        haskey(get(pilot["settings"],model.name,Dict()),label) &&
            any(t->get(t,"selected",false) && !isempty(t["runs"]),get(trials,label,())) ||
            error("Missing screening result for $(model.name) / $label. Run a new screen or restrict only_methods.")
    end
    return SC.compare(; only_methods, selected, ensemble_settings=pilot["settings"],
        configuration_search=screen_provenance(screenfile,pilot), references=pilot, seed_start=8401,
        output=joinpath(@__DIR__,"ensemble-results.toml"), kwargs...)
end

"""Rebuild the published comparison from saved measurements without sampling."""
function report(; ensemblefile=joinpath(@__DIR__,"ensemble-results-2026-09-17.toml"),
                  signalfile=joinpath(@__DIR__,"signal-background-results-2026-09-17.toml"),
                  screenfile=joinpath(@__DIR__,"ensemble-screen-2026-09-17.toml"),
                  output=joinpath(@__DIR__,"results-2026-09-22-comparison.toml"))
    combined = TOML.parsefile(joinpath(@__DIR__,"results-2026-09-16-comparison.toml"))
    for path in (signalfile,ensemblefile)
        fresh = TOML.parsefile(path)
        metadata = copy(fresh["metadata"])
        metadata["file_sha256"] = bytes2hex(SC.sha256(read(path)))
        combined["sources"][basename(path)] = metadata
        for name in fresh["model_order"]
            record = deepcopy(fresh["models"][name])
            for rows in values(record["runs"]), row in rows
                row["source"] = basename(path)
            end
            if haskey(combined["models"],name)
                combined["models"][name]["reference"] == record["reference"] ||
                    error("Reference changed for $name")
                merge!(combined["models"][name]["runs"],record["runs"])
            else
                combined["models"][name] = record
                push!(combined["model_order"],name)
            end
        end
        combined["method_order"] = path == signalfile ? fresh["method_order"] :
            unique(vcat(combined["method_order"],fresh["method_order"]))
        # Keep the CUDA-capable run's environment. Raw sources retain all older
        # versions, seeds, and timing protocols rather than relabelling them.
        path == signalfile && (combined["metadata"] = fresh["metadata"])
    end
    screen = TOML.parsefile(screenfile)
    combined["ensemble_settings"] = screen["settings"]
    combined["ensemble_sweeps"] = 2^14
    combined["configuration_search"] = screen_provenance(screenfile,screen)
    combined["ensemble_selection"] = selected_moves(screen)
    # The old load summary belongs only to its archived measurements.
    delete!(combined,"host_load")
    open(io->TOML.print(io,combined),output,"w")
    open(io->SC.print_table(combined;io),replace(output,".toml"=>".md"),"w")
    return combined
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    ARGS == ["--screen"] ? EnsembleComparison.screen() :
        ARGS == ["--report"] ? EnsembleComparison.SC.print_table(EnsembleComparison.report()) :
        isempty(ARGS) ? EnsembleComparison.compare() :
        error("Usage: ensemble.jl [--screen | --report]")
end
