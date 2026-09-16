include("compare.jl")
using .SamplerComparison, TOML

isempty(setdiff(ARGS, ["--resume", "--report"])) || error("Usage: refresh.jl [--resume] [--report]")
output = joinpath(@__DIR__, "results-2026-09-16-refresh.toml")
references = TOML.parsefile(joinpath(@__DIR__, "accuracy-2026-09-16.toml"))
load_before = Sys.loadavg()
report = "--report" in ARGS ? TOML.parsefile(output) : SamplerComparison.compare(;
    cuda=true, resume="--resume" in ARGS, repeats=3, timing_samples=3,
    seed_start=7101, references, output,
    pilot=(nsamples=4096, scale_limits=(0.25, 2.0)),
    only_methods=(:dmpmc, :lais, :gramis, :ensemble, :stretch, :snooker),
    reuse=joinpath(@__DIR__, "results-2026-09-16-partial.toml"),
)
affinity = Sys.islinux() ? only(filter(line -> startswith(line, "Cpus_allowed_list:"),
    readlines("/proc/self/status"))) : "CPU affinity not set by this script"
if !("--report" in ARGS)
    report["host_load"] = Dict(
        "before"=>string(load_before), "after"=>string(Sys.loadavg()),
        "note"=>"Shared host, $(Sys.CPU_THREADS) logical CPUs. $affinity. Load averages:",
    )
    open(io -> TOML.print(io, report), output, "w")
end

# Keep unchanged rows from the published run. No repeat of NUTS, MH, or slice.
base_name = "results-2026-09-16-long.toml"
combined = TOML.parsefile(joinpath(@__DIR__,base_name))
for model in values(combined["models"]), rows in values(model["runs"]), row in rows
    row["source"] = base_name
end
combined["sources"] = Dict(base_name=>combined["metadata"],
    basename(output)=>report["metadata"], report["reused_source"]=>report["reused_metadata"])
for name in combined["model_order"]
    merge!(combined["models"][name]["runs"],report["models"][name]["runs"])
end
combined["method_order"] = unique(vcat(combined["method_order"],report["method_order"]))
for key in ("proposal_pilot","timing_samples","host_load")
    combined[key] = report[key]
end
combined_path = joinpath(@__DIR__,"results-2026-09-16-comparison.toml")
open(io -> TOML.print(io,combined),combined_path,"w")
open(io -> SamplerComparison.print_table(combined;io),replace(combined_path,".toml"=>".md"),"w")
SamplerComparison.print_table(combined)
