module BatchComparison

include("compare.jl")
using .SamplerComparison, Statistics, TOML, Printf
const SC = SamplerComparison
const PILOT = (nsamples=4096,scale_limits=(0.25,2.0))

"""Check the four batch likelihoods against scalar values, including a partial chunk."""
function check_values(device=identity)
    errors = Dict{String,Float64}()
    for model in SC.models()[1:4]
        rng = SC.Xoshiro(9300)
        theta = 0.1SC.randn(rng,model.dimension,37)
        expected = [SC.logtarget(x,model.data) for x in eachcol(theta)]
        data = device(merge(model.data,(;workspace=similar(model.data.X,model.observations,17))))
        values = device(zeros(37))
        SC.regression_batch!(values,device(theta),data)
        actual = Array(values)
        isapprox(actual,expected;rtol=2e-12,atol=2e-10) || error("Batch values differ: $(model.name)")
        errors[model.name] = maximum(abs,actual-expected)
    end
    return errors
end

function print_table(report; io=stdout)
    println(io,"# Scalar and batch regression targets\n")
    println(io,report["conditions"],"\n")
    println(io,"| Model | Method | Device | Mode | Seconds [range] | Weight ESS/s | Host MiB | Host allocations | Max mean error / SD | Max variance error | Accuracy |")
    println(io,"|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|:--|")
    for name in report["model_order"], key in sort!(collect(keys(report["models"][name]["runs"])))
        rows = report["models"][name]["runs"][key]
        isempty(rows) && continue
        ref = report["models"][name]["reference"]
        row = first(rows)
        lo,hi = extrema(Iterators.flatten(r["timing_seconds"] for r in rows))
        errors = [SC.moment_errors(r,ref) for r in rows]
        @printf(io,"| %s | %s | %s | %s | %.4g [%.4g, %.4g] | %.4g | %.3g | %.0f | %.3g | %.3g | %s |\n",
            name,row["method"],row["device"],row["mode"],mean(r["seconds"] for r in rows),
            lo,hi,
            mean(r["ess"] for r in rows)/mean(r["seconds"] for r in rows),
            mean(r["host_bytes"] for r in rows)/2^20,mean(r["host_allocations"] for r in rows),
            maximum(e.mean_error for e in errors),maximum(e.variance_error for e in errors),
            all(r->SC.accurate_moments(r,ref),rows) ? "pass" : "warn")
    end
    println(io,"\nTime includes fitting, pilot, scratch setup, transfers, sampling and posterior mean.")
    println(io,"Each seed uses the median of up to ",report["timing_samples"]," timed executions and diagnostics from its last same-seed execution.")
    println(io,"Host allocations exclude device allocations. CUDA pools are warm. Raw times, GC times, load and BLAS threads remain in TOML.")
    println(io,"Accuracy warns at >0.2 posterior-SD mean error or >30% marginal variance error.")
    println(io,"Julia ",report["metadata"]["julia"],", threads ",report["metadata"]["threads"],
        "; ",report["metadata"]["cpu"],". Samples: ",report["nsamples"],"; seeds: ",join(report["seeds"],", "),".")
    println(io,"\nManifest SHA-256: `",report["metadata"]["manifest_sha256"],"`.\n\n| Package | Version |\n|:--|:--|")
    for (name,version) in sort!(collect(report["metadata"]["packages"]);by=first)
        println(io,"| ",name," | ",version," |")
    end
end

function compare(; output=joinpath(@__DIR__,"batch-results.toml"), resume=false,
                  cpu=false,cuda=true,nsamples=2^18,repeats=3,timing_samples=3,capacity=8192,
                  selected=1:4,methods=SC.IMPORTANCE_METHODS)
    ispath(output) && !resume && error("Output already exists: $output")
    nsamples % 64 == 0 || error("Sample budget must divide into four rounds and sixteen proposals")
    capacity > 0 || error("Batch capacity must be positive")
    SC.BLAS.set_num_threads(1)
    SC.FFTW.set_num_threads(1)
    cases = SC.models()[collect(selected)]
    referencefile = joinpath(@__DIR__,"results-2026-09-17-comparison.toml")
    references = TOML.parsefile(referencefile)
    devices = Pair{String,Any}[]
    cpu && push!(devices,"CPU"=>identity)
    if cuda
        SC.CUDA.allowscalar(false)
        SC.CUDA.functional() || error("CUDA is not functional")
        push!(devices,"CUDA"=>SC.MLDataDevices.with_eltype(SC.MLDataDevices.CUDADevice(),nothing))
    end
    isempty(devices) && error("Select at least one device")
    meta = SC.metadata()
    cuda && (meta["gpu"] = SC.CUDA.name(SC.CUDA.device()))
    for file in ("batch.jl","batch_targets.jl","models.jl")
        meta[file*"_sha256"] = bytes2hex(SC.sha256(read(joinpath(@__DIR__,file))))
    end
    meta["reference_sha256"] = bytes2hex(SC.sha256(read(referencefile)))
    meta["cpu_affinity"] = Sys.islinux() ? only(filter(line->startswith(line,"Cpus_allowed_list:"),
        readlines("/proc/self/status"))) : "Not set by this script"
    report = Dict{String,Any}("metadata"=>meta,"nsamples"=>nsamples,"capacity"=>capacity,
        "seeds"=>collect(9301:9300+repeats),"timing_samples"=>timing_samples,
        "devices"=>first.(devices),"methods"=>collect(String.(methods)),
        "model_order"=>[m.name for m in cases],"models"=>Dict{String,Any}(),
        "conditions"=>"Provisional shared-host measurements. CPU contention affects GPU setup too. Do not replace published CPU timings.")
    if resume
        saved = TOML.parsefile(output)
        for key in ("metadata","nsamples","capacity","seeds","timing_samples","devices","methods","model_order")
            saved[key] == report[key] || error("Resume configuration differs: $key")
        end
        report = saved
    end
    save() = open(io->TOML.print(io,report),output,"w")
    for (_,device) in devices
        check_values(device)
    end
    for (model_index,model) in enumerate(cases)
        reference = references["models"][model.name]
        (reference["dimension"],reference["observations"]) ==
            (model.dimension,model.observations) || error("Reference model differs: $(model.name)")
        record = get!(report["models"],model.name) do
            Dict{String,Any}("reference"=>reference["reference"],"dimension"=>model.dimension,
                "observations"=>model.observations,"runs"=>Dict{String,Any}())
        end
        for (method_index,method) in enumerate(methods), (device_index,(label,device)) in enumerate(devices)
            for batch_capacity in (0,capacity)
                SC.BLAS.set_num_threads(label == "CPU" && batch_capacity > 0 ? 16 : 1)
                SC.run_method(method,device === identity ? nothing : device,model,900_001;
                    nsamples=4096,warmup=32,pilot=PILOT,batch_capacity)
            end
            for (index,seed) in enumerate(report["seeds"])
                # Counterbalance order across seeds, models and methods.
                order = isodd(index+model_index+method_index+device_index) ? (0,capacity) : (capacity,0)
                for batch_capacity in order
                    mode = iszero(batch_capacity) ? "scalar" : "batch"
                    rows = get!(record["runs"],"$method / $label / $mode",Dict{String,Any}[])
                    any(r->r["seed"]==seed,rows) && continue
                    SC.BLAS.set_num_threads(label == "CPU" && batch_capacity > 0 ? 16 : 1)
                    load_before = collect(Sys.loadavg())
                    row = SC.measure_run(method,device === identity ? nothing : device,model,seed;
                        nsamples,pilot=PILOT,batch_capacity,timing_samples)
                    merge!(row,Dict("method"=>String(method),"device"=>label,"mode"=>mode,
                        "blas_threads"=>SC.BLAS.get_num_threads(),"load_before"=>load_before,
                        "load_after"=>collect(Sys.loadavg()),"source"=>basename(output)))
                    push!(rows,row)
                    save()
                    @info "Batch comparison" model=model.name method label mode seed seconds=row["seconds"]
                end
            end
        end
    end
    SC.BLAS.set_num_threads(1)
    open(io->print_table(report;io),replace(output,".toml"=>".md"),"w")
    print_table(report)
    return report
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 2 && ARGS[1] == "--report"
        BatchComparison.print_table(BatchComparison.TOML.parsefile(ARGS[2]))
    else
        isempty(setdiff(ARGS,["--cpu","--cpu-only","--resume"])) || error("Usage: batch.jl [--cpu | --cpu-only] [--resume] or --report file.toml")
        BatchComparison.compare(;cpu="--cpu" in ARGS || "--cpu-only" in ARGS,
            cuda=!("--cpu-only" in ARGS),resume="--resume" in ARGS)
    end
end
