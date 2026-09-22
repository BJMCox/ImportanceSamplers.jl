module BatchComparison

include("compare.jl")
using .SamplerComparison, BenchmarkTools, Statistics, TOML, Printf
const SC = SamplerComparison
const PILOT = (nsamples=4096,scale_limits=(0.25,2.0))

function measure_pair(method,device,model,seed; nsamples,capacity,timing_samples,batch_first)
    gpu = device !== identity
    sampling_device = gpu ? device : nothing
    captured = [Ref{Any}(nothing),Ref{Any}(nothing)]
    capacities = (0,capacity)
    blas_threads = (1,gpu ? 1 : 16)
    benches = map(eachindex(capacities)) do mode
        result,batch_capacity = captured[mode],capacities[mode]
        @benchmarkable begin
            $result[] = SC.run_method($method,$sampling_device,$model,$seed;
                nsamples=$nsamples,pilot=PILOT,batch_capacity=$batch_capacity)
            $gpu && SC.CUDA.synchronize()
        end samples=1 evals=1 gctrial=false
    end
    for mode in (batch_first ? (2,1) : (1,2))
        SC.BLAS.set_num_threads(blas_threads[mode])
        run(benches[mode];samples=1) # Warm the complete workload and measurement wrapper.
    end
    trials = [BenchmarkTools.Trial(bench.params) for bench in benches]
    executions = Dict{String,Any}[]
    for block in 1:timing_samples÷2
        order = xor(batch_first,iseven(block)) ? (2,1,1,2) : (1,2,2,1)
        for (position,mode) in enumerate(order)
            captured[mode][] = nothing
            GC.gc(true)
            gpu && SC.CUDA.synchronize()
            SC.BLAS.set_num_threads(blas_threads[mode])
            started,load_before = time(),collect(Sys.loadavg())
            trial = run(benches[mode];samples=1,warmup=false)
            push!(trials[mode],only(trial.times),only(trial.gctimes),trial.memory,trial.allocs)
            push!(executions,Dict("seed"=>seed,"block"=>block,"position"=>position,
                "mode"=>mode==1 ? "scalar" : "batch","seconds"=>only(trial.times)/1e9,
                "gc_seconds"=>only(trial.gctimes)/1e9,"host_bytes"=>trial.memory,
                "host_allocations"=>trial.allocs,"started_unix"=>started,
                "load_before"=>load_before,"load_after"=>collect(Sys.loadavg()),
                "blas_threads"=>blas_threads[mode]))
        end
    end
    # Keep diagnostics outside both balanced blocks, using the last same-seed result.
    rows = [SC.summarize_run(trials[mode],captured[mode][],seed) for mode in 1:2]
    rows[1]["draws"] == rows[2]["draws"] == nsamples || error("Scalar/batch sample budgets differ")
    return rows,executions
end

"""Check the four batch likelihoods against scalar values, including a partial chunk."""
function check_values(device=identity)
    errors = Dict{String,Float64}()
    for model in SC.models()[1:4]
        rng = SC.Xoshiro(9300)
        theta = 0.1SC.randn(rng,model.dimension,37)
        expected = [SC.logtarget(x,model.data) for x in eachcol(theta)]
        workspace = device === identity ? similar(model.data.X,model.observations,17) : nothing
        data = device(merge(model.data,(;workspace,batch_capacity=17)))
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
    if haskey(report,"timing_protocol")
        println(io,"Each seed interleaves ",report["timing_samples"]," executions per mode in balanced ABBA/BAAB blocks. Diagnostics follow both modes. Full GC precedes each execution outside timing.")
        println(io,"\n| Model | Method | Device | Batch/scalar paired time ratio: median [range] |\n|:--|:--|:--|--:|")
        for name in report["model_order"], method in report["methods"], device in report["devices"]
            entries = filter(r->r["model"]==name && r["method"]==method && r["device"]==device,report["executions"])
            ratios = map(unique((r["seed"],r["block"]) for r in entries)) do key
                block = filter(r->(r["seed"],r["block"])==key,entries)
                mean(r["seconds"] for r in block if r["mode"]=="batch") /
                    mean(r["seconds"] for r in block if r["mode"]=="scalar")
            end
            isempty(ratios) || @printf(io,"| %s | %s | %s | %.3g [%.3g, %.3g] |\n",name,method,device,median(ratios),extrema(ratios)...)
        end
        println(io,"\nA ratio above one means batching took longer. Interleaving reduces drift but cannot eliminate shared-host contention.")
    else
        println(io,"Each seed uses the median of up to ",report["timing_samples"]," timed executions and diagnostics from its last same-seed execution.")
    end
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
                  cpu=false,cuda=true,nsamples=2^18,repeats=3,timing_samples=4,capacity=8192,
                  selected=1:4,methods=SC.IMPORTANCE_METHODS)
    ispath(output) && !resume && error("Output already exists: $output")
    nsamples % 64 == 0 || error("Sample budget must divide into four rounds and sixteen proposals")
    capacity > 0 || error("Batch capacity must be positive")
    timing_samples >= 4 && timing_samples % 4 == 0 || error("ABBA/BAAB needs timing_samples divisible by four")
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
    cuda && (meta["gpu_uuid"] = string(SC.CUDA.uuid(SC.CUDA.device())))
    for file in ("batch.jl","batch_targets.jl","models.jl")
        meta[file*"_sha256"] = bytes2hex(SC.sha256(read(joinpath(@__DIR__,file))))
    end
    meta["reference_sha256"] = bytes2hex(SC.sha256(read(referencefile)))
    meta["cpu_affinity"] = Sys.islinux() ? only(filter(line->startswith(line,"Cpus_allowed_list:"),
        readlines("/proc/self/status"))) : "Not set by this script"
    report = Dict{String,Any}("metadata"=>meta,"nsamples"=>nsamples,"capacity"=>capacity,
        "seeds"=>collect(9301:9300+repeats),"timing_samples"=>timing_samples,
        "timing_protocol"=>"ABBA/BAAB single executions","executions"=>Dict{String,Any}[],
        "devices"=>first.(devices),"methods"=>collect(String.(methods)),
        "model_order"=>[m.name for m in cases],"models"=>Dict{String,Any}(),
        "conditions"=>"Provisional shared-host measurements. CPU contention affects GPU setup too. Do not replace published CPU timings.")
    if resume
        saved = TOML.parsefile(output)
        for key in ("metadata","nsamples","capacity","seeds","timing_samples","timing_protocol","devices","methods","model_order")
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
            for (index,seed) in enumerate(report["seeds"])
                modes = ("scalar","batch")
                stored = [get!(record["runs"],"$method / $label / $mode",Dict{String,Any}[]) for mode in modes]
                present = [any(r->r["seed"]==seed,rows) for rows in stored]
                if all(present)
                    entries = filter(r->r["model"]==model.name && r["method"]==String(method) &&
                        r["device"]==label && r["seed"]==seed,report["executions"])
                    length(entries)==2timing_samples || error("Saved pair lacks raw executions")
                    for (mode,destination) in zip(modes,stored)
                        saved_row = only(filter(r->r["seed"]==seed,destination))
                        saved_row["timing_seconds"] == [r["seconds"] for r in entries if r["mode"]==mode] ||
                            error("Saved pair timing differs from its raw executions")
                    end
                    continue
                end
                any(present) && error("Cannot resume an incomplete scalar/batch pair")
                rows,executions = measure_pair(method,device,model,seed;nsamples,capacity,timing_samples,
                    batch_first=isodd(index+model_index+method_index+device_index))
                for (mode,row,destination) in zip(modes,rows,stored)
                    merge!(row,Dict("method"=>String(method),"device"=>label,"mode"=>mode,
                        "blas_threads"=>label=="CPU" && mode=="batch" ? 16 : 1,
                        "source"=>basename(output)))
                    push!(destination,row)
                end
                for entry in executions
                    merge!(entry,Dict("model"=>model.name,"method"=>String(method),"device"=>label))
                end
                append!(report["executions"],executions)
                save()
                @info "Paired batch comparison" model=model.name method label seed
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
