module SamplerComparison

using BenchmarkTools, Distributions, ForwardDiff, LinearAlgebra, Optim, Random,
      Random123, Statistics, Printf, Pkg, TOML, SHA
import AbstractMCMC, AdvancedHMC, AdvancedMH, CUDA, EnsembleMCMC, ImportanceSamplers,
       LogDensityProblems, MCMCDiagnosticTools, MLDataDevices, SliceSampling
using FFTW

include("models.jl")
const IS = ImportanceSamplers
const HMC = AdvancedHMC

function nuts(target, seed, draws, warmup; acceptance=0.8)
    rng = Xoshiro(seed)
    d = LogDensityProblems.dimension(target)
    metric = HMC.DiagEuclideanMetric(d)
    h = HMC.Hamiltonian(metric,
        x -> LogDensityProblems.logdensity(target, x),
        x -> LogDensityProblems.logdensity_and_gradient(target, x))
    initial = 0.1randn(rng, d)
    integrator = HMC.Leapfrog(HMC.find_good_stepsize(rng, h, initial))
    kernel = HMC.HMCKernel(HMC.Trajectory{HMC.MultinomialTS}(integrator, HMC.GeneralisedNoUTurn()))
    adaptor = HMC.StanHMCAdaptor(HMC.MassMatrixAdaptor(metric), HMC.StepSizeAdaptor(acceptance, integrator))
    samples, stats = HMC.sample(rng, h, kernel, initial, draws+warmup, adaptor, warmup;
                               drop_warmup=true, verbose=false, progress=false)
    return (; samples=stack(samples), divergences=count(s -> s.numerical_error, stats))
end

function chain(method, target, seed, draws, warmup; acceptance=0.8)
    method === :nuts && return nuts(target, seed, draws, warmup; acceptance)
    rng = Xoshiro(seed)
    d = LogDensityProblems.dimension(target)
    sampler = method === :mh ? AdvancedMH.RWMH(MvNormal(zeros(d), (2.38^2/d)*I)) :
              SliceSampling.RandPermGibbs(SliceSampling.SliceSteppingOut(2.0))
    samples = AbstractMCMC.sample(rng, AbstractMCMC.LogDensityModel(target), sampler, draws;
                                  initial_params=0.1randn(rng,d), discard_initial=warmup,
                                  progress=false)
    return (; samples=stack(s.params for s in samples), divergences=0)
end

function chains(method, target, seed, draws, warmup, nchains; acceptance=0.8)
    results = Vector{NamedTuple{(:samples,:divergences),Tuple{Matrix{Float64},Int}}}(undef,nchains)
    Threads.@threads for j in 1:nchains
        results[j] = chain(method, target, seed+10_000j, draws, warmup; acceptance)
    end
    # Axes are draw, independent chain, coefficient. Ensemble walkers never
    # enter this array: they are dependent and do not count as separate chains.
    values = Array{Float64}(undef, draws, nchains, LogDensityProblems.dimension(target))
    for j in 1:nchains
        values[:,j,:] .= transpose(target.fit.location .+ target.fit.factor*results[j].samples)
    end
    return (; values, divergences=sum(r.divergences for r in results))
end

function run_method(method, device, model, seed; nsamples=65_536, warmup=1024,
                    nchains=Threads.nthreads(:default))
    fit = laplace(model)
    rng = Xoshiro(seed)
    if method in (:is, :amis)
        proposal = IS.FactorStudentT(8.0, fit.location, 1.2fit.factor)
        algorithm = method === :is ? IS.ImportanceSampling(proposal; nsamples) :
                    IS.AMIS(proposal; rounds=4, round_size=nsamples÷4)
        prepared = IS.prepare_sampler(rng, logtarget, model.data, algorithm)
        device === nothing || (prepared = device(prepared))
        result = IS.importance_sample!(prepared)
        # This includes the device reduction, final transfer, and its sync.
        estimate = Array(mean(result))
        return (; estimate, samples=result, divergences=0, draws=length(result))
    end
    target = WhitenedTarget(model.data, fit)
    if method === :ensemble
        nwalkers = 4model.dimension
        state = EnsembleMCMC.initialize(Philox4x((UInt64(seed),UInt64(1))),
            x -> LogDensityProblems.logdensity(target,x), randn(rng,model.dimension,nwalkers);
            move=EnsembleMCMC.DEMove(), executor=EnsembleMCMC.ThreadedExecutor())
        EnsembleMCMC.step!(state,256)
        result = EnsembleMCMC.sample!(state,cld(nsamples,nwalkers))
        estimate = fit.location + fit.factor*vec(mean(result.positions; dims=(2,3)))
        return (; estimate, samples=nothing, divergences=0, draws=length(result.positions)÷model.dimension)
    end
    result = chains(method,target,seed,cld(nsamples,nchains),warmup,nchains)
    return (; estimate=vec(mean(result.values; dims=(1,2))), samples=result.values,
            result.divergences, draws=size(result.values,1)*nchains)
end

# Diagnostics run after the timed call. Weight ESS measures weight concentration,
# not autocorrelation or accuracy. Dependent ensemble walkers have no chain ESS here.
diagnostics(::Nothing) = Dict{String,Any}()
diagnostics(samples::IS.WeightedSamples) =
    Dict("ess"=>inv(sum(abs2,IS.normalized_weights(samples))))
function diagnostics(samples::AbstractArray)
    result = MCMCDiagnosticTools.ess_rhat(samples;
        autocov_method=MCMCDiagnosticTools.FFTAutocovMethod())
    return Dict("ess"=>minimum(result.ess), "max_rhat"=>maximum(result.rhat))
end

function reference(model; draws=8192, nchains=Threads.nthreads(:default))
    if model.name == "linear"
        X,y = model.data.X,model.data.y
        precision = cholesky(Symmetric(X'X + I/4))
        variance = diag(precision \ Matrix{Float64}(I,model.dimension,model.dimension))
        return Dict("mean"=>precision\(X'y), "variance"=>variance,
                    "mcse"=>zeros(model.dimension), "kind"=>"analytic Gaussian posterior")
    end
    target = WhitenedTarget(model.data,laplace(model))
    result = chains(:nuts,target,730_001,draws,2048,nchains; acceptance=0.95)
    diagnostic = MCMCDiagnosticTools.ess_rhat(result.values;
        autocov_method=MCMCDiagnosticTools.FFTAutocovMethod())
    mcse = MCMCDiagnosticTools.mcse(result.values;
        autocov_method=MCMCDiagnosticTools.FFTAutocovMethod())
    maximum(diagnostic.rhat) < 1.01 || error("Reference Rhat failed for $(model.name)")
    result.divergences == 0 || error("Reference NUTS diverged for $(model.name)")
    return Dict("mean"=>vec(mean(result.values; dims=(1,2))),
                "variance"=>vec(var(reshape(result.values,:,model.dimension); dims=1)),
                "mcse"=>vec(mcse), "kind"=>"independent long NUTS run",
                "chain_means"=>[vec(mean(view(result.values,:,j,:);dims=1)) for j in 1:nchains],
                "min_bulk_ess"=>minimum(diagnostic.ess),
                "max_rhat"=>maximum(diagnostic.rhat), "draws_per_chain"=>draws,
                "chains"=>nchains)
end

function metadata()
    root = normpath(joinpath(@__DIR__,"../.."))
    packages = Dict(info.name=>string(info.version) for info in values(Pkg.dependencies())
                    if info.is_direct_dep && info.version !== nothing)
    return Dict("julia"=>string(VERSION), "cpu"=>Sys.cpu_info()[1].model,
        "threads"=>Threads.nthreads(:default), "blas_threads"=>BLAS.get_num_threads(),
        "word_size"=>Sys.WORD_SIZE, "platform"=>Sys.MACHINE,
        "revision"=>readchomp(`git -C $root rev-parse HEAD`),
        "benchmark_sha256"=>bytes2hex(sha256(read(@__FILE__,String)*read(joinpath(@__DIR__,"models.jl"),String))),
        "manifest_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"Manifest.toml")))),
        "packages"=>packages)
end

function rate(rows, ref)
    mse = mean(sum((r["estimate"] .- ref["mean"]).^2 ./ ref["variance"])/length(ref["mean"])
               for r in rows)
    return inv(mse*mean(r["seconds"] for r in rows))
end

function interval(rows, ref)
    rng = Xoshiro(140_991)
    samples = map(1:1000) do _
        bootstrap_ref = copy(ref)
        if haskey(ref,"chain_means")
            bootstrap_ref["mean"] = mean(rand(rng,ref["chain_means"],length(ref["chain_means"])))
        end
        rate(rand(rng,rows,length(rows)),bootstrap_ref)
    end
    return quantile(samples,[0.025,0.975])
end

function print_table(report; io=stdout, accuracy=false)
    names = report["model_order"]
    labels = filter(report["method_order"]) do label
        accuracy || all(name->haskey(first(report["models"][name]["runs"][label]),"ess"),names)
    end
    headers = ["$(replace(name,'_'=>' ')) ($(report["models"][name]["dimension"]))" for name in names]
    println(io,"| Sampler / device | ",join(headers," | ")," |")
    println(io,"|:--|",join(fill("--:",length(names)),"|"),"|")
    for label in labels
        cells = map(names) do name
            rows = report["models"][name]["runs"][label]
            any(r->r["divergences"]>0,rows) && return "divergences"
            value = accuracy ? rate(rows,report["models"][name]["reference"]) :
                    mean(r["ess"] for r in rows)/mean(r["seconds"] for r in rows)
            @sprintf("%.2g",value)
        end
        println(io,"| ",label," | ",join(cells," | ")," |")
    end
    println(io,accuracy ? "\nAccuracy-based ESS/s from repeated posterior-mean error." :
        "\nESS/s: weight ESS for IS/AMIS, minimum bulk ESS across parameters for MCMC. These are different diagnostics, not a common accuracy score.")
    println(io,"Includes fresh preparation, warmup/adaptation, and posterior means. Excludes compilation and post-run diagnostics.")
    println(io,"\nJulia ",report["metadata"]["julia"],". ",report["metadata"]["cpu"],
        ". Julia threads: ",report["metadata"]["threads"],". BLAS threads: ",report["metadata"]["blas_threads"],".")
    haskey(report["metadata"],"gpu") && println(io,"GPU: ",report["metadata"]["gpu"],".")
    println(io,"\n",report["repeats"]," independent seeds. IS samples: ",report["nsamples"],
        ". MCMC samples: ",report["mcmc_samples"],". MCMC warmup: 1024 per chain, or 256 ensemble sweeps. AMIS: four rounds.")
    println(io,"\n| Model | Sampler / device | Mean seconds | ",
        accuracy ? "Accuracy ESS/s, 95% bootstrap interval" : "ESS/s range across seeds | Max R-hat | Max mean error / posterior SD",
        " |\n|:--|:--|--:|--:|",accuracy ? "" : "--:|--:|")
    for name in names, label in labels
        rows = report["models"][name]["runs"][label]
        ref = report["models"][name]["reference"]
        lo,hi = accuracy ? interval(rows,ref) : extrema(r["ess"]/r["seconds"] for r in rows)
        @printf(io,"| %s | %s | %.3g | %.2g–%.2g |",name,label,mean(r["seconds"] for r in rows),lo,hi)
        if !accuracy
            rhat = haskey(first(rows),"max_rhat") ? @sprintf("%.3f",maximum(r["max_rhat"] for r in rows)) : "—"
            error = maximum(abs.(mean(r["estimate"] for r in rows) .- ref["mean"]) ./ sqrt.(ref["variance"]))
            @printf(io," %s | %.3g |",rhat,error)
        end
        println(io)
    end
    println(io,"\n| Package | Version |\n|:--|:--|")
    for (name,version) in sort!(collect(report["metadata"]["packages"]); by=first)
        println(io,"| ",name," | ",version," |")
    end
    println(io,"\nSource: `",report["metadata"]["revision"],"`. Manifest SHA-256: `",report["metadata"]["manifest_sha256"],"`.")
end

"""
    compare(; cuda=false, resume=false, repeats=5, nsamples=65_536, mcmc_samples=8192, reference_draws=8192,
              references=nothing,
              selected=eachindex(models()), output="results.toml")

Measure fresh end-to-end runs with BenchmarkTools, then print a Markdown table.
Each result checkpoints to TOML. Use `resume=true` to continue the same run.
Reference runs use different seeds. Pass a saved report as `references` to reuse
its reference moments. Normal data generation and compilation stay
outside timing. All methods receive the same timed Laplace initialization.
"""
function compare(; cuda=false, resume=false, repeats=5, nsamples=65_536, mcmc_samples=8192, reference_draws=8192,
                  references=nothing,
                  selected=eachindex(models()), output=joinpath(@__DIR__,"results.toml"))
    ispath(output) && !resume && error("Output already exists: $output (use --resume)")
    BLAS.set_num_threads(1)
    FFTW.set_num_threads(1)
    nsamples % 4 == 0 || error("nsamples must divide into four AMIS rounds")
    cases = models()[collect(selected)]
    methods = [(:is,nothing,"IS / CPU"),(:amis,nothing,"AMIS / CPU"),
               (:nuts,nothing,"AdvancedHMC NUTS / CPU"),(:mh,nothing,"AdvancedMH RWMH / CPU"),
               (:slice,nothing,"SliceSampling / CPU"),(:ensemble,nothing,"EnsembleMCMC DE / CPU")]
    report = Dict{String,Any}("metadata"=>metadata(), "repeats"=>repeats,"nsamples"=>nsamples,"mcmc_samples"=>mcmc_samples,
        "model_order"=>[m.name for m in cases], "models"=>Dict{String,Any}())
    if cuda
        CUDA.allowscalar(false)
        CUDA.functional() || error("CUDA is not functional")
        device = MLDataDevices.with_eltype(MLDataDevices.CUDADevice(),nothing)
        methods = vcat(methods,[(:is,device,"IS / CUDA"),(:amis,device,"AMIS / CUDA")])
        report["metadata"]["gpu"] = CUDA.name(CUDA.device())
    end
    report["method_order"] = [m[3] for m in methods]
    if resume
        saved = TOML.parsefile(output)
        for key in ("repeats", "nsamples", "mcmc_samples", "model_order", "method_order")
            saved[key] == report[key] || error("Resume configuration differs: $key")
        end
        saved["metadata"] == report["metadata"] || error("Resume environment differs")
        report = saved
    end
    save() = open(io->TOML.print(io,report),output,"w")
    for model in cases
        point = fill(0.1,model.dimension)
        gradient = similar(point)
        evaluate(gradient,point,model.data)
        isapprox(gradient,ForwardDiff.gradient(x->logtarget(x,model.data),point); rtol=1e-10) ||
            error("Analytic gradient failed for $(model.name)")
        record = get!(report["models"],model.name) do
            @info "Reference" model=model.name dimension=model.dimension
            ref = references === nothing ? reference(model; draws=reference_draws) :
                  references["models"][model.name]["reference"]
            Dict{String,Any}("reference"=>ref,
                "dimension"=>model.dimension,"observations"=>model.observations,
                "runs"=>Dict{String,Any}())
        end
        save()
        for (method,device,label) in methods
            rows = get!(record["runs"],label,Dict{String,Any}[])
            length(rows) == repeats && continue
            @info "Benchmark" model=model.name sampler=label
            count = method in (:is,:amis) ? nsamples : mcmc_samples
            # Compile once, then retain the actual timed result. @btimed would
            # execute each expensive chain again for warmup and result capture.
            captured = Ref(run_method(method,device,model,900_001; nsamples=count))
            for seed in 1001+length(rows):1000+repeats
                bench = @benchmarkable $captured[] = run_method($method,$device,$model,$seed; nsamples=$count) samples=1 evals=1 gctrial=false
                measured = minimum(run(bench; warmup=false))
                value = captured[]
                push!(rows,Dict("seed"=>seed,"seconds"=>measured.time/1e9,"host_bytes"=>measured.memory,
                    "host_allocations"=>measured.allocs,"estimate"=>value.estimate,
                    "divergences"=>value.divergences,"draws"=>value.draws,
                    diagnostics(value.samples)...))
                save()
            end
        end
    end
    print_table(report)
    return report
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS)==2 && ARGS[1] in ("--report","--accuracy-report")
        SamplerComparison.print_table(SamplerComparison.TOML.parsefile(ARGS[2]); accuracy=ARGS[1]=="--accuracy-report")
    else
        unknown = setdiff(ARGS,["--cuda","--resume"])
        isempty(unknown) || error("Usage: compare.jl [--cuda] [--resume] or compare.jl --report results.toml")
        SamplerComparison.compare(; cuda="--cuda" in ARGS, resume="--resume" in ARGS)
    end
end
