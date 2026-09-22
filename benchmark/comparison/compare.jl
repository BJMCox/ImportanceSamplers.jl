module SamplerComparison

using BenchmarkTools, Distributions, ForwardDiff, LinearAlgebra, Optim, Random,
      Random123, Statistics, Printf, Pkg, TOML, SHA
import AbstractMCMC, AdvancedHMC, AdvancedMH, CUDA, EnsembleMCMC, ImportanceSamplers,
       LogDensityProblems, MCMCDiagnosticTools, MLDataDevices, SliceSampling
using FFTW

include("models.jl")
include("batch_targets.jl")
const IS = ImportanceSamplers
const HMC = AdvancedHMC
const IMPORTANCE_METHODS = (:is, :amis, :dmpmc, :cais, :lais, :gramis)
const ENSEMBLE_MOVES = (ensemble=EnsembleMCMC.DEMove(),
    stretch=EnsembleMCMC.StretchMove(), snooker=EnsembleMCMC.DESnookerMove(),
    gaussian=EnsembleMCMC.GaussianReplacementMove())

function ensemble_options(settings, model, label, method; sweeps=2^14)
    # Only linear has an exact Gaussian stationary start. The dimension-based
    # stretch width helped linear, but hurt schools in the separate ESS review.
    move = method === :gaussian ? (shrinkage=0.5,) :
        method === :ensemble ? (gamma0=2.38/sqrt(2model.dimension), sigma=1e-5) :
        (scale=method === :snooker ? 1.7 : model.name == "linear" ? 1+2.151/sqrt(model.dimension) : 2.0,)
    defaults = (; walkers=4model.dimension, sweeps, warmup=model.name == "linear" ? 0 : 1024, move...)
    overrides = get(get(settings,model.name,Dict()),label,Dict())
    return merge(defaults, (; (Symbol(k)=>v for (k,v) in overrides)...))
end

ensemble_move(::Val{:ensemble}, p) = EnsembleMCMC.DEMove(; p.gamma0, p.sigma)
ensemble_move(::Val{:stretch}, p) = EnsembleMCMC.StretchMove(; p.scale)
ensemble_move(::Val{:snooker}, p) = EnsembleMCMC.DESnookerMove(; p.scale)
ensemble_move(::Val{:gaussian}, p) = EnsembleMCMC.GaussianReplacementMove(; p.shrinkage)

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

const DEFAULT_POPULATION = (scale=1.2, count=16, rounds=4)

function population_proposal(family, location, factor; dof=8.0)
    family === :gaussian && return IS.FactorGaussian(location,factor)
    family === :student_t && return IS.FactorStudentT(dof,location,factor)
    error("Unknown population family: $family")
end

function tune_population(rng, device, model, bank, pilot; family=:student_t,
                         target=logtarget, data=model.data)
    start = time_ns()
    logsumexp = IS.LogExpFunctions.logsumexp
    # A separate stream prevents the main sampler from replaying pilot draws.
    prepared = IS.prepare_sampler(Xoshiro(rand(rng,UInt64)),target,data,
        IS.ImportanceSampling(bank;nsamples=pilot.nsamples,mis_scheme=IS.RandomMixture()))
    device === nothing || (prepared = device(prepared))
    samples = IS.importance_sample!(prepared)
    transfer = device === nothing ? identity : device
    d = size(samples.samples,1)
    nu = family === :gaussian ? nothing : first(bank.proposals).family.dof
    logkernel = family === :gaussian ? (r -> -r/2) : (r -> -((nu+d)/2)*log1p(r/nu))
    radii = reduce(hcat,[vec(sum(abs2,
        LowerTriangular(transfer(q.scale.factor)) \
            (samples.samples .- transfer(q.location));dims=1)) for q in bank.proposals])
    constants = transpose(transfer([q.lognormalizer+log(m/sum(bank.masses))
        for (q,m) in zip(bank.proposals,bank.masses)]))
    logmixture(logscale) = vec(logsumexp(constants .- d*logscale .+
        logkernel.(radii.*exp(-2logscale));dims=2))
    # E_q0[pi^2/(q_s*q0)] estimates the weight second moment under q_s.
    # Ordinary pi/q_s weights on q0 draws would optimize the wrong sampling law.
    numerator = 2 .* samples.logweights .+ logmixture(0.0)
    objective(logscale) = logsumexp(numerator .- logmixture(logscale))
    fit = Optim.optimize(objective,log(pilot.scale_limits[1]),log(pilot.scale_limits[2]),
        Optim.Brent();iterations=32,abs_tol=1e-3)
    scale = exp(Optim.minimizer(fit))
    fitted = IS.ProposalBank([population_proposal(family,q.location,scale*q.scale.factor;dof=nu)
        for q in bank.proposals],copy(bank.masses))
    copyto!(fitted.masses,bank.masses)
    return (; bank=fitted, scale, draws=length(samples), seconds=(time_ns()-start)/1e9)
end

function run_method(method, device, model, seed; nsamples=2^18, warmup=1024, nchains=16,
                    population=DEFAULT_POPULATION, pilot=nothing, ensemble=nothing,
                    batch_capacity=0)
    fit = laplace(model)
    rng = Xoshiro(seed)
    if method in IMPORTANCE_METHODS
        data = iszero(batch_capacity) ? model.data : merge(model.data,
            (;workspace=similar(model.data.X,model.observations,batch_capacity)))
        target = IS.LogTarget(logtarget; grad=method === :gramis ? evaluate : nothing,
            batch=iszero(batch_capacity) ? nothing : regression_batch!)
        pilot_stats = nothing
        proposal = IS.FactorStudentT(8.0, fit.location, 1.2fit.factor)
        rounds = method in (:dmpmc, :lais) ? population.rounds : 4
        options = (; rounds, round_size=nsamples÷rounds)
        algorithm = if method === :is
            IS.ImportanceSampling(proposal; nsamples)
        elseif method === :amis
            IS.AMIS(proposal; options...)
        else
            settings = method in (:dmpmc, :lais) ? population : DEFAULT_POPULATION
            family = get(settings,:family,:student_t)
            locations = fit.location .+ 0.25fit.factor*randn(rng,model.dimension,settings.count)
            bank = IS.ProposalBank([
                population_proposal(family,locations[:,j],settings.scale*fit.factor) for j in 1:settings.count
            ],ones(settings.count))
            if method in (:dmpmc,:lais) && pilot !== nothing
                tuned = tune_population(rng,device,model,bank,pilot;family,target,data)
                bank = tuned.bank
                pilot_stats = (;tuned.draws,tuned.seconds,tuned.scale)
            end
            if method === :dmpmc
                IS.DeterministicMixturePMC(bank; options...)
            elseif method === :cais
                IS.CAIS(bank; options...)
            elseif method === :gramis
                IS.FirstOrderGRAMIS(bank; repulsion_strength=0.1, options...)
            else
                covariance = (2.38^2/model.dimension)*Symmetric(fit.factor*fit.factor')
                transition = IS.RAM(covariance; tuning=IS.WarmupTuning(warmup))
                IS.LAIS(bank; transition, options...)
            end
        end
        prepared = IS.prepare_sampler(rng, target, data, algorithm)
        device === nothing || (prepared = device(prepared))
        result = IS.importance_sample!(prepared)
        # This includes the device reduction, final transfer, and its sync.
        estimate = Array(mean(result))
        return (; estimate, samples=result, divergences=0, draws=length(result), pilot=pilot_stats)
    end
    target = WhitenedTarget(model.data, fit)
    if haskey(ENSEMBLE_MOVES,method)
        options = ensemble === nothing ?
            merge(ensemble_options(Dict(),model,"",method; sweeps=cld(nsamples,4model.dimension)),(;warmup)) : ensemble
        nwalkers = options.walkers
        state = EnsembleMCMC.initialize(Philox4x((UInt64(seed),UInt64(1))),
            x -> LogDensityProblems.logdensity(target,x), randn(rng,model.dimension,nwalkers);
            move=ensemble_move(Val(method),options), executor=EnsembleMCMC.ThreadedExecutor())
        EnsembleMCMC.step!(state,options.warmup)
        result = EnsembleMCMC.sample!(state,options.sweeps)
        positions = reshape(fit.location .+ fit.factor*reshape(result.positions,model.dimension,:),
                            size(result.positions))
        return (; estimate=vec(mean(positions; dims=(2,3))), samples=(;positions),
                divergences=0, draws=length(positions)÷model.dimension, ensemble=options)
    end
    result = chains(method,target,seed,cld(nsamples,nchains),warmup,nchains)
    return (; estimate=vec(mean(result.values; dims=(1,2))), samples=result.values,
            result.divergences, draws=size(result.values,1)*nchains)
end

# Diagnostics run after the timed call. Weight ESS measures weight concentration,
# not autocorrelation or accuracy. Ensemble diagnostics use the sweep-mean process.
diagnostics(::Nothing) = Dict{String,Any}()
diagnostics(samples::IS.WeightedSamples) =
    Dict("ess"=>inv(sum(abs2,IS.normalized_weights(samples))),
         "variance"=>Array(var(samples; corrected=false)))
function diagnostics(samples::AbstractArray)
    result = MCMCDiagnosticTools.ess_rhat(samples;
        autocov_method=MCMCDiagnosticTools.FFTAutocovMethod())
    return Dict("ess"=>minimum(result.ess), "max_rhat"=>maximum(result.rhat),
                "variance"=>vec(var(reshape(samples,:,size(samples,3)); dims=1,corrected=false)))
end

function diagnostics(samples::NamedTuple{(:positions,)})
    x = samples.positions  # coordinate, walker, sweep
    d,walkers,sweeps = size(x)
    means = reshape(transpose(dropdims(mean(x;dims=2);dims=2)),sweeps,1,d)
    variance = vec(var(reshape(x,d,:);dims=2,corrected=false))
    # Goodman & Weare, section 3: diagnose the ensemble-average time series.
    # Its MCSE includes cross-walker lag covariance. Walkers are not chains.
    mcse = MCMCDiagnosticTools.mcse(means;
        autocov_method=MCMCDiagnosticTools.FFTAutocovMethod(),maxlag=sweeps÷2-1)
    return Dict("ess"=>minimum(variance ./ abs2.(mcse)),"variance"=>variance,
        "ensemble_mean_mcse"=>vec(mcse),"walkers"=>walkers,"sweeps"=>sweeps,
        "max_rhat"=>maximum(MCMCDiagnosticTools.rhat(means)))
end

function measure_run(method, device, model, seed; timing_samples=1, kwargs...)
    captured = Ref{Any}()
    bench = @benchmarkable $captured[] = run_method($method,$device,$model,$seed; $kwargs...) samples=1 evals=1 gctrial=false
    trial = run(bench; samples=timing_samples, seconds=5.0, warmup=false)
    return summarize_run(trial,captured[],seed)
end

function summarize_run(trial,value,seed)
    measured = median(trial)
    row = Dict("seed"=>seed,"seconds"=>measured.time/1e9,"host_bytes"=>measured.memory,
        "host_allocations"=>measured.allocs,"estimate"=>value.estimate,
        "divergences"=>value.divergences,"draws"=>value.draws,
        "timing_seconds"=>trial.times./1e9,"gc_seconds"=>trial.gctimes./1e9,
        diagnostics(value.samples)...)
    pilot = get(value,:pilot,nothing)
    pilot === nothing || merge!(row,Dict("pilot_draws"=>pilot.draws,
        "pilot_seconds"=>pilot.seconds,"pilot_scale"=>pilot.scale))
    ensemble = get(value,:ensemble,nothing)
    ensemble === nothing || (row["ensemble"] = Dict(string(k)=>v for (k,v) in pairs(ensemble)))
    return row
end

function population_settings(settings, model, label)
    entry = get(get(settings,model.name,Dict()),label,nothing)
    entry === nothing && return DEFAULT_POPULATION
    return (scale=entry["scale"],count=entry["count"],rounds=entry["rounds"],
            family=Symbol(get(entry,"family","student_t")))
end

function moment_errors(row, reference)
    mean_error = maximum(abs.(row["estimate"] .- reference["mean"]) ./ sqrt.(reference["variance"]))
    variance_error = haskey(row,"variance") ?
        maximum(abs.(row["variance"] ./ reference["variance"] .- 1)) : missing
    return (;mean_error,variance_error)
end

function accurate_moments(row, reference)
    error = moment_errors(row,reference)
    return error.mean_error <= 0.2 && (ismissing(error.variance_error) || error.variance_error <= 0.3)
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
        "benchmark_sha256"=>bytes2hex(sha256(join(read(joinpath(@__DIR__,file),String)
            for file in ("compare.jl","models.jl","signal_background.jl")))),
        "data_sha256"=>Dict(file=>bytes2hex(sha256(read(joinpath(@__DIR__,"data","signal-background",file))))
            for file in ("sample_table.csv","summary_dataset_table.csv")),
        "package_revisions"=>Dict(info.name=>info.git_revision for info in values(Pkg.dependencies())
            if info.is_direct_dep && info.git_revision !== nothing),
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
    println(io,accuracy ? "Posterior-mean accuracy per second.\n" : "Effective sample size per second (ESS/s)\\*.\n")
    names = report["model_order"]
    labels = filter(report["method_order"]) do label
        accuracy || all(name->haskey(first(report["models"][name]["runs"][label]),"ess"),names)
    end
    selection = get(report,"ensemble_selection",Dict())
    headline = isempty(selection) ? labels :
        [filter(label->!startswith(label,"EnsembleMCMC "),labels); "EnsembleMCMC (selected) / CPU"]
    rows_for(label,name) = get(report["models"][name]["runs"],
        label == "EnsembleMCMC (selected) / CPU" ? get(selection,name,"") : label,())
    scores = map(Iterators.product(headline,names)) do (label,name)
        rows = rows_for(label,name)
        isempty(rows) ? -Inf : accuracy ? rate(rows,report["models"][name]["reference"]) :
            mean(r["ess"] for r in rows)/mean(r["seconds"] for r in rows)
    end
    headers = ["$(uppercasefirst(replace(name,'_'=>' '))) ($(report["models"][name]["dimension"]))" for name in names]
    for device in ("CPU","CUDA")
        indices = findall(label->endswith(label," / $device"),headline)
        isempty(indices) && continue
        best = maximum(scores[indices,:]; dims=1)
        println(io,"### ",device,"\n\n| Sampler | ",join(headers," | ")," |")
        println(io,"|:--|",join(fill("--:",length(names)),"|"),"|")
        for i in indices
            label = headline[i]
            cells = map(eachindex(names)) do j
                rows = rows_for(label,names[j])
                isempty(rows) && return "—"
                mark = (any(r->r["divergences"]>0,rows) ? "†" : "") *
                       (any(r->get(r,"max_rhat",0.0)>1.01,rows) ? "‡" : "")
                if haskey(first(rows),"ess") && !haskey(first(rows),"max_rhat")
                    maximum(r["ess"] for r in rows) > 10minimum(r["ess"] for r in rows) && (mark *= "§")
                end
                reference = report["models"][names[j]]["reference"]
                any(r->!accurate_moments(r,reference),rows) && (mark *= "¶")
                value = @sprintf("%.2g",scores[i,j])
                (device == "CPU" && scores[i,j] == best[j] ? "**$value**" : value) * mark
            end
            println(io,"| ",replace(label," / $device"=>"")," | ",join(cells," | ")," |")
        end
        println(io)
    end
    if !isempty(selection)
        choices = ["$name: " * (isempty(get(selection,name,"")) ? "no eligible move" :
            replace(selection[name],"EnsembleMCMC "=>""," / CPU"=>"")) for name in names]
        println(io,"EnsembleMCMC moves: ",join(choices,"; "),".\n")
        println(io,"Moves maximize screening mean ESS / mean seconds among accuracy-eligible, width-selected candidates. Reporting seeds do not select the move. All held-out move rows remain below; width-screen trials remain in the screen artifact.\n")
    end
    println(io,accuracy ? "\nAccuracy-based ESS/s estimates posterior-mean precision per unit time." :
        "\n\\* ESS denotes weight ESS for importance sampling, minimum bulk ESS for independent-chain MCMC, and minimum mean ESS for EnsembleMCMC. Ensemble mean ESS is marginal variance divided by the squared MCSE of the sweep-mean process. These diagnostics do not define an equal-accuracy comparison.")
    println(io,"\nBold denotes the highest measured CPU rate per model. GPU results appear separately.")
    println(io,"\n† At least one retained NUTS transition diverged. ‡ At least one run had maximum R-hat above 1.01. § Weight ESS varied by more than a factor of ten across seeds.")
    println(io,"\n¶ At least one run exceeded a mean error of 0.2 posterior standard deviations or a marginal variance error of 30%.")
    println(io,"\nEnsemble R-hat splits the sweep-mean time series, not the walkers. Settings below apply when recorded; archived ensemble runs used 4d walkers and rounded their pooled draw budget to complete sweeps.")
    ensemble_rows = [(name,label,first(report["models"][name]["runs"][label])) for name in names for label in labels
        if haskey(first(report["models"][name]["runs"][label]),"ensemble")]
    if !isempty(ensemble_rows)
        println(io,"\n| Model | Ensemble move | Walkers | Retained sweeps | Warmup sweeps | Move settings |\n|:--|:--|--:|--:|--:|:--|")
        for (name,label,row) in ensemble_rows
            p = row["ensemble"]
            move = join(["$k=$(p[k])" for k in ("gamma0","sigma","scale","shrinkage") if haskey(p,k)],", ")
            println(io,"| $name | $label | ",p["walkers"]," | ",p["sweeps"]," | ",p["warmup"]," | $move |")
        end
    end
    search = get(report,"configuration_search",Dict())
    if !isempty(search)
        println(io,"\nOffline configuration search: `",get(search,"source","unrecorded"),
            "`, SHA-256 `",get(search,"sha256","unrecorded"),"`. Screening seeds: ",
            join(get(search,"seeds",Int[]),", "),". Selection: ",get(search,"selection","unrecorded"),".")
        failed = get(search,"no_eligible_candidate",String[])
        isempty(failed) || println(io,"\nNo screening candidate met the moment checks for ",
            join(failed,", "),". Those cases retain the baseline; held-out diagnostics remain separate.")
    end
    haskey(report,"host_load") && println(io,"\nHost conditions: ",report["host_load"]["note"],
        " ",report["host_load"]["before"],"; ",report["host_load"]["after"],".")
    println(io,"\nElapsed time includes initialization, warmup or adaptation, sampling, and posterior-mean estimation. Compilation and post-run diagnostics are excluded.")
    haskey(report,"timing_samples") && println(io,haskey(report,"sources") ?
        "\nRefreshed rows use the median of up to " : "\nTimes use the median of up to ",
        report["timing_samples"]," executions per seed under a five-second BenchmarkTools budget. A complete execution may exceed that budget. Raw times and GC times are retained.")
    haskey(report,"sources") && println(io,"\nArchived rows retain one timed execution per seed and mean checks. They did not record per-run variances. A dash denotes an unavailable variance check, not a pass.")
    pilot = get(report,"proposal_pilot",Dict())
    if !isempty(pilot)
        println(io,"\nDM-PMC and LAIS use an independent ",pilot["nsamples"],
            "-draw width pilot. Its full cost is timed. Only proposal widths pass to the main run; its settings and retained sample count are unchanged.")
        println(io,"\n| Model | Sampler / device | Mean pilot seconds | Width multiplier range |\n|:--|:--|--:|--:|")
        for name in names, label in labels
            rows = report["models"][name]["runs"][label]
            haskey(first(rows),"pilot_seconds") || continue
            lo,hi = extrema(r["pilot_scale"] for r in rows)
            @printf(io,"| %s | %s | %.3g | %.3g–%.3g |\n",name,label,
                mean(r["pilot_seconds"] for r in rows),lo,hi)
        end
    end
    println(io,"\nJulia ",report["metadata"]["julia"],". ",report["metadata"]["cpu"],
        ". Julia threads: ",report["metadata"]["threads"],". BLAS threads: ",report["metadata"]["blas_threads"],".")
    haskey(report["metadata"],"gpu") && println(io,"GPU: ",report["metadata"]["gpu"],".")
    println(io,"\n",report["repeats"]," independent seeds. IS samples: ",report["nsamples"],
        ". MCMC samples: ",report["mcmc_samples"],". MCMC chains: ",get(report,"mcmc_chains",report["metadata"]["threads"]),
        ". MCMC warmup: 1024 per chain. Ensemble budgets and warmup are recorded per row.")
    settings = get(report,"population_settings",Dict())
    if !isempty(settings)
        println(io,"\nPopulation settings supplied to this run. Per-run initialization and adaptation remain timed.\n")
        println(io,"| Model | Sampler / device | Family | Initial scale / Laplace factor | Proposals | Rounds |\n|:--|:--|:--|--:|--:|--:|")
        for name in names, (label,p) in sort!(collect(get(settings,name,Dict()));by=first)
            println(io,"| $name | $label | ",get(p,"family","student_t")," | ",
                p["scale"]," | ",p["count"]," | ",p["rounds"]," |")
        end
    end
    println(io,"\n| Model | Sampler / device | Mean seconds | ",
        accuracy ? "Accuracy ESS/s, 95% bootstrap interval" : "ESS/s range across seeds | Max R-hat | Max pooled-mean error / posterior SD | Max per-run variance error",
        " |\n|:--|:--|--:|--:|",accuracy ? "" : "--:|--:|--:|")
    for name in names, label in labels
        rows = report["models"][name]["runs"][label]
        ref = report["models"][name]["reference"]
        lo,hi = accuracy ? interval(rows,ref) : extrema(r["ess"]/r["seconds"] for r in rows)
        @printf(io,"| %s | %s | %.3g | %.2g–%.2g |",name,label,mean(r["seconds"] for r in rows),lo,hi)
        if !accuracy
            rhat = haskey(first(rows),"max_rhat") ? @sprintf("%.3f",maximum(r["max_rhat"] for r in rows)) : "—"
            error = maximum(abs.(mean(r["estimate"] for r in rows) .- ref["mean"]) ./ sqrt.(ref["variance"]))
            variance_error = all(r->haskey(r,"variance"),rows) ?
                @sprintf("%.3g",maximum(moment_errors(r,ref).variance_error for r in rows)) : "—"
            @printf(io," %s | %.3g | %s |",rhat,error,variance_error)
        end
        println(io)
    end
    if haskey(report,"timing_samples")
        println(io,"\n| Model | Sampler / device | Timed executions | Raw seconds range |\n|:--|:--|--:|--:|")
        for name in names, label in labels
            times = reduce(vcat,(get(r,"timing_seconds",[r["seconds"]]) for r in report["models"][name]["runs"][label]))
            @printf(io,"| %s | %s | %d | %.3g–%.3g |\n",name,label,length(times),extrema(times)...)
        end
    end
    if haskey(report,"sources")
        println(io,"\n## Measurement sources\n\nUnchanged rows reuse archived measurements. Source files retain their original seeds, timing protocol, and hashes.\n")
        println(io,"| Model | Sampler / device | Source |\n|:--|:--|:--|")
        for name in names, label in labels
            sources = unique(r["source"] for r in report["models"][name]["runs"][label])
            println(io,"| $name | $label | ",join(["[$s]($s)" for s in sources],", ")," |")
        end
    end
    haskey(report,"sources") && println(io,
        "\nThis package table describes only the report-environment source. The linked measurement sources retain their own package versions and revisions.")
    println(io,"\n| Package | Version | Source revision |\n|:--|:--|:--|")
    for (name,version) in sort!(collect(report["metadata"]["packages"]); by=first)
        revision = get(get(report["metadata"],"package_revisions",Dict()),name,"—")
        println(io,"| ",name," | ",version," | ",revision," |")
    end
    println(io,haskey(report,"sources") ? "\nReport environment source: `" : "\nSource: `",
        report["metadata"]["revision"],"`. Manifest SHA-256: `",report["metadata"]["manifest_sha256"],"`.")
end

"""
    compare(; cuda=false, cpu=true, resume=false, repeats=3, timing_samples=3, nsamples=2^18, mcmc_draws=2^14, nchains=16, reference_draws=8192,
              ensemble_sweeps=2^14, ensemble_settings=Dict(), configuration_search=Dict(),
              references=nothing, settings=Dict(), pilot=nothing, seed_start=1001,
              selected=eachindex(models()), only_methods=nothing, reuse=nothing, output="results.toml")

Measure fresh end-to-end runs with BenchmarkTools, then print a Markdown table.
Each result checkpoints to TOML. Use `resume=true` to continue the same run.
Reference runs use different seeds. Pass a saved report as `references` to reuse
its reference moments. Normal data generation and compilation stay
outside timing. All methods receive the same timed Laplace initialization.
Each seed uses up to `timing_samples` executions within a five-second timing
budget. The median supplies its elapsed time. All raw times remain in the report.
An optional `pilot=(nsamples=4096, scale_limits=(0.25,2.0))` tunes only lower
proposal widths for DM-PMC and LAIS. Its cost is timed, and its samples and
settings remain separate from the main sampling budget and round schedule.
Population settings accept `family="gaussian"` or `"student_t"` (the default).
Gaussian factors define covariance directly. Student-t factors define scale.
Ensembles retain `ensemble_sweeps` time steps per walker, not a pooled draw
count. `ensemble_settings[model][label]` overrides walkers, sweeps, warmup, and
move parameters. Every result records the resolved settings. Freeze them before
the held-out seeds. Linear alone defaults to zero warmup and a narrower Stretch move.
`configuration_search` records an offline search's source, hashes, and selection
status. Resume requires the same search provenance as well as the same settings.
Use `only_methods` to measure a subset. `reuse` accepts a compatible earlier
report for unchanged importance-sampler calls. Ensemble rows need retained
sweep diagnostics and are measured anew. Every reused row keeps its source.
"""
function compare(; cuda=false, cpu=true, resume=false, repeats=3, timing_samples=3, nsamples=2^18, mcmc_draws=2^14, nchains=16, reference_draws=8192,
                  ensemble_sweeps=2^14, ensemble_settings=Dict(), configuration_search=Dict(),
                  references=nothing, settings=Dict(), pilot=nothing, seed_start=1001,
                  selected=eachindex(models()), only_methods=nothing, reuse=nothing,
                  output=joinpath(@__DIR__,"results.toml"))
    ispath(output) && !resume && error("Output already exists: $output (use --resume)")
    BLAS.set_num_threads(1)
    FFTW.set_num_threads(1)
    nsamples % 64 == 0 || error("nsamples must divide across four rounds and sixteen proposals")
    cases = models()[collect(selected)]
    methods = [(:is,nothing,"IS / CPU"),(:amis,nothing,"AMIS / CPU"),
               (:dmpmc,nothing,"DM-PMC / CPU"),(:cais,nothing,"CAIS / CPU"),(:lais,nothing,"LAIS-RAM / CPU"),
               (:gramis,nothing,"First-order GRAMIS-CAIS / CPU"),
               (:nuts,nothing,"AdvancedHMC NUTS / CPU"),(:mh,nothing,"AdvancedMH RWMH / CPU"),
               (:slice,nothing,"SliceSampling / CPU"),(:ensemble,nothing,"EnsembleMCMC DE / CPU"),
               (:stretch,nothing,"EnsembleMCMC Stretch / CPU"),(:snooker,nothing,"EnsembleMCMC snooker / CPU"),
               (:gaussian,nothing,"EnsembleMCMC Gaussian replacement / CPU")]
    only_methods === nothing || filter!(m->m[1] in only_methods,methods)
    mcmc_samples = mcmc_draws*nchains
    report = Dict{String,Any}("metadata"=>metadata(), "repeats"=>repeats,"timing_samples"=>timing_samples,"nsamples"=>nsamples,"mcmc_samples"=>mcmc_samples,
        "mcmc_chains"=>nchains,"mcmc_draws_per_chain"=>mcmc_draws,
        "ensemble_sweeps"=>ensemble_sweeps,"ensemble_settings"=>ensemble_settings,
        "configuration_search"=>configuration_search,
        "seeds"=>collect(seed_start:seed_start+repeats-1),"population_settings"=>settings,
        "proposal_pilot"=>pilot === nothing ? Dict() :
            Dict("nsamples"=>pilot.nsamples,"scale_limits"=>collect(pilot.scale_limits)),
        "model_order"=>[m.name for m in cases], "models"=>Dict{String,Any}())
    if cuda
        CUDA.allowscalar(false)
        CUDA.functional() || error("CUDA is not functional")
        device = MLDataDevices.with_eltype(MLDataDevices.CUDADevice(),nothing)
        methods = reduce(vcat,[method in IMPORTANCE_METHODS ?
            [(method,nothing,label),(method,device,replace(label,"CPU"=>"CUDA"))] :
            [(method,nothing,label)] for (method,_,label) in methods])
        report["metadata"]["gpu"] = CUDA.name(CUDA.device())
    end
    cpu || filter!(m->m[2] !== nothing,methods)
    isempty(methods) && error("Enable at least one device")
    report["method_order"] = [m[3] for m in methods]
    if reuse !== nothing && !resume
        saved = TOML.parsefile(reuse)
        for key in ("repeats","timing_samples","nsamples","mcmc_samples","mcmc_chains","seeds","population_settings","proposal_pilot")
            saved[key] == report[key] || error("Reuse configuration differs: $key")
        end
        for key in ("julia","cpu","threads","blas_threads","manifest_sha256")
            saved["metadata"][key] == report["metadata"][key] || error("Reuse environment differs: $key")
        end
        reusable = [label for (method,_,label) in methods if method in IMPORTANCE_METHODS]
        for name in intersect(report["model_order"],keys(saved["models"]))
            record = deepcopy(saved["models"][name])
            filter!(pair->first(pair) in reusable,record["runs"])
            for rows in values(record["runs"]), row in rows
                row["source"] = basename(reuse)
            end
            report["models"][name] = record
        end
        report["reused_source"] = basename(reuse)
        report["reused_metadata"] = saved["metadata"]
    end
    if resume
        saved = TOML.parsefile(output)
        for key in ("repeats", "timing_samples", "nsamples", "mcmc_samples", "mcmc_chains", "ensemble_sweeps", "ensemble_settings", "configuration_search", "model_order", "method_order", "seeds", "population_settings", "proposal_pilot")
            saved[key] == report[key] || error("Resume configuration differs: $key")
        end
        saved["metadata"] == report["metadata"] || error("Resume environment differs")
        report = saved
    end
    save() = open(io->TOML.print(io,report),output,"w")
    for model in cases
        for point in (zeros(model.dimension), fill(0.1,model.dimension))
            gradient = similar(point)
            evaluate(gradient,point,model.data)
            isapprox(gradient,ForwardDiff.gradient(x->logtarget(x,model.data),point); rtol=1e-10) ||
                error("Analytic gradient failed for $(model.name)")
        end
        record = get!(report["models"],model.name) do
            @info "Reference" model=model.name dimension=model.dimension
            ref = references !== nothing && haskey(references["models"],model.name) ?
                  references["models"][model.name]["reference"] : reference(model; draws=reference_draws,nchains=8)
            Dict{String,Any}("reference"=>ref,
                "dimension"=>model.dimension,"observations"=>model.observations,
                "runs"=>Dict{String,Any}())
        end
        save()
        for (method,device,label) in methods
            population = population_settings(settings,model,label)
            ensemble = haskey(ENSEMBLE_MOVES,method) ?
                ensemble_options(ensemble_settings,model,label,method; sweeps=ensemble_sweeps) : nothing
            if method in (:dmpmc,:lais)
                nsamples % (population.count*population.rounds) == 0 || error("Population budget does not divide: $label")
            end
            rows = get!(record["runs"],label,Dict{String,Any}[])
            length(rows) == repeats && continue
            @info "Benchmark" model=model.name sampler=label
            count = method in IMPORTANCE_METHODS ? nsamples : mcmc_samples
            # Compile once, then retain the actual timed result. @btimed would
            # execute each expensive chain again for warmup and result capture.
            # Compile with a short run. Population covariance fits require more
            # than d+1 samples per proposal, even for this untimed warmup.
            compile_count = method in IMPORTANCE_METHODS ? max(4096,64*(model.dimension+2),population.count*population.rounds) : 1024
            compile_ensemble = ensemble === nothing ? nothing : merge(ensemble,(;sweeps=32,warmup=32))
            run_method(method,device,model,900_001; nsamples=compile_count,warmup=128,nchains,population,pilot,ensemble=compile_ensemble)
            for seed in seed_start+length(rows):seed_start+repeats-1
                push!(rows,measure_run(method,device,model,seed; nsamples=count,nchains,population,pilot,ensemble,timing_samples))
                rows[end]["source"] = basename(output)
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
