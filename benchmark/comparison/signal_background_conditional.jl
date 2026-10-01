module SignalBackgroundConditional

include("compare.jl")
include("signal_background_proposal.jl")
using .SignalBackgroundProposals: SignalBackgroundProposal
using BenchmarkTools, LinearAlgebra, Optim, Printf, Random, SHA, Statistics, TOML
import CUDA, MLDataDevices
const SC = SamplerComparison
const IS = SC.IS
const WIDTH_PILOT = (nsamples=4096, scale_limits=(0.25, 2.0))

function fitted_t(x, weights, dof)
    T = eltype(x)
    w = weights ./ sum(weights)
    location = x*w
    residual = x .- location
    scale = Matrix(Symmetric((residual .* transpose(w))*transpose(residual)))
    for i in axes(scale, 1)
        scale[i, i] += T(1e-6)*max(scale[i, i], eps(T))
    end
    scale .*= (dof - T(2))/dof
    return IS.FactorStudentT(dof, location, cholesky(Symmetric(scale)))
end

"""
    fit_proposal(samples)

Fit the frozen signal-background proposal from CPU pilot draws and log weights.
The three global Student-t components split weighted background-mean quantiles
at 0.5 and 0.9. Five conditional Student-t fits model the detector effects.
The global mixture retains 5% of the exact prior. Degrees of freedom are eight.
This is a model-specific proposal fit, not a change to LAIS.
"""
function fit_proposal(samples)
    x, w = Array(samples.samples), Array(IS.normalized_weights(samples))
    T = eltype(x)
    dof = T(8)
    order = sortperm(view(x, 3, :))
    cumulative = cumsum(w[order])
    edges = [0, searchsortedfirst(cumulative, T(0.5)),
        searchsortedfirst(cumulative, T(0.9)), length(w)]
    groups = [order[edges[i]+1:edges[i+1]] for i in 1:3]
    bank = IS.ProposalBank([fitted_t(x[1:4, ids], w[ids], dof) for ids in groups],
        [sum(w[ids]) for ids in groups])

    sigma = T(0.1) .+ T(0.9).*SC.sigmoid.(view(x, 2, :))
    logmean = log.(T(1e-10) .+ (T(20)-T(1e-10)).*SC.sigmoid.(view(x, 3, :))) .- sigma.^2/T(2)
    coefficients = Matrix{T}(undef, 3, 5)
    for j in 1:5
        z = view(x, 4+j, :)
        function fit_at(logb)
            b = exp(logb)
            precision = one(T) .+ b.*sigma.^2
            slope = sigma ./ precision
            adjusted = z .+ b.*slope.*logmean
            a = sum(w.*precision.*slope.*adjusted)/sum(w.*precision.*slope.^2)
            variance = sum(w.*precision.*(adjusted.-a.*slope).^2)
            return (; a, b, variance, objective=log(variance)-sum(w.*log.(precision)))
        end
        optimum = Optim.optimize(t -> fit_at(t).objective, log(T(1e-3)), log(T(1e3)), Optim.Brent())
        fit = fit_at(Optim.minimizer(optimum))
        coefficients[:, j] = [fit.a, fit.b, sqrt(fit.variance*(dof-T(2))/dof)]
    end
    return SignalBackgroundProposal(bank, coefficients; prior_mass=T(0.05))
end

function run_conditional(device, model, seed; nsamples=2^18, pilot_draws=2^16)
    # Pilot and production streams are disjoint. Only the fitted law is reused.
    pilot = SC.run_method(:lais, nothing, model, seed+2_000_000;
        nsamples=pilot_draws, pilot=WIDTH_PILOT)
    proposal = fit_proposal(pilot.samples)
    sampler = IS.prepare_sampler(Xoshiro(seed+3_000_000), IS.LogTarget(SC.logtarget),
        model.data, IS.ImportanceSampling(proposal; nsamples))
    device === nothing || (sampler = device(sampler))
    samples = IS.importance_sample!(sampler)
    estimate = Array(mean(samples))
    device === nothing || CUDA.synchronize()
    return (; samples, estimate, draws=length(samples), divergences=0)
end

function print_table(report; io=stdout)
    reference = report["reference"]
    for device in ("CPU", "CUDA")
        labels = filter(label -> endswith(label, device), report["method_order"])
        isempty(labels) && continue
        println(io, "## ", device, "\n\n| Workflow | Median seconds (range) | Weight ESS/s | Moment checks passed |")
        println(io, "|:--|--:|--:|--:|")
        for label in labels
            rows = filter(row -> row["label"] == label, report["runs"])
            isempty(rows) && continue
            times = [row["seconds"] for row in rows]
            rate = median(row["ess"]/row["seconds"] for row in rows)
            passed = count(row -> SC.accurate_moments(row, reference), rows)
            @printf(io, "| %s | %.4g (%.4g–%.4g) | %.4g | %d/%d |\n",
                label, median(times), extrema(times)..., rate, passed, length(rows))
        end
        println(io)
    end
    println(io, "Times include initialization, all pilots, fitting, transfer, production and the final mean. Compilation and diagnostics are excluded.")
    println(io, "Weight ESS is not tail-functional ESS. Moment checks use the unchanged archived reference and are not tail-coverage guarantees.\n")
    meta = report["metadata"]
    println(io, "Julia ", meta["julia"], "; ", meta["threads"], " CPU threads; ", meta["blas_threads"], " BLAS threads.")
    println(io, "CPU: ", meta["cpu"], ".")
    haskey(meta, "gpu") && println(io, "GPU: ", meta["gpu"], ".")
    println(io, "Retained draws: ", report["nsamples"], "; conditional-fit pilot draws: ", report["pilot_draws"], ".")
    println(io, "Source: `", meta["revision"], "`. Manifest SHA-256: `", meta["manifest_sha256"], "`.")
    println(io, "\n| Package | Version | Source revision |\n|:--|:--|:--|")
    for (name, version) in sort!(collect(meta["packages"]); by=first)
        println(io, "| ", name, " | ", version, " | ", get(meta["package_revisions"], name, "—"), " |")
    end
end

function benchmark(path; cuda=false, seeds=10101:10106, nsamples=2^18, pilot_draws=2^16)
    ispath(path) && error("Results already exist: $path")
    BLAS.set_num_threads(1)
    model = SC.signal_background()
    devices = cuda ? [nothing, MLDataDevices.with_eltype(MLDataDevices.CUDADevice(), nothing)] : [nothing]
    cuda && CUDA.allowscalar(false)
    operations = [(label="$(conditional ? "Conditional IS" : "LAIS-RAM") / $(device === nothing ? "CPU" : "CUDA")",
        run=conditional ?
            (seed -> run_conditional(device, model, seed; nsamples, pilot_draws)) :
            (seed -> SC.run_method(:lais, device, model, seed; nsamples, pilot=WIDTH_PILOT)))
        for device in devices for conditional in (false, true)]
    reference_file = joinpath(@__DIR__, "signal-background-results-2026-09-28-gates.toml")
    report = Dict{String,Any}("metadata"=>SC.metadata(), "nsamples"=>nsamples,
        "pilot_draws"=>pilot_draws, "width_pilot_draws"=>WIDTH_PILOT.nsamples,
        "seeds"=>collect(seeds), "method_order"=>[op.label for op in operations],
        "reference"=>TOML.parsefile(reference_file)["models"][model.name]["reference"],
        "reference_sha256"=>bytes2hex(sha256(read(reference_file))),
        "source_sha256"=>Dict(file=>bytes2hex(sha256(read(joinpath(@__DIR__, file))))
            for file in ("signal_background_conditional.jl", "signal_background_proposal.jl")),
        "runs"=>Any[])
    cuda && (report["metadata"]["gpu"] = CUDA.name(CUDA.device()))
    for op in operations
        op.run(10000)
    end
    captured = Ref{Any}()
    for (block, seed) in enumerate(seeds)
        order = isodd(block) ? operations : reverse(operations)
        for op in order
            GC.gc(false)
            load = collect(Sys.loadavg())
            operation = op.run
            bench = @benchmarkable $captured[] = $operation($seed) samples=1 evals=1 gctrial=false
            trial = run(bench; samples=1, seconds=30.0, warmup=false)
            row = SC.summarize_run(trial, captured[], seed)
            merge!(row, Dict("label"=>op.label, "block"=>block, "load"=>load))
            push!(report["runs"], row)
            open(io -> TOML.print(io, report; sorted=true), path, "w")
        end
    end
    print_table(report)
    return report
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 2 && ARGS[1] == "--report"
        SignalBackgroundConditional.print_table(SignalBackgroundConditional.TOML.parsefile(ARGS[2]))
    else
        paths = filter(!=("--cuda"), ARGS)
        length(paths) == 1 || error("Usage: signal_background_conditional.jl [--cuda] output.toml, or --report output.toml")
        SignalBackgroundConditional.benchmark(only(paths); cuda="--cuda" in ARGS)
    end
end
