module TimeToAccuracy

using BenchmarkTools, ImportanceSamplers, LinearAlgebra, QuadGK, Random, Statistics
const KA = ImportanceSamplers.KernelAbstractions

function gaussian_target(x, p)
    a, b = x[1] - p.mean[1], x[2] - p.mean[2]
    return p.constant - (a*a - 2p.correlation*a*b + b*b) / (2p.determinant)
end
function gaussian_gradient!(g, x, p)
    a, b = x[1] - p.mean[1], x[2] - p.mean[2]
    g[1], g[2] = (-a + p.correlation*b) / p.determinant, (-b + p.correlation*a) / p.determinant
    return nothing
end
student_target(x, p) = p.constant - (p.dof + 2)/2 * log1p(sum(abs2, x) / p.dof)
function student_gradient!(g, x, p)
    scale = -(p.dof + 2) / (p.dof + sum(abs2, x))
    for j in eachindex(g)
        g[j] = scale * x[j]
    end
    return nothing
end
mixture_terms(x, p) = (
    p.constants[1] - ((x[1] + p.offset)^2 + x[2]^2) / (2p.variance),
    p.constants[2] - ((x[1] - p.offset)^2 + x[2]^2) / (2p.variance),
)
function mixture_target(x, p)
    a, b = mixture_terms(x, p)
    return max(a, b) + log1p(exp(-abs(a - b)))
end
function mixture_gradient!(g, x, p)
    a, b = mixture_terms(x, p)
    mass = inv(1 + exp(b - a))
    g[1] = -(x[1] + (2mass - 1)*p.offset) / p.variance
    g[2] = -x[2] / p.variance
    return nothing
end
softplus(x) = max(x, zero(x)) + log1p(exp(-abs(x)))
function logistic_target(x, p)
    value = -2log(p.scale) - log(oftype(p.scale, 2)*oftype(p.scale, pi))
    for j in eachindex(x)
        value += p.successes[j]*x[j] - p.totals[j]*softplus(x[j]) - (x[j]/p.scale)^2/2
    end
    return value
end
function logistic_gradient!(g, x, p)
    for j in eachindex(x)
        g[j] = p.successes[j] - p.totals[j]/(1 + exp(-x[j])) - x[j]/p.scale^2
    end
    return nothing
end

# Two independent Bernoulli groups with one coefficient each. Quadrature is an
# independent CPU oracle, not sampler working precision or a measured operation.
function logistic_oracle(p; rtol=1e-11)
    moments = map(1:2) do j
        scale = Float64(p.scale)
        density(x) = exp(p.successes[j]*x - p.totals[j]*softplus(x) - (x/scale)^2/2) /
                     (scale*sqrt(2pi))
        integrals = [first(quadgk(x -> x^k*density(x), -Inf, Inf; rtol)) for k in 0:2]
        (; logz=log(integrals[1]), mean=integrals[2]/integrals[1], second=integrals[3]/integrals[1])
    end
    means, seconds = getproperty.(moments, :mean), getproperty.(moments, :second)
    variances = seconds .- means.^2
    return (; logz=sum(m.logz for m in moments), mean=means,
            second=means*means' + Diagonal(variances), sd=sqrt.(variances))
end

function cases(::Type{T}=Float64) where {T}
    logtwopi = log(T(2)*T(pi))
    correlation = T(0.7)
    determinant = one(T) - correlation^2
    means = T[0.4, -0.3]
    gaussian = (; name=:gaussian, target=gaussian_target, gradient=gaussian_gradient!,
        context=(; mean=Tuple(means), correlation, determinant, constant=-logtwopi-log(determinant)/2),
        oracle=(; logz=0.0, mean=Float64.(means),
                  second=[1.0 Float64(correlation); Float64(correlation) 1.0] + Float64.(means)*Float64.(means)', sd=ones(2)))
    student = (; name=:student, target=student_target, gradient=student_gradient!,
        context=(; dof=T(7), constant=-logtwopi),
        oracle=(; logz=0.0, mean=zeros(2), second=7/5*Matrix{Float64}(I,2,2), sd=fill(sqrt(7/5),2)))
    mixture = (; name=:mixture, target=mixture_target, gradient=mixture_gradient!,
        context=(; offset=T(3), variance=T(0.49),
                   constants=(log(T(0.35))-logtwopi-log(T(0.49)), log(T(0.65))-logtwopi-log(T(0.49)))),
        oracle=(; logz=0.0, mean=[0.9,0.0], second=Diagonal([9.49,0.49]), sd=sqrt.([8.68,0.49])))
    context = (; successes=(14,4), totals=(20,12), scale=T(1.5))
    logistic = (; name=:logistic, target=logistic_target, gradient=logistic_gradient!,
        context, oracle=logistic_oracle(context))
    return (; gaussian, student, mixture, logistic)
end

function algorithms(case, total; T=Float64, rounds=4)
    total % (4rounds) == 0 || throw(ArgumentError("budget must divide across rounds and four proposals"))
    heavy = case.name === :student
    factor = T(2)*Matrix{T}(I,2,2)
    proposal(mean, factor) = heavy ? FactorStudentT(T(5),mean,factor) : FactorGaussian(mean,factor)
    locations = (T[-0.25,-0.25], T[-0.25,0.25], T[0.25,-0.25], T[0.25,0.25])
    bank = ProposalBank([proposal(mean,factor) for mean in locations], ones(T,4))
    # Match initial mean and covariance, although one proposal and a bank have
    # different density shapes. GRAMIS requires distinct initial centres.
    scale = sqrt(T(4) + T(0.25)^2 * (heavy ? T(3)/T(5) : one(T)))
    single = proposal(zeros(T,2),scale*Matrix{T}(I,2,2))
    options = (; rounds, round_size=total ÷ rounds)
    return (
        base=ImportanceSampling(single; nsamples=total),
        stratified=ImportanceSampling(bank; nsamples=total, mis_scheme=StratifiedMixture()),
        random=ImportanceSampling(bank; nsamples=total, mis_scheme=RandomMixture()),
        standard=ImportanceSampling(bank; nsamples=total, mis_scheme=StandardMIS()),
        partial=ImportanceSampling(bank; nsamples=total, mis_scheme=PartialDeterministicMixture(((1,2),(3,4)))),
        amis=AMIS(single; options...), npmc=NPMC(single; options...),
        dmpmc_global=DeterministicMixturePMC(bank; options...),
        dmpmc_local=DeterministicMixturePMC(bank; options..., resampling=LocalResampling()),
        apis=APIS(bank; options...), cais=CAIS(bank; options...),
        lais_rw=LAIS(bank; options..., transition=RandomWalkMetropolis(T(0.5))),
        lais_ram=LAIS(bank; options..., transition=RAM(T(0.5); tuning=ContinuousTuning())),
        lais_smh=LAIS(bank; options..., transition=SampleMetropolisHastings(single)),
        gramis=FirstOrderGRAMIS(bank; options..., repulsion_strength=T(0.1)),
    )
end

function fresh(device, case, algorithm, seed, threaded)
    prepared = prepare_sampler(Xoshiro(seed),
        LogTarget(case.target; grad=case.gradient), case.context, algorithm; threaded)
    # Default preparation already owns CPU storage. Do not rebuild it just to
    # select CPU again. Explicit converting devices still apply their policy.
    device isa ImportanceSamplers.MLDataDevices.CPUDevice{Missing} || (prepared = device(prepared))
    result = importance_sample!(prepared)
    KA.synchronize(KA.get_backend(result.logweights))
    return result
end

function assess(result, reference)
    x, w = Array(result.samples), Array(normalized_weights(result))
    means, seconds = x*w, (x .* reshape(w,1,:))*x'
    return (; actual_samples=length(w),
        z_error=exp(lognormalizer(result)-reference.logz)-1,
        mean_error=sqrt(sum(abs2,(means-reference.mean)./reference.sd)/length(means)),
        second_error=sqrt(sum(abs2,(seconds-reference.second)./(reference.sd*reference.sd'))/length(seconds)),
        ess=inv(sum(abs2,w)),
        target_evaluations=get(result.diagnostics,:target_evaluations,length(w)),
        gradient_evaluations=get(result.diagnostics,:gradient_evaluations,0))
end

"""
    measure(device, case, algorithm; seeds=1:20, threaded=true)

Measure complete fresh preparation plus synchronized sampling, never reused
adaptation. Each seed uses one BenchmarkTools measurement after an unmeasured
fresh warmup. Postprocessing and the independent oracle stay outside timing.
Bytes/allocations are host allocations, not peak host or device memory.
Forced per-seed GC sweeps are disabled. Natural GC stays inside timing.
"""
function measure(device, case, algorithm; seeds=1:20, threaded=true)
    return map(seeds) do seed
        measured = @btimed fresh($device,$case,$algorithm,$seed,$threaded) samples=1 evals=1 gctrial=false
        (; seed, seconds=measured.time, gc_seconds=measured.gctime,
           host_bytes=measured.bytes, host_allocations=measured.alloc,
           assess(measured.value,case.oracle)...)
    end
end

function rmse_summary(errors)
    rng = Xoshiro(711)
    bootstrap = [sqrt(sum(abs2,rand(rng,errors,length(errors)))/length(errors)) for _ in 1:1000]
    return (; rmse=sqrt(sum(abs2,errors)/length(errors)),
            interval95=Tuple(quantile(bootstrap,[0.025,0.975])))
end

function summarize(rows)
    times = getproperty.(rows,:seconds)
    z = rmse_summary(getproperty.(rows,:z_error))
    means = rmse_summary(getproperty.(rows,:mean_error))
    seconds = rmse_summary(getproperty.(rows,:second_error))
    cost = mean(times)
    return (; seeds=length(rows), mean_ms=1000cost, median_ms=1000median(times), range_ms=1000 .* extrema(times),
        host_bytes=median(getproperty.(rows,:host_bytes)),
        host_allocations=median(getproperty.(rows,:host_allocations)),
        ess=median(getproperty.(rows,:ess)), normalizer=z, mean=means, second_moment=seconds,
        mse_seconds=(; normalizer=z.rmse^2*cost, mean=means.rmse^2*cost, second=seconds.rmse^2*cost))
end

end
