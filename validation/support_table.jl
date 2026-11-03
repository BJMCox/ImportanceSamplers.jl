module SamplerSupportTable

using ImportanceSamplers, Random
import MLDataDevices

scalar_target(x::T) where {T} = -(abs2(x) + log(T(2) * T(pi))) / T(2)
function vector_target(x)
    T = eltype(x)
    total = zero(T)
    for value in x
        total += abs2(value)
    end
    return -(total + length(x) * log(T(2) * T(pi))) / T(2)
end
gradient!(g, x) = (g .= .-x; nothing)
product_target(x) = scalar_target(x.offset) + vector_target(x.coefficients)
named_target(x) = -sum(abs2, x.weights) - x.rate - abs2(x.offset)
function named_batch!(out, x)
    out .= .-vec(sum(abs2, x.weights; dims=1)) .- x.rate .- abs2.(x.offset)
    return nothing
end

case(name, example, algorithm; target=vector_target, precision=Float32,
    keywords=(;), self_target=false, limits=(;)) =
    (; name, example, algorithm, target, precision, keywords, self_target, limits)

# These are finite execution examples, not a Cartesian product of capabilities.
function support_cases()
    proposal = FactorGaussian(zeros(Float32, 2), Float32[1.2 0; 0.2 1.1])
    bank = ProposalBank([proposal,
        FactorGaussian(Float32[0.5, -0.5], Float32[1.1 0; -0.1 1.2])])
    student_bank = ProposalBank([
        FactorStudentT(6f0, Float32[-0.5, 0], Float32[1.2 0; 0.2 1.1]),
        FactorStudentT(6f0, Float32[0.5, 0], Float32[1.1 0; -0.1 1.2]),
    ])
    product = ProductProposal((offset=SphericalGaussian(0f0, 1f0), coefficients=proposal))
    cooperative = (reactant_cpu=:reactant_cpu_cooperative_kernels,)
    return (
        case("Plain IS", "scalar Gaussian", ImportanceSampling(
            SphericalGaussian(0f0, 1f0); nsamples=256); target=scalar_target, self_target=true),
        case("Plain IS", "factor Student-t", ImportanceSampling(
            FactorStudentT(6.0, zeros(2), [1.2 0; 0.2 1.1]); nsamples=256);
            precision=Float64),
        case("Static MIS: stratified", "factor Gaussian bank",
            ImportanceSampling(bank; nsamples=256, mis_scheme=StratifiedMixture())),
        case("Static MIS: random", "factor Gaussian bank",
            ImportanceSampling(bank; nsamples=256, mis_scheme=RandomMixture())),
        case("Static MIS: standard", "factor Gaussian bank",
            ImportanceSampling(bank; nsamples=256, mis_scheme=StandardMIS())),
        case("Static MIS: partial DM", "factor Gaussian bank", ImportanceSampling(
            bank; nsamples=256, mis_scheme=PartialDeterministicMixture(((1,), (2,))))),
        case("AMIS", "factor Gaussian", AMIS(proposal; rounds=2, round_size=128); limits=cooperative),
        case("N-PMC", "scalar Student-t", NPMC(SphericalStudentT(6f0, 0f0, 1f0);
            rounds=2, round_size=128); target=scalar_target),
        case("DM-PMC: global", "factor Gaussian bank",
            DeterministicMixturePMC(bank; rounds=2, round_size=128)),
        case("DM-PMC: local", "factor Gaussian bank", DeterministicMixturePMC(
            bank; rounds=2, round_size=128, resampling=LocalResampling()); limits=cooperative),
        case("APIS", "factor Gaussian bank", APIS(bank; rounds=2, round_size=128); limits=cooperative),
        case("CAIS", "factor Student-t bank", CAIS(student_bank; rounds=2, round_size=128); limits=cooperative),
        case("First-order GRAMIS", "factor Gaussian bank; supplied gradient",
            FirstOrderGRAMIS(bank; rounds=2, round_size=128, repulsion_strength=0f0);
            target=LogTarget(vector_target; grad=gradient!), limits=cooperative),
        case("LAIS: RWM", "factor Gaussian bank", LAIS(bank;
            transition=RandomWalkMetropolis(0.25f0), rounds=2, round_size=128)),
        case("LAIS: RAM", "factor Gaussian bank; four warmup moves", LAIS(bank;
            transition=RAM(0.25f0; tuning=WarmupTuning(4)), rounds=2, round_size=128);
            limits=(reactant_cpu=:reactant_cpu_transition_tuning,)),
        case("LAIS: SMH", "factor Gaussian bank; Student-t upper proposal", LAIS(bank;
            transition=SampleMetropolisHastings(SphericalStudentT(6f0, zeros(Float32, 2), 1f0)),
            rounds=2, round_size=128); limits=cooperative),
        case("Plain IS: named product", "scalar + correlated Gaussian vector",
            ImportanceSampling(product; nsamples=256); target=product_target),
        case("Plain IS: named batch", "simplex + positive + identity",
            ImportanceSampling(SphericalGaussian(zeros(Float32, 4), 1f0); nsamples=256);
            target=LogTarget(named_target; batch=named_batch!),
            keywords=(transform=(weights=1:2=>SimplexTransform(3),
                rate=3=>PositiveTransform(), offset=4),)),
    )
end

sample_arrays(x::AbstractArray) = (x,)
sample_arrays(x::NamedTuple) = values(x)

function check_support_case(row; device=MLDataDevices.CPUDevice(),
    backend=:cpu, isresident=x -> x isa Array, threaded=true)
    sampler = prepare_sampler(Xoshiro(73), row.target, row.algorithm;
        threaded, row.keywords...)
    prepared = try
        device(sampler)
    catch exception
        if haskey(row.limits, backend) && exception isa SamplerDeviceError &&
            exception.reason === row.limits[backend]
            return :rejected
        end
        rethrow()
    end
    haskey(row.limits, backend) && error("$(row.name): expected backend restriction was not enforced")
    result = importance_sample!(prepared)
    all(isresident, (sample_arrays(result.samples)..., result.logweights)) ||
        error("$(row.name): output is not resident on the requested device")
    host = MLDataDevices.CPUDevice()(result)
    length(host) == 256 || error("$(row.name): incorrect sample count")
    arrays = (sample_arrays(host.samples)..., host.logweights)
    all(x -> eltype(x) === row.precision, arrays) ||
        error("$(row.name): sample or log-weight precision changed")
    all(x -> all(isfinite, x), arrays) || error("$(row.name): nonfinite output")
    weights = normalized_weights(host)
    isapprox(sum(weights), one(row.precision); atol=32eps(row.precision)) ||
        error("$(row.name): weights do not normalize")
    if row.self_target
        maximum(abs, host.logweights) <= 16eps(row.precision) ||
            error("plain IS: normalized self-target identity failed")
    end
    return :passed
end

function checked_support_table(; cases=support_cases(), backend=:cpu, kwargs...)
    backends = (:cpu=>"CPU", :cuda=>"CUDA", :metal=>"Metal",
        :reactant_cpu=>"Reactant CPU", :reactant_gpu=>"Reactant GPU")
    backend in first.(backends) || throw(ArgumentError("unknown validation backend: $backend"))
    io = IOBuffer()
    println(io, "| Method | Checked example | Precision | ", join(last.(backends), " | "), " |")
    println(io, "|:--|:--|:--|:--|:--|:--|:--|:--|")
    for row in cases
        status = check_support_case(row; backend, kwargs...)
        cells = map(backends) do (key, _)
            key === backend ? string(status) : haskey(row.limits, key) ? "restricted" : "not run"
        end
        println(io, "| $(row.name) | $(row.example) | $(row.precision) | ", join(cells, " | "), " |")
    end
    return String(take!(io))
end

end
