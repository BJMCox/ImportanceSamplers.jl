using BenchmarkTools
using CUDA
using DensityInterface
using ImportanceSamplers
using MLDataDevices
using Random

const STATIC_MIS_SMOKE = "--smoke" in ARGS
const STATIC_MIS_BENCHMARK_SEED = 0x737461746963626d
const STATIC_MIS_BENCHMARK_SAMPLES = STATIC_MIS_SMOKE ? 3 : 30
const STATIC_MIS_BENCHMARK_SECONDS = STATIC_MIS_SMOKE ? 0.2 : 5.0
const STATIC_MIS_CPU_CASES = STATIC_MIS_SMOKE ?
    ((nsamples=256, dimension=2, proposal_count=4),) :
    Tuple(
        (; nsamples, dimension, proposal_count) for
        nsamples in (10_000, 100_000) for
        dimension in (1, 4, 16) for
        proposal_count in (2, 8, 32)
    )
const STATIC_MIS_CUDA_TRANSFER_CASE = STATIC_MIS_SMOKE ?
    first(STATIC_MIS_CPU_CASES) :
    (nsamples=10_000, dimension=4, proposal_count=8)

struct BenchmarkGaussian{T<:AbstractFloat}
    location::Vector{T}
    scale::T
    density_evaluations::Base.RefValue{Int}
end

function Random.rand(rng::Random.AbstractRNG, proposal::BenchmarkGaussian{T}) where {T}
    return proposal.location .+ proposal.scale .* randn(rng, T, length(proposal.location))
end

function DensityInterface.logdensityof(proposal::BenchmarkGaussian, sample)
    proposal.density_evaluations[] += 1
    dimension = length(proposal.location)
    value = -dimension * (log(proposal.scale) + 0.5 * log(2pi))
    for coordinate in eachindex(sample, proposal.location)
        value -= 0.5 * abs2(
            (sample[coordinate] - proposal.location[coordinate]) / proposal.scale,
        )
    end
    return value
end

function benchmark_target(sample)::Float64
    value = -0.5 * length(sample) * log(2pi)
    for coordinate in eachindex(sample)
        value -= 0.5 * abs2(sample[coordinate])
    end
    return value
end

scheme_cases(proposal_count) = (
    (label=:stratified_mixture, value=StratifiedMixture()),
    (label=:random_mixture, value=RandomMixture()),
    (label=:standard_mis, value=StandardMIS()),
    (
        label=:partial_deterministic_mixture,
        value=PartialDeterministicMixture((
            Tuple(1:2:proposal_count),
            Tuple(2:2:proposal_count),
        )),
    ),
)

function benchmark_banks(proposal_count, dimension)
    locations = [
        fill(0.25 * (proposal - (proposal_count + 1) / 2), dimension) for
        proposal in 1:proposal_count
    ]
    scales = [0.7 + 0.05 * mod(proposal, 4) for proposal in 1:proposal_count]
    masses = Float64[mod(proposal, 5) + 1 for proposal in 1:proposal_count]
    generic = ProposalBank([
        BenchmarkGaussian(locations[proposal], scales[proposal], Ref(0)) for
        proposal in 1:proposal_count
    ], masses)
    packed = ProposalBank([
        SphericalGaussian(locations[proposal], scales[proposal]) for
        proposal in 1:proposal_count
    ], masses)
    return (; generic, packed)
end

function reset_density_evaluations!(bank)
    for proposal in bank.proposals
        proposal.density_evaluations[] = 0
    end
    return bank
end

function run_trial(benchmark)
    return run(
        benchmark;
        samples=STATIC_MIS_BENCHMARK_SAMPLES,
        seconds=STATIC_MIS_BENCHMARK_SECONDS,
        evals=1,
    )
end

function trial_record(trial; nsamples=nothing)
    estimate = median(trial)
    throughput = isnothing(nsamples) ? nothing : nsamples / (estimate.time / 1.0e9)
    return (
        time_ns=estimate.time,
        allocations=estimate.allocs,
        bytes=estimate.memory,
        samples_per_second=throughput,
    )
end

function expected_density_evaluations(scheme, result, proposal_count)
    nsamples = length(result.logweights)
    scheme isa StandardMIS && return nsamples
    scheme isa PartialDeterministicMixture || return nsamples * proposal_count

    group_sizes = Dict(
        proposal_id => length(group) for group in scheme.groups for
        proposal_id in group
    )
    return sum(group_sizes[proposal_id] for proposal_id in result.provenance.proposal_id)
end

function benchmark_cpu_case(
    bank,
    bank_kind,
    scheme_case,
    nsamples,
    dimension,
    proposal_count,
)
    algorithm = ImportanceSampling(
        bank;
        nsamples,
        mis_scheme=scheme_case.value,
    )
    preparation_rng = Xoshiro(STATIC_MIS_BENCHMARK_SEED)
    preparation = @benchmarkable prepare_sampler(
        $preparation_rng,
        benchmark_target,
        $algorithm;
        threaded=false,
    )
    preparation_record = trial_record(run_trial(preparation))

    sampler = prepare_sampler(
        Xoshiro(STATIC_MIS_BENCHMARK_SEED),
        benchmark_target,
        algorithm;
        threaded=false,
    )
    importance_sample!(sampler)
    execution = @benchmarkable importance_sample!($sampler)
    execution_record = trial_record(
        run_trial(execution);
        nsamples,
    )

    summary_sampler = prepare_sampler(
        Xoshiro(STATIC_MIS_BENCHMARK_SEED),
        benchmark_target,
        algorithm;
        threaded=false,
    )
    bank_kind === :generic && reset_density_evaluations!(bank)
    result = importance_sample!(summary_sampler)
    observed_density_evaluations = nothing
    if bank_kind === :generic
        observed_density_evaluations = sum(
            proposal.density_evaluations[] for proposal in bank.proposals
        )
    end
    weights = normalized_weights(result)
    effective_sample_size = inv(sum(abs2, weights))
    return (
        nsamples,
        dimension,
        proposal_count,
        bank=bank_kind,
        scheme=scheme_case.label,
        preparation=preparation_record,
        execution=execution_record,
        expected_proposal_density_evaluations=expected_density_evaluations(
            scheme_case.value,
            result,
            proposal_count,
        ),
        observed_proposal_density_evaluations=observed_density_evaluations,
        effective_sample_size,
        ess_fraction=effective_sample_size / nsamples,
        effective_samples_per_second=(
            effective_sample_size / (execution_record.time_ns / 1.0e9)
        ),
    )
end

function cuda_result_transfer_record(bank, nsamples)
    CUDA.functional() || return nothing
    CUDA.allowscalar(false)
    physical = CUDA.device()
    device = MLDataDevices.CUDADevice{typeof(physical),Nothing}(physical)
    sampler = prepare_sampler(
        Xoshiro(STATIC_MIS_BENCHMARK_SEED),
        benchmark_target,
        ImportanceSampling(
            bank;
            nsamples,
            mis_scheme=StratifiedMixture(),
        );
        threaded=true,
    ) |> device
    result = importance_sample!(sampler)
    CUDA.synchronize()
    copy_result(value) = begin
        copied = MLDataDevices.cpu_device()(value)
        CUDA.synchronize()
        copied
    end
    copy_result(result)
    transfer = @benchmarkable copy_result($result)
    payload_bytes = sizeof(result.samples) + sizeof(result.logweights) +
                    sizeof(result.provenance.proposal_id)
    return merge(trial_record(run_trial(transfer)), (; payload_bytes))
end

function main()
    cpu = [
        benchmark_cpu_case(
            bank,
            bank_kind,
            scheme_case,
            case.nsamples,
            case.dimension,
            case.proposal_count,
        ) for
        case in STATIC_MIS_CPU_CASES for
        (bank_kind, bank) in pairs(
            benchmark_banks(case.proposal_count, case.dimension),
        ) for
        scheme_case in scheme_cases(case.proposal_count)
    ]
    transfer_case = STATIC_MIS_CUDA_TRANSFER_CASE
    transfer_banks = benchmark_banks(
        transfer_case.proposal_count,
        transfer_case.dimension,
    )
    return (
        command="julia --project=benchmark benchmark/static_mis.jl" *
                (STATIC_MIS_SMOKE ? " --smoke" : ""),
        julia=VERSION,
        smoke=STATIC_MIS_SMOKE,
        cpu,
        cuda_result_transfer=cuda_result_transfer_record(
            transfer_banks.packed,
            transfer_case.nsamples,
        ),
    )
end

result = main()
show(stdout, MIME("text/plain"), result)
println()
result
