using CUDA
using ImportanceSamplers
using LinearAlgebra
using MLDataDevices
using Pkg
using Random
using Test

include(joinpath(@__DIR__, "first_order_gramis_capability_contract.jl"))

const IS = ImportanceSamplers
const GRAMIS_CUDA_VALIDATION_SEED = 0x4752414d49534355
const GRAMIS_CUDA_EXECUTION_SEED = 0x4752414d49534532
const GRAMIS_CUDA_REFERENCE_ROUND_SIZE = 65_536
const GRAMIS_CUDA_REFERENCE_MEAN_ATOL = 0.035
const GRAMIS_CUDA_REFERENCE_COVARIANCE_ATOL = 0.055
const GRAMIS_CUDA_CHOLESKY_RTOL = (Float32=3f-4, Float64=3e-12)

function gram_is_validation_source_provenance(; required=false)
    value = get(ENV, "IMPORTANCE_SAMPLERS_SOURCE_COMMIT", "")
    supplied = !isempty(value)
    required && !supplied && error(
        "CUDA validation requires IMPORTANCE_SAMPLERS_SOURCE_COMMIT",
    )
    supplied && !occursin(r"^[0-9a-f]{40}$", value) && error(
        "IMPORTANCE_SAMPLERS_SOURCE_COMMIT must be 40 lowercase hex characters",
    )
    return (
        environment_variable="IMPORTANCE_SAMPLERS_SOURCE_COMMIT",
        commit=supplied ? value : nothing,
        supplied,
    )
end

const GRAMIS_CUDA_SOURCE = gram_is_validation_source_provenance(required=true)
const GRAMIS_CAPABILITY_ROWS = checked_first_order_gramis_capability_rows()

struct CUDAFirstOrderGRAMISTarget{T} end
struct CUDAFirstOrderGRAMISGradient end
struct CUDAFirstOrderGRAMISRoundTwoFailureGradient{T}
    locations::NTuple{2,T}
end
struct CUDAFirstOrderGRAMISMeanOnlyTarget{T}
    locations::NTuple{2,T}
end
struct CUDAFirstOrderGRAMISZeroGradient end

function (::CUDAFirstOrderGRAMISTarget{T})(sample)::T where {T}
    value = zero(T)
    @inbounds for row in 1:length(sample)
        value += abs2(sample[row])
    end
    return -T(0.5) * value
end

function (::CUDAFirstOrderGRAMISGradient)(destination, sample)
    @inbounds for row in 1:length(destination)
        destination[row] = -sample[row]
    end
    return destination
end

function (gradient::CUDAFirstOrderGRAMISRoundTwoFailureGradient{T})(
    destination,
    sample,
) where {T}
    initial = (sample[1] == gradient.locations[1] ||
               sample[1] == gradient.locations[2]) && iszero(sample[2])
    @inbounds for row in eachindex(destination)
        destination[row] = initial ? -sample[row] : T(NaN)
    end
    return destination
end

function (target::CUDAFirstOrderGRAMISMeanOnlyTarget{T})(sample)::T where {T}
    at_left = sample[1] == target.locations[1] && iszero(sample[2])
    at_right = sample[1] == target.locations[2] && iszero(sample[2])
    return at_left || at_right ? zero(T) : T(-Inf)
end

function (::CUDAFirstOrderGRAMISZeroGradient)(destination, sample)
    @inbounds for row in 1:length(destination)
        destination[row] = zero(eltype(destination))
    end
    return destination
end

function gram_is_cuda_device(physical=CUDA.device())
    return MLDataDevices.CUDADevice{typeof(physical),Nothing}(physical)
end

function gram_is_covariance_batch(::Type{T}, dimension, proposal_count) where {T}
    covariances = Array{T}(undef, dimension, dimension, proposal_count)
    for proposal_slot in 1:proposal_count
        seed = reshape(
            T.(1:(dimension * dimension)),
            dimension,
            dimension,
        ) / T(dimension + proposal_slot + 3)
        covariances[:, :, proposal_slot] .=
            seed * transpose(seed) + T(dimension + proposal_slot) * I
    end
    return covariances
end

function gram_is_cuda_preflight(::Type{T}) where {T}
    row = only(filter(
        candidate -> candidate.type === T &&
                     candidate.gradient === :explicit_inplace,
        GRAMIS_CAPABILITY_ROWS,
    ))
    bank = first_order_gramis_capability_bank(T)
    source = prepare_sampler(
        Random.Xoshiro(GRAMIS_CUDA_VALIDATION_SEED),
        first_order_gramis_capability_target(row),
        FirstOrderGRAMIS(
            bank;
            rounds=1,
            round_size=8,
            repulsion_strength=T(0.1),
        );
        threaded=true,
    )
    prepared = gram_is_cuda_device()(source)
    return (
        prepared,
        resident=IS._backend_state_resident(
            prepared.device,
            IS._transferred_backend_state(
                prepared.algorithm,
                prepared.method_state,
                prepared.target,
                prepared.random_buffers,
            ),
        ),
    )
end

function gram_is_cuda_sampler(
    ::Type{T};
    repulsion_strength=zero(T),
    fallback=false,
    rounds=2,
    round_size=8,
    device=:cuda,
    round_two_failure=false,
) where {T}
    bank = first_order_gramis_capability_bank(T)
    row = only(filter(
        candidate -> candidate.type === T &&
                     candidate.gradient === :explicit_inplace,
        GRAMIS_CAPABILITY_ROWS,
    ))
    gradient = round_two_failure ?
               CUDAFirstOrderGRAMISRoundTwoFailureGradient{T}((
                   -one(T),
                   one(T),
               )) : CUDAFirstOrderGRAMISGradient()
    target = fallback ?
             LogTarget(
        CUDAFirstOrderGRAMISMeanOnlyTarget{T}((-one(T), one(T)));
        grad=CUDAFirstOrderGRAMISZeroGradient(),
    ) : round_two_failure ? LogTarget(
        CUDAFirstOrderGRAMISTarget{T}();
        grad=gradient,
    ) : first_order_gramis_capability_target(row)
    source = prepare_sampler(
        Random.Xoshiro(GRAMIS_CUDA_EXECUTION_SEED),
        target,
        FirstOrderGRAMIS(
            bank;
            rounds,
            round_size,
            repulsion_strength=repulsion_strength,
        );
        threaded=true,
    )
    if device === :cpu
        return source
    elseif device === :cuda
        return gram_is_cuda_device()(source)
    end
    error("unknown FirstOrderGRAMIS validation device $device")
end

function gram_is_cuda_population_bits(sampler)
    state = sampler.method_state.committed
    return (
        locations=map(bitstring, vec(Array(state.locations))),
        factors=map(bitstring, vec(Array(state.factors))),
        lognormalizers=map(bitstring, Array(state.lognormalizers)),
    )
end

function gram_is_host_weighted_moments(result)
    samples = Array(result.samples)
    logweights = Array(result.logweights)
    maximum_logweight = maximum(logweights)
    weights = exp.(logweights .- maximum_logweight)
    weights ./= sum(weights)
    mean = samples * weights
    centered = samples .- mean
    covariance = (centered .* reshape(weights, 1, :)) * transpose(centered)
    return (; mean, covariance)
end

function gram_is_package_versions()
    wanted = Set((
        "Adapt",
        "CUDA",
        "ImportanceSamplers",
        "KernelAbstractions",
        "MLDataDevices",
    ))
    return sort!(
        [
            (dependency.name, something(dependency.version, "unversioned")) for
            dependency in values(Pkg.dependencies()) if dependency.name in wanted
        ];
        by=first,
    )
end

function gram_is_cuda_validation_environment()
    device = CUDA.device()
    return (
        gpu=CUDA.name(device),
        capability=CUDA.capability(device),
        driver=CUDA.driver_version(),
        runtime=CUDA.runtime_version(),
        julia=VERSION,
        packages=gram_is_package_versions(),
        source=GRAMIS_CUDA_SOURCE,
        seeds=(
            preflight=GRAMIS_CUDA_VALIDATION_SEED,
            execution=GRAMIS_CUDA_EXECUTION_SEED,
        ),
        reference_round_size=GRAMIS_CUDA_REFERENCE_ROUND_SIZE,
        reference_mean_atol=GRAMIS_CUDA_REFERENCE_MEAN_ATOL,
        reference_covariance_atol=GRAMIS_CUDA_REFERENCE_COVARIANCE_ATOL,
        cholesky_rtol=GRAMIS_CUDA_CHOLESKY_RTOL,
        scalar_indexing_allowed=false,
    )
end

function gram_is_multi_device_restoration_test()
    devices = collect(CUDA.devices())
    length(devices) >= 2 || return (
        available=false,
        passed=nothing,
        device_count=length(devices),
    )
    caller = CUDA.device()
    requested = first(device for device in devices if device != caller)
    source = gram_is_cuda_sampler(Float32; round_size=8, device=:cpu)
    transfer_observed_device = caller
    sampler = try
        gram_is_cuda_device(requested)(source)
    finally
        transfer_observed_device = CUDA.device()
        CUDA.device!(caller)
    end
    @test transfer_observed_device == caller
    execution_observed_device = caller
    result = try
        importance_sample!(sampler)
    finally
        execution_observed_device = CUDA.device()
        CUDA.device!(caller)
    end
    @test execution_observed_device == caller
    @test IS._backend_state_resident(
        sampler.device,
        IS._transferred_backend_state(
            sampler.algorithm,
            sampler.method_state,
            sampler.target,
            sampler.random_buffers,
        ),
    )
    return (
        available=true,
        passed=true,
        device_count=length(devices),
        caller=string(caller),
        requested=string(requested),
        result_samples=length(result),
    )
end

CUDA.allowscalar(false)

@testset "FirstOrderGRAMIS shared capability contract" begin
    caller_device = CUDA.device()
    @test CUDA.name(caller_device) == FIRST_ORDER_GRAMIS_CUDA_HARDWARE
    @test Tuple(
        (row.type, row.gradient) for row in GRAMIS_CAPABILITY_ROWS
        if row.cuda.status === :supported
    ) == ((Float32, :explicit_inplace), (Float64, :explicit_inplace))
    for (row_index, row) in enumerate(GRAMIS_CAPABILITY_ROWS)
        row.cpu.status === :supported || continue
        row.cuda.status === :rejected || continue
        source = prepare_sampler(
            Random.Xoshiro(GRAMIS_CUDA_VALIDATION_SEED + row_index),
            first_order_gramis_capability_target(row),
            FirstOrderGRAMIS(
                first_order_gramis_capability_bank(row.type);
                rounds=1,
                round_size=8,
                repulsion_strength=zero(row.type),
            );
            threaded=true,
        )
        rejection = try
            gram_is_cuda_device()(source)
            nothing
        catch error
            error
        end
        @test rejection isa SamplerDeviceError
        @test rejection.reason === row.cuda.reason
    end
    @test CUDA.device() == caller_device
end

@testset "FirstOrderGRAMIS CUDA agrees with CPU reference moments" begin
    caller_device = CUDA.device()
    for T in (Float32, Float64)
        cpu_sampler = gram_is_cuda_sampler(
            T;
            round_size=GRAMIS_CUDA_REFERENCE_ROUND_SIZE,
            device=:cpu,
        )
        cuda_sampler = gram_is_cuda_sampler(
            T;
            round_size=GRAMIS_CUDA_REFERENCE_ROUND_SIZE,
        )
        cpu = gram_is_host_weighted_moments(importance_sample!(cpu_sampler))
        gpu_result = importance_sample!(cuda_sampler)
        gpu = gram_is_host_weighted_moments(gpu_result)
        mean_atol = T(GRAMIS_CUDA_REFERENCE_MEAN_ATOL)
        covariance_atol = T(GRAMIS_CUDA_REFERENCE_COVARIANCE_ATOL)
        @test gpu.mean ≈ cpu.mean atol = mean_atol rtol = zero(T)
        @test gpu.covariance ≈ cpu.covariance atol = covariance_atol rtol = zero(T)
        @test gpu.mean ≈ zeros(T, 2) atol = mean_atol rtol = zero(T)
        @test gpu.covariance ≈ Matrix{T}(I, 2, 2) atol = covariance_atol rtol = zero(T)
        @test IS._backend_state_resident(
            cuda_sampler.device,
            IS._transferred_backend_state(
                cuda_sampler.algorithm,
                cuda_sampler.method_state,
                cuda_sampler.target,
                cuda_sampler.random_buffers,
            ),
        )
        @test gpu_result.diagnostics.transfers.count <= 192
        @test gpu_result.diagnostics.transfers.bytes <= 4_096
    end
    @test CUDA.device() == caller_device
end

@testset "FirstOrderGRAMIS CUDA repeated calls execute on resident state" begin
    caller_device = CUDA.device()
    for T in (Float32, Float64)
        sampler = gram_is_cuda_sampler(T; round_size=256)
        initial = gram_is_cuda_population_bits(sampler)
        first_result = importance_sample!(sampler)
        first = gram_is_cuda_population_bits(sampler)
        second_result = importance_sample!(sampler)
        second = gram_is_cuda_population_bits(sampler)
        @test first != initial
        @test second != first
        @test length(first_result) == length(second_result) == 512
        @test extrema(Array(first_result.provenance.round)) == (1, 2)
        @test extrema(Array(second_result.provenance.round)) == (1, 2)
        @test first_result.diagnostics.transfers.count ==
              second_result.diagnostics.transfers.count
        @test first_result.diagnostics.transfers.bytes ==
              second_result.diagnostics.transfers.bytes
        @test IS._backend_state_resident(
            sampler.device,
            IS._transferred_backend_state(
                sampler.algorithm,
                sampler.method_state,
                sampler.target,
                sampler.random_buffers,
            ),
        )
    end
    @test CUDA.device() == caller_device
end

const GRAMIS_CUDA_MULTI_DEVICE_RESTORATION =
    gram_is_multi_device_restoration_test()


@testset "FirstOrderGRAMIS CUDA preflight and pooled adapters" begin
    caller_device = CUDA.device()
    for T in (Float32, Float64)
        preflight = gram_is_cuda_preflight(T)
        @test preflight.resident

        source = gram_is_covariance_batch(T, 3, 3)
        factors = cat(
            (
                Matrix(cholesky(Hermitian(source[:, :, slot])).L)
                for slot in 1:3
            )...;
            dims=3,
        )
        device_factors = CuArray(factors)
        pooled = CUDA.zeros(T, 3, 3)
        execution = IS._KernelExecution(IS._ThreadedCPUExecution())
        IS._pooled_covariance!(pooled, device_factors, execution)
        pooled_reference = dropdims(sum(source; dims=3); dims=3) / T(3)
        @test Array(pooled) ≈ pooled_reference rtol =
            T === Float32 ? 4f-5 : 4e-13

        IS._factor_pooled_covariance!(pooled)
        means = CuArray(reshape(T.(1:12), 3, 4) / T(7))
        whitened = similar(means)
        IS._whiten_means!(whitened, pooled, means)
        expected = cholesky(Hermitian(pooled_reference)).L \ Array(means)
        @test Array(whitened) ≈ expected rtol =
            T === Float32 ? 5f-5 : 5e-13
    end
    @test CUDA.device() == caller_device
end
@testset "FirstOrderGRAMIS CUDA end-to-end sampler" begin
    caller_device = CUDA.device()
    for row in GRAMIS_CAPABILITY_ROWS
        row.cuda.status === :supported || continue
        T = row.type
        strength = T === Float32 ? zero(T) : T(0.1)
        sampler = gram_is_cuda_sampler(T; repulsion_strength=strength)
        result = importance_sample!(sampler)
        @test eltype(result.samples) === T
        @test eltype(result.logweights) === T
        @test result.diagnostics.transfers.count <= 192
        @test result.diagnostics.transfers.bytes <= 4_096
        @test IS._backend_state_resident(
            sampler.device,
            IS._transferred_backend_state(
                sampler.algorithm,
                sampler.method_state,
                sampler.target,
                sampler.random_buffers,
            ),
        )
    end
    @test CUDA.device() == caller_device
end

@testset "FirstOrderGRAMIS CUDA fallback preserves factors" begin
    caller_device = CUDA.device()
    sampler = gram_is_cuda_sampler(Float64; fallback=true)
    before = current_proposal(cpu_device(), sampler)
    result = importance_sample!(sampler)
    after = current_proposal(cpu_device(), sampler)

    @test all(==(-Inf), Array(result.logweights))
    @test [proposal.scale.factor for proposal in after.proposals] ==
          [proposal.scale.factor for proposal in before.proposals]
    @test CUDA.device() == caller_device
end

@testset "FirstOrderGRAMIS CUDA call transaction rolls back after round one" begin
    caller_device = CUDA.device()
    sampler = gram_is_cuda_sampler(
        Float64;
        rounds=2,
        round_size=8,
        round_two_failure=true,
    )
    before = gram_is_cuda_population_bits(sampler)
    failure = try
        importance_sample!(sampler)
        nothing
    catch error
        error
    end
    @test failure isa FirstOrderGRAMISRoundError
    if failure isa FirstOrderGRAMISRoundError
        @test failure.round == 2
        @test failure.phase === :derivative
        @test failure.diagnostics.completed_rounds == 1
        @test failure.diagnostics.pre_call_state_preserved === true
    end
    @test gram_is_cuda_population_bits(sampler) == before
    @test CUDA.device() == caller_device
end

@testset "FirstOrderGRAMIS CUDA covariance failure rolls back" begin
    caller_device = CUDA.device()
    sampler = gram_is_cuda_sampler(Float64)
    fill!(sampler.method_state.covariance_rate, NaN)
    before = gram_is_cuda_population_bits(sampler)
    failure = try
        importance_sample!(sampler)
        nothing
    catch error
        error
    end
    @test failure isa FirstOrderGRAMISRoundError
    @test failure.phase === :covariance
    @test failure.diagnostics.transfers.count <= 192
    @test failure.diagnostics.transfers.bytes <= 4_096
    @test gram_is_cuda_population_bits(sampler) == before
    @test CUDA.device() == caller_device
end

@testset "FirstOrderGRAMIS CUDA repulsion failures roll back" begin
    caller_device = CUDA.device()
    for (T, case) in ((Float32, :pooled_covariance), (Float64, :force))
        sampler = gram_is_cuda_sampler(
            T;
            repulsion_strength=T(0.1),
            fallback=case === :pooled_covariance,
        )
        if case === :pooled_covariance
            fill!(sampler.method_state.committed.factors, sqrt(floatmax(T)))
        else
            fill!(sampler.method_state.repulsion_strength, T(Inf))
        end
        before = gram_is_cuda_population_bits(sampler)
        failure = try
            importance_sample!(sampler)
            nothing
        catch error
            error
        end
        @test failure isa FirstOrderGRAMISRoundError
        @test failure.phase === :repulsion
        @test failure.cause.reason === (
            case === :pooled_covariance ?
            :pooled_covariance_nonfinite : :force_nonfinite
        )
        @test failure.diagnostics.transfers.count <= 192
        @test failure.diagnostics.transfers.bytes <= 4_096
        @test gram_is_cuda_population_bits(sampler) == before
    end
    @test CUDA.device() == caller_device
end

const GRAMIS_CUDA_CAPABILITY_RESULT = (
    status=:passed,
    environment=gram_is_cuda_validation_environment(),
    coverage=(
        scalar_types=Tuple(
            row.type for row in GRAMIS_CAPABILITY_ROWS
            if row.cuda.status === :supported
        ),
        capability_rows=Tuple((
            label=row.label,
            type=row.type,
            gradient=row.gradient,
            cuda_status=row.cuda.status,
            cuda_reason=row.cuda.reason,
            hardware=row.cuda.hardware,
            context=row.cuda.context,
            evidence=row.cuda.evidence,
        ) for row in GRAMIS_CAPABILITY_ROWS),
        residence=true,
        cpu_reference_agreement=true,
        all_zero_fallback=true,
        call_transaction_rollback=true,
        repeated_resident_execution=true,
        caller_device_restoration=(
            current_device_checks=true,
            multi_device=GRAMIS_CUDA_MULTI_DEVICE_RESTORATION,
        ),
        bounded_explicit_transfers=true,
        scalar_indexing_disabled=true,
    ),
)
