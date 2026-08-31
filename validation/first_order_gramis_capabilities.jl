using CUDA
using ImportanceSamplers
using LinearAlgebra
using MLDataDevices
using Pkg
using Random
using Test

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

function gram_is_cuda_population_cholesky(::Type{T}, dimension, proposal_count) where {T}
    source = gram_is_covariance_batch(T, dimension, proposal_count)
    covariances = CuArray(source)
    factors = similar(covariances)
    info = CUDA.fill(Int32(-1), proposal_count)
    status = CUDA.fill(IS._GRAMIS_COVARIANCE_READY, proposal_count)
    device = gram_is_cuda_device()

    IS._factor_population!(device, factors, covariances, info, status)
    CUDA.synchronize()

    wrapper_factors = [CuArray(source[:, :, slot]) for slot in 1:proposal_count]
    wrapper_factors, _ = CUDA.cuSOLVER.potrfBatched!(
        'L',
        wrapper_factors,
    )
    CUDA.synchronize()
    wrapper_lower = cat(
        (tril(Array(factor)) for factor in wrapper_factors)...;
        dims=3,
    )
    return (
        source,
        factors=Array(factors),
        info=Array(info),
        wrapper_factors=wrapper_lower,
    )
end

function gram_is_cuda_local_weights(
    ::Type{T},
    local_logweights;
    max_iterations=64,
) where {T}
    sample_count = 4
    proposal_count = length(local_logweights) ÷ sample_count
    counts = fill(sample_count, proposal_count, 1)
    starts = IS._first_order_gramis_group_starts(counts)
    normalized_weights = CUDA.fill(T(NaN), length(local_logweights))
    local_ess = CUDA.fill(T(NaN), proposal_count)
    tempering_powers = CUDA.fill(T(NaN), proposal_count)
    status = CUDA.fill(UInt8(0xff), proposal_count)
    device_logweights = CuArray(local_logweights)
    device_starts = CuArray(starts)
    device_counts = CuArray(counts)
    thresholds = CUDA.fill(3, proposal_count, 1)
    backend = IS.KernelAbstractions.get_backend(normalized_weights)
    kernel = IS._cooperative_local_weights_kernel!(
        backend,
        IS._GRAMIS_REDUCTION_WORKGROUP_SIZE,
    )
    kernel(
        normalized_weights,
        local_ess,
        tempering_powers,
        status,
        device_logweights,
        device_starts,
        device_counts,
        thresholds,
        1,
        T(1.0e-6),
        max_iterations;
        ndrange=IS._GRAMIS_REDUCTION_WORKGROUP_SIZE * proposal_count,
        workgroupsize=IS._GRAMIS_REDUCTION_WORKGROUP_SIZE,
    )
    CUDA.synchronize()
    return (
        normalized_weights=Array(normalized_weights),
        local_ess=Array(local_ess),
        tempering_powers=Array(tempering_powers),
        status=Array(status),
    )
end


function gram_is_cuda_preflight(::Type{T}) where {T}
    bank = ProposalBank([
        FactorGaussian(T[-1, 0], T[1 0; 0.1 0.8]),
        FactorGaussian(T[1, 0], T[0.9 0; -0.2 1.1]),
    ])
    source = prepare_sampler(
        Random.Xoshiro(GRAMIS_CUDA_VALIDATION_SEED),
        LogTarget(
            CUDAFirstOrderGRAMISTarget{T}();
            grad=CUDAFirstOrderGRAMISGradient(),
        ),
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
    bank = ProposalBank([
        FactorGaussian(T[-1, 0], T[1 0; 0.1 0.8]),
        FactorGaussian(T[1, 0], T[0.9 0; -0.2 1.1]),
    ])
    gradient = round_two_failure ?
               CUDAFirstOrderGRAMISRoundTwoFailureGradient{T}((
                   -one(T),
                   one(T),
               )) : CUDAFirstOrderGRAMISGradient()
    target = fallback ?
             LogTarget(
        CUDAFirstOrderGRAMISMeanOnlyTarget{T}((-one(T), one(T)));
        grad=CUDAFirstOrderGRAMISZeroGradient(),
    ) : LogTarget(
        CUDAFirstOrderGRAMISTarget{T}();
        grad=gradient,
    )
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

@testset "FirstOrderGRAMIS CUDA population Cholesky capabilities" begin
    caller_device = CUDA.device()
    for T in (Float32, Float64), dimension in (
        2,
        IS._GRAMIS_CHOLESKY_WORKGROUP_SIZE + 3,
    )
        result = gram_is_cuda_population_cholesky(T, dimension, 3)
        @test result.info == zeros(Int32, 3)
        @test result.factors ≈ result.wrapper_factors rtol =
            T === Float32 ? GRAMIS_CUDA_CHOLESKY_RTOL.Float32 :
            GRAMIS_CUDA_CHOLESKY_RTOL.Float64
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


@testset "FirstOrderGRAMIS CUDA cooperative local weights" begin
    caller_device = CUDA.device()
    for T in (Float32, Float64)
        raw_ready = T[0, 0, 0, 0]
        active = T[0, -4, -8, -12]
        all_zero = fill(T(-Inf), 4)
        result = gram_is_cuda_local_weights(
            T,
            vcat(raw_ready, active, all_zero),
        )

        @test result.status == UInt8[
            IS._GRAMIS_COVARIANCE_READY,
            IS._GRAMIS_COVARIANCE_READY,
            IS._GRAMIS_ALL_ZERO_LOCAL,
        ]
        @test result.tempering_powers[1] == one(T)
        @test zero(T) < result.tempering_powers[2] < one(T)
        @test result.tempering_powers[3] == zero(T)
        @test sum(result.normalized_weights[1:4]) ≈ one(T) rtol = T(4.0e-6)
        @test sum(result.normalized_weights[5:8]) ≈ one(T) rtol = T(4.0e-6)
        @test result.local_ess[1] == T(4)
        @test result.local_ess[2] >= T(3)
        @test result.local_ess[3] == zero(T)

        fallback = gram_is_cuda_local_weights(T, active; max_iterations=1)
        @test fallback.status == UInt8[IS._GRAMIS_TEMPERING_FALLBACK]
        @test fallback.tempering_powers == zeros(T, 1)
        @test sum(fallback.normalized_weights) ≈ one(T) rtol = T(4.0e-6)
    end
    @test CUDA.device() == caller_device
end


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
    for (T, strength) in ((Float32, 0f0), (Float64, 0.1))
        sampler = gram_is_cuda_sampler(T; repulsion_strength=strength)
        result = importance_sample!(sampler)
        @test result.diagnostics.transfers.count <= 192
        @test result.diagnostics.transfers.bytes <= 4_096
        if !iszero(strength)
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
    end
    @test CUDA.device() == caller_device
end

@testset "FirstOrderGRAMIS CUDA fallback preserves factors" begin
    caller_device = CUDA.device()
    for T in (Float32, Float64)
        sampler = gram_is_cuda_sampler(T; fallback=true)
        before_factors = map(
            bitstring,
            vec(Array(sampler.method_state.committed.factors)),
        )
        result = importance_sample!(sampler)
        @test all(==(-Inf), Array(result.logweights))
        @test all(
            ==(IS._GRAMIS_ALL_ZERO_LOCAL),
            Array(result.diagnostics.fallback_status),
        )
        @test map(
            bitstring,
            vec(Array(sampler.method_state.committed.factors)),
        ) == before_factors
        @test result.diagnostics.transfers.count <= 192
        @test result.diagnostics.transfers.bytes <= 4_096
    end
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
        scalar_types=(Float32, Float64),
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
