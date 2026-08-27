using CUDA
using ImportanceSamplers
using LinearAlgebra
using MLDataDevices
using Pkg
using Random
using Test

include(joinpath(@__DIR__, "..", "cuda_plain_is_support.jl"))

const IS = ImportanceSamplers
const AMIS_CUDA_SEED = 0x616d697363756461
const AMIS_CUDA_HARDWARE = "NVIDIA A100-PCIE-40GB"

struct AMISQuadraticTarget{T} end

@inline function (::AMISQuadraticTarget{T})(sample::Real) where {T}
    centered = sample - T(0.75)
    return -T(0.5) * abs2(centered)
end

@inline function (::AMISQuadraticTarget{T})(sample::AbstractVector) where {T}
    value = zero(T)
    @inbounds for coordinate in eachindex(sample)
        centered = sample[coordinate] - T(0.25) * T(coordinate)
        value += abs2(centered)
    end
    return -T(0.5) * value
end

amis_proposal(::Type{T}, ::Val{:scalar}) where {T} =
    SphericalGaussian(T(-1), T(1.5))

amis_proposal(::Type{T}, ::Val{:factor}) where {T} = FactorGaussian(
    T[-1, 0.5, 1.25],
    T[1.2 0 0; -0.2 0.9 0; 0.15 0.25 1.1],
)

function reported_transfer_record(transfers)
    reason_names = fieldnames(typeof(transfers.reasons))
    reason_values = map(reason_names) do reason
        record = getfield(transfers.reasons, reason)
        (count=record.count, bytes=record.bytes)
    end
    reasons = NamedTuple{reason_names}(reason_values)
    @test sum(record.count for record in values(reasons)) == transfers.count
    @test sum(record.bytes for record in values(reasons)) == transfers.bytes
    return (count=transfers.count, bytes=transfers.bytes, reasons)
end

function assert_amis_transfers(transfers, rounds, ::Type{T}) where {T}
    record = reported_transfer_record(transfers)
    @test record.count == 3rounds
    @test record.bytes == rounds * (3sizeof(UInt64) + 2sizeof(T))
    @test record.reasons.failure_snapshot == (
        count=rounds,
        bytes=rounds * 3sizeof(UInt64),
    )
    @test record.reasons.summary_maximum == (
        count=rounds,
        bytes=rounds * sizeof(T),
    )
    @test record.reasons.summary_scaled_sum == (
        count=rounds,
        bytes=rounds * sizeof(T),
    )
    for reason in (
        :cdf_maximum,
        :cdf_sum,
        :summary_scaled_square_sum,
    )
        @test getfield(record.reasons, reason) == (count=0, bytes=0)
    end
    return record
end

function assert_amis_residence(prepared, result)
    state = prepared.method_state
    history = state.history
    workspace = state.workspace
    buffers = prepared.random_buffers
    scale_storage = history isa IS._AMISScalarHistory ?
                    history.scales : history.factors
    arrays = (
        state.logcounts,
        history.means,
        scale_storage,
        history.lognormalizers,
        workspace.samples,
        workspace.logtargets,
        workspace.lognumerators,
        workspace.logweights,
        workspace.normalized_weights,
        workspace.centered_scaled,
        workspace.covariance,
        workspace.candidate_mean,
        workspace.candidate_scale,
        workspace.candidate_lognormalizer,
        buffers.uniform,
        buffers.normal,
        buffers.failure_scratch.record.storage,
        result.samples,
        result.logweights,
        result.provenance.round,
    )
    @test all(array -> array isa CUDA.AnyCuArray, arrays)
    @test state.schedule isa Tuple
    @test state.offsets isa Tuple
    return nothing
end

function public_execution_case(
    device,
    ::Type{T},
    kind;
    repeated=false,
) where {T}
    schedule = [9, 11, 13]
    proposal = amis_proposal(T, Val(kind))
    source = prepare_sampler(
        Random.Xoshiro(AMIS_CUDA_SEED + UInt(sizeof(T)) + UInt(kind === :factor)),
        AMISQuadraticTarget{T}(),
        AMIS(proposal; rounds=length(schedule), round_size=schedule);
        threaded=true,
    )
    prepared = device(source)
    caller_device = CUDA.device()
    expected_rounds = reduce(
        vcat,
        [fill(round, count) for (round, count) in pairs(schedule)],
    )
    initial_parameters = current_proposal(MLDataDevices.cpu_device(), prepared)
    @test CUDA.device() == caller_device
    @test_throws ArgumentError current_proposal(prepared)
    first_result = importance_sample!(prepared)
    CUDA.synchronize()
    @test CUDA.device() == caller_device
    assert_amis_residence(prepared, first_result)
    first_samples = Array(first_result.samples)
    first_logweights = Array(first_result.logweights)
    first_rounds = Array(first_result.provenance.round)
    @test length(first_result) == sum(schedule)
    @test first_rounds == expected_rounds
    @test all(isfinite, first_logweights)
    first_transfers = assert_amis_transfers(
        first_result.diagnostics.transfers,
        length(schedule),
        T,
    )
    @test first_result.diagnostics.target_evaluations == sum(schedule)
    @test first_result.diagnostics.proposal_evaluations ==
          length(schedule) * sum(schedule)
    learned = current_proposal(MLDataDevices.cpu_device(), prepared)
    @test CUDA.device() == caller_device
    @test learned.location != initial_parameters.location
    @test_throws ArgumentError current_proposal(MLDataDevices.cpu_device(Float32), prepared)
    @test_throws ArgumentError current_proposal(device, prepared)

    second_transfers = nothing
    repeated_persistence = nothing
    if repeated
        first_learned_location = deepcopy(learned.location)
        second_result = importance_sample!(prepared)
        CUDA.synchronize()
        assert_amis_residence(prepared, second_result)
        @test Array(first_result.samples) == first_samples
        @test Array(first_result.logweights) == first_logweights
        @test Array(first_result.provenance.round) == first_rounds
        @test first_result.samples !== second_result.samples
        @test first_result.logweights !== second_result.logweights
        @test first_result.provenance.round !== second_result.provenance.round
        second_learned = current_proposal(MLDataDevices.cpu_device(), prepared)
        @test second_learned.location != first_learned_location
        second_transfers = assert_amis_transfers(
            second_result.diagnostics.transfers,
            length(schedule),
            T,
        )
        repeated_persistence = true
    end
    return (
        scalar_type=T,
        proposal=kind,
        schedule=Tuple(schedule),
        count=length(first_result),
        provenance=true,
        finite_weights=true,
        residence=true,
        explicit_current_proposal=true,
        first_transfers,
        second_transfers,
        repeated_persistence,
    )
end

function transfer_shape_case(device, ::Type{T}) where {T}
    function run(kind, schedule)
        source = prepare_sampler(
            Random.Xoshiro(AMIS_CUDA_SEED + UInt(sum(schedule))),
            AMISQuadraticTarget{T}(),
            AMIS(
                amis_proposal(T, Val(kind));
                rounds=length(schedule),
                round_size=schedule,
            );
            threaded=true,
        )
        result = importance_sample!(device(source))
        CUDA.synchronize()
        transfers = assert_amis_transfers(
            result.diagnostics.transfers,
            length(schedule),
            T,
        )
        return (count=transfers.count, bytes=transfers.bytes)
    end
    scalar = run(:scalar, [5, 17])
    factor = run(:factor, [9, 13])
    three_rounds = run(:factor, [4, 7, 10])
    @test scalar == factor
    @test three_rounds.count == 9
    @test three_rounds.bytes == 3 * (3sizeof(UInt64) + 2sizeof(T))
    return (; scalar, factor, three_rounds)
end

function wrong_device_pre_rng_case(device, ::Type{T}) where {T}
    devices = collect(CUDA.devices())
    length(devices) > 1 || return (tested=false, reason=:one_visible_device)
    caller = CUDA.device()
    other = first(filter(!=(caller), devices))
    wrong = MLDataDevices.CUDADevice{typeof(other),Nothing}(other)
    algorithm = AMIS(
        amis_proposal(T, Val(:scalar));
        rounds=2,
        round_size=[7, 9],
    )
    make_source() = prepare_sampler(
        Random.Xoshiro(AMIS_CUDA_SEED + 0x100),
        AMISQuadraticTarget{T}(),
        algorithm;
        threaded=true,
    )
    rejected = device(make_source())
    control = device(make_source())
    rejected.device = wrong
    error = try
        importance_sample!(rejected)
        nothing
    catch cause
        cause
    end
    @test error isa SamplerDeviceError
    @test error.reason === :device_residency_mismatch
    rejected.device = device
    after_rejection = importance_sample!(rejected)
    control_result = importance_sample!(control)
    @test Array(after_rejection.samples) == Array(control_result.samples)
    @test Array(after_rejection.logweights) == Array(control_result.logweights)
    @test CUDA.device() == caller
    return (tested=true, reason=:device_residency_mismatch, pre_rng=true)
end

function degenerate_scalar_covariance_case(device)
    T = Float32
    proposal = SphericalGaussian(zero(T), nextfloat(zero(T)))
    algorithm = AMIS(proposal; rounds=1, round_size=1)
    make_source(seed; threaded=true) = prepare_sampler(
        Random.Xoshiro(seed),
        AMISQuadraticTarget{T}(),
        algorithm;
        threaded,
    )

    cpu = make_source(AMIS_CUDA_SEED + 0x200; threaded=false)
    cpu_before = current_proposal(cpu)
    cpu_failure = try
        importance_sample!(cpu)
        nothing
    catch cause
        cause
    end
    @test cpu_failure isa AMISRoundError
    @test cpu_failure.round == 1
    @test cpu_failure.phase === :fit_proposal
    @test cpu_failure.cause isa LinearAlgebra.PosDefException
    @test cpu_failure.cause.info == 1
    @test current_proposal(cpu) == cpu_before

    caller = CUDA.device()
    prepared = device(make_source(AMIS_CUDA_SEED + 0x201))
    before = current_proposal(MLDataDevices.cpu_device(), prepared)
    first_failure = try
        importance_sample!(prepared)
        nothing
    catch cause
        cause
    end
    @test CUDA.device() == caller
    @test first_failure isa AMISRoundError
    @test first_failure.round == cpu_failure.round == 1
    @test first_failure.phase === cpu_failure.phase === :fit_proposal
    @test first_failure.cause isa LinearAlgebra.PosDefException
    @test first_failure.cause.info == cpu_failure.cause.info == 1
    @test current_proposal(MLDataDevices.cpu_device(), prepared) == before
    @test !prepared.running
    first_normals = Array(prepared.random_buffers.normal)
    first_storage = Array(prepared.random_buffers.failure_scratch.record.storage)
    first_snapshot = IS._decode_native_failure(
        first_storage[1],
        first_storage[2],
    )
    @test first_snapshot.count == 1
    @test first_snapshot.reason_bits == IS._AMIS_COVARIANCE_INVALID
    @test iszero(first_storage[3])

    second_failure = try
        importance_sample!(prepared)
        nothing
    catch cause
        cause
    end
    @test CUDA.device() == caller
    @test second_failure isa AMISRoundError
    @test second_failure.phase === :fit_proposal
    @test second_failure.cause isa LinearAlgebra.PosDefException
    @test second_failure.cause.info == 1
    @test current_proposal(MLDataDevices.cpu_device(), prepared) == before
    @test Array(prepared.random_buffers.normal) != first_normals
    @test !prepared.running
    return (
        scalar_type=T,
        round_size=1,
        scale=proposal.scale.scale,
        cpu_cuda_parity=true,
        rollback=true,
        rng_advanced=true,
        caller_device_restored=true,
        failure_reason=:finite_positive_covariance,
        failure_snapshot=true,
    )
end

function environment_record()
    root = normpath(joinpath(@__DIR__, "..", ".."))
    gpu = CUDA.device()
    return (
        commit=readchomp(
            addenv(
                `git -C $root rev-parse HEAD`,
                "GIT_CONFIG_GLOBAL" => "/dev/null",
            ),
        ),
        gpu=CUDA.name(gpu),
        capability=CUDA.capability(gpu),
        driver=CUDA.driver_version(),
        runtime=CUDA.runtime_version(),
        julia=VERSION,
        packages=cuda_package_versions((
            "Adapt",
            "CUDA",
            "ImportanceSamplers",
            "KernelAbstractions",
            "MLDataDevices",
        )),
        seed=AMIS_CUDA_SEED,
        allowscalar=false,
        cuda_solver_status_transfer=
            "CUDA.jl internally reads one solver status scalar; package transfer accounting cannot observe it",
    )
end

function main()
    device = cuda_device()
    caller_device = CUDA.device()
    @test CUDA.name(caller_device) == AMIS_CUDA_HARDWARE
    rows = Dict{Tuple{DataType,Symbol},NamedTuple}()
    for T in (Float32, Float64), kind in (:scalar, :factor)
        rows[(T, kind)] = public_execution_case(
            device,
            T,
            kind;
            repeated=T === Float64 && kind === :factor,
        )
        @test CUDA.device() == caller_device
    end
    transfer_shape = transfer_shape_case(device, Float64)
    wrong_device = wrong_device_pre_rng_case(device, Float64)
    degenerate_scalar_covariance = degenerate_scalar_covariance_case(device)
    @test CUDA.device() == caller_device
    return (
        environment=environment_record(),
        rows=(
            float32_scalar=rows[(Float32, :scalar)],
            float64_scalar=rows[(Float64, :scalar)],
            float32_factor=rows[(Float32, :factor)],
            float64_factor=rows[(Float64, :factor)],
        ),
        transfer_shape,
        wrong_device,
        degenerate_scalar_covariance,
    )
end

main()
