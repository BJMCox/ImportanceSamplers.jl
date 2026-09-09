module CUDALAISValidation

using CUDA, ImportanceSamplers, LinearAlgebra, MLDataDevices, Random, Test
const IS = ImportanceSamplers
include("../../test/support/lais.jl")

Random.randn!(rng::LAISScriptedRNG, out::CUDA.AnyCuArray{T}) where {T<:AbstractFloat} =
    invoke(Random.randn!, Tuple{LAISScriptedRNG,AbstractArray}, rng, out)
Random.rand!(rng::LAISScriptedRNG, out::CUDA.AnyCuArray{T}) where {T<:AbstractFloat} =
    invoke(Random.rand!, Tuple{LAISScriptedRNG,AbstractArray}, rng, out)

# Present the same prescribed move buffers to the batched GPU implementation.
function Random.randn!(rng::LAISScriptedRNG, out::CUDA.AnyCuArray{T,3}) where {T<:AbstractFloat}
    for step in axes(out, 3)
        invoke(Random.randn!, Tuple{LAISScriptedRNG,AbstractArray}, rng, view(out, :, :, step))
    end
    return out
end
function Random.rand!(rng::LAISScriptedRNG, out::CUDA.AnyCuArray{T,2}) where {T<:AbstractFloat}
    for step in axes(out, 2)
        invoke(Random.rand!, Tuple{LAISScriptedRNG,AbstractArray}, rng, view(out, :, step))
    end
    return out
end

function check_warmup_batch_boundary(device)
    normals = [[isodd(step) ? -0.1 : 0.1, 0.2] for step in 1:258]
    centres = zeros(2)
    factor = 1.0
    for step in 1:258
        centres .+= factor .* normals[step]
        step <= 257 && (factor *= sqrt(1 + 0.766step^(-0.6)))
    end
    algorithm = LAIS(ProposalBank([SphericalGaussian(0.0, 1.0), SphericalGaussian(0.0, 1.0)]);
        transition=RAM(1.0; tuning=WarmupTuning(257)), rounds=1, round_size=2)
    base = device(prepare_sampler(Xoshiro(21), x -> 0.0, algorithm))
    sampler = scripted_sampler(base, [normals; [zeros(2)]], [[0.1, 0.1] for _ in 1:258])
    result = cpu_device()(importance_sample!(sampler))
    @test result.samples ≈ centres
    @test result.diagnostics.transition.warmup_target_evaluations == 514
end

struct QuadraticTarget{T} end
(::QuadraticTarget{T})(x) where {T} = -sum(abs2, x) / T(20)

scalar_target(x) = -abs2(x) / 2
shifted_target(x, p) = -(abs2(x[1] - p.location[1]) + abs2(x[2] - p.location[2])) / 20
failure_target(x) = x > 3 ? Inf : 0.0

function check_transfer_budget(result, rounds)
    # Batch snapshots, one packed weight summary per round, and two counts.
    # Bounds allow fewer transfers without encoding a kernel-launch count.
    snapshots = 1 + 2rounds
    max_count = snapshots + rounds + 2
    max_bytes = 3sizeof(UInt64) * snapshots + 3rounds * sizeof(eltype(result.logweights)) + 2sizeof(Int)
    transfers = result.diagnostics.transfers
    @test transfers.count <= max_count && transfers.bytes <= max_bytes
end

function check_failure_rollback(device)
    base = device(prepare_sampler(Xoshiro(19), failure_target,
        LAIS(ProposalBank([SphericalGaussian(0.0, 1.0)]);
            transition=RAM(1.0; tuning=ContinuousTuning()), rounds=2, round_size=1)))
    normals = [[u] for u in (1.0, 0.0, 2.0, 0.5, 0.0, 0.5, 0.0)]
    sampler = scripted_sampler(base, normals, [[0.1] for _ in 1:4])
    try
        importance_sample!(sampler)
    catch error
        error isa LAISRoundError || rethrow()
    end
    result = cpu_device()(importance_sample!(sampler))
    check_transfer_budget(result, 2)
    @test result.samples ≈ [0.5, 0.5 + 0.5sqrt(1.766)]
    @test result.logweights ≈ fill(log(2pi) / 2, 2)
    @test result.diagnostics.target_evaluations == 5
end

function check_fixed_rwm(device)
    normals = [[1.0, 1.0], [-0.5, 0.5, -0.5, 0.5],
        [-1.0, -1.0], [-0.5, 0.5, -0.5, 0.5]]
    uniforms = [[0.5, 0.9], [0.5, 0.9]]
    algorithm = LAIS(ProposalBank([SphericalGaussian(-1.0, 1.0), SphericalGaussian(1.0, 2.0)]);
        transition=RandomWalkMetropolis(1.0), rounds=2, round_size=4)
    reference = importance_sample!(prepare_sampler(LAISScriptedRNG(normals, uniforms, 1, 1),
        scalar_target, algorithm))
    base = device(prepare_sampler(Xoshiro(18), scalar_target, algorithm))
    result = cpu_device()(importance_sample!(scripted_sampler(base, normals, uniforms)))
    check_transfer_budget(result, 2)
    @test result.samples ≈ reference.samples
    @test result.logweights ≈ reference.logweights
    @test result.provenance == reference.provenance
end

function check_warmup_rollback(device)
    oracle = lais_ram_oracle(Float64)
    algorithm = LAIS(ProposalBank([FactorGaussian(zeros(2), Matrix{Float64}(I, 2, 2))]);
        transition=RAM([1.0 1.0; 1.0 2.0]; tuning=WarmupTuning(2)), rounds=1, round_size=1)
    base = device(prepare_sampler(Xoshiro(20), QuadraticTarget{Float64}(), algorithm))
    # Zero direction fails the first move. Later queued moves cannot erase it.
    normals = [zeros(2), ones(2), ones(2), oracle.normals..., zeros(2)]
    uniforms = [[0.1], [0.1], [0.1], ([u] for u in oracle.uniforms)...]
    sampler = scripted_sampler(base, normals, uniforms)
    try
        importance_sample!(sampler)
    catch error
        error isa LAISRoundError || rethrow()
    end
    result = cpu_device()(importance_sample!(sampler))
    @test result.samples ≈ oracle.centres[:, 3:3]
    @test result.diagnostics.transition.warmup_target_evaluations == 2
end

function check_retarget_and_reuse(device, source, oracle, algorithm, normals, uniforms)
    context = (; location=[0.5, -0.5])
    changed = retarget(Xoshiro(91), source, shifted_target, context)
    learned = ProposalBank([FactorGaussian(oracle.centres[:, end], Matrix{Float64}(I, 2, 2))])
    reference = device(prepare_sampler(Xoshiro(91), shifted_target, context,
        LAIS(learned; transition=RAM(cholesky(Symmetric(oracle.factor * oracle.factor'));
            tuning=ContinuousTuning()), rounds=3, round_size=1);
        factor_execution=BatchedFactorExecution()))
    actual = cpu_device()(importance_sample!(changed))
    expected = cpu_device()(importance_sample!(reference))
    @test actual.samples ≈ expected.samples
    @test actual.logweights ≈ expected.logweights

    cpu = prepare_sampler(LAISScriptedRNG(normals, uniforms, 1, 1), QuadraticTarget{Float64}(), algorithm)
    importance_sample!(cpu)
    expected_reuse = importance_sample!(cpu)
    actual_reuse = cpu_device()(importance_sample!(source))
    @test actual_reuse.samples ≈ expected_reuse.samples
    @test actual_reuse.logweights ≈ expected_reuse.logweights
end

# The controlled RNG is a reproducer fixture, not a production injection hook.
# All scripted buffers move once, before sampling, and stay on the device.
function scripted_sampler(base, normals, uniforms)
    rng = LAISScriptedRNG(Tuple(CuArray.(normals)), Tuple(CuArray.(uniforms)), 1, 1)
    return IS._PreparedImportanceSampler(rng, base.random_buffers, base.target,
        base.algorithm, base.method_state, base.device, base.factor_execution,
        base.threaded, false, false)
end

function main()
    CUDA.allowscalar(false)
    CUDA.functional() || error("CUDA is required for this reproducer")
    physical = CUDA.device()
    device = MLDataDevices.CUDADevice{typeof(physical),Nothing}(physical)
    @testset "CUDA LAIS recurrence and resident results" begin
        check_fixed_rwm(device)
        check_failure_rollback(device)
        check_warmup_rollback(device)
        check_warmup_batch_boundary(device)
        for (T, warmup, policy) in ((Float32, false, FusedFactorExecution()),
            (Float64, false, BatchedFactorExecution()),
            (Float64, true, FusedFactorExecution()))
            oracle = lais_ram_oracle(T)
            bank = ProposalBank([FactorGaussian(zeros(T, 2), Matrix{T}(I, 2, 2))])
            tuning = warmup ? WarmupTuning(2) : ContinuousTuning()
            algorithm = LAIS(bank; transition=RAM(T[1 1; 1 2]; tuning),
                rounds=warmup ? 1 : 3, round_size=1)
            base = device(prepare_sampler(Xoshiro(17), QuadraticTarget{T}(),
                algorithm; factor_execution=policy))
            normals = warmup ? [oracle.normals; [zeros(T, 2)]] :
                reduce(vcat, [[u, zeros(T, 2)] for u in oracle.normals])
            uniforms = [[v] for v in oracle.uniforms]
            normals, uniforms = [normals; normals], [uniforms; uniforms]
            sampler = scripted_sampler(base, normals, uniforms)
            result = importance_sample!(sampler)
            CUDA.synchronize()
            check_transfer_budget(result, warmup ? 1 : 3)
            @test all(x -> x isa CUDA.AnyCuArray,
                (result.samples, result.logweights, result.provenance.round, result.provenance.proposal_id))
            host = cpu_device()(result)
            expected = warmup ? oracle.centres[:, 3:3] : oracle.centres
            @test host.samples ≈ expected
            @test host.logweights ≈ [QuadraticTarget{T}()(x) + log(T(2pi)) for x in eachcol(expected)]
            if T === Float64 && !warmup
                check_retarget_and_reuse(device, sampler, oracle, algorithm, normals, uniforms)
            end
        end
    end
    return (; device=CUDA.name(physical))
end

end
