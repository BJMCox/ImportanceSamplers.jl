using CUDA
using ImportanceSamplers
using LinearAlgebra
using MLDataDevices
using Pkg
using Random
using Test

include(joinpath(@__DIR__, "..", "cuda_plain_is_support.jl"))
include(joinpath(@__DIR__, "..", "dm_pmc_capabilities.jl"))

const IS = ImportanceSamplers
const KA = IS.KernelAbstractions
const DM_PMC_CUDA_SEED = 0x646d706d63637564

struct DMPMCQuadraticTarget{T} end

@inline function (::DMPMCQuadraticTarget{T})(sample) where {T}
    value = zero(T)
    @inbounds for coordinate in eachindex(sample)
        value += abs2(sample[coordinate])
    end
    return -T(0.5) * value
end

@inline function cuda_factor_logdensity(sample, context, slot)
    T = eltype(context.lognormalizers)
    factor = context.factors
    location = context.locations
    z1 = (sample[1] - location[1, slot]) / factor[1, 1, slot]
    z2 = (sample[2] - location[2, slot] - factor[2, 1, slot] * z1) /
         factor[2, 2, slot]
    z3 = (
        sample[3] - location[3, slot] - factor[3, 1, slot] * z1 -
        factor[3, 2, slot] * z2
    ) / factor[3, 3, slot]
    z4 = (
        sample[4] - location[4, slot] - factor[4, 1, slot] * z1 -
        factor[4, 2, slot] * z2 - factor[4, 3, slot] * z3
    ) / factor[4, 4, slot]
    return context.lognormalizers[slot] -
           T(0.5) * (abs2(z1) + abs2(z2) + abs2(z3) + abs2(z4))
end

@inline function cuda_factor_mixture_target(sample, context)
    value = oftype(context.logmasses[1], -Inf)
    @inbounds for slot in axes(context.locations, 2)
        term = context.logmasses[slot] +
               cuda_factor_logdensity(sample, context, slot)
        largest = max(value, term)
        value = largest == -Inf ? largest :
                largest + log1p(exp(min(value, term) - largest))
    end
    return value
end

function factor_target_context(bank)
    state = IS._prepare_method_state(
        ImportanceSampling(bank; nsamples=17, mis_scheme=StratifiedMixture()),
    )
    packed = state.bank
    return (
        locations=copy(packed.locations),
        factors=copy(packed.factors),
        lognormalizers=copy(packed.lognormalizers),
        logmasses=copy(packed.logmasses),
    )
end

function deterministic_buffers(::Type{T}, schedule, dimension, proposal_count) where {T}
    capacity = maximum(schedule)
    base = collect(range(T(-1.4), T(1.6); length=dimension * capacity))
    normal_batches = [circshift(base, round - 1) for round in eachindex(schedule)]
    distinct = T[0, 0.2, 0.55, prevfloat(one(T))]
    uniform_batches = [
        round == 1 ? fill(T(0.2), proposal_count) : distinct[1:proposal_count] for
        round in eachindex(schedule)
    ]
    return normal_batches, uniform_batches
end

function prefilled_trajectory!(sampler, normal_batches, uniform_batches)
    state = sampler.method_state
    bank = state.bank
    plan = state.plan
    workspace = state.workspace
    buffers = sampler.random_buffers
    backend = KA.get_backend(buffers.normals)
    execution = IS._ThreadedCPUExecution()
    target = IS._bind_resolved_target(
        sampler.target,
        IS._dm_pmc_binding_sample(bank),
    )
    target_evaluator, target_failures = IS._native_target_evaluator(
        backend,
        target,
        eltype(workspace.round_logweights),
        buffers.failure_scratch.target_failures,
    )
    samples = Vector{Matrix{eltype(bank.locations)}}()
    logweights = Vector{Vector{eltype(workspace.round_logweights)}}()
    proposal_ids = Vector{Vector{Int}}()
    ancestors = Vector{Vector{Int}}()

    for round in eachindex(plan.schedule)
        IS._reset_native_failure_scratch!(buffers.failure_scratch)
        round_size = plan.schedule[round]
        round_samples = IS._sample_view(workspace.round_samples, 1:round_size)
        round_logweights = view(workspace.round_logweights, 1:round_size)
        round_proposal_ids = view(workspace.round_proposal_ids, 1:round_size)
        assignments = view(plan.assignments, 1:round_size, round)
        cdf = view(workspace.resampling_cdf, 1:round_size)

        copyto!(buffers.normals, normal_batches[round])
        IS._launch_mis_round!(
            round_samples,
            IS._MISRoundOutput(round_logweights, round_proposal_ids),
            buffers.failure_scratch.record.storage,
            buffers.normals,
            target_evaluator,
            bank,
            assignments,
            IS._RealizedMixtureDenominator(plan.logcoefficients, round),
            workspace.solve_scratch,
            execution,
        )
        snapshot = IS._device_failure_snapshot(buffers.failure_scratch.record)
        IS._throw_native_failures(
            snapshot.failure,
            snapshot.draw_failure,
            target_failures,
            IS._NoSampleTransform(),
        )
        IS._resampling_cdf!(cdf, round_logweights)
        copyto!(buffers.resampling_uniforms, uniform_batches[round])
        IS._resample_and_gather!(
            cdf,
            buffers.resampling_uniforms,
            workspace.ancestors,
            round_samples,
            workspace.candidate_locations,
            execution,
        )
        push!(samples, Array(round_samples))
        push!(logweights, Array(round_logweights))
        push!(proposal_ids, Array(round_proposal_ids))
        push!(ancestors, Array(workspace.ancestors))
        copyto!(bank.locations, workspace.candidate_locations)
        KA.synchronize(backend)
    end
    return (
        samples=reduce(hcat, samples),
        logweights=reduce(vcat, logweights),
        proposal_ids=reduce(vcat, proposal_ids),
        ancestors,
        locations=Array(bank.locations),
    )
end

maximum_error(left, right) = maximum(abs, left .- right; init=zero(eltype(left)))

function parity_case(device, ::Type{T}, kind; zero_mass=false, schedule=[9, 11]) where {T}
    bank = dm_pmc_validation_bank(T, Val(kind); zero_mass)
    algorithm = DeterministicMixturePMC(
        bank;
        rounds=length(schedule),
        round_size=schedule,
    )
    cpu = prepare_sampler(
        Xoshiro(DM_PMC_CUDA_SEED),
        DMPMCQuadraticTarget{T}(),
        algorithm;
        threaded=true,
    )
    gpu_source = prepare_sampler(
        Xoshiro(DM_PMC_CUDA_SEED),
        DMPMCQuadraticTarget{T}(),
        algorithm;
        threaded=true,
    )
    gpu = device(gpu_source)
    normal_batches, uniform_batches = deterministic_buffers(
        T,
        schedule,
        DM_PMC_CUDA_DIMENSION,
        count(!iszero, bank.masses),
    )
    cpu_result = prefilled_trajectory!(cpu, normal_batches, uniform_batches)
    gpu_result = prefilled_trajectory!(gpu, normal_batches, uniform_batches)
    tolerance = kind === :diagonal ?
                dm_pmc_diagonal_tolerance(T) :
                dm_pmc_factor_tolerance(T, DM_PMC_CUDA_DIMENSION)
    sample_error = maximum_error(cpu_result.samples, gpu_result.samples)
    logweight_error = maximum_error(cpu_result.logweights, gpu_result.logweights)
    location_error = maximum_error(cpu_result.locations, gpu_result.locations)
    @test sample_error <= tolerance
    @test logweight_error <= tolerance
    @test location_error <= tolerance
    @test gpu_result.proposal_ids == cpu_result.proposal_ids
    @test gpu_result.ancestors == cpu_result.ancestors
    return (
        scalar_type=T,
        bank=kind,
        schedule=Tuple(schedule),
        zero_mass,
        tolerance,
        sample_error,
        logweight_error,
        location_error,
        duplicate_ancestors=length(unique(first(gpu_result.ancestors))) == 1,
    )
end

function assert_dm_pmc_residence(prepared, result)
    state = prepared.method_state
    bank = state.bank
    plan = state.plan
    workspace = state.workspace
    buffers = prepared.random_buffers
    scale_storage = bank isa IS._PackedDiagonalGaussianBank ?
                    bank.scales : bank.factors
    arrays = Any[
        bank.locations,
        scale_storage,
        bank.lognormalizers,
        bank.logmasses,
        bank.cdf,
        bank.proposal_ids,
        plan.counts,
        plan.assignments,
        plan.logcoefficients,
        workspace.round_samples,
        workspace.round_logweights,
        workspace.round_proposal_ids,
        workspace.resampling_cdf,
        workspace.ancestors,
        workspace.candidate_locations,
        buffers.normals,
        buffers.resampling_uniforms,
        buffers.failure_scratch.record.storage,
        result.samples,
        result.logweights,
        result.provenance.round,
        result.provenance.proposal_id,
    ]
    workspace.solve_scratch isa IS._NoMISSolveScratch ||
        push!(arrays, workspace.solve_scratch)
    @test all(array -> array isa CUDA.AnyCuArray, arrays)
    return nothing
end

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

function assert_dm_pmc_reported_transfers(
    transfers,
    rounds,
    ::Type{T},
    resampling=GlobalResampling(),
) where {T}
    record = reported_transfer_record(transfers)
    is_local = resampling isa LocalResampling
    @test record.count == (is_local ? 5rounds : 6rounds)
    expected_bytes = is_local ?
                     3sizeof(UInt64) + sizeof(Int) + 3sizeof(T) :
                     3sizeof(UInt64) + 5sizeof(T)
    @test record.bytes == rounds * expected_bytes
    @test record.reasons.failure_snapshot == (
        count=rounds,
        bytes=rounds * 3sizeof(UInt64),
    )
    for reason in (
        :logweight_maximum,
        :logweight_scaled_sum,
        :logweight_scaled_square_sum,
    )
        @test getfield(record.reasons, reason) == (
            count=rounds,
            bytes=rounds * sizeof(T),
        )
    end
    if is_local
        @test record.reasons.local_resampling_validity == (
            count=rounds,
            bytes=rounds * sizeof(Int),
        )
        @test iszero(record.reasons.cdf_maximum.count)
        @test iszero(record.reasons.cdf_sum.count)
    else
        for reason in (:cdf_maximum, :cdf_sum)
            @test getfield(record.reasons, reason) == (
                count=rounds,
                bytes=rounds * sizeof(T),
            )
        end
        @test iszero(record.reasons.local_resampling_validity.count)
    end
    return record
end

function public_execution_case(
    device,
    ::Type{T},
    kind;
    repeated=false,
    resampling=GlobalResampling(),
) where {T}
    schedule = [9, 11, 13]
    bank = dm_pmc_validation_bank(T, Val(kind))
    source = prepare_sampler(
        Xoshiro(DM_PMC_CUDA_SEED + 0x100),
        DMPMCQuadraticTarget{T}(),
        DeterministicMixturePMC(
            bank;
            rounds=length(schedule),
            round_size=schedule,
            resampling,
        );
        threaded=true,
    )
    prepared = device(source)
    initial_locations = Array(prepared.method_state.bank.locations)
    active_ids = Array(prepared.method_state.bank.proposal_ids)
    assignments = Array(prepared.method_state.plan.assignments)
    expected_rounds = reduce(vcat, [fill(round, size) for (round, size) in pairs(schedule)])
    expected_proposal_ids = reduce(
        vcat,
        [
            active_ids[assignments[1:size, round]] for
            (round, size) in pairs(schedule)
        ],
    )
    for proposal in prepared.algorithm.bank.proposals
        fill!(proposal.location, T(NaN))
    end
    first_result = importance_sample!(prepared)
    CUDA.synchronize()
    assert_dm_pmc_residence(prepared, first_result)
    retained_after_first = Array(prepared.method_state.bank.locations)
    @test_throws ArgumentError current_proposal(prepared)
    host_bank = current_proposal(MLDataDevices.cpu_device(), prepared)
    @test host_bank.masses == bank.masses
    @test length(host_bank.proposals) == length(bank.proposals)
    for (slot, proposal_id) in pairs(active_ids)
        @test host_bank.proposals[proposal_id].location ≈
              retained_after_first[:, slot]
    end
    host_bank.proposals[first(active_ids)].location[1] = T(1.0e6)
    host_bank.masses[first(active_ids)] = zero(T)
    @test Array(prepared.method_state.bank.locations) == retained_after_first
    @test prepared.algorithm.bank.masses == bank.masses
    first_snapshot = (
        samples=Array(first_result.samples),
        logweights=Array(first_result.logweights),
        round=Array(first_result.provenance.round),
        proposal_id=Array(first_result.provenance.proposal_id),
    )
    diagnostics = first_result.diagnostics
    first_transfers = assert_dm_pmc_reported_transfers(
        diagnostics.transfers,
        length(schedule),
        T,
        resampling,
    )
    @test length(first_result) == sum(schedule)
    @test first_snapshot.round == expected_rounds
    @test first_snapshot.proposal_id == expected_proposal_ids
    @test all(isfinite, first_snapshot.logweights)
    @test diagnostics.method === :deterministic_mixture_pmc
    @test diagnostics.resampling === IS._pmc_resampling_name(resampling)
    @test diagnostics.execution === :threaded
    @test diagnostics.threaded
    @test diagnostics.rounds == length(schedule)
    @test diagnostics.round_sizes == schedule
    @test length(diagnostics.round_ess) == length(schedule)
    @test length(diagnostics.round_lognormalizers) == length(schedule)
    @test all(isfinite, diagnostics.round_ess)
    @test all(isfinite, diagnostics.round_lognormalizers)
    @test diagnostics.failures == 0
    @test retained_after_first != initial_locations

    if resampling isa LocalResampling
        final_round = lastindex(schedule)
        final_indices = findall(==(final_round), first_snapshot.round)
        for (slot, proposal_id) in pairs(active_ids)
            generated = filter(final_indices) do index
                first_snapshot.proposal_id[index] == proposal_id
            end
            @test any(generated) do index
                first_snapshot.samples[:, index] == retained_after_first[:, slot]
            end
        end
    end

    second_transfers = nothing
    earlier_result_independent = nothing
    retained_population_repeated = nothing
    if repeated
        second_result = importance_sample!(prepared)
        CUDA.synchronize()
        assert_dm_pmc_residence(prepared, second_result)
        @test Array(first_result.samples) == first_snapshot.samples
        @test Array(first_result.logweights) == first_snapshot.logweights
        @test Array(first_result.provenance.round) == first_snapshot.round
        @test Array(first_result.provenance.proposal_id) == first_snapshot.proposal_id
        @test first_result.samples !== second_result.samples
        @test first_result.logweights !== second_result.logweights
        @test first_result.provenance.round !== second_result.provenance.round
        @test first_result.provenance.proposal_id !==
              second_result.provenance.proposal_id
        @test Array(prepared.method_state.bank.locations) != retained_after_first
        second_transfers = assert_dm_pmc_reported_transfers(
            second_result.diagnostics.transfers,
            length(schedule),
            T,
            resampling,
        )
        earlier_result_independent = true
        retained_population_repeated = true
    end
    return (
        scalar_type=T,
        bank=kind,
        schedule=Tuple(schedule),
        count=length(first_result),
        provenance=true,
        residence=true,
        finite_raw_logweights=true,
        diagnostics=true,
        first_transfers,
        explicit_current_proposal=true,
        second_transfers,
        retained_population=true,
        retained_population_repeated,
        earlier_result_independent,
        resampling=IS._pmc_resampling_name(resampling),
    )
end

function transfer_scaling_case(device, ::Type{T}) where {T}
    configured = dm_pmc_validation_bank(T, Val(:diagonal))
    smaller_bank = ProposalBank(
        configured.proposals[1:3],
        configured.masses[1:3],
    )
    function run(bank, schedule)
        prepared = prepare_sampler(
            Xoshiro(DM_PMC_CUDA_SEED + UInt(length(schedule))),
            DMPMCQuadraticTarget{T}(),
            DeterministicMixturePMC(
                bank;
                rounds=length(schedule),
                round_size=schedule,
            );
            threaded=true,
        ) |> device
        result = importance_sample!(prepared)
        CUDA.synchronize()
        transfers = result.diagnostics.transfers
        assert_dm_pmc_reported_transfers(transfers, length(schedule), T)
        return (count=transfers.count, bytes=transfers.bytes)
    end
    two_rounds = run(configured, [9, 11])
    two_rounds_other_shape = run(smaller_bank, [5, 17])
    three_rounds = run(configured, [9, 11, 13])
    @test two_rounds == two_rounds_other_shape
    @test two_rounds.count == 12
    @test three_rounds.count == 18
    @test two_rounds.bytes == 2 * (3sizeof(UInt64) + 5sizeof(T))
    @test three_rounds.bytes == 3 * (3sizeof(UInt64) + 5sizeof(T))
    return (; two_rounds, two_rounds_other_shape, three_rounds)
end

function static_factor_case(device, ::Type{T}) where {T}
    bank = dm_pmc_validation_bank(T, Val(:factor))
    context = factor_target_context(bank)
    algorithm = ImportanceSampling(
        bank;
        nsamples=257,
        mis_scheme=StratifiedMixture(),
    )
    cpu = importance_sample(
        Xoshiro(DM_PMC_CUDA_SEED),
        cuda_factor_mixture_target,
        context,
        algorithm;
        threaded=false,
    )
    prepared = prepare_sampler(
        Xoshiro(DM_PMC_CUDA_SEED),
        cuda_factor_mixture_target,
        context,
        algorithm;
        threaded=true,
    ) |> device
    gpu = importance_sample!(prepared)
    CUDA.synchronize()
    tolerance = dm_pmc_factor_tolerance(T, DM_PMC_CUDA_DIMENSION)
    cpu_error = maximum(abs, cpu.logweights; init=zero(T))
    gpu_error = maximum(abs, Array(gpu.logweights); init=zero(T))
    @test cpu_error <= tolerance
    @test gpu_error <= tolerance
    @test prepared.method_state.bank.factors isa CUDA.AnyCuArray
    @test prepared.random_buffers.solve_scratch isa CUDA.AnyCuArray
    return (; scalar_type=T, tolerance, cpu_error, gpu_error)
end

function strict_resampling_and_ess_case(::Type{T}) where {T}
    cdf = CuArray(T[0, 0.2, 0.2, 1])
    uniforms = CuArray(T[0, 0.2, 0.2, prevfloat(one(T))])
    ancestors = CUDA.zeros(Int, 4)
    samples = CuArray(reshape(T.(1:16), 4, 4))
    candidates = similar(samples)
    IS._resample_and_gather!(
        cdf,
        uniforms,
        ancestors,
        samples,
        candidates,
        IS._ThreadedCPUExecution(),
    )
    selected = Array(ancestors)
    @test selected == [2, 4, 4, 4]
    transfers = IS._ResultTransferCounter(0, 0)
    summary = IS._logweight_summary(
        CuArray(T[floatmax(T), floatmax(T)]),
        transfers,
    )
    @test summary.ess == T(2)
    @test isfinite(summary.lognormalizer)
    @test transfers.count == 3
    @test transfers.bytes == 3sizeof(T)
    reason_record = reported_transfer_record(transfers)
    @test reason_record.reasons.logweight_maximum ==
          (count=1, bytes=sizeof(T))
    @test reason_record.reasons.logweight_scaled_sum ==
          (count=1, bytes=sizeof(T))
    @test reason_record.reasons.logweight_scaled_square_sum ==
          (count=1, bytes=sizeof(T))
    return (; selected, ess=summary.ess, transfers=reason_record)
end

function factor_batch_case(device, ::Type{T}) where {T}
    dimension = 32
    round_size = 4096
    proposal_count = 4
    locations = Matrix{T}(undef, dimension, proposal_count)
    factors = Array{T}(undef, dimension, dimension, proposal_count)
    proposals = map(1:proposal_count) do proposal
        location = T.(range(-0.3, 0.3; length=dimension)) .+
                   T(0.2proposal)
        factor = Matrix{T}(I, dimension, dimension)
        for index in 1:dimension
            factor[index, index] = T(0.7) + T(0.04proposal) + T(0.004index)
            index > 1 && (factor[index, index - 1] = T(0.015proposal))
        end
        locations[:, proposal] = location
        factors[:, :, proposal] = factor
        FactorGaussian(location, factor)
    end
    masses = T[1, 2, 3, 4]
    prepared = prepare_sampler(
        Xoshiro(DM_PMC_CUDA_SEED + 0x500),
        DMPMCQuadraticTarget{T}(),
        DeterministicMixturePMC(
            ProposalBank(proposals, masses);
            rounds=2,
            round_size,
        );
        factor_execution=BatchedFactorExecution(),
        threaded=true,
    ) |> device
    state = prepared.method_state
    denominator = IS._RealizedMixtureDenominator(state.plan.logcoefficients, 1)
    first_logcoefficients = Array(view(state.plan.logcoefficients, :, 1))
    @test IS._use_factor_batch_mis_path(
        prepared.device,
        state.bank,
        denominator,
        T,
        prepared.factor_execution,
    )
    result = importance_sample!(prepared)
    CUDA.synchronize()
    assert_dm_pmc_residence(prepared, result)
    @test length(result) == 2round_size
    samples = Array(result.samples)[:, 1:round_size]
    observed = Array(result.logweights)[1:round_size]
    expected = map(eachcol(samples)) do sample
        logdenominator = T(-Inf)
        for proposal in 1:proposal_count
            standardized = LowerTriangular(view(factors, :, :, proposal)) \
                           (sample - view(locations, :, proposal))
            logdensity = -T(0.5dimension) * log(T(2pi)) -
                         sum(log, diag(view(factors, :, :, proposal))) -
                         T(0.5) * sum(abs2, standardized)
            term = first_logcoefficients[proposal] + logdensity
            largest = max(logdenominator, term)
            logdenominator = largest == -Inf ? largest :
                             largest + log1p(exp(min(logdenominator, term) - largest))
        end
        DMPMCQuadraticTarget{T}()(sample) - logdenominator
    end
    tolerance = T(8192) * eps(T)
    @test observed ≈ expected rtol = tolerance atol = tolerance
    return (scalar_type=T, dimension, round_size)
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
        seed=DM_PMC_CUDA_SEED,
        allowscalar=false,
    )
end

function main()
    device = cuda_device()
    caller_device = CUDA.device()
    @test CUDA.name(caller_device) == DM_PMC_CUDA_HARDWARE
    private_prefilled_parity = Dict{Tuple{DataType,Symbol},NamedTuple}()
    public_execution = Dict{Tuple{DataType,Symbol},NamedTuple}()
    static_factor = NamedTuple[]
    for T in (Float32, Float64), kind in (:diagonal, :factor)
        private_prefilled_parity[(T, kind)] = parity_case(device, T, kind)
        public_execution[(T, kind)] = public_execution_case(
            device,
            T,
            kind;
            repeated=T === Float64 && kind === :factor,
        )
        kind === :factor && push!(static_factor, static_factor_case(device, T))
        @test CUDA.device() == caller_device
    end
    unequal = parity_case(
        device,
        Float64,
        :diagonal;
        zero_mass=true,
        schedule=[7, 10],
    )
    @test unequal.zero_mass
    @test unequal.schedule == (7, 10)
    @test private_prefilled_parity[(Float64, :diagonal)].duplicate_ancestors
    transfer_scaling = transfer_scaling_case(device, Float64)
    strict = (
        Float32 => strict_resampling_and_ess_case(Float32),
        Float64 => strict_resampling_and_ess_case(Float64),
    )
    factor_batches = [factor_batch_case(device, T) for T in (Float32, Float64)]
    local_resampling = public_execution_case(
        device,
        Float32,
        :diagonal;
        resampling=LocalResampling(),
    )
    @test Tuple(row.label for row in DM_PMC_CUDA_CAPABILITY_ROWS) == (
        :float32_diagonal,
        :float64_diagonal,
        :float32_factor,
        :float64_factor,
        :unequal_masses_with_zero,
        :unequal_round_sizes,
        :duplicate_resampled_ancestors,
        :repeated_prepared_execution,
        :local_resampling,
    )
    rows = (
        float32_diagonal=(
            public_execution=public_execution[(Float32, :diagonal)],
            private_prefilled_parity=private_prefilled_parity[(Float32, :diagonal)],
        ),
        float64_diagonal=(
            public_execution=public_execution[(Float64, :diagonal)],
            private_prefilled_parity=private_prefilled_parity[(Float64, :diagonal)],
        ),
        float32_factor=(
            public_execution=public_execution[(Float32, :factor)],
            private_prefilled_parity=private_prefilled_parity[(Float32, :factor)],
        ),
        float64_factor=(
            public_execution=public_execution[(Float64, :factor)],
            private_prefilled_parity=private_prefilled_parity[(Float64, :factor)],
        ),
        unequal_masses_with_zero=(private_prefilled_parity=unequal,),
        unequal_round_sizes=(schedule=unequal.schedule,),
        duplicate_resampled_ancestors=(
            duplicate=private_prefilled_parity[(Float64, :diagonal)].duplicate_ancestors,
            strict,
        ),
        repeated_prepared_execution=merge(
            public_execution[(Float64, :factor)],
            (; transfer_scaling),
        ),
        local_resampling=(public_execution=local_resampling,),
    )
    @test all(row -> hasproperty(rows, row.label), DM_PMC_CUDA_CAPABILITY_ROWS)
    @test CUDA.device() == caller_device
    return (
        environment=environment_record(),
        static_factor,
        private_prefilled_parity,
        public_execution,
        factor_batches,
        local_resampling,
        rows,
    )
end

main()
