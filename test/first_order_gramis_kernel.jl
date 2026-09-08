using Test
using ImportanceSamplers
import LinearAlgebra
import Random

const GRAMISKernelIS = ImportanceSamplers

struct GRAMISKernelTarget{T} end

function (::GRAMISKernelTarget{T})(sample)::T where {T}
    value = sample isa Real ? sample : only(sample)
    return -abs2(value) / T(2)
end

struct GRAMISSubtractionOverflowTarget{T}
    value::T
end

function (target::GRAMISSubtractionOverflowTarget{T})(sample)::T where {T}
    return target.value
end

function gram_is_factor_policy_result(algorithm, policy)
    return importance_sample(
        Random.Xoshiro(0x4752414d49534b45),
        GRAMISKernelTarget{Float64}(),
        algorithm;
        threaded=false,
        factor_execution=policy,
    )
end

@testset "shared factor-policy output characterization before GRAMIS adaptation" begin
    bank = ProposalBank(
        [
            FactorGaussian([-1.0], reshape([0.75], 1, 1)),
            FactorGaussian([1.0], reshape([1.25], 1, 1)),
        ],
        [1.0, 1.0],
    )
    algorithms = (
        static=ImportanceSampling(
            bank;
            nsamples=8,
            mis_scheme=StratifiedMixture(),
        ),
        dm_pmc=DeterministicMixturePMC(bank; rounds=1, round_size=8),
        amis=AMIS(first(bank.proposals); rounds=1, round_size=8),
    )

    for (name, algorithm) in pairs(algorithms)
        fused = gram_is_factor_policy_result(algorithm, FusedFactorExecution())
        batched = gram_is_factor_policy_result(algorithm, BatchedFactorExecution())

        @test fused.samples == batched.samples
        @test fused.logweights == batched.logweights
        @test fused.provenance == batched.provenance
        @test fused.diagnostics.factor_execution_policy === :fused
        @test batched.diagnostics.factor_execution_policy === :batched
        @test length(fused) == 8
    end
end

function gram_is_scalar_logdensity(sample, location, scale)
    T = promote_type(typeof(sample), typeof(location), typeof(scale))
    standardized = (sample - location) / scale
    return -log(scale) - log(T(2pi)) / T(2) - abs2(standardized) / T(2)
end

function gram_is_one_round_oracle(
    ::Type{T},
    locations,
    scales,
    normals,
    assignments,
    counts,
) where {T}
    sample_count = length(assignments)
    samples = Vector{T}(undef, sample_count)
    logtargets = Vector{T}(undef, sample_count)
    returned_logweights = Vector{T}(undef, sample_count)
    local_logweights = Vector{T}(undef, sample_count)
    generating_logdensities = Vector{T}(undef, sample_count)
    for sample_index in eachindex(assignments)
        slot = assignments[sample_index]
        sample = locations[slot] + scales[slot] * normals[sample_index]
        logtarget = -abs2(sample) / T(3) + sample / T(5) - T(0.7)
        logmixture = T(-Inf)
        for proposal_slot in eachindex(locations)
            logcoefficient = log(T(counts[proposal_slot]) / T(sample_count))
            logdensity = gram_is_scalar_logdensity(
                sample,
                locations[proposal_slot],
                scales[proposal_slot],
            )
            logmixture = GRAMISKernelIS.LogExpFunctions.logaddexp(
                logmixture,
                logcoefficient + logdensity,
            )
        end
        generating_logdensity = gram_is_scalar_logdensity(
            sample,
            locations[slot],
            scales[slot],
        )
        samples[sample_index] = sample
        logtargets[sample_index] = logtarget
        returned_logweights[sample_index] = logtarget - logmixture
        local_logweights[sample_index] = logtarget - generating_logdensity
        generating_logdensities[sample_index] = generating_logdensity
    end
    return (;
        samples,
        logtargets,
        returned_logweights,
        local_logweights,
        generating_logdensities,
        proposal_ids=assignments,
    )
end

struct GRAMISFrozenRoundTarget{T}
    calls::Threads.Atomic{Int}
end

GRAMISFrozenRoundTarget{T}() where {T} =
    GRAMISFrozenRoundTarget{T}(Threads.Atomic{Int}(0))

function (target::GRAMISFrozenRoundTarget{T})(sample)::T where {T}
    Threads.atomic_add!(target.calls, 1)
    value = only(sample)
    return -abs2(value) / T(3) + value / T(5) - T(0.7)
end

struct GRAMISCountedFrozenValue{T}
    calls::Threads.Atomic{Int}
end

struct GRAMISCountedFrozenGradient{T}
    calls::Threads.Atomic{Int}
end

struct GRAMISBacktrackingTarget{T}
    calls::Vector{Threads.Atomic{Int}}
end

struct GRAMISAlwaysAcceptTarget{T}
    calls::Threads.Atomic{Int}
end

struct GRAMISAtomicReadVector{A<:AbstractVector{Bool}} <: AbstractVector{Bool}
    storage::A
    reads::Threads.Atomic{Int}
end

struct GRAMISRoundOneValue{T}
    calls::Threads.Atomic{Int}
end

struct GRAMISRoundOneGradient
    calls::Threads.Atomic{Int}
end

struct GRAMISReadCountingArray{T,N,A<:AbstractArray{T,N},R} <: AbstractArray{T,N}
    storage::A
    reads::R
end

struct GRAMISOperationCountingMatrix{T,A<:Matrix{T}} <: AbstractMatrix{T}
    storage::A
    factorizations::Base.RefValue{Int}
    solves::Base.RefValue{Int}
end

Base.size(array::GRAMISReadCountingArray) = size(array.storage)
Base.IndexStyle(::Type{<:GRAMISReadCountingArray}) = IndexCartesian()
Base.size(array::GRAMISAtomicReadVector) = size(array.storage)
Base.IndexStyle(::Type{<:GRAMISAtomicReadVector}) = IndexLinear()
Base.size(matrix::GRAMISOperationCountingMatrix) = size(matrix.storage)
Base.IndexStyle(::Type{<:GRAMISOperationCountingMatrix}) = IndexLinear()
Base.getindex(matrix::GRAMISOperationCountingMatrix, index::Int) =
    matrix.storage[index]

function Base.setindex!(
    matrix::GRAMISOperationCountingMatrix,
    value,
    index::Int,
)
    matrix.storage[index] = value
    return value
end

Base.getindex(matrix::GRAMISOperationCountingMatrix, row::Int, column::Int) =
    matrix.storage[row, column]

function Base.setindex!(
    matrix::GRAMISOperationCountingMatrix,
    value,
    row::Int,
    column::Int,
)
    matrix.storage[row, column] = value
    return value
end

function LinearAlgebra.LAPACK.potrf!(
    uplo::AbstractChar,
    matrix::GRAMISOperationCountingMatrix,
)
    matrix.factorizations[] += 1
    _, info = LinearAlgebra.LAPACK.potrf!(uplo, matrix.storage)
    return matrix, info
end

function LinearAlgebra.ldiv!(
    factor::LinearAlgebra.LowerTriangular{
        T,
        <:GRAMISOperationCountingMatrix{T},
    },
    right_hand_side::AbstractMatrix{T},
) where {T}
    factor.data.solves[] += 1
    LinearAlgebra.ldiv!(
        LinearAlgebra.LowerTriangular(factor.data.storage),
        right_hand_side,
    )
    return right_hand_side
end

function Base.getindex(
    array::GRAMISReadCountingArray{T,N},
    indices::Vararg{Int,N},
) where {T,N}
    gram_is_count_read!(array.reads)
    return array.storage[indices...]
end

gram_is_count_read!(reads::Base.RefValue{Int}) = (reads[] += 1)
gram_is_count_read!(reads::Threads.Atomic{Int}) = Threads.atomic_add!(reads, 1)

function Base.getindex(array::GRAMISAtomicReadVector, index::Int)
    Threads.atomic_add!(array.reads, 1)
    return array.storage[index]
end

function Base.setindex!(array::GRAMISAtomicReadVector, value, index::Int)
    array.storage[index] = value
    return value
end

function (target::GRAMISCountedFrozenValue{T})(sample)::T where {T}
    Threads.atomic_add!(target.calls, 1)
    return -abs2(sample[1]) / T(2) - abs2(sample[2]) / T(4)
end

function (gradient::GRAMISCountedFrozenGradient{T})(destination, sample) where {T}
    Threads.atomic_add!(gradient.calls, 1)
    destination[1] = -sample[1]
    destination[2] = -sample[2] / T(2)
    return destination
end

function (target::GRAMISBacktrackingTarget{T})(sample)::T where {T}
    value = only(sample)
    proposal_slot = floor(Int, value / T(10)) + 1
    Threads.atomic_add!(target.calls[proposal_slot], 1)
    if proposal_slot == 1
        return zero(T)
    elseif proposal_slot == 2
        return value <= T(10.25) ? zero(T) : -one(T)
    end
    return -one(T)
end

function (target::GRAMISAlwaysAcceptTarget{T})(sample)::T where {T}
    Threads.atomic_add!(target.calls, 1)
    return zero(T)
end

function (target::GRAMISRoundOneValue{T})(sample)::T where {T}
    Threads.atomic_add!(target.calls, 1)
    return -abs2(only(sample)) / T(2)
end

function (gradient::GRAMISRoundOneGradient)(destination, sample)
    Threads.atomic_add!(gradient.calls, 1)
    destination[1] = -sample[1]
    return destination
end

gram_is_kernel_state_target(sample) = -sum(abs2, sample) / 2

function gram_is_kernel_state_gradient!(destination, sample)
    destination .= -sample
    return destination
end

function gram_is_kernel_state(algorithm)
    sampler = prepare_sampler(
        Random.Xoshiro(0x4752414d49535354),
        LogTarget(
            gram_is_kernel_state_target;
            grad=gram_is_kernel_state_gradient!,
        ),
        algorithm;
        threaded=false,
    )
    return sampler.method_state
end

function run_gram_is_frozen_round(
    ::Type{T},
    execution,
    factor_execution;
    failed_sample=nothing,
) where {T}
    locations = T[-2, 0.5, 2.5]
    scales = T[0.75, 1.25, 0.5]
    bank = GRAMISKernelIS._first_order_gramis_factor_bank(ProposalBank([
        FactorGaussian(T[locations[1]], reshape(T[scales[1]], 1, 1)),
        FactorGaussian(T[locations[2]], reshape(T[scales[2]], 1, 1)),
        FactorGaussian(T[locations[3]], reshape(T[scales[3]], 1, 1)),
    ]))
    normals = T[-1.5, -0.25, 0.5, 1.25, -1, 0.25, 1.5, -0.75, 0, 1]
    assignments = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    counts = [4, 3, 3]
    sample_count = length(assignments)
    samples = fill(T(101), 1, sample_count)
    returned_logweights = fill(T(102), sample_count)
    local_logweights = fill(T(103), sample_count)
    generating_logdensities = fill(T(104), sample_count)
    proposal_ids = fill(105, sample_count)
    round_ids = fill(106, sample_count)
    failures = zeros(UInt64, 3)
    sampled_normals = copy(normals)
    isnothing(failed_sample) || (sampled_normals[failed_sample] = T(Inf))
    denominator = GRAMISKernelIS._RealizedMixtureDenominator(
        reshape(log.(T.(counts) ./ T(sample_count)), :, 1),
        1,
    )
    counted_target = GRAMISFrozenRoundTarget{T}()
    target = GRAMISKernelIS._NativeDeviceTarget{
        T,
        GRAMISFrozenRoundTarget{T},
    }(counted_target)
    GRAMISKernelIS._first_order_gramis_sample_round!(
        samples,
        returned_logweights,
        local_logweights,
        generating_logdensities,
        proposal_ids,
        round_ids,
        failures,
        sampled_normals,
        target,
        bank,
        assignments,
        denominator,
        GRAMISKernelIS._allocate_mis_solve_scratch(normals, bank, sample_count),
        7,
        execution,
        GRAMISKernelIS.MLDataDevices.CPUDevice(),
        factor_execution,
    )
    oracle = gram_is_one_round_oracle(
        T,
        locations,
        scales,
        normals,
        assignments,
        counts,
    )
    return (;
        samples,
        returned_logweights,
        local_logweights,
        generating_logdensities,
        proposal_ids,
        round_ids,
        failures,
        target_calls=counted_target.calls[],
        oracle,
    )
end

function run_gram_is_subtraction_overflow(::Type{T}, branch, execution) where {T}
    large_normal = T(0.9) * sqrt(floatmax(T))
    if branch === :returned
        locations = T[0, 1]
        normals = T[large_normal]
        assignments = [1]
        counts = [1, 0]
    elseif branch === :local
        locations = T[0, large_normal]
        normals = T[large_normal, 0]
        assignments = [1, 2]
        counts = [1, 1]
    else
        error("unknown subtraction-overflow branch")
    end
    bank = GRAMISKernelIS._first_order_gramis_factor_bank(ProposalBank([
        FactorGaussian(T[location], reshape(T[1], 1, 1)) for location in locations
    ]))
    sample_count = length(assignments)
    samples = fill(T(101), 1, sample_count)
    returned_logweights = fill(T(102), sample_count)
    local_logweights = fill(T(103), sample_count)
    generating_logdensities = fill(T(104), sample_count)
    proposal_ids = fill(105, sample_count)
    round_ids = fill(106, sample_count)
    failures = zeros(UInt64, 3)
    denominator = GRAMISKernelIS._RealizedMixtureDenominator(
        reshape(log.(T.(counts) ./ T(sample_count)), :, 1),
        1,
    )
    target_value = T(0.75) * floatmax(T)
    target = GRAMISKernelIS._NativeDeviceTarget{
        T,
        GRAMISSubtractionOverflowTarget{T},
    }(GRAMISSubtractionOverflowTarget(target_value))
    GRAMISKernelIS._first_order_gramis_sample_round!(
        samples,
        returned_logweights,
        local_logweights,
        generating_logdensities,
        proposal_ids,
        round_ids,
        failures,
        normals,
        target,
        bank,
        assignments,
        denominator,
        GRAMISKernelIS._allocate_mis_solve_scratch(normals, bank, sample_count),
        7,
        execution,
        GRAMISKernelIS.MLDataDevices.CPUDevice(),
        FusedFactorExecution(),
    )
    return (;
        samples,
        returned_logweights,
        local_logweights,
        generating_logdensities,
        proposal_ids,
        round_ids,
        failures,
        target_value,
    )
end

@testset "serial fused subtraction failures match the portable path" begin
    for T in (Float32, Float64), branch in (:returned, :local)
        serial = run_gram_is_subtraction_overflow(
            T,
            branch,
            GRAMISKernelIS._SerialCPUExecution(),
        )
        portable = run_gram_is_subtraction_overflow(
            T,
            branch,
            GRAMISKernelIS._ThreadedCPUExecution(),
        )
        for field in (
            :samples,
            :returned_logweights,
            :local_logweights,
            :generating_logdensities,
            :proposal_ids,
            :round_ids,
            :failures,
        )
            @test getproperty(serial, field) == getproperty(portable, field)
        end
        decoded = GRAMISKernelIS._decode_native_failure(
            serial.failures[1],
            serial.failures[2],
        )
        @test decoded.count == 1
        @test decoded.first_logical_index == 1
        @test decoded.reason_bits == GRAMISKernelIS._NATIVE_LOGWEIGHT_INVALID
        @test iszero(serial.failures[3])

        if branch === :returned
            @test serial.returned_logweights == fill(T(-Inf), 1)
            @test serial.local_logweights == fill(T(-Inf), 1)
            @test serial.generating_logdensities == fill(T(-Inf), 1)
            @test serial.proposal_ids == [0]
            @test serial.round_ids == [0]
        else
            @test isfinite(serial.returned_logweights[1])
            @test serial.local_logweights[1] == serial.target_value
            @test isfinite(serial.generating_logdensities[1])
            @test serial.proposal_ids[1] == 1
            @test serial.round_ids[1] == 0
            @test serial.proposal_ids[2] == 2
            @test serial.round_ids[2] == 7
        end
    end
end

@testset "failed frozen-round samples clear fused and batched adaptation state" begin
    failed_sample = 3
    for T in (Float32, Float64)
        results = (
            fused_serial=run_gram_is_frozen_round(
                T,
                GRAMISKernelIS._SerialCPUExecution(),
                FusedFactorExecution();
                failed_sample,
            ),
            fused_threaded=run_gram_is_frozen_round(
                T,
                GRAMISKernelIS._ThreadedCPUExecution(),
                FusedFactorExecution();
                failed_sample,
            ),
            batched_serial=run_gram_is_frozen_round(
                T,
                GRAMISKernelIS._SerialCPUExecution(),
                BatchedFactorExecution();
                failed_sample,
            ),
            batched_threaded=run_gram_is_frozen_round(
                T,
                GRAMISKernelIS._ThreadedCPUExecution(),
                BatchedFactorExecution();
                failed_sample,
            ),
        )

        for result in results
            decoded = GRAMISKernelIS._decode_native_failure(
                result.failures[1],
                result.failures[2],
            )
            @test result.returned_logweights[failed_sample] == T(-Inf)
            @test result.local_logweights[failed_sample] == T(-Inf)
            @test result.generating_logdensities[failed_sample] == T(-Inf)
            @test iszero(result.proposal_ids[failed_sample])
            @test iszero(result.round_ids[failed_sample])
            @test decoded.first_logical_index == failed_sample
            @test decoded.reason_bits == GRAMISKernelIS._NATIVE_GENERATED_NONFINITE
        end

        reference = results.batched_serial
        for result in results
            @test result.returned_logweights[failed_sample] ==
                  reference.returned_logweights[failed_sample]
            @test result.local_logweights[failed_sample] ==
                  reference.local_logweights[failed_sample]
            @test result.generating_logdensities[failed_sample] ==
                  reference.generating_logdensities[failed_sample]
            @test result.proposal_ids[failed_sample] ==
                  reference.proposal_ids[failed_sample]
            @test result.round_ids[failed_sample] == reference.round_ids[failed_sample]
            @test result.failures == reference.failures
        end
    end
end

@testset "FirstOrderGRAMIS frozen round uses realized and local denominators" begin
    for T in (Float32, Float64), factor_execution in (
        FusedFactorExecution(),
        BatchedFactorExecution(),
    )
        serial = run_gram_is_frozen_round(
            T,
            GRAMISKernelIS._SerialCPUExecution(),
            factor_execution,
        )
        threaded = run_gram_is_frozen_round(
            T,
            GRAMISKernelIS._ThreadedCPUExecution(),
            factor_execution,
        )
        oracle = serial.oracle

        @test vec(serial.samples) ≈ oracle.samples rtol = 8eps(T)
        @test serial.returned_logweights ≈
              oracle.returned_logweights rtol = 32eps(T)
        @test serial.local_logweights ≈ oracle.local_logweights rtol = 32eps(T)
        @test serial.generating_logdensities ≈
              oracle.generating_logdensities rtol = 16eps(T)
        @test serial.local_logweights .+ serial.generating_logdensities ≈
              oracle.logtargets rtol = 16eps(T)
        @test serial.proposal_ids == oracle.proposal_ids
        @test serial.round_ids == fill(7, 10)
        @test iszero(serial.failures)
        @test serial.target_calls == 10
        @test serial.samples == threaded.samples
        @test serial.returned_logweights == threaded.returned_logweights
        @test serial.local_logweights == threaded.local_logweights
        @test serial.generating_logdensities == threaded.generating_logdensities
        @test serial.proposal_ids == threaded.proposal_ids
        @test serial.round_ids == threaded.round_ids
        @test serial.failures == threaded.failures
        @test threaded.target_calls == 10

        returned_denominators = oracle.logtargets .- serial.returned_logweights
        local_denominators = oracle.logtargets .- serial.local_logweights
        @test returned_denominators ≈
              oracle.logtargets .- oracle.returned_logweights rtol = 16eps(T)
        @test local_denominators ≈ oracle.generating_logdensities rtol = 16eps(T)
        @test any(!isapprox(returned_denominators[index], local_denominators[index]) for
                  index in eachindex(returned_denominators))
    end
end

@testset "realized-mixture finite-law conditional unbiasedness identity" begin
    counts = [4, 3, 3]
    total = 10
    proposals = Rational{Int}[
        1//2 1//4 1//4
        1//4 1//2 1//4
        1//4 1//4 1//2
    ]
    pi_h = Rational{Int}[2//5, -1//7, 3//11]
    psi = [
        sum((counts[slot] // total) * proposals[slot, point] for slot in 1:3) for
        point in 1:3
    ]
    pointwise = [
        sum(
            (counts[slot] // total) * proposals[slot, point] *
            pi_h[point] / psi[point] for slot in 1:3
        ) for point in 1:3
    ]

    @test vec(sum(proposals; dims=2)) == fill(1//1, 3)
    @test psi == Rational{Int}[7//20, 13//40, 13//40]
    @test pointwise == pi_h
    @test sum(pointwise) == 204//385

    round_sizes = [10, 13]
    round_sums = Rational{Int}[17//6, -5//4]
    combined = sum(
        (round_sizes[round] // sum(round_sizes)) *
        (round_sums[round] / round_sizes[round]) for round in 1:2
    )
    @test combined == sum(round_sums) / sum(round_sizes) == 19//276
end

@testset "FirstOrderGRAMIS preconditions gradients with the frozen covariance" begin
    for T in (Float32, Float64), execution in (
        GRAMISKernelIS._SerialCPUExecution(),
        GRAMISKernelIS._ThreadedCPUExecution(),
    )
        factors = zeros(T, 2, 2, 2)
        factors[:, :, 1] .= T[2 0; 1 3]
        factors[:, :, 2] .= T[1 0; -2 4]
        gradients = T[4 -3; -2 5]
        moves = fill(T(99), 2, 2)

        @test @inferred(GRAMISKernelIS._precondition_gradients!(
            moves,
            gradients,
            factors,
            execution,
        )) === nothing

        # Literal products by Sigma = L * L', not solves by Sigma or L.
        @test moves == T[12 -13; -12 106]
        @test gradients == T[4 -3; -2 5]
        @test factors[:, :, 1] == T[2 0; 1 3]
        @test factors[:, :, 2] == T[1 0; -2 4]
    end
end

@testset "FirstOrderGRAMIS preconditioning uses two triangular passes" begin
    proposal_count = 3
    for dimension in (2, 5, 9)
        factors = zeros(Float64, dimension, dimension, proposal_count)
        gradients = Matrix{Float64}(undef, dimension, proposal_count)
        for proposal_slot in 1:proposal_count
            for column in 1:dimension, row in column:dimension
                factors[row, column, proposal_slot] =
                    (row + 2column + proposal_slot) / 10
            end
            gradients[:, proposal_slot] .=
                (1:dimension) .- (dimension + proposal_slot) / 3
        end
        reads = Ref(0)
        counted_factors = GRAMISReadCountingArray(factors, reads)
        moves = similar(gradients)

        GRAMISKernelIS._precondition_gradients!(
            moves,
            gradients,
            counted_factors,
            GRAMISKernelIS._SerialCPUExecution(),
        )

        for proposal_slot in 1:proposal_count
            factor = factors[:, :, proposal_slot]
            @test moves[:, proposal_slot] ≈
                  factor * (factor' * gradients[:, proposal_slot])
        end
        @test reads[] == proposal_count * dimension * (dimension + 1)
    end
end

@testset "FirstOrderGRAMIS evaluates each frozen gradient once" begin
    for T in (Float32, Float64), execution in (
        GRAMISKernelIS._SerialCPUExecution(),
        GRAMISKernelIS._ThreadedCPUExecution(),
    )
        locations = T[1 3; -2 4]
        values = fill(T(99), 2)
        gradients = fill(T(98), 2, 2)
        value_operation = GRAMISCountedFrozenValue{T}(Threads.Atomic{Int}(0))
        gradient_operation =
            GRAMISCountedFrozenGradient{T}(Threads.Atomic{Int}(0))
        prepared = GRAMISKernelIS._prepare_target(
            LogTarget(value_operation; grad=gradient_operation),
            FactorGaussian(T[0, 0], Matrix{T}(LinearAlgebra.I, 2, 2)),
        )
        bound_value = GRAMISKernelIS._bind_resolved_target(
            prepared,
            view(locations, :, 1),
        )
        worker_count = execution isa GRAMISKernelIS._SerialCPUExecution ?
                       1 : length(Threads.threadpooltids(:default))
        bound_gradient = GRAMISKernelIS._prepare_bound_gradient(
            prepared,
            view(locations, :, 1),
            worker_count,
        )

        @test @inferred(GRAMISKernelIS._evaluate_frozen_gradients!(
            values,
            gradients,
            bound_value,
            bound_gradient,
            locations,
            execution,
        )) === nothing

        @test values == T[-1.5, -8.5]
        @test gradients == T[-1 -3; 1 -2]
        @test value_operation.calls[] == 2
        @test gradient_operation.calls[] == 2
        @test locations == T[1 3; -2 4]
    end
end

@testset "FirstOrderGRAMIS backtracking boundaries and inactive proposals" begin
    for T in (Float32, Float64), execution in (
        GRAMISKernelIS._SerialCPUExecution(),
        GRAMISKernelIS._ThreadedCPUExecution(),
    )
        locations = reshape(T[0, 10, 20], 1, :)
        moves = ones(T, 1, 3)
        frozen_values = zeros(T, 3)
        candidate_locations = fill(T(99), 1, 3)
        candidate_values = fill(T(98), 3)
        active_mask = fill(false, 3)
        steps = fill(T(97), 3)
        trials = fill(96, 3)
        target = GRAMISBacktrackingTarget{T}([
            Threads.Atomic{Int}(0) for _ in 1:3
        ])

        @test @inferred(GRAMISKernelIS._backtrack_means!(
            candidate_locations,
            candidate_values,
            active_mask,
            steps,
            trials,
            target,
            frozen_values,
            locations,
            moves,
            3,
            execution,
        )) === nothing

        @test candidate_locations == reshape(T[1, 10.25, 20], 1, :)
        @test candidate_values == zeros(T, 3)
        @test active_mask == fill(false, 3)
        @test steps == T[1, 0.25, 0]
        @test trials == [1, 3, 3]
        @test getindex.(target.calls) == [1, 3, 3]
        @test locations == reshape(T[0, 10, 20], 1, :)
        @test moves == ones(T, 1, 3)
    end
end

@testset "FirstOrderGRAMIS CPU backtracking stops after all proposals accept" begin
    for T in (Float32, Float64), execution in (
        GRAMISKernelIS._SerialCPUExecution(),
        GRAMISKernelIS._ThreadedCPUExecution(),
    )
        locations = reshape(T[0, 10], 1, :)
        moves = ones(T, 1, 2)
        frozen_values = zeros(T, 2)
        candidate_locations = similar(locations)
        candidate_values = similar(frozen_values)
        active_mask = GRAMISAtomicReadVector(
            fill(false, 2),
            Threads.Atomic{Int}(0),
        )
        steps = similar(frozen_values)
        trials = similar(frozen_values, Int)
        target = GRAMISAlwaysAcceptTarget{T}(Threads.Atomic{Int}(0))

        @test @inferred(GRAMISKernelIS._backtrack_means!(
            candidate_locations,
            candidate_values,
            active_mask,
            steps,
            trials,
            target,
            frozen_values,
            locations,
            moves,
            20,
            execution,
        )) === nothing

        @test candidate_locations == locations .+ moves
        @test candidate_values == frozen_values
        @test active_mask.storage == fill(false, 2)
        @test steps == ones(T, 2)
        @test trials == ones(Int, 2)
        @test target.calls[] == 2
        @test active_mask.reads[] == 6
    end
end

function gram_is_gradient_move_allocation_counts()
    T = Float64
    execution = GRAMISKernelIS._SerialCPUExecution()
    locations = reshape(T[-1, 1], 1, :)
    values = zeros(T, 2)
    gradients = zeros(T, 1, 2)
    factors = ones(T, 1, 1, 2)
    moves = zeros(T, 1, 2)
    candidate_locations = similar(locations)
    candidate_values = similar(values)
    active_mask = similar(values, Bool)
    steps = similar(values)
    trials = similar(values, Int)
    value_operation = GRAMISRoundOneValue{T}(Threads.Atomic{Int}(0))
    gradient_operation = GRAMISRoundOneGradient(Threads.Atomic{Int}(0))
    prepared = GRAMISKernelIS._prepare_target(
        LogTarget(value_operation; grad=gradient_operation),
        FactorGaussian(T[0], reshape(T[1], 1, 1)),
    )
    target = GRAMISKernelIS._bind_resolved_target(
        prepared,
        view(locations, :, 1),
    )
    gradient = GRAMISKernelIS._prepare_bound_gradient(
        prepared,
        view(locations, :, 1),
        1,
    )

    GRAMISKernelIS._evaluate_frozen_gradients!(
        values,
        gradients,
        target,
        gradient,
        locations,
        execution,
    )
    GRAMISKernelIS._precondition_gradients!(
        moves,
        gradients,
        factors,
        execution,
    )
    GRAMISKernelIS._backtrack_means!(
        candidate_locations,
        candidate_values,
        active_mask,
        steps,
        trials,
        target,
        values,
        locations,
        moves,
        4,
        execution,
    )

    gradient_allocations = @allocated GRAMISKernelIS._evaluate_frozen_gradients!(
        values,
        gradients,
        target,
        gradient,
        locations,
        execution,
    )
    preconditioner_allocations = @allocated GRAMISKernelIS._precondition_gradients!(
        moves,
        gradients,
        factors,
        execution,
    )
    backtracking_allocations = @allocated GRAMISKernelIS._backtrack_means!(
        candidate_locations,
        candidate_values,
        active_mask,
        steps,
        trials,
        target,
        values,
        locations,
        moves,
        4,
        execution,
    )
    return (
        gradient_allocations,
        preconditioner_allocations,
        backtracking_allocations,
    )
end

@testset "FirstOrderGRAMIS serial gradient move reuses workspaces" begin
    @test gram_is_gradient_move_allocation_counts() == (0, 0, 0)
end

@testset "FirstOrderGRAMIS pooled covariance is the mean frozen covariance" begin
    for T in (Float32, Float64), dimension in (1, 2, 4), execution in (
        GRAMISKernelIS._SerialCPUExecution(),
        GRAMISKernelIS._ThreadedCPUExecution(),
    )
        proposal_count = 3
        factors = zeros(T, dimension, dimension, proposal_count)
        expected = zeros(T, dimension, dimension)
        for proposal_slot in 1:proposal_count
            for column in 1:dimension, row in column:dimension
                factors[row, column, proposal_slot] =
                    T(row + 2column + proposal_slot) / T(7)
            end
            factor = factors[:, :, proposal_slot]
            expected .+= factor * transpose(factor)
        end
        expected ./= T(proposal_count)
        pooled = fill(T(-99), dimension, dimension)

        @test @inferred(GRAMISKernelIS._pooled_covariance!(
            pooled,
            factors,
            execution,
        )) === nothing
        @test pooled ≈ expected rtol = 16eps(T)
        @test pooled ≈ transpose(pooled) rtol = 16eps(T)
    end
end

@testset "FirstOrderGRAMIS whitens one complete mean matrix in place" begin
    for T in (Float32, Float64), dimension in (1, 2, 5)
        proposal_count = 4
        covariance = Matrix{T}(LinearAlgebra.I, dimension, dimension)
        for row in 1:dimension
            covariance[row, row] = T(row + 1)
        end
        for row in 2:dimension
            covariance[row, 1] = covariance[1, row] = T(row) / T(10)
        end
        factor_storage = Matrix(LinearAlgebra.cholesky(
            LinearAlgebra.Hermitian(covariance),
        ).L)
        factorizations = Ref(0)
        solves = Ref(0)
        counted_factor = GRAMISOperationCountingMatrix(
            factor_storage,
            factorizations,
            solves,
        )
        means = reshape(
            T.(1:(dimension * proposal_count)),
            dimension,
            proposal_count,
        ) / T(3)
        frozen_means = copy(means)
        whitened = fill(T(-99), size(means))
        expected = LinearAlgebra.LowerTriangular(factor_storage) \ means

        @test @inferred(GRAMISKernelIS._whiten_means!(
            whitened,
            counted_factor,
            means,
        )) === nothing
        @test whitened ≈ expected rtol = 32eps(T)
        @test means == frozen_means
        @test factorizations[] == 0
        @test solves[] == 1
    end
end

function gram_is_package_repulsion_oracle(
    means,
    factors,
    strength,
    softening,
)
    T = eltype(means)
    dimension, proposal_count = size(means)
    pooled_covariance = zeros(T, dimension, dimension)
    for proposal_slot in 1:proposal_count
        factor = factors[:, :, proposal_slot]
        pooled_covariance .+= factor * transpose(factor)
    end
    pooled_covariance ./= T(proposal_count)
    pooled_factor = LinearAlgebra.cholesky(
        LinearAlgebra.Hermitian(pooled_covariance),
    ).L
    whitened = pooled_factor \ means
    forces = zeros(T, size(means))
    collision_counts = zeros(Int, proposal_count)
    softening2 = abs2(softening)
    for proposal_slot in 1:proposal_count
        for peer_slot in 1:proposal_count
            peer_slot == proposal_slot && continue
            distance2 = zero(T)
            for row in 1:dimension
                distance2 += abs2(
                    whitened[row, proposal_slot] - whitened[row, peer_slot],
                )
            end
            iszero(distance2) && (collision_counts[proposal_slot] += 1)
            denominator =
                (distance2 + softening2) ^ (T(dimension) / T(2))
            for row in 1:dimension
                forces[row, proposal_slot] +=
                    (means[row, proposal_slot] - means[row, peer_slot]) /
                    denominator
            end
        end
    end
    forces .*= strength / T(proposal_count - 1)
    return (; forces, collision_counts, pooled_covariance, whitened)
end

function gram_is_repulsion_fixture(::Type{T}, dimension, proposal_count) where {T}
    means = Matrix{T}(undef, dimension, proposal_count)
    factors = zeros(T, dimension, dimension, proposal_count)
    for proposal_slot in 1:proposal_count
        for row in 1:dimension
            means[row, proposal_slot] =
                T(3row - 2proposal_slot + row * proposal_slot) / T(5)
        end
        for column in 1:dimension, row in column:dimension
            factors[row, column, proposal_slot] = row == column ?
                T(row + proposal_slot + 2) / T(3) :
                T(row - column + proposal_slot) / T(11)
        end
    end
    return means, factors
end

function gram_is_run_repulsion(
    means,
    factors,
    strength,
    softening,
    execution;
    pooled_covariance=zeros(eltype(means), size(means, 1), size(means, 1)),
)
    forces = similar(means)
    whitened = similar(means)
    collision_counts = Vector{Int}(undef, size(means, 2))
    GRAMISKernelIS._repulsion!(
        forces,
        collision_counts,
        pooled_covariance,
        whitened,
        means,
        factors,
        strength,
        softening,
        execution,
    )
    return (; forces, collision_counts, pooled_covariance, whitened)
end

@testset "FirstOrderGRAMIS package repulsion matches an independent oracle" begin
    for T in (Float32, Float64), dimension in (1, 2, 3, 5), proposal_count in (2, 4)
        means, factors = gram_is_repulsion_fixture(T, dimension, proposal_count)
        strength = T(0.7)
        softening = T(0.4)
        oracle = gram_is_package_repulsion_oracle(
            means,
            factors,
            strength,
            softening,
        )
        serial = gram_is_run_repulsion(
            means,
            factors,
            strength,
            softening,
            GRAMISKernelIS._SerialCPUExecution(),
        )
        threaded = gram_is_run_repulsion(
            means,
            factors,
            strength,
            softening,
            GRAMISKernelIS._ThreadedCPUExecution(),
        )

        @test serial.forces ≈ oracle.forces rtol = 128eps(T)
        @test serial.collision_counts == oracle.collision_counts
        @test threaded.forces == serial.forces
        @test threaded.collision_counts == serial.collision_counts
    end
end

@testset "FirstOrderGRAMIS repulsion is affine covariant" begin
    for T in (Float32, Float64), dimension in (2, 4), execution in (
        GRAMISKernelIS._SerialCPUExecution(),
        GRAMISKernelIS._ThreadedCPUExecution(),
    )
        means, factors = gram_is_repulsion_fixture(T, dimension, 4)
        transform = Matrix{T}(LinearAlgebra.I, dimension, dimension)
        for row in 1:dimension
            transform[row, row] = T(row + 1) / T(2)
        end
        for row in 2:dimension
            transform[row, 1] = T(row) / T(7)
        end
        translation = T.(1:dimension) / T(3)
        transformed_means = transform * means .+ translation
        transformed_factors = similar(factors)
        for proposal_slot in axes(factors, 3)
            transformed_covariance =
                transform * factors[:, :, proposal_slot] *
                transpose(factors[:, :, proposal_slot]) * transpose(transform)
            transformed_factors[:, :, proposal_slot] .= Matrix(
                LinearAlgebra.cholesky(
                    LinearAlgebra.Hermitian(transformed_covariance),
                ).L,
            )
        end
        strength = T(0.35)
        softening = T(0.8)
        original = gram_is_run_repulsion(
            means,
            factors,
            strength,
            softening,
            execution,
        )
        transformed = gram_is_run_repulsion(
            transformed_means,
            transformed_factors,
            strength,
            softening,
            execution,
        )

        @test transformed.forces ≈ transform * original.forces rtol = 512eps(T)
        @test transformed.collision_counts == original.collision_counts
    end
end

@testset "FirstOrderGRAMIS exact and near collisions are bounded" begin
    for T in (Float32, Float64), dimension in (1, 2, 5), execution in (
        GRAMISKernelIS._SerialCPUExecution(),
        GRAMISKernelIS._ThreadedCPUExecution(),
    )
        softening = T(0.75)
        factors = zeros(T, dimension, dimension, 2)
        for proposal_slot in 1:2, row in 1:dimension
            factors[row, row, proposal_slot] = one(T)
        end
        exact_means = zeros(T, dimension, 2)
        exact = gram_is_run_repulsion(
            exact_means,
            factors,
            one(T),
            softening,
            execution,
        )
        @test exact.forces == zeros(T, dimension, 2)
        @test exact.collision_counts == [1, 1]

        near_means = copy(exact_means)
        near_means[1, 2] = softening / T(2)
        near = gram_is_run_repulsion(
            near_means,
            factors,
            one(T),
            softening,
            execution,
        )
        @test all(isfinite, near.forces)
        @test near.forces[:, 1] == -near.forces[:, 2]
        @test near.collision_counts == [0, 0]
    end
end

@testset "FirstOrderGRAMIS collision detection preserves subnormal distinctions" begin
    for T in (Float32, Float64), execution in (
        GRAMISKernelIS._SerialCPUExecution(),
        GRAMISKernelIS._ThreadedCPUExecution(),
    )
        smallest_positive = nextfloat(zero(T))
        factors = ones(T, 1, 1, 2)

        @testset "$(T), $(typeof(execution)), exact" begin
            exact = gram_is_run_repulsion(
                zeros(T, 1, 2),
                factors,
                one(T),
                smallest_positive,
                execution,
            )
            @test exact.forces == zeros(T, 1, 2)
            @test exact.collision_counts == [1, 1]
        end

        @testset "$(T), $(typeof(execution)), separated" begin
            separated = gram_is_run_repulsion(
                reshape(T[0, smallest_positive], 1, 2),
                factors,
                one(T),
                one(T),
                execution,
            )
            @test separated.forces ==
                  reshape(T[-smallest_positive, smallest_positive], 1, 2)
            @test separated.collision_counts == [0, 0]
        end
    end
end

@testset "FirstOrderGRAMIS active repulsion performs one factor and one solve" begin
    T = Float64
    dimension = 3
    proposal_count = 4
    means, factors = gram_is_repulsion_fixture(T, dimension, proposal_count)
    factor_reads = Ref(0)
    mean_reads = Ref(0)
    counted_factors = GRAMISReadCountingArray(factors, factor_reads)
    counted_means = GRAMISReadCountingArray(means, mean_reads)
    factorizations = Ref(0)
    solves = Ref(0)
    pooled = GRAMISOperationCountingMatrix(
        zeros(T, dimension, dimension),
        factorizations,
        solves,
    )

    forces = similar(means)
    whitened = similar(means)
    collision_counts = Vector{Int}(undef, proposal_count)
    @test @inferred(GRAMISKernelIS._repulsion!(
        forces,
        collision_counts,
        pooled,
        whitened,
        counted_means,
        counted_factors,
        T(0.4),
        T(0.5),
        GRAMISKernelIS._SerialCPUExecution(),
    )) === nothing
    expected_factor_reads = proposal_count * sum(
        2min(row, column) for row in 1:dimension, column in 1:dimension
    )
    expected_mean_reads =
        dimension * proposal_count +
        2dimension * proposal_count * (proposal_count - 1)

    @test all(isfinite, forces)
    @test factorizations[] == 1
    @test solves[] == 1
    @test factor_reads[] == expected_factor_reads
    @test mean_reads[] == expected_mean_reads
end

@testset "FirstOrderGRAMIS zero strength skips every repulsion operation" begin
    T = Float64
    dimension = 3
    proposal_count = 4
    means, factors = gram_is_repulsion_fixture(T, dimension, proposal_count)
    factor_reads = Ref(0)
    mean_reads = Ref(0)
    factorizations = Ref(0)
    solves = Ref(0)
    pooled_storage = fill(T(-91), dimension, dimension)
    pooled = GRAMISOperationCountingMatrix(
        pooled_storage,
        factorizations,
        solves,
    )
    whitened = fill(T(-92), dimension, proposal_count)
    forces = fill(T(-93), dimension, proposal_count)
    collision_counts = fill(-94, proposal_count)

    @test @inferred(GRAMISKernelIS._repulsion!(
        forces,
        collision_counts,
        pooled,
        whitened,
        GRAMISReadCountingArray(means, mean_reads),
        GRAMISReadCountingArray(factors, factor_reads),
        zero(T),
        T(0.5),
        GRAMISKernelIS._SerialCPUExecution(),
    )) === nothing
    @test forces == zeros(T, dimension, proposal_count)
    @test collision_counts == fill(
        GRAMISKernelIS._GRAMIS_COLLISIONS_UNAVAILABLE,
        proposal_count,
    )
    @test pooled_storage == fill(T(-91), dimension, dimension)
    @test whitened == fill(T(-92), dimension, proposal_count)
    @test factorizations[] == 0
    @test solves[] == 0
    @test factor_reads[] == 0
    @test mean_reads[] == 0
end

@testset "FirstOrderGRAMIS nonfinite force raises a typed repulsion failure" begin
    for T in (Float32, Float64)
        means = zeros(T, 2, 2)
        means[1, 2] = eps(T)
        factors = zeros(T, 2, 2, 2)
        factors[:, :, 1] .= Matrix{T}(LinearAlgebra.I, 2, 2)
        factors[:, :, 2] .= Matrix{T}(LinearAlgebra.I, 2, 2)
        error = try
            gram_is_run_repulsion(
                means,
                factors,
                floatmax(T),
                eps(T),
                GRAMISKernelIS._SerialCPUExecution(),
            )
            nothing
        catch cause
            cause
        end
        @test error isa GRAMISKernelIS._FirstOrderGRAMISRepulsionError
        @test error.reason == :force_nonfinite
    end
end

function gram_is_repulsion_allocation_counts()
    T = Float64
    means, factors = gram_is_repulsion_fixture(T, 3, 4)
    forces = similar(means)
    collision_counts = Vector{Int}(undef, 4)
    pooled_covariance = zeros(T, 3, 3)
    whitened = similar(means)
    execution = GRAMISKernelIS._SerialCPUExecution()
    GRAMISKernelIS._repulsion!(
        forces,
        collision_counts,
        pooled_covariance,
        whitened,
        means,
        factors,
        T(0.4),
        T(0.5),
        execution,
    )
    return @allocated GRAMISKernelIS._repulsion!(
        forces,
        collision_counts,
        pooled_covariance,
        whitened,
        means,
        factors,
        T(0.4),
        T(0.5),
        execution,
    )
end

@testset "FirstOrderGRAMIS serial repulsion reuses every workspace" begin
    @test gram_is_repulsion_allocation_counts() == 0
end
