using Test
using ImportanceSamplers
import DensityInterface
import Random

const ISK = ImportanceSamplers

struct StaticMISKernelExternalGaussian{T<:AbstractFloat}
    location::T
end

struct StaticMISKernelScalarTarget{T} end
function (::StaticMISKernelScalarTarget{T})(sample::T)::T where {T}
    return -abs2(sample) / T(3)
end

struct StaticMISKernelVectorTarget{T} end
function (::StaticMISKernelVectorTarget{T})(sample::AbstractVector{T})::T where {T}
    return -sum(abs2, sample) / T(3)
end

struct StaticMISKernelTargetFailure <: Exception
    sample::Float64
end

mutable struct StaticMISKernelFailTarget
    fail::Bool
    sample::Float64
end

function (target::StaticMISKernelFailTarget)(sample::Float64)::Float64
    target.fail && sample == target.sample && throw(
        StaticMISKernelTargetFailure(sample),
    )
    return -abs2(sample)
end

struct StaticMISKernelConstantTarget{T}
    value::T
end
(target::StaticMISKernelConstantTarget{T})(::T) where {T} = target.value

mutable struct StaticMISKernelPrefilledRNG{T} <: Random.AbstractRNG
    uniforms::Vector{T}
    normals::Vector{T}
    calls::Vector{Symbol}
end

function Random.rand!(rng::StaticMISKernelPrefilledRNG{T}, values::Vector{T}) where {T}
    push!(rng.calls, :uniform)
    copyto!(values, rng.uniforms)
    return values
end

function Random.randn!(rng::StaticMISKernelPrefilledRNG{T}, values::Vector{T}) where {T}
    push!(rng.calls, :normal)
    copyto!(values, rng.normals)
    return values
end

function caught_static_mis_kernel_failure(f)
    try
        f()
    catch error
        return error
    end
    return nothing
end

function Random.rand(
    rng::Random.AbstractRNG,
    proposal::StaticMISKernelExternalGaussian{T},
) where {T}
    return proposal.location + randn(rng, T)
end

function DensityInterface.logdensityof(
    proposal::StaticMISKernelExternalGaussian,
    sample::Real,
)
    offset = sample - proposal.location
    return -oftype(offset, 0.5) * abs2(offset) - oftype(offset, 0.5 * log(2pi))
end

@testset "packed native Gaussian proposal banks" begin
    for T in (Float32, Float64)
        scalar_bank = ProposalBank(
            [
                SphericalGaussian(T(-2), T(0.5)),
                SphericalGaussian(T(3), T(2)),
            ],
            T[3, 1],
        )
        scalar = @inferred ISK._pack_native_gaussian_bank(scalar_bank)

        @test scalar isa ISK._PackedDiagonalGaussianBank
        @test scalar.layout isa ISK._ScalarGaussianLayout
        @test size(scalar.locations) == (1, 2)
        @test size(scalar.scales) == (1, 2)
        @test scalar.locations == reshape(T[3, -2], 1, 2)
        @test scalar.scales == reshape(T[2, 0.5], 1, 2)
        @test scalar.proposal_ids == [2, 1]
        @test exp.(scalar.logmasses) ≈ T[0.25, 0.75]
        @test scalar.cdf == T[0.25, 1]

        vector_bank = ProposalBank(
            Any[
                SphericalGaussian(T[-2, -1], T(0.5)),
                SphericalGaussian(T[99, 99, 99], T(1)),
                DiagonalGaussian(T[3, 4], T[2, 3]),
            ],
            T[3, 0, 1],
        )
        vector = ISK._pack_native_gaussian_bank(vector_bank)

        @test vector isa ISK._PackedDiagonalGaussianBank
        @test vector.layout isa ISK._VectorGaussianLayout
        @test size(vector.locations) == (2, 2)
        @test size(vector.scales) == (2, 2)
        @test vector.locations == T[3 -2; 4 -1]
        @test vector.scales == T[2 0.5; 3 0.5]
        @test vector.proposal_ids == [3, 1]
        @test exp.(vector.logmasses) ≈ T[0.25, 0.75]
        @test vector.cdf == T[0.25, 1]
    end

end

@testset "packed native Gaussian bank validation precedes RNG" begin
    invalid_banks = (
        ProposalBank(
            Any[
                SphericalGaussian(0.0, 1.0),
                SphericalGaussian([0.0], 1.0),
            ],
        ),
        ProposalBank(
            Any[
                SphericalGaussian(zeros(2), 1.0),
                DiagonalGaussian(zeros(3), ones(3)),
            ],
        ),
        ProposalBank(
            Any[
                SphericalGaussian(zeros(Float32, 2), 1.0f0),
                DiagonalGaussian(zeros(Float64, 2), ones(Float64, 2)),
            ],
        ),
    )

    for (index, bank) in pairs(invalid_banks)
        rng = Random.Xoshiro(0x5400 + index)
        expected_rng = copy(rng)
        error_type = index == 2 ? DimensionMismatch : ArgumentError
        @test_throws error_type prepare_sampler(
            rng,
            sample -> -sum(abs2, sample),
            ImportanceSampling(bank; nsamples=8);
            threaded=false,
        )
        @test rand(rng) == rand(expected_rng)
    end
end

@testset "generic proposal banks retain generic preparation" begin
    proposals = [
        StaticMISKernelExternalGaussian(-1.0),
        StaticMISKernelExternalGaussian(1.0),
    ]
    sampler = @inferred prepare_sampler(
        Random.Xoshiro(0x5404),
        sample -> -abs2(sample) / 2,
        ImportanceSampling(ProposalBank(proposals); nsamples=8);
        threaded=false,
    )

    @test sampler.method_state.bank isa ISK._ActiveProposalBank
    @test sampler.random_buffers isa ISK._StaticMISRandomBuffers

    factor_sampler = prepare_sampler(
        Random.Xoshiro(0x5405),
        StaticMISKernelVectorTarget{Float64}(),
        ImportanceSampling(
            ProposalBank(
                [
                    FactorGaussian([0.0, 0.0], [1.0 0.0; 0.25 1.0]),
                    FactorGaussian([1.0, 1.0], [1.0 0.0; 0.25 1.0]),
                ],
            );
            nsamples=4,
        );
        threaded=false,
    )
    transformed_sampler = prepare_sampler(
        Random.Xoshiro(0x5406),
        StaticMISKernelVectorTarget{Float64}(),
        ImportanceSampling(
            ProposalBank(
                [
                    TransformedProposal(
                        SphericalGaussian([0.0, 0.0], 1.0),
                        IdentityTransform(),
                    ),
                    TransformedProposal(
                        SphericalGaussian([1.0, 1.0], 1.0),
                        IdentityTransform(),
                    ),
                ],
            );
            nsamples=4,
        );
        threaded=false,
    )
    product_sampler = prepare_sampler(
        Random.Xoshiro(0x5407),
        sample -> -abs2(sample.x),
        ImportanceSampling(
            ProposalBank(
                [
                    ProductProposal((x=SphericalGaussian(0.0, 1.0),)),
                    ProductProposal((x=SphericalGaussian(1.0, 1.0),)),
                ],
            );
            nsamples=4,
        );
        threaded=false,
    )

    for generic_sampler in (
        factor_sampler,
        transformed_sampler,
        product_sampler,
    )
        @test generic_sampler.method_state.bank isa ISK._ActiveProposalBank
        @test length(importance_sample!(generic_sampler)) == 4
    end
end

function static_mis_kernel_assignment(uniform, sample_index, nsamples, scheme, cdf)
    selected = scheme isa RandomMixture ?
               uniform : ((sample_index - 1) + uniform) / nsamples
    return searchsortedfirst(cdf, selected)
end

function static_mis_kernel_sample(bank, normals, sample_index, slot)
    dimension = size(bank.locations, 1)
    offset = (sample_index - 1) * dimension
    values = [
        bank.locations[coordinate, slot] +
        bank.scales[coordinate, slot] * normals[offset + coordinate] for
        coordinate in 1:dimension
    ]
    return bank.layout isa ISK._ScalarGaussianLayout ? only(values) : values
end

function static_mis_kernel_logdensity(bank, sample, slot)
    squared_radius = if sample isa Real
        abs2((sample - bank.locations[1, slot]) / bank.scales[1, slot])
    else
        sum(eachindex(sample)) do coordinate
            abs2(
                (sample[coordinate] - bank.locations[coordinate, slot]) /
                bank.scales[coordinate, slot],
            )
        end
    end
    return bank.lognormalizers[slot] -
           oftype(bank.lognormalizers[slot], 0.5) * squared_radius
end

function static_mis_kernel_denominator(
    packed,
    configured_bank,
    sample,
    generating_slot,
    scheme,
)
    slots = if scheme isa StandardMIS
        [generating_slot]
    elseif scheme isa PartialDeterministicMixture
        proposal_id = packed.proposal_ids[generating_slot]
        group = only(filter(group -> proposal_id in group, scheme.groups))
        [
            findfirst(==(id), packed.proposal_ids) for id in group if
            !iszero(configured_bank.masses[id])
        ]
    else
        collect(eachindex(packed.proposal_ids))
    end
    coefficients = if scheme isa StandardMIS
        [one(eltype(packed.cdf))]
    elseif scheme isa PartialDeterministicMixture
        masses = [configured_bank.masses[packed.proposal_ids[slot]] for slot in slots]
        masses ./ sum(masses)
    else
        exp.(packed.logmasses[slots])
    end
    terms = [
        coefficients[index] *
        exp(static_mis_kernel_logdensity(packed, sample, slot)) for
        (index, slot) in pairs(slots)
    ]
    return log(sum(terms))
end

function static_mis_kernel_oracle(rng, bank, packed, scheme, target, nsamples)
    T = eltype(packed.locations)
    dimension = size(packed.locations, 1)
    uniforms = Vector{eltype(packed.cdf)}(undef, nsamples)
    normals = Vector{T}(undef, dimension * nsamples)
    Random.rand!(rng, uniforms)
    Random.randn!(rng, normals)

    assignments = [
        static_mis_kernel_assignment(
            uniforms[sample_index],
            sample_index,
            nsamples,
            scheme,
            packed.cdf,
        ) for sample_index in 1:nsamples
    ]
    samples = [
        static_mis_kernel_sample(packed, normals, sample_index, assignments[sample_index]) for
        sample_index in 1:nsamples
    ]
    logweights = [
        target(samples[sample_index]) - static_mis_kernel_denominator(
            packed,
            bank,
            samples[sample_index],
            assignments[sample_index],
            scheme,
        ) for sample_index in 1:nsamples
    ]
    maximum_logweight = maximum(logweights)
    reduction = maximum_logweight + log(
        sum(weight -> exp(weight - maximum_logweight), logweights) / nsamples,
    )
    return (
        assignments=assignments,
        samples=samples,
        logweights=logweights,
        proposal_ids=packed.proposal_ids[assignments],
        lognormalizer=reduction,
    )
end

@testset "packed native Gaussian CPU parity" begin
    schemes = (
        StratifiedMixture(),
        RandomMixture(),
        StandardMIS(),
        PartialDeterministicMixture(((1, 3), (2,))),
    )
    for T in (Float32, Float64), vector_layout in (false, true), scheme in schemes
        proposals = if vector_layout
            [
                SphericalGaussian(T[-2, -1], T(0.75)),
                DiagonalGaussian(T[2, 1], T[1.5, 0.5]),
                SphericalGaussian(T[0, 3], T(1.25)),
            ]
        else
            [
                SphericalGaussian(T(-2), T(0.75)),
                SphericalGaussian(T(2), T(1.5)),
                SphericalGaussian(T(0), T(1.25)),
            ]
        end
        bank = ProposalBank(proposals, T[1, 3, 2])
        target = vector_layout ?
                 (sample -> -sum(abs2, sample) / T(3)) :
                 (sample -> -abs2(sample) / T(3))
        nsamples = 37

        for threaded in (false, true)
            sampler = prepare_sampler(
                Random.Xoshiro(0x5410),
                target,
                ImportanceSampling(bank; nsamples, mis_scheme=scheme);
                threaded,
            )
            packed = sampler.method_state.bank
            oracle = static_mis_kernel_oracle(
                copy(sampler.rng),
                bank,
                packed,
                scheme,
                target,
                nsamples,
            )
            result = importance_sample!(sampler)
            result_samples = vector_layout ? eachcol(result.samples) : result.samples

            @test sampler.random_buffers isa ISK._PackedStaticMISRandomBuffers
            @test sampler.random_buffers.assignments == oracle.assignments
            @test collect(result_samples) == oracle.samples
            @test result.logweights ≈ oracle.logweights rtol = 32eps(T)
            @test result.provenance.proposal_id == oracle.proposal_ids
            @test lognormalizer(result) ≈ oracle.lognormalizer rtol = 64eps(T)
        end
    end
end

@testset "packed native Gaussian inference" begin
    for T in (Float32, Float64)
        scalar_bank = ProposalBank(
            [
                SphericalGaussian(T(-1), T(0.75)),
                SphericalGaussian(T(1), T(1.25)),
            ],
            T[1, 3],
        )
        scalar_sampler = @inferred prepare_sampler(
            Random.Xoshiro(0x5420),
            StaticMISKernelScalarTarget{T}(),
            ImportanceSampling(scalar_bank; nsamples=16);
            threaded=false,
        )
        scalar_result = @inferred importance_sample!(scalar_sampler)

        vector_bank = ProposalBank(
            [
                SphericalGaussian(T[-1, 0], T(0.75)),
                DiagonalGaussian(T[1, 0], T[1.25, 0.5]),
            ],
            T[1, 3],
        )
        vector_sampler = @inferred prepare_sampler(
            Random.Xoshiro(0x5421),
            StaticMISKernelVectorTarget{T}(),
            ImportanceSampling(vector_bank; nsamples=16);
            threaded=false,
        )
        vector_result = @inferred importance_sample!(vector_sampler)

        @test scalar_result.samples isa Vector{T}
        @test scalar_result.logweights isa Vector{T}
        @test vector_result.samples isa Matrix{T}
        @test vector_result.logweights isa Vector{T}
    end
end

@testset "packed native Gaussian failure and buffer accounting" begin
    bank = ProposalBank(
        [SphericalGaussian(0.0, 1.0), SphericalGaussian(0.0, 1.0)],
        [1.0, 1.0],
    )
    rng = StaticMISKernelPrefilledRNG(
        fill(0.25, 3),
        [0.0, 1.0, 2.0],
        Symbol[],
    )
    target = StaticMISKernelFailTarget(true, 1.0)
    sampler = prepare_sampler(
        rng,
        target,
        ImportanceSampling(bank; nsamples=3);
        threaded=true,
    )
    buffers = sampler.random_buffers
    uniform_buffer = buffers.uniform
    normal_buffer = buffers.normal
    assignment_buffer = buffers.assignments
    scratch = buffers.failure_scratch
    target_failures = scratch.target_failures
    failure = caught_static_mis_kernel_failure() do
        importance_sample!(sampler)
    end

    @test failure isa SamplerExecutionError
    @test (failure.phase, failure.sample_index) == (:target, 2)
    @test failure.captured.ex isa StaticMISKernelTargetFailure
    @test failure.captured.ex.sample == 1.0
    @test rng.calls == [:uniform, :normal]
    @test all(isnothing, target_failures.slots)
    @test target_failures.first_index[] == typemax(Int)

    target.fail = false
    result = @inferred importance_sample!(sampler)
    @test sampler.random_buffers === buffers
    @test sampler.random_buffers.uniform === uniform_buffer
    @test sampler.random_buffers.normal === normal_buffer
    @test sampler.random_buffers.assignments === assignment_buffer
    @test sampler.random_buffers.failure_scratch === scratch
    @test scratch.record.storage == zeros(UInt64, 3)
    @test result.diagnostics.transfers.count == 0
    @test result.diagnostics.transfers.bytes == 0
    @test rng.calls == [:uniform, :normal, :uniform, :normal]

    draw_first = prepare_sampler(
        StaticMISKernelPrefilledRNG(
            fill(0.25, 2),
            [Inf, 1.0],
            Symbol[],
        ),
        StaticMISKernelFailTarget(true, 1.0),
        ImportanceSampling(bank; nsamples=2);
        threaded=true,
    )
    draw_failure = caught_static_mis_kernel_failure() do
        importance_sample!(draw_first)
    end
    @test draw_failure isa SamplerExecutionError
    @test (draw_failure.phase, draw_failure.sample_index) == (:proposal_draw, 1)
    @test draw_failure.captured.ex isa DomainError

    draw_after_density = prepare_sampler(
        StaticMISKernelPrefilledRNG(
            fill(0.25, 3),
            [2sqrt(floatmax(Float64)), Inf, Inf],
            Symbol[],
        ),
        StaticMISKernelConstantTarget(0.0),
        ImportanceSampling(bank; nsamples=3);
        threaded=true,
    )
    phase_priority_failure = caught_static_mis_kernel_failure() do
        importance_sample!(draw_after_density)
    end
    @test phase_priority_failure isa SamplerExecutionError
    @test (phase_priority_failure.phase, phase_priority_failure.sample_index) ==
          (:proposal_draw, 2)
    @test phase_priority_failure.captured.ex isa DomainError
end

@testset "packed native Gaussian log-value truth table" begin
    for T in (Float32, Float64), threaded in (false, true)
        bank = ProposalBank(
            [SphericalGaussian(zero(T), one(T)), SphericalGaussian(zero(T), one(T))],
            T[1, 1],
        )
        algorithm = ImportanceSampling(bank; nsamples=2)

        zero_weight = importance_sample(
            Random.Xoshiro(0x5430),
            StaticMISKernelConstantTarget(T(-Inf)),
            algorithm;
            threaded,
        )
        @test zero_weight.logweights == fill(T(-Inf), 2)
        @test lognormalizer(zero_weight) == T(-Inf)

        for invalid_target in (T(NaN), T(Inf))
            failure = caught_static_mis_kernel_failure() do
                importance_sample(
                    Random.Xoshiro(0x5431),
                    StaticMISKernelConstantTarget(invalid_target),
                    algorithm;
                    threaded,
                )
            end
            @test failure isa SamplerExecutionError
            @test (failure.phase, failure.sample_index) == (:target, 1)
            @test failure.captured.ex isa DomainError
        end

        overflow_normal = T(0.75) * sqrt(floatmax(T))
        overflow = prepare_sampler(
            StaticMISKernelPrefilledRNG(
                T[0.25, 0.75],
                T[overflow_normal, 0],
                Symbol[],
            ),
            StaticMISKernelConstantTarget(floatmax(T)),
            algorithm;
            threaded,
        )
        failure = caught_static_mis_kernel_failure() do
            importance_sample!(overflow)
        end
        @test failure isa SamplerExecutionError
        @test (failure.phase, failure.sample_index) == (:logweight, 1)
        @test failure.captured.ex isa DomainError
    end

    function packed_table_denominator(logdensities, denominator; generating_slot=1)
        T = eltype(logdensities)
        bank = ISK._PackedDiagonalGaussianBank(
            zeros(T, 1, 2),
            ones(T, 1, 2),
            collect(logdensities),
            fill(-log(T(2)), 2),
            T[0.5, 1],
            [1, 2],
            ISK._ScalarGaussianLayout(),
        )
        return ISK._mis_logdenominator_core(
            T,
            bank,
            denominator,
            generating_slot,
            zero(T),
        )
    end

    full = ISK._FullMixtureDenominator()
    partial = ISK._PartialMixtureDenominator([1, 1], [1, 3], [1, 2], fill(-log(2.0), 2))
    for denominator in (full, partial)
        finite, finite_generating, finite_reason =
            packed_table_denominator([0.0, -Inf], denominator)
        @test finite == -log(2.0)
        @test finite_generating == 0.0
        @test iszero(finite_reason)

        positive_infinity, _, positive_infinity_reason =
            packed_table_denominator([0.0, Inf], denominator)
        @test positive_infinity == Inf
        @test iszero(positive_infinity_reason)

        generating_positive_infinity, generating_logdensity, generating_reason =
            packed_table_denominator([Inf, -Inf], denominator)
        @test generating_positive_infinity == Inf
        @test generating_logdensity == Inf
        @test iszero(generating_reason)

        for logdensities in ([-Inf, 0.0], [NaN, 0.0])
            _, _, reason = packed_table_denominator(logdensities, denominator)
            @test reason == ISK._NATIVE_PROPOSAL_INVALID
        end

        reduced_nan, _, reduced_nan_reason =
            packed_table_denominator([0.0, NaN], denominator)
        @test isnan(reduced_nan)
        @test reduced_nan_reason == ISK._NATIVE_PROPOSAL_INVALID

        reduced_minus_infinity, _, reduced_minus_infinity_reason =
            packed_table_denominator([-Inf, -Inf], denominator)
        @test reduced_minus_infinity == -Inf
        @test reduced_minus_infinity_reason == ISK._NATIVE_PROPOSAL_INVALID
    end

    generating = ISK._GeneratingDenominator()
    for (logdensity, expected_reason) in (
        (0.0, UInt16(0)),
        (Inf, UInt16(0)),
        (-Inf, ISK._NATIVE_PROPOSAL_INVALID),
        (NaN, ISK._NATIVE_PROPOSAL_INVALID),
    )
        denominator, generating_logdensity, reason =
            packed_table_denominator([logdensity, 0.0], generating)
        @test isequal(denominator, logdensity)
        @test isequal(generating_logdensity, logdensity)
        @test reason == expected_reason
    end
end
@testset "packed native Gaussian execution allocations" begin
    nsamples = 4_096
    for T in (Float32, Float64), dimension in (1, 4)
        proposals = if dimension == 1
            [
                SphericalGaussian(T(-1), T(0.75)),
                SphericalGaussian(T(1), T(1.25)),
            ]
        else
            [
                SphericalGaussian(fill(T(-1), dimension), T(0.75)),
                DiagonalGaussian(
                    fill(T(1), dimension),
                    fill(T(1.25), dimension),
                ),
            ]
        end
        target = dimension == 1 ?
                 StaticMISKernelScalarTarget{T}() :
                 StaticMISKernelVectorTarget{T}()
        sampler = prepare_sampler(
            Random.Xoshiro(0x5440),
            target,
            ImportanceSampling(
                ProposalBank(proposals, T[1, 3]);
                nsamples,
            );
            threaded=false,
        )
        importance_sample!(sampler)
        allocation = @allocated importance_sample!(sampler)
        required_result_storage = nsamples * (
            dimension * sizeof(T) + sizeof(T) + sizeof(Int)
        )

        @test allocation <= required_result_storage + 64_000
        @test all(
            value -> !(value isa AbstractMatrix),
            (
                sampler.random_buffers.uniform,
                sampler.random_buffers.normal,
                sampler.random_buffers.assignments,
                sampler.random_buffers.failure_scratch.record.storage,
                sampler.random_buffers.failure_scratch.target_failures.slots,
            ),
        )
    end
end
