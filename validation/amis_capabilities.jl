const AMIS_CAPABILITY_ROWS = (
    (
        label=:float32_scalar,
        type=Float32,
        geometry=:scalar,
        proposal=T -> SphericalGaussian(T(-1.25), T(1.8)),
        cpu=:supported,
        cuda=(hardware="NVIDIA A100-PCIE-40GB", status=:supported),
    ),
    (
        label=:float64_scalar,
        type=Float64,
        geometry=:scalar,
        proposal=T -> SphericalGaussian(T(-1.25), T(1.8)),
        cpu=:supported,
        cuda=(hardware="NVIDIA A100-PCIE-40GB", status=:supported),
    ),
    (
        label=:float32_factor,
        type=Float32,
        geometry=:factor,
        proposal=T -> FactorGaussian(
            T[-1.0, 0.75, 1.5],
            T[1.4 0 0; -0.2 1.2 0; 0.1 0.25 1.35],
        ),
        cpu=:supported,
        cuda=(hardware="NVIDIA A100-PCIE-40GB", status=:supported),
    ),
    (
        label=:float64_factor,
        type=Float64,
        geometry=:factor,
        proposal=T -> FactorGaussian(
            T[-1.0, 0.75, 1.5],
            T[1.4 0 0; -0.2 1.2 0; 0.1 0.25 1.35],
        ),
        cpu=:supported,
        cuda=(hardware="NVIDIA A100-PCIE-40GB", status=:supported),
    ),
)

const AMIS_SCHEDULE_CAPABILITY_ROWS = (
    (label=:equal, schedule=rounds -> fill(257, rounds)),
    (label=:unequal, schedule=rounds -> collect(257:113:(257 + 113 * (rounds - 1)))),
)

const AMIS_BENCHMARK_TYPE_GEOMETRIES = (
    (Float32, :scalar),
    (Float32, :factor),
    (Float64, :scalar),
    (Float64, :factor),
)

const AMIS_BENCHMARK_ROWS = Tuple(
    (device, T, geometry) for device in (:cpu, :cuda) for
    (T, geometry) in AMIS_BENCHMARK_TYPE_GEOMETRIES
)

# Each non-baseline cell changes one scaling axis while all schedules remain unequal.
const AMIS_PERFORMANCE_SCALING_CELLS = (
    (label=:baseline, dimension=4, schedule=(512, 1_024, 2_048, 4_608)),
    (
        label=:rounds,
        dimension=4,
        schedule=(128, 256, 512, 768, 1_024, 1_280, 1_664, 2_560),
    ),
    (label=:dimension, dimension=16, schedule=(512, 1_024, 2_048, 4_608)),
    (label=:total_samples, dimension=4, schedule=(1_024, 2_048, 4_096, 9_216)),
)

const AMIS_SCALING_ROUNDS = (2, 4, 8)
const AMIS_SCALING_DIMENSIONS = (1, 4, 16)

struct AMISCapabilityTarget{T<:AbstractFloat} end

function (::AMISCapabilityTarget{T})(sample)::T where {T}
    radius = sample isa Number ? abs2(sample) : sum(abs2, sample)
    return -T(0.5) * radius
end

function checked_amis_capability_table()
    rows = map(enumerate(AMIS_CAPABILITY_ROWS)) do (row_index, row)
        row.cpu === :supported || error(
            "AMIS capability row $(row.label) is not advertised for CPU",
        )
        row.cuda.status === :supported || error(
            "AMIS capability row $(row.label) is not advertised for CUDA",
        )
        for (schedule_index, schedule_row) in enumerate(AMIS_SCHEDULE_CAPABILITY_ROWS)
            schedule = schedule_row.schedule(2)
            sampler = prepare_sampler(
                Random.Xoshiro(10row_index + schedule_index),
                AMISCapabilityTarget{row.type}(),
                AMIS(
                    row.proposal(row.type);
                    rounds=length(schedule),
                    round_size=schedule,
                );
                threaded=false,
            )
            result = importance_sample!(sampler)
            length(result) == sum(schedule) || error(
                "AMIS capability row $(row.label) returned the wrong count",
            )
            result.diagnostics.round_sizes == schedule || error(
                "AMIS capability row $(row.label) returned the wrong schedule",
            )
            result.provenance.round == vcat(
                (fill(round, count) for (round, count) in enumerate(schedule))...,
            ) || error(
                "AMIS capability row $(row.label) returned the wrong provenance",
            )
            current_proposal(sampler)
        end
        geometry = row.geometry === :scalar ?
                   "scalar spherical Gaussian" : "factor Gaussian"
        return "| `$(row.type)` | $geometry | public execution during docs build | " *
               "$(row.cuda.hardware) reproducer |"
    end
    return "| Scalar type | Geometry | CPU evidence | CUDA evidence |\n" *
           "|:--|:--|:--|:--|\n" * join(rows, '\n')
end
