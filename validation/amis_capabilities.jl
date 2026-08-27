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

const AMIS_BENCHMARK_ROWS = (
    (:cpu, Float32, :scalar),
    (:cpu, Float64, :factor),
    (:cuda, Float32, :factor),
    (:cuda, Float64, :factor),
)

const AMIS_SCALING_ROUNDS = (2, 4, 8)
const AMIS_SCALING_DIMENSIONS = (1, 4, 16)
