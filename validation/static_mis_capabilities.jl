const STATIC_MIS_COMPLETE_SCHEMES = (
    (label=:stratified_mixture, name="`StratifiedMixture`", value=StratifiedMixture()),
    (label=:random_mixture, name="`RandomMixture`", value=RandomMixture()),
    (label=:standard_mis, name="`StandardMIS`", value=StandardMIS()),
    (
        label=:partial_deterministic_mixture,
        name="`PartialDeterministicMixture`",
        value=PartialDeterministicMixture(((1, 3), (2, 4))),
    ),
)

const STATIC_MIS_CAPABILITY_ROWS = (
    (
        label=:generic,
        bank="concrete homogeneous external proposals",
        cpu="generic CPU execution",
        accelerator="rejected: `:generic_proposal_cpu_only`",
        reason=:generic_proposal_cpu_only,
        a100=false,
    ),
    (
        label=:scalar_spherical,
        bank="scalar spherical Gaussians",
        cpu="packed CPU execution",
        accelerator="A100 execution with `Float32` and `Float64`",
        reason=nothing,
        a100=true,
    ),
    (
        label=:vector_spherical,
        bank="vector spherical Gaussians",
        cpu="packed CPU execution",
        accelerator="A100 execution with `Float32` and `Float64`",
        reason=nothing,
        a100=true,
    ),
    (
        label=:vector_diagonal,
        bank="vector diagonal Gaussians",
        cpu="packed CPU execution",
        accelerator="CUDA execution; not directly A100-validated",
        reason=nothing,
        a100=false,
    ),
    (
        label=:mixed_native,
        bank="mixed spherical/diagonal Gaussians with one layout, dimension, and float type",
        cpu="packed CPU execution",
        accelerator="CUDA execution; not directly A100-validated",
        reason=nothing,
        a100=false,
    ),
    (
        label=:factor,
        bank="factor Gaussian banks",
        cpu="generic CPU execution",
        accelerator="rejected: `:factor_proposal_cpu_only`",
        reason=:factor_proposal_cpu_only,
        a100=false,
    ),
    (
        label=:transformed,
        bank="transformed proposal banks",
        cpu="generic CPU execution",
        accelerator="rejected: `:transformed_proposal_cpu_only`",
        reason=:transformed_proposal_cpu_only,
        a100=false,
    ),
    (
        label=:product,
        bank="product proposal banks",
        cpu="generic CPU execution",
        accelerator="rejected: `:product_proposal_cpu_only`",
        reason=:product_proposal_cpu_only,
        a100=false,
    ),
)

const STATIC_MIS_PREPARATION_REJECTIONS = (
    (
        label=:mixed_dimension,
        input="mixed positive-mass vector dimensions",
        error=DimensionMismatch,
    ),
    (
        label=:mixed_float,
        input="mixed positive-mass floating types",
        error=ArgumentError,
    ),
    (
        label=:mixed_layout,
        input="mixed scalar and vector sample layouts",
        error=ArgumentError,
    ),
)

const STATIC_MIS_A100_TYPES = (Float32, Float64)
const STATIC_MIS_A100_LAYOUTS = (
    :scalar_spherical,
    :vector_spherical,
)
