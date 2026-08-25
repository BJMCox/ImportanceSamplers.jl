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

const _STATIC_MIS_A100_ALL_SCHEMES = (
    hardware="NVIDIA A100-PCIE-40GB",
    types=(Float32, Float64),
    schemes=Tuple(scheme.label for scheme in STATIC_MIS_COMPLETE_SCHEMES),
)

const STATIC_MIS_CAPABILITY_ROWS = (
    (
        bank="concrete homogeneous external proposals",
        cpu="generic CPU execution",
        factory=T -> ProposalBank(fill(CapabilityGaussian(), 4), T[1, 3, 2, 4]),
        device=:generic_proposal_cpu_only,
        direct=nothing,
    ),
    (
        bank="scalar spherical Gaussians",
        cpu="packed CPU execution",
        factory=T -> ProposalBank(
            [SphericalGaussian(T(x), T(s)) for (x, s) in ((-0.45, 1.0), (-0.1, 1.1), (0.2, 0.9), (0.5, 1.2))],
            T[1, 3, 0, 2],
        ),
        device=:supported,
        direct=merge(_STATIC_MIS_A100_ALL_SCHEMES, (sample_layout=:scalar,)),
    ),
    (
        bank="vector spherical Gaussians",
        cpu="packed CPU execution",
        factory=T -> ProposalBank(
            [SphericalGaussian(fill(T(x), 2), T(s)) for (x, s) in ((-1, 0.7), (0, 1.1), (1, 0.8), (2, 1.35))],
            T[1, 3, 0, 2],
        ),
        device=:supported,
        direct=merge(_STATIC_MIS_A100_ALL_SCHEMES, (sample_layout=:vector,)),
    ),
    (
        bank="vector diagonal Gaussians",
        cpu="packed CPU execution",
        factory=T -> ProposalBank([DiagonalGaussian(fill(T(x), 2), T[s, 2s]) for (x, s) in ((-1, 0.7), (0, 1.1), (1, 0.8), (2, 1.35))]),
        device=:supported,
        direct=nothing,
    ),
    (
        bank="mixed spherical/diagonal Gaussians with one layout, dimension, and float type",
        cpu="packed CPU execution",
        factory=T -> ProposalBank(Any[
            SphericalGaussian(zeros(T, 2), one(T)),
            DiagonalGaussian(ones(T, 2), T[0.75, 1.25]),
            SphericalGaussian(fill(T(2), 2), T(1.1)),
            DiagonalGaussian(fill(T(3), 2), T[1.25, 0.75]),
        ], T[1, 3, 2, 4]),
        device=:supported,
        direct=nothing,
    ),
    (
        bank="factor Gaussian banks",
        cpu="generic CPU execution",
        factory=T -> ProposalBank(fill(FactorGaussian(zeros(T, 2), T[1 0; 0.2 1.1]), 4)),
        device=:factor_proposal_cpu_only,
        direct=nothing,
    ),
    (
        bank="transformed proposal banks",
        cpu="generic CPU execution",
        factory=T -> ProposalBank(fill(TransformedProposal(SphericalGaussian(zero(T), one(T)), PositiveTransform()), 4)),
        device=:transformed_proposal_cpu_only,
        direct=nothing,
    ),
    (
        bank="product proposal banks",
        cpu="generic CPU execution",
        factory=T -> ProposalBank(fill(ProductProposal((left=CapabilityGaussian(), right=CapabilityGaussian())), 4)),
        device=:product_proposal_cpu_only,
        direct=nothing,
    ),
)

const STATIC_MIS_PREPARATION_REJECTIONS = (
    (input="mixed positive-mass vector dimensions", error=DimensionMismatch,
        factory=() -> ProposalBank(Any[SphericalGaussian(zeros(2), 1.0), DiagonalGaussian(zeros(3), ones(3))]),
    ),
    (input="mixed positive-mass floating types", error=ArgumentError,
        factory=() -> ProposalBank(Any[SphericalGaussian(zeros(Float32, 2), 1.0f0), DiagonalGaussian(zeros(Float64, 2), ones(Float64, 2))]),
    ),
    (input="mixed scalar and vector sample layouts", error=ArgumentError,
        factory=() -> ProposalBank(Any[SphericalGaussian(0.0, 1.0), SphericalGaussian([0.0], 1.0)]),
    ),
)
