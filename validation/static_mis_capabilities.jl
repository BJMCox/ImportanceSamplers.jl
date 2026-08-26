const STATIC_MIS_COMPLETE_SCHEMES = (
    (
        label=:stratified_mixture,
        name="`StratifiedMixture`",
        value=StratifiedMixture(),
        factory=_ -> StratifiedMixture(),
    ),
    (
        label=:random_mixture,
        name="`RandomMixture`",
        value=RandomMixture(),
        factory=_ -> RandomMixture(),
    ),
    (
        label=:standard_mis,
        name="`StandardMIS`",
        value=StandardMIS(),
        factory=_ -> StandardMIS(),
    ),
    (
        label=:partial_deterministic_mixture,
        name="`PartialDeterministicMixture`",
        value=PartialDeterministicMixture(((1, 3), (2, 4))),
        factory=proposal_count -> PartialDeterministicMixture(
            (Tuple(1:2:proposal_count), Tuple(2:2:proposal_count)),
        ),
    ),
)

const _STATIC_MIS_A100_ALL_SCHEMES = (
    hardware="NVIDIA A100-PCIE-40GB",
    types=(Float32, Float64),
    schemes=Tuple(scheme.label for scheme in STATIC_MIS_COMPLETE_SCHEMES),
)
const _STATIC_MIS_DIRECT_SPHERICAL_PARAMETERS =
    ((-0.45, 1.0), (-0.1, 1.1), (0.2, 0.9), (0.5, 1.2))

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
            [SphericalGaussian(T(x), T(s)) for (x, s) in _STATIC_MIS_DIRECT_SPHERICAL_PARAMETERS],
            T[1, 3, 0, 2],
        ),
        device=:supported,
        direct=merge(_STATIC_MIS_A100_ALL_SCHEMES, (sample_layout=:scalar,)),
    ),
    (
        bank="vector spherical Gaussians",
        cpu="packed CPU execution",
        factory=T -> ProposalBank(
            [SphericalGaussian(fill(T(x), 2), T(s)) for (x, s) in _STATIC_MIS_DIRECT_SPHERICAL_PARAMETERS],
            T[1, 3, 0, 2],
        ),
        device=:supported,
        direct=merge(_STATIC_MIS_A100_ALL_SCHEMES, (sample_layout=:vector,)),
    ),
    (
        bank="vector diagonal Gaussians",
        cpu="packed CPU execution",
        factory=T -> ProposalBank(
            [
                DiagonalGaussian(fill(T(x), 2), T[s, 1.05s]) for
                (x, s) in _STATIC_MIS_DIRECT_SPHERICAL_PARAMETERS
            ],
            T[1, 3, 0, 2],
        ),
        device=:supported,
        direct=merge(_STATIC_MIS_A100_ALL_SCHEMES, (sample_layout=:vector,)),
    ),
    (
        bank="mixed spherical/diagonal Gaussians with one layout, dimension, and float type",
        cpu="packed CPU execution",
        factory=T -> ProposalBank(Any[
            SphericalGaussian(fill(T(-0.45), 2), one(T)),
            DiagonalGaussian(fill(T(-0.1), 2), T[1.0, 1.2]),
            SphericalGaussian(fill(T(0.2), 2), T(0.9)),
            DiagonalGaussian(fill(T(0.5), 2), T[1.2, 1.0]),
        ], T[1, 3, 0, 2]),
        device=:supported,
        direct=merge(_STATIC_MIS_A100_ALL_SCHEMES, (sample_layout=:vector,)),
    ),
    (
        bank="factor Gaussian banks",
        cpu="packed CPU execution",
        factory=T -> ProposalBank(fill(FactorGaussian(zeros(T, 2), T[1 0; 0.2 1.1]), 4)),
        # Task 5 directly validated Float32/Float64 factor execution with
        # StratifiedMixture on an A100. Keep this row out of the older
        # all-scheme direct matrix until that matrix gains a factor oracle.
        device=:supported,
        direct=nothing,
        evidence=(
            hardware="NVIDIA A100-PCIE-40GB",
            types=(Float32, Float64),
            schemes=(:stratified_mixture,),
            reproducer=:cuda_dm_pmc,
        ),
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

static_mis_expected_sample_size(row, expected_mean, nsamples) =
    row.direct.sample_layout === :scalar ? (nsamples,) : (length(expected_mean), nsamples)

function static_mis_stratified_counts_within_bound(counts, masses, nsamples)
    expected = nsamples .* masses
    return all(abs.(counts .- expected) .< 2 .+ 16 .* eps.(float.(expected)))
end

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
