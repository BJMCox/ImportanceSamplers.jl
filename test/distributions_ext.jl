import Distributions
import DensityInterface
import LinearAlgebra: Symmetric, cholesky
import Random

@testset "Distributions normals configure adaptive sampling" begin
    for T in (Float32, Float64)
        proposal = Distributions.Normal(T(0.5), T(1.25))
        algorithm = AMIS(proposal; rounds=1, round_size=32)
        target(sample) = Distributions.logpdf(proposal, sample)

        result = importance_sample(Random.Xoshiro(0x4d91), target, algorithm)

        @test length(result) == 32
        @test all(isapprox(zero(T); atol=8eps(T)), result.logweights)
    end

    for T in (Float32, Float64)
        location = T[0.5, -0.25]
        cases = (
            (
                proposal=Distributions.MvNormal(location, T(1.25)),
                native=SphericalGaussian(location, sqrt(T(1.25))),
            ),
            (
                proposal=Distributions.MvNormal(location, T[0.75, 1.5]),
                native=DiagonalGaussian(location, sqrt.(T[0.75, 1.5])),
            ),
            (
                proposal=Distributions.MvNormal(location, T[1.0 0.3; 0.3 2.0]),
                native=FactorGaussian(
                    location,
                    cholesky(Symmetric(T[1.0 0.3; 0.3 2.0])),
                ),
            ),
        )
        for case in cases
            proposal = case.proposal
            algorithm = AMIS(proposal; rounds=1, round_size=64)
            target = sample -> Distributions.logpdf(proposal, sample)

            result = importance_sample(Random.Xoshiro(0x74a3), target, algorithm)

            @test typeof(algorithm.proposal) === typeof(case.native)
            @test length(result) == 64
            @test all(isapprox(zero(T); atol=32eps(T)), result.logweights)
        end
    end
end

@testset "Distributions normal banks configure adaptive sampling" begin
    proposals = [
        Distributions.MvNormal([-1.0, 0.0], 1.5),
        Distributions.MvNormal([1.0, 0.0], 1.5),
    ]
    bank = ProposalBank(proposals, [1.0, 2.0])
    algorithm = DeterministicMixturePMC(bank; rounds=1, round_size=64)
    target(sample) = -sum(abs2, sample) / 2

    result = importance_sample(Random.Xoshiro(0x1c87), target, algorithm)

    @test length(result) == 64
    @test all(isfinite, result.logweights)
    @test Set(result.provenance.proposal_id) == Set((1, 2))
end

@testset "Distributions Student-t proposals use native storage" begin
    scalar_cases = (
        (
            source=Distributions.TDist(5.0f0),
            native=SphericalStudentT(5.0f0, 0.0f0, 1.0f0),
        ),
        (
            source=Distributions.Cauchy(1.5, 2.0),
            native=SphericalStudentT(1.0, 1.5, 2.0),
        ),
    )
    for (index, case) in enumerate(scalar_cases)
        algorithm = ImportanceSampling(case.source; nsamples=64)
        result = importance_sample(
            Random.Xoshiro(index),
            sample -> Distributions.logpdf(case.source, sample),
            algorithm;
            threaded=false,
        )
        @test typeof(algorithm.proposal) === typeof(case.native)
        @test maximum(abs, result.logweights) <= 64eps(eltype(result.logweights))
    end

    location = [0.5, -0.25]
    multivariate_cases = (
        (
            source=Distributions.IsoTDist(7.0, location, 1.25),
            native=SphericalStudentT(7.0, location, 1.25),
        ),
        (
            source=Distributions.DiagTDist(7.0, location, [0.75, 1.5]),
            native=DiagonalStudentT(7.0, location, [0.75, 1.5]),
        ),
        (
            source=Distributions.MvTDist(7.0, location, [1.0 0.3; 0.3 2.0]),
            native=FactorStudentT(
                7.0,
                location,
                cholesky(Symmetric([1.0 0.3; 0.3 2.0])),
            ),
        ),
    )
    for (index, case) in enumerate(multivariate_cases)
        algorithm = ImportanceSampling(case.source; nsamples=64)
        result = importance_sample(
            Random.Xoshiro(100 + index),
            sample -> Distributions.logpdf(case.source, sample),
            algorithm;
            threaded=false,
        )
        @test typeof(algorithm.proposal) === typeof(case.native)
        @test maximum(abs, result.logweights) <= 256eps()
    end
end

@testset "native Student-t density matches Distributions" begin
    for T in (Float32, Float64), dof in T.((0.5, 1, 8, 100, 1000)), dimension in (1, 8)
        location = zeros(T, dimension)
        sample = fill(T(0.75), dimension)
        source = Distributions.IsoTDist(Float64(dof), zeros(Float64, dimension), 1.0)
        native = SphericalStudentT(dof, location, one(T))
        tolerance = T === Float32 ? T(2e-5) : T(2e-12)
        @test DensityInterface.logdensityof(native, sample) ≈
              Distributions.logpdf(source, Float64.(sample)) rtol=tolerance atol=tolerance
    end
end
