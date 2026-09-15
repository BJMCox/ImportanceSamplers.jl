module ReactantExecutionValidation
using ImportanceSamplers, Reactant, CUDA, Random, Test

logtarget(x, p) = -((x[1] - p[1])^2 + (x[2] - p[2])^2) / 2
function gradient!(g, x, p)
    g[1] = p[1] - x[1]
    g[2] = p[2] - x[2]
    return nothing
end

function run()
    CUDA.allowscalar(false)
    Reactant.set_default_backend("gpu")
    device = ImportanceSamplers.MLDataDevices.with_eltype(
        ImportanceSamplers.MLDataDevices.ReactantDevice(), nothing)
    proposal = FactorGaussian(zeros(Float32, 2), Float32[1.4 0; 0.3 1.2])
    bank = ProposalBank([proposal,
        FactorGaussian(Float32[0.5, -0.3], Float32[1.1 0; -0.2 1.4])])
    static = map((StratifiedMixture(), RandomMixture(), StandardMIS(),
        PartialDeterministicMixture(((1,), (2,))))) do scheme
        ImportanceSampling(bank; nsamples=32768, mis_scheme=scheme)
    end
    @testset "Reactant retained execution preserves live state and owned results" begin
        for algorithm in (ImportanceSampling(proposal; nsamples=32768),
            AMIS(proposal; rounds=3, round_size=16384),
            NPMC(proposal; rounds=3, round_size=16384), static...,
            APIS(bank; rounds=3, round_size=16384),
            DeterministicMixturePMC(bank; rounds=3, round_size=16384),
            DeterministicMixturePMC(bank; rounds=3, round_size=16384, resampling=LocalResampling()),
            CAIS(bank; rounds=3, round_size=16384),
            LAIS(bank; transition=RandomWalkMetropolis(0.1f0), rounds=3, round_size=16384),
            FirstOrderGRAMIS(bank; rounds=3, round_size=16384, repulsion_strength=0.1f0))
            p = Float32[0.25, -0.15]
            sampler = device(prepare_sampler(Xoshiro(41), LogTarget(logtarget; grad=gradient!), p, algorithm))
            first = importance_sample!(sampler)
            saved = Array(first.samples)
            saved_weights = Array(first.logweights)
            @test maximum(abs.(saved * Array(normalized_weights(first)) - p)) < 0.06
            @test abs(lognormalizer(first) - log(2f0 * Float32(pi))) < 0.04
            if algorithm isa ImportanceSampling && algorithm.proposal isa ProposalBank
                ids = Array(first.provenance.proposal_id)
                logdensity = ImportanceSamplers.DensityInterface.logdensityof
                full = algorithm.mis_scheme isa Union{StratifiedMixture,RandomMixture}
                expected = map(eachcol(saved), ids) do x, id
                    logq = full ? log(sum(exp(logdensity(q, x)) for q in bank.proposals) / 2) :
                        logdensity(bank.proposals[id], x)
                    logtarget(x, p) - logq
                end
                @test maximum(abs.(saved_weights .- expected)) < 2e-5
            end

            # Change values in the existing resident context, not its shape or type.
            moved = Float32[0.75, -0.5]
            copyto!(sampler.target.context, moved)
            second = importance_sample!(sampler)
            current = Array(second.samples)
            @test maximum(abs.(current * Array(normalized_weights(second)) - moved)) < 0.06
            @test Array(first.samples) == saved
            @test Array(first.logweights) == saved_weights
            @test current != saved

            # A failed run must not poison the next use of the prepared state.
            copyto!(sampler.target.context, fill(Float32(NaN), 2))
            @test_throws Union{SamplerExecutionError, AMISRoundError, NPMCRoundError,
                APISRoundError, DMPMCRoundError, CAISRoundError, LAISRoundError,
                FirstOrderGRAMISRoundError} importance_sample!(sampler)
            copyto!(sampler.target.context, moved)
            recovered = importance_sample!(sampler)
            @test abs(lognormalizer(recovered) - log(2f0 * Float32(pi))) < 0.04
        end
    end
    return nothing
end
end

ReactantExecutionValidation.run()
