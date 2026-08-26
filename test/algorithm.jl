@testset "ImportanceSampling constructor" begin
    proposal = TestScalarProposal(0.0)
    algorithm = @inferred ImportanceSampling(proposal; nsamples=8)

    @test algorithm.proposal === proposal
    @test algorithm.nsamples === 8
    @test algorithm.mis_scheme isa ImportanceSamplers._SingleProposalScheme
    @test fieldnames(typeof(algorithm)) == (:proposal, :nsamples, :mis_scheme)
    @test algorithm isa AbstractImportanceSampler
    @test_throws UndefKeywordError ImportanceSampling(proposal)
    @test_throws ArgumentError ImportanceSampling(proposal; nsamples=true)
    @test_throws ArgumentError ImportanceSampling(proposal; nsamples=0)
    @test_throws ArgumentError ImportanceSampling(proposal; nsamples=-1)
end
