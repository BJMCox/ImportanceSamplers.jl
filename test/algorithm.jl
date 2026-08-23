@testset "ImportanceSampling constructor" begin
    proposal = TestScalarProposal(0.0)
    algorithm = ImportanceSampling(proposal; nsamples=8)

    @test algorithm.proposal === proposal
    @test algorithm.nsamples === 8
    @test fieldnames(typeof(algorithm)) == (:proposal, :nsamples)
    @test algorithm isa AbstractImportanceSampler
    @test_throws UndefKeywordError ImportanceSampling(proposal)
    @test_throws ArgumentError ImportanceSampling(proposal; nsamples=true)
    @test_throws ArgumentError ImportanceSampling(proposal; nsamples=0)
    @test_throws ArgumentError ImportanceSampling(proposal; nsamples=-1)
end
