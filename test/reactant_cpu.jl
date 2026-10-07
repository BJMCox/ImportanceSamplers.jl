using ImportanceSamplers, DensityInterface, MLDataDevices, Reactant, Random, Test
using CUDA # Reactant's KernelAbstractions compiler needs this even on CPU.

Reactant.set_default_backend("cpu")

@testset "Reactant CPU factor-Student-t sampling" begin
    device = MLDataDevices.with_eltype(ReactantDevice(), nothing)
    proposal = FactorStudentT(5f0, zeros(Float32, 2), Float32[1 0; 0.4 0.8])
    logtarget(x) = -sum(abs2, x) / 2f0
    # An uneven count exercises a partially filled final workgroup.
    sampler = device(prepare_sampler(Xoshiro(42), logtarget,
        ImportanceSampling(proposal; nsamples=129)))
    result = importance_sample!(sampler)
    points, weights = Array(result.samples), Array(result.logweights)
    expected = [logtarget(x) - logdensityof(proposal, x) for x in eachcol(points)]
    @test size(points, 2) == length(weights) == 129
    @test weights ≈ expected rtol=32eps(Float32)
end
