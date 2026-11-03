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

@testset "Reactant CPU transition preflight preserves frozen execution" begin
    device = MLDataDevices.with_eltype(ReactantDevice(), nothing)
    bank = ProposalBank([FactorGaussian(zeros(Float32, 2), Float32[1 0; 0.2 1])])
    scalar(x) = -(abs2(x[1]) + abs2(x[2])) / 2f0
    batch!(out, xs) = (out .= .-vec(sum(abs2, xs; dims=1)) ./ 2f0; nothing)
    for (target, transition) in (
        (scalar, RAM(0.25f0; tuning=ContinuousTuning())),
        (scalar, RAM(0.25f0; tuning=WarmupTuning(2))),
        (LogTarget(scalar; batch=batch!), RAM(0.25f0; tuning=WarmupTuning(2))),
        (scalar, SampleMetropolisHastings(SphericalStudentT(6f0, zeros(Float32, 2), 1f0))),
    )
        algorithm = LAIS(bank; transition, rounds=1, round_size=8)
        source = prepare_sampler(Xoshiro(72), target, algorithm)
        @test_throws SamplerDeviceError device(source)
        @test importance_sample!(source).samples ==
            importance_sample(Xoshiro(72), target, algorithm).samples
    end
    algorithm = LAIS(bank; transition=RAM(0.25f0; tuning=WarmupTuning(0)),
        rounds=1, round_size=8)
    sampler = device(prepare_sampler(Xoshiro(73), scalar, algorithm))
    result = importance_sample!(sampler)
    host = MLDataDevices.CPUDevice()(result)
    proposal = only(current_proposal(MLDataDevices.CPUDevice(), sampler).proposals)
    expected = [scalar(x) - logdensityof(proposal, x) for x in eachcol(host.samples)]
    @test host.logweights ≈ expected rtol=32eps(Float32)
    @test result.diagnostics.transition.warmup_proposals == 0
end
