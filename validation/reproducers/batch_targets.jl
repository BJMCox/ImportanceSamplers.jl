using ImportanceSamplers, Random, Test
import MLDataDevices

function batch_reference_logtarget(x, p)
    total = zero(eltype(x))
    for i in eachindex(x)
        total += abs2(x[i] - p.centre[i])
    end
    return -total / 2
end

batch_named_logtarget(x, p) = sum(log, x.mixture; init=zero(x.scale)) - x.scale / p.scale - abs2(x.offset) / 2
function batch_named_values!(values, samples, p)
    values .= vec(sum(log, samples.mixture; dims=1)) .- samples.scale ./ p.scale .-
              abs2.(samples.offset) ./ 2
    return nothing
end

function verify_named_batch_target(device; T=Float32)
    transform = (mixture=1:2 => SimplexTransform(3),
        scale=3 => PositiveTransform(), offset=4 => IdentityTransform())
    algorithm = ImportanceSampling(SphericalGaussian(zeros(T, 4), T(0.5)); nsamples=128)
    p = (; scale=T(2))
    scalar = LogTarget(batch_named_logtarget)
    batch = LogTarget(batch_named_logtarget; batch=batch_named_values!)
    a = device(prepare_sampler(Xoshiro(18), scalar, p, algorithm; transform))
    b = device(prepare_sampler(Xoshiro(18), batch, p, algorithm; transform))
    expected, actual = importance_sample!(a), importance_sample!(b)
    @testset "named batch layout on $(typeof(device))" begin
        @test all(Array(x) ≈ Array(y) for (x, y) in zip(actual.samples, expected.samples))
        @test Array(actual.logweights) ≈ Array(expected.logweights) rtol=8sqrt(eps(T))
    end
    return nothing
end

function batch_reference_values!(values, samples, p)
    values .= .-vec(sum(abs2, samples .- p.centre; dims=1)) ./ 2
    return nothing
end

function batch_reference_gradient!(g, x, p)
    for i in eachindex(x)
        g[i] = p.centre[i] - x[i]
    end
    return nothing
end

"""Compare scalar and explicit batch execution on the same selected device."""
function verify_batch_targets(device; T=Float32, select=nothing)
    proposal = FactorGaussian(T[0.2, -0.3], T[1.3 0; 0.2 1.1])
    student = FactorStudentT(T(7), T[0.2, -0.3], T[1.3 0; 0.2 1.1])
    bank = ProposalBank([proposal, FactorGaussian(T[-0.2, 0.3], T[1.1 0; -0.2 1.3])], T[1, 1])
    algorithms = (
        plain=ImportanceSampling(proposal; nsamples=128),
        student=ImportanceSampling(student; nsamples=128),
        mis=ImportanceSampling(bank; nsamples=128),
        apis=APIS(bank; rounds=2, round_size=128),
        dm_pmc=DeterministicMixturePMC(bank; rounds=2, round_size=128),
        cais=CAIS(bank; rounds=2, round_size=128),
        amis=AMIS(proposal; rounds=3, round_size=[128, 256, 192]),
        npmc=NPMC(proposal; rounds=3, round_size=128),
        lais=LAIS(bank; transition=RandomWalkMetropolis(T(0.3)), rounds=2, round_size=128),
        ram=LAIS(bank; transition=RAM(T(0.3); tuning=WarmupTuning(3)), rounds=2, round_size=128),
        smh=LAIS(bank; transition=SampleMetropolisHastings(proposal; moves=3), rounds=2, round_size=128),
        gramis=FirstOrderGRAMIS(bank; rounds=2, round_size=128, repulsion_strength=T(0)),
    )
    scalar = LogTarget(batch_reference_logtarget; grad=batch_reference_gradient!)
    batch = LogTarget(batch_reference_logtarget;
        grad=batch_reference_gradient!, batch=batch_reference_values!)
    p = (; centre=T[0.1, -0.2])
    @testset "explicit batch targets on $(typeof(device))" begin
        for (name, algorithm) in pairs(algorithms)
            isnothing(select) || name in select || continue
            @testset "$name" begin
                a = device(prepare_sampler(Xoshiro(55), scalar, p, algorithm))
                b = device(prepare_sampler(Xoshiro(55), batch, p, algorithm))
                expected, actual = importance_sample!(a), importance_sample!(b)
                tolerance = 8sqrt(eps(T))
                @test Array(actual.samples) ≈ Array(expected.samples) rtol=tolerance atol=tolerance
                @test Array(actual.logweights) ≈ Array(expected.logweights) rtol=tolerance atol=tolerance
            end
        end
    end
    return nothing
end
