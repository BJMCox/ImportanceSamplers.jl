using Test

saved_args = copy(ARGS)
empty!(ARGS)
append!(ARGS, ("--smoke", "--cpu-only"))
benchmark_run = try
    include(joinpath(@__DIR__, "amis.jl"))
finally
    empty!(ARGS)
    append!(ARGS, saved_args)
end

@testset "AMIS benchmark evidence contract" begin
    @test benchmark_run.environment.hostname == gethostname()
    @test benchmark_run.environment.commit ==
          readchomp(`git -C $(benchmark_run.environment.checkout) rev-parse HEAD`)
    @test Tuple((row.scalar_type, row.geometry) for row in benchmark_run.rows) == (
        (Float32, :scalar),
        (Float32, :factor),
        (Float64, :scalar),
        (Float64, :factor),
    )
    @test all(
        record -> record.total_samples == sum(record.schedule) &&
                  record.target_evaluations == record.total_samples &&
                  record.proposal_evaluations ==
                  length(record.schedule) * record.total_samples &&
                  isfinite(record.ess) && record.ess > 0 &&
                  record.benchmarktools_evaluations == 1,
        (record for row in benchmark_run.rows for record in row.records),
    )
    @test all(
        guard -> ismissing(guard.device_allocated_bytes),
        benchmark_run.guards,
    )
end

function guard_fixture(commit; hostname="benchmark-host", host_allocations=7,
                       host_allocated_bytes=96, device_allocated_bytes=missing,
                       device=:cpu)
    records = ntuple(5) do replicate
        (;
            seed=0x10 + UInt(replicate),
            seconds=1.0,
            samples_per_second=100.0,
            host_allocations,
            host_allocated_bytes,
            benchmarktools_evaluations=1,
        )
    end
    guard = (;
        method=:static_mis,
        device,
        workload=64,
        prepared_samplers=5,
        allocation_samplers=device === :cuda ? 1 : 0,
        device_allocated_bytes,
        replicate_seeds=Tuple(record.seed for record in records),
        records,
        samples_per_second=(minimum=100.0, median=100.0, maximum=100.0),
        host_allocations=(minimum=7, median=7, maximum=7),
        host_allocated_bytes=(minimum=96, median=96, maximum=96),
    )
    environment = (;
        commit,
        checkout="/loaded/checkout",
        hostname,
        julia=v"1.12.7",
        cpu="test-cpu",
        cpu_threads=8,
        julia_threads=8,
        cuda=device === :cuda ?
             (gpu="test-gpu", capability=v"8.0", driver=v"13.0",
              runtime=v"12.9", allowscalar=false) : nothing,
        packages=(("BenchmarkTools", v"1.8.0"),
                  ("ImportanceSamplers", v"0.0.1")),
    )
    return (;
        schema_version=1,
        command="test fixture",
        mode=(smoke=true, long=false, scaling=false, thread_scaling=false,
              guards_only=true, factor_only=false),
        seed=0x10,
        benchmark_replicates=5,
        guard_sample_count=64,
        warmup=false,
        benchmarktools_evaluations_per_sampler=1,
        devices=(device,),
        environment,
        rows=(),
        scaling_rows=(),
        guards=(guard,),
        unavailable_cuda=false,
    )
end

compare_fixture(base, candidate) = compare_guard_runs(
    base,
    candidate;
    expected_base_commit=base.environment.commit,
    expected_candidate_commit=candidate.environment.commit,
)

@testset "AMIS exact-base guard comparison" begin
    base_commit = repeat("a", 40)
    candidate_commit = repeat("b", 40)
    base = guard_fixture(base_commit)
    candidate = guard_fixture(candidate_commit)
    comparison = compare_fixture(base, candidate)
    @test only(comparison).throughput_ratio == 1.0
    @test !only(comparison).new_host_allocation

    wrong_host = merge(
        candidate,
        (; environment=merge(candidate.environment, (; hostname="other-host"))),
    )
    @test_throws ErrorException compare_fixture(base, wrong_host)

    host_growth = guard_fixture(
        candidate_commit;
        host_allocations=8,
        host_allocated_bytes=112,
    )
    @test_throws ErrorException compare_fixture(base, host_growth)

    cuda_base = guard_fixture(base_commit; device=:cuda, device_allocated_bytes=32)
    cuda_growth = guard_fixture(
        candidate_commit;
        device=:cuda,
        device_allocated_bytes=64,
    )
    @test_throws ErrorException compare_fixture(cuda_base, cuda_growth)
end
