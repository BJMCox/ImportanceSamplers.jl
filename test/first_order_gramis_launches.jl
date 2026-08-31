using Test
using ImportanceSamplers
import LinearAlgebra
import Random

const GRAMISLaunchIS = ImportanceSamplers
const GRAMISLaunchKA = GRAMISLaunchIS.KernelAbstractions

const GRAMIS_LAUNCH_FUNCTIONS = (
    local_weights=typeof(GRAMISLaunchIS.cpu__first_order_gramis_local_weights_kernel!),
    group_starts=typeof(GRAMISLaunchIS.cpu__local_group_starts_kernel!),
    local_summary=typeof(GRAMISLaunchIS.cpu__local_weight_summary_kernel!),
    tempering=typeof(GRAMISLaunchIS.cpu__tempering_power_kernel!),
    covariance_fit=typeof(GRAMISLaunchIS.cpu__fit_local_covariances_kernel!),
    covariance_blend=typeof(GRAMISLaunchIS.cpu__blend_local_covariances_kernel!),
)
const GRAMIS_LAUNCH_RECORDS = NamedTuple[]
const GRAMIS_RECORD_LAUNCHES = Ref(false)

for function_type in values(GRAMIS_LAUNCH_FUNCTIONS)
    @eval function (kernel::GRAMISLaunchKA.Kernel{
            GRAMISLaunchKA.CPU,
            GRAMISLaunchKA.NDIteration.DynamicSize,
            GRAMISLaunchKA.NDIteration.DynamicSize,
            $function_type,
        })(
        args...;
        ndrange=nothing,
        workgroupsize=nothing,
    )
        GRAMIS_RECORD_LAUNCHES[] && push!(
            GRAMIS_LAUNCH_RECORDS,
            (; function_type=$function_type, ndrange, workgroupsize),
        )
        ndrange, workgroupsize, iterspace, dynamic =
            GRAMISLaunchKA.launch_config(
            kernel,
            ndrange,
            workgroupsize,
        )
        isempty(GRAMISLaunchKA.blocks(iterspace)) && return nothing
        GRAMISLaunchKA.__run(
            kernel,
            ndrange,
            iterspace,
            args,
            dynamic,
            kernel.backend.static,
        )
        return nothing
    end
end

struct GRAMISLaunchTarget{T} end

function (::GRAMISLaunchTarget{T})(sample)::T where {T}
    return -sum(abs2, sample) / T(2)
end

function gram_is_launch_gradient!(destination, sample)
    destination .= .-sample
    return destination
end

@testset "FirstOrderGRAMIS public execution applies every local launch policy" begin
    T = Float64
    dimension = 2
    proposal_count = 16
    round_size = 160
    proposals = map(1:proposal_count) do proposal_slot
        phase = T(2pi * (proposal_slot - 1) / proposal_count)
        location = T[sin(phase), cos(phase)] / T(10)
        FactorGaussian(
            location,
            Matrix{T}(LinearAlgebra.I, dimension, dimension),
        )
    end
    sampler = prepare_sampler(
        Random.Xoshiro(0x6c61756e63686573),
        LogTarget(GRAMISLaunchTarget{T}(); grad=gram_is_launch_gradient!),
        FirstOrderGRAMIS(
            ProposalBank(proposals);
            rounds=1,
            round_size,
            repulsion_strength=zero(T),
            covariance_ess_threshold=3,
        );
        threaded=true,
    )

    empty!(GRAMIS_LAUNCH_RECORDS)
    GRAMIS_RECORD_LAUNCHES[] = true
    result = try
        importance_sample!(sampler)
    finally
        GRAMIS_RECORD_LAUNCHES[] = false
    end

    @test length(result) == round_size
    pool_threads = Threads.nthreads(:default)
    expected_ranges = (
        local_weights=round_size,
        group_starts=1,
        local_summary=proposal_count,
        tempering=proposal_count,
        covariance_fit=dimension * dimension * proposal_count,
        covariance_blend=proposal_count,
    )
    for (name, function_type) in pairs(GRAMIS_LAUNCH_FUNCTIONS)
        matching = filter(
            record -> record.function_type === function_type,
            GRAMIS_LAUNCH_RECORDS,
        )
        @test length(matching) == 1
        length(matching) == 1 || continue
        record = only(matching)
        ndrange = expected_ranges[name]
        expected_workgroupsize = min(
            1_024,
            max(1, fld(ndrange, pool_threads)),
        )
        @test record.ndrange == ndrange
        @test record.workgroupsize == expected_workgroupsize
        record.workgroupsize == expected_workgroupsize || continue
        @test cld(ndrange, record.workgroupsize) >=
              min(pool_threads, ndrange)
    end
end
