using Test
using ImportanceSamplers
import LinearAlgebra
import Random

const GRAMISLaunchIS = ImportanceSamplers
const GRAMISLaunchKA = GRAMISLaunchIS.KernelAbstractions

const GRAMIS_COOPERATIVE_LOCAL_WEIGHTS_FUNCTION =
    typeof(GRAMISLaunchIS.cpu__cooperative_local_weights_kernel!)
const GRAMIS_GROUP_STARTS_FUNCTION =
    isdefined(GRAMISLaunchIS, :cpu__local_group_starts_kernel!) ?
    typeof(GRAMISLaunchIS.cpu__local_group_starts_kernel!) : nothing
const GRAMIS_RECORDED_LAUNCH_FUNCTIONS = (
    local_weights=typeof(GRAMISLaunchIS.cpu__first_order_gramis_local_weights_kernel!),
    group_starts=GRAMIS_GROUP_STARTS_FUNCTION,
    local_summary=typeof(GRAMISLaunchIS.cpu__local_weight_summary_kernel!),
    tempering=typeof(GRAMISLaunchIS.cpu__tempering_power_kernel!),
    cooperative_local_weights=GRAMIS_COOPERATIVE_LOCAL_WEIGHTS_FUNCTION,
    covariance_fit=typeof(GRAMISLaunchIS.cpu__fit_local_covariances_kernel!),
    covariance_blend=typeof(GRAMISLaunchIS.cpu__blend_local_covariances_kernel!),
)
const GRAMIS_LAUNCH_RECORDS = NamedTuple[]
const GRAMIS_RECORD_LAUNCHES = Ref(false)

for (name, function_type) in pairs(GRAMIS_RECORDED_LAUNCH_FUNCTIONS)
    isnothing(function_type) && continue
    workgroupsize_type = name === :cooperative_local_weights ?
                         GRAMISLaunchKA.NDIteration.StaticSize{(256,)} :
                         GRAMISLaunchKA.NDIteration.DynamicSize
    @eval begin
        function (kernel::GRAMISLaunchKA.Kernel{
                GRAMISLaunchKA.CPU,
                $workgroupsize_type,
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
    cpu_expected_ranges = (
        local_weights=round_size,
        local_summary=proposal_count,
        tempering=proposal_count,
        covariance_fit=dimension * dimension * proposal_count,
        covariance_blend=proposal_count,
    )
    if !isnothing(GRAMIS_GROUP_STARTS_FUNCTION)
        @test count(
            record -> record.function_type === GRAMIS_GROUP_STARTS_FUNCTION,
            GRAMIS_LAUNCH_RECORDS,
        ) == 0
    end
    @test count(
        record -> record.function_type ===
                  GRAMIS_COOPERATIVE_LOCAL_WEIGHTS_FUNCTION,
        GRAMIS_LAUNCH_RECORDS,
    ) == 0
    for (name, ndrange) in pairs(cpu_expected_ranges)
        function_type = getproperty(GRAMIS_RECORDED_LAUNCH_FUNCTIONS, name)
        matching = filter(
            record -> record.function_type === function_type,
            GRAMIS_LAUNCH_RECORDS,
        )
        @test length(matching) == 1
        length(matching) == 1 || continue
        record = only(matching)
        expected_workgroupsize = name === :cooperative_local_weights ? 256 : min(
            1_024,
            max(1, fld(ndrange, pool_threads)),
        )
        @test record.ndrange == ndrange
        @test record.workgroupsize == expected_workgroupsize
        record.workgroupsize == expected_workgroupsize || continue
        @test cld(ndrange, record.workgroupsize) >=
              min(pool_threads, ndrange)
    end

    empty!(GRAMIS_LAUNCH_RECORDS)
    GRAMIS_RECORD_LAUNCHES[] = true
    try
        GRAMISLaunchIS._prepare_local_covariance_weights!(
            sampler.method_state,
            1,
            GRAMISLaunchIS._KernelExecution(
                GRAMISLaunchIS._ThreadedCPUExecution(),
            ),
        )
    finally
        GRAMIS_RECORD_LAUNCHES[] = false
    end

    for name in (:local_summary, :tempering)
        function_type = getproperty(GRAMIS_RECORDED_LAUNCH_FUNCTIONS, name)
        @test count(
            record -> record.function_type === function_type,
            GRAMIS_LAUNCH_RECORDS,
        ) == 0
    end
    cooperative_launches = filter(
        record -> record.function_type ===
                  GRAMIS_COOPERATIVE_LOCAL_WEIGHTS_FUNCTION,
        GRAMIS_LAUNCH_RECORDS,
    )
    @test length(cooperative_launches) == 1
    if length(cooperative_launches) == 1
        launch = only(cooperative_launches)
        @test launch.ndrange == 256 * proposal_count
        @test launch.workgroupsize == 256
    end
end
