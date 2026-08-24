using CUDA
using ImportanceSamplers
using MLDataDevices
using Test

const IS = ImportanceSamplers

@testset "public preserving device dispatch" begin
    default_policy, preserving_policy = withenv(
        "MLDATADEVICES_SILENCE_WARN_NO_GPU" => "1",
    ) do
        (
            MLDataDevices.gpu_device(nothing; force=false),
            MLDataDevices.gpu_device(nothing, nothing; force=false),
        )
    end

    @test eltype(default_policy) === Missing
    @test eltype(preserving_policy) === Nothing
end

@testset "public CUDA RNG extension" begin
    extension_module = Base.get_extension(IS, :ImportanceSamplersCUDAExt)
    @test extension_module !== nothing

    device = MLDataDevices.CUDADevice()
    seed = UInt64(0x63756461726e67)
    first_rng = @inferred IS._owned_backend_rng(device, seed)
    second_rng = @inferred IS._owned_backend_rng(device, seed)

    @test first_rng isa CUDA.RNG
    @test second_rng isa CUDA.RNG
    @test first_rng !== second_rng
    if CUDA.functional()
        default_rng = CUDA.default_rng()
        @test first_rng !== default_rng
        @test second_rng !== default_rng
    else
        @test_skip first_rng !== CUDA.default_rng()
        @test_skip second_rng !== CUDA.default_rng()
    end
end
