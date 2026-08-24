using CUDA
using ImportanceSamplers
using MLDataDevices
using Test

const IS = ImportanceSamplers

@testset "public preserving device dispatch" begin
    physical = CUDA.device()
    default_policy = MLDataDevices.CUDADevice(physical)
    preserving_policy =
        MLDataDevices.CUDADevice{typeof(physical),Nothing}(physical)

    @test eltype(default_policy) === Missing
    @test eltype(preserving_policy) === Nothing
    @test IS._backend_functional(preserving_policy) === CUDA.functional()
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
