import KernelAbstractions

@testset "portable kernel smoke test" begin
    output = zeros(Int, 4)
    backend = KernelAbstractions.get_backend(output)

    @test backend isa KernelAbstractions.CPU

    ImportanceSamplers._kernel_smoke!(output)
    KernelAbstractions.synchronize(backend)

    @test output == collect(1:4)
end
