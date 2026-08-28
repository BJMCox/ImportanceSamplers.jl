using Test
using ImportanceSamplers

const IS = ImportanceSamplers

@testset "adaptive schedule" begin
    @test IS._validate_adaptive_schedule(3, 7) == 7
    @test IS._validate_adaptive_schedule(3, [2, 3, 5]) == [2, 3, 5]
    @test IS._resolve_adaptive_schedule(3, 7) == [7, 7, 7]
    @test IS._resolve_adaptive_schedule(3, [2, 3, 5]) == [2, 3, 5]
    @test_throws ArgumentError IS._validate_adaptive_schedule(0, 1)
    @test_throws ArgumentError IS._validate_adaptive_schedule(2, 0)
    @test_throws DimensionMismatch IS._validate_adaptive_schedule(2, [1])
end
