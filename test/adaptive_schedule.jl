using Test
using ImportanceSamplers

const IS = ImportanceSamplers

@testset "adaptive schedule" begin
    @test IS._validate_adaptive_schedule(3, 7) == 7
    @test IS._validate_adaptive_schedule(3, [2, 3, 5]) == [2, 3, 5]
    @test IS._resolve_adaptive_schedule(3, 7) == [7, 7, 7]
    @test IS._resolve_adaptive_schedule(3, [2, 3, 5]) == [2, 3, 5]
    @test_throws ArgumentError IS._adaptive_sample_budget(2, typemax(Int))
    @test_throws ArgumentError IS._adaptive_sample_budget(
        2,
        [typemax(Int), 1],
    )
    @test_throws ArgumentError IS._validate_adaptive_schedule(0, 1)
    @test_throws ArgumentError IS._validate_adaptive_schedule(2, 0)
    @test_throws DimensionMismatch IS._validate_adaptive_schedule(2, [1])
end

@testset "deterministic allocation plan characterization" begin
    proposal_ids = [7, 2, 9]
    bank = (; proposal_ids)

    @test_throws ArgumentError IS._deterministic_allocation_plan(
        (; proposal_ids=[1]),
        [1.0],
        [typemax(Int), 1],
    )
    terminal_offset_error = try
        IS._deterministic_allocation_plan(
            (; proposal_ids=[1]),
            [1.0],
            [typemax(Int)],
        )
        nothing
    catch error
        error
    end
    @test terminal_offset_error isa ArgumentError
    @test occursin(
        "adaptive allocation offsets exceed Int",
        sprint(showerror, terminal_offset_error),
    )

    for T in (Float32, Float64)
        equal_schedule = [3, 4, 5, 6]
        equal_plan = IS._deterministic_allocation_plan(
            bank,
            T[1, 1, 1],
            equal_schedule,
        )
        expected_equal_logcoefficients = if T === Float32
            Float32[
                -1.0986123 -1.3862944 -0.9162907 -1.0986123
                -1.0986123 -0.6931472 -1.6094380 -1.0986123
                -1.0986123 -1.3862944 -0.9162907 -1.0986123
            ]
        else
            Float64[
                -1.0986122886681098 -1.3862943611198906 -0.916290731874155 -1.0986122886681098
                -1.0986122886681098 -0.6931471805599453 -1.6094379124341003 -1.0986122886681098
                -1.0986122886681098 -1.3862943611198906 -0.916290731874155 -1.0986122886681098
            ]
        end

        @test equal_plan isa IS._DeterministicAllocationPlan
        @test equal_plan.schedule === equal_schedule
        @test equal_plan.counts == [
            1 1 2 2
            1 2 1 2
            1 1 2 2
        ]
        @test equal_plan.assignments == [
            1 1 1 1
            2 2 1 1
            3 2 2 2
            0 3 3 2
            0 0 3 3
            0 0 0 3
        ]
        @test equal_plan.logcoefficients == expected_equal_logcoefficients
        @test equal_plan.offsets == [1, 4, 8, 13, 19]
        @test [
            proposal_ids[view(equal_plan.assignments, 1:round_size, round)] for
            (round, round_size) in pairs(equal_schedule)
        ] == [[7, 2, 9], [7, 2, 2, 9], [7, 7, 2, 9, 9], [7, 7, 2, 2, 9, 9]]

        unequal_schedule = [6, 7]
        unequal_plan = IS._deterministic_allocation_plan(
            bank,
            T[1, 3, 2],
            unequal_schedule,
        )
        expected_unequal_logcoefficients = if T === Float32
            Float32[
                -1.7917595 -1.9459101
                -0.6931472 -0.55961573
                -1.0986123 -1.2527629
            ]
        else
            Float64[
                -1.791759469228055 -1.9459101490553135
                -0.6931471805599453 -0.5596157879354228
                -1.0986122886681098 -1.252762968495368
            ]
        end

        @test unequal_plan.counts == [
            1 1
            3 4
            2 2
        ]
        @test unequal_plan.assignments == [
            1 1
            2 2
            2 2
            2 2
            3 2
            3 3
            0 3
        ]
        @test unequal_plan.logcoefficients == expected_unequal_logcoefficients
        @test unequal_plan.offsets == [1, 7, 14]
        @test [
            proposal_ids[view(unequal_plan.assignments, 1:round_size, round)] for
            (round, round_size) in pairs(unequal_schedule)
        ] == [[7, 2, 2, 2, 9, 9], [7, 2, 2, 2, 2, 9, 9]]
    end
end
