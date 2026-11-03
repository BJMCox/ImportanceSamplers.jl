include("../validation/support_table.jl")

@testset "documented sampler execution contracts" begin
    for row in SamplerSupportTable.support_cases()
        @test SamplerSupportTable.check_support_case(row) === :passed
    end
end
