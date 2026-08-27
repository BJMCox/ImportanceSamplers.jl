using Test
using ImportanceSamplers
import LogExpFunctions

const IS = ImportanceSamplers

@testset "log-mixture accumulator" begin
    for T in (Float32, Float64)
        a = log(T(2)) + log(T(0.25))
        b = log(T(3)) + log(T(0.5))
        numerator = IS._append_logmixture(T(-Inf), log(T(2)), log(T(0.25)))
        numerator = IS._append_logmixture(numerator, log(T(3)), log(T(0.5)))
        @test numerator ≈ LogExpFunctions.logaddexp(a, b)
        @test IS._logweight_from_logmixture(log(T(4)), numerator, log(T(5)))[1] ≈
              log(T(4)) - numerator + log(T(5))
    end
end
