const IS = ImportanceSamplers

function _transform_forward_derivative(transform, z::T) where {T}
    step = cbrt(eps(T)) * max(one(T), abs(z))
    upper = IS._transform_with_logjac(transform, z + step)[1]
    lower = IS._transform_with_logjac(transform, z - step)[1]
    return (upper - lower) / (T(2) * step)
end

function _transform_error_reason(f)
    error = try
        f()
        nothing
    catch caught
        caught
    end
    return error isa InvalidTransformError ? error.reason : nothing
end

function _scalar_transform_cases(::Type{T}) where {T}
    return (
        (IdentityTransform(), (-20, -2, 0, 2, 20)),
        (PositiveTransform(), (-20, -2, 0, 2, 20)),
        (SoftplusTransform(), (-20, -2, 0, 2, 20)),
        (IntervalTransform(T(-3), nothing), (-2, 0, 2, 5)),
        (IntervalTransform(nothing, T(5)), (-2, 0, 2, 5)),
        (IntervalTransform(T(-3), T(5)), (-5, -2, 0, 2, 5)),
    )
end

const _transform_api_ready = all(
    name -> isdefined(ImportanceSamplers, name),
    (
        :AbstractSampleTransform,
        :IdentityTransform,
        :PositiveTransform,
        :SoftplusTransform,
        :IntervalTransform,
        :InvalidTransformError,
        :_transform_with_logjac,
        :_inverse_with_logjac,
    ),
)

@testset "scalar transform public API" begin
    @test _transform_api_ready
end

if _transform_api_ready

@testset "scalar transform round trips and Jacobians" begin
    for T in (Float32, Float64), (transform, inputs) in _scalar_transform_cases(T), input in inputs
        z = T(input)
        x, logabsjac = @inferred IS._transform_with_logjac(transform, z)
        recovered_z, recovered_logabsjac = @inferred IS._inverse_with_logjac(transform, x)
        @test x isa T
        @test logabsjac isa T
        @test recovered_z isa T
        @test recovered_logabsjac isa T
        @test recovered_z ≈ z rtol = 64eps(T) atol = 64eps(T)
        @test recovered_logabsjac ≈ logabsjac rtol = 64eps(T) atol = 64eps(T)
    end
end

@testset "scalar transform finite-difference Jacobians" begin
    for T in (Float32, Float64), (transform, input) in (
        (IdentityTransform(), -0.5),
        (PositiveTransform(), 0.4),
        (SoftplusTransform(), -0.4),
        (IntervalTransform(T(-3), nothing), 0.4),
        (IntervalTransform(nothing, T(5)), -0.4),
        (IntervalTransform(T(-3), T(5)), 0.4),
    )
        z = T(input)
        _, logabsjac = IS._transform_with_logjac(transform, z)
        derivative = _transform_forward_derivative(transform, z)
        @test log(abs(derivative)) ≈ logabsjac rtol = sqrt(eps(T)) atol = sqrt(eps(T))
    end
end

@testset "one-sided scalar interval shifts" begin
    for T in (Float32, Float64)
        lower = T(-3)
        upper = T(5)
        z = T(1.25)
        lower_value, lower_logabsjac = IS._transform_with_logjac(
            IntervalTransform(lower, nothing),
            z,
        )
        upper_value, upper_logabsjac = IS._transform_with_logjac(
            IntervalTransform(nothing, upper),
            z,
        )
        @test lower_value == lower + exp(z)
        @test upper_value == upper - exp(z)
        @test lower_logabsjac == z
        @test upper_logabsjac == z
    end
end

@testset "scalar transform extreme finite inputs" begin
    for T in (Float32, Float64)
        positive_limit = prevfloat(log(floatmax(T)))
        positive_value, positive_logabsjac = IS._transform_with_logjac(
            PositiveTransform(),
            positive_limit,
        )
        @test isfinite(positive_value)
        @test positive_logabsjac == positive_limit

        softplus_value, softplus_logabsjac = IS._transform_with_logjac(
            SoftplusTransform(),
            floatmax(T),
        )
        @test softplus_value == floatmax(T)
        @test softplus_logabsjac == zero(T)

        interval_value, interval_logabsjac = IS._transform_with_logjac(
            IntervalTransform(T(-3), T(5)),
            zero(T),
        )
        @test interval_value == T(1)
        @test interval_logabsjac ≈ log(T(2))
    end
end

@testset "wide bounded intervals and strict scalar boundaries" begin
    for T in (Float32, Float64)
        lower = -floatmax(T)
        upper = floatmax(T)
        transform = IntervalTransform(lower, upper)
        midpoint, midpoint_logabsjac = IS._transform_with_logjac(transform, zero(T))
        midpoint_z, midpoint_inverse_logabsjac = IS._inverse_with_logjac(transform, midpoint)
        @test midpoint == zero(T)
        @test isfinite(midpoint_logabsjac)
        @test midpoint_z == zero(T)
        @test midpoint_inverse_logabsjac ≈ midpoint_logabsjac rtol = 64eps(T)
        for z in (T(-5), T(5))
            value, logabsjac = IS._transform_with_logjac(transform, z)
            recovered_z, recovered_logabsjac = IS._inverse_with_logjac(transform, value)
            @test lower < value < upper
            @test isfinite(logabsjac)
            @test recovered_z ≈ z rtol = 128eps(T) atol = 128eps(T)
            @test recovered_logabsjac ≈ logabsjac rtol = 128eps(T)
        end
        for near_boundary in (nextfloat(lower), prevfloat(upper))
            z, logabsjac = IS._inverse_with_logjac(transform, near_boundary)
            @test isfinite(z)
            @test isfinite(logabsjac)
        end

        underflow_z = log(nextfloat(zero(T))) - one(T)
        for transform in (PositiveTransform(), SoftplusTransform())
            @test _transform_error_reason(
                () -> IS._transform_with_logjac(transform, underflow_z),
            ) === :outside_support
        end
        for z in (-floatmax(T), floatmax(T))
            @test _transform_error_reason(
                () -> IS._transform_with_logjac(IntervalTransform(T(-3), T(5)), z),
            ) === :outside_support
        end

        rounding_z = log(eps(T) / T(4))
        @test _transform_error_reason(
            () -> IS._transform_with_logjac(IntervalTransform(T(-3), nothing), rounding_z),
        ) === :outside_support
        @test _transform_error_reason(
            () -> IS._transform_with_logjac(IntervalTransform(nothing, T(5)), rounding_z),
        ) === :outside_support

        for z in (zero(T), -zero(T))
            identity_value, identity_logabsjac = IS._transform_with_logjac(IdentityTransform(), z)
            _, positive_logabsjac = IS._transform_with_logjac(PositiveTransform(), z)
            @test signbit(identity_value) == signbit(z)
            @test identity_logabsjac == zero(T)
            @test signbit(positive_logabsjac) == signbit(z)
        end
        for invalid in (T(NaN), T(Inf), T(-Inf))
            @test _transform_error_reason(
                () -> IS._transform_with_logjac(IdentityTransform(), invalid),
            ) === :nonfinite_input
        end
    end
end

@testset "scalar transform validation and invalid arithmetic" begin
    @test_throws ArgumentError IntervalTransform(nothing, nothing)
    @test_throws ArgumentError IntervalTransform(0.0, 0.0)
    @test_throws ArgumentError IntervalTransform(1.0, 0.0)
    @test_throws ArgumentError IntervalTransform(NaN, 1.0)
    @test_throws ArgumentError IntervalTransform(0.0, NaN)
    @test_throws ArgumentError IntervalTransform(Inf, nothing)
    @test_throws ArgumentError IntervalTransform(nothing, -Inf)
    @test_throws ArgumentError IntervalTransform(0, 1)
    @test_throws ArgumentError IntervalTransform(0.0f0, 1.0)
    @test_throws ArgumentError IntervalTransform("zero", nothing)

    overflow = try
        IS._transform_with_logjac(PositiveTransform(), floatmax(Float64))
        nothing
    catch error
        error
    end
    @test overflow isa InvalidTransformError
    @test overflow.reason === :nonfinite_output
    @test isnothing(overflow.location)
    @test !any(field_type -> field_type <: AbstractArray, fieldtypes(typeof(overflow)))
    @test_throws ArgumentError InvalidTransformError(:outside_support, [1.0, 2.0])

    invalid_input = try
        IS._inverse_with_logjac(PositiveTransform(), NaN)
        nothing
    catch error
        error
    end
    @test invalid_input.reason === :nonfinite_input
end
end

const _simplex_transform_api_ready = isdefined(ImportanceSamplers, :SimplexTransform)

@testset "simplex transform public API" begin
    @test _simplex_transform_api_ready
    @test :SimplexTransform in names(ImportanceSamplers)
end

if _simplex_transform_api_ready

@testset "simplex transform dimension validation" begin
    for dimension in (-3, 0, 1, 2.0, Int32(3), true)
        @test_throws ArgumentError SimplexTransform(dimension)
    end

    transform = SimplexTransform(3)
    @test !any(field_type -> field_type <: AbstractArray, fieldtypes(typeof(transform)))
    @test_throws DimensionMismatch IS._transform_with_logjac(transform, zeros(3))
    @test_throws DimensionMismatch IS._inverse_with_logjac(transform, fill(1 / 2, 2))
end

@testset "simplex transform range and round trips" begin
    for T in (Float32, Float64), K in (2, 3, 10)
        transform = SimplexTransform(K)
        z = T[T(sin(index)) / T(3) for index in 1:(K - 1)]
        x, logabsjac = @inferred IS._transform_with_logjac(transform, z)
        recovered_z, recovered_logabsjac = @inferred IS._inverse_with_logjac(transform, x)

        @test x isa Vector{T}
        @test length(x) == K
        @test all(isfinite, x)
        @test all(>(zero(T)), x)
        @test sum(x) ≈ one(T) rtol = T(8) * eps(T) atol = T(8) * eps(T)
        @test logabsjac isa T
        @test recovered_z isa Vector{T}
        @test length(recovered_z) == K - 1
        @test recovered_z ≈ z rtol = T(256) * eps(T) atol = T(256) * eps(T)
        @test recovered_logabsjac ≈ logabsjac rtol = T(64) * eps(T) atol = T(64) * eps(T)
    end
end

@testset "simplex zero is exactly uniform where representable" begin
    for T in (Float32, Float64), K in (2, 4)
        x, _ = IS._transform_with_logjac(SimplexTransform(K), zeros(T, K - 1))
        @test x == fill(inv(T(K)), K)
        @test sum(x) == one(T)
    end
end

@testset "simplex stable finite arithmetic" begin
    for T in (Float32, Float64)
        transform = SimplexTransform(4)
        z = T[20, -20, 15]
        x, logabsjac = IS._transform_with_logjac(transform, z)
        recovered_z, recovered_logabsjac = IS._inverse_with_logjac(transform, x)
        @test all(isfinite, x)
        @test all(>(zero(T)), x)
        @test isfinite(logabsjac)
        @test recovered_z ≈ z rtol = T(512) * eps(T) atol = T(512) * eps(T)
        @test recovered_logabsjac ≈ logabsjac rtol = T(512) * eps(T)
    end
end

@testset "simplex Float32 large-dimension reductions" begin
    @test _transform_error_reason(
        () -> IS._inverse_with_logjac(
            SimplexTransform(3),
            Float32[0.2, 0.3, 0.5 + 32eps(Float32)],
        ),
    ) === :outside_support

    for dimension in (1_000, 10_000)
        transform = SimplexTransform(dimension)
        _, logabsjac = IS._transform_with_logjac(
            transform,
            zeros(Float32, dimension - 1),
        )
        expected =
            0.5 * log(Float64(dimension)) -
            Float64(dimension) * log(Float64(dimension))
        @test abs(Float64(logabsjac) - expected) <= Float64(eps(Float32(expected)))
    end

    dimension = 10_000
    coordinates = Float32[
        0.25f0 * Float32(sin(index)) for index in 1:(dimension - 1)
    ]
    simplex, _ = IS._transform_with_logjac(SimplexTransform(dimension), coordinates)
    @test abs(sum(Float64, simplex) - 1) <= 2eps(Float32)

    dimension = 100_000
    transform = SimplexTransform(dimension)
    simplex, forward_logabsjac = IS._transform_with_logjac(
        transform,
        zeros(Float32, dimension - 1),
    )
    recovered, inverse_logabsjac = @inferred IS._inverse_with_logjac(transform, simplex)
    @test maximum(abs, recovered) <= 16eps(Float32) * log(Float32(dimension))
    @test abs(Float64(inverse_logabsjac) - Float64(forward_logabsjac)) <=
          Float64(eps(forward_logabsjac))
end

@testset "simplex orthonormal and permutation-neutral geometry" begin
    K = 5
    scale = 0.25
    embedded_columns = Matrix{Float64}(undef, K, K - 1)
    for column in 1:(K - 1)
        z = zeros(K - 1)
        z[column] = scale
        x, _ = IS._transform_with_logjac(SimplexTransform(K), z)
        logx = log.(x)
        embedded_columns[:, column] = (logx .- sum(logx) / K) ./ scale
    end

    for left in 1:(K - 1), right in 1:(K - 1)
        expected = left == right ? 1.0 : 0.0
        @test sum(embedded_columns[:, left] .* embedded_columns[:, right]) ≈ expected atol = 2e-14
    end
    for left in 1:K, right in 1:K
        expected = (left == right ? 1.0 : 0.0) - 1 / K
        @test sum(embedded_columns[left, :] .* embedded_columns[right, :]) ≈ expected atol = 2e-14
    end

    weights = [0.05, 0.15, 0.25, 0.20, 0.35]
    reference_coordinates, _ = IS._inverse_with_logjac(SimplexTransform(K), weights)
    reference_norm = sum(abs2, reference_coordinates)
    for permutation in ([2, 1, 3, 5, 4], [5, 4, 3, 2, 1], [3, 5, 1, 4, 2])
        coordinates, _ = IS._inverse_with_logjac(SimplexTransform(K), weights[permutation])
        @test sum(abs2, coordinates) ≈ reference_norm atol = 2e-14
    end
end

@testset "simplex full coordinate Jacobian" begin
    for T in (Float32, Float64), K in (2, 3, 7)
        x, logabsjac = IS._transform_with_logjac(SimplexTransform(K), zeros(T, K - 1))
        expected = T(0.5) * log(T(K)) + sum(log, x)
        @test logabsjac ≈ expected rtol = T(8) * eps(T) atol = T(8) * eps(T)
        @test logabsjac - sum(log, x) ≈ T(0.5) * log(T(K)) atol = T(8) * eps(T)
    end

    transform = SimplexTransform(3)
    z = [0.2, -0.4]
    x, logabsjac = IS._transform_with_logjac(transform, z)
    step = cbrt(eps(Float64))
    columns = ntuple(2) do column
        offset = zeros(2)
        offset[column] = step
        upper = IS._transform_with_logjac(transform, z + offset)[1]
        lower = IS._transform_with_logjac(transform, z - offset)[1]
        (upper - lower) / (2step)
    end
    determinant = columns[1][1] * columns[2][2] - columns[1][2] * columns[2][1]
    @test log(abs(determinant)) ≈ logabsjac rtol = 2e-9 atol = 2e-9
end

@testset "simplex inverse support validation" begin
    transform = SimplexTransform(3)
    @test _transform_error_reason(
        () -> IS._transform_with_logjac(transform, [0.0, Inf]),
    ) === :nonfinite_input
    @test _transform_error_reason(
        () -> IS._inverse_with_logjac(transform, [0.0, 0.5, 0.5]),
    ) === :outside_support
    @test _transform_error_reason(
        () -> IS._inverse_with_logjac(transform, [0.2, 0.3, 0.6]),
    ) === :outside_support
    @test _transform_error_reason(
        () -> IS._inverse_with_logjac(transform, [0.2, NaN, 0.8]),
    ) === :nonfinite_input
end

end
