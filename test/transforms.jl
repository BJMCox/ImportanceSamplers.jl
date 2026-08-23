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
