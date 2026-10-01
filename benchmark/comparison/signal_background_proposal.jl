module SignalBackgroundProposals

using Random
import ImportanceSamplers as IS
using ImportanceSamplers.KernelAbstractions: @kernel, @index

"""
    SignalBackgroundProposal(globals::ProposalBank, coefficients; prior_mass)

Frozen proposal for the signal-background benchmark's nine logit/noncentred
coordinates. `globals` contains four-dimensional factor Student-t proposals.
The three coefficient rows define each detector's conditional location and
scale. All Student-t laws use the same degrees of freedom.

This benchmark-local adapter uses private native IS hooks. It is not a public
custom-proposal API. Fit on CPU, then transfer the prepared sampler to device.
"""
struct SignalBackgroundProposal{B,C,T}
    globals::B
    coefficients::C
    prior_mass::T
    dof::T
    lognormalizer::T
end

function SignalBackgroundProposal(globals::IS.ProposalBank, coefficients; prior_mass)
    packed = IS._pack_native_radial_bank(globals)
    packed isa IS._PackedFactorBank || throw(ArgumentError("globals require factor proposals"))
    T = eltype(packed.locations)
    size(packed.locations, 1) == 4 && size(coefficients) == (3, 5) ||
        throw(DimensionMismatch("expected four global coordinates and five detector fits"))
    eltype(coefficients) === T && prior_mass isa T ||
        throw(ArgumentError("proposal parameters must share a floating type"))
    family = first(globals.proposals).family
    family isa IS.StudentTFamily || throw(ArgumentError("globals require Student-t proposals"))
    all(q -> q.family.dof == family.dof, globals.proposals) ||
        throw(ArgumentError("all Student-t laws must share degrees of freedom"))
    isfinite(prior_mass) && zero(T) <= prior_mass <= one(T) ||
        throw(ArgumentError("prior mass must lie in [0, 1]"))
    all(isfinite, coefficients) && all(>=(zero(T)), view(coefficients, 2, :)) &&
        all(>(zero(T)), view(coefficients, 3, :)) ||
        throw(ArgumentError("conditional fits require finite coefficients and positive scales"))
    residual = IS.SphericalStudentT(family.dof, zero(T), one(T))
    return SignalBackgroundProposal(packed, copy(coefficients), prior_mass,
        family.dof, residual.lognormalizer)
end

IS.Adapt.@adapt_structure SignalBackgroundProposal

IS._supports_native_fused_cpu(::SignalBackgroundProposal) = true
IS._accelerator_proposal_limit(::SignalBackgroundProposal) = nothing
IS._native_fused_components(q::SignalBackgroundProposal) = (q, IS._NoSampleTransform())
IS._native_fused_float_type(q::SignalBackgroundProposal) = eltype(q.coefficients)
IS._proposal_dimension(::SignalBackgroundProposal) = 9
IS._native_normal_stride(q::SignalBackgroundProposal) =
    9 + 6 * IS._radial_normal_stride(IS.StudentTFamily(q.dof, q.lognormalizer))
IS._native_uniform_stride(q::SignalBackgroundProposal) =
    1 + max(8, IS._radial_uniform_stride(IS.StudentTFamily(q.dof, q.lognormalizer))) +
    5 * IS._radial_uniform_stride(IS.StudentTFamily(q.dof, q.lognormalizer))
IS._allocate_native_samples(prototype, q::SignalBackgroundProposal, count) =
    similar(prototype, eltype(q.coefficients), 9, count)

@inline function detector_parameters(q, x, j)
    T = eltype(q.coefficients)
    sigma = T(0.1) + T(0.9) * IS.LogExpFunctions.logistic(x[2])
    m = T(1e-10) + (T(20) - T(1e-10)) * IS.LogExpFunctions.logistic(x[3])
    a, b, c = q.coefficients[1, j], q.coefficients[2, j], q.coefficients[3, j]
    precision = one(T) + b * sigma^2
    return sigma * (a - b * (log(m) - sigma^2 / T(2))) / precision, c / sqrt(precision)
end

@inline function draw!(x, q, normals, uniforms, normal_offset, uniform_offset)
    T = eltype(x)
    family = IS.StudentTFamily(q.dof, q.lognormalizer)
    radial_normals = IS._radial_normal_stride(family)
    radial_uniforms = IS._radial_uniform_stride(family)
    choice = uniforms[uniform_offset + 1]
    if isone(q.prior_mass) || choice < q.prior_mass
        for j in 1:4
            u = uniforms[uniform_offset + 2j]
            zero(T) < u < one(T) || (u = uniforms[uniform_offset + 2j + 1])
            zero(T) < u < one(T) || return IS._NATIVE_PROPOSAL_DRAW_EXHAUSTED
            x[j] = log(u) - log1p(-u)
        end
    else
        u = (choice - q.prior_mass) / (one(T) - q.prior_mass)
        component = 1
        while component < length(q.globals.cdf) && q.globals.cdf[component] < u
            component += 1
        end
        multiplier, reason = IS._student_t_radial_multiplier(q.dof, normals,
            uniforms, normal_offset + 9, uniform_offset + 1)
        iszero(reason) || return reason
        scaled = IS._ScaledNormals(normals, multiplier)
        for j in 1:4
            x[j] = IS._native_gaussian_coordinate(q.globals, scaled,
                normal_offset, j, component)
        end
    end
    for j in 1:5
        multiplier, reason = IS._student_t_radial_multiplier(q.dof, normals,
            uniforms, normal_offset + 9 + j * radial_normals,
            uniform_offset + 1 + max(8, radial_uniforms) + (j - 1) * radial_uniforms)
        iszero(reason) || return reason
        location, scale = detector_parameters(q, x, j)
        x[4+j] = location + scale * multiplier * normals[normal_offset + 3 + j]
    end
    return UInt16(0)
end

@inline function proposal_logdensity(q, x, scratch, slot)
    T = eltype(x)
    logprior = -sum(IS.LogExpFunctions.log1pexp(x[j]) +
        IS.LogExpFunctions.log1pexp(-x[j]) for j in 1:4)
    density = log(q.prior_mass) + logprior
    for component in eachindex(q.globals.logmasses)
        logq = IS._packed_gaussian_logdensity!(q.globals, x, component, scratch, slot)
        density = IS.LogExpFunctions.logaddexp(density,
            log1p(-q.prior_mass) + q.globals.logmasses[component] + logq)
    end
    for j in 1:5
        location, scale = detector_parameters(q, x, j)
        radius = abs2((x[4+j] - location) / scale)
        density += q.lognormalizer - log(scale) -
            (q.dof + one(T)) / T(2) * log1p(radius / q.dof)
    end
    return density
end

function Random.rand(rng::Random.AbstractRNG, q::SignalBackgroundProposal)
    T = eltype(q.coefficients)
    x = Vector{T}(undef, 9)
    reason = draw!(x, q, randn(rng, T, IS._native_normal_stride(q)),
        rand(rng, T, IS._native_uniform_stride(q)), 1, 0)
    iszero(reason) || error("conditional proposal draw exhausted")
    return x
end

IS.DensityInterface.logdensityof(q::SignalBackgroundProposal, x::AbstractVector) =
    proposal_logdensity(q, x, Matrix{eltype(x)}(undef, 4, 1), 1)

@kernel function conditional_draws!(samples, logweights, failures, q, normals, uniforms)
    slot = @index(Global, Linear)
    sample = view(samples, :, slot)
    reason = draw!(sample, q, normals, uniforms,
        (slot - 1) * IS._native_normal_stride(q) + 1,
        (slot - 1) * IS._native_uniform_stride(q))
    if iszero(reason) && !IS._factor_batch_sample_finite(samples, slot)
        reason = IS._NATIVE_GENERATED_NONFINITE
    end
    if iszero(reason)
        logweights[slot] = proposal_logdensity(q, sample, normals, slot)
        reason = IS._native_proposal_reason(logweights[slot])
    end
    if !iszero(reason)
        IS._record_native_failure!(failures, slot, 0, reason)
        logweights[slot] = eltype(logweights)(NaN)
        for j in axes(samples, 1)
            samples[j, slot] = zero(eltype(samples))
        end
    end
end

# Every NaN here has a recorded failure. Skip its target so a second error
# cannot hide the draw failure in the shared finish kernel.
struct DrawTarget{T,W}
    target::T
    logweights::W
end
IS.Adapt.@adapt_structure DrawTarget
@inline (target::DrawTarget)(sample, slot) = isnan(target.logweights[slot]) ?
    (zero(eltype(target.logweights)), UInt16(0), true) : target.target(sample, slot)

function IS._launch_native_fused!(samples, logweights, failures, uniforms, normals,
    target, q::SignalBackgroundProposal, transform::IS._NoSampleTransform, execution)
    if target isa IS._NativeBatchTarget
        IS._launch_native_fused!(samples, logweights, failures, uniforms, normals,
            IS._deferred_target(target), q, transform, execution)
        return IS._finish_batch_weights!(logweights, target.target, samples,
            failures.storage, execution; transform)
    end
    backend = IS.KernelAbstractions.get_backend(normals)
    launch = (; ndrange=length(logweights),
        workgroupsize=IS._native_workgroupsize(execution, length(logweights)))
    conditional_draws!(backend)(samples, logweights, failures.storage, q,
        reshape(normals, IS._native_normal_stride(q), length(logweights)), uniforms; launch...)
    IS._native_factor_batch_finish_kernel!(backend)(samples, logweights,
        failures.storage, DrawTarget(target, logweights); launch...)
    IS.KernelAbstractions.synchronize(backend)
    return nothing
end

end
