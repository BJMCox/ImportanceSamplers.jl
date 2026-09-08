import Random, LinearAlgebra

mutable struct LAISScriptedRNG{N,U} <: Random.AbstractRNG
    normals::N
    uniforms::U
    ni::Int
    ui::Int
end

function Random.randn!(rng::LAISScriptedRNG, out::AbstractArray)
    copyto!(out, rng.normals[rng.ni])
    rng.ni += 1
    return out
end

function Random.rand!(rng::LAISScriptedRNG, out::AbstractArray)
    copyto!(out, rng.uniforms[rng.ui])
    rng.ui += 1
    return out
end

function lais_ram_oracle(::Type{T}) where {T}
    factor = T[1 0; 1 1]
    centre = zeros(T, 2)
    normals = [T[1, 1], T[8, 3], T[-0.25, 0.5]]
    uniforms = T[0.1, 0.9, 0.01]
    centres = Matrix{T}(undef, 2, 3)
    for step in 1:3
        u = normals[step]
        candidate = centre + factor * u
        logalpha = min(zero(T), (sum(abs2, centre) - sum(abs2, candidate)) / T(20))
        direction = factor * (u / LinearAlgebra.norm(u))
        coefficient = T(step)^(-T(0.6)) * (exp(logalpha) - T(0.234))
        covariance = factor * factor' + coefficient * direction * direction'
        factor = Matrix(LinearAlgebra.cholesky(LinearAlgebra.Symmetric(covariance)).L)
        log(uniforms[step]) < logalpha && (centre = candidate)
        centres[:, step] = centre
    end
    return (; centres, normals, uniforms, factor)
end
