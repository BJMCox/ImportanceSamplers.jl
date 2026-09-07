module ImportanceSamplersDistributionsExt

import Distributions
import ImportanceSamplers
import LinearAlgebra
import Statistics

function ImportanceSamplers._prepare_proposal_input(proposal::Distributions.Normal)
    location, scale = Distributions.params(proposal)
    return ImportanceSamplers.SphericalGaussian(location, scale)
end

function ImportanceSamplers._prepare_proposal_input(proposal::Distributions.IsoNormal)
    location = collect(Statistics.mean(proposal))
    scale = first(Statistics.std(proposal))
    return ImportanceSamplers.SphericalGaussian(location, scale)
end

function ImportanceSamplers._prepare_proposal_input(proposal::Distributions.DiagNormal)
    location = collect(Statistics.mean(proposal))
    scales = collect(Statistics.std(proposal))
    return ImportanceSamplers.DiagonalGaussian(location, scales)
end

function ImportanceSamplers._prepare_proposal_input(proposal::Distributions.FullNormal)
    location = collect(Statistics.mean(proposal))
    covariance = Matrix(Statistics.cov(proposal))
    factor = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(covariance))
    return ImportanceSamplers.FactorGaussian(location, factor)
end

function ImportanceSamplers._prepare_proposal_input(proposal::Distributions.TDist)
    dof = only(Distributions.params(proposal))
    T = typeof(dof)
    return ImportanceSamplers.SphericalStudentT(dof, zero(T), one(T))
end

function ImportanceSamplers._prepare_proposal_input(proposal::Distributions.Cauchy)
    location, scale = Distributions.params(proposal)
    return ImportanceSamplers.SphericalStudentT(one(location), location, scale)
end

function ImportanceSamplers._prepare_proposal_input(proposal::Distributions.IsoTDist)
    dof, location, _ = Distributions.params(proposal)
    scale_matrix = Distributions.scale(proposal)
    scale = sqrt(first(LinearAlgebra.diag(scale_matrix)))
    return ImportanceSamplers.SphericalStudentT(dof, collect(location), scale)
end

function ImportanceSamplers._prepare_proposal_input(proposal::Distributions.DiagTDist)
    dof, location, _ = Distributions.params(proposal)
    scales = sqrt.(LinearAlgebra.diag(Distributions.scale(proposal)))
    return ImportanceSamplers.DiagonalStudentT(dof, collect(location), scales)
end

function ImportanceSamplers._prepare_proposal_input(proposal::Distributions.GenericMvTDist)
    dof, location, _ = Distributions.params(proposal)
    factor = LinearAlgebra.cholesky(
        LinearAlgebra.Symmetric(Distributions.scale(proposal)),
    )
    return ImportanceSamplers.FactorStudentT(dof, collect(location), factor)
end

function ImportanceSamplers._prepare_proposal_inputs(
    proposals::AbstractVector{<:Union{
        Distributions.Normal,
        Distributions.MvNormal,
        Distributions.TDist,
        Distributions.Cauchy,
        Distributions.IsoTDist,
        Distributions.DiagTDist,
        Distributions.GenericMvTDist,
    }},
)
    return map(ImportanceSamplers._prepare_proposal_input, proposals)
end

end
