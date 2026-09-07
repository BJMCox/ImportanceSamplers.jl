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

function ImportanceSamplers._prepare_proposal_inputs(
    proposals::AbstractVector{<:Union{Distributions.Normal,Distributions.MvNormal}},
)
    return map(ImportanceSamplers._prepare_proposal_input, proposals)
end

end
