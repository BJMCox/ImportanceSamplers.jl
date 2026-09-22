# These targets share their scalar log density across CPU and CUDA. Constants
# independent of the sampled parameters are omitted throughout.
# Both branches have derivative 1/2 at zero; a max/abs composition does not
# preserve that derivative under ForwardDiff's tie rules.
softplus(x) = x > zero(x) ? x + log1p(exp(-x)) : log1p(exp(x))
sigmoid(x) = exp(-softplus(-x))

include("signal_background.jl")

observation(::Val{:linear}, y, eta, logscale) = (-(y-eta)^2/2, y-eta, zero(eta))
observation(::Val{:logistic}, y, eta, logscale) = (y*eta-softplus(eta), y-sigmoid(eta), zero(eta))
observation(::Val{:poisson}, y, eta, logscale) = (y*eta-exp(eta), y-exp(eta), zero(eta))
function observation(::Val{:robust}, y, eta, logscale)
    residual = y-eta
    scaled = residual^2 * exp(-2logscale) / 4
    return (-logscale-2.5log1p(scaled),
            5residual/(4exp(2logscale)+residual^2), -1+5scaled/(1+scaled))
end

function evaluate(g, theta, p)
    d = size(p.X, 2)
    logscale = p.family isa Val{:robust} ? theta[d+1] : zero(eltype(theta))
    value = -sum(abs2, view(theta, 1:d))/8 - logscale^2/2
    if g !== nothing
        for j in 1:d
            g[j] = -theta[j]/4
        end
        p.family isa Val{:robust} && (g[d+1] = -logscale)
    end
    for i in eachindex(p.y)
        eta = zero(eltype(theta))
        for j in 1:d
            eta += p.X[i,j]*theta[j]
        end
        v, slope, scale_slope = observation(p.family, p.y[i], eta, logscale)
        value += v
        if g !== nothing
            for j in 1:d
                g[j] += p.X[i,j]*slope
            end
            p.family isa Val{:robust} && (g[d+1] += scale_slope)
        end
    end
    return value
end

function evaluate(g, theta, p::NamedTuple{(:schools, :errors)})
    mu, logtau = theta[9], theta[10]
    tau = exp(logtau)
    value = -sum(abs2, view(theta, 1:8))/2 - mu^2/50 - tau^2/50 + logtau
    if g !== nothing
        g[9], g[10] = -mu/25, 1-tau^2/25
    end
    for j in 1:8
        residual = p.schools[j]-mu-tau*theta[j]
        slope = residual/p.errors[j]^2
        value -= residual*slope/2
        if g !== nothing
            g[j] = -theta[j]+tau*slope
            g[9] += slope
            g[10] += tau*theta[j]*slope
        end
    end
    return value
end

logtarget(theta, p) = evaluate(nothing, theta, p)

function regression(family, d, n, seed)
    rng = Xoshiro(seed)
    X = randn(rng, n, d)
    for j in 3:d
        X[:,j] .= 0.8 .* X[:,j-1] .+ 0.6 .* X[:,j]
    end
    X[:,1] .= 1
    beta = [0.5cos(j)/sqrt(d) for j in 1:d]
    eta = X*beta
    y = if family === :linear
        eta + randn(rng, n)
    elseif family === :logistic
        Float64.(rand(rng, n) .< sigmoid.(eta))
    elseif family === :poisson
        Float64[rand(rng, Poisson(exp(x))) for x in eta]
    else
        eta + 0.7rand(rng, TDist(4), n)
    end
    return (; name=String(family), dimension=d+(family===:robust), observations=n,
            data=(; X, y, family=Val(family)))
end

function models()
    return [
        regression(:linear, 32, 1024, 101),
        regression(:logistic, 12, 1024, 102),
        regression(:poisson, 12, 1024, 103),
        regression(:robust, 12, 512, 104),
        (; name="eight_schools", dimension=10, observations=8,
           data=(; schools=[28.,8.,-3.,7.,-1.,1.,18.,12.],
                   errors=[15.,10.,16.,11.,9.,11.,10.,18.])),
        signal_background(),
    ]
end

# Every measured run pays for the same fit. MCMC also receives this location
# and full-covariance preconditioner, rather than a deliberately poor start.
function laplace(model)
    loss(x) = -logtarget(x, model.data)
    function gradient!(g, x)
        evaluate(g, x, model.data)
        g .*= -1
        return g
    end
    fit = Optim.optimize(loss, gradient!, zeros(model.dimension), Optim.BFGS(),
                         Optim.Options(iterations=1000, g_abstol=1e-7))
    Optim.converged(fit) || error("Laplace fit did not converge: $(model.name)")
    location = Optim.minimizer(fit)
    precision = cholesky(Symmetric(ForwardDiff.hessian(loss, location)))
    covariance = precision \ Matrix{Float64}(I, model.dimension, model.dimension)
    return (; location, factor=Matrix(cholesky(Symmetric(covariance)).L))
end

struct WhitenedTarget{P,F}
    data::P
    fit::F
end
LogDensityProblems.dimension(p::WhitenedTarget) = length(p.fit.location)
LogDensityProblems.capabilities(::Type{<:WhitenedTarget}) = LogDensityProblems.LogDensityOrder{1}()
function LogDensityProblems.logdensity(p::WhitenedTarget, z)
    return logtarget(p.fit.location + p.fit.factor*z, p.data)
end
function LogDensityProblems.logdensity_and_gradient(p::WhitenedTarget, z)
    theta = p.fit.location + p.fit.factor*z
    g = similar(theta)
    value = evaluate(g, theta, p.data)
    return value, p.fit.factor'*g
end
