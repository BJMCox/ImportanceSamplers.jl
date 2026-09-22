# BAT's paper model in noncentred coordinates. The four bounded parent priors
# use logits. The five detector effects have independent standard normal priors.
function signal_background()
    directory = joinpath(@__DIR__, "data", "signal-background")
    rows(file) = split.(readlines(joinpath(directory, file))[2:end], ',')
    summary = rows("summary_dataset_table.csv")
    events = rows("sample_table.csv")
    exposure = [parse(Float64, row[2]) for row in summary]
    efficiency = [parse(Float64, row[3]) for row in summary]
    dataset = [parse(Int, row[1]) for row in events]
    energy = [parse(Float64, row[2]) for row in events]
    all(count(==(i), dataset) == parse(Int, row[6]) for (i,row) in enumerate(summary)) ||
        error("Signal-background event counts do not match the source summary")
    return (; name="signal-background", dimension=4+length(exposure), observations=length(energy),
        data=(; exposure, efficiency, dataset, energy))
end

function evaluate(g, theta, p::NamedTuple{(:exposure, :efficiency, :dataset, :energy)})
    uS, usigma, um, ulambda = ntuple(i -> sigmoid(theta[i]), 4)
    logS = log(10.0) - softplus(-theta[1])
    S = exp(logS)
    sigma = 0.1 + 0.9usigma
    m = 1e-10 + (20.0-1e-10)*um
    lambda = 1e-10 + (100.0-1e-10)*ulambda
    # Uniform prior widths cancel the interval Jacobians. Constants omitted
    # below do not depend on theta, including the Poisson log-factorials.
    value = -sum(softplus(theta[i])+softplus(-theta[i]) for i in 1:4)
    signal_score = zero(value)
    lambda_score = zero(value)
    for j in eachindex(p.exposure)
        z = theta[4+j]
        background = p.exposure[j]*exp(log(m)-sigma^2/2+sigma*z)
        signal = p.exposure[j]*p.efficiency[j]*S
        value -= background + signal + z^2/2
        signal_score -= signal
        g === nothing || (g[4+j] = -background)
    end
    for i in eachindex(p.energy)
        j, energy = p.dataset[i], p.energy[i]
        logbackground = log(m)-sigma^2/2+sigma*theta[4+j] - log(lambda)-energy/lambda
        logsignal = logS+log(p.efficiency[j])-log(2.0)-log(2pi)/2-(energy-100)^2/8
        lograte = logbackground > logsignal ?
            logbackground + log1p(exp(logsignal-logbackground)) :
            logsignal + log1p(exp(logbackground-logsignal))
        value += log(p.exposure[j])+lograte
        if g !== nothing
            responsibility = exp(logbackground-lograte)
            g[4+j] += responsibility
            signal_score += exp(logsignal-lograte)
            lambda_score += responsibility*(energy/lambda-1)
        end
    end
    if g !== nothing
        sigma_score = zero(value)
        mean_score = zero(value)
        for j in eachindex(p.exposure)
            score, z = g[4+j], theta[4+j]
            sigma_score += score*(z-sigma)
            mean_score += score
            g[4+j] = sigma*score-z
        end
        g[1] = signal_score*(1-uS) + 1-2uS
        g[2] = sigma_score*0.9usigma*(1-usigma) + 1-2usigma
        g[3] = mean_score*(20.0-1e-10)*um*(1-um)/m + 1-2um
        g[4] = lambda_score*(100.0-1e-10)*ulambda*(1-ulambda)/lambda + 1-2ulambda
    end
    return value
end
