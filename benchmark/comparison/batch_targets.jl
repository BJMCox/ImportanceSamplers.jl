# Benchmark callback, not package machinery. The scalar target remains the
# density oracle and gradient source. Scratch allocation/transfer is timed.
observation_value(family, y, eta, logscale) = first(observation(family,y,eta,logscale))

function regression_batch!(values, samples, p)
    d = size(p.X,2)
    capacity = size(p.workspace,2)
    for start in 1:capacity:length(values)
        columns = start:min(start+capacity-1,length(values))
        beta = view(samples,1:d,columns)
        eta = view(p.workspace,:,1:length(columns))
        logscale = p.family isa Val{:robust} ? view(samples,d+1:d+1,columns) : zero(eltype(samples))
        mul!(eta,p.X,beta)
        eta .= observation_value.(Ref(p.family),p.y,eta,logscale)
        output = view(values,columns)
        output .= vec(sum(eta;dims=1)) .- vec(sum(abs2,beta;dims=1)) ./ 8
        if p.family isa Val{:robust}
            output .-= abs2.(view(samples,d+1,columns)) ./ 2
        end
    end
    return nothing
end
