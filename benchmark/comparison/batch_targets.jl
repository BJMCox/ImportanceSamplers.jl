# Benchmark callback, not package machinery. The scalar target remains the
# density oracle and gradient source. Scratch allocation/transfer is timed.
function regression_batch!(values, samples, p)
    d = size(p.X,2)
    capacity = isnothing(p.workspace) ? p.batch_capacity : size(p.workspace,2)
    workspace = isnothing(p.workspace) ?
        similar(samples,size(p.X,1),min(capacity,length(values))) : p.workspace
    device = MLDataDevices.get_device(values)
    for start in 1:capacity:length(values)
        columns = start:min(start+capacity-1,length(values))
        beta = view(samples,1:d,columns)
        eta = view(workspace,:,1:length(columns))
        logscale = p.family isa Val{:robust} ? view(samples,d+1:d+1,columns) : zero(eltype(samples))
        mul!(eta,p.X,beta)
        regression_reduce!(device,view(values,columns),beta,eta,logscale,p)
    end
    return nothing
end

function regression_reduce!(device, values, beta, eta, logscale, p)
    eta .= observation_value.(Ref(p.family),p.y,eta,logscale)
    values .= vec(sum(eta;dims=1)) .- vec(sum(abs2,beta;dims=1)) ./ 8
    p.family isa Val{:robust} && (values .-= vec(logscale).^2 ./ 2)
    return nothing
end

# BLAS and Julia threading run in separate phases. Fuse the CPU
# likelihood and sum so each sample needs no intermediate likelihood array.
function regression_reduce!(::MLDataDevices.CPUDevice, values, beta, eta, logscale, p)
    Threads.@threads for j in eachindex(values)
        scale = logscale isa Number ? logscale : logscale[1,j]
        value = -sum(abs2,view(beta,:,j))/8 - scale^2/2
        for i in axes(eta,1)
            value += observation_value(p.family,p.y[i],eta[i,j],scale)
        end
        values[j] = value
    end
    return nothing
end
