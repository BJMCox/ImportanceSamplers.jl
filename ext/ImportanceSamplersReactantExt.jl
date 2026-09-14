module ImportanceSamplersReactantExt

import ImportanceSamplers as IS
import Reactant
import Random
import MLDataDevices
import ADTypes
import DifferentiationInterface as DI
import KernelAbstractions as KA
import LinearAlgebra as LA

const _ReactantStorage = Union{Reactant.AnyConcreteRArray,
    SubArray{T,N,P} where {T,N,P<:Reactant.AnyConcreteRArray}}

struct _ReactantRNG{R} <: Random.AbstractRNG
    rng::R
end

IS._owned_backend_rng(device::MLDataDevices.ReactantDevice, seed::UInt64) =
    _ReactantRNG(device(Random.Xoshiro(seed)))

function Random.rand!(rng::_ReactantRNG, values::AbstractArray)
    iszero(length(values)) || (Reactant.@jit Random.rand!(rng.rng, values))
    return values
end

function Random.randn!(rng::_ReactantRNG, values::AbstractArray)
    iszero(length(values)) || (Reactant.@jit Random.randn!(rng.rng, values))
    return values
end

# Reactant arguments remain managed arrays, not isbits kernel pointers. Its live
# preflight compiles the target and gradients before sampling consumes the RNG.
function IS._preflight_kernel_argument(device::MLDataDevices.ReactantDevice, kernel, argument)
    try
        KA.argconvert(kernel, argument)
    catch
        throw(IS.SamplerDeviceError(device, :kernel_argument_unsupported))
    end
    return nothing
end

function IS._preflight_accelerator_method(
    device::MLDataDevices.ReactantDevice, target, algorithm::IS.FirstOrderGRAMIS,
    state::IS._PreparedFirstOrderGRAMIS, buffers, factor_execution,
)
    # Unraised cooperative kernels retain NVVM barriers on Reactant's CPU
    # backend. Reject before its compiler aborts the Julia process.
    Reactant.XLA.device_kind(Reactant.XLA.device(state.committed.locations)) == "cpu" &&
        throw(IS.SamplerDeviceError(device, :reactant_cpu_cooperative_kernels))
    return invoke(IS._preflight_accelerator_method,
        Tuple{Any,Any,IS.FirstOrderGRAMIS,IS._PreparedFirstOrderGRAMIS,Any,Any},
        device, target, algorithm, state, buffers, factor_execution)
end

# Preserve the existing solve workspace. Reactant cannot trace the coalesced
# PermutedDimsArray/reshape view used by the native CUDA kernel.
IS._fused_mis_solve_scratch(scratch::Reactant.AnyConcreteRArray, backend) = scratch

function _pooled_covariance!(covariance, factors)
    packed = reshape(factors, size(factors, 1), :)
    covariance .= (packed * transpose(packed)) / eltype(factors)(size(factors, 3))
    return nothing
end

function IS._pooled_covariance!(covariance::Reactant.AnyConcreteRArray, factors, ::IS._KernelExecution)
    Reactant.@jit _pooled_covariance!(covariance, factors)
    return nothing
end

function _factor!(factor)
    decomposition = LA.cholesky(LA.Hermitian(factor, :L); check=false)
    factor .= transpose(decomposition.factors)
    return decomposition.info
end

function IS._factor_pooled_covariance!(factor::Reactant.AnyConcreteRArray)
    success = Reactant.@jit _factor!(factor)
    Bool(success) || throw(IS._FirstOrderGRAMISRepulsionError(
        0, :pooled_factorization_failed, :nonfinite_result))
    return nothing
end

function IS._gaussian_potrf!(::MLDataDevices.ReactantDevice, factor::Reactant.AnyConcreteRArray)
    success = Reactant.@jit _factor!(factor)
    Bool(success) || throw(LA.PosDefException(0))
    return factor
end

function IS._preflight_gaussian_factorization!(
    device::MLDataDevices.ReactantDevice, state::IS._PreparedMomentSampler{S,O,L,H},
) where {S,O,L,H<:IS._FactorProposalHistory}
    # Moment-sampler support stays GPU-only while cooperative CPU lowering
    # remains unvalidated. Check before launching any adaptive kernels.
    Reactant.XLA.device_kind(Reactant.XLA.device(state.workspace.covariance)) == "cpu" &&
        throw(IS.SamplerDeviceError(device, :reactant_cpu_cooperative_kernels))
    IS._gaussian_potrf!(device, state.workspace.covariance)
    return nothing
end

function IS._weighted_moments!(mean::Reactant.AnyConcreteRArray, covariance, centered, samples, weights)
    Reactant.@jit IS._weighted_moments!(mean, covariance, centered, samples, weights)
    return nothing
end

function _store_factor_candidate!(history, slot, workspace)
    history.means[:, slot] = workspace.candidate_mean
    history.factors[:, :, slot] = workspace.candidate_scale
    history.lognormalizers[slot:slot] = workspace.candidate_lognormalizer
    return nothing
end

function IS._store_gaussian_candidate!(
    history::IS._FactorProposalHistory{<:Reactant.AnyConcreteRArray}, slot, workspace::IS._MomentWorkspace,
)
    Reactant.@jit _store_factor_candidate!(history, slot, workspace)
    return nothing
end

function IS._clip_logweights!(clipped::_ReactantStorage, raw, threshold)
    Reactant.@jit IS._clip_logweights!(clipped, raw, threshold)
    return clipped
end

function _normalize_weights!(weights, logweights, count)
    active = view(weights, 1:count)
    values = view(logweights, 1:count)
    maximum_logweight = maximum(values; dims=1)
    active .= exp.(values .- ifelse.(isfinite.(maximum_logweight), maximum_logweight, zero(eltype(values))))
    total = sum(active; dims=1)
    squared_sum = sum(abs2, active; dims=1)
    active ./= ifelse.(iszero.(total), one(eltype(active)), total)
    return vcat(maximum_logweight, total, squared_sum)
end

function IS._normalize_gaussian_weights!(
    weights::_ReactantStorage, logweights, count, transfers::IS._ResultTransferCounter,
)
    moments = Array(Reactant.@jit _normalize_weights!(weights, logweights, count))
    IS._record_reported_transfer!(transfers, 1, sizeof(moments), Val(:logweight_moments))
    isfinite(moments[2]) && moments[2] > 0 || throw(IS.AllZeroWeightsError())
    return IS._logweight_summary(moments..., count)
end

function _whiten_means!(output, factor, means)
    output .= Reactant.Ops.triangular_solve(factor, means;
        left_side=true, lower=true, unit_diagonal=false, transpose_a='N')
    return nothing
end

function IS._whiten_means!(output::Reactant.AnyConcreteRArray, factor, means)
    Reactant.@jit _whiten_means!(output, factor, means)
    return nothing
end

function _logweight_moments(values)
    maximum_logweight = maximum(values; dims=1)
    shifted = exp.(values .- ifelse.(isfinite.(maximum_logweight), maximum_logweight, zero(eltype(values))))
    return vcat(maximum_logweight, sum(shifted; dims=1), sum(abs2, shifted; dims=1))
end

function IS._logweight_moments(values::_ReactantStorage)
    return Tuple(Array(Reactant.@jit _logweight_moments(values)))
end

function _logsumexp(values)
    maximum_logweight = maximum(values; dims=1)
    shifted = exp.(values .- ifelse.(isfinite.(maximum_logweight), maximum_logweight, zero(eltype(values))))
    return vcat(maximum_logweight, sum(shifted; dims=1))
end

function IS._logsumexp_accumulator(values::_ReactantStorage)
    # Reactant tensors cannot store the struct-valued reduction used by CUDA.
    return IS._LogSumExpAccumulator(Array(Reactant.@jit _logsumexp(values))...)
end

IS._normalized_weights(values::_ReactantStorage, total) = Reactant.@jit IS._normalized_weights(values, total)

function _resampling_cdf!(cdf, logweights)
    moments = _normalize_weights!(cdf, logweights, length(cdf))
    cdf .= cumsum(cdf)
    return moments[1:2]
end

function IS._resampling_cdf!(
    cdf::_ReactantStorage, logweights, transfers::IS._ResultTransferCounter=IS._ResultTransferCounter(0, 0),
)
    summary = Array(Reactant.@jit _resampling_cdf!(cdf, logweights))
    # The CDF maximum and sum share one transfer, attributed to cdf_sum.
    IS._record_reported_transfer!(transfers, 1, sizeof(summary), Val(:cdf_sum))
    isfinite(summary[2]) && summary[2] > 0 || throw(IS.AllZeroWeightsError())
    return cdf
end

_allfinite(values) = all(isfinite.(values))
IS._local_means_valid(values::Reactant.AnyConcreteRArray) = Bool(Reactant.@jit _allfinite(values))

function _gramis_diagnostics(status, steps, trials)
    return vcat(sum(Int.(status .== IS._POPULATION_ALL_ZERO_LOCAL); dims=1),
        sum(Int.(status .== IS._POPULATION_TEMPERING_FAILED); dims=1),
        sum(Int.(iszero.(steps)); dims=1), sum(trials; dims=1))
end

function IS._first_order_gramis_diagnostic_summary(
    ::MLDataDevices.ReactantDevice, status, steps, trials, transfers, ::IS._KernelExecution,
)
    values = Array(Reactant.@jit _gramis_diagnostics(status, steps, trials))
    transfers.count += 1
    transfers.bytes += sizeof(values)
    return NamedTuple{(:all_zero, :tempering, :backtracking, :target_trials)}(Tuple(values))
end

struct _BoundReactantGradient{F,T,S,E} <: IS._BoundBatchGradient
    compiled::F
    target::T
    seeds::S
    status::E
end

IS._record_gradient_transfers!(transfers, ::_BoundReactantGradient) =
    IS._record_scalar_transfer!(transfers, Int32)

@inline _target_value(f, p, x) =
    (IS._batch_target_value(f, x, p), UInt16(0), 0)

@inline function _target_value(f::IS._NamedADLogDensity, p, x)
    logical, logjac, reason, block = IS._coordinate_to_logical(f.layout, x)
    iszero(reason) || return (oftype(logjac, NaN), reason, block)
    return (IS._batch_target_value(f.logdensity, logical, p) + logjac, reason, block)
end

KA.@kernel function _target_kernel!(values, locations, f, context, status)
    column = KA.@index(Global, Linear)
    value, reason, block = _target_value(f, context[], view(locations, :, column))
    values[column] = value
    status[1, column] = reason
    status[2, column] = block
end

function _batch_target!(values, locations, f, context, status)
    _target_kernel!(KA.get_backend(locations), 64)(
        values, locations, f, context, status; ndrange=size(locations, 2))
    return nothing
end

function _gradient!(values, gradients, locations, target, seeds, status)
    # Prepare inside the trace. DI preparations contain the concrete input types.
    DI.pullback!(_batch_target!, values, (gradients,), target.adtype, locations,
        (seeds,), DI.Constant(target.logdensity), DI.Constant(Ref(target.context)),
        DI.Constant(status))
    # Exceptions stay on the host. Return one scalar, not per-sample host reads.
    return minimum(ifelse.(iszero.(view(status, 1, :)),
        typemax(Int32), Int32.(1:size(locations, 2))))
end

function IS._prepare_accelerator_gradient(
    target::IS._PreparedLogTarget{F,P,A,Nothing},
    locations::Reactant.AnyConcreteRArray, values,
) where {F,P,A<:ADTypes.AutoEnzyme}
    ADTypes.mode(target.adtype) isa Union{ADTypes.ReverseMode,ADTypes.ForwardOrReverseMode} ||
        IS._reject_cpu_gradient_on_accelerator(view(locations, :, 1))
    seeds = similar(values)
    fill!(seeds, one(eltype(seeds)))
    status = similar(locations, Int32, 2, size(locations, 2))
    gradients = similar(locations)
    # Pre-AD tensor optimization can lose accumulated slice tangents. Keep
    # optimization after AD, where the numerical regression preserves them.
    options = Reactant.CompileOptions(; optimization_passes=:after_enzyme,
        raise=true, raise_first=true, excluded_passes=["transpose_is_reshape", "cse_reshape"])
    compiled = Reactant.@compile compile_options=options _gradient!(
        values, gradients, locations, target, seeds, status)
    return _BoundReactantGradient(compiled, target, seeds, status)
end

function IS._batch_value_and_gradient!(
    values, gradients, bound::_BoundReactantGradient, locations,
)
    first_failure = Int(bound.compiled(
        values, gradients, locations, bound.target, bound.seeds, bound.status))
    if first_failure != typemax(Int32)
        reason, block = Array(view(bound.status, :, first_failure))
        IS._throw_named_transform_failure(bound.target.logdensity.layout, UInt16(reason), Int(block))
    end
    return nothing
end

end
