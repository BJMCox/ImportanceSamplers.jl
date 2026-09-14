struct _LocatedTransform{L,T}
    location::L
    transform::T
end

struct _NamedTransformLayout{B<:NamedTuple} <: AbstractSampleTransform
    blocks::B
end

struct _FlatTransformLayout{B<:NamedTuple} <: AbstractSampleTransform
    blocks::B
    dimension::Int
end

_logical_block_length(block::_LocatedTransform{Int}) = 1
_logical_block_length(block::_LocatedTransform{<:UnitRange}) = length(block.location)
_logical_block_length(block::_LocatedTransform{<:UnitRange,SimplexTransform}) =
    block.transform.dimension

_logical_block_shape(block::_LocatedTransform{Int}) = ()
_logical_block_shape(block::_LocatedTransform{<:UnitRange}) = (_logical_block_length(block),)

function _allocate_flat_samples(prototype, ::Type{T}, layout::_FlatTransformLayout, count) where {T}
    return map(layout.blocks) do block
        similar(prototype, T, _logical_block_shape(block)..., count)
    end
end

function _prepare_transformed_proposal(base, transform::AbstractSampleTransform)
    _validate_known_transform_input(transform, _known_proposal_transform_input(base))
    return TransformedProposal(base, transform, _PreparedProposalToken())
end

function _prepare_transformed_proposal(base, specification::NamedTuple)
    if isempty(specification) || all(value -> value isa AbstractSampleTransform, specification)
        return _prepare_named_transformed_proposal(base, specification)
    elseif all(value -> value isa Pair, specification)
        return _prepare_flat_transformed_proposal(base, specification)
    end
    throw(
        ArgumentError(
            "a named transform layout must contain only transforms or only selector pairs",
        ),
    )
end

function _prepare_transformed_proposal(base, transform)
    throw(
        ArgumentError(
            "transform must be an AbstractSampleTransform or a named transform layout",
        ),
    )
end

function _prepare_named_transformed_proposal(base, specification::NamedTuple)
    throw(
        ArgumentError(
            "named partial transforms require a ProductProposal base with a named schema",
        ),
    )
end

@generated function _complete_named_transform_blocks(
    specification::NamedTuple{SpecifiedNames},
    base_blocks::NamedTuple{Names},
    ::Val{Names},
) where {SpecifiedNames,Names}
    expressions = map(Names) do name
        transform = name in SpecifiedNames ?
                    :(getfield(specification, $(QuoteNode(name)))) :
                    :(IdentityTransform())
        :(_prepare_named_transform_block(
            getfield(base_blocks, $(QuoteNode(name))), $(QuoteNode(name)), $transform,
        ))
    end
    return :(NamedTuple{$Names}(($(expressions...),)))
end

_known_proposal_transform_input(base) = nothing
_known_proposal_transform_input(base::_GaussianProposal) = base.location

_validate_known_transform_input(transform, input) = nothing
function _validate_known_transform_input(
    transform::Union{PositiveTransform,SoftplusTransform},
    input,
)
    input === nothing || input isa _NativeGaussianFloat || input isa Int || throw(
        DimensionMismatch("$(typeof(transform)) requires one unconstrained scalar"),
    )
    return nothing
end
function _validate_known_transform_input(transform::IntervalTransform{T}, input) where {T}
    input === nothing || input isa _NativeGaussianFloat || input isa Int || throw(
        DimensionMismatch("$(typeof(transform)) requires one unconstrained scalar"),
    )
    input isa _NativeGaussianFloat && typeof(input) !== T && throw(
        ArgumentError(
            "interval endpoint type must match native Gaussian coordinate type: " *
            "got $T and $(typeof(input))",
        ),
    )
    return nothing
end
function _validate_known_transform_input(transform::SimplexTransform, input)
    input === nothing && return nothing
    required = transform.dimension - 1
    input isa AbstractVector && length(input) == required || throw(
        DimensionMismatch("SimplexTransform($(transform.dimension)) requires $required coordinates"),
    )
    return nothing
end

function _prepare_named_transform_block(base, location, transform)
    _validate_known_transform_input(transform, _known_proposal_transform_input(base))
    return _LocatedTransform(location, transform)
end

function _prepare_named_transformed_proposal(
    base::ProductProposal{NamedTuple{Names,Types}},
    specification::NamedTuple,
) where {Names,Types}
    unknown_names = filter(name -> !(name in Names), keys(specification))
    isempty(unknown_names) || throw(
        ArgumentError("unknown named transform fields: $(join(unknown_names, ", "))"),
    )
    blocks = _complete_named_transform_blocks(specification, base.blocks, Val(Names))
    return TransformedProposal(base, _NamedTransformLayout(blocks), _PreparedProposalToken())
end

function _validated_flat_selector(selector, dimension)
    if selector isa Int && !(selector isa Bool)
        1 <= selector <= dimension || throw(
            ArgumentError(
                "flat transform selector $selector is out of bounds for dimension $dimension",
            ),
        )
        return selector
    elseif selector isa UnitRange{Int}
        isempty(selector) && throw(ArgumentError("flat transform ranges must be nonempty"))
        first(selector) >= 1 && last(selector) <= dimension || throw(
            ArgumentError(
                "flat transform selector $selector is out of bounds for dimension $dimension",
            ),
        )
        return selector
    end
    throw(
        ArgumentError(
            "flat transform selectors must be Int values or contiguous UnitRange{Int} values",
        ),
    )
end

_selector_indices(selector::Int) = selector:selector
_selector_indices(selector::UnitRange{Int}) = selector

function _prepare_flat_transform_block(pair::Pair, dimension, selected)
    selector = _validated_flat_selector(first(pair), dimension)
    transform = last(pair)
    transform isa AbstractSampleTransform || throw(
        ArgumentError("flat transform selector values must be sample transforms"),
    )
    _validate_known_transform_input(transform, selector)
    for index in _selector_indices(selector)
        selected[index] && throw(ArgumentError("flat transform selectors must be disjoint"))
        selected[index] = true
    end
    return _LocatedTransform(selector, transform)
end

function _prepare_flat_transform_layout(base, specification::NamedTuple)
    dimension = _proposal_dimension(base)
    dimension isa Int && dimension > 0 || throw(
        ArgumentError("flat selector transforms require a base with a known positive dimension"),
    )
    isempty(specification) && throw(ArgumentError("flat selector transforms must contain blocks"))

    selected = falses(dimension)
    blocks = map(pair -> _prepare_flat_transform_block(pair, dimension, selected), specification)
    all(selected) || throw(
        ArgumentError("flat transform selectors must collectively cover the base dimension"),
    )

    return _FlatTransformLayout(blocks, dimension)
end

function _prepare_flat_transformed_proposal(base, specification::NamedTuple)
    return TransformedProposal(
        base,
        _prepare_flat_transform_layout(base, specification),
        _PreparedProposalToken(),
    )
end

function _with_transform_location(f, block::_LocatedTransform, value)
    try
        return f(block.transform, value)
    catch error
        error isa InvalidTransformError || rethrow()
        throw(InvalidTransformError(error.reason, block.location))
    end
end

_transform_block(block, value) =
    _with_transform_location(_transform_with_logjac, block, value)
_inverse_transform_block(block, value) =
    _with_transform_location(_inverse_with_logjac, block, value)

_sum_transform_logjacs(results::Tuple{T}) where {T} = last(first(results))
_sum_transform_logjacs(results::Tuple) =
    last(first(results)) + _sum_transform_logjacs(Base.tail(results))

function _split_transform_results(results::NamedTuple)
    return map(first, results), _sum_transform_logjacs(values(results))
end

function _transform_with_logjac(
    layout::_NamedTransformLayout{B},
    coordinate::NamedTuple{Names},
) where {Names,B<:NamedTuple{Names}}
    return _split_transform_results(map(_transform_block, layout.blocks, coordinate))
end

function _transform_with_logjac(layout::_NamedTransformLayout, coordinate::NamedTuple)
    throw(ArgumentError("named transform input fields must match the prepared layout"))
end

function _inverse_with_logjac(
    layout::_NamedTransformLayout{B},
    logical_value::NamedTuple{Names},
) where {Names,B<:NamedTuple{Names}}
    return _split_transform_results(
        map(_inverse_transform_block, layout.blocks, logical_value),
    )
end

function _inverse_with_logjac(layout::_NamedTransformLayout, logical_value::NamedTuple)
    throw(ArgumentError("named transform input fields must match the prepared layout"))
end

@inline _selected_coordinate(coordinate::AbstractVector, selector::Int) =
    @inbounds coordinate[selector]
@inline _selected_coordinate(coordinate::AbstractVector, selector::UnitRange{Int}) =
    @inbounds coordinate[selector]

function _transform_selected_block(coordinate::AbstractVector, block::_LocatedTransform)
    return _transform_block(block, _selected_coordinate(coordinate, block.location))
end

function _transform_with_logjac(layout::_FlatTransformLayout, coordinate::AbstractVector)
    length(coordinate) == layout.dimension || throw(
        DimensionMismatch(
            "flat transform requires $(layout.dimension) unconstrained coordinates",
        ),
    )
    results = map(Base.Fix1(_transform_selected_block, coordinate), layout.blocks)
    return _split_transform_results(results)
end

function _store_selected_coordinate!(
    coordinate::Vector{T},
    selector::Int,
    selected::T,
) where {T<:_TransformFloat}
    @inbounds coordinate[selector] = selected
    return coordinate
end

function _store_selected_coordinate!(
    coordinate::Vector{T},
    selector::UnitRange{Int},
    selected::AbstractVector{T},
) where {T<:_TransformFloat}
    length(selected) == length(selector) || throw(
        DimensionMismatch("inverse transform output does not match selector $selector"),
    )
    copyto!(coordinate, first(selector), selected, firstindex(selected), length(selector))
    return coordinate
end

_store_inverse_blocks!(coordinate, ::Tuple{}, ::Tuple{}) = coordinate

function _store_inverse_blocks!(coordinate, blocks::Tuple, results::Tuple)
    block = first(blocks)
    selected, _ = first(results)
    _store_selected_coordinate!(coordinate, block.location, selected)
    return _store_inverse_blocks!(coordinate, Base.tail(blocks), Base.tail(results))
end

function _inverse_with_logjac(
    layout::_FlatTransformLayout{B},
    logical_value::NamedTuple{Names},
) where {Names,B<:NamedTuple{Names}}
    results = map(_inverse_transform_block, layout.blocks, logical_value)
    T = _transform_float_type(first(first(values(results))))
    coordinate = Vector{T}(undef, layout.dimension)
    _store_inverse_blocks!(coordinate, values(layout.blocks), values(results))
    return coordinate, _sum_transform_logjacs(values(results))
end

function _inverse_with_logjac(layout::_FlatTransformLayout, logical_value::NamedTuple)
    throw(ArgumentError("flat transform input fields must match the prepared layout"))
end

function Random.rand(rng::Random.AbstractRNG, proposal::TransformedProposal)
    logical_value, _ = _transform_with_logjac(
        proposal.transform,
        Random.rand(rng, proposal.base),
    )
    return logical_value
end

_transform_float_type(value::T) where {T<:_TransformFloat} = T
_transform_float_type(value::AbstractVector{T}) where {T<:_TransformFloat} = T

function _transform_float_type(value::NamedTuple)
    isempty(value) && throw(ArgumentError("transformed samples must contain numeric leaves"))
    return _transform_float_type(first(values(value)))
end

_zero_transform_input(::AbstractSampleTransform, ::T) where {T<:_TransformFloat} =
    zero(T)
_zero_transform_input(::AbstractSampleTransform, value::AbstractVector{T}) where {T<:_TransformFloat} =
    zeros(T, length(value))
_zero_transform_input(transform::SimplexTransform, ::AbstractVector{T}) where {T<:_TransformFloat} =
    zeros(T, transform.dimension - 1)

_zero_transform_input(block::_LocatedTransform, value) =
    _zero_transform_input(block.transform, value)

_zero_transform_input(layout::_NamedTransformLayout{B}, value::NamedTuple{Names}) where
    {Names,B<:NamedTuple{Names}} = map(_zero_transform_input, layout.blocks, value)

_zero_transform_input(layout::_FlatTransformLayout{B}, value::NamedTuple{Names}) where
    {Names,B<:NamedTuple{Names}} = zeros(_transform_float_type(value), layout.dimension)

function _out_of_support_logdensity(proposal::TransformedProposal, logical_value)
    coordinate = _zero_transform_input(proposal.transform, logical_value)
    _, logabsjac = _transform_with_logjac(proposal.transform, coordinate)
    density = DensityInterface.logdensityof(proposal.base, coordinate) - logabsjac
    return oftype(density, -Inf)
end

function DensityInterface.logdensityof(proposal::TransformedProposal, logical_value)
    inverse = try
        _inverse_with_logjac(proposal.transform, logical_value)
    catch error
        if error isa InvalidTransformError
            return _out_of_support_logdensity(proposal, logical_value)
        end
        rethrow()
    end
    coordinate, logabsjac = inverse
    return DensityInterface.logdensityof(proposal.base, coordinate) - logabsjac
end

@inline DensityInterface.DensityKind(::TransformedProposal) = DensityInterface.HasDensity()

_accelerator_proposal_limit(proposal::TransformedProposal) =
    _accelerator_proposal_limit(proposal.base)
