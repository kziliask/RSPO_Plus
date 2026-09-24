using ChainRulesCore
using LinearAlgebra
using Distances

import InferOpt

"""
    RSPOPlusLoss(optimizer; lambda=1.0)

Custom RSPO+ surrogate loss built around an optimizer of the form
`optimizer(theta; kwargs...)`.

The loss is defined on predicted objective coefficients `theta_pred` and true
objective coefficients `theta_true`. Any instance-specific optimizer arguments
should be passed at call time through `kwargs...`.

# Interface
- `loss(theta_pred, theta_true; kwargs...)`
- `loss(theta_pred, theta_true, y_true; kwargs...)`
- `loss(theta_pred, (; theta_true, y_true); kwargs...)`

The third positional argument is accepted for InferOpt compatibility and is not
used by the RSPO+ formula.

# Optional precomputation
You can pass a cached optimizer output through `v_opt=...` to skip the oracle
call, where:
    v_opt = optimizer(theta_true; kwargs...)
"""
struct RSPOPlusLoss{F,T<:AbstractFloat} <: InferOpt.AbstractLossLayer
    optimizer::F
    lambda::T
end

function RSPOPlusLoss(optimizer; lambda=1.0)
    lambda > 0 || throw(ArgumentError("lambda must be positive, got $lambda"))
    return RSPOPlusLoss(optimizer, float(lambda))
end

function Base.show(io::IO, loss::RSPOPlusLoss)
    return print(io, "RSPOPlusLoss(", loss.optimizer, ", ", loss.lambda, ")")
end

sq_dist(x::AbstractVector, y::AbstractVector) = sqeuclidean(x, y)
sq_dist(x::AbstractMatrix, y::AbstractMatrix) = sum(colwise(SqEuclidean(), x, y))

function _get_theta_true(target)
    if hasproperty(target, :theta_true)
        return getproperty(target, :theta_true)
    elseif hasproperty(target, Symbol("θ_true"))
        return getproperty(target, Symbol("θ_true"))
    end
    throw(ArgumentError("RSPOPlusLoss target must contain `theta_true` or `θ_true`."))
end

function _check_lambda_kw(loss::RSPOPlusLoss, kwargs)
    provided_lambda =
        if haskey(kwargs, :lambda)
            kwargs[:lambda]
        elseif haskey(kwargs, Symbol("λ"))
            kwargs[Symbol("λ")]
        else
            return nothing
        end

    if provided_lambda != loss.lambda
        throw(
            ArgumentError(
                "keyword lambda $(provided_lambda) does not match loss.lambda $(loss.lambda)",
            ),
        )
    end
    return nothing
end

function compute_loss_and_gradient(
    loss::RSPOPlusLoss,
    theta_pred::AbstractArray,
    theta_true::AbstractArray;
    v_opt=nothing,
    kwargs...,
)
    lambda = loss.lambda
    optimizer = loss.optimizer
    _check_lambda_kw(loss, kwargs)

    p_u = (theta_true .- 2 .* theta_pred) ./ lambda
    u = 2 .* theta_pred .- theta_true
    v = theta_true

    opt_u = optimizer(u; kwargs...)
    opt_v = isnothing(v_opt) ? optimizer(v; kwargs...) : v_opt

    loss_value = (lambda / 2) * (sq_dist(p_u, opt_v) - sq_dist(p_u, opt_u))
    grad = -2 .* (opt_u .- opt_v)

    return loss_value, grad
end

function compute_loss_and_gradient(
    loss::RSPOPlusLoss,
    theta_pred::AbstractArray,
    theta_true::AbstractArray,
    ::AbstractArray;
    v_opt=nothing,
    kwargs...,
)
    return compute_loss_and_gradient(loss, theta_pred, theta_true; v_opt=v_opt, kwargs...)
end

function compute_loss_and_gradient(
    loss::RSPOPlusLoss,
    theta_pred::AbstractArray,
    target::NamedTuple;
    v_opt=nothing,
    kwargs...,
)
    theta_true = _get_theta_true(target)
    return compute_loss_and_gradient(loss, theta_pred, theta_true; v_opt=v_opt, kwargs...)
end

function (loss::RSPOPlusLoss)(
    theta_pred::AbstractArray,
    theta_true::AbstractArray;
    v_opt=nothing,
    kwargs...,
)
    loss_value, _ =
        compute_loss_and_gradient(loss, theta_pred, theta_true; v_opt=v_opt, kwargs...)
    return loss_value
end

function (loss::RSPOPlusLoss)(
    theta_pred::AbstractArray,
    theta_true::AbstractArray,
    y_true::AbstractArray;
    v_opt=nothing,
    kwargs...,
)
    loss_value, _ =
        compute_loss_and_gradient(loss, theta_pred, theta_true, y_true; v_opt=v_opt, kwargs...)
    return loss_value
end

function (loss::RSPOPlusLoss)(
    theta_pred::AbstractArray,
    target::NamedTuple;
    v_opt=nothing,
    kwargs...,
)
    loss_value, _ = compute_loss_and_gradient(loss, theta_pred, target; v_opt=v_opt, kwargs...)
    return loss_value
end

function ChainRulesCore.rrule(
    loss::RSPOPlusLoss,
    theta_pred::AbstractArray,
    theta_true::AbstractArray;
    v_opt=nothing,
    kwargs...,
)
    loss_value, grad =
        compute_loss_and_gradient(loss, theta_pred, theta_true; v_opt=v_opt, kwargs...)
    pullback(dl) = NoTangent(), dl .* grad, NoTangent()
    return loss_value, pullback
end

function ChainRulesCore.rrule(
    loss::RSPOPlusLoss,
    theta_pred::AbstractArray,
    theta_true::AbstractArray,
    y_true::AbstractArray;
    v_opt=nothing,
    kwargs...,
)
    loss_value, grad = compute_loss_and_gradient(
        loss,
        theta_pred,
        theta_true,
        y_true;
        v_opt=v_opt,
        kwargs...,
    )
    pullback(dl) = NoTangent(), dl .* grad, NoTangent(), NoTangent()
    return loss_value, pullback
end

function ChainRulesCore.rrule(
    loss::RSPOPlusLoss,
    theta_pred::AbstractArray,
    target::NamedTuple;
    v_opt=nothing,
    kwargs...,
)
    loss_value, grad = compute_loss_and_gradient(loss, theta_pred, target; v_opt=v_opt, kwargs...)
    pullback(dl) = NoTangent(), dl .* grad, NoTangent()
    return loss_value, pullback
end
