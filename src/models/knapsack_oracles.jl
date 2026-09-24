using JuMP
using Gurobi
import InferOpt
using LinearAlgebra

const KNAPSACK_MOI = JuMP.MOI

if !@isdefined(get_knapsack_gurobi_env)
    const KNAPSACK_GUROBI_ENV = Ref{Gurobi.Env}()

    function get_knapsack_gurobi_env()
        if !isassigned(KNAPSACK_GUROBI_ENV)
            KNAPSACK_GUROBI_ENV[] = Gurobi.Env()
        end
        return KNAPSACK_GUROBI_ENV[]
    end
end

"""
    KnapsackInstance(weights, capacity)

Container for a fractional knapsack instance used by the LP and projection
oracles. The oracle objective is the full vector `theta`.
"""
struct KnapsackInstance{T<:AbstractFloat,V<:AbstractVector{T}}
    weights::V
    capacity::T
end

function KnapsackInstance(
    weights::AbstractVector{<:Real},
    capacity::Real,
)
    T = promote_type(Float64, eltype(weights), typeof(float(capacity)))
    return KnapsackInstance(Vector{T}(weights), T(capacity))
end

"""
    KnapsackProjectionCache

Persistent JuMP/Gurobi state for repeated regularized projections onto a
fractional knapsack feasible set.
"""
mutable struct KnapsackProjectionCache
    jump_model::Model
    w::Vector{VariableRef}
    lambda::Float64
    weights::Vector{Float64}
    capacity::Float64
    last_solution::Vector{Float64}
end

function full_objective(theta::AbstractVector, instance::KnapsackInstance)
    length(theta) == length(instance.weights) ||
        throw(ArgumentError("theta and instance.weights must have the same length"))
    return Vector{Float64}(theta)
end

function _knapsack_sortperm(full_theta::AbstractVector, weights::AbstractVector, sense::Symbol)
    ratios = full_theta ./ weights
    if sense === :min
        return sortperm(ratios)
    elseif sense === :max
        return sortperm(ratios; rev=true)
    end
    throw(ArgumentError("sense must be :min or :max, got $sense"))
end

"""
    solve_fractional_knapsack(theta; instance, sense=:min)

Analytic greedy oracle for the fractional knapsack LP on the full objective
`theta`.

- `sense=:min` matches the RSPO+/paper cost-minimization convention.
- `sense=:max` reproduces the greedy direction used by the old synthetic code.
"""
function solve_fractional_knapsack(
    theta::AbstractVector{<:Real};
    instance::KnapsackInstance,
    sense::Symbol=:min,
)
    full_theta = full_objective(theta, instance)
    order = _knapsack_sortperm(full_theta, instance.weights, sense)

    decision = zeros(Float64, length(full_theta))
    remaining_capacity = instance.capacity

    for idx in order
        remaining_capacity <= 0 && break
        if sense === :min && full_theta[idx] >= 0
            break
        elseif sense === :max && full_theta[idx] <= 0
            break
        end

        weight = instance.weights[idx]
        if weight <= remaining_capacity
            decision[idx] = 1.0
            remaining_capacity -= weight
        else
            decision[idx] = remaining_capacity / weight
            remaining_capacity = 0.0
        end
    end

    objective_value = dot(full_theta, decision)
    return decision, objective_value
end

function linear_knapsack_decision(
    theta::AbstractVector{<:Real};
    instance::KnapsackInstance,
    direction::Symbol=:min,
)
    decision, _ = solve_fractional_knapsack(theta; instance=instance, sense=direction)
    return decision
end

function linear_knapsack_decision(
    theta::AbstractMatrix{<:Real};
    instance::KnapsackInstance,
    direction::Symbol=:min,
)
    decisions = Matrix{Float64}(undef, size(theta, 1), size(theta, 2))
    for col_idx in axes(theta, 2)
        decisions[:, col_idx] = linear_knapsack_decision(
            view(theta, :, col_idx);
            instance=instance,
            direction=direction,
        )
    end
    return decisions
end

"""
    linear_maximizer(direction; instance)

InferOpt-compatible linear maximizer wrapper for the fractional knapsack oracle.
For `direction=:min`, the sign flip is absorbed into the `LinearMaximizer`, so
losses like `InferOpt.SPOPlusLoss` can train directly on the cost-minimization
convention used by this codebase.
"""
function _knapsack_linear_maximizer(direction::Symbol; instance::KnapsackInstance)
    if direction === :min
        return InferOpt.LinearMaximizer(
            θ -> linear_knapsack_decision(θ; instance=instance, direction=:min);
            g=y -> -y,
        )
    elseif direction === :max
        return InferOpt.LinearMaximizer(
            θ -> linear_knapsack_decision(θ; instance=instance, direction=:max),
        )
    end

    throw(ArgumentError("direction must be :min or :max, got $direction"))
end

function linear_maximizer(direction::Symbol; instance)
    if instance isa KnapsackInstance
        return _knapsack_linear_maximizer(direction; instance=instance)
    elseif @isdefined(BatteryDispatchInstance) && instance isa BatteryDispatchInstance
        return _battery_linear_maximizer(direction; instance=instance)
    end

    throw(ArgumentError("unsupported instance type $(typeof(instance))"))
end

function _build_knapsack_jump_model(
    instance::KnapsackInstance,
    lambda::Real,
    theta::AbstractVector{<:Real},
)
    dim = length(instance.weights)
    length(theta) == dim ||
        throw(ArgumentError("theta length $(length(theta)) does not match instance dimension $dim"))

    model = Model(() -> Gurobi.Optimizer(get_knapsack_gurobi_env()))
    set_silent(model)

    @variable(model, 0 <= w[1:dim] <= 1)
    @constraint(model, sum(instance.weights[i] * w[i] for i in 1:dim) <= instance.capacity)
    objective =
        sum(Float64(theta[i]) * w[i] for i in 1:dim) +
        (float(lambda) / 2) * sum(w[i]^2 for i in 1:dim)
    set_objective_sense(model, KNAPSACK_MOI.MIN_SENSE)
    set_objective_function(model, objective)

    return model, w
end

function init_knapsack_projection_cache(instance::KnapsackInstance; lambda::Real)
    lambda > 0 || throw(ArgumentError("lambda must be positive, got $lambda"))

    dim = length(instance.weights)
    model, w = _build_knapsack_jump_model(instance, lambda, zeros(Float64, dim))

    return KnapsackProjectionCache(
        model,
        w,
        float(lambda),
        copy(instance.weights),
        float(instance.capacity),
        zeros(Float64, dim),
    )
end

function _knapsack_cache_matches_instance(
    cache::KnapsackProjectionCache,
    instance::KnapsackInstance,
)
    return cache.weights == instance.weights && cache.capacity == float(instance.capacity)
end

function _rebuild_knapsack_projection_cache!(
    cache::KnapsackProjectionCache,
    instance::KnapsackInstance;
    lambda::Real,
)
    dim = length(instance.weights)
    model, w = _build_knapsack_jump_model(instance, lambda, zeros(Float64, dim))

    cache.jump_model = model
    cache.w = w
    cache.lambda = float(lambda)
    cache.weights = copy(instance.weights)
    cache.capacity = float(instance.capacity)
    cache.last_solution = zeros(Float64, dim)
    return cache
end

function reset_knapsack_projection_cache!(cache::KnapsackProjectionCache)
    fill!(cache.last_solution, 0.0)
    return cache
end

function _batched_projection_caches(
    cache::Nothing,
    instance::KnapsackInstance,
    lambda::Real,
    n_cols::Int,
)
    return fill(nothing, n_cols)
end

function _batched_projection_caches(
    cache::KnapsackProjectionCache,
    instance::KnapsackInstance,
    lambda::Real,
    n_cols::Int,
)
    return fill(cache, n_cols)
end

function _batched_projection_caches(
    caches::AbstractVector{<:Union{Nothing,KnapsackProjectionCache}},
    instance::KnapsackInstance,
    lambda::Real,
    n_cols::Int,
)
    length(caches) == n_cols ||
        throw(ArgumentError("cache vector length $(length(caches)) does not match $n_cols columns"))
    return caches
end

function _set_knapsack_projection_objective!(
    cache::KnapsackProjectionCache,
    theta::AbstractVector{<:Real},
    lambda::Real,
)
    dim = length(cache.weights)
    length(theta) == dim ||
        throw(ArgumentError("theta length $(length(theta)) does not match cache dimension $dim"))

    objective =
        sum(Float64(theta[i]) * cache.w[i] for i in 1:dim) +
        (float(lambda) / 2) * sum(cache.w[i]^2 for i in 1:dim)
    set_objective_sense(cache.jump_model, KNAPSACK_MOI.MIN_SENSE)
    set_objective_function(cache.jump_model, objective)
    cache.lambda = float(lambda)
    return cache
end

"""
    solve_fractional_knapsack_projection(theta; instance, cache=nothing, lambda, use_warm_start=true)

Solve the Gurobi-backed regularized projection problem and return a named tuple
with the decision, objective value, exit flag, solver info, and the cache used.
"""
function solve_fractional_knapsack_projection(
    theta::AbstractVector{<:Real};
    instance::KnapsackInstance,
    cache::Union{Nothing,KnapsackProjectionCache}=nothing,
    lambda::Real,
    use_warm_start::Bool=true,
)
    lambda > 0 || throw(ArgumentError("lambda must be positive, got $lambda"))

    local_cache =
        isnothing(cache) ? init_knapsack_projection_cache(instance; lambda=lambda) : cache
    if !_knapsack_cache_matches_instance(local_cache, instance)
        _rebuild_knapsack_projection_cache!(local_cache, instance; lambda=lambda)
    end

    objective = full_objective(theta, instance)
    _set_knapsack_projection_objective!(local_cache, objective, lambda)

    for idx in eachindex(local_cache.w)
        start_value = use_warm_start ? local_cache.last_solution[idx] : 0.0
        set_start_value(local_cache.w[idx], start_value)
    end

    optimize!(local_cache.jump_model)

    status = termination_status(local_cache.jump_model)
    if status == OPTIMAL || status == LOCALLY_SOLVED || status == ALMOST_OPTIMAL
        solution = value.(local_cache.w)
        local_cache.last_solution .= solution
        return (
            decision=Vector{Float64}(solution),
            objective_value=objective_value(local_cache.jump_model),
            exitflag=1,
            info=(status=status,),
            cache=local_cache,
        )
    end

    error("Gurobi knapsack projection solve failed with status $status")
end

"""
    projection_optimizer(theta; instance, cache=nothing, lambda)

Solve the regularized fractional knapsack problem

`min_w theta' * w + (lambda / 2) * ||w||^2`

subject to the client's knapsack constraints. When a cache is provided, the
underlying JuMP/Gurobi model is reused and warm-started from the previous
solution.
"""
function _knapsack_projection_optimizer(
    theta::AbstractVector{<:Real};
    instance::KnapsackInstance,
    cache::Union{Nothing,KnapsackProjectionCache}=nothing,
    lambda::Real,
    use_warm_start::Bool=true,
)
    result = solve_fractional_knapsack_projection(
        theta;
        instance=instance,
        cache=cache,
        lambda=lambda,
        use_warm_start=use_warm_start,
    )
    return result.decision
end

function _knapsack_projection_optimizer(
    theta::AbstractMatrix{<:Real};
    instance::KnapsackInstance,
    cache::Union{
        Nothing,
        KnapsackProjectionCache,
        AbstractVector{<:Union{Nothing,KnapsackProjectionCache}},
    }=nothing,
    lambda::Real,
    use_warm_start::Bool=true,
)
    n_cols = size(theta, 2)
    decisions = Matrix{Float64}(undef, size(theta, 1), n_cols)
    caches = _batched_projection_caches(cache, instance, lambda, n_cols)

    for col_idx in 1:n_cols
        decisions[:, col_idx] = projection_optimizer(
            view(theta, :, col_idx);
            instance=instance,
            cache=caches[col_idx],
            lambda=lambda,
            use_warm_start=use_warm_start,
        )
    end

    return decisions
end

function projection_optimizer(
    theta::AbstractVector{<:Real};
    instance,
    cache=nothing,
    lambda::Real,
    use_warm_start::Bool=true,
)
    if instance isa KnapsackInstance
        return _knapsack_projection_optimizer(
            theta;
            instance=instance,
            cache=cache,
            lambda=lambda,
            use_warm_start=use_warm_start,
        )
    elseif @isdefined(BatteryDispatchInstance) && instance isa BatteryDispatchInstance
        return _battery_projection_optimizer(
            theta;
            instance=instance,
            cache=cache,
            lambda=lambda,
            use_warm_start=use_warm_start,
        )
    end

    throw(ArgumentError("unsupported instance type $(typeof(instance))"))
end

function projection_optimizer(
    theta::AbstractMatrix{<:Real};
    instance,
    cache=nothing,
    lambda::Real,
    use_warm_start::Bool=true,
)
    if instance isa KnapsackInstance
        return _knapsack_projection_optimizer(
            theta;
            instance=instance,
            cache=cache,
            lambda=lambda,
            use_warm_start=use_warm_start,
        )
    elseif @isdefined(BatteryDispatchInstance) && instance isa BatteryDispatchInstance
        return _battery_projection_optimizer(
            theta;
            instance=instance,
            cache=cache,
            lambda=lambda,
            use_warm_start=use_warm_start,
        )
    end

    throw(ArgumentError("unsupported instance type $(typeof(instance))"))
end
