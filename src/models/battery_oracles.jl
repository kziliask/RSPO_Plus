using JuMP
using Gurobi
import InferOpt

const BATTERY_MOI = JuMP.MOI

if !@isdefined(get_battery_gurobi_env)
    const BATTERY_GUROBI_ENV = Ref{Gurobi.Env}()

    function get_battery_gurobi_env()
        if !isassigned(BATTERY_GUROBI_ENV)
            BATTERY_GUROBI_ENV[] = Gurobi.Env()
        end
        return BATTERY_GUROBI_ENV[]
    end
end

"""
    BatteryDispatchInstance(capacity; horizon=24)

Exact-fill battery-dispatch instance with one decision variable per hour. The
feasible set is:

`sum(w) = capacity`, `0 <= w_h <= 1`.
"""
struct BatteryDispatchInstance{T<:AbstractFloat}
    capacity::T
    horizon::Int
end

function BatteryDispatchInstance(capacity::Real; horizon::Integer=24)
    horizon > 0 || throw(ArgumentError("horizon must be positive, got $horizon"))
    capacity_float = float(capacity)
    0.0 <= capacity_float <= horizon || throw(
        ArgumentError("capacity must lie in [0, horizon], got capacity=$capacity_float and horizon=$horizon"),
    )
    return BatteryDispatchInstance(capacity_float, Int(horizon))
end

mutable struct BatteryDispatchProjectionCache
    jump_model::Model
    w::Vector{VariableRef}
    lambda::Float64
    capacity::Float64
    horizon::Int
    last_solution::Vector{Float64}
end

function _battery_sortperm(theta::AbstractVector{<:Real}, sense::Symbol)
    if sense === :min
        return sortperm(theta)
    elseif sense === :max
        return sortperm(theta; rev=true)
    end
    throw(ArgumentError("sense must be :min or :max, got $sense"))
end

"""
    solve_battery_dispatch(theta; instance, sense=:min)

Analytic exact-fill battery oracle. It ranks hours by price and fills the
cheapest slots first for minimization (or the most expensive for maximization),
using a fractional final hour when needed.
"""
function solve_battery_dispatch(
    theta::AbstractVector{<:Real};
    instance::BatteryDispatchInstance,
    sense::Symbol=:min,
)
    length(theta) == instance.horizon || throw(
        ArgumentError("theta length $(length(theta)) does not match horizon $(instance.horizon)"),
    )

    order = _battery_sortperm(theta, sense)
    decision = zeros(Float64, instance.horizon)
    remaining_capacity = instance.capacity

    for idx in order
        remaining_capacity <= 0 && break
        take = min(1.0, remaining_capacity)
        decision[idx] = take
        remaining_capacity -= take
    end

    return decision, dot(theta, decision)
end

function battery_dispatch_decision(
    theta::AbstractVector{<:Real};
    instance::BatteryDispatchInstance,
    direction::Symbol=:min,
)
    decision, _ = solve_battery_dispatch(theta; instance=instance, sense=direction)
    return decision
end

function battery_dispatch_decision(
    theta::AbstractMatrix{<:Real};
    instance::BatteryDispatchInstance,
    direction::Symbol=:min,
)
    decisions = Matrix{Float64}(undef, size(theta, 1), size(theta, 2))
    for col_idx in axes(theta, 2)
        decisions[:, col_idx] = battery_dispatch_decision(
            view(theta, :, col_idx);
            instance=instance,
            direction=direction,
        )
    end
    return decisions
end

function _battery_linear_maximizer(direction::Symbol; instance::BatteryDispatchInstance)
    if direction === :min
        return InferOpt.LinearMaximizer(
            θ -> battery_dispatch_decision(θ; instance=instance, direction=:min);
            g=y -> -y,
        )
    elseif direction === :max
        return InferOpt.LinearMaximizer(
            θ -> battery_dispatch_decision(θ; instance=instance, direction=:max),
        )
    end

    throw(ArgumentError("direction must be :min or :max, got $direction"))
end

function _build_battery_jump_model(
    instance::BatteryDispatchInstance,
    lambda::Real,
    theta::AbstractVector{<:Real},
)
    length(theta) == instance.horizon || throw(
        ArgumentError("theta length $(length(theta)) does not match horizon $(instance.horizon)"),
    )

    model = Model(() -> Gurobi.Optimizer(get_battery_gurobi_env()))
    set_silent(model)

    @variable(model, 0 <= w[1:(instance.horizon)] <= 1)
    @constraint(model, sum(w) == instance.capacity)
    objective =
        sum(Float64(theta[idx]) * w[idx] for idx in 1:(instance.horizon)) +
        (float(lambda) / 2) * sum(w[idx]^2 for idx in 1:(instance.horizon))
    set_objective_sense(model, BATTERY_MOI.MIN_SENSE)
    set_objective_function(model, objective)

    return model, w
end

function init_battery_projection_cache(instance::BatteryDispatchInstance; lambda::Real)
    lambda > 0 || throw(ArgumentError("lambda must be positive, got $lambda"))
    model, w = _build_battery_jump_model(instance, lambda, zeros(Float64, instance.horizon))
    return BatteryDispatchProjectionCache(
        model,
        w,
        float(lambda),
        instance.capacity,
        instance.horizon,
        zeros(Float64, instance.horizon),
    )
end

function _battery_cache_matches_instance(
    cache::BatteryDispatchProjectionCache,
    instance::BatteryDispatchInstance,
)
    return cache.capacity == instance.capacity && cache.horizon == instance.horizon
end

function _rebuild_battery_projection_cache!(
    cache::BatteryDispatchProjectionCache,
    instance::BatteryDispatchInstance;
    lambda::Real,
)
    model, w = _build_battery_jump_model(instance, lambda, zeros(Float64, instance.horizon))
    cache.jump_model = model
    cache.w = w
    cache.lambda = float(lambda)
    cache.capacity = instance.capacity
    cache.horizon = instance.horizon
    cache.last_solution = zeros(Float64, instance.horizon)
    return cache
end

function reset_battery_projection_cache!(cache::BatteryDispatchProjectionCache)
    fill!(cache.last_solution, 0.0)
    return cache
end

function _battery_batched_projection_caches(
    cache::Nothing,
    instance::BatteryDispatchInstance,
    lambda::Real,
    n_cols::Int,
)
    return fill(nothing, n_cols)
end

function _battery_batched_projection_caches(
    cache::BatteryDispatchProjectionCache,
    instance::BatteryDispatchInstance,
    lambda::Real,
    n_cols::Int,
)
    return fill(cache, n_cols)
end

function _battery_batched_projection_caches(
    caches::AbstractVector{<:Union{Nothing,BatteryDispatchProjectionCache}},
    instance::BatteryDispatchInstance,
    lambda::Real,
    n_cols::Int,
)
    length(caches) == n_cols || throw(
        ArgumentError("cache vector length $(length(caches)) does not match $n_cols columns"),
    )
    return caches
end

function _set_battery_projection_objective!(
    cache::BatteryDispatchProjectionCache,
    theta::AbstractVector{<:Real},
    lambda::Real,
)
    length(theta) == cache.horizon || throw(
        ArgumentError("theta length $(length(theta)) does not match cache horizon $(cache.horizon)"),
    )

    objective =
        sum(Float64(theta[idx]) * cache.w[idx] for idx in 1:(cache.horizon)) +
        (float(lambda) / 2) * sum(cache.w[idx]^2 for idx in 1:(cache.horizon))
    set_objective_sense(cache.jump_model, BATTERY_MOI.MIN_SENSE)
    set_objective_function(cache.jump_model, objective)
    cache.lambda = float(lambda)
    return cache
end

function solve_battery_dispatch_projection(
    theta::AbstractVector{<:Real};
    instance::BatteryDispatchInstance,
    cache::Union{Nothing,BatteryDispatchProjectionCache}=nothing,
    lambda::Real,
    use_warm_start::Bool=true,
)
    lambda > 0 || throw(ArgumentError("lambda must be positive, got $lambda"))

    local_cache =
        isnothing(cache) ? init_battery_projection_cache(instance; lambda=lambda) : cache
    if !_battery_cache_matches_instance(local_cache, instance)
        _rebuild_battery_projection_cache!(local_cache, instance; lambda=lambda)
    end

    _set_battery_projection_objective!(local_cache, theta, lambda)

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

    error("Gurobi battery projection solve failed with status $status")
end

function _battery_projection_optimizer(
    theta::AbstractVector{<:Real};
    instance::BatteryDispatchInstance,
    cache::Union{Nothing,BatteryDispatchProjectionCache}=nothing,
    lambda::Real,
    use_warm_start::Bool=true,
)
    result = solve_battery_dispatch_projection(
        theta;
        instance=instance,
        cache=cache,
        lambda=lambda,
        use_warm_start=use_warm_start,
    )
    return result.decision
end

function _battery_projection_optimizer(
    theta::AbstractMatrix{<:Real};
    instance::BatteryDispatchInstance,
    cache::Union{
        Nothing,
        BatteryDispatchProjectionCache,
        AbstractVector{<:Union{Nothing,BatteryDispatchProjectionCache}},
    }=nothing,
    lambda::Real,
    use_warm_start::Bool=true,
)
    n_cols = size(theta, 2)
    decisions = Matrix{Float64}(undef, size(theta, 1), n_cols)
    caches = _battery_batched_projection_caches(cache, instance, lambda, n_cols)

    for col_idx in 1:n_cols
        decisions[:, col_idx] = _battery_projection_optimizer(
            view(theta, :, col_idx);
            instance=instance,
            cache=caches[col_idx],
            lambda=lambda,
            use_warm_start=use_warm_start,
        )
    end

    return decisions
end
