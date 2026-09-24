using Flux
import InferOpt
using LinearAlgebra
using Optimisers
using Random
using Statistics

Base.@kwdef struct SyntheticKnapsackTrainingConfig
    seed::Int = 42
    hidden_dim::Int = 32
    rounds::Int = 50
    local_epochs::Int = 1
    batch_size::Int = 32
    client_fraction::Float64 = 0.2
    validation_client_fraction::Float64 = 0.25
    validation_max_samples_per_client::Int = 64
    lambda0::Float64 = 2.0
    kappa_lambda::Float64 = 0.05
    per_client_lambda::Bool = false
    lr0::Float64 = 1e-3
    kappa_lr::Float64 = 0.05
    per_client_lr::Bool = false
    lr_lambda_alpha::Float64 = 0.1
    clip_norm::Float64 = 1.0
    freeze_tau::Float64 = 0.05
    freeze_eps::Float64 = 1e-8
    stop_after_freeze_rounds::Int = 3
    shuffle_batches::Bool = true
    use_warm_start::Bool = true
end

struct SyntheticKnapsackClientData
    client_id::Int
    instance::KnapsackInstance
    normalization::FeatureNormalizationStats{Float64,Vector{Float64}}
    train_x::Matrix{Float64}
    train_theta_true::Matrix{Float64}
    train_y_true::Matrix{Float64}
    val_x::Matrix{Float64}
    val_theta_true::Matrix{Float64}
    val_y_true::Matrix{Float64}
    test_x::Matrix{Float64}
    test_theta_true::Matrix{Float64}
    test_y_true::Matrix{Float64}
end

struct ValidationClientData
    client_id::Int
    instance::KnapsackInstance
    x::Matrix{Float64}
    theta_true::Matrix{Float64}
    z_star::Vector{Float64}
end

struct ValidationMonitor
    clients::Vector{ValidationClientData}
end

struct SchedulerTrace
    lambda_values::Vector{Float64}
    lr_values::Vector{Float64}
    bound_values::Vector{Float64}
    frozen_after_round::Union{Nothing,Int}
    per_client_lambda0::Union{Nothing,Vector{Float64}}
end

SchedulerTrace(lv, lr, bv, far) = SchedulerTrace(lv, lr, bv, far, nothing)

abstract type SyntheticKnapsackObjective end

struct RSPOPlusObjective <: SyntheticKnapsackObjective end

struct InferOptObjective{B,T} <: SyntheticKnapsackObjective
    name::Symbol
    loss_builder::B
    target_getter::T
end

struct PerturbedMSEObjective{B} <: SyntheticKnapsackObjective
    name::Symbol
    layer_builder::B
end

struct MSEObjective <: SyntheticKnapsackObjective end

function SPOPlusObjective(; α::Real=2.0)
    alpha = float(α)
    return InferOptObjective(
        :spo_plus,
        client -> InferOpt.SPOPlusLoss(linear_maximizer(:min; instance=client.instance); α=alpha),
        (client, batch_indices) -> view(client.train_theta_true, :, batch_indices),
    )
end

objective_name(::RSPOPlusObjective) = :rspo_plus
objective_name(objective::InferOptObjective) = objective.name
objective_name(objective::PerturbedMSEObjective) = objective.name
objective_name(::MSEObjective) = :mse

objective_uses_lambda(::SyntheticKnapsackObjective) = false
objective_uses_lambda(::RSPOPlusObjective) = true

objective_tracks_bound(::SyntheticKnapsackObjective) = false
objective_tracks_bound(::RSPOPlusObjective) = true

function objective_method(objective::SyntheticKnapsackObjective, phase::Symbol)
    phase in (:fed, :local, :centralized) ||
        throw(ArgumentError("phase must be :fed, :local, or :centralized, got $phase"))
    return Symbol(phase, "_", objective_name(objective))
end

struct FedTrainingResult
    method::Symbol
    model
    client_data::Vector{SyntheticKnapsackClientData}
    round_losses::Vector{Float64}
    selected_clients::Vector{Vector{Int}}
    trace::SchedulerTrace
end

function FedTrainingResult(
    model,
    client_data::Vector{SyntheticKnapsackClientData},
    round_losses::Vector{Float64},
    selected_clients::Vector{Vector{Int}},
    trace::SchedulerTrace,
)
    return FedTrainingResult(:fed_rspo_plus, model, client_data, round_losses, selected_clients, trace)
end

const FedRSPOPlusResult = FedTrainingResult

struct LocalClientTrainingResult
    client_id::Int
    model
    round_losses::Vector{Float64}
    trace::SchedulerTrace
end

const LocalClientRSPOPlusResult = LocalClientTrainingResult

struct LocalTrainingResult
    method::Symbol
    clients::Vector{LocalClientTrainingResult}
end

struct SyntheticKnapsackTwoStageTrainingResult
    warm_start::FedTrainingResult
    personalization::LocalTrainingResult
end

function LocalTrainingResult(clients::Vector{LocalClientTrainingResult})
    return LocalTrainingResult(:local_rspo_plus, clients)
end

const LocalRSPOPlusResult = LocalTrainingResult

function _validate_training_config(
    config::SyntheticKnapsackTrainingConfig,
    objective::SyntheticKnapsackObjective=RSPOPlusObjective(),
)
    config.hidden_dim > 0 || throw(ArgumentError("hidden_dim must be positive"))
    config.rounds > 0 || throw(ArgumentError("rounds must be positive"))
    config.local_epochs > 0 || throw(ArgumentError("local_epochs must be positive"))
    config.batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    config.client_fraction > 0 || throw(ArgumentError("client_fraction must be positive"))
    config.validation_client_fraction > 0 ||
        throw(ArgumentError("validation_client_fraction must be positive"))
    if objective_uses_lambda(objective)
        config.per_client_lambda || config.lambda0 > 0 ||
            throw(ArgumentError("lambda0 must be positive (or set per_client_lambda=true)"))
        config.kappa_lambda >= 0 || throw(ArgumentError("kappa_lambda must be nonnegative"))
    end
    config.lr0 > 0 || throw(ArgumentError("lr0 must be positive"))
    config.kappa_lr >= 0 || throw(ArgumentError("kappa_lr must be nonnegative"))
    config.lr_lambda_alpha >= 0 || throw(ArgumentError("lr_lambda_alpha must be nonnegative"))
    config.clip_norm > 0 || throw(ArgumentError("clip_norm must be positive"))
    config.freeze_eps > 0 || throw(ArgumentError("freeze_eps must be positive"))
    config.stop_after_freeze_rounds >= 0 ||
        throw(ArgumentError("stop_after_freeze_rounds must be nonnegative"))
    return config
end

function build_synthetic_knapsack_model(
    input_dim::Int,
    output_dim::Int;
    hidden_dim::Int=32,
    rng::AbstractRNG=Random.default_rng(),
)
    input_dim > 0 || throw(ArgumentError("input_dim must be positive"))
    output_dim > 0 || throw(ArgumentError("output_dim must be positive"))
    hidden_dim > 0 || throw(ArgumentError("hidden_dim must be positive"))

    init = Flux.glorot_uniform(rng)
    model = Flux.Chain(
        Flux.Dense(input_dim => hidden_dim, Flux.relu; init=init),
        Flux.Dense(hidden_dim => output_dim; init=init),
    )
    return Flux.f64(model)
end

function _optimizer_rule(config::SyntheticKnapsackTrainingConfig, lr::Real)
    return Optimisers.OptimiserChain(Optimisers.ClipNorm(config.clip_norm), Optimisers.Adam(lr))
end

function _subset_count(value::Real, total::Int)
    total > 0 || return 0
    if value <= 1
        return clamp(ceil(Int, value * total), 1, total)
    end
    return clamp(round(Int, value), 1, total)
end

function _sample_subset(
    rng::AbstractRNG,
    candidates::AbstractVector{Int},
    subset_spec::Real,
)
    isempty(candidates) && return Int[]
    n_select = _subset_count(subset_spec, length(candidates))
    order = randperm(rng, length(candidates))[1:n_select]
    return sort(candidates[order])
end

function _batch_index_sets(
    rng::AbstractRNG,
    n_samples::Int,
    batch_size::Int;
    shuffle::Bool=true,
)
    n_samples == 0 && return Vector{Vector{Int}}()

    order = shuffle ? randperm(rng, n_samples) : collect(1:n_samples)
    batches = Vector{Vector{Int}}()
    for start in 1:batch_size:n_samples
        stop = min(start + batch_size - 1, n_samples)
        push!(batches, order[start:stop])
    end
    return batches
end

function _concat_float_matrices(
    matrices::AbstractVector{<:AbstractMatrix{<:Real}},
    n_rows::Int,
)
    nonempty = Matrix{Float64}[Matrix{Float64}(matrix) for matrix in matrices if size(matrix, 2) > 0]
    isempty(nonempty) && return zeros(Float64, n_rows, 0)
    return reduce(hcat, nonempty)
end

function _synthetic_decision_labels(
    theta_true::AbstractMatrix{<:Real},
    instance::KnapsackInstance,
)
    y_true = zeros(Float64, length(instance.weights), size(theta_true, 2))
    for col_idx in axes(theta_true, 2)
        y_true[:, col_idx], _ = solve_fractional_knapsack(
            view(theta_true, :, col_idx);
            instance=instance,
            sense=:min,
        )
    end
    return y_true
end

function prepare_synthetic_knapsack_client_data(
    dataset::SyntheticKnapsackDataset;
    normalize_x::Bool=true,
)
    _validate_config(dataset.config)
    n_clients = dataset.config.n_clients
    train_stats =
        normalize_x ? fit_all_client_normalization_stats(dataset.train, n_clients) :
        fill(FeatureNormalizationStats(Float64[], Float64[]), n_clients)

    client_data = Vector{SyntheticKnapsackClientData}(undef, n_clients)
    for client_id in 1:n_clients
        normalization = train_stats[client_id]
        weights, capacity, train_x, train_theta_true = get_client_split_data(
            dataset.train,
            dataset.clients,
            client_id;
            normalize_x=normalize_x,
            normalization_stats=normalize_x ? normalization : nothing,
        )
        _, _, val_x, val_theta_true = get_client_split_data(
            dataset.val,
            dataset.clients,
            client_id;
            normalize_x=normalize_x,
            normalization_stats=normalize_x ? normalization : nothing,
        )
        _, _, test_x, test_theta_true = get_client_split_data(
            dataset.test,
            dataset.clients,
            client_id;
            normalize_x=normalize_x,
            normalization_stats=normalize_x ? normalization : nothing,
        )
        instance = KnapsackInstance(weights, capacity)
        train_y_true = _synthetic_decision_labels(train_theta_true, instance)
        val_y_true = _synthetic_decision_labels(val_theta_true, instance)
        test_y_true = _synthetic_decision_labels(test_theta_true, instance)
        client_data[client_id] = SyntheticKnapsackClientData(
            client_id,
            instance,
            normalization,
            train_x,
            train_theta_true,
            train_y_true,
            val_x,
            val_theta_true,
            val_y_true,
            test_x,
            test_theta_true,
            test_y_true,
        )
    end

    return client_data
end

function prepare_centralized_synthetic_knapsack_client_data(
    dataset::SyntheticKnapsackDataset;
    normalize_x::Bool=true,
)
    client_data = prepare_synthetic_knapsack_client_data(dataset; normalize_x=normalize_x)
    pooled_train_x = _concat_float_matrices([client.train_x for client in client_data], dataset.config.p)
    pooled_train_theta_true = _concat_float_matrices(
        [client.train_theta_true for client in client_data],
        dataset.config.dim,
    )
    pooled_train_y_true = _concat_float_matrices(
        [client.train_y_true for client in client_data],
        dataset.config.dim,
    )

    return [
        SyntheticKnapsackClientData(
            client.client_id,
            client.instance,
            client.normalization,
            pooled_train_x,
            pooled_train_theta_true,
            pooled_train_y_true,
            client.val_x,
            client.val_theta_true,
            client.val_y_true,
            client.test_x,
            client.test_theta_true,
            client.test_y_true,
        ) for client in client_data
    ]
end

function _sample_validation_columns(
    rng::AbstractRNG,
    x::Matrix{Float64},
    theta_true::Matrix{Float64},
    max_samples::Int,
)
    n_samples = size(x, 2)
    if max_samples <= 0 || n_samples <= max_samples
        return x, theta_true
    end

    indices = sort(randperm(rng, n_samples)[1:max_samples])
    return x[:, indices], theta_true[:, indices]
end

function build_validation_monitor(
    client_data::Vector{SyntheticKnapsackClientData},
    config::SyntheticKnapsackTrainingConfig;
    rng::AbstractRNG=MersenneTwister(config.seed),
    client_ids::Union{Nothing,AbstractVector{Int}}=nothing,
)
    if isnothing(client_ids)
        available_client_ids = [
            client.client_id for client in client_data if size(client.val_x, 2) > 0
        ]
        selected_client_ids =
            _sample_subset(rng, available_client_ids, config.validation_client_fraction)
    else
        selected_client_ids = collect(client_ids)
    end

    monitor_clients = ValidationClientData[]
    for client_id in selected_client_ids
        client = client_data[client_id]
        size(client.val_x, 2) == 0 && continue

        x, theta_true = _sample_validation_columns(
            rng,
            client.val_x,
            client.val_theta_true,
            config.validation_max_samples_per_client,
        )
        z_star = Float64[
            solve_fractional_knapsack(view(theta_true, :, col); instance=client.instance)[2] for
            col in axes(theta_true, 2)
        ]
        push!(
            monitor_clients,
            ValidationClientData(client_id, client.instance, x, theta_true, z_star),
        )
    end

    return ValidationMonitor(monitor_clients)
end

function pointwise_rspo_losses(
    theta_pred::AbstractMatrix{<:Real},
    theta_true::AbstractMatrix{<:Real};
    instance::KnapsackInstance,
    cache::Union{Nothing,KnapsackProjectionCache}=nothing,
    lambda::Real,
    use_warm_start::Bool=true,
    v_opt::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
)
    p_u = (theta_true .- 2 .* theta_pred) ./ lambda
    u = 2 .* theta_pred .- theta_true
    opt_u = projection_optimizer(
        u;
        instance=instance,
        cache=cache,
        lambda=lambda,
        use_warm_start=use_warm_start,
    )
    opt_v =
        isnothing(v_opt) ? projection_optimizer(
            theta_true;
            instance=instance,
            cache=cache,
            lambda=lambda,
            use_warm_start=use_warm_start,
        ) : Matrix{Float64}(v_opt)

    return vec(
        (lambda / 2) .* (
            sum(abs2, p_u .- opt_v; dims=1) .- sum(abs2, p_u .- opt_u; dims=1)
        ),
    )
end

"""
    compute_freeze_bound(model, monitor; lambda, lambdas, freeze_eps, use_warm_start)

Compute the global regularization-bias ratio across all validation samples:

    mean_i[c_i^T w_{it}^{*,reg} - z_i^*] / (mean_i[|z_i^*|] + ε)

where `c_i` is the ground-truth cost, `w_{it}^{*,reg}` is the optimal regularized
decision at the current lambda, and `z_i^*` is the unregularized optimal objective.
The mean is taken over the union of all validation samples in `monitor`, so this is
a single ratio of averages rather than an average of per-sample or per-client ratios.

This metric is non-negative and goes to 0 as λ → 0 (regularization vanishes).
"""
function compute_freeze_bound(
    model,
    monitor::ValidationMonitor;
    lambda::Union{Nothing,Real}=nothing,
    lambdas::Union{Nothing,AbstractVector{<:Real}}=nothing,
    freeze_eps::Real=1e-8,
    use_warm_start::Bool=true,
)
    isempty(monitor.clients) && return Inf

    numerator_sum = 0.0
    denominator_sum = 0.0
    n_validation_samples = 0

    for client in monitor.clients
        lam = _resolve_knapsack_bound_lambda(
            client.client_id;
            lambda=lambda,
            lambdas=lambdas,
        )
        cache = init_knapsack_projection_cache(client.instance; lambda=lam)
        w_star_reg = projection_optimizer(
            client.theta_true;
            instance=client.instance,
            cache=cache,
            lambda=lam,
            use_warm_start=use_warm_start,
        )

        reg_objectives = vec(sum(client.theta_true .* w_star_reg; dims=1))
        numerator_sum += sum(reg_objectives) - sum(client.z_star)
        denominator_sum += sum(abs, client.z_star)
        n_validation_samples += length(client.z_star)
    end

    n_validation_samples == 0 && return Inf
    mean_bias = numerator_sum / n_validation_samples
    mean_abs_z_star = denominator_sum / n_validation_samples
    return mean_bias / (mean_abs_z_star + freeze_eps)
end

function _require_objective_lambda(lambda::Real)
    isfinite(lambda) && lambda > 0 ||
        throw(ArgumentError("lambda must be positive and finite, got $lambda"))
    return float(lambda)
end

function _resolve_knapsack_bound_lambda(
    client_id::Integer;
    lambda::Union{Nothing,Real}=nothing,
    lambdas::Union{Nothing,AbstractVector{<:Real}}=nothing,
)
    if !isnothing(lambda) && !isnothing(lambdas)
        throw(ArgumentError("pass either `lambda` or `lambdas`, not both"))
    elseif !isnothing(lambdas)
        return _require_objective_lambda(lambdas[client_id])
    elseif !isnothing(lambda)
        return _require_objective_lambda(lambda)
    end
    throw(ArgumentError("pass either `lambda` or `lambdas`"))
end

function compute_knapsack_client_c_norm(client::SyntheticKnapsackClientData)
    n = size(client.train_theta_true, 2)
    n == 0 && return NaN
    return mean(sqrt.(sum(abs2, client.train_theta_true; dims=1)))
end

function compute_knapsack_feasible_radius(instance::KnapsackInstance)
    order = sortperm(instance.weights)
    w = zeros(Float64, length(instance.weights))
    remaining = instance.capacity
    for idx in order
        remaining <= 0 && break
        if instance.weights[idx] <= remaining
            w[idx] = 1.0
            remaining -= instance.weights[idx]
        else
            w[idx] = remaining / instance.weights[idx]
            remaining = 0.0
        end
    end
    return norm(w)
end

function compute_knapsack_per_client_lambda0(
    client_data::AbstractVector{SyntheticKnapsackClientData};
    fallback::Float64=1.0,
)
    return [let
        c_norm = compute_knapsack_client_c_norm(c)
        radius = compute_knapsack_feasible_radius(c.instance)
        (isnan(c_norm) || radius <= 0) ? fallback : max(c_norm / radius, 1e-6)
    end for c in client_data]
end

function _synthetic_initial_client_lambda0s(
    objective::SyntheticKnapsackObjective,
    client_data::AbstractVector{SyntheticKnapsackClientData},
    config::SyntheticKnapsackTrainingConfig,
)
    if config.per_client_lambda && objective_uses_lambda(objective)
        return compute_knapsack_per_client_lambda0(client_data; fallback=config.lambda0)
    end
    return fill(config.lambda0, length(client_data))
end

function _synthetic_initial_client_lr0s(
    objective::SyntheticKnapsackObjective,
    client_data::AbstractVector{SyntheticKnapsackClientData},
    config::SyntheticKnapsackTrainingConfig,
    client_lambda0s::AbstractVector{<:Real},
)
    length(client_lambda0s) == length(client_data) || throw(
        ArgumentError("client_lambda0s length must match client_data length"),
    )
    if config.per_client_lr && objective_uses_lambda(objective)
        return [config.lr_lambda_alpha * client_lambda0s[i] / 4 for i in eachindex(client_data)]
    end
    return fill(config.lr0, length(client_data))
end

_initial_client_objective_state(::SyntheticKnapsackObjective, client, config) = nothing

function _initial_client_objective_state(
    ::RSPOPlusObjective,
    client::SyntheticKnapsackClientData,
    config::SyntheticKnapsackTrainingConfig,
)
    return init_knapsack_projection_cache(client.instance; lambda=config.lambda0)
end

function _build_loss_layer(
    ::RSPOPlusObjective,
    client::SyntheticKnapsackClientData;
    lambda::Real,
    config::SyntheticKnapsackTrainingConfig,
)
    return RSPOPlusLoss(projection_optimizer; lambda=_require_objective_lambda(lambda))
end

function _build_loss_layer(
    objective::InferOptObjective,
    client::SyntheticKnapsackClientData;
    lambda::Real,
    config::SyntheticKnapsackTrainingConfig,
)
    return objective.loss_builder(client)
end

function _build_loss_layer(
    objective::PerturbedMSEObjective,
    client::SyntheticKnapsackClientData;
    lambda::Real,
    config::SyntheticKnapsackTrainingConfig,
)
    return objective.layer_builder(client)
end

function _build_loss_layer(
    ::MSEObjective,
    client::SyntheticKnapsackClientData;
    lambda::Real,
    config::SyntheticKnapsackTrainingConfig,
)
    return nothing
end

_prepare_client_training_state(::SyntheticKnapsackObjective, client, config, state, lambda) = state

function _prepare_client_training_state(
    ::RSPOPlusObjective,
    client::SyntheticKnapsackClientData,
    config::SyntheticKnapsackTrainingConfig,
    state,
    lambda::Real,
)
    resolved_lambda = _require_objective_lambda(lambda)
    local_cache =
        isnothing(state) ? init_knapsack_projection_cache(client.instance; lambda=resolved_lambda) :
        state
    v_opt_train = projection_optimizer(
        client.train_theta_true;
        instance=client.instance,
        cache=local_cache,
        lambda=resolved_lambda,
        use_warm_start=config.use_warm_start,
    )
    return (cache=local_cache, v_opt_train=v_opt_train)
end

_persist_client_training_state(::SyntheticKnapsackObjective, state) = state
_persist_client_training_state(::RSPOPlusObjective, state) = state.cache

function _batch_training_loss(
    ::RSPOPlusObjective,
    loss_layer,
    theta_pred,
    theta_batch,
    batch_indices,
    client::SyntheticKnapsackClientData,
    state,
    config::SyntheticKnapsackTrainingConfig;
    lambda::Real,
)
    resolved_lambda = _require_objective_lambda(lambda)
    return loss_layer(
        theta_pred,
        theta_batch;
        instance=client.instance,
        cache=state.cache,
        lambda=resolved_lambda,
        use_warm_start=config.use_warm_start,
        v_opt=view(state.v_opt_train, :, batch_indices),
    ) / length(batch_indices)
end

function _batch_training_loss(
    objective::InferOptObjective,
    loss_layer,
    theta_pred,
    theta_batch,
    batch_indices,
    client::SyntheticKnapsackClientData,
    state,
    config::SyntheticKnapsackTrainingConfig;
    lambda::Real,
)
    targets = objective.target_getter(client, batch_indices)
    total = zero(eltype(theta_pred))
    for (batch_pos, _) in enumerate(batch_indices)
        total += loss_layer(view(theta_pred, :, batch_pos), view(targets, :, batch_pos))
    end
    return total / length(batch_indices)
end

function _batch_training_loss(
    ::PerturbedMSEObjective,
    loss_layer,
    theta_pred,
    theta_batch,
    batch_indices,
    client::SyntheticKnapsackClientData,
    state,
    config::SyntheticKnapsackTrainingConfig;
    lambda::Real,
)
    total = zero(eltype(theta_pred))
    for (batch_pos, sample_idx) in enumerate(batch_indices)
        y_hat = loss_layer(view(theta_pred, :, batch_pos))
        total += sum(abs2, y_hat .- view(client.train_y_true, :, sample_idx))
    end
    return total / (length(batch_indices) * size(client.train_y_true, 1))
end

function _batch_training_loss(
    ::MSEObjective,
    loss_layer,
    theta_pred,
    theta_batch,
    batch_indices,
    client::SyntheticKnapsackClientData,
    state,
    config::SyntheticKnapsackTrainingConfig;
    lambda::Real,
)
    return sum(abs2, theta_pred .- theta_batch) / length(theta_pred)
end

function _compute_objective_bound(
    ::SyntheticKnapsackObjective,
    model,
    monitor::ValidationMonitor,
    config::SyntheticKnapsackTrainingConfig;
    kwargs...,
)
    return NaN
end

function _compute_objective_bound(
    ::RSPOPlusObjective,
    model,
    monitor::ValidationMonitor,
    config::SyntheticKnapsackTrainingConfig;
    lambda::Union{Nothing,Real}=nothing,
    lambdas::Union{Nothing,AbstractVector{<:Real}}=nothing,
)
    if !isnothing(lambda) && !isnothing(lambdas)
        throw(ArgumentError("pass either `lambda` or `lambdas`, not both"))
    elseif !isnothing(lambdas)
        return compute_freeze_bound(
            model,
            monitor;
            lambdas=lambdas,
            freeze_eps=config.freeze_eps,
            use_warm_start=config.use_warm_start,
        )
    elseif !isnothing(lambda)
        return compute_freeze_bound(
            model,
            monitor;
            lambda=_require_objective_lambda(lambda),
            freeze_eps=config.freeze_eps,
            use_warm_start=config.use_warm_start,
        )
    end
    throw(ArgumentError("pass either `lambda` or `lambdas`"))
end

function train_client_model!(
    model,
    opt_state,
    client::SyntheticKnapsackClientData,
    objective::SyntheticKnapsackObjective,
    config::SyntheticKnapsackTrainingConfig,
    rng::AbstractRNG;
    state=nothing,
    lambda::Real=NaN,
)
    n_samples = size(client.train_x, 2)
    n_samples == 0 && return 0.0, state

    loss_layer = _build_loss_layer(objective, client; lambda=lambda, config=config)
    objective_state = _prepare_client_training_state(objective, client, config, state, lambda)

    total_loss = 0.0
    total_samples = 0

    for _ in 1:config.local_epochs
        for batch_indices in _batch_index_sets(
            rng,
            n_samples,
            config.batch_size;
            shuffle=config.shuffle_batches,
        )
            x_batch = view(client.train_x, :, batch_indices)
            theta_batch = view(client.train_theta_true, :, batch_indices)

            batch_loss, grads = Flux.withgradient(model) do m
                theta_pred = m(x_batch)
                _batch_training_loss(
                    objective,
                    loss_layer,
                    theta_pred,
                    theta_batch,
                    batch_indices,
                    client,
                    objective_state,
                    config;
                    lambda=lambda,
                )
            end
            Flux.update!(opt_state, model, grads[1])

            total_loss += batch_loss * length(batch_indices)
            total_samples += length(batch_indices)
        end
    end

    return total_loss / total_samples, _persist_client_training_state(objective, objective_state)
end

function fed_synthetic_knapsack(
    dataset::SyntheticKnapsackDataset;
    objective::SyntheticKnapsackObjective=RSPOPlusObjective(),
    config::SyntheticKnapsackTrainingConfig=SyntheticKnapsackTrainingConfig(),
    model=nothing,
)
    _validate_training_config(config, objective)

    rng = MersenneTwister(config.seed)
    model_rng = MersenneTwister(config.seed)
    client_data = prepare_synthetic_knapsack_client_data(dataset)
    validation_monitor =
        objective_tracks_bound(objective) ? build_validation_monitor(client_data, config; rng=rng) :
        ValidationMonitor(ValidationClientData[])

    global_model =
        isnothing(model) ? build_synthetic_knapsack_model(
            dataset.config.p,
            dataset.config.dim;
            hidden_dim=config.hidden_dim,
            rng=model_rng,
        ) : Flux.f64(deepcopy(model))
    flat_global, rebuild = Flux.destructure(global_model)

    frozen = Ref(false)
    client_lambda0s = _synthetic_initial_client_lambda0s(objective, client_data, config)
    lambda_scheds = if objective_uses_lambda(objective)
        [create_inverse_time_scheduler(client_lambda0s[i], config.kappa_lambda; frozen=frozen)
         for i in eachindex(client_data)]
    else
        nothing
    end
    client_lr0s = _synthetic_initial_client_lr0s(objective, client_data, config, client_lambda0s)
    lr_scheds = [
        create_inverse_time_scheduler(client_lr0s[i], config.kappa_lr; frozen=frozen)
        for i in eachindex(client_data)
    ]

    round_losses = Float64[]
    selected_clients = Vector{Vector{Int}}()
    lambda_values = Float64[]
    lr_values = Float64[]
    bound_values = Float64[]
    objective_states = [
        _initial_client_objective_state(objective, client, config) for client in client_data
    ]
    frozen_after_round = nothing

    for round in 1:config.rounds
        lambdas = if objective_uses_lambda(objective)
            [next_schedule_value!(lambda_scheds[i]) for i in eachindex(lambda_scheds)]
        else
            fill(NaN, length(client_data))
        end
        lrs = [next_schedule_value!(lr_scheds[i]) for i in eachindex(lr_scheds)]
        push!(lambda_values, mean(lambdas))
        push!(lr_values, mean(lrs))

        candidate_ids = [client.client_id for client in client_data if size(client.train_x, 2) > 0]
        active_client_ids = _sample_subset(rng, candidate_ids, config.client_fraction)
        push!(selected_clients, active_client_ids)

        client_deltas = Vector{Vector{Float64}}()
        client_sizes = Int[]
        client_losses = Float64[]

        for client_id in active_client_ids
            local_model = rebuild(copy(flat_global))
            opt_state = Flux.setup(_optimizer_rule(config, lrs[client_id]), local_model)
            round_loss, objective_states[client_id] = train_client_model!(
                local_model,
                opt_state,
                client_data[client_id],
                objective,
                config,
                rng;
                state=objective_states[client_id],
                lambda=lambdas[client_id],
            )
            flat_local, _ = Flux.destructure(local_model)
            push!(client_deltas, flat_local .- flat_global)
            push!(client_sizes, size(client_data[client_id].train_x, 2))
            push!(client_losses, round_loss)
        end

        if isempty(client_deltas)
            push!(round_losses, NaN)
        else
            weights = client_sizes ./ sum(client_sizes)
            average_delta = zeros(Float64, length(flat_global))
            for (delta, weight) in zip(client_deltas, weights)
                average_delta .+= weight .* delta
            end
            flat_global .+= average_delta
            push!(round_losses, sum(weights .* client_losses))
        end

        global_model = rebuild(copy(flat_global))
        bound = _compute_objective_bound(
            objective,
            global_model,
            validation_monitor,
            config;
            lambdas=lambdas,
        )
        push!(bound_values, bound)

        if objective_tracks_bound(objective) && !frozen[] && bound <= config.freeze_tau
            frozen[] = true
            frozen_after_round = round
        end
    end

    pcl = config.per_client_lambda ? client_lambda0s : nothing
    return FedTrainingResult(
        objective_method(objective, :fed),
        global_model,
        client_data,
        round_losses,
        selected_clients,
        SchedulerTrace(lambda_values, lr_values, bound_values, frozen_after_round, pcl),
    )
end

function local_synthetic_knapsack(
    dataset::SyntheticKnapsackDataset;
    objective::SyntheticKnapsackObjective=RSPOPlusObjective(),
    config::SyntheticKnapsackTrainingConfig=SyntheticKnapsackTrainingConfig(),
    model=nothing,
)
    _validate_training_config(config, objective)

    rng = MersenneTwister(config.seed)
    model_rng = MersenneTwister(config.seed)
    client_data = prepare_synthetic_knapsack_client_data(dataset)
    base_model =
        isnothing(model) ? build_synthetic_knapsack_model(
            dataset.config.p,
            dataset.config.dim;
            hidden_dim=config.hidden_dim,
            rng=model_rng,
        ) : Flux.f64(deepcopy(model))
    flat_init, rebuild = Flux.destructure(base_model)

    local_models = [rebuild(copy(flat_init)) for _ in eachindex(client_data)]

    client_lambda0s = _synthetic_initial_client_lambda0s(objective, client_data, config)
    client_lr0s = _synthetic_initial_client_lr0s(objective, client_data, config, client_lambda0s)

    opt_states = [
        Flux.setup(_optimizer_rule(config, client_lr0s[client_id]), local_models[client_id]) for
        client_id in eachindex(client_data)
    ]
    objective_states = [
        _initial_client_objective_state(objective, client, config) for client in client_data
    ]
    validation_monitors = [
        objective_tracks_bound(objective) ?
        build_validation_monitor(client_data, config; rng=rng, client_ids=[client_id]) :
        ValidationMonitor(ValidationClientData[]) for
        client_id in eachindex(client_data)
    ]

    frozen_flags = [Ref(false) for _ in eachindex(client_data)]
    lambda_scheds = [
        objective_uses_lambda(objective) ?
        create_inverse_time_scheduler(client_lambda0s[i], config.kappa_lambda; frozen=frozen_flags[i]) :
        nothing for
        i in eachindex(client_data)
    ]
    lr_scheds = [
        create_inverse_time_scheduler(client_lr0s[i], config.kappa_lr; frozen=frozen_flags[i]) for
        i in eachindex(client_data)
    ]

    round_losses = [Float64[] for _ in eachindex(client_data)]
    lambda_values = [Float64[] for _ in eachindex(client_data)]
    lr_values = [Float64[] for _ in eachindex(client_data)]
    bound_values = [Float64[] for _ in eachindex(client_data)]
    frozen_after_round = Vector{Union{Nothing,Int}}(fill(nothing, length(client_data)))

    for round in 1:config.rounds
        for client_id in eachindex(client_data)
            freeze_round = frozen_after_round[client_id]
            if objective_tracks_bound(objective) &&
               !isnothing(freeze_round) &&
               round > freeze_round + config.stop_after_freeze_rounds
                continue
            end

            lambda =
                objective_uses_lambda(objective) ?
                next_schedule_value!(lambda_scheds[client_id]) : NaN
            lr = next_schedule_value!(lr_scheds[client_id])
            push!(lambda_values[client_id], lambda)
            push!(lr_values[client_id], lr)

            Optimisers.adjust!(opt_states[client_id], lr)
            round_loss, objective_states[client_id] = train_client_model!(
                local_models[client_id],
                opt_states[client_id],
                client_data[client_id],
                objective,
                config,
                rng;
                state=objective_states[client_id],
                lambda=lambda,
            )
            push!(round_losses[client_id], round_loss)

            bound = _compute_objective_bound(
                objective,
                local_models[client_id],
                validation_monitors[client_id],
                config;
                lambda=lambda,
            )
            push!(bound_values[client_id], bound)

            if objective_tracks_bound(objective) &&
               !frozen_flags[client_id][] &&
               bound <= config.freeze_tau
                frozen_flags[client_id][] = true
                frozen_after_round[client_id] = round
            end
        end
    end

    pcl = config.per_client_lambda ? client_lambda0s : nothing
    return LocalTrainingResult(
        objective_method(objective, :local),
        [
        LocalClientTrainingResult(
            client_id,
            local_models[client_id],
            round_losses[client_id],
            SchedulerTrace(
                lambda_values[client_id],
                lr_values[client_id],
                bound_values[client_id],
                frozen_after_round[client_id],
                isnothing(pcl) ? nothing : [pcl[client_id]],
            ),
        ) for client_id in eachindex(client_data)
    ],
    )
end

function centralized_synthetic_knapsack(
    dataset::SyntheticKnapsackDataset;
    objective::SyntheticKnapsackObjective=SPOPlusObjective(),
    config::SyntheticKnapsackTrainingConfig=SyntheticKnapsackTrainingConfig(),
    model=nothing,
)
    objective_uses_lambda(objective) && throw(
        ArgumentError(
            "centralized training is only implemented for objectives without lambda schedules; got $(objective_name(objective))",
        ),
    )
    _validate_training_config(config, objective)

    rng = MersenneTwister(config.seed)
    model_rng = MersenneTwister(config.seed)
    client_data = prepare_centralized_synthetic_knapsack_client_data(dataset)
    base_model =
        isnothing(model) ? build_synthetic_knapsack_model(
            dataset.config.p,
            dataset.config.dim;
            hidden_dim=config.hidden_dim,
            rng=model_rng,
        ) : Flux.f64(deepcopy(model))
    flat_init, rebuild = Flux.destructure(base_model)

    local_models = [rebuild(copy(flat_init)) for _ in eachindex(client_data)]
    opt_states = [
        Flux.setup(_optimizer_rule(config, config.lr0), local_models[client_id]) for
        client_id in eachindex(client_data)
    ]
    lr_scheds = [
        create_inverse_time_scheduler(config.lr0, config.kappa_lr) for _ in eachindex(client_data)
    ]
    objective_states = [
        _initial_client_objective_state(objective, client, config) for client in client_data
    ]

    round_losses = [Float64[] for _ in eachindex(client_data)]
    lambda_values = [Float64[] for _ in eachindex(client_data)]
    lr_values = [Float64[] for _ in eachindex(client_data)]
    bound_values = [Float64[] for _ in eachindex(client_data)]

    for _ in 1:config.rounds
        for client_id in eachindex(client_data)
            lr = next_schedule_value!(lr_scheds[client_id])
            push!(lambda_values[client_id], NaN)
            push!(lr_values[client_id], lr)
            push!(bound_values[client_id], NaN)

            Optimisers.adjust!(opt_states[client_id], lr)
            round_loss, objective_states[client_id] = train_client_model!(
                local_models[client_id],
                opt_states[client_id],
                client_data[client_id],
                objective,
                config,
                rng;
                state=objective_states[client_id],
                lambda=NaN,
            )
            push!(round_losses[client_id], round_loss)
        end
    end

    return LocalTrainingResult(
        objective_method(objective, :centralized),
        [
            LocalClientTrainingResult(
                client_id,
                local_models[client_id],
                round_losses[client_id],
                SchedulerTrace(
                    lambda_values[client_id],
                    lr_values[client_id],
                    bound_values[client_id],
                    nothing,
                    nothing,
                ),
            ) for client_id in eachindex(client_data)
        ],
    )
end

function run_synthetic_knapsack_two_stage(
    dataset::SyntheticKnapsackDataset;
    objective::SyntheticKnapsackObjective=RSPOPlusObjective(),
    warm_start_objective::Union{Nothing,SyntheticKnapsackObjective}=nothing,
    personalization_objective::Union{Nothing,SyntheticKnapsackObjective}=nothing,
    warm_start_config::SyntheticKnapsackTrainingConfig=SyntheticKnapsackTrainingConfig(),
    personalization_config::SyntheticKnapsackTrainingConfig=SyntheticKnapsackTrainingConfig(),
    model=nothing,
)
    resolved_warm_start_objective = something(warm_start_objective, objective)
    resolved_personalization_objective = something(personalization_objective, objective)
    same_objective =
        objective_name(resolved_warm_start_objective) ==
        objective_name(resolved_personalization_objective)

    t_total = time()
    if same_objective
        println(
            "=== TWO-STAGE SYNTHETIC KNAPSACK: $(objective_name(resolved_warm_start_objective)) ===",
        )
    else
        println(
            "=== TWO-STAGE SYNTHETIC KNAPSACK: $(objective_name(resolved_warm_start_objective)) -> $(objective_name(resolved_personalization_objective)) ===",
        )
    end
    println("--- Stage 1: Federated warm start ($(warm_start_config.rounds) rounds) ---")
    warm_start = fed_synthetic_knapsack(
        dataset;
        objective=resolved_warm_start_objective,
        config=warm_start_config,
        model=model,
    )
    println("--- Stage 1 complete [$(round(time() - t_total; digits=1))s elapsed] ---")
    t_stage2 = time()
    println("--- Stage 2: Local personalization ($(personalization_config.rounds) rounds) ---")
    personalization = local_synthetic_knapsack(
        dataset;
        objective=resolved_personalization_objective,
        config=personalization_config,
        model=warm_start.model,
    )
    println("--- Stage 2 complete [$(round(time() - t_stage2; digits=1))s] ---")
    println("=== TWO-STAGE COMPLETE [$(round(time() - t_total; digits=1))s total] ===")
    return SyntheticKnapsackTwoStageTrainingResult(warm_start, personalization)
end

function _synthetic_perturbed_layer_builder(;
    nb_samples::Integer=10,
    epsilon::Real=1.0,
    threaded::Bool=false,
    seed=nothing,
)
    nb_samples > 0 || throw(ArgumentError("nb_samples must be positive, got $nb_samples"))
    isfinite(epsilon) && epsilon > 0 ||
        throw(ArgumentError("epsilon must be positive and finite, got $epsilon"))
    nb_samples_int = Int(nb_samples)
    epsilon_float = float(epsilon)
    return client -> InferOpt.PerturbedAdditive(
        linear_maximizer(:min; instance=client.instance);
        ε=epsilon_float,
        nb_samples=nb_samples_int,
        threaded=threaded,
        seed=seed,
    )
end

function PerturbedFYLObjective(;
    nb_samples::Integer=10,
    epsilon::Real=1.0,
    threaded::Bool=false,
    seed=nothing,
)
    perturbed_builder = _synthetic_perturbed_layer_builder(
        ;
        nb_samples=nb_samples,
        epsilon=epsilon,
        threaded=threaded,
        seed=seed,
    )
    return InferOptObjective(
        :perturbed_fyl_mult,
        client -> InferOpt.FenchelYoungLoss(perturbed_builder(client)),
        (client, batch_indices) -> view(client.train_y_true, :, batch_indices),
    )
end

function PerturbedMSEObjective(;
    nb_samples::Integer=10,
    epsilon::Real=1.0,
    threaded::Bool=false,
    seed=nothing,
)
    return PerturbedMSEObjective(
        :dpo_perturbed_mse_mult,
        _synthetic_perturbed_layer_builder(
            ;
            nb_samples=nb_samples,
            epsilon=epsilon,
            threaded=threaded,
            seed=seed,
        ),
    )
end

function synthetic_knapsack_objective_from_name(
    name::Symbol;
    spo_alpha::Real=2.0,
    perturbed_nb_samples::Integer=10,
    perturbed_epsilon::Real=1.0,
    perturbed_threaded::Bool=false,
    perturbed_seed=nothing,
)
    if name === :rspo_plus
        return RSPOPlusObjective()
    elseif name === :spo_plus
        return SPOPlusObjective(; α=spo_alpha)
    elseif name === :pfyl || name === :perturbed_fyl || name === :perturbed_fyl_mult
        return PerturbedFYLObjective(
            ;
            nb_samples=perturbed_nb_samples,
            epsilon=perturbed_epsilon,
            threaded=perturbed_threaded,
            seed=perturbed_seed,
        )
    elseif name === :dpo || name === :perturbed_mse || name === :dpo_perturbed_mse_mult
        return PerturbedMSEObjective(
            ;
            nb_samples=perturbed_nb_samples,
            epsilon=perturbed_epsilon,
            threaded=perturbed_threaded,
            seed=perturbed_seed,
        )
    elseif name === :mse
        return MSEObjective()
    end

    throw(
        ArgumentError(
            "unknown synthetic knapsack objective `$name` (expected rspo_plus, spo_plus, perturbed_fyl, dpo_perturbed_mse_mult, or mse)",
        ),
    )
end

function fed_rspo_plus(
    dataset::SyntheticKnapsackDataset;
    config::SyntheticKnapsackTrainingConfig=SyntheticKnapsackTrainingConfig(),
    model=nothing,
)
    return fed_synthetic_knapsack(dataset; objective=RSPOPlusObjective(), config=config, model=model)
end

function local_rspo_plus(
    dataset::SyntheticKnapsackDataset;
    config::SyntheticKnapsackTrainingConfig=SyntheticKnapsackTrainingConfig(),
    model=nothing,
)
    return local_synthetic_knapsack(
        dataset;
        objective=RSPOPlusObjective(),
        config=config,
        model=model,
    )
end

function fed_spo_plus(
    dataset::SyntheticKnapsackDataset;
    config::SyntheticKnapsackTrainingConfig=SyntheticKnapsackTrainingConfig(),
    model=nothing,
    α::Real=2.0,
)
    return fed_synthetic_knapsack(
        dataset;
        objective=SPOPlusObjective(; α=α),
        config=config,
        model=model,
    )
end

function local_spo_plus(
    dataset::SyntheticKnapsackDataset;
    config::SyntheticKnapsackTrainingConfig=SyntheticKnapsackTrainingConfig(),
    model=nothing,
    α::Real=2.0,
)
    return local_synthetic_knapsack(
        dataset;
        objective=SPOPlusObjective(; α=α),
        config=config,
        model=model,
    )
end

function centralized_spo_plus(
    dataset::SyntheticKnapsackDataset;
    config::SyntheticKnapsackTrainingConfig=SyntheticKnapsackTrainingConfig(),
    model=nothing,
    α::Real=2.0,
)
    return centralized_synthetic_knapsack(
        dataset;
        objective=SPOPlusObjective(; α=α),
        config=config,
        model=model,
    )
end

function fed_perturbed_fyl(
    dataset::SyntheticKnapsackDataset;
    config::SyntheticKnapsackTrainingConfig=SyntheticKnapsackTrainingConfig(),
    model=nothing,
    nb_samples::Integer=10,
    epsilon::Real=1.0,
    threaded::Bool=false,
    seed=nothing,
)
    return fed_synthetic_knapsack(
        dataset;
        objective=PerturbedFYLObjective(
            ;
            nb_samples=nb_samples,
            epsilon=epsilon,
            threaded=threaded,
            seed=seed,
        ),
        config=config,
        model=model,
    )
end

function local_perturbed_fyl(
    dataset::SyntheticKnapsackDataset;
    config::SyntheticKnapsackTrainingConfig=SyntheticKnapsackTrainingConfig(),
    model=nothing,
    nb_samples::Integer=10,
    epsilon::Real=1.0,
    threaded::Bool=false,
    seed=nothing,
)
    return local_synthetic_knapsack(
        dataset;
        objective=PerturbedFYLObjective(
            ;
            nb_samples=nb_samples,
            epsilon=epsilon,
            threaded=threaded,
            seed=seed,
        ),
        config=config,
        model=model,
    )
end

function fed_pfyl(args...; kwargs...)
    return fed_perturbed_fyl(args...; kwargs...)
end

function local_pfyl(args...; kwargs...)
    return local_perturbed_fyl(args...; kwargs...)
end

function fed_perturbed_mse(
    dataset::SyntheticKnapsackDataset;
    config::SyntheticKnapsackTrainingConfig=SyntheticKnapsackTrainingConfig(),
    model=nothing,
    nb_samples::Integer=10,
    epsilon::Real=1.0,
    threaded::Bool=false,
    seed=nothing,
)
    return fed_synthetic_knapsack(
        dataset;
        objective=PerturbedMSEObjective(
            ;
            nb_samples=nb_samples,
            epsilon=epsilon,
            threaded=threaded,
            seed=seed,
        ),
        config=config,
        model=model,
    )
end

function local_perturbed_mse(
    dataset::SyntheticKnapsackDataset;
    config::SyntheticKnapsackTrainingConfig=SyntheticKnapsackTrainingConfig(),
    model=nothing,
    nb_samples::Integer=10,
    epsilon::Real=1.0,
    threaded::Bool=false,
    seed=nothing,
)
    return local_synthetic_knapsack(
        dataset;
        objective=PerturbedMSEObjective(
            ;
            nb_samples=nb_samples,
            epsilon=epsilon,
            threaded=threaded,
            seed=seed,
        ),
        config=config,
        model=model,
    )
end

function fed_dpo(args...; kwargs...)
    return fed_perturbed_mse(args...; kwargs...)
end

function local_dpo(args...; kwargs...)
    return local_perturbed_mse(args...; kwargs...)
end

function fed_mse(
    dataset::SyntheticKnapsackDataset;
    config::SyntheticKnapsackTrainingConfig=SyntheticKnapsackTrainingConfig(),
    model=nothing,
)
    return fed_synthetic_knapsack(dataset; objective=MSEObjective(), config=config, model=model)
end

function local_mse(
    dataset::SyntheticKnapsackDataset;
    config::SyntheticKnapsackTrainingConfig=SyntheticKnapsackTrainingConfig(),
    model=nothing,
)
    return local_synthetic_knapsack(dataset; objective=MSEObjective(), config=config, model=model)
end
