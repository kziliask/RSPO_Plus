using Flux
import InferOpt
using LinearAlgebra
using Optimisers
using Random
using Statistics

Base.@kwdef struct PJMBatteryTrainingConfig
    seed::Int = 42
    hidden_dim::Int = 32
    rounds::Int = 50
    local_epochs::Int = 1
    batch_size::Int = 32
    client_fraction::Float64 = 0.2
    validation_client_fraction::Float64 = 1.0
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
    prox_mu::Float64 = 0.0
end

struct PJMBatteryNormalizationStats{T<:AbstractFloat,V<:AbstractVector{T}}
    mean::V
    std::V
end

struct PJMBatteryClientData
    client_id::Int
    client_name::String
    instance::BatteryDispatchInstance
    normalization::PJMBatteryNormalizationStats{Float64,Vector{Float64}}
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

struct PJMBatteryValidationClientData
    client_id::Int
    instance::BatteryDispatchInstance
    x::Matrix{Float64}
    theta_true::Matrix{Float64}
    z_star::Vector{Float64}
end

struct PJMBatteryValidationMonitor
    clients::Vector{PJMBatteryValidationClientData}
end

struct PJMBatterySchedulerTrace
    lambda_values::Vector{Float64}
    lr_values::Vector{Float64}
    bound_values::Vector{Float64}
    frozen_after_round::Union{Nothing,Int}
    per_client_lambda0::Union{Nothing,Vector{Float64}}
end

abstract type PJMBatteryObjective end

struct PJMRSPOPlusObjective <: PJMBatteryObjective end

struct PJMInferOptObjective{B,T} <: PJMBatteryObjective
    name::Symbol
    loss_builder::B
    target_getter::T
end

struct PJMPerturbedMSEObjective{B} <: PJMBatteryObjective
    name::Symbol
    layer_builder::B
end

struct PJMMSEObjective <: PJMBatteryObjective end

function PJMSPOPlusObjective(; α::Real=2.0)
    alpha = float(α)
    return PJMInferOptObjective(
        :spo_plus,
        client -> InferOpt.SPOPlusLoss(linear_maximizer(:min; instance=client.instance); α=alpha),
        (client, batch_indices) -> view(client.train_theta_true, :, batch_indices),
    )
end

function _pjm_perturbed_layer_builder(;
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

function PJMPerturbedFYLObjective(;
    nb_samples::Integer=10,
    epsilon::Real=1.0,
    threaded::Bool=false,
    seed=nothing,
)
    perturbed_builder = _pjm_perturbed_layer_builder(
        ;
        nb_samples=nb_samples,
        epsilon=epsilon,
        threaded=threaded,
        seed=seed,
    )
    return PJMInferOptObjective(
        :perturbed_fyl_mult,
        client -> InferOpt.FenchelYoungLoss(perturbed_builder(client)),
        (client, batch_indices) -> view(client.train_y_true, :, batch_indices),
    )
end

function PJMPerturbedMSEObjective(;
    nb_samples::Integer=10,
    epsilon::Real=1.0,
    threaded::Bool=false,
    seed=nothing,
)
    return PJMPerturbedMSEObjective(
        :dpo_perturbed_mse_mult,
        _pjm_perturbed_layer_builder(
            ;
            nb_samples=nb_samples,
            epsilon=epsilon,
            threaded=threaded,
            seed=seed,
        ),
    )
end

_pjm_objective_name(::PJMRSPOPlusObjective) = :rspo_plus
_pjm_objective_name(objective::PJMInferOptObjective) = objective.name
_pjm_objective_name(objective::PJMPerturbedMSEObjective) = objective.name
_pjm_objective_name(::PJMMSEObjective) = :mse

_pjm_objective_uses_lambda(::PJMBatteryObjective) = false
_pjm_objective_uses_lambda(::PJMRSPOPlusObjective) = true

_pjm_objective_tracks_bound(::PJMBatteryObjective) = false
_pjm_objective_tracks_bound(::PJMRSPOPlusObjective) = true

function _pjm_objective_method(objective::PJMBatteryObjective, phase::Symbol)
    phase in (:fed, :fedprox, :local, :ditto, :centralized) || throw(
        ArgumentError("phase must be :fed, :fedprox, :local, :ditto, or :centralized, got $phase"),
    )
    return Symbol(phase, "_", _pjm_objective_name(objective))
end

struct PJMBatteryFedTrainingResult
    method::Symbol
    model
    client_data::Vector{PJMBatteryClientData}
    round_losses::Vector{Float64}
    selected_clients::Vector{Vector{Int}}
    trace::PJMBatterySchedulerTrace
end

struct PJMBatteryLocalClientTrainingResult
    client_id::Int
    model
    round_losses::Vector{Float64}
    trace::PJMBatterySchedulerTrace
end

struct PJMBatteryLocalTrainingResult
    method::Symbol
    clients::Vector{PJMBatteryLocalClientTrainingResult}
end

struct PJMBatteryTwoStageTrainingResult
    warm_start::PJMBatteryFedTrainingResult
    personalization::PJMBatteryLocalTrainingResult
end

function pjm_objective_from_name(
    name::Symbol;
    spo_alpha::Real=2.0,
    perturbed_nb_samples::Integer=10,
    perturbed_epsilon::Real=1.0,
    perturbed_threaded::Bool=false,
    perturbed_seed=nothing,
)
    if name === :rspo_plus
        return PJMRSPOPlusObjective()
    elseif name === :spo_plus
        return PJMSPOPlusObjective(; α=spo_alpha)
    elseif name === :pfyl || name === :perturbed_fyl || name === :perturbed_fyl_mult
        return PJMPerturbedFYLObjective(
            ;
            nb_samples=perturbed_nb_samples,
            epsilon=perturbed_epsilon,
            threaded=perturbed_threaded,
            seed=perturbed_seed,
        )
    elseif name === :dpo || name === :perturbed_mse || name === :dpo_perturbed_mse_mult
        return PJMPerturbedMSEObjective(
            ;
            nb_samples=perturbed_nb_samples,
            epsilon=perturbed_epsilon,
            threaded=perturbed_threaded,
            seed=perturbed_seed,
        )
    elseif name === :mse
        return PJMMSEObjective()
    end

    throw(
        ArgumentError(
            "unknown objective `$name` (expected rspo_plus, spo_plus, perturbed_fyl, dpo_perturbed_mse_mult, or mse)",
        ),
    )
end

function _validate_pjm_training_config(
    config::PJMBatteryTrainingConfig,
    objective::PJMBatteryObjective=PJMRSPOPlusObjective(),
)
    config.hidden_dim > 0 || throw(ArgumentError("hidden_dim must be positive"))
    config.rounds > 0 || throw(ArgumentError("rounds must be positive"))
    config.local_epochs > 0 || throw(ArgumentError("local_epochs must be positive"))
    config.batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    config.client_fraction > 0 || throw(ArgumentError("client_fraction must be positive"))
    config.validation_client_fraction > 0 || throw(
        ArgumentError("validation_client_fraction must be positive"),
    )
    if _pjm_objective_uses_lambda(objective)
        config.per_client_lambda || config.lambda0 > 0 || throw(
            ArgumentError("lambda0 must be positive (or set per_client_lambda=true)"),
        )
        config.kappa_lambda >= 0 || throw(ArgumentError("kappa_lambda must be nonnegative"))
    end
    config.lr0 > 0 || throw(ArgumentError("lr0 must be positive"))
    config.kappa_lr >= 0 || throw(ArgumentError("kappa_lr must be nonnegative"))
    config.clip_norm > 0 || throw(ArgumentError("clip_norm must be positive"))
    config.prox_mu >= 0 || throw(ArgumentError("prox_mu must be nonnegative"))
    config.freeze_eps > 0 || throw(ArgumentError("freeze_eps must be positive"))
    config.stop_after_freeze_rounds >= 0 || throw(
        ArgumentError("stop_after_freeze_rounds must be nonnegative"),
    )
    return config
end

function build_pjm_battery_model(
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

function _pjm_optimizer_rule(config::PJMBatteryTrainingConfig, lr::Real)
    return Optimisers.OptimiserChain(Optimisers.ClipNorm(config.clip_norm), Optimisers.Adam(lr))
end

function _pjm_subset_count(value::Real, total::Int)
    total > 0 || return 0
    if value <= 1
        return clamp(ceil(Int, value * total), 1, total)
    end
    return clamp(round(Int, value), 1, total)
end

function _pjm_sample_subset(
    rng::AbstractRNG,
    candidates::AbstractVector{Int},
    subset_spec::Real,
)
    isempty(candidates) && return Int[]
    n_select = _pjm_subset_count(subset_spec, length(candidates))
    order = randperm(rng, length(candidates))[1:n_select]
    return sort(candidates[order])
end

function _pjm_batch_index_sets(
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

function _pjm_concat_float_matrices(
    matrices::AbstractVector{<:AbstractMatrix{<:Real}},
    n_rows::Int,
)
    nonempty = Matrix{Float64}[Matrix{Float64}(matrix) for matrix in matrices if size(matrix, 2) > 0]
    isempty(nonempty) && return zeros(Float64, n_rows, 0)
    return reduce(hcat, nonempty)
end

function _pjm_decision_labels(
    theta_true::AbstractMatrix{<:Real},
    instance::BatteryDispatchInstance,
)
    y_true = zeros(Float64, instance.horizon, size(theta_true, 2))
    for col_idx in axes(theta_true, 2)
        y_true[:, col_idx], _ = solve_battery_dispatch(
            view(theta_true, :, col_idx);
            instance=instance,
            sense=:min,
        )
    end
    return y_true
end

function fit_pjm_feature_normalization_stats(x::AbstractMatrix{<:Real})
    mean_vec = vec(mean(x; dims=2))
    std_vec = vec(std(x; dims=2) .+ 1e-8)
    return PJMBatteryNormalizationStats(mean_vec, std_vec)
end

function apply_pjm_feature_normalization(
    x::AbstractMatrix{<:Real},
    stats::PJMBatteryNormalizationStats,
)
    return (x .- stats.mean) ./ stats.std
end

function _pjm_split_indices(split::PJMBatterySplit, client_id::Int)
    return findall(==(client_id), split.client_ids)
end

function _empty_pjm_normalization(feature_dim::Int)
    return PJMBatteryNormalizationStats(zeros(feature_dim), ones(feature_dim))
end

function _get_pjm_client_split_data(
    split::PJMBatterySplit,
    clients::Vector{PJMBatteryClient},
    client_id::Int;
    normalize_x::Bool=true,
    normalization_stats::Union{Nothing,PJMBatteryNormalizationStats}=nothing,
)
    indices = _pjm_split_indices(split, client_id)
    client = clients[client_id]
    x = split.x[:, indices]
    theta_true = split.theta_true[:, indices]

    if normalize_x && !isempty(indices)
        stats =
            isnothing(normalization_stats) ? fit_pjm_feature_normalization_stats(x) :
            normalization_stats
        x = apply_pjm_feature_normalization(x, stats)
    end

    return BatteryDispatchInstance(client.capacity; horizon=size(split.theta_true, 1)), x, theta_true
end

function prepare_pjm_battery_client_data(
    dataset::PJMBatteryDataset;
    normalize_x::Bool=true,
)
    n_clients = length(dataset.clients)
    feature_dim = size(dataset.train.x, 1)
    train_stats =
        if normalize_x
            [let
                indices = _pjm_split_indices(dataset.train, client_id)
                isempty(indices) ? _empty_pjm_normalization(feature_dim) :
                fit_pjm_feature_normalization_stats(dataset.train.x[:, indices])
            end for client_id in 1:n_clients]
        else
            [_empty_pjm_normalization(feature_dim) for _ in 1:n_clients]
        end

    client_data = Vector{PJMBatteryClientData}(undef, n_clients)
    for client_id in 1:n_clients
        normalization = train_stats[client_id]
        instance, train_x, train_theta_true = _get_pjm_client_split_data(
            dataset.train,
            dataset.clients,
            client_id;
            normalize_x=normalize_x,
            normalization_stats=normalize_x ? normalization : nothing,
        )
        _, val_x, val_theta_true = _get_pjm_client_split_data(
            dataset.val,
            dataset.clients,
            client_id;
            normalize_x=normalize_x,
            normalization_stats=normalize_x ? normalization : nothing,
        )
        _, test_x, test_theta_true = _get_pjm_client_split_data(
            dataset.test,
            dataset.clients,
            client_id;
            normalize_x=normalize_x,
            normalization_stats=normalize_x ? normalization : nothing,
        )
        train_y_true = _pjm_decision_labels(train_theta_true, instance)
        val_y_true = _pjm_decision_labels(val_theta_true, instance)
        test_y_true = _pjm_decision_labels(test_theta_true, instance)
        client = dataset.clients[client_id]
        client_data[client_id] = PJMBatteryClientData(
            client_id,
            client.client_name,
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

function prepare_centralized_pjm_battery_client_data(
    dataset::PJMBatteryDataset;
    normalize_x::Bool=true,
)
    client_data = prepare_pjm_battery_client_data(dataset; normalize_x=normalize_x)
    pooled_train_x = _pjm_concat_float_matrices(
        [client.train_x for client in client_data],
        length(PJM_FEATURE_COLUMNS),
    )
    pooled_train_theta_true = _pjm_concat_float_matrices(
        [client.train_theta_true for client in client_data],
        length(PJM_TARGET_COLUMNS),
    )
    pooled_train_y_true = _pjm_concat_float_matrices(
        [client.train_y_true for client in client_data],
        length(PJM_TARGET_COLUMNS),
    )

    return [
        PJMBatteryClientData(
            client.client_id,
            client.client_name,
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

function _pjm_sample_validation_columns(
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

function build_pjm_validation_monitor(
    client_data::Vector{PJMBatteryClientData},
    config::PJMBatteryTrainingConfig;
    rng::AbstractRNG=MersenneTwister(config.seed),
    client_ids::Union{Nothing,AbstractVector{Int}}=nothing,
)
    if isnothing(client_ids)
        available_client_ids = [
            client.client_id for client in client_data if size(client.val_x, 2) > 0
        ]
        selected_client_ids =
            _pjm_sample_subset(rng, available_client_ids, config.validation_client_fraction)
    else
        selected_client_ids = collect(client_ids)
    end

    monitor_clients = PJMBatteryValidationClientData[]
    for client_id in selected_client_ids
        client = client_data[client_id]
        size(client.val_x, 2) == 0 && continue

        x, theta_true = _pjm_sample_validation_columns(
            rng,
            client.val_x,
            client.val_theta_true,
            config.validation_max_samples_per_client,
        )
        z_star = Float64[
            solve_battery_dispatch(view(theta_true, :, col); instance=client.instance)[2] for
            col in axes(theta_true, 2)
        ]
        push!(
            monitor_clients,
            PJMBatteryValidationClientData(client_id, client.instance, x, theta_true, z_star),
        )
    end

    return PJMBatteryValidationMonitor(monitor_clients)
end

function _pjm_require_objective_lambda(lambda::Real)
    isfinite(lambda) && lambda > 0 || throw(
        ArgumentError("lambda must be positive and finite, got $lambda"),
    )
    return float(lambda)
end

function _resolve_pjm_bound_lambda(
    client_id::Integer;
    lambda::Union{Nothing,Real}=nothing,
    lambdas::Union{Nothing,AbstractVector{<:Real}}=nothing,
)
    if !isnothing(lambda) && !isnothing(lambdas)
        throw(ArgumentError("pass either `lambda` or `lambdas`, not both"))
    elseif !isnothing(lambdas)
        return _pjm_require_objective_lambda(lambdas[client_id])
    elseif !isnothing(lambda)
        return _pjm_require_objective_lambda(lambda)
    end
    throw(ArgumentError("pass either `lambda` or `lambdas`"))
end

function compute_pjm_freeze_bound(
    model,
    monitor::PJMBatteryValidationMonitor;
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
        lam = _resolve_pjm_bound_lambda(
            client.client_id;
            lambda=lambda,
            lambdas=lambdas,
        )
        cache = init_battery_projection_cache(client.instance; lambda=lam)
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

function compute_pjm_client_c_norm(client::PJMBatteryClientData)
    n = size(client.train_theta_true, 2)
    n == 0 && return NaN
    return mean(sqrt.(sum(abs2, client.train_theta_true; dims=1)))
end

function compute_pjm_feasible_radius(instance::BatteryDispatchInstance)
    whole = floor(Int, instance.capacity)
    frac = instance.capacity - whole
    return sqrt(whole + frac^2)
end

function compute_pjm_per_client_lambda0(
    client_data::AbstractVector{PJMBatteryClientData};
    fallback::Float64=1.0,
)
    return [let
        c_norm = compute_pjm_client_c_norm(client)
        radius = compute_pjm_feasible_radius(client.instance)
        (isnan(c_norm) || radius <= 0) ? fallback : max(c_norm / radius, 1e-6)
    end for client in client_data]
end

_initial_pjm_objective_state(::PJMBatteryObjective, client, config) = nothing

function _initial_pjm_objective_state(
    ::PJMRSPOPlusObjective,
    client::PJMBatteryClientData,
    config::PJMBatteryTrainingConfig,
)
    return init_battery_projection_cache(client.instance; lambda=max(config.lambda0, 1e-6))
end

function _build_pjm_loss_layer(
    ::PJMRSPOPlusObjective,
    client::PJMBatteryClientData;
    lambda::Real,
    config::PJMBatteryTrainingConfig,
)
    return RSPOPlusLoss(projection_optimizer; lambda=_pjm_require_objective_lambda(lambda))
end

function _build_pjm_loss_layer(
    objective::PJMInferOptObjective,
    client::PJMBatteryClientData;
    lambda::Real,
    config::PJMBatteryTrainingConfig,
)
    return objective.loss_builder(client)
end

function _build_pjm_loss_layer(
    objective::PJMPerturbedMSEObjective,
    client::PJMBatteryClientData;
    lambda::Real,
    config::PJMBatteryTrainingConfig,
)
    return objective.layer_builder(client)
end

function _build_pjm_loss_layer(
    ::PJMMSEObjective,
    client::PJMBatteryClientData;
    lambda::Real,
    config::PJMBatteryTrainingConfig,
)
    return nothing
end

_prepare_pjm_client_training_state(::PJMBatteryObjective, client, config, state, lambda) = state

function _prepare_pjm_client_training_state(
    ::PJMRSPOPlusObjective,
    client::PJMBatteryClientData,
    config::PJMBatteryTrainingConfig,
    state,
    lambda::Real,
)
    resolved_lambda = _pjm_require_objective_lambda(lambda)
    local_cache =
        isnothing(state) ? init_battery_projection_cache(client.instance; lambda=resolved_lambda) :
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

_persist_pjm_client_training_state(::PJMBatteryObjective, state) = state
_persist_pjm_client_training_state(::PJMRSPOPlusObjective, state) = state.cache

function _pjm_batch_training_loss(
    ::PJMRSPOPlusObjective,
    loss_layer,
    theta_pred,
    theta_batch,
    batch_indices,
    client::PJMBatteryClientData,
    state,
    config::PJMBatteryTrainingConfig;
    lambda::Real,
)
    resolved_lambda = _pjm_require_objective_lambda(lambda)
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

function _pjm_batch_training_loss(
    objective::PJMInferOptObjective,
    loss_layer,
    theta_pred,
    theta_batch,
    batch_indices,
    client::PJMBatteryClientData,
    state,
    config::PJMBatteryTrainingConfig;
    lambda::Real,
)
    targets = objective.target_getter(client, batch_indices)
    total = zero(eltype(theta_pred))
    for (batch_pos, _) in enumerate(batch_indices)
        total += loss_layer(view(theta_pred, :, batch_pos), view(targets, :, batch_pos))
    end
    return total / length(batch_indices)
end

function _pjm_batch_training_loss(
    ::PJMPerturbedMSEObjective,
    loss_layer,
    theta_pred,
    theta_batch,
    batch_indices,
    client::PJMBatteryClientData,
    state,
    config::PJMBatteryTrainingConfig;
    lambda::Real,
)
    total = zero(eltype(theta_pred))
    for (batch_pos, sample_idx) in enumerate(batch_indices)
        y_hat = loss_layer(view(theta_pred, :, batch_pos))
        total += sum(abs2, y_hat .- view(client.train_y_true, :, sample_idx))
    end
    return total / (length(batch_indices) * size(client.train_y_true, 1))
end

function _pjm_batch_training_loss(
    ::PJMMSEObjective,
    loss_layer,
    theta_pred,
    theta_batch,
    batch_indices,
    client::PJMBatteryClientData,
    state,
    config::PJMBatteryTrainingConfig;
    lambda::Real,
)
    return sum(abs2, theta_pred .- theta_batch) / length(theta_pred)
end

function _compute_pjm_objective_bound(
    ::PJMBatteryObjective,
    model,
    monitor::PJMBatteryValidationMonitor,
    config::PJMBatteryTrainingConfig;
    kwargs...,
)
    return NaN
end

function _compute_pjm_objective_bound(
    ::PJMRSPOPlusObjective,
    model,
    monitor::PJMBatteryValidationMonitor,
    config::PJMBatteryTrainingConfig;
    lambda::Union{Nothing,Real}=nothing,
    lambdas::Union{Nothing,AbstractVector{<:Real}}=nothing,
)
    if !isnothing(lambda) && !isnothing(lambdas)
        throw(ArgumentError("pass either `lambda` or `lambdas`, not both"))
    elseif !isnothing(lambdas)
        return compute_pjm_freeze_bound(
            model,
            monitor;
            lambdas=lambdas,
            freeze_eps=config.freeze_eps,
            use_warm_start=config.use_warm_start,
        )
    elseif !isnothing(lambda)
        return compute_pjm_freeze_bound(
            model,
            monitor;
            lambda=_pjm_require_objective_lambda(lambda),
            freeze_eps=config.freeze_eps,
            use_warm_start=config.use_warm_start,
        )
    end
    throw(ArgumentError("pass either `lambda` or `lambdas`"))
end

function _pjm_prox_penalty(
    model,
    reference_params::Union{Nothing,AbstractVector},
    mu::Real,
)
    (isnothing(reference_params) || mu <= 0) && return 0.0

    params = Flux.trainables(model)
    length(params) == length(reference_params) || throw(
        ArgumentError(
            "proximal reference has $(length(reference_params)) parameter tensors, expected $(length(params))",
        ),
    )

    penalty = 0.0
    for (param, reference) in zip(params, reference_params)
        size(param) == size(reference) || throw(
            ArgumentError(
                "proximal reference parameter shape $(size(reference)) does not match model shape $(size(param))",
            ),
        )
        penalty += sum(abs2, param .- reference)
    end
    return float(mu) * penalty / 2
end

function train_pjm_client_model!(
    model,
    opt_state,
    client::PJMBatteryClientData,
    objective::PJMBatteryObjective,
    config::PJMBatteryTrainingConfig,
    rng::AbstractRNG;
    state=nothing,
    lambda::Real=NaN,
    prox_reference=nothing,
    prox_mu::Real=config.prox_mu,
)
    n_samples = size(client.train_x, 2)
    n_samples == 0 && return 0.0, state

    loss_layer = _build_pjm_loss_layer(objective, client; lambda=lambda, config=config)
    objective_state = _prepare_pjm_client_training_state(objective, client, config, state, lambda)

    total_loss = 0.0
    total_samples = 0

    for _ in 1:config.local_epochs
        for batch_indices in _pjm_batch_index_sets(
            rng,
            n_samples,
            config.batch_size;
            shuffle=config.shuffle_batches,
        )
            x_batch = view(client.train_x, :, batch_indices)
            theta_batch = view(client.train_theta_true, :, batch_indices)

            batch_loss, grads = Flux.withgradient(model) do m
                theta_pred = m(x_batch)
                base_loss = _pjm_batch_training_loss(
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
                return base_loss + _pjm_prox_penalty(m, prox_reference, prox_mu)
            end
            Flux.update!(opt_state, model, grads[1])

            total_loss += batch_loss * length(batch_indices)
            total_samples += length(batch_indices)
        end
    end

    return total_loss / total_samples, _persist_pjm_client_training_state(objective, objective_state)
end

function _fed_pjm_battery(
    dataset::PJMBatteryDataset;
    objective::PJMBatteryObjective=PJMRSPOPlusObjective(),
    config::PJMBatteryTrainingConfig=PJMBatteryTrainingConfig(),
    model=nothing,
    phase::Symbol=:fed,
)
    _validate_pjm_training_config(config, objective)

    rng = MersenneTwister(config.seed)
    model_rng = MersenneTwister(config.seed)
    client_data = prepare_pjm_battery_client_data(dataset)
    validation_monitor =
        _pjm_objective_tracks_bound(objective) ?
        build_pjm_validation_monitor(client_data, config; rng=rng) :
        PJMBatteryValidationMonitor(PJMBatteryValidationClientData[])

    global_model =
        isnothing(model) ? build_pjm_battery_model(
            length(PJM_FEATURE_COLUMNS),
            length(PJM_TARGET_COLUMNS);
            hidden_dim=config.hidden_dim,
            rng=model_rng,
        ) : Flux.f64(deepcopy(model))
    flat_global, rebuild = Flux.destructure(global_model)

    frozen = Ref(false)
    client_lambda0s = if config.per_client_lambda && _pjm_objective_uses_lambda(objective)
        compute_pjm_per_client_lambda0(client_data; fallback=config.lambda0)
    else
        fill(config.lambda0, length(client_data))
    end
    lambda_scheds = if _pjm_objective_uses_lambda(objective)
        [
            create_inverse_time_scheduler(client_lambda0s[idx], config.kappa_lambda; frozen=frozen) for
            idx in eachindex(client_data)
        ]
    else
        nothing
    end
    client_lr0s = if config.per_client_lr && _pjm_objective_uses_lambda(objective)
        [config.lr_lambda_alpha * client_lambda0s[idx] / 4 for idx in eachindex(client_data)]
    else
        fill(config.lr0, length(client_data))
    end
    lr_scheds = [
        create_inverse_time_scheduler(client_lr0s[idx], config.kappa_lr; frozen=frozen) for
        idx in eachindex(client_data)
    ]

    round_losses = Float64[]
    selected_clients = Vector{Vector{Int}}()
    lambda_values = Float64[]
    lr_values = Float64[]
    bound_values = Float64[]
    objective_states = [
        _initial_pjm_objective_state(objective, client, config) for client in client_data
    ]
    frozen_after_round = nothing

    for round in 1:config.rounds
        lambdas = if _pjm_objective_uses_lambda(objective)
            [next_schedule_value!(lambda_scheds[idx]) for idx in eachindex(lambda_scheds)]
        else
            fill(NaN, length(client_data))
        end
        lrs = [next_schedule_value!(lr_scheds[idx]) for idx in eachindex(lr_scheds)]
        push!(lambda_values, mean(lambdas))
        push!(lr_values, mean(lrs))

        candidate_ids = [client.client_id for client in client_data if size(client.train_x, 2) > 0]
        active_client_ids = _pjm_sample_subset(rng, candidate_ids, config.client_fraction)
        push!(selected_clients, active_client_ids)

        client_deltas = Vector{Vector{Float64}}()
        client_sizes = Int[]
        client_losses = Float64[]

        for client_id in active_client_ids
            local_model = rebuild(copy(flat_global))
            prox_reference =
                config.prox_mu > 0 ? [copy(param) for param in Flux.trainables(local_model)] :
                nothing
            opt_state = Flux.setup(_pjm_optimizer_rule(config, lrs[client_id]), local_model)
            round_loss, objective_states[client_id] = train_pjm_client_model!(
                local_model,
                opt_state,
                client_data[client_id],
                objective,
                config,
                rng;
                state=objective_states[client_id],
                lambda=lambdas[client_id],
                prox_reference=prox_reference,
                prox_mu=config.prox_mu,
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
        bound = _compute_pjm_objective_bound(
            objective,
            global_model,
            validation_monitor,
            config;
            lambdas=lambdas,
        )
        push!(bound_values, bound)

        if _pjm_objective_tracks_bound(objective) && !frozen[] && bound <= config.freeze_tau
            frozen[] = true
            frozen_after_round = round
        end
    end

    pcl = config.per_client_lambda ? client_lambda0s : nothing
    return PJMBatteryFedTrainingResult(
        _pjm_objective_method(objective, phase),
        global_model,
        client_data,
        round_losses,
        selected_clients,
        PJMBatterySchedulerTrace(lambda_values, lr_values, bound_values, frozen_after_round, pcl),
    )
end

function fed_pjm_battery(
    dataset::PJMBatteryDataset;
    objective::PJMBatteryObjective=PJMRSPOPlusObjective(),
    config::PJMBatteryTrainingConfig=PJMBatteryTrainingConfig(),
    model=nothing,
)
    return _fed_pjm_battery(dataset; objective=objective, config=config, model=model, phase=:fed)
end

function fedprox_pjm_battery(
    dataset::PJMBatteryDataset;
    objective::PJMBatteryObjective=PJMRSPOPlusObjective(),
    config::PJMBatteryTrainingConfig=PJMBatteryTrainingConfig(),
    model=nothing,
)
    return _fed_pjm_battery(
        dataset;
        objective=objective,
        config=config,
        model=model,
        phase=:fedprox,
    )
end

function local_pjm_battery(
    dataset::PJMBatteryDataset;
    objective::PJMBatteryObjective=PJMRSPOPlusObjective(),
    config::PJMBatteryTrainingConfig=PJMBatteryTrainingConfig(),
    model=nothing,
)
    _validate_pjm_training_config(config, objective)

    rng = MersenneTwister(config.seed)
    model_rng = MersenneTwister(config.seed)
    client_data = prepare_pjm_battery_client_data(dataset)
    base_model =
        isnothing(model) ? build_pjm_battery_model(
            length(PJM_FEATURE_COLUMNS),
            length(PJM_TARGET_COLUMNS);
            hidden_dim=config.hidden_dim,
            rng=model_rng,
        ) : Flux.f64(deepcopy(model))
    flat_init, rebuild = Flux.destructure(base_model)

    local_models = [rebuild(copy(flat_init)) for _ in eachindex(client_data)]
    client_lambda0s = if config.per_client_lambda && _pjm_objective_uses_lambda(objective)
        compute_pjm_per_client_lambda0(client_data; fallback=config.lambda0)
    else
        fill(config.lambda0, length(client_data))
    end
    client_lr0s = if config.per_client_lr && _pjm_objective_uses_lambda(objective)
        [config.lr_lambda_alpha * client_lambda0s[idx] / 4 for idx in eachindex(client_data)]
    else
        fill(config.lr0, length(client_data))
    end

    opt_states = [
        Flux.setup(_pjm_optimizer_rule(config, client_lr0s[client_id]), local_models[client_id]) for
        client_id in eachindex(client_data)
    ]
    objective_states = [
        _initial_pjm_objective_state(objective, client, config) for client in client_data
    ]
    validation_monitors = [
        _pjm_objective_tracks_bound(objective) ?
        build_pjm_validation_monitor(client_data, config; rng=rng, client_ids=[client_id]) :
        PJMBatteryValidationMonitor(PJMBatteryValidationClientData[]) for
        client_id in eachindex(client_data)
    ]

    frozen_flags = [Ref(false) for _ in eachindex(client_data)]
    lambda_scheds = [
        _pjm_objective_uses_lambda(objective) ?
        create_inverse_time_scheduler(client_lambda0s[idx], config.kappa_lambda; frozen=frozen_flags[idx]) :
        nothing for idx in eachindex(client_data)
    ]
    lr_scheds = [
        create_inverse_time_scheduler(client_lr0s[idx], config.kappa_lr; frozen=frozen_flags[idx]) for
        idx in eachindex(client_data)
    ]

    round_losses = [Float64[] for _ in eachindex(client_data)]
    lambda_values = [Float64[] for _ in eachindex(client_data)]
    lr_values = [Float64[] for _ in eachindex(client_data)]
    bound_values = [Float64[] for _ in eachindex(client_data)]
    frozen_after_round = Vector{Union{Nothing,Int}}(fill(nothing, length(client_data)))

    for round in 1:config.rounds
        for client_id in eachindex(client_data)
            freeze_round = frozen_after_round[client_id]
            if _pjm_objective_tracks_bound(objective) &&
               !isnothing(freeze_round) &&
               round > freeze_round + config.stop_after_freeze_rounds
                continue
            end

            lambda =
                _pjm_objective_uses_lambda(objective) ?
                next_schedule_value!(lambda_scheds[client_id]) : NaN
            lr = next_schedule_value!(lr_scheds[client_id])
            push!(lambda_values[client_id], lambda)
            push!(lr_values[client_id], lr)

            Optimisers.adjust!(opt_states[client_id], lr)
            round_loss, objective_states[client_id] = train_pjm_client_model!(
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

            bound = _compute_pjm_objective_bound(
                objective,
                local_models[client_id],
                validation_monitors[client_id],
                config;
                lambda=lambda,
            )
            push!(bound_values[client_id], bound)

            if _pjm_objective_tracks_bound(objective) &&
               !frozen_flags[client_id][] &&
               bound <= config.freeze_tau
                frozen_flags[client_id][] = true
                frozen_after_round[client_id] = round
            end
        end
    end

    pcl = config.per_client_lambda ? client_lambda0s : nothing
    return PJMBatteryLocalTrainingResult(
        _pjm_objective_method(objective, :local),
        [
            PJMBatteryLocalClientTrainingResult(
                client_id,
                local_models[client_id],
                round_losses[client_id],
                PJMBatterySchedulerTrace(
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

function centralized_pjm_battery(
    dataset::PJMBatteryDataset;
    objective::PJMBatteryObjective=PJMSPOPlusObjective(),
    config::PJMBatteryTrainingConfig=PJMBatteryTrainingConfig(),
    model=nothing,
)
    _pjm_objective_uses_lambda(objective) && throw(
        ArgumentError(
            "centralized training is only implemented for objectives without lambda schedules; got $(_pjm_objective_name(objective))",
        ),
    )
    _validate_pjm_training_config(config, objective)

    rng = MersenneTwister(config.seed)
    model_rng = MersenneTwister(config.seed)
    client_data = prepare_centralized_pjm_battery_client_data(dataset)
    base_model =
        isnothing(model) ? build_pjm_battery_model(
            length(PJM_FEATURE_COLUMNS),
            length(PJM_TARGET_COLUMNS);
            hidden_dim=config.hidden_dim,
            rng=model_rng,
        ) : Flux.f64(deepcopy(model))
    flat_init, rebuild = Flux.destructure(base_model)

    local_models = [rebuild(copy(flat_init)) for _ in eachindex(client_data)]
    opt_states = [
        Flux.setup(_pjm_optimizer_rule(config, config.lr0), local_models[client_id]) for
        client_id in eachindex(client_data)
    ]
    lr_scheds = [
        create_inverse_time_scheduler(config.lr0, config.kappa_lr) for _ in eachindex(client_data)
    ]
    objective_states = [
        _initial_pjm_objective_state(objective, client, config) for client in client_data
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
            round_loss, objective_states[client_id] = train_pjm_client_model!(
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

    return PJMBatteryLocalTrainingResult(
        _pjm_objective_method(objective, :centralized),
        [
            PJMBatteryLocalClientTrainingResult(
                client_id,
                local_models[client_id],
                round_losses[client_id],
                PJMBatterySchedulerTrace(
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

function run_pjm_battery_two_stage(
    dataset::PJMBatteryDataset;
    objective::PJMBatteryObjective=PJMRSPOPlusObjective(),
    warm_start_config::PJMBatteryTrainingConfig=PJMBatteryTrainingConfig(),
    personalization_config::PJMBatteryTrainingConfig=PJMBatteryTrainingConfig(),
    model=nothing,
)
    warm_start = fed_pjm_battery(
        dataset;
        objective=objective,
        config=warm_start_config,
        model=model,
    )
    personalization = local_pjm_battery(
        dataset;
        objective=objective,
        config=personalization_config,
        model=warm_start.model,
    )
    return PJMBatteryTwoStageTrainingResult(warm_start, personalization)
end

function run_pjm_battery_ditto(
    dataset::PJMBatteryDataset;
    objective::PJMBatteryObjective=PJMRSPOPlusObjective(),
    global_config::PJMBatteryTrainingConfig=PJMBatteryTrainingConfig(),
    personalization_config::PJMBatteryTrainingConfig=PJMBatteryTrainingConfig(),
    model=nothing,
)
    _validate_pjm_training_config(global_config, objective)
    _validate_pjm_training_config(personalization_config, objective)
    global_config.rounds == personalization_config.rounds || throw(
        ArgumentError(
            "Ditto requires matching round counts for the global and personalization configs; got $(global_config.rounds) and $(personalization_config.rounds)",
        ),
    )

    global_rng = MersenneTwister(global_config.seed)
    global_model_rng = MersenneTwister(global_config.seed)
    personalization_rng = MersenneTwister(personalization_config.seed)
    client_data = prepare_pjm_battery_client_data(dataset)
    global_validation_monitor =
        _pjm_objective_tracks_bound(objective) ?
        build_pjm_validation_monitor(client_data, global_config; rng=global_rng) :
        PJMBatteryValidationMonitor(PJMBatteryValidationClientData[])

    base_model =
        isnothing(model) ? build_pjm_battery_model(
            length(PJM_FEATURE_COLUMNS),
            length(PJM_TARGET_COLUMNS);
            hidden_dim=global_config.hidden_dim,
            rng=global_model_rng,
        ) : Flux.f64(deepcopy(model))
    flat_global, rebuild = Flux.destructure(base_model)
    personalized_models = [rebuild(copy(flat_global)) for _ in eachindex(client_data)]

    global_frozen = Ref(false)
    global_lambda0s = if global_config.per_client_lambda && _pjm_objective_uses_lambda(objective)
        compute_pjm_per_client_lambda0(client_data; fallback=global_config.lambda0)
    else
        fill(global_config.lambda0, length(client_data))
    end
    global_lambda_scheds = if _pjm_objective_uses_lambda(objective)
        [
            create_inverse_time_scheduler(global_lambda0s[idx], global_config.kappa_lambda; frozen=global_frozen) for
            idx in eachindex(client_data)
        ]
    else
        nothing
    end
    global_lr0s = if global_config.per_client_lr && _pjm_objective_uses_lambda(objective)
        [global_config.lr_lambda_alpha * global_lambda0s[idx] / 4 for idx in eachindex(client_data)]
    else
        fill(global_config.lr0, length(client_data))
    end
    global_lr_scheds = [
        create_inverse_time_scheduler(global_lr0s[idx], global_config.kappa_lr; frozen=global_frozen) for
        idx in eachindex(client_data)
    ]

    personalization_lambda0s =
        if personalization_config.per_client_lambda && _pjm_objective_uses_lambda(objective)
            compute_pjm_per_client_lambda0(client_data; fallback=personalization_config.lambda0)
        else
            fill(personalization_config.lambda0, length(client_data))
        end
    personalization_lr0s =
        if personalization_config.per_client_lr && _pjm_objective_uses_lambda(objective)
            [
                personalization_config.lr_lambda_alpha * personalization_lambda0s[idx] / 4 for
                idx in eachindex(client_data)
            ]
        else
            fill(personalization_config.lr0, length(client_data))
        end

    personalization_opt_states = [
        Flux.setup(
            _pjm_optimizer_rule(personalization_config, personalization_lr0s[client_id]),
            personalized_models[client_id],
        ) for client_id in eachindex(client_data)
    ]
    personalization_validation_monitors = [
        _pjm_objective_tracks_bound(objective) ?
        build_pjm_validation_monitor(
            client_data,
            personalization_config;
            rng=personalization_rng,
            client_ids=[client_id],
        ) : PJMBatteryValidationMonitor(PJMBatteryValidationClientData[]) for
        client_id in eachindex(client_data)
    ]
    personalization_frozen_flags = [Ref(false) for _ in eachindex(client_data)]
    personalization_lambda_scheds = [
        _pjm_objective_uses_lambda(objective) ?
        create_inverse_time_scheduler(
            personalization_lambda0s[idx],
            personalization_config.kappa_lambda;
            frozen=personalization_frozen_flags[idx],
        ) : nothing for idx in eachindex(client_data)
    ]
    personalization_lr_scheds = [
        create_inverse_time_scheduler(
            personalization_lr0s[idx],
            personalization_config.kappa_lr;
            frozen=personalization_frozen_flags[idx],
        ) for idx in eachindex(client_data)
    ]

    global_round_losses = Float64[]
    selected_clients = Vector{Vector{Int}}()
    global_lambda_values = Float64[]
    global_lr_values = Float64[]
    global_bound_values = Float64[]
    global_objective_states = [
        _initial_pjm_objective_state(objective, client, global_config) for client in client_data
    ]
    global_frozen_after_round = nothing

    personalization_round_losses = [Float64[] for _ in eachindex(client_data)]
    personalization_lambda_values = [Float64[] for _ in eachindex(client_data)]
    personalization_lr_values = [Float64[] for _ in eachindex(client_data)]
    personalization_bound_values = [Float64[] for _ in eachindex(client_data)]
    personalization_frozen_after_round =
        Vector{Union{Nothing,Int}}(fill(nothing, length(client_data)))
    personalization_objective_states = [
        _initial_pjm_objective_state(objective, client, personalization_config) for
        client in client_data
    ]

    for round in 1:global_config.rounds
        global_lambdas = if _pjm_objective_uses_lambda(objective)
            [next_schedule_value!(global_lambda_scheds[idx]) for idx in eachindex(global_lambda_scheds)]
        else
            fill(NaN, length(client_data))
        end
        global_lrs = [next_schedule_value!(global_lr_scheds[idx]) for idx in eachindex(client_data)]
        push!(global_lambda_values, mean(global_lambdas))
        push!(global_lr_values, mean(global_lrs))

        candidate_ids = [client.client_id for client in client_data if size(client.train_x, 2) > 0]
        active_client_ids = _pjm_sample_subset(global_rng, candidate_ids, global_config.client_fraction)
        push!(selected_clients, active_client_ids)

        round_start_model = rebuild(copy(flat_global))
        personalization_reference =
            personalization_config.prox_mu > 0 ?
            [copy(param) for param in Flux.trainables(round_start_model)] : nothing

        client_deltas = Vector{Vector{Float64}}()
        client_sizes = Int[]
        client_losses = Float64[]

        for client_id in active_client_ids
            local_model = rebuild(copy(flat_global))
            global_prox_reference =
                global_config.prox_mu > 0 ? [copy(param) for param in Flux.trainables(local_model)] :
                nothing
            global_opt_state = Flux.setup(
                _pjm_optimizer_rule(global_config, global_lrs[client_id]),
                local_model,
            )
            round_loss, global_objective_states[client_id] = train_pjm_client_model!(
                local_model,
                global_opt_state,
                client_data[client_id],
                objective,
                global_config,
                global_rng;
                state=global_objective_states[client_id],
                lambda=global_lambdas[client_id],
                prox_reference=global_prox_reference,
                prox_mu=global_config.prox_mu,
            )
            flat_local, _ = Flux.destructure(local_model)
            push!(client_deltas, flat_local .- flat_global)
            push!(client_sizes, size(client_data[client_id].train_x, 2))
            push!(client_losses, round_loss)

            freeze_round = personalization_frozen_after_round[client_id]
            if _pjm_objective_tracks_bound(objective) &&
               !isnothing(freeze_round) &&
               round > freeze_round + personalization_config.stop_after_freeze_rounds
                continue
            end

            personalization_lambda =
                _pjm_objective_uses_lambda(objective) ?
                next_schedule_value!(personalization_lambda_scheds[client_id]) : NaN
            personalization_lr = next_schedule_value!(personalization_lr_scheds[client_id])
            push!(personalization_lambda_values[client_id], personalization_lambda)
            push!(personalization_lr_values[client_id], personalization_lr)

            Optimisers.adjust!(personalization_opt_states[client_id], personalization_lr)
            personalization_loss, personalization_objective_states[client_id] =
                train_pjm_client_model!(
                    personalized_models[client_id],
                    personalization_opt_states[client_id],
                    client_data[client_id],
                    objective,
                    personalization_config,
                    personalization_rng;
                    state=personalization_objective_states[client_id],
                    lambda=personalization_lambda,
                    prox_reference=personalization_reference,
                    prox_mu=personalization_config.prox_mu,
                )
            push!(personalization_round_losses[client_id], personalization_loss)

            personalization_bound = _compute_pjm_objective_bound(
                objective,
                personalized_models[client_id],
                personalization_validation_monitors[client_id],
                personalization_config;
                lambda=personalization_lambda,
            )
            push!(personalization_bound_values[client_id], personalization_bound)

            if _pjm_objective_tracks_bound(objective) &&
               !personalization_frozen_flags[client_id][] &&
               personalization_bound <= personalization_config.freeze_tau
                personalization_frozen_flags[client_id][] = true
                personalization_frozen_after_round[client_id] = round
            end
        end

        if isempty(client_deltas)
            push!(global_round_losses, NaN)
        else
            weights = client_sizes ./ sum(client_sizes)
            average_delta = zeros(Float64, length(flat_global))
            for (delta, weight) in zip(client_deltas, weights)
                average_delta .+= weight .* delta
            end
            flat_global .+= average_delta
            push!(global_round_losses, sum(weights .* client_losses))
        end

        global_model = rebuild(copy(flat_global))
        global_bound = _compute_pjm_objective_bound(
            objective,
            global_model,
            global_validation_monitor,
            global_config;
            lambdas=global_lambdas,
        )
        push!(global_bound_values, global_bound)

        if _pjm_objective_tracks_bound(objective) &&
           !global_frozen[] &&
           global_bound <= global_config.freeze_tau
            global_frozen[] = true
            global_frozen_after_round = round
        end
    end

    global_phase = global_config.prox_mu > 0 ? :fedprox : :fed
    global_per_client_lambda0 = global_config.per_client_lambda ? global_lambda0s : nothing
    personalization_per_client_lambda0 =
        personalization_config.per_client_lambda ? personalization_lambda0s : nothing
    global_result = PJMBatteryFedTrainingResult(
        _pjm_objective_method(objective, global_phase),
        rebuild(copy(flat_global)),
        client_data,
        global_round_losses,
        selected_clients,
        PJMBatterySchedulerTrace(
            global_lambda_values,
            global_lr_values,
            global_bound_values,
            global_frozen_after_round,
            global_per_client_lambda0,
        ),
    )
    personalization_result = PJMBatteryLocalTrainingResult(
        _pjm_objective_method(objective, :ditto),
        [
            PJMBatteryLocalClientTrainingResult(
                client_id,
                personalized_models[client_id],
                personalization_round_losses[client_id],
                PJMBatterySchedulerTrace(
                    personalization_lambda_values[client_id],
                    personalization_lr_values[client_id],
                    personalization_bound_values[client_id],
                    personalization_frozen_after_round[client_id],
                    isnothing(personalization_per_client_lambda0) ?
                    nothing : [personalization_per_client_lambda0[client_id]],
                ),
            ) for client_id in eachindex(client_data)
        ],
    )
    return PJMBatteryTwoStageTrainingResult(global_result, personalization_result)
end

function fed_pjm_rspo_plus(
    dataset::PJMBatteryDataset;
    config::PJMBatteryTrainingConfig=PJMBatteryTrainingConfig(),
    model=nothing,
)
    return fed_pjm_battery(dataset; objective=PJMRSPOPlusObjective(), config=config, model=model)
end

function fedprox_pjm_rspo_plus(
    dataset::PJMBatteryDataset;
    config::PJMBatteryTrainingConfig=PJMBatteryTrainingConfig(),
    model=nothing,
)
    return fedprox_pjm_battery(
        dataset;
        objective=PJMRSPOPlusObjective(),
        config=config,
        model=model,
    )
end

function local_pjm_rspo_plus(
    dataset::PJMBatteryDataset;
    config::PJMBatteryTrainingConfig=PJMBatteryTrainingConfig(),
    model=nothing,
)
    return local_pjm_battery(dataset; objective=PJMRSPOPlusObjective(), config=config, model=model)
end

function fed_pjm_spo_plus(
    dataset::PJMBatteryDataset;
    config::PJMBatteryTrainingConfig=PJMBatteryTrainingConfig(),
    model=nothing,
    α::Real=2.0,
)
    return fed_pjm_battery(
        dataset;
        objective=PJMSPOPlusObjective(; α=α),
        config=config,
        model=model,
    )
end

function fedprox_pjm_spo_plus(
    dataset::PJMBatteryDataset;
    config::PJMBatteryTrainingConfig=PJMBatteryTrainingConfig(),
    model=nothing,
    α::Real=2.0,
)
    return fedprox_pjm_battery(
        dataset;
        objective=PJMSPOPlusObjective(; α=α),
        config=config,
        model=model,
    )
end

function local_pjm_spo_plus(
    dataset::PJMBatteryDataset;
    config::PJMBatteryTrainingConfig=PJMBatteryTrainingConfig(),
    model=nothing,
    α::Real=2.0,
)
    return local_pjm_battery(
        dataset;
        objective=PJMSPOPlusObjective(; α=α),
        config=config,
        model=model,
    )
end

function centralized_pjm_spo_plus(
    dataset::PJMBatteryDataset;
    config::PJMBatteryTrainingConfig=PJMBatteryTrainingConfig(),
    model=nothing,
    α::Real=2.0,
)
    return centralized_pjm_battery(
        dataset;
        objective=PJMSPOPlusObjective(; α=α),
        config=config,
        model=model,
    )
end

function fed_pjm_perturbed_fyl(
    dataset::PJMBatteryDataset;
    config::PJMBatteryTrainingConfig=PJMBatteryTrainingConfig(),
    model=nothing,
    nb_samples::Integer=10,
    epsilon::Real=1.0,
    threaded::Bool=false,
    seed=nothing,
)
    return fed_pjm_battery(
        dataset;
        objective=PJMPerturbedFYLObjective(
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

function local_pjm_perturbed_fyl(
    dataset::PJMBatteryDataset;
    config::PJMBatteryTrainingConfig=PJMBatteryTrainingConfig(),
    model=nothing,
    nb_samples::Integer=10,
    epsilon::Real=1.0,
    threaded::Bool=false,
    seed=nothing,
)
    return local_pjm_battery(
        dataset;
        objective=PJMPerturbedFYLObjective(
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

function fed_pjm_pfyl(args...; kwargs...)
    return fed_pjm_perturbed_fyl(args...; kwargs...)
end

function local_pjm_pfyl(args...; kwargs...)
    return local_pjm_perturbed_fyl(args...; kwargs...)
end

function fed_pjm_perturbed_mse(
    dataset::PJMBatteryDataset;
    config::PJMBatteryTrainingConfig=PJMBatteryTrainingConfig(),
    model=nothing,
    nb_samples::Integer=10,
    epsilon::Real=1.0,
    threaded::Bool=false,
    seed=nothing,
)
    return fed_pjm_battery(
        dataset;
        objective=PJMPerturbedMSEObjective(
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

function local_pjm_perturbed_mse(
    dataset::PJMBatteryDataset;
    config::PJMBatteryTrainingConfig=PJMBatteryTrainingConfig(),
    model=nothing,
    nb_samples::Integer=10,
    epsilon::Real=1.0,
    threaded::Bool=false,
    seed=nothing,
)
    return local_pjm_battery(
        dataset;
        objective=PJMPerturbedMSEObjective(
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

function fed_pjm_dpo(args...; kwargs...)
    return fed_pjm_perturbed_mse(args...; kwargs...)
end

function local_pjm_dpo(args...; kwargs...)
    return local_pjm_perturbed_mse(args...; kwargs...)
end

function fed_pjm_mse(
    dataset::PJMBatteryDataset;
    config::PJMBatteryTrainingConfig=PJMBatteryTrainingConfig(),
    model=nothing,
)
    return fed_pjm_battery(dataset; objective=PJMMSEObjective(), config=config, model=model)
end

function fedprox_pjm_mse(
    dataset::PJMBatteryDataset;
    config::PJMBatteryTrainingConfig=PJMBatteryTrainingConfig(),
    model=nothing,
)
    return fedprox_pjm_battery(dataset; objective=PJMMSEObjective(), config=config, model=model)
end

function local_pjm_mse(
    dataset::PJMBatteryDataset;
    config::PJMBatteryTrainingConfig=PJMBatteryTrainingConfig(),
    model=nothing,
)
    return local_pjm_battery(dataset; objective=PJMMSEObjective(), config=config, model=model)
end
