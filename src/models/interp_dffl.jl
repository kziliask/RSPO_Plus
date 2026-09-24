using Flux
using LinearAlgebra
using Random
using Serialization
using Statistics
using TOML

const INTERP_DFFL_CONDITION = :client_wise_80_20

Base.@kwdef struct InterpDFFLConfig
    seed::Int = 42
    fit_fraction::Float64 = 0.8
    split_seed_offset::Int = 1_000_003
    lambda_grid::Vector{Float64} = collect(0.0:0.05:1.0)
    hidden_dim::Int = 64
    local_epochs::Int = 50
    federated_rounds::Int = 50
    federated_local_epochs::Int = 5
    batch_size::Int = 64
    client_fraction::Float64 = 0.2
    learning_rate::Float64 = 1e-3
    clip_norm::Float64 = 1.0
    output_norm_bound::Float64 = 20.0
    shuffle_batches::Bool = true
end

struct L2NormBound
    radius::Float64
end

function L2NormBound(radius::Real)
    isfinite(radius) && radius > 0 ||
        throw(ArgumentError("output norm bound must be positive and finite, got $radius"))
    return L2NormBound(float(radius))
end

function (layer::L2NormBound)(values::AbstractVector{<:Real})
    value_norm = sqrt(sum(abs2, values) + eps(Float64))
    scale = min(1.0, layer.radius / value_norm)
    return values .* scale
end

function (layer::L2NormBound)(values::AbstractMatrix{<:Real})
    value_norms = sqrt.(sum(abs2, values; dims=1) .+ eps(Float64))
    scales = min.(1.0, layer.radius ./ value_norms)
    return values .* scales
end

struct InterpDFFLSplitMetadata
    condition::Symbol
    source_split::Symbol
    fit_fraction::Float64
    split_seed_base::Int
    split_seed_by_client::Dict{Int,Int}
    test_seed_by_client::Dict{Int,Int}
    fit_sample_ids::Dict{Int,Vector{Int}}
    calibration_sample_ids::Dict{Int,Vector{Int}}
    test_sample_ids::Dict{Int,Vector{Int}}
end

struct InterpDFFLClientCalibration
    client_id::Int
    lambda_spo_plus::Float64
    lambda_mse::Float64
    spo_plus_losses::Vector{Float64}
    mse_losses::Vector{Float64}
end

struct InterpDFFLTrainingResult
    config::InterpDFFLConfig
    dataset_config::SyntheticKnapsackConfig
    split_metadata::InterpDFFLSplitMetadata
    local_endpoint::LocalTrainingResult
    federated_endpoint::FedTrainingResult
    client_data::Vector{SyntheticKnapsackClientData}
    calibrations::Dict{Int,InterpDFFLClientCalibration}
    objective_sense::Symbol
end

struct InterpDFFLClientEvaluation
    selector::Symbol
    client_id::Int
    n_test_samples::Int
    lambda::Float64
    mse::Float64
    absolute_regret::Float64
    relative_regret::Float64
end

struct InterpDFFLMetricSummary
    mean::Float64
    std::Float64
    n::Int
end

struct InterpDFFLAggregateEvaluation
    n_clients::Int
    total_test_samples::Int
    mse::InterpDFFLMetricSummary
    absolute_regret::InterpDFFLMetricSummary
    relative_regret::InterpDFFLMetricSummary
end

struct InterpDFFLEvaluationResult
    selector::Symbol
    clients::Vector{InterpDFFLClientEvaluation}
    aggregate::InterpDFFLAggregateEvaluation
end

function _validate_interp_dffl_config(config::InterpDFFLConfig)
    0 < config.fit_fraction < 1 ||
        throw(ArgumentError("fit_fraction must lie strictly between zero and one"))
    config.hidden_dim > 0 || throw(ArgumentError("hidden_dim must be positive"))
    config.local_epochs > 0 || throw(ArgumentError("local_epochs must be positive"))
    config.federated_rounds > 0 ||
        throw(ArgumentError("federated_rounds must be positive"))
    config.federated_local_epochs > 0 ||
        throw(ArgumentError("federated_local_epochs must be positive"))
    config.batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    config.client_fraction > 0 ||
        throw(ArgumentError("client_fraction must be positive"))
    isfinite(config.learning_rate) && config.learning_rate > 0 ||
        throw(ArgumentError("learning_rate must be positive and finite"))
    isfinite(config.clip_norm) && config.clip_norm > 0 ||
        throw(ArgumentError("clip_norm must be positive and finite"))
    isfinite(config.output_norm_bound) && config.output_norm_bound > 0 ||
        throw(ArgumentError("output_norm_bound must be positive and finite"))
    _validate_interp_lambda_grid(config.lambda_grid)
    return config
end

function _validate_interp_lambda_grid(lambda_grid::AbstractVector{<:Real})
    isempty(lambda_grid) && throw(ArgumentError("lambda grid must not be empty"))
    all(isfinite, lambda_grid) ||
        throw(ArgumentError("lambda grid must contain only finite values"))
    all(lambda -> 0 <= lambda <= 1, lambda_grid) ||
        throw(ArgumentError("lambda grid values must lie in [0, 1]"))
    issorted(lambda_grid; lt=<) ||
        throw(ArgumentError("lambda grid must be strictly increasing"))
    first(lambda_grid) == 0 ||
        throw(ArgumentError("lambda grid must include the local endpoint 0"))
    last(lambda_grid) == 1 ||
        throw(ArgumentError("lambda grid must include the federated endpoint 1"))
    return lambda_grid
end

function build_interp_dffl_model(
    input_dim::Int,
    output_dim::Int;
    hidden_dim::Int=64,
    output_norm_bound::Real=20.0,
    rng::AbstractRNG=Random.default_rng(),
)
    input_dim > 0 || throw(ArgumentError("input_dim must be positive"))
    output_dim > 0 || throw(ArgumentError("output_dim must be positive"))
    hidden_dim > 0 || throw(ArgumentError("hidden_dim must be positive"))

    init = Flux.glorot_uniform(rng)
    model = Flux.Chain(
        Flux.Dense(input_dim => hidden_dim, Flux.relu; init=init),
        Flux.Dense(hidden_dim => output_dim; init=init),
        L2NormBound(output_norm_bound),
    )
    return Flux.f64(model)
end

function interpolate_objective_predictions(
    local_predictions::AbstractArray{<:Real},
    federated_predictions::AbstractArray{<:Real},
    lambda::Real,
)
    size(local_predictions) == size(federated_predictions) || throw(
        ArgumentError(
            "local and federated prediction shapes must match; got $(size(local_predictions)) and $(size(federated_predictions))",
        ),
    )
    isfinite(lambda) && 0 <= lambda <= 1 ||
        throw(ArgumentError("interpolation lambda must be finite and lie in [0, 1]"))
    lambda == 0 && return copy(local_predictions)
    lambda == 1 && return copy(federated_predictions)
    return (1 - lambda) .* local_predictions .+ lambda .* federated_predictions
end

function spoplus_tilted_total_objective(
    predicted_uncertain::AbstractArray{<:Real},
    true_uncertain::AbstractArray{<:Real},
    known_offset::AbstractArray{<:Real},
)
    size(predicted_uncertain) == size(true_uncertain) == size(known_offset) || throw(
        ArgumentError("prediction, truth, and known offset shapes must match"),
    )
    return 2 .* predicted_uncertain .- true_uncertain .+ known_offset
end

function _total_objective(
    uncertain::AbstractVector{<:Real},
    known_offset::Union{Nothing,AbstractVector{<:Real}},
)
    isnothing(known_offset) && return Vector{Float64}(uncertain)
    length(uncertain) == length(known_offset) ||
        throw(ArgumentError("uncertain objective and known offset lengths must match"))
    return Vector{Float64}(uncertain .+ known_offset)
end

function _tilted_total_objective(
    predicted_uncertain::AbstractVector{<:Real},
    true_uncertain::AbstractVector{<:Real},
    known_offset::Union{Nothing,AbstractVector{<:Real}},
)
    length(predicted_uncertain) == length(true_uncertain) ||
        throw(ArgumentError("prediction and truth lengths must match"))
    if isnothing(known_offset)
        return Vector{Float64}(2 .* predicted_uncertain .- true_uncertain)
    end
    return Vector{Float64}(
        spoplus_tilted_total_objective(
            predicted_uncertain,
            true_uncertain,
            known_offset,
        ),
    )
end

function spoplus_loss_minimization(
    predicted_uncertain::AbstractVector{<:Real},
    true_uncertain::AbstractVector{<:Real};
    instance::KnapsackInstance,
    known_offset::Union{Nothing,AbstractVector{<:Real}}=nothing,
)
    true_total = _total_objective(true_uncertain, known_offset)
    tilted_total =
        _tilted_total_objective(predicted_uncertain, true_uncertain, known_offset)
    w_true, _ = solve_fractional_knapsack(true_total; instance=instance, sense=:min)
    w_tilt, _ = solve_fractional_knapsack(tilted_total; instance=instance, sense=:min)
    return dot(tilted_total, w_true .- w_tilt)
end

function spoplus_loss_maximization(
    predicted_uncertain::AbstractVector{<:Real},
    true_uncertain::AbstractVector{<:Real};
    instance::KnapsackInstance,
    known_offset::Union{Nothing,AbstractVector{<:Real}}=nothing,
)
    true_total = _total_objective(true_uncertain, known_offset)
    tilted_total =
        _tilted_total_objective(predicted_uncertain, true_uncertain, known_offset)
    w_true, _ = solve_fractional_knapsack(true_total; instance=instance, sense=:max)
    w_tilt, _ = solve_fractional_knapsack(tilted_total; instance=instance, sense=:max)
    return dot(tilted_total, w_tilt .- w_true)
end

function mean_spoplus_loss(
    predicted_uncertain::AbstractMatrix{<:Real},
    true_uncertain::AbstractMatrix{<:Real};
    instance::KnapsackInstance,
    sense::Symbol=:min,
    known_offset::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
)
    size(predicted_uncertain) == size(true_uncertain) ||
        throw(ArgumentError("prediction and truth shapes must match"))
    if !isnothing(known_offset)
        size(known_offset) == size(true_uncertain) ||
            throw(ArgumentError("known offset and truth shapes must match"))
    end
    n_samples = size(true_uncertain, 2)
    n_samples > 0 || throw(ArgumentError("SPO+ calibration requires at least one sample"))

    loss_function =
        sense === :min ? spoplus_loss_minimization :
        sense === :max ? spoplus_loss_maximization :
        throw(ArgumentError("sense must be :min or :max, got $sense"))
    total_loss = 0.0
    for sample_idx in axes(true_uncertain, 2)
        offset =
            isnothing(known_offset) ? nothing : view(known_offset, :, sample_idx)
        total_loss += loss_function(
            view(predicted_uncertain, :, sample_idx),
            view(true_uncertain, :, sample_idx);
            instance=instance,
            known_offset=offset,
        )
    end
    return total_loss / n_samples
end

function prediction_mse(
    predictions::AbstractArray{<:Real},
    targets::AbstractArray{<:Real},
)
    size(predictions) == size(targets) ||
        throw(ArgumentError("prediction and target shapes must match"))
    isempty(targets) && throw(ArgumentError("MSE calibration requires at least one target"))
    return mean(abs2, predictions .- targets)
end

function select_interp_lambda(
    local_predictions::AbstractArray{<:Real},
    federated_predictions::AbstractArray{<:Real},
    loss_function;
    lambda_grid::AbstractVector{<:Real}=collect(0.0:0.05:1.0),
)
    _validate_interp_lambda_grid(lambda_grid)
    size(local_predictions) == size(federated_predictions) ||
        throw(ArgumentError("local and federated prediction shapes must match"))

    losses = Vector{Float64}(undef, length(lambda_grid))
    best_lambda = float(first(lambda_grid))
    best_loss = Inf
    for (idx, lambda) in pairs(lambda_grid)
        predictions =
            interpolate_objective_predictions(local_predictions, federated_predictions, lambda)
        candidate_loss = float(loss_function(predictions))
        isfinite(candidate_loss) || throw(
            ArgumentError(
                "non-finite calibration loss for lambda $(float(lambda)): $candidate_loss",
            ),
        )
        losses[idx] = candidate_loss
        if candidate_loss < best_loss
            best_loss = candidate_loss
            best_lambda = float(lambda)
        end
    end
    return (lambda=best_lambda, losses=losses)
end

function _copy_synthetic_config_with_counts(
    config::SyntheticKnapsackConfig;
    n_train_total::Int,
    n_val_total::Int,
    n_test_total::Int,
)
    return SyntheticKnapsackConfig(
        seed=config.seed,
        n_clients=config.n_clients,
        p=config.p,
        dim=config.dim,
        deg=config.deg,
        epsilon_noise=config.epsilon_noise,
        eta_obj=config.eta_obj,
        eta_constr=config.eta_constr,
        eta_constr_affects_weights=config.eta_constr_affects_weights,
        eta_data_dist=config.eta_data_dist,
        data_imbalance=config.data_imbalance,
        n_train_total=n_train_total,
        n_val_total=n_val_total,
        n_test_total=n_test_total,
        capacity_ratio=config.capacity_ratio,
        obj_low=config.obj_low,
        obj_high=config.obj_high,
        weight_low=config.weight_low,
        weight_high=config.weight_high,
    )
end

function _synthetic_split_subset(
    split::SyntheticKnapsackSplit,
    indices::AbstractVector{Int},
)
    return SyntheticKnapsackSplit(
        Matrix{Float64}(split.x[:, indices]),
        Matrix{Float64}(split.theta_true[:, indices]),
        Vector{Int}(split.client_ids[indices]),
        Vector{Int}(split.sample_ids[indices]),
    )
end

function _empty_synthetic_split(config::SyntheticKnapsackConfig)
    return SyntheticKnapsackSplit(
        zeros(Float64, config.p, 0),
        zeros(Float64, config.dim, 0),
        Int[],
        Int[],
    )
end

function split_synthetic_knapsack_for_interp_dffl(
    dataset::SyntheticKnapsackDataset,
    config::InterpDFFLConfig=InterpDFFLConfig(),
)
    _validate_interp_dffl_config(config)

    fit_indices = Int[]
    calibration_indices = Int[]
    fit_sample_ids = Dict{Int,Vector{Int}}()
    calibration_sample_ids = Dict{Int,Vector{Int}}()
    test_sample_ids = Dict{Int,Vector{Int}}()
    split_seed_by_client = Dict{Int,Int}()
    test_seed_by_client = Dict{Int,Int}()
    split_seed_base = config.seed + config.split_seed_offset

    for client_id in 1:dataset.config.n_clients
        client_indices = client_split_indices(dataset.train, client_id)
        length(client_indices) >= 2 || throw(
            ArgumentError(
                "client $client_id needs at least two training observations for an 80/20 split",
            ),
        )
        client_seed = split_seed_base + client_id
        split_seed_by_client[client_id] = client_seed
        test_seed_by_client[client_id] =
            dataset.config.seed + _RNG_OFFSETS.split_test + 1_000 * client_id
        permutation = randperm(MersenneTwister(client_seed), length(client_indices))
        n_fit = clamp(floor(Int, config.fit_fraction * length(client_indices)), 1, length(client_indices) - 1)
        client_fit_indices = sort(client_indices[permutation[1:n_fit]])
        client_calibration_indices = sort(client_indices[permutation[(n_fit + 1):end]])
        append!(fit_indices, client_fit_indices)
        append!(calibration_indices, client_calibration_indices)
        fit_sample_ids[client_id] =
            Vector{Int}(dataset.train.sample_ids[client_fit_indices])
        calibration_sample_ids[client_id] =
            Vector{Int}(dataset.train.sample_ids[client_calibration_indices])
        client_test_indices = client_split_indices(dataset.test, client_id)
        test_sample_ids[client_id] =
            Vector{Int}(dataset.test.sample_ids[client_test_indices])
    end

    fit_split = _synthetic_split_subset(dataset.train, fit_indices)
    calibration_split = _synthetic_split_subset(dataset.train, calibration_indices)
    split_config = _copy_synthetic_config_with_counts(
        dataset.config;
        n_train_total=length(fit_indices),
        n_val_total=length(calibration_indices),
        n_test_total=length(dataset.test.sample_ids),
    )
    metadata = summarize_synthetic_knapsack_dataset(
        dataset.clients,
        fit_split,
        calibration_split,
        dataset.test,
    )
    split_dataset = SyntheticKnapsackDataset(
        split_config,
        dataset.clients,
        dataset.true_weight_matrix,
        fit_split,
        calibration_split,
        dataset.test,
        metadata,
    )
    split_metadata = InterpDFFLSplitMetadata(
        INTERP_DFFL_CONDITION,
        :train,
        config.fit_fraction,
        split_seed_base,
        split_seed_by_client,
        test_seed_by_client,
        fit_sample_ids,
        calibration_sample_ids,
        test_sample_ids,
    )
    return split_dataset, split_metadata
end

function _fit_only_synthetic_dataset(dataset::SyntheticKnapsackDataset)
    empty_split = _empty_synthetic_split(dataset.config)
    fit_only_config = _copy_synthetic_config_with_counts(
        dataset.config;
        n_train_total=length(dataset.train.sample_ids),
        n_val_total=0,
        n_test_total=0,
    )
    metadata = summarize_synthetic_knapsack_dataset(
        dataset.clients,
        dataset.train,
        empty_split,
        empty_split,
    )
    return SyntheticKnapsackDataset(
        fit_only_config,
        dataset.clients,
        dataset.true_weight_matrix,
        dataset.train,
        empty_split,
        empty_split,
        metadata,
    )
end

function interp_dffl_endpoint_configs(config::InterpDFFLConfig)
    _validate_interp_dffl_config(config)
    common = (
        seed=config.seed,
        hidden_dim=config.hidden_dim,
        batch_size=config.batch_size,
        validation_client_fraction=1.0,
        validation_max_samples_per_client=0,
        lambda0=NaN,
        kappa_lambda=0.0,
        per_client_lambda=false,
        lr0=config.learning_rate,
        kappa_lr=0.0,
        per_client_lr=false,
        lr_lambda_alpha=0.0,
        clip_norm=config.clip_norm,
        freeze_tau=NaN,
        freeze_eps=1e-8,
        stop_after_freeze_rounds=0,
        shuffle_batches=config.shuffle_batches,
        use_warm_start=false,
    )
    local_config = SyntheticKnapsackTrainingConfig(
        ;
        common...,
        rounds=config.local_epochs,
        local_epochs=1,
        client_fraction=1.0,
    )
    federated_config = SyntheticKnapsackTrainingConfig(
        ;
        common...,
        rounds=config.federated_rounds,
        local_epochs=config.federated_local_epochs,
        client_fraction=config.client_fraction,
    )
    return (local_config=local_config, federated_config=federated_config)
end

function _local_endpoint_map(result::LocalTrainingResult)
    client_map = Dict(client.client_id => client for client in result.clients)
    length(client_map) == length(result.clients) ||
        throw(ArgumentError("duplicate client ids in local endpoint results"))
    return client_map
end

function _client_data_map(client_data::Vector{SyntheticKnapsackClientData})
    client_map = Dict(client.client_id => client for client in client_data)
    length(client_map) == length(client_data) ||
        throw(ArgumentError("duplicate client ids in prepared client data"))
    return client_map
end

function train_interp_dffl(
    dataset::SyntheticKnapsackDataset;
    config::InterpDFFLConfig=InterpDFFLConfig(seed=dataset.config.seed),
)
    _validate_interp_dffl_config(config)
    split_dataset, split_metadata =
        split_synthetic_knapsack_for_interp_dffl(dataset, config)
    fit_only_dataset = _fit_only_synthetic_dataset(split_dataset)
    endpoint_configs = interp_dffl_endpoint_configs(config)
    common_initialization = build_interp_dffl_model(
        dataset.config.p,
        dataset.config.dim;
        hidden_dim=config.hidden_dim,
        output_norm_bound=config.output_norm_bound,
        rng=MersenneTwister(config.seed),
    )

    local_endpoint = local_synthetic_knapsack(
        fit_only_dataset;
        objective=SPOPlusObjective(),
        config=endpoint_configs.local_config,
        model=common_initialization,
    )
    federated_endpoint = fed_synthetic_knapsack(
        fit_only_dataset;
        objective=SPOPlusObjective(),
        config=endpoint_configs.federated_config,
        model=common_initialization,
    )

    client_data = prepare_synthetic_knapsack_client_data(split_dataset)
    local_models = _local_endpoint_map(local_endpoint)
    calibrations = Dict{Int,InterpDFFLClientCalibration}()
    for client in sort(client_data; by=client -> client.client_id)
        size(client.val_x, 2) > 0 ||
            throw(ArgumentError("client $(client.client_id) has no calibration observations"))
        local_predictions = local_models[client.client_id].model(client.val_x)
        federated_predictions = federated_endpoint.model(client.val_x)

        spo_selection = select_interp_lambda(
            local_predictions,
            federated_predictions,
            predictions -> mean_spoplus_loss(
                predictions,
                client.val_theta_true;
                instance=client.instance,
                sense=:min,
            );
            lambda_grid=config.lambda_grid,
        )
        mse_selection = select_interp_lambda(
            local_predictions,
            federated_predictions,
            predictions -> prediction_mse(predictions, client.val_theta_true);
            lambda_grid=config.lambda_grid,
        )
        calibrations[client.client_id] = InterpDFFLClientCalibration(
            client.client_id,
            spo_selection.lambda,
            mse_selection.lambda,
            spo_selection.losses,
            mse_selection.losses,
        )
    end

    return InterpDFFLTrainingResult(
        config,
        dataset.config,
        split_metadata,
        local_endpoint,
        federated_endpoint,
        client_data,
        calibrations,
        :min,
    )
end

function _canonical_interp_selector(selector::Symbol)
    selector in (:spo_plus, :spoplus, :interp_dffl) && return :spo_plus
    selector in (:mse, :interp_mse) && return :mse
    throw(ArgumentError("selector must be :spo_plus or :mse, got $selector"))
end

function _selected_interp_lambda(
    calibration::InterpDFFLClientCalibration,
    selector::Symbol,
)
    canonical_selector = _canonical_interp_selector(selector)
    return canonical_selector === :spo_plus ? calibration.lambda_spo_plus : calibration.lambda_mse
end

function _predict_interp_dffl_normalized(
    result::InterpDFFLTrainingResult,
    client_id::Int,
    normalized_features::AbstractVecOrMat{<:Real};
    selector::Symbol=:spo_plus,
)
    local_models = _local_endpoint_map(result.local_endpoint)
    haskey(local_models, client_id) ||
        throw(ArgumentError("unknown client id $client_id"))
    haskey(result.calibrations, client_id) ||
        throw(ArgumentError("missing calibration for client $client_id"))
    lambda = _selected_interp_lambda(result.calibrations[client_id], selector)
    local_predictions = local_models[client_id].model(normalized_features)
    federated_predictions = result.federated_endpoint.model(normalized_features)
    return interpolate_objective_predictions(local_predictions, federated_predictions, lambda)
end

function predict_interp_dffl(
    result::InterpDFFLTrainingResult,
    client_id::Int,
    features::AbstractVecOrMat{<:Real};
    selector::Symbol=:spo_plus,
    normalized::Bool=false,
)
    client_map = _client_data_map(result.client_data)
    haskey(client_map, client_id) ||
        throw(ArgumentError("unknown client id $client_id"))
    normalized_features =
        normalized ? features :
        apply_feature_normalization(features, client_map[client_id].normalization)
    return _predict_interp_dffl_normalized(
        result,
        client_id,
        normalized_features;
        selector=selector,
    )
end

function decision_regret_minimization(
    predicted_uncertain::AbstractVector{<:Real},
    true_uncertain::AbstractVector{<:Real};
    instance::KnapsackInstance,
    known_offset::Union{Nothing,AbstractVector{<:Real}}=nothing,
)
    predicted_total = _total_objective(predicted_uncertain, known_offset)
    true_total = _total_objective(true_uncertain, known_offset)
    predicted_decision, _ =
        solve_fractional_knapsack(predicted_total; instance=instance, sense=:min)
    _, true_optimal_objective =
        solve_fractional_knapsack(true_total; instance=instance, sense=:min)
    return dot(true_total, predicted_decision) - true_optimal_objective
end

function decision_regret_maximization(
    predicted_uncertain::AbstractVector{<:Real},
    true_uncertain::AbstractVector{<:Real};
    instance::KnapsackInstance,
    known_offset::Union{Nothing,AbstractVector{<:Real}}=nothing,
)
    predicted_total = _total_objective(predicted_uncertain, known_offset)
    true_total = _total_objective(true_uncertain, known_offset)
    predicted_decision, _ =
        solve_fractional_knapsack(predicted_total; instance=instance, sense=:max)
    _, true_optimal_objective =
        solve_fractional_knapsack(true_total; instance=instance, sense=:max)
    return true_optimal_objective - dot(true_total, predicted_decision)
end

function _summarize_interp_metric(values)
    finite_values = Float64[float(value) for value in values if isfinite(value)]
    isempty(finite_values) && return InterpDFFLMetricSummary(NaN, NaN, 0)
    return InterpDFFLMetricSummary(
        mean(finite_values),
        length(finite_values) == 1 ? 0.0 : std(finite_values; corrected=false),
        length(finite_values),
    )
end

function evaluate_interp_dffl(
    result::InterpDFFLTrainingResult;
    selector::Symbol=:spo_plus,
)
    canonical_selector = _canonical_interp_selector(selector)
    client_evaluations = InterpDFFLClientEvaluation[]
    for client in sort(result.client_data; by=client -> client.client_id)
        n_test_samples = size(client.test_x, 2)
        lambda =
            _selected_interp_lambda(result.calibrations[client.client_id], canonical_selector)
        if n_test_samples == 0
            push!(
                client_evaluations,
                InterpDFFLClientEvaluation(
                    canonical_selector,
                    client.client_id,
                    0,
                    lambda,
                    NaN,
                    NaN,
                    NaN,
                ),
            )
            continue
        end

        predictions = _predict_interp_dffl_normalized(
            result,
            client.client_id,
            client.test_x;
            selector=canonical_selector,
        )
        regrets = Float64[
            decision_regret_minimization(
                view(predictions, :, sample_idx),
                view(client.test_theta_true, :, sample_idx);
                instance=client.instance,
            ) for sample_idx in axes(client.test_theta_true, 2)
        ]
        true_objectives = Float64[
            solve_fractional_knapsack(
                view(client.test_theta_true, :, sample_idx);
                instance=client.instance,
                sense=:min,
            )[2] for sample_idx in axes(client.test_theta_true, 2)
        ]
        absolute_regret = mean(regrets)
        denominator = mean(abs, true_objectives)
        relative_regret = denominator <= 0 ? NaN : absolute_regret / denominator
        push!(
            client_evaluations,
            InterpDFFLClientEvaluation(
                canonical_selector,
                client.client_id,
                n_test_samples,
                lambda,
                prediction_mse(predictions, client.test_theta_true),
                absolute_regret,
                relative_regret,
            ),
        )
    end

    aggregate = InterpDFFLAggregateEvaluation(
        length(client_evaluations),
        sum(client.n_test_samples for client in client_evaluations),
        _summarize_interp_metric(client.mse for client in client_evaluations),
        _summarize_interp_metric(client.absolute_regret for client in client_evaluations),
        _summarize_interp_metric(client.relative_regret for client in client_evaluations),
    )
    return InterpDFFLEvaluationResult(canonical_selector, client_evaluations, aggregate)
end

function evaluate_interp_dffl_selectors(result::InterpDFFLTrainingResult)
    return (
        spo_plus=evaluate_interp_dffl(result; selector=:spo_plus),
        mse=evaluate_interp_dffl(result; selector=:mse),
    )
end

function validate_temporal_interp_split(
    fit_dates,
    calibration_dates,
    test_dates,
)
    isempty(fit_dates) && throw(ArgumentError("fitting dates must not be empty"))
    isempty(calibration_dates) &&
        throw(ArgumentError("calibration dates must not be empty"))
    isempty(test_dates) && throw(ArgumentError("test dates must not be empty"))
    maximum(fit_dates) < minimum(calibration_dates) ||
        throw(ArgumentError("every fitting date must precede every calibration date"))
    maximum(calibration_dates) < minimum(test_dates) ||
        throw(ArgumentError("every calibration date must precede every test date"))
    return true
end

function _interp_csv_escape(value)
    text = string(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text)
        return string('"', replace(text, "\"" => "\"\""), '"')
    end
    return text
end

function _write_interp_csv(path::AbstractString, header, rows)
    open(path, "w") do io
        println(io, join(_interp_csv_escape.(header), ","))
        for row in rows
            println(io, join(_interp_csv_escape.(row), ","))
        end
    end
    return path
end

function _interp_config_dict(result::InterpDFFLTrainingResult)
    config = result.config
    dataset = result.dataset_config
    return Dict(
        "method" => "interp_dffl",
        "condition" => string(result.split_metadata.condition),
        "objective_sense" => "minimization_after_paper_sign_transform",
        "dataset" => Dict(
            string(name) => getfield(dataset, name) for name in fieldnames(typeof(dataset))
        ),
        "interpolation" => Dict(
            "seed" => config.seed,
            "fit_fraction" => config.fit_fraction,
            "split_seed_offset" => config.split_seed_offset,
            "lambda_grid" => config.lambda_grid,
        ),
        "endpoint_training" => Dict(
            "architecture" => string(
                "Dense(p,",
                config.hidden_dim,
                ",relu)-Dense(",
                config.hidden_dim,
                ",d)-L2NormBound(",
                config.output_norm_bound,
                ")",
            ),
            "hidden_dim" => config.hidden_dim,
            "local_epochs" => config.local_epochs,
            "federated_rounds" => config.federated_rounds,
            "federated_local_epochs" => config.federated_local_epochs,
            "batch_size" => config.batch_size,
            "client_fraction" => config.client_fraction,
            "aggregation" => "sample_size_weighted_fedavg",
            "optimizer" => "Adam",
            "learning_rate" => config.learning_rate,
            "gradient_clip_norm" => config.clip_norm,
            "output_l2_norm_bound" => config.output_norm_bound,
            "shared_deterministic_initialization" => true,
        ),
    )
end

function export_interp_dffl_run(
    result::InterpDFFLTrainingResult;
    root::AbstractString=joinpath("results", "experiment1", "synthetic_knapsack", "interp_dffl"),
    run_id::Union{Nothing,AbstractString}=nothing,
)
    resolved_run_id = something(
        run_id,
        join(
            (
                "seed$(result.dataset_config.seed)",
                "obj$(result.dataset_config.eta_obj)",
                "constr$(result.dataset_config.eta_constr)",
                "data$(result.dataset_config.eta_data_dist)",
            ),
            "_",
        ),
    )
    export_dir = joinpath(root, resolved_run_id)
    mkpath(export_dir)
    evaluations = evaluate_interp_dffl_selectors(result)

    config_path = joinpath(export_dir, "config.toml")
    open(config_path, "w") do io
        TOML.print(io, _interp_config_dict(result); sorted=true)
    end

    artifact_path = joinpath(export_dir, "artifact.jls")
    open(artifact_path, "w") do io
        serialize(io, result)
    end

    calibration_rows = (
        (
            selector,
            result.split_metadata.condition,
            client_id,
            lambda,
            selector === :spo_plus ?
            calibration.spo_plus_losses[lambda_idx] :
            calibration.mse_losses[lambda_idx],
        ) for (client_id, calibration) in sort(collect(result.calibrations); by=first)
        for selector in (:spo_plus, :mse)
        for (lambda_idx, lambda) in pairs(result.config.lambda_grid)
    )
    calibration_path = _write_interp_csv(
        joinpath(export_dir, "calibration_curves.csv"),
        ("selector", "condition", "client_id", "lambda", "validation_loss"),
        calibration_rows,
    )

    selected_path = _write_interp_csv(
        joinpath(export_dir, "selected_lambdas.csv"),
        ("client_id", "lambda_spo_plus", "lambda_mse"),
        (
            (client_id, calibration.lambda_spo_plus, calibration.lambda_mse) for
            (client_id, calibration) in sort(collect(result.calibrations); by=first)
        ),
    )

    split_rows = (
        (
            split,
            client_id,
            sample_id,
            split === :fit ?
            result.split_metadata.split_seed_by_client[client_id] :
            split === :calibration ?
            result.split_metadata.split_seed_by_client[client_id] :
            result.split_metadata.test_seed_by_client[client_id],
        ) for client_id in sort(collect(keys(result.calibrations)))
        for split in (:fit, :calibration, :test)
        for sample_id in (
            split === :fit ?
            result.split_metadata.fit_sample_ids[client_id] :
            split === :calibration ?
            result.split_metadata.calibration_sample_ids[client_id] :
            result.split_metadata.test_sample_ids[client_id]
        )
    )
    split_path = _write_interp_csv(
        joinpath(export_dir, "split_metadata.csv"),
        ("split", "client_id", "sample_id", "seed"),
        split_rows,
    )

    preprocessing_rows = (
        (
            client.client_id,
            feature_idx,
            client.normalization.mean[feature_idx],
            client.normalization.std[feature_idx],
        ) for client in sort(result.client_data; by=client -> client.client_id)
        for feature_idx in eachindex(client.normalization.mean)
    )
    preprocessing_path = _write_interp_csv(
        joinpath(export_dir, "preprocessing.csv"),
        ("client_id", "feature_index", "mean", "std"),
        preprocessing_rows,
    )

    optimization_rows = (
        (
            client.client_id,
            client.instance.capacity,
            join(client.instance.weights, ";"),
            result.objective_sense,
        ) for client in sort(result.client_data; by=client -> client.client_id)
    )
    optimization_path = _write_interp_csv(
        joinpath(export_dir, "client_optimization.csv"),
        ("client_id", "capacity", "weights", "objective_sense"),
        optimization_rows,
    )

    evaluation_client_rows = (
        (
            evaluation.selector,
            client.client_id,
            client.n_test_samples,
            client.lambda,
            client.mse,
            client.absolute_regret,
            client.relative_regret,
        ) for evaluation in (evaluations.spo_plus, evaluations.mse)
        for client in evaluation.clients
    )
    evaluation_clients_path = _write_interp_csv(
        joinpath(export_dir, "evaluation_clients.csv"),
        (
            "selector",
            "client_id",
            "n_test_samples",
            "lambda",
            "mse",
            "absolute_regret",
            "relative_regret",
        ),
        evaluation_client_rows,
    )

    evaluation_aggregate_rows = (
        (
            evaluation.selector,
            metric,
            getfield(evaluation.aggregate, metric).mean,
            getfield(evaluation.aggregate, metric).std,
            getfield(evaluation.aggregate, metric).n,
        ) for evaluation in (evaluations.spo_plus, evaluations.mse)
        for metric in (:mse, :absolute_regret, :relative_regret)
    )
    evaluation_aggregate_path = _write_interp_csv(
        joinpath(export_dir, "evaluation_aggregate.csv"),
        ("selector", "metric", "mean", "std", "n"),
        evaluation_aggregate_rows,
    )

    return (
        dir=export_dir,
        result=result,
        evaluations=evaluations,
        paths=(
            config=config_path,
            artifact=artifact_path,
            calibration_curves=calibration_path,
            selected_lambdas=selected_path,
            split_metadata=split_path,
            preprocessing=preprocessing_path,
            client_optimization=optimization_path,
            evaluation_clients=evaluation_clients_path,
            evaluation_aggregate=evaluation_aggregate_path,
        ),
    )
end
