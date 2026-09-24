using Flux
using LinearAlgebra
using Random
using Statistics

const PJM_INTERP_DFFL_PROTOCOL = :matched_pjm_temporal
const PJM_INTERP_DFFL_SELECTORS = (
    :local_endpoint,
    :federated_endpoint,
    :calibrated_spo,
    :calibrated_mse,
)

Base.@kwdef struct PJMInterpDFFLConfig
    seed::Int = 42
    lambda_grid::Vector{Float64} = collect(0.0:0.05:1.0)
    hidden_dim::Int = 64
    federated_rounds::Int = 10
    federated_local_epochs::Int = 3
    federated_client_fraction::Float64 = 0.4
    local_rounds::Int = 45
    local_epochs::Int = 1
    batch_size::Int = 64
    learning_rate::Float64 = 1e-3
    clip_norm::Float64 = 1.0
    spo_alpha::Float64 = 2.0
    shuffle_batches::Bool = true
end

struct PJMInterpDFFLClientCalibration
    client_id::Int
    lambda_spo_plus::Float64
    lambda_mse::Float64
    spo_plus_losses::Vector{Float64}
    mse_losses::Vector{Float64}
end

struct PJMInterpDFFLTrainingResult
    config::PJMInterpDFFLConfig
    federated_config::PJMBatteryTrainingConfig
    local_config::PJMBatteryTrainingConfig
    federated_endpoint::PJMBatteryFedTrainingResult
    local_endpoint::PJMBatteryLocalTrainingResult
    client_data::Vector{PJMBatteryClientData}
    calibrations::Dict{Int,PJMInterpDFFLClientCalibration}
end

struct PJMInterpDFFLClientEvaluation
    selector::Symbol
    client_id::Int
    client_name::String
    n_test_samples::Int
    lambda::Float64
    mse::Float64
    absolute_regret::Float64
    relative_regret::Float64
end

struct PJMInterpDFFLMetricSummary
    mean::Float64
    std::Float64
    n::Int
end

struct PJMInterpDFFLAggregateEvaluation
    n_clients::Int
    total_test_samples::Int
    mse::PJMInterpDFFLMetricSummary
    absolute_regret::PJMInterpDFFLMetricSummary
    relative_regret::PJMInterpDFFLMetricSummary
end

struct PJMInterpDFFLEvaluationResult
    selector::Symbol
    clients::Vector{PJMInterpDFFLClientEvaluation}
    aggregate::PJMInterpDFFLAggregateEvaluation
end

function _validate_pjm_interp_lambda_grid(lambda_grid::AbstractVector{<:Real})
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

function _validate_pjm_interp_dffl_config(config::PJMInterpDFFLConfig)
    _validate_pjm_interp_lambda_grid(config.lambda_grid)
    config.hidden_dim > 0 || throw(ArgumentError("hidden_dim must be positive"))
    config.federated_rounds > 0 ||
        throw(ArgumentError("federated_rounds must be positive"))
    config.federated_local_epochs > 0 ||
        throw(ArgumentError("federated_local_epochs must be positive"))
    config.federated_client_fraction > 0 ||
        throw(ArgumentError("federated_client_fraction must be positive"))
    config.local_rounds > 0 || throw(ArgumentError("local_rounds must be positive"))
    config.local_epochs > 0 || throw(ArgumentError("local_epochs must be positive"))
    config.batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    isfinite(config.learning_rate) && config.learning_rate > 0 ||
        throw(ArgumentError("learning_rate must be positive and finite"))
    isfinite(config.clip_norm) && config.clip_norm > 0 ||
        throw(ArgumentError("clip_norm must be positive and finite"))
    isfinite(config.spo_alpha) && config.spo_alpha > 1 ||
        throw(ArgumentError("spo_alpha must be finite and greater than one"))
    return config
end

function pjm_interp_dffl_endpoint_configs(config::PJMInterpDFFLConfig)
    _validate_pjm_interp_dffl_config(config)
    federated = PJMBatteryTrainingConfig(
        seed=config.seed,
        hidden_dim=config.hidden_dim,
        rounds=config.federated_rounds,
        local_epochs=config.federated_local_epochs,
        batch_size=config.batch_size,
        client_fraction=config.federated_client_fraction,
        validation_client_fraction=0.4,
        validation_max_samples_per_client=8,
        lambda0=NaN,
        kappa_lambda=0.0,
        per_client_lambda=false,
        lr0=config.learning_rate,
        kappa_lr=0.0,
        per_client_lr=false,
        lr_lambda_alpha=0.0,
        clip_norm=config.clip_norm,
        freeze_tau=1.0,
        freeze_eps=1e-8,
        stop_after_freeze_rounds=3,
        shuffle_batches=config.shuffle_batches,
        use_warm_start=true,
    )
    local_endpoint = PJMBatteryTrainingConfig(
        seed=config.seed,
        hidden_dim=config.hidden_dim,
        rounds=config.local_rounds,
        local_epochs=config.local_epochs,
        batch_size=config.batch_size,
        client_fraction=1.0,
        validation_client_fraction=1.0,
        validation_max_samples_per_client=8,
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
        stop_after_freeze_rounds=3,
        shuffle_batches=config.shuffle_batches,
        use_warm_start=true,
    )
    return (federated=federated, local_endpoint=local_endpoint)
end

function pjm_interpolate_objective_predictions(
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

function pjm_mean_spoplus_loss(
    predictions::AbstractMatrix{<:Real},
    theta_true::AbstractMatrix{<:Real};
    instance::BatteryDispatchInstance,
)
    size(predictions) == size(theta_true) ||
        throw(ArgumentError("prediction and truth shapes must match"))
    size(theta_true, 2) > 0 ||
        throw(ArgumentError("SPO+ calibration requires at least one sample"))

    total_loss = 0.0
    for sample_idx in axes(theta_true, 2)
        true_cost = view(theta_true, :, sample_idx)
        tilted_cost = 2 .* view(predictions, :, sample_idx) .- true_cost
        w_true, _ =
            solve_battery_dispatch(true_cost; instance=instance, sense=:min)
        w_tilt, _ =
            solve_battery_dispatch(tilted_cost; instance=instance, sense=:min)
        total_loss += dot(tilted_cost, w_true .- w_tilt)
    end
    return total_loss / size(theta_true, 2)
end

function pjm_prediction_mse(
    predictions::AbstractArray{<:Real},
    targets::AbstractArray{<:Real},
)
    size(predictions) == size(targets) ||
        throw(ArgumentError("prediction and target shapes must match"))
    isempty(targets) && throw(ArgumentError("MSE calibration requires at least one target"))
    return mean(abs2, predictions .- targets)
end

function pjm_select_interp_lambda(
    local_predictions::AbstractArray{<:Real},
    federated_predictions::AbstractArray{<:Real},
    loss_function;
    lambda_grid::AbstractVector{<:Real}=collect(0.0:0.05:1.0),
)
    _validate_pjm_interp_lambda_grid(lambda_grid)
    size(local_predictions) == size(federated_predictions) ||
        throw(ArgumentError("local and federated prediction shapes must match"))

    losses = Vector{Float64}(undef, length(lambda_grid))
    best_lambda = float(first(lambda_grid))
    best_loss = Inf
    for (idx, lambda) in pairs(lambda_grid)
        predictions = pjm_interpolate_objective_predictions(
            local_predictions,
            federated_predictions,
            lambda,
        )
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
    return (lambda=best_lambda, losses=losses, best_loss=best_loss)
end

function _pjm_interp_local_model(
    result::PJMBatteryLocalTrainingResult,
    client_id::Int,
)
    idx = findfirst(client -> client.client_id == client_id, result.clients)
    isnothing(idx) && throw(ArgumentError("missing local endpoint for client $client_id"))
    return result.clients[idx].model
end

function _calibrate_pjm_interp_dffl(
    federated_endpoint::PJMBatteryFedTrainingResult,
    local_endpoint::PJMBatteryLocalTrainingResult,
    client_data::Vector{PJMBatteryClientData},
    config::PJMInterpDFFLConfig,
)
    calibrations = Dict{Int,PJMInterpDFFLClientCalibration}()
    for client in client_data
        size(client.val_x, 2) > 0 ||
            throw(ArgumentError("client $(client.client_id) has no PJM validation samples"))
        local_predictions = Matrix{Float64}(
            _pjm_interp_local_model(local_endpoint, client.client_id)(client.val_x),
        )
        federated_predictions =
            Matrix{Float64}(federated_endpoint.model(client.val_x))

        spo_selection = pjm_select_interp_lambda(
            local_predictions,
            federated_predictions,
            predictions -> pjm_mean_spoplus_loss(
                predictions,
                client.val_theta_true;
                instance=client.instance,
            );
            lambda_grid=config.lambda_grid,
        )
        mse_selection = pjm_select_interp_lambda(
            local_predictions,
            federated_predictions,
            predictions -> pjm_prediction_mse(predictions, client.val_theta_true);
            lambda_grid=config.lambda_grid,
        )
        calibrations[client.client_id] = PJMInterpDFFLClientCalibration(
            client.client_id,
            spo_selection.lambda,
            mse_selection.lambda,
            spo_selection.losses,
            mse_selection.losses,
        )
    end
    return calibrations
end

"""
    train_pjm_interp_dffl(dataset; config=PJMInterpDFFLConfig())

Train independent local SPO+ endpoints and one federated SPO+ endpoint on the
PJM training period, starting every endpoint from the same seed-specific model.
The client-specific interpolation weights are selected only on the chronological
PJM validation period.
"""
function train_pjm_interp_dffl(
    dataset::PJMBatteryDataset;
    config::PJMInterpDFFLConfig=PJMInterpDFFLConfig(),
)
    _validate_pjm_interp_dffl_config(config)
    endpoint_configs = pjm_interp_dffl_endpoint_configs(config)
    shared_init_model = build_pjm_battery_model(
        length(PJM_FEATURE_COLUMNS),
        length(PJM_TARGET_COLUMNS);
        hidden_dim=config.hidden_dim,
        rng=MersenneTwister(config.seed),
    )
    objective = PJMSPOPlusObjective(; α=config.spo_alpha)
    federated_endpoint = fed_pjm_battery(
        dataset;
        objective=objective,
        config=endpoint_configs.federated,
        model=shared_init_model,
    )
    local_endpoint = local_pjm_battery(
        dataset;
        objective=objective,
        config=endpoint_configs.local_endpoint,
        model=shared_init_model,
    )
    client_data = federated_endpoint.client_data
    calibrations = _calibrate_pjm_interp_dffl(
        federated_endpoint,
        local_endpoint,
        client_data,
        config,
    )
    return PJMInterpDFFLTrainingResult(
        config,
        endpoint_configs.federated,
        endpoint_configs.local_endpoint,
        federated_endpoint,
        local_endpoint,
        client_data,
        calibrations,
    )
end

function _pjm_interp_selector_lambda(
    result::PJMInterpDFFLTrainingResult,
    client_id::Int,
    selector::Symbol,
)
    selector === :local_endpoint && return 0.0
    selector === :federated_endpoint && return 1.0
    calibration = get(
        result.calibrations,
        client_id,
        nothing,
    )
    isnothing(calibration) &&
        throw(ArgumentError("missing interpolation calibration for client $client_id"))
    selector === :calibrated_spo && return calibration.lambda_spo_plus
    selector === :calibrated_mse && return calibration.lambda_mse
    throw(
        ArgumentError(
            "selector must be one of $(collect(PJM_INTERP_DFFL_SELECTORS)), got $selector",
        ),
    )
end

function _evaluate_pjm_interp_client(
    result::PJMInterpDFFLTrainingResult,
    client::PJMBatteryClientData,
    selector::Symbol,
)
    n_test_samples = size(client.test_x, 2)
    lambda = _pjm_interp_selector_lambda(result, client.client_id, selector)
    if n_test_samples == 0
        return PJMInterpDFFLClientEvaluation(
            selector,
            client.client_id,
            client.client_name,
            0,
            lambda,
            NaN,
            NaN,
            NaN,
        )
    end

    local_predictions = Matrix{Float64}(
        _pjm_interp_local_model(result.local_endpoint, client.client_id)(client.test_x),
    )
    federated_predictions =
        Matrix{Float64}(result.federated_endpoint.model(client.test_x))
    predictions = pjm_interpolate_objective_predictions(
        local_predictions,
        federated_predictions,
        lambda,
    )

    regrets = Vector{Float64}(undef, n_test_samples)
    true_objectives = Vector{Float64}(undef, n_test_samples)
    for sample_idx in 1:n_test_samples
        pred_decision, _ = solve_battery_dispatch(
            view(predictions, :, sample_idx);
            instance=client.instance,
            sense=:min,
        )
        _, true_objective = solve_battery_dispatch(
            view(client.test_theta_true, :, sample_idx);
            instance=client.instance,
            sense=:min,
        )
        pred_objective =
            dot(view(client.test_theta_true, :, sample_idx), pred_decision)
        regrets[sample_idx] = pred_objective - true_objective
        true_objectives[sample_idx] = true_objective
    end
    absolute_regret = mean(regrets)
    denominator = mean(abs.(true_objectives))
    relative_regret = denominator <= 0 ? NaN : absolute_regret / denominator
    return PJMInterpDFFLClientEvaluation(
        selector,
        client.client_id,
        client.client_name,
        n_test_samples,
        lambda,
        pjm_prediction_mse(predictions, client.test_theta_true),
        absolute_regret,
        relative_regret,
    )
end

function _pjm_interp_metric_summary(values)
    finite_values = Float64[float(value) for value in values if isfinite(value)]
    isempty(finite_values) && return PJMInterpDFFLMetricSummary(NaN, NaN, 0)
    return PJMInterpDFFLMetricSummary(
        mean(finite_values),
        length(finite_values) == 1 ? 0.0 : std(finite_values; corrected=false),
        length(finite_values),
    )
end

function evaluate_pjm_interp_dffl(
    result::PJMInterpDFFLTrainingResult;
    selector::Symbol=:calibrated_spo,
)
    selector in PJM_INTERP_DFFL_SELECTORS || throw(
        ArgumentError(
            "selector must be one of $(collect(PJM_INTERP_DFFL_SELECTORS)), got $selector",
        ),
    )
    clients = [
        _evaluate_pjm_interp_client(result, client, selector) for
        client in result.client_data
    ]
    aggregate = PJMInterpDFFLAggregateEvaluation(
        length(clients),
        sum(client.n_test_samples for client in clients),
        _pjm_interp_metric_summary(client.mse for client in clients),
        _pjm_interp_metric_summary(client.absolute_regret for client in clients),
        _pjm_interp_metric_summary(client.relative_regret for client in clients),
    )
    return PJMInterpDFFLEvaluationResult(selector, clients, aggregate)
end

function evaluate_all_pjm_interp_dffl(result::PJMInterpDFFLTrainingResult)
    return Dict(
        selector => evaluate_pjm_interp_dffl(result; selector=selector) for
        selector in PJM_INTERP_DFFL_SELECTORS
    )
end
