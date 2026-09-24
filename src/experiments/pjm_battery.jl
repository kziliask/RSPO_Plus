using Dates
using SHA
using Statistics
using TOML

struct PJMBatteryMetricSummary
    mean::Float64
    std::Float64
    n::Int
end

struct PJMBatteryAggregateSchedulerDiagnostics
    final_lambda::PJMBatteryMetricSummary
    final_lr::PJMBatteryMetricSummary
    final_bound::PJMBatteryMetricSummary
    freeze_round::PJMBatteryMetricSummary
    frozen_fraction::Float64
end

struct PJMBatteryClientEvaluation
    client_id::Int
    client_name::String
    n_test_samples::Int
    mse::Float64
    absolute_regret::Float64
    relative_regret::Float64
    regularized_decision_gap::Float64
    final_lambda::Float64
    final_lr::Float64
    final_bound::Float64
    frozen_after_round::Union{Nothing,Int}
end

struct PJMBatteryAggregateEvaluation
    n_clients::Int
    total_test_samples::Int
    mse::PJMBatteryMetricSummary
    absolute_regret::PJMBatteryMetricSummary
    relative_regret::PJMBatteryMetricSummary
    regularized_decision_gap::PJMBatteryMetricSummary
    scheduler::PJMBatteryAggregateSchedulerDiagnostics
end

struct PJMBatteryEvaluationResult
    method::Symbol
    clients::Vector{PJMBatteryClientEvaluation}
    aggregate::PJMBatteryAggregateEvaluation
end

struct PJMBatteryTwoStageEvaluationResult
    warm_start::PJMBatteryEvaluationResult
    personalization::PJMBatteryEvaluationResult
end

function _pjm_final_trace_value(values::AbstractVector{<:Real})
    return isempty(values) ? NaN : float(values[end])
end

function _pjm_summarize_metric(values)
    finite_values = Float64[float(value) for value in values if isfinite(value)]
    if isempty(finite_values)
        return PJMBatteryMetricSummary(NaN, NaN, 0)
    end
    return PJMBatteryMetricSummary(
        mean(finite_values),
        length(finite_values) == 1 ? 0.0 : std(finite_values; corrected=false),
        length(finite_values),
    )
end

function _pjm_as_float_matrix(values::AbstractMatrix{<:Real})
    return Matrix{Float64}(values)
end

function _pjm_mean_regularized_decision_gap(
    theta_pred::AbstractMatrix{<:Real},
    theta_true::AbstractMatrix{<:Real},
    instance::BatteryDispatchInstance;
    lambda::Real,
    use_warm_start::Bool=true,
)
    if !isfinite(lambda) || lambda <= 0
        return NaN
    end

    w_hat = projection_optimizer(
        theta_pred;
        instance=instance,
        cache=init_battery_projection_cache(instance; lambda=lambda),
        lambda=lambda,
        use_warm_start=use_warm_start,
    )
    w_star_reg = projection_optimizer(
        theta_true;
        instance=instance,
        cache=init_battery_projection_cache(instance; lambda=lambda),
        lambda=lambda,
        use_warm_start=use_warm_start,
    )
    gaps = sqrt.(sum(abs2, w_hat .- w_star_reg; dims=1))
    return mean(vec(gaps))
end

function _evaluate_pjm_client_metrics(
    client::PJMBatteryClientData,
    theta_pred::AbstractMatrix{<:Real},
    trace::PJMBatterySchedulerTrace;
    include_regularized_decision_gap::Bool=true,
    use_warm_start::Bool=true,
)
    theta_true = client.test_theta_true
    size(theta_pred) == size(theta_true) || throw(
        ArgumentError(
            "model output shape $(size(theta_pred)) does not match test theta shape $(size(theta_true)) for client $(client.client_id)",
        ),
    )

    final_lambda = _pjm_final_trace_value(trace.lambda_values)
    final_lr = _pjm_final_trace_value(trace.lr_values)
    final_bound = _pjm_final_trace_value(trace.bound_values)
    n_test_samples = size(theta_true, 2)

    if n_test_samples == 0
        return PJMBatteryClientEvaluation(
            client.client_id,
            client.client_name,
            0,
            NaN,
            NaN,
            NaN,
            NaN,
            final_lambda,
            final_lr,
            final_bound,
            trace.frozen_after_round,
        )
    end

    theta_pred_matrix = _pjm_as_float_matrix(theta_pred)
    regrets = Vector{Float64}(undef, n_test_samples)
    true_objectives = Vector{Float64}(undef, n_test_samples)

    for sample_idx in 1:n_test_samples
        pred_decision, _ = solve_battery_dispatch(
            view(theta_pred_matrix, :, sample_idx);
            instance=client.instance,
            sense=:min,
        )
        _, true_objective = solve_battery_dispatch(
            view(theta_true, :, sample_idx);
            instance=client.instance,
            sense=:min,
        )
        pred_objective = dot(view(theta_true, :, sample_idx), pred_decision)
        regrets[sample_idx] = pred_objective - true_objective
        true_objectives[sample_idx] = true_objective
    end

    absolute_regret = mean(regrets)
    denominator = mean(abs.(true_objectives))
    relative_regret = denominator <= 0 ? NaN : absolute_regret / denominator
    regularized_decision_gap =
        include_regularized_decision_gap ? _pjm_mean_regularized_decision_gap(
            theta_pred_matrix,
            theta_true,
            client.instance;
            lambda=final_lambda,
            use_warm_start=use_warm_start,
        ) : NaN

    return PJMBatteryClientEvaluation(
        client.client_id,
        client.client_name,
        n_test_samples,
        mean(abs2.(theta_pred_matrix .- theta_true)),
        absolute_regret,
        relative_regret,
        regularized_decision_gap,
        final_lambda,
        final_lr,
        final_bound,
        trace.frozen_after_round,
    )
end

function _aggregate_pjm_client_evaluations(clients::Vector{PJMBatteryClientEvaluation})
    freeze_rounds = [float(something(client.frozen_after_round, NaN)) for client in clients]
    frozen_fraction =
        isempty(clients) ? NaN : mean(
            [isnothing(client.frozen_after_round) ? 0.0 : 1.0 for client in clients],
        )

    return PJMBatteryAggregateEvaluation(
        length(clients),
        sum(client.n_test_samples for client in clients),
        _pjm_summarize_metric(client.mse for client in clients),
        _pjm_summarize_metric(client.absolute_regret for client in clients),
        _pjm_summarize_metric(client.relative_regret for client in clients),
        _pjm_summarize_metric(client.regularized_decision_gap for client in clients),
        PJMBatteryAggregateSchedulerDiagnostics(
            _pjm_summarize_metric(client.final_lambda for client in clients),
            _pjm_summarize_metric(client.final_lr for client in clients),
            _pjm_summarize_metric(client.final_bound for client in clients),
            _pjm_summarize_metric(freeze_rounds),
            frozen_fraction,
        ),
    )
end

function _evaluate_pjm_client_collection(
    method::Symbol,
    client_data::Vector{PJMBatteryClientData},
    models::AbstractVector,
    traces::AbstractVector{PJMBatterySchedulerTrace};
    include_regularized_decision_gap::Bool=true,
    use_warm_start::Bool=true,
)
    length(models) == length(client_data) || throw(
        ArgumentError("expected $(length(client_data)) models, got $(length(models))"),
    )
    length(traces) == length(client_data) || throw(
        ArgumentError("expected $(length(client_data)) traces, got $(length(traces))"),
    )

    client_metrics = Vector{PJMBatteryClientEvaluation}(undef, length(client_data))
    for idx in eachindex(client_data)
        client = client_data[idx]
        theta_pred =
            size(client.test_x, 2) == 0 ? zeros(Float64, client.instance.horizon, 0) :
            models[idx](client.test_x)
        client_metrics[idx] = _evaluate_pjm_client_metrics(
            client,
            theta_pred,
            traces[idx];
            include_regularized_decision_gap=include_regularized_decision_gap,
            use_warm_start=use_warm_start,
        )
    end

    return PJMBatteryEvaluationResult(
        method,
        client_metrics,
        _aggregate_pjm_client_evaluations(client_metrics),
    )
end

function evaluate_fed_pjm_battery(
    result::PJMBatteryFedTrainingResult;
    include_regularized_decision_gap::Bool=true,
    use_warm_start::Bool=true,
)
    traces = fill(result.trace, length(result.client_data))
    models = fill(result.model, length(result.client_data))
    return _evaluate_pjm_client_collection(
        result.method,
        result.client_data,
        models,
        traces;
        include_regularized_decision_gap=include_regularized_decision_gap,
        use_warm_start=use_warm_start,
    )
end

function evaluate_local_pjm_battery(
    result::PJMBatteryLocalTrainingResult,
    client_data::Vector{PJMBatteryClientData};
    include_regularized_decision_gap::Bool=true,
    use_warm_start::Bool=true,
)
    models = [client.model for client in result.clients]
    traces = [client.trace for client in result.clients]
    return _evaluate_pjm_client_collection(
        result.method,
        client_data,
        models,
        traces;
        include_regularized_decision_gap=include_regularized_decision_gap,
        use_warm_start=use_warm_start,
    )
end

function evaluate_local_pjm_battery(
    result::PJMBatteryLocalTrainingResult,
    dataset::PJMBatteryDataset;
    kwargs...,
)
    return evaluate_local_pjm_battery(result, prepare_pjm_battery_client_data(dataset); kwargs...)
end

function evaluate_pjm_battery_two_stage(
    result::PJMBatteryTwoStageTrainingResult,
    dataset::PJMBatteryDataset;
    include_regularized_decision_gap::Bool=true,
    use_warm_start::Bool=true,
)
    return PJMBatteryTwoStageEvaluationResult(
        evaluate_fed_pjm_battery(
            result.warm_start;
            include_regularized_decision_gap=include_regularized_decision_gap,
            use_warm_start=use_warm_start,
        ),
        evaluate_local_pjm_battery(
            result.personalization,
            dataset;
            include_regularized_decision_gap=include_regularized_decision_gap,
            use_warm_start=use_warm_start,
        ),
    )
end

_pjm_result_method(result::PJMBatteryFedTrainingResult) = result.method
_pjm_result_method(result::PJMBatteryLocalTrainingResult) = result.method

_pjm_slugify(value) = replace(string(value), "." => "p", "-" => "m", "," => "", " " => "")

function _pjm_path_component(value::AbstractString; max_length::Int=120)
    length(value) <= max_length && return String(value)
    digest = bytes2hex(SHA.sha1(value))[1:12]
    prefix_length = max(max_length - length(digest) - 2, 16)
    return string(first(value, prefix_length), "__", digest)
end

function _pjm_config_slug(config)
    parts = [
        string(name, "-", _pjm_slugify(getfield(config, name)))
        for name in fieldnames(typeof(config)) if !isnothing(getfield(config, name))
    ]
    return join(parts, "_")
end

function default_pjm_battery_config_id(
    dataset_config::PJMBatteryDatasetConfig,
    training_config::PJMBatteryTrainingConfig,
)
    return string(_pjm_config_slug(dataset_config), "__", _pjm_config_slug(training_config))
end

function default_pjm_battery_seed_tag(
    dataset_config::PJMBatteryDatasetConfig,
    training_config::PJMBatteryTrainingConfig,
)
    return "train$(training_config.seed)"
end

function _pjm_config_to_dict(config)
    return Dict(
        string(name) => let value = getfield(config, name)
            value isa Tuple ? collect(value) : value
        end for name in fieldnames(typeof(config)) if !isnothing(getfield(config, name))
    )
end

function _write_pjm_config_snapshot(
    path::AbstractString,
    method::Symbol,
    dataset_config::PJMBatteryDatasetConfig,
    training_config::PJMBatteryTrainingConfig,
    config_id::AbstractString,
    seed_tag::AbstractString,
    objective_metadata::Union{Nothing,AbstractDict}=nothing,
)
    payload = Dict(
        "run" => Dict(
            "method" => string(method),
            "config_id" => config_id,
            "seed_tag" => seed_tag,
            "exported_at" => string(Dates.now()),
        ),
        "dataset" => _pjm_config_to_dict(dataset_config),
        "training" => _pjm_config_to_dict(training_config),
    )
    if !isnothing(objective_metadata)
        payload["objective"] =
            Dict(string(key) => value for (key, value) in pairs(objective_metadata))
    end

    open(path, "w") do io
        TOML.print(io, payload)
    end
    return path
end

function _write_pjm_training_rounds(path::AbstractString, result::PJMBatteryFedTrainingResult)
    header = ["scope", "client_id", "round", "loss", "lambda", "lr", "bound", "selected_clients"]
    rows = (
        [
            "global",
            "0",
            string(round),
            string(result.round_losses[round]),
            string(result.trace.lambda_values[round]),
            string(result.trace.lr_values[round]),
            string(result.trace.bound_values[round]),
            join(string.(result.selected_clients[round]), "|"),
        ] for round in eachindex(result.round_losses)
    )
    return _write_pjm_csv_rows(path, header, rows)
end

function _write_pjm_training_rounds(path::AbstractString, result::PJMBatteryLocalTrainingResult)
    header = ["scope", "client_id", "round", "loss", "lambda", "lr", "bound", "selected_clients"]
    rows = (
        [
            "client",
            string(client_result.client_id),
            string(round),
            string(client_result.round_losses[round]),
            string(client_result.trace.lambda_values[round]),
            string(client_result.trace.lr_values[round]),
            string(client_result.trace.bound_values[round]),
            "",
        ] for client_result in result.clients for round in eachindex(client_result.round_losses)
    )
    return _write_pjm_csv_rows(path, header, rows)
end

function _write_pjm_scheduler_summary(
    path::AbstractString,
    result::PJMBatteryFedTrainingResult,
)
    header = ["scope", "client_id", "frozen_after_round", "final_lambda", "final_lr", "final_bound", "lambda0"]
    pcl = result.trace.per_client_lambda0
    rows = if isnothing(pcl)
        [[
            "global",
            "0",
            isnothing(result.trace.frozen_after_round) ? "" : string(result.trace.frozen_after_round),
            string(_pjm_final_trace_value(result.trace.lambda_values)),
            string(_pjm_final_trace_value(result.trace.lr_values)),
            string(_pjm_final_trace_value(result.trace.bound_values)),
            "",
        ]]
    else
        [[
            "global",
            "0",
            isnothing(result.trace.frozen_after_round) ? "" : string(result.trace.frozen_after_round),
            string(_pjm_final_trace_value(result.trace.lambda_values)),
            string(_pjm_final_trace_value(result.trace.lr_values)),
            string(_pjm_final_trace_value(result.trace.bound_values)),
            join(string.(pcl), ";"),
        ]]
    end
    return _write_pjm_csv_rows(path, header, rows)
end

function _write_pjm_scheduler_summary(
    path::AbstractString,
    result::PJMBatteryLocalTrainingResult,
)
    header = ["scope", "client_id", "frozen_after_round", "final_lambda", "final_lr", "final_bound", "lambda0"]
    rows = (
        [
            "client",
            string(client_result.client_id),
            isnothing(client_result.trace.frozen_after_round) ? "" :
            string(client_result.trace.frozen_after_round),
            string(_pjm_final_trace_value(client_result.trace.lambda_values)),
            string(_pjm_final_trace_value(client_result.trace.lr_values)),
            string(_pjm_final_trace_value(client_result.trace.bound_values)),
            isnothing(client_result.trace.per_client_lambda0) ? "" :
            join(string.(client_result.trace.per_client_lambda0), ";"),
        ] for client_result in result.clients
    )
    return _write_pjm_csv_rows(path, header, rows)
end

function _write_pjm_evaluation_clients(
    path::AbstractString,
    evaluation::PJMBatteryEvaluationResult,
)
    header = [
        "client_id",
        "client_name",
        "n_test_samples",
        "mse",
        "absolute_regret",
        "relative_regret",
        "regularized_decision_gap",
        "final_lambda",
        "final_lr",
        "final_bound",
        "frozen_after_round",
    ]
    rows = (
        [
            string(client.client_id),
            client.client_name,
            string(client.n_test_samples),
            string(client.mse),
            string(client.absolute_regret),
            string(client.relative_regret),
            string(client.regularized_decision_gap),
            string(client.final_lambda),
            string(client.final_lr),
            string(client.final_bound),
            isnothing(client.frozen_after_round) ? "" : string(client.frozen_after_round),
        ] for client in evaluation.clients
    )
    return _write_pjm_csv_rows(path, header, rows)
end

function _write_pjm_evaluation_aggregate(
    path::AbstractString,
    evaluation::PJMBatteryEvaluationResult,
)
    aggregate = evaluation.aggregate
    scheduler = aggregate.scheduler
    rows = [
        ["mse", string(aggregate.mse.mean), string(aggregate.mse.std), string(aggregate.mse.n)],
        [
            "absolute_regret",
            string(aggregate.absolute_regret.mean),
            string(aggregate.absolute_regret.std),
            string(aggregate.absolute_regret.n),
        ],
        [
            "relative_regret",
            string(aggregate.relative_regret.mean),
            string(aggregate.relative_regret.std),
            string(aggregate.relative_regret.n),
        ],
        [
            "regularized_decision_gap",
            string(aggregate.regularized_decision_gap.mean),
            string(aggregate.regularized_decision_gap.std),
            string(aggregate.regularized_decision_gap.n),
        ],
        [
            "final_lambda",
            string(scheduler.final_lambda.mean),
            string(scheduler.final_lambda.std),
            string(scheduler.final_lambda.n),
        ],
        [
            "final_lr",
            string(scheduler.final_lr.mean),
            string(scheduler.final_lr.std),
            string(scheduler.final_lr.n),
        ],
        [
            "final_bound",
            string(scheduler.final_bound.mean),
            string(scheduler.final_bound.std),
            string(scheduler.final_bound.n),
        ],
        [
            "freeze_round",
            string(scheduler.freeze_round.mean),
            string(scheduler.freeze_round.std),
            string(scheduler.freeze_round.n),
        ],
        [
            "frozen_fraction",
            string(scheduler.frozen_fraction),
            "0.0",
            string(aggregate.n_clients),
        ],
    ]
    return _write_pjm_csv_rows(path, ["metric", "mean", "std", "n"], rows)
end

function export_pjm_battery_run(
    result::Union{PJMBatteryFedTrainingResult,PJMBatteryLocalTrainingResult},
    dataset::PJMBatteryDataset;
    config::PJMBatteryTrainingConfig,
    evaluation::Union{Nothing,PJMBatteryEvaluationResult}=nothing,
    objective_metadata::Union{Nothing,AbstractDict}=nothing,
    root::AbstractString=joinpath("results", "experiment3", "pjm_battery"),
    config_id::Union{Nothing,AbstractString}=nothing,
    seed_tag::Union{Nothing,AbstractString}=nothing,
)
    method = _pjm_result_method(result)
    resolved_evaluation =
        if isnothing(evaluation)
            result isa PJMBatteryFedTrainingResult ? evaluate_fed_pjm_battery(result) :
            evaluate_local_pjm_battery(result, dataset)
        else
            evaluation
        end
    resolved_config_id =
        something(config_id, default_pjm_battery_config_id(dataset.config, config))
    resolved_seed_tag =
        something(seed_tag, default_pjm_battery_seed_tag(dataset.config, config))
    safe_config_id = _pjm_path_component(resolved_config_id)
    safe_seed_tag = _pjm_path_component(resolved_seed_tag; max_length=64)

    export_dir = joinpath(root, string(method), safe_config_id, safe_seed_tag)
    mkpath(export_dir)

    paths = (
        config=_write_pjm_config_snapshot(
            joinpath(export_dir, "config.toml"),
            method,
            dataset.config,
            config,
            safe_config_id,
            safe_seed_tag,
            objective_metadata,
        ),
        dataset_metadata=export_pjm_dataset_metadata(
            dataset.metadata,
            joinpath(export_dir, "dataset_metadata.csv"),
        ),
        partition_manifest=export_pjm_partition_manifest(
            dataset,
            joinpath(export_dir, "partition_manifest.csv"),
        ),
        training_rounds=_write_pjm_training_rounds(joinpath(export_dir, "training_rounds.csv"), result),
        scheduler_summary=_write_pjm_scheduler_summary(
            joinpath(export_dir, "scheduler_summary.csv"),
            result,
        ),
        evaluation_clients=_write_pjm_evaluation_clients(
            joinpath(export_dir, "evaluation_clients.csv"),
            resolved_evaluation,
        ),
        evaluation_aggregate=_write_pjm_evaluation_aggregate(
            joinpath(export_dir, "evaluation_aggregate.csv"),
            resolved_evaluation,
        ),
    )

    return (; dir=export_dir, method=method, evaluation=resolved_evaluation, paths...)
end

function export_pjm_battery_two_stage_run(
    result::PJMBatteryTwoStageTrainingResult,
    dataset::PJMBatteryDataset;
    warm_start_config::PJMBatteryTrainingConfig,
    personalization_config::PJMBatteryTrainingConfig,
    evaluation::Union{Nothing,PJMBatteryTwoStageEvaluationResult}=nothing,
    objective_metadata::Union{Nothing,AbstractDict}=nothing,
    root::AbstractString=joinpath("results", "experiment3", "pjm_battery"),
    warm_start_config_id::Union{Nothing,AbstractString}=nothing,
    warm_start_seed_tag::Union{Nothing,AbstractString}=nothing,
    personalization_config_id::Union{Nothing,AbstractString}=nothing,
    personalization_seed_tag::Union{Nothing,AbstractString}=nothing,
)
    resolved_evaluation =
        isnothing(evaluation) ? evaluate_pjm_battery_two_stage(result, dataset) : evaluation

    warm_start_export = export_pjm_battery_run(
        result.warm_start,
        dataset;
        config=warm_start_config,
        evaluation=resolved_evaluation.warm_start,
        objective_metadata=objective_metadata,
        root=root,
        config_id=something(
            warm_start_config_id,
            string(
                "warm_start__",
                default_pjm_battery_config_id(dataset.config, warm_start_config),
            ),
        ),
        seed_tag=something(
            warm_start_seed_tag,
            default_pjm_battery_seed_tag(dataset.config, warm_start_config),
        ),
    )
    personalization_export = export_pjm_battery_run(
        result.personalization,
        dataset;
        config=personalization_config,
        evaluation=resolved_evaluation.personalization,
        objective_metadata=objective_metadata,
        root=root,
        config_id=something(
            personalization_config_id,
            string(
                "personalization__",
                default_pjm_battery_config_id(dataset.config, personalization_config),
            ),
        ),
        seed_tag=something(
            personalization_seed_tag,
            default_pjm_battery_seed_tag(dataset.config, personalization_config),
        ),
    )

    return (
        warm_start=warm_start_export,
        personalization=personalization_export,
        evaluation=resolved_evaluation,
    )
end
