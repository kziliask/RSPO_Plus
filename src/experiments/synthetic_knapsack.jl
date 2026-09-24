using Dates
using SHA
using Statistics
using TOML

struct MetricSummary
    mean::Float64
    std::Float64
    n::Int
end

struct AggregateSchedulerDiagnostics
    final_lambda::MetricSummary
    final_lr::MetricSummary
    final_bound::MetricSummary
    freeze_round::MetricSummary
    frozen_fraction::Float64
end

struct SyntheticKnapsackClientEvaluation
    client_id::Int
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

struct SyntheticKnapsackAggregateEvaluation
    n_clients::Int
    total_test_samples::Int
    mse::MetricSummary
    absolute_regret::MetricSummary
    relative_regret::MetricSummary
    regularized_decision_gap::MetricSummary
    scheduler::AggregateSchedulerDiagnostics
end

struct SyntheticKnapsackEvaluationResult
    method::Symbol
    clients::Vector{SyntheticKnapsackClientEvaluation}
    aggregate::SyntheticKnapsackAggregateEvaluation
end

struct SyntheticKnapsackTwoStageEvaluationResult
    warm_start::SyntheticKnapsackEvaluationResult
    personalization::SyntheticKnapsackEvaluationResult
end

function _final_trace_value(values::AbstractVector{<:Real})
    return isempty(values) ? NaN : float(values[end])
end

function _summarize_metric(values)
    finite_values = Float64[float(value) for value in values if isfinite(value)]
    if isempty(finite_values)
        return MetricSummary(NaN, NaN, 0)
    end

    return MetricSummary(
        mean(finite_values),
        length(finite_values) == 1 ? 0.0 : std(finite_values; corrected=false),
        length(finite_values),
    )
end

function _as_float_matrix(values::AbstractMatrix{<:Real})
    return Matrix{Float64}(values)
end

function _mean_regularized_decision_gap(
    theta_pred::AbstractMatrix{<:Real},
    theta_true::AbstractMatrix{<:Real},
    instance::KnapsackInstance;
    lambda::Real,
    use_warm_start::Bool=true,
)
    if !isfinite(lambda) || lambda <= 0
        return NaN
    end

    w_hat = projection_optimizer(
        theta_pred;
        instance=instance,
        cache=init_knapsack_projection_cache(instance; lambda=lambda),
        lambda=lambda,
        use_warm_start=use_warm_start,
    )
    w_star_reg = projection_optimizer(
        theta_true;
        instance=instance,
        cache=init_knapsack_projection_cache(instance; lambda=lambda),
        lambda=lambda,
        use_warm_start=use_warm_start,
    )
    gaps = sqrt.(sum(abs2, w_hat .- w_star_reg; dims=1))
    return mean(vec(gaps))
end

function _evaluate_client_metrics(
    client::SyntheticKnapsackClientData,
    theta_pred::AbstractMatrix{<:Real},
    trace::SchedulerTrace;
    include_regularized_decision_gap::Bool=true,
    use_warm_start::Bool=true,
)
    theta_true = client.test_theta_true
    size(theta_pred) == size(theta_true) || throw(
        ArgumentError(
            "model output shape $(size(theta_pred)) does not match test theta shape $(size(theta_true)) for client $(client.client_id)",
        ),
    )

    final_lambda = _final_trace_value(trace.lambda_values)
    final_lr = _final_trace_value(trace.lr_values)
    final_bound = _final_trace_value(trace.bound_values)
    n_test_samples = size(theta_true, 2)

    if n_test_samples == 0
        return SyntheticKnapsackClientEvaluation(
            client.client_id,
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

    theta_pred_matrix = _as_float_matrix(theta_pred)
    regrets = Vector{Float64}(undef, n_test_samples)
    true_objectives = Vector{Float64}(undef, n_test_samples)

    for sample_idx in 1:n_test_samples
        pred_decision, _ = solve_fractional_knapsack(
            view(theta_pred_matrix, :, sample_idx);
            instance=client.instance,
            sense=:min,
        )
        _, true_objective = solve_fractional_knapsack(
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
        include_regularized_decision_gap ? _mean_regularized_decision_gap(
            theta_pred_matrix,
            theta_true,
            client.instance;
            lambda=final_lambda,
            use_warm_start=use_warm_start,
        ) : NaN

    return SyntheticKnapsackClientEvaluation(
        client.client_id,
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

function _aggregate_client_evaluations(clients::Vector{SyntheticKnapsackClientEvaluation})
    freeze_rounds = [float(something(client.frozen_after_round, NaN)) for client in clients]
    frozen_fraction =
        isempty(clients) ? NaN : mean(
            [isnothing(client.frozen_after_round) ? 0.0 : 1.0 for client in clients],
        )

    return SyntheticKnapsackAggregateEvaluation(
        length(clients),
        sum(client.n_test_samples for client in clients),
        _summarize_metric(client.mse for client in clients),
        _summarize_metric(client.absolute_regret for client in clients),
        _summarize_metric(client.relative_regret for client in clients),
        _summarize_metric(client.regularized_decision_gap for client in clients),
        AggregateSchedulerDiagnostics(
            _summarize_metric(client.final_lambda for client in clients),
            _summarize_metric(client.final_lr for client in clients),
            _summarize_metric(client.final_bound for client in clients),
            _summarize_metric(freeze_rounds),
            frozen_fraction,
        ),
    )
end

function _evaluate_client_collection(
    method::Symbol,
    client_data::Vector{SyntheticKnapsackClientData},
    models::AbstractVector,
    traces::AbstractVector{SchedulerTrace};
    include_regularized_decision_gap::Bool=true,
    use_warm_start::Bool=true,
)
    length(models) == length(client_data) ||
        throw(ArgumentError("expected $(length(client_data)) models, got $(length(models))"))
    length(traces) == length(client_data) ||
        throw(ArgumentError("expected $(length(client_data)) traces, got $(length(traces))"))

    client_metrics = Vector{SyntheticKnapsackClientEvaluation}(undef, length(client_data))
    for idx in eachindex(client_data)
        client = client_data[idx]
        theta_pred =
            size(client.test_x, 2) == 0 ? zeros(Float64, length(client.instance.weights), 0) :
            models[idx](client.test_x)
        client_metrics[idx] = _evaluate_client_metrics(
            client,
            theta_pred,
            traces[idx];
            include_regularized_decision_gap=include_regularized_decision_gap,
            use_warm_start=use_warm_start,
        )
    end

    return SyntheticKnapsackEvaluationResult(
        method,
        client_metrics,
        _aggregate_client_evaluations(client_metrics),
    )
end

function evaluate_fed_training(
    result::FedTrainingResult;
    include_regularized_decision_gap::Bool=true,
    use_warm_start::Bool=true,
)
    traces = fill(result.trace, length(result.client_data))
    models = fill(result.model, length(result.client_data))
    return _evaluate_client_collection(
        result.method,
        result.client_data,
        models,
        traces;
        include_regularized_decision_gap=include_regularized_decision_gap,
        use_warm_start=use_warm_start,
    )
end

function evaluate_fed_training(
    result::FedTrainingResult,
    dataset::SyntheticKnapsackDataset;
    normalize_x::Bool=true,
    include_regularized_decision_gap::Bool=true,
    use_warm_start::Bool=true,
)
    client_data = prepare_synthetic_knapsack_client_data(dataset; normalize_x=normalize_x)
    traces = fill(result.trace, length(client_data))
    models = fill(result.model, length(client_data))
    return _evaluate_client_collection(
        result.method,
        client_data,
        models,
        traces;
        include_regularized_decision_gap=include_regularized_decision_gap,
        use_warm_start=use_warm_start,
    )
end

function evaluate_local_training(
    result::LocalTrainingResult,
    client_data::Vector{SyntheticKnapsackClientData};
    include_regularized_decision_gap::Bool=true,
    use_warm_start::Bool=true,
)
    client_result_map = Dict(client_result.client_id => client_result for client_result in result.clients)
    length(client_result_map) == length(result.clients) ||
        throw(ArgumentError("duplicate client ids found in local result"))

    models = Vector{Any}(undef, length(client_data))
    traces = Vector{SchedulerTrace}(undef, length(client_data))
    for (idx, client) in pairs(client_data)
        haskey(client_result_map, client.client_id) ||
            throw(ArgumentError("missing local result for client $(client.client_id)"))
        client_result = client_result_map[client.client_id]
        models[idx] = client_result.model
        traces[idx] = client_result.trace
    end

    return _evaluate_client_collection(
        result.method,
        client_data,
        models,
        traces;
        include_regularized_decision_gap=include_regularized_decision_gap,
        use_warm_start=use_warm_start,
    )
end

function evaluate_local_training(
    result::LocalTrainingResult,
    dataset::SyntheticKnapsackDataset;
    normalize_x::Bool=true,
    include_regularized_decision_gap::Bool=true,
    use_warm_start::Bool=true,
)
    client_data = prepare_synthetic_knapsack_client_data(dataset; normalize_x=normalize_x)
    return evaluate_local_training(
        result,
        client_data;
        include_regularized_decision_gap=include_regularized_decision_gap,
        use_warm_start=use_warm_start,
    )
end

function evaluate_fed_rspo_plus(args...; kwargs...)
    return evaluate_fed_training(args...; kwargs...)
end

function evaluate_local_rspo_plus(args...; kwargs...)
    return evaluate_local_training(args...; kwargs...)
end

function evaluate_synthetic_knapsack_two_stage(
    result::SyntheticKnapsackTwoStageTrainingResult,
    dataset::SyntheticKnapsackDataset;
    kwargs...,
)
    return SyntheticKnapsackTwoStageEvaluationResult(
        evaluate_fed_training(result.warm_start, dataset; kwargs...),
        evaluate_local_training(result.personalization, dataset; kwargs...),
    )
end

function _metric_delta(left::Real, right::Real)
    return (isfinite(left) && isfinite(right)) ? float(left) - float(right) : NaN
end

function build_synthetic_knapsack_comparison_rows(
    dataset::SyntheticKnapsackDataset,
    two_stage_evaluation::SyntheticKnapsackTwoStageEvaluationResult,
    local_only_evaluation::SyntheticKnapsackEvaluationResult;
    objective_name::Symbol,
)
    warm_start = two_stage_evaluation.warm_start
    personalization = two_stage_evaluation.personalization
    warm_start_aggregate = warm_start.aggregate
    personalization_aggregate = personalization.aggregate
    local_only_aggregate = local_only_evaluation.aggregate

    local_by_client = Dict(client.client_id => client for client in local_only_evaluation.clients)
    length(local_by_client) == length(local_only_evaluation.clients) ||
        throw(ArgumentError("duplicate client ids found in local-only evaluation"))

    rows = Vector{NamedTuple}(undef, length(personalization.clients))
    for (idx, client) in enumerate(personalization.clients)
        haskey(local_by_client, client.client_id) || throw(
            ArgumentError("missing local-only evaluation for client $(client.client_id)"),
        )
        local_client = local_by_client[client.client_id]

        rows[idx] = (
            objective=objective_name,
            warm_start_method=warm_start.method,
            two_stage_method=personalization.method,
            local_only_method=local_only_evaluation.method,
            seed=dataset.config.seed,
            eta_obj=dataset.config.eta_obj,
            eta_constr=dataset.config.eta_constr,
            eta_data_dist=dataset.config.eta_data_dist,
            data_imbalance=dataset.config.data_imbalance,
            client_id=client.client_id,
            objective_pairwise_mean=dataset.metadata.objective_pairwise_mean,
            capacity_pairwise_mean=dataset.metadata.capacity_pairwise_mean,
            feature_mean_pairwise_mean=dataset.metadata.feature_mean_pairwise_mean,
            warm_start_mse_mean=warm_start_aggregate.mse.mean,
            warm_start_absolute_regret_mean=warm_start_aggregate.absolute_regret.mean,
            warm_start_relative_regret_mean=warm_start_aggregate.relative_regret.mean,
            warm_start_regularized_decision_gap_mean=warm_start_aggregate.regularized_decision_gap.mean,
            two_stage_mse_mean=personalization_aggregate.mse.mean,
            two_stage_absolute_regret_mean=personalization_aggregate.absolute_regret.mean,
            two_stage_relative_regret_mean=personalization_aggregate.relative_regret.mean,
            two_stage_regularized_decision_gap_mean=personalization_aggregate.regularized_decision_gap.mean,
            local_only_mse_mean=local_only_aggregate.mse.mean,
            local_only_absolute_regret_mean=local_only_aggregate.absolute_regret.mean,
            local_only_relative_regret_mean=local_only_aggregate.relative_regret.mean,
            local_only_regularized_decision_gap_mean=local_only_aggregate.regularized_decision_gap.mean,
            two_stage_client_n_test_samples=client.n_test_samples,
            local_only_client_n_test_samples=local_client.n_test_samples,
            two_stage_client_mse=client.mse,
            two_stage_client_absolute_regret=client.absolute_regret,
            two_stage_client_relative_regret=client.relative_regret,
            two_stage_client_regularized_decision_gap=client.regularized_decision_gap,
            two_stage_final_lambda=client.final_lambda,
            two_stage_final_lr=client.final_lr,
            two_stage_final_bound=client.final_bound,
            two_stage_frozen_after_round=something(client.frozen_after_round, ""),
            local_only_client_mse=local_client.mse,
            local_only_client_absolute_regret=local_client.absolute_regret,
            local_only_client_relative_regret=local_client.relative_regret,
            local_only_client_regularized_decision_gap=local_client.regularized_decision_gap,
            local_only_final_lambda=local_client.final_lambda,
            local_only_final_lr=local_client.final_lr,
            local_only_final_bound=local_client.final_bound,
            local_only_frozen_after_round=something(local_client.frozen_after_round, ""),
            local_only_minus_two_stage_mse=_metric_delta(local_client.mse, client.mse),
            local_only_minus_two_stage_absolute_regret=_metric_delta(
                local_client.absolute_regret,
                client.absolute_regret,
            ),
            local_only_minus_two_stage_relative_regret=_metric_delta(
                local_client.relative_regret,
                client.relative_regret,
            ),
            local_only_minus_two_stage_regularized_decision_gap=_metric_delta(
                local_client.regularized_decision_gap,
                client.regularized_decision_gap,
            ),
        )
    end

    return rows
end

function _result_method(result::FedTrainingResult)
    return result.method
end

function _result_method(result::LocalTrainingResult)
    return result.method
end

function _slugify(value)
    return replace(string(value), "." => "p", "-" => "m", "," => "", " " => "")
end

function _synthetic_path_component(value::AbstractString; max_length::Int=120)
    length(value) <= max_length && return String(value)
    digest = bytes2hex(SHA.sha1(value))[1:12]
    prefix_length = max(max_length - length(digest) - 2, 16)
    return string(first(value, prefix_length), "__", digest)
end

function _config_slug(config; exclude::Tuple=(:seed,))
    parts = [
        string(name, "-", _slugify(getfield(config, name))) for
        name in fieldnames(typeof(config)) if !(name in exclude)
    ]
    return join(parts, "_")
end

function default_synthetic_knapsack_config_id(
    dataset_config::SyntheticKnapsackConfig,
    training_config::SyntheticKnapsackTrainingConfig,
)
    return string(
        _config_slug(dataset_config),
        "__",
        _config_slug(training_config),
    )
end

function default_synthetic_knapsack_seed_tag(
    dataset_config::SyntheticKnapsackConfig,
    training_config::SyntheticKnapsackTrainingConfig,
)
    if dataset_config.seed == training_config.seed
        return "seed$(training_config.seed)"
    end
    return "train$(training_config.seed)_data$(dataset_config.seed)"
end

function _config_to_dict(config)
    return Dict(string(name) => getfield(config, name) for name in fieldnames(typeof(config)))
end

function _write_config_snapshot(
    path::AbstractString,
    method::Symbol,
    dataset_config::SyntheticKnapsackConfig,
    training_config::SyntheticKnapsackTrainingConfig,
    config_id::AbstractString,
    seed_tag::AbstractString,
)
    payload = Dict(
        "run" => Dict(
            "method" => string(method),
            "config_id" => config_id,
            "seed_tag" => seed_tag,
            "exported_at" => string(Dates.now()),
        ),
        "dataset" => _config_to_dict(dataset_config),
        "training" => _config_to_dict(training_config),
    )

    open(path, "w") do io
        TOML.print(io, payload)
    end
    return path
end

function _write_training_rounds(path::AbstractString, result::FedRSPOPlusResult)
    header = [
        "scope",
        "client_id",
        "round",
        "loss",
        "lambda",
        "lr",
        "bound",
        "selected_clients",
    ]
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
    return _write_csv_rows(path, header, rows)
end

function _write_training_rounds(path::AbstractString, result::LocalRSPOPlusResult)
    header = [
        "scope",
        "client_id",
        "round",
        "loss",
        "lambda",
        "lr",
        "bound",
        "selected_clients",
    ]
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
    return _write_csv_rows(path, header, rows)
end

function _write_scheduler_summary(path::AbstractString, result::FedRSPOPlusResult)
    header = ["scope", "client_id", "frozen_after_round", "final_lambda", "final_lr", "final_bound", "lambda0"]
    pcl = result.trace.per_client_lambda0
    rows = if isnothing(pcl)
        [[
            "global",
            "0",
            isnothing(result.trace.frozen_after_round) ? "" : string(result.trace.frozen_after_round),
            string(_final_trace_value(result.trace.lambda_values)),
            string(_final_trace_value(result.trace.lr_values)),
            string(_final_trace_value(result.trace.bound_values)),
            "",
        ]]
    else
        [[
            "global",
            "0",
            isnothing(result.trace.frozen_after_round) ? "" : string(result.trace.frozen_after_round),
            string(_final_trace_value(result.trace.lambda_values)),
            string(_final_trace_value(result.trace.lr_values)),
            string(_final_trace_value(result.trace.bound_values)),
            join(string.(pcl), ";"),
        ]]
    end
    return _write_csv_rows(path, header, rows)
end

function _write_scheduler_summary(path::AbstractString, result::LocalRSPOPlusResult)
    header = ["scope", "client_id", "frozen_after_round", "final_lambda", "final_lr", "final_bound", "lambda0"]
    rows = (
        [
            "client",
            string(client_result.client_id),
            isnothing(client_result.trace.frozen_after_round) ? "" :
            string(client_result.trace.frozen_after_round),
            string(_final_trace_value(client_result.trace.lambda_values)),
            string(_final_trace_value(client_result.trace.lr_values)),
            string(_final_trace_value(client_result.trace.bound_values)),
            isnothing(client_result.trace.per_client_lambda0) ? "" :
            join(string.(client_result.trace.per_client_lambda0), ";"),
        ] for client_result in result.clients
    )
    return _write_csv_rows(path, header, rows)
end

function _write_evaluation_clients(
    path::AbstractString,
    evaluation::SyntheticKnapsackEvaluationResult,
)
    header = [
        "client_id",
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
    return _write_csv_rows(path, header, rows)
end

function _write_evaluation_aggregate(
    path::AbstractString,
    evaluation::SyntheticKnapsackEvaluationResult,
)
    aggregate = evaluation.aggregate
    scheduler = aggregate.scheduler
    header = ["metric", "mean", "std", "n"]
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
    return _write_csv_rows(path, header, rows)
end

function export_synthetic_knapsack_run(
    result::Union{FedTrainingResult,LocalTrainingResult},
    dataset::SyntheticKnapsackDataset;
    config::SyntheticKnapsackTrainingConfig,
    evaluation::Union{Nothing,SyntheticKnapsackEvaluationResult}=nothing,
    root::AbstractString=joinpath("results", "experiment1", "synthetic_knapsack"),
    config_id::Union{Nothing,AbstractString}=nothing,
    seed_tag::Union{Nothing,AbstractString}=nothing,
)
    method = _result_method(result)
    resolved_evaluation =
        if isnothing(evaluation)
            if result isa FedTrainingResult
                evaluate_fed_training(result, dataset)
            else
                evaluate_local_training(result, dataset)
            end
        else
            evaluation
        end
    resolved_config_id =
        something(config_id, default_synthetic_knapsack_config_id(dataset.config, config))
    resolved_seed_tag =
        something(seed_tag, default_synthetic_knapsack_seed_tag(dataset.config, config))
    safe_config_id = _synthetic_path_component(resolved_config_id)
    safe_seed_tag = _synthetic_path_component(resolved_seed_tag; max_length=64)

    export_dir = joinpath(root, string(method), safe_config_id, safe_seed_tag)
    mkpath(export_dir)

    paths = (
        config=_write_config_snapshot(
            joinpath(export_dir, "config.toml"),
            method,
            dataset.config,
            config,
            safe_config_id,
            safe_seed_tag,
        ),
        dataset_metadata=export_synthetic_knapsack_metadata(
            dataset.metadata,
            joinpath(export_dir, "dataset_metadata.csv"),
        ),
        sample_counts=export_synthetic_knapsack_sample_counts(
            dataset.metadata,
            joinpath(export_dir, "sample_counts.csv"),
        ),
        training_rounds=_write_training_rounds(joinpath(export_dir, "training_rounds.csv"), result),
        scheduler_summary=_write_scheduler_summary(
            joinpath(export_dir, "scheduler_summary.csv"),
            result,
        ),
        evaluation_clients=_write_evaluation_clients(
            joinpath(export_dir, "evaluation_clients.csv"),
            resolved_evaluation,
        ),
        evaluation_aggregate=_write_evaluation_aggregate(
            joinpath(export_dir, "evaluation_aggregate.csv"),
            resolved_evaluation,
        ),
    )

    return (; dir=export_dir, method=method, evaluation=resolved_evaluation, paths...)
end

function export_synthetic_knapsack_two_stage_run(
    result::SyntheticKnapsackTwoStageTrainingResult,
    dataset::SyntheticKnapsackDataset;
    warm_start_config::SyntheticKnapsackTrainingConfig,
    personalization_config::SyntheticKnapsackTrainingConfig,
    evaluation::Union{Nothing,SyntheticKnapsackTwoStageEvaluationResult}=nothing,
    root::AbstractString=joinpath("results", "experiment1", "synthetic_knapsack"),
    warm_start_config_id::Union{Nothing,AbstractString}=nothing,
    warm_start_seed_tag::Union{Nothing,AbstractString}=nothing,
    personalization_config_id::Union{Nothing,AbstractString}=nothing,
    personalization_seed_tag::Union{Nothing,AbstractString}=nothing,
)
    resolved_evaluation =
        isnothing(evaluation) ? evaluate_synthetic_knapsack_two_stage(result, dataset) : evaluation

    warm_start_export = export_synthetic_knapsack_run(
        result.warm_start,
        dataset;
        config=warm_start_config,
        evaluation=resolved_evaluation.warm_start,
        root=root,
        config_id=something(
            warm_start_config_id,
            string(
                "warm_start__",
                default_synthetic_knapsack_config_id(dataset.config, warm_start_config),
            ),
        ),
        seed_tag=something(
            warm_start_seed_tag,
            default_synthetic_knapsack_seed_tag(dataset.config, warm_start_config),
        ),
    )
    personalization_export = export_synthetic_knapsack_run(
        result.personalization,
        dataset;
        config=personalization_config,
        evaluation=resolved_evaluation.personalization,
        root=root,
        config_id=something(
            personalization_config_id,
            string(
                "personalization__",
                default_synthetic_knapsack_config_id(
                    dataset.config,
                    personalization_config,
                ),
            ),
        ),
        seed_tag=something(
            personalization_seed_tag,
            default_synthetic_knapsack_seed_tag(dataset.config, personalization_config),
        ),
    )

    return (
        warm_start=warm_start_export,
        personalization=personalization_export,
        evaluation=resolved_evaluation,
    )
end
