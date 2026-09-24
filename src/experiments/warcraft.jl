using Dates
using SHA
using Statistics
using TOML

struct WarcraftMetricSummary
    mean::Float64
    std::Float64
    n::Int
end

struct WarcraftAggregateSchedulerDiagnostics
    final_lambda::WarcraftMetricSummary
    final_lr::WarcraftMetricSummary
    final_bound::WarcraftMetricSummary
    freeze_round::WarcraftMetricSummary
    frozen_fraction::Float64
end

struct WarcraftClientEvaluation
    client_id::Int
    n_test_samples::Int
    pred_cost::Float64
    opt_cost::Float64
    cost_ratio::Float64
    gap::Float64
    regularized_decision_gap::Float64
    final_lambda::Float64
    final_lr::Float64
    final_bound::Float64
    frozen_after_round::Union{Nothing,Int}
end

struct WarcraftSampleEvaluation
    client_id::Int
    sample_id::Int
    pred_distance::Float64
    opt_distance::Float64
    model_output::Matrix{Float64}
    chosen_path::Matrix{Float64}
    optimal_path::Matrix{Float64}
end

struct WarcraftAggregateEvaluation
    n_clients::Int
    total_test_samples::Int
    pred_cost::WarcraftMetricSummary
    opt_cost::WarcraftMetricSummary
    cost_ratio::WarcraftMetricSummary
    gap::WarcraftMetricSummary
    regularized_decision_gap::WarcraftMetricSummary
    scheduler::WarcraftAggregateSchedulerDiagnostics
end

struct WarcraftEvaluationResult
    method::Symbol
    clients::Vector{WarcraftClientEvaluation}
    aggregate::WarcraftAggregateEvaluation
    samples::Vector{WarcraftSampleEvaluation}
end

struct WarcraftTwoStageEvaluationResult
    warm_start::WarcraftEvaluationResult
    personalization::WarcraftEvaluationResult
end

function _warcraft_final_trace_value(values::AbstractVector{<:Real})
    return isempty(values) ? NaN : float(values[end])
end

function _warcraft_summarize_metric(values)
    finite_values = Float64[float(value) for value in values if isfinite(value)]
    if isempty(finite_values)
        return WarcraftMetricSummary(NaN, NaN, 0)
    end
    return WarcraftMetricSummary(
        mean(finite_values),
        length(finite_values) == 1 ? 0.0 : std(finite_values; corrected=false),
        length(finite_values),
    )
end

function _warcraft_regularized_decision_gap(
    samples::Vector{WarcraftSample},
    predictions::Vector{Matrix{Float64}};
    lambda::Real,
    use_warm_start::Bool=true,
)
    (!isfinite(lambda) || lambda <= 0 || isempty(samples)) && return NaN
    gaps = Float64[]
    for (sample, prediction) in zip(samples, predictions)
        cache = init_warcraft_projection_cache(sample.instance; lambda=lambda)
        y_hat = projection_optimizer(
            prediction;
            instance=sample.instance,
            cache=cache,
            lambda=lambda,
            use_warm_start=use_warm_start,
        )
        y_star = projection_optimizer(
            sample.theta_true;
            instance=sample.instance,
            cache=cache,
            lambda=lambda,
            use_warm_start=use_warm_start,
        )
        push!(gaps, norm(y_hat .- y_star))
    end
    return mean(gaps)
end

function _warcraft_predict_samples(model, samples::Vector{WarcraftSample}; batch_size::Int=64)
    isempty(samples) && return Matrix{Float64}[]
    predictions = Matrix{Float64}[]
    for start_idx in 1:batch_size:length(samples)
        stop_idx = min(start_idx + batch_size - 1, length(samples))
        batch_indices = collect(start_idx:stop_idx)
        batch = _batch_images(samples, batch_indices)
        batch_predictions = max.(model(batch), zero(Float32))
        for local_idx in 1:length(batch_indices)
            push!(predictions, Matrix{Float64}(view(batch_predictions, :, :, local_idx)))
        end
    end
    return predictions
end

function _warcraft_matrix_literal(matrix::AbstractMatrix{<:Real})
    row_literals = [
        string(
            "[",
            join((string(float(matrix[row_idx, col_idx])) for col_idx in axes(matrix, 2)), ","),
            "]",
        ) for row_idx in axes(matrix, 1)
    ]
    return string("[", join(row_literals, ","), "]")
end

function _evaluate_warcraft_client(
    client::WarcraftClientData,
    model,
    trace::SchedulerTrace;
    include_regularized_decision_gap::Bool=true,
    batch_size::Int=64,
    use_warm_start::Bool=true,
)
    final_lambda = _warcraft_final_trace_value(trace.lambda_values)
    final_lr = _warcraft_final_trace_value(trace.lr_values)
    final_bound = _warcraft_final_trace_value(trace.bound_values)
    n_test_samples = length(client.test)

    if n_test_samples == 0
        return (
            WarcraftClientEvaluation(
                client.client_id,
                0,
                NaN,
                NaN,
                NaN,
                NaN,
                NaN,
                final_lambda,
                final_lr,
                final_bound,
                trace.frozen_after_round,
            ),
            WarcraftSampleEvaluation[],
        )
    end

    predictions = _warcraft_predict_samples(model, client.test; batch_size=batch_size)
    pred_costs = Float64[]
    opt_costs = Float64[]
    cost_ratios = Float64[]
    gaps = Float64[]
    sample_evaluations = WarcraftSampleEvaluation[]
    for (sample, prediction) in zip(client.test, predictions)
        y_hat = solve_warcraft_path(prediction; instance=sample.instance)
        opt_cost = warcraft_path_cost(sample.theta_true, sample.y_true)
        pred_cost = warcraft_path_cost(sample.theta_true, y_hat)
        push!(pred_costs, pred_cost)
        push!(opt_costs, opt_cost)
        ratio = opt_cost == 0 ? NaN : pred_cost / opt_cost
        push!(cost_ratios, ratio)
        push!(gaps, ratio - 1.0)
        push!(
            sample_evaluations,
            WarcraftSampleEvaluation(
                client.client_id,
                sample.sample_id,
                pred_cost,
                opt_cost,
                copy(prediction),
                Matrix{Float64}(y_hat),
                Matrix{Float64}(sample.y_true),
            ),
        )
    end

    regularized_gap =
        include_regularized_decision_gap ? _warcraft_regularized_decision_gap(
            client.test,
            predictions;
            lambda=final_lambda,
            use_warm_start=use_warm_start,
        ) : NaN

    return (
        WarcraftClientEvaluation(
            client.client_id,
            n_test_samples,
            mean(pred_costs),
            mean(opt_costs),
            mean(cost_ratios),
            mean(gaps),
            regularized_gap,
            final_lambda,
            final_lr,
            final_bound,
            trace.frozen_after_round,
        ),
        sample_evaluations,
    )
end

function _aggregate_warcraft_client_evaluations(clients::Vector{WarcraftClientEvaluation})
    freeze_rounds = [float(something(client.frozen_after_round, NaN)) for client in clients]
    frozen_fraction =
        isempty(clients) ? NaN : mean(
            [isnothing(client.frozen_after_round) ? 0.0 : 1.0 for client in clients],
        )

    return WarcraftAggregateEvaluation(
        length(clients),
        sum(client.n_test_samples for client in clients),
        _warcraft_summarize_metric(client.pred_cost for client in clients),
        _warcraft_summarize_metric(client.opt_cost for client in clients),
        _warcraft_summarize_metric(client.cost_ratio for client in clients),
        _warcraft_summarize_metric(client.gap for client in clients),
        _warcraft_summarize_metric(client.regularized_decision_gap for client in clients),
        WarcraftAggregateSchedulerDiagnostics(
            _warcraft_summarize_metric(client.final_lambda for client in clients),
            _warcraft_summarize_metric(client.final_lr for client in clients),
            _warcraft_summarize_metric(client.final_bound for client in clients),
            _warcraft_summarize_metric(freeze_rounds),
            frozen_fraction,
        ),
    )
end

function evaluate_fed_warcraft(
    result::WarcraftFedTrainingResult;
    include_regularized_decision_gap::Bool=true,
    batch_size::Int=64,
    use_warm_start::Bool=true,
)
    traces = fill(result.trace, length(result.client_data))
    clients = Vector{WarcraftClientEvaluation}(undef, length(result.client_data))
    samples = WarcraftSampleEvaluation[]
    for (idx, client) in enumerate(result.client_data)
        client_evaluation, client_samples = _evaluate_warcraft_client(
            client,
            result.model,
            traces[idx];
            include_regularized_decision_gap=include_regularized_decision_gap,
            batch_size=batch_size,
            use_warm_start=use_warm_start,
        )
        clients[idx] = client_evaluation
        append!(samples, client_samples)
    end
    return WarcraftEvaluationResult(
        result.method,
        clients,
        _aggregate_warcraft_client_evaluations(clients),
        samples,
    )
end

function evaluate_fed_warcraft(
    result::WarcraftFedTrainingResult,
    dataset::WarcraftDataset;
    kwargs...,
)
    fresh_client_data = prepare_warcraft_client_data(dataset)
    fresh_result = WarcraftFedTrainingResult(
        result.method,
        result.model,
        fresh_client_data,
        result.round_losses,
        result.selected_clients,
        result.trace,
    )
    return evaluate_fed_warcraft(fresh_result; kwargs...)
end

function evaluate_local_warcraft(
    result::WarcraftLocalTrainingResult,
    client_data::Vector{WarcraftClientData};
    include_regularized_decision_gap::Bool=true,
    batch_size::Int=64,
    use_warm_start::Bool=true,
)
    result_by_client = Dict(entry.client_id => entry for entry in result.clients)
    clients = WarcraftClientEvaluation[]
    samples = WarcraftSampleEvaluation[]
    for client in client_data
        local_result = result_by_client[client.client_id]
        client_evaluation, client_samples =
            _evaluate_warcraft_client(
                client,
                local_result.model,
                local_result.trace;
                include_regularized_decision_gap=include_regularized_decision_gap,
                batch_size=batch_size,
                use_warm_start=use_warm_start,
            )
        push!(
            clients,
            client_evaluation,
        )
        append!(samples, client_samples)
    end
    return WarcraftEvaluationResult(
        result.method,
        clients,
        _aggregate_warcraft_client_evaluations(clients),
        samples,
    )
end

function evaluate_local_warcraft(
    result::WarcraftLocalTrainingResult,
    dataset::WarcraftDataset;
    kwargs...,
)
    return evaluate_local_warcraft(result, prepare_warcraft_client_data(dataset); kwargs...)
end

function evaluate_warcraft_two_stage(
    result::WarcraftTwoStageTrainingResult,
    dataset::WarcraftDataset;
    kwargs...,
)
    return WarcraftTwoStageEvaluationResult(
        evaluate_fed_warcraft(result.warm_start, dataset; kwargs...),
        evaluate_local_warcraft(result.personalization, dataset; kwargs...),
    )
end

function evaluate_rspo_plus_warcraft_experiment(args...; kwargs...)
    return evaluate_warcraft_two_stage(args...; kwargs...)
end

function _warcraft_metric_delta(left::Real, right::Real)
    return (isfinite(left) && isfinite(right)) ? float(left) - float(right) : NaN
end

function build_warcraft_comparison_client_rows(
    two_stage_evaluation::WarcraftTwoStageEvaluationResult,
    local_only_evaluation::WarcraftEvaluationResult,
)
    warm_start = two_stage_evaluation.warm_start
    personalization = two_stage_evaluation.personalization

    warm_by_client = Dict(client.client_id => client for client in warm_start.clients)
    personalization_by_client = Dict(
        client.client_id => client for client in personalization.clients
    )
    local_by_client = Dict(client.client_id => client for client in local_only_evaluation.clients)

    rows = Vector{NamedTuple}(undef, length(personalization.clients))
    for (idx, client) in enumerate(personalization.clients)
        haskey(warm_by_client, client.client_id) || throw(
            ArgumentError("missing warm-start evaluation for client $(client.client_id)"),
        )
        haskey(local_by_client, client.client_id) || throw(
            ArgumentError("missing local-only evaluation for client $(client.client_id)"),
        )
        warm_client = warm_by_client[client.client_id]
        local_client = local_by_client[client.client_id]
        personalization_client = personalization_by_client[client.client_id]

        rows[idx] = (
            client_id=client.client_id,
            n_test_samples=client.n_test_samples,
            warm_start_method=warm_start.method,
            two_stage_method=personalization.method,
            local_only_method=local_only_evaluation.method,
            warm_start_pred_cost=warm_client.pred_cost,
            warm_start_opt_cost=warm_client.opt_cost,
            warm_start_cost_ratio=warm_client.cost_ratio,
            warm_start_gap=warm_client.gap,
            warm_start_regularized_decision_gap=warm_client.regularized_decision_gap,
            warm_start_final_lambda=warm_client.final_lambda,
            warm_start_final_lr=warm_client.final_lr,
            warm_start_final_bound=warm_client.final_bound,
            warm_start_frozen_after_round=something(warm_client.frozen_after_round, ""),
            two_stage_pred_cost=personalization_client.pred_cost,
            two_stage_opt_cost=personalization_client.opt_cost,
            two_stage_cost_ratio=personalization_client.cost_ratio,
            two_stage_gap=personalization_client.gap,
            two_stage_regularized_decision_gap=personalization_client.regularized_decision_gap,
            two_stage_final_lambda=personalization_client.final_lambda,
            two_stage_final_lr=personalization_client.final_lr,
            two_stage_final_bound=personalization_client.final_bound,
            two_stage_frozen_after_round=something(
                personalization_client.frozen_after_round,
                "",
            ),
            local_only_pred_cost=local_client.pred_cost,
            local_only_opt_cost=local_client.opt_cost,
            local_only_cost_ratio=local_client.cost_ratio,
            local_only_gap=local_client.gap,
            local_only_regularized_decision_gap=local_client.regularized_decision_gap,
            local_only_final_lambda=local_client.final_lambda,
            local_only_final_lr=local_client.final_lr,
            local_only_final_bound=local_client.final_bound,
            local_only_frozen_after_round=something(local_client.frozen_after_round, ""),
            local_only_minus_two_stage_pred_cost=_warcraft_metric_delta(
                local_client.pred_cost,
                personalization_client.pred_cost,
            ),
            local_only_minus_two_stage_cost_ratio=_warcraft_metric_delta(
                local_client.cost_ratio,
                personalization_client.cost_ratio,
            ),
            local_only_minus_two_stage_gap=_warcraft_metric_delta(
                local_client.gap,
                personalization_client.gap,
            ),
            local_only_minus_two_stage_regularized_decision_gap=_warcraft_metric_delta(
                local_client.regularized_decision_gap,
                personalization_client.regularized_decision_gap,
            ),
        )
    end

    return rows
end

function _warcraft_comparison_aggregate_row(
    metric::AbstractString,
    warm_start::WarcraftMetricSummary,
    two_stage::WarcraftMetricSummary,
    local_only::WarcraftMetricSummary,
)
    return (
        metric=metric,
        warm_start_mean=warm_start.mean,
        warm_start_std=warm_start.std,
        warm_start_n=warm_start.n,
        two_stage_mean=two_stage.mean,
        two_stage_std=two_stage.std,
        two_stage_n=two_stage.n,
        local_only_mean=local_only.mean,
        local_only_std=local_only.std,
        local_only_n=local_only.n,
        local_only_minus_two_stage_mean=_warcraft_metric_delta(
            local_only.mean,
            two_stage.mean,
        ),
    )
end

function _warcraft_fraction_summary(value::Real, n::Int)
    return WarcraftMetricSummary(float(value), 0.0, n)
end

function build_warcraft_comparison_aggregate_rows(
    two_stage_evaluation::WarcraftTwoStageEvaluationResult,
    local_only_evaluation::WarcraftEvaluationResult,
)
    warm_start = two_stage_evaluation.warm_start.aggregate
    personalization = two_stage_evaluation.personalization.aggregate
    local_only = local_only_evaluation.aggregate

    return [
        _warcraft_comparison_aggregate_row(
            "pred_cost",
            warm_start.pred_cost,
            personalization.pred_cost,
            local_only.pred_cost,
        ),
        _warcraft_comparison_aggregate_row(
            "opt_cost",
            warm_start.opt_cost,
            personalization.opt_cost,
            local_only.opt_cost,
        ),
        _warcraft_comparison_aggregate_row(
            "cost_ratio",
            warm_start.cost_ratio,
            personalization.cost_ratio,
            local_only.cost_ratio,
        ),
        _warcraft_comparison_aggregate_row(
            "gap",
            warm_start.gap,
            personalization.gap,
            local_only.gap,
        ),
        _warcraft_comparison_aggregate_row(
            "regularized_decision_gap",
            warm_start.regularized_decision_gap,
            personalization.regularized_decision_gap,
            local_only.regularized_decision_gap,
        ),
        _warcraft_comparison_aggregate_row(
            "final_lambda",
            warm_start.scheduler.final_lambda,
            personalization.scheduler.final_lambda,
            local_only.scheduler.final_lambda,
        ),
        _warcraft_comparison_aggregate_row(
            "final_lr",
            warm_start.scheduler.final_lr,
            personalization.scheduler.final_lr,
            local_only.scheduler.final_lr,
        ),
        _warcraft_comparison_aggregate_row(
            "final_bound",
            warm_start.scheduler.final_bound,
            personalization.scheduler.final_bound,
            local_only.scheduler.final_bound,
        ),
        _warcraft_comparison_aggregate_row(
            "freeze_round",
            warm_start.scheduler.freeze_round,
            personalization.scheduler.freeze_round,
            local_only.scheduler.freeze_round,
        ),
        _warcraft_comparison_aggregate_row(
            "frozen_fraction",
            _warcraft_fraction_summary(
                warm_start.scheduler.frozen_fraction,
                warm_start.n_clients,
            ),
            _warcraft_fraction_summary(
                personalization.scheduler.frozen_fraction,
                personalization.n_clients,
            ),
            _warcraft_fraction_summary(
                local_only.scheduler.frozen_fraction,
                local_only.n_clients,
            ),
        ),
    ]
end

function _slugify(value)
    return replace(string(value), "." => "p", "-" => "m", "," => "", " " => "")
end

function _warcraft_config_slug(config; exclude::Tuple=(:seed,))
    parts = [
        string(name, "-", _slugify(getfield(config, name))) for
        name in fieldnames(typeof(config)) if !(name in exclude)
    ]
    return join(parts, "_")
end

function _warcraft_path_component(value::AbstractString; max_length::Int=120)
    length(value) <= max_length && return String(value)
    digest = bytes2hex(SHA.sha1(value))[1:12]
    prefix_length = max(max_length - length(digest) - 2, 16)
    return string(first(value, prefix_length), "__", digest)
end

function default_warcraft_config_id(
    dataset_config::WarcraftDatasetConfig,
    training_config::WarcraftTrainingConfig,
)
    return string(_warcraft_config_slug(dataset_config), "__", _warcraft_config_slug(training_config))
end

function default_warcraft_seed_tag(
    dataset_config::WarcraftDatasetConfig,
    training_config::WarcraftTrainingConfig,
)
    if dataset_config.seed == training_config.seed
        return "seed$(training_config.seed)"
    end
    return "train$(training_config.seed)_data$(dataset_config.seed)"
end

function default_warcraft_comparison_config_id(
    dataset_config::WarcraftDatasetConfig,
    warm_start_config::WarcraftTrainingConfig,
    personalization_config::WarcraftTrainingConfig,
    local_only_config::WarcraftTrainingConfig,
)
    return string(
        _warcraft_config_slug(dataset_config),
        "__warm__",
        _warcraft_config_slug(warm_start_config),
        "__personal__",
        _warcraft_config_slug(personalization_config),
        "__local__",
        _warcraft_config_slug(local_only_config),
    )
end

function default_warcraft_comparison_seed_tag(
    dataset_config::WarcraftDatasetConfig,
    warm_start_config::WarcraftTrainingConfig,
    personalization_config::WarcraftTrainingConfig,
    local_only_config::WarcraftTrainingConfig,
)
    return string(
        "warm",
        warm_start_config.seed,
        "_personal",
        personalization_config.seed,
        "_local",
        local_only_config.seed,
        "_data",
        dataset_config.seed,
    )
end

_warcraft_config_value(value::Symbol) = string(value)
_warcraft_config_value(value) = value

function _warcraft_config_to_dict(config)
    return Dict(
        string(name) => _warcraft_config_value(getfield(config, name)) for
        name in fieldnames(typeof(config))
    )
end

function _write_warcraft_config_snapshot(
    path::AbstractString,
    method::Symbol,
    dataset_config::WarcraftDatasetConfig,
    training_config::WarcraftTrainingConfig,
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
        "dataset" => _warcraft_config_to_dict(dataset_config),
        "training" => _warcraft_config_to_dict(training_config),
    )
    open(path, "w") do io
        TOML.print(io, payload)
    end
    return path
end

function _write_warcraft_comparison_config_snapshot(
    path::AbstractString,
    dataset_config::WarcraftDatasetConfig,
    warm_start_config::WarcraftTrainingConfig,
    personalization_config::WarcraftTrainingConfig,
    local_only_config::WarcraftTrainingConfig,
    config_id::AbstractString,
    seed_tag::AbstractString,
    warm_start_method::Symbol,
    personalization_method::Symbol,
    local_only_method::Symbol,
)
    payload = Dict(
        "run" => Dict(
            "comparison" => "two_stage_vs_local",
            "config_id" => config_id,
            "seed_tag" => seed_tag,
            "exported_at" => string(Dates.now()),
        ),
        "dataset" => _warcraft_config_to_dict(dataset_config),
        "methods" => Dict(
            "warm_start" => string(warm_start_method),
            "two_stage" => string(personalization_method),
            "local_only" => string(local_only_method),
        ),
        "warm_start_training" => _warcraft_config_to_dict(warm_start_config),
        "personalization_training" => _warcraft_config_to_dict(personalization_config),
        "local_only_training" => _warcraft_config_to_dict(local_only_config),
    )
    open(path, "w") do io
        TOML.print(io, payload)
    end
    return path
end

function _write_warcraft_training_rounds(path::AbstractString, result::WarcraftFedTrainingResult)
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
    return _write_warcraft_csv_rows(path, header, rows)
end

function _write_warcraft_training_rounds(path::AbstractString, result::WarcraftLocalTrainingResult)
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
    return _write_warcraft_csv_rows(path, header, rows)
end

function _write_warcraft_scheduler_summary(path::AbstractString, result::WarcraftFedTrainingResult)
    header = ["scope", "client_id", "frozen_after_round", "final_lambda", "final_lr", "final_bound", "lambda0"]
    pcl = result.trace.per_client_lambda0
    rows = [[
        "global",
        "0",
        isnothing(result.trace.frozen_after_round) ? "" : string(result.trace.frozen_after_round),
        string(_warcraft_final_trace_value(result.trace.lambda_values)),
        string(_warcraft_final_trace_value(result.trace.lr_values)),
        string(_warcraft_final_trace_value(result.trace.bound_values)),
        isnothing(pcl) ? "" : join(string.(pcl), ";"),
    ]]
    return _write_warcraft_csv_rows(path, header, rows)
end

function _write_warcraft_scheduler_summary(path::AbstractString, result::WarcraftLocalTrainingResult)
    header = ["scope", "client_id", "frozen_after_round", "final_lambda", "final_lr", "final_bound", "lambda0"]
    rows = (
        [
            "client",
            string(client_result.client_id),
            isnothing(client_result.trace.frozen_after_round) ? "" :
            string(client_result.trace.frozen_after_round),
            string(_warcraft_final_trace_value(client_result.trace.lambda_values)),
            string(_warcraft_final_trace_value(client_result.trace.lr_values)),
            string(_warcraft_final_trace_value(client_result.trace.bound_values)),
            isnothing(client_result.trace.per_client_lambda0) ? "" :
            join(string.(client_result.trace.per_client_lambda0), ";"),
        ] for client_result in result.clients
    )
    return _write_warcraft_csv_rows(path, header, rows)
end

function _write_warcraft_evaluation_clients(
    path::AbstractString,
    evaluation::WarcraftEvaluationResult,
)
    header = [
        "client_id",
        "n_test_samples",
        "pred_cost",
        "opt_cost",
        "cost_ratio",
        "gap",
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
            string(client.pred_cost),
            string(client.opt_cost),
            string(client.cost_ratio),
            string(client.gap),
            string(client.regularized_decision_gap),
            string(client.final_lambda),
            string(client.final_lr),
            string(client.final_bound),
            isnothing(client.frozen_after_round) ? "" : string(client.frozen_after_round),
        ] for client in evaluation.clients
    )
    return _write_warcraft_csv_rows(path, header, rows)
end

function _write_warcraft_evaluation_aggregate(
    path::AbstractString,
    evaluation::WarcraftEvaluationResult,
)
    aggregate = evaluation.aggregate
    scheduler = aggregate.scheduler
    rows = [
        ["pred_cost", string(aggregate.pred_cost.mean), string(aggregate.pred_cost.std), string(aggregate.pred_cost.n)],
        ["opt_cost", string(aggregate.opt_cost.mean), string(aggregate.opt_cost.std), string(aggregate.opt_cost.n)],
        ["cost_ratio", string(aggregate.cost_ratio.mean), string(aggregate.cost_ratio.std), string(aggregate.cost_ratio.n)],
        ["gap", string(aggregate.gap.mean), string(aggregate.gap.std), string(aggregate.gap.n)],
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
    return _write_warcraft_csv_rows(path, ["metric", "mean", "std", "n"], rows)
end

function _write_warcraft_evaluation_detailed(
    path::AbstractString,
    evaluation::WarcraftEvaluationResult,
)
    header = [
        "client_id",
        "test_sample_id",
        "pred_distance",
        "optimal_distance",
        "model_output",
        "chosen_path",
        "optimal_path",
    ]
    rows = (
        [
            string(sample.client_id),
            string(sample.sample_id),
            string(sample.pred_distance),
            string(sample.opt_distance),
            _warcraft_matrix_literal(sample.model_output),
            _warcraft_matrix_literal(sample.chosen_path),
            _warcraft_matrix_literal(sample.optimal_path),
        ] for sample in evaluation.samples
    )
    return _write_warcraft_csv_rows(path, header, rows)
end

_result_method(result::WarcraftFedTrainingResult) = result.method
_result_method(result::WarcraftLocalTrainingResult) = result.method

function export_warcraft_run(
    result::Union{WarcraftFedTrainingResult,WarcraftLocalTrainingResult},
    dataset::WarcraftDataset;
    config::WarcraftTrainingConfig,
    evaluation::Union{Nothing,WarcraftEvaluationResult}=nothing,
    root::AbstractString=joinpath("results", "experiment2", "warcraft"),
    config_id::Union{Nothing,AbstractString}=nothing,
    seed_tag::Union{Nothing,AbstractString}=nothing,
)
    method = _result_method(result)
    resolved_evaluation =
        if isnothing(evaluation)
            result isa WarcraftFedTrainingResult ? evaluate_fed_warcraft(result, dataset) :
            evaluate_local_warcraft(result, dataset)
        else
            evaluation
        end
    resolved_config_id =
        something(config_id, default_warcraft_config_id(dataset.config, config))
    resolved_seed_tag =
        something(seed_tag, default_warcraft_seed_tag(dataset.config, config))
    safe_config_id = _warcraft_path_component(resolved_config_id)
    safe_seed_tag = _warcraft_path_component(resolved_seed_tag; max_length=64)

    export_dir = joinpath(root, string(method), safe_config_id, safe_seed_tag)
    mkpath(export_dir)

    paths = (
        config=_write_warcraft_config_snapshot(
            joinpath(export_dir, "config.toml"),
            method,
            dataset.config,
            config,
            safe_config_id,
            safe_seed_tag,
        ),
        dataset_metadata=export_warcraft_dataset_metadata(
            dataset.metadata,
            joinpath(export_dir, "dataset_metadata.csv"),
        ),
        partition_manifest=export_warcraft_partition_manifest(
            dataset.partition_manifest,
            dataset.metadata.terrain_values,
            joinpath(export_dir, "partition_manifest.csv"),
        ),
        training_rounds=_write_warcraft_training_rounds(
            joinpath(export_dir, "training_rounds.csv"),
            result,
        ),
        scheduler_summary=_write_warcraft_scheduler_summary(
            joinpath(export_dir, "scheduler_summary.csv"),
            result,
        ),
        evaluation_clients=_write_warcraft_evaluation_clients(
            joinpath(export_dir, "evaluation_clients.csv"),
            resolved_evaluation,
        ),
        evaluation_detailed=_write_warcraft_evaluation_detailed(
            joinpath(export_dir, "evaluation_detailed.csv"),
            resolved_evaluation,
        ),
        evaluation_aggregate=_write_warcraft_evaluation_aggregate(
            joinpath(export_dir, "evaluation_aggregate.csv"),
            resolved_evaluation,
        ),
    )
    return (; dir=export_dir, method=method, evaluation=resolved_evaluation, paths...)
end

function export_warcraft_two_stage_run(
    result::WarcraftTwoStageTrainingResult,
    dataset::WarcraftDataset;
    warm_start_config::WarcraftTrainingConfig,
    personalization_config::WarcraftTrainingConfig,
    evaluation::Union{Nothing,WarcraftTwoStageEvaluationResult}=nothing,
    root::AbstractString=joinpath("results", "experiment2", "warcraft"),
    warm_start_config_id::Union{Nothing,AbstractString}=nothing,
    warm_start_seed_tag::Union{Nothing,AbstractString}=nothing,
    personalization_config_id::Union{Nothing,AbstractString}=nothing,
    personalization_seed_tag::Union{Nothing,AbstractString}=nothing,
)
    resolved_evaluation =
        isnothing(evaluation) ? evaluate_warcraft_two_stage(result, dataset) : evaluation

    warm_start_export = export_warcraft_run(
        result.warm_start,
        dataset;
        config=warm_start_config,
        evaluation=resolved_evaluation.warm_start,
        root=root,
        config_id=something(
            warm_start_config_id,
            string(
                "warm_start__",
                default_warcraft_config_id(dataset.config, warm_start_config),
            ),
        ),
        seed_tag=something(
            warm_start_seed_tag,
            default_warcraft_seed_tag(dataset.config, warm_start_config),
        ),
    )
    personalization_export = export_warcraft_run(
        result.personalization,
        dataset;
        config=personalization_config,
        evaluation=resolved_evaluation.personalization,
        root=root,
        config_id=something(
            personalization_config_id,
            string(
                "personalization__",
                default_warcraft_config_id(dataset.config, personalization_config),
            ),
        ),
        seed_tag=something(
            personalization_seed_tag,
            default_warcraft_seed_tag(dataset.config, personalization_config),
        ),
    )

    return (
        warm_start=warm_start_export,
        personalization=personalization_export,
        evaluation=resolved_evaluation,
    )
end

function export_warcraft_comparison_summary(
    two_stage_result::WarcraftTwoStageTrainingResult,
    local_only_result::WarcraftLocalTrainingResult,
    dataset::WarcraftDataset;
    warm_start_config::WarcraftTrainingConfig,
    personalization_config::WarcraftTrainingConfig,
    local_only_config::WarcraftTrainingConfig,
    two_stage_evaluation::Union{Nothing,WarcraftTwoStageEvaluationResult}=nothing,
    local_only_evaluation::Union{Nothing,WarcraftEvaluationResult}=nothing,
    root::AbstractString=joinpath("results", "experiment2", "warcraft"),
    config_id::Union{Nothing,AbstractString}=nothing,
    seed_tag::Union{Nothing,AbstractString}=nothing,
)
    resolved_two_stage_evaluation =
        isnothing(two_stage_evaluation) ?
        evaluate_warcraft_two_stage(two_stage_result, dataset) : two_stage_evaluation
    resolved_local_only_evaluation =
        isnothing(local_only_evaluation) ?
        evaluate_local_warcraft(local_only_result, dataset) : local_only_evaluation

    resolved_config_id = something(
        config_id,
        default_warcraft_comparison_config_id(
            dataset.config,
            warm_start_config,
            personalization_config,
            local_only_config,
        ),
    )
    resolved_seed_tag = something(
        seed_tag,
        default_warcraft_comparison_seed_tag(
            dataset.config,
            warm_start_config,
            personalization_config,
            local_only_config,
        ),
    )
    safe_config_id = _warcraft_path_component(resolved_config_id)
    safe_seed_tag = _warcraft_path_component(resolved_seed_tag; max_length=64)

    export_dir = joinpath(root, "comparison", safe_config_id, safe_seed_tag)
    mkpath(export_dir)

    client_rows = build_warcraft_comparison_client_rows(
        resolved_two_stage_evaluation,
        resolved_local_only_evaluation,
    )
    aggregate_rows = build_warcraft_comparison_aggregate_rows(
        resolved_two_stage_evaluation,
        resolved_local_only_evaluation,
    )

    paths = (
        config=_write_warcraft_comparison_config_snapshot(
            joinpath(export_dir, "config.toml"),
            dataset.config,
            warm_start_config,
            personalization_config,
            local_only_config,
            safe_config_id,
            safe_seed_tag,
            resolved_two_stage_evaluation.warm_start.method,
            resolved_two_stage_evaluation.personalization.method,
            resolved_local_only_evaluation.method,
        ),
        comparison_clients=_write_warcraft_csv_rows(
            joinpath(export_dir, "comparison_clients.csv"),
            String.(collect(keys(client_rows[1]))),
            ([row[key] for key in keys(client_rows[1])] for row in client_rows),
        ),
        comparison_aggregate=_write_warcraft_csv_rows(
            joinpath(export_dir, "comparison_aggregate.csv"),
            String.(collect(keys(aggregate_rows[1]))),
            ([row[key] for key in keys(aggregate_rows[1])] for row in aggregate_rows),
        ),
    )

    return (
        dir=export_dir,
        two_stage_evaluation=resolved_two_stage_evaluation,
        local_only_evaluation=resolved_local_only_evaluation,
        paths...,
    )
end

function export_rspo_plus_warcraft_experiment(args...; kwargs...)
    return export_warcraft_two_stage_run(args...; kwargs...)
end
