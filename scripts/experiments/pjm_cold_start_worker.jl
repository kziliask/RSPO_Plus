using Pkg
using Random

Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "..", "..", "src", "models", "knapsack_oracles.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "models", "battery_oracles.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "models", "RSPOPlusLoss.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "models", "scheduling.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "datagen", "pjm_battery.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "models", "pjm_battery_training.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "experiments", "pjm_battery.jl"))

function _pjm_budget_local_config(seed::Int)
    return PJMBatteryTrainingConfig(
        seed=seed,
        hidden_dim=64,
        rounds=45,
        local_epochs=1,
        batch_size=64,
        client_fraction=1.0,
        validation_client_fraction=1.0,
        validation_max_samples_per_client=8,
        lambda0=55.0,
        kappa_lambda=1.0,
        per_client_lambda=true,
        lr0=1e-3,
        kappa_lr=0.1,
        per_client_lr=true,
        lr_lambda_alpha=0.001,
        clip_norm=1.0,
        freeze_tau=0.03,
        freeze_eps=1e-8,
        stop_after_freeze_rounds=3,
        shuffle_batches=true,
        use_warm_start=true,
    )
end

function _pjm_budget_warm_config(seed::Int)
    return PJMBatteryTrainingConfig(
        seed=seed,
        hidden_dim=64,
        rounds=10,
        local_epochs=3,
        batch_size=64,
        client_fraction=0.4,
        validation_client_fraction=0.4,
        validation_max_samples_per_client=8,
        lambda0=2.0,
        kappa_lambda=0.0,
        per_client_lambda=true,
        lr0=1e-3,
        kappa_lr=0.0,
        per_client_lr=true,
        lr_lambda_alpha=0.01,
        clip_norm=1.0,
        freeze_tau=1.0,
        freeze_eps=1e-8,
        stop_after_freeze_rounds=3,
        shuffle_batches=true,
        use_warm_start=true,
    )
end

function _pjm_budget_personal_config(seed::Int)
    return PJMBatteryTrainingConfig(
        seed=seed + 1,
        hidden_dim=64,
        rounds=15,
        local_epochs=1,
        batch_size=64,
        client_fraction=1.0,
        validation_client_fraction=1.0,
        validation_max_samples_per_client=8,
        lambda0=55.0,
        kappa_lambda=1.0,
        per_client_lambda=true,
        lr0=1e-3,
        kappa_lr=0.1,
        per_client_lr=true,
        lr_lambda_alpha=0.001,
        clip_norm=1.0,
        freeze_tau=0.03,
        freeze_eps=1e-8,
        stop_after_freeze_rounds=3,
        shuffle_batches=true,
        use_warm_start=true,
    )
end


function _pjm_cold_start_subset_split(
    split::PJMBatterySplit,
    indices::AbstractVector{Int};
    name::Symbol=split.name,
    reindexed_client_id::Union{Nothing,Int}=nothing,
)
    client_ids =
        isnothing(reindexed_client_id) ? split.client_ids[indices] :
        fill(reindexed_client_id, length(indices))
    return PJMBatterySplit(
        name,
        split.x[:, indices],
        split.theta_true[:, indices],
        client_ids,
        split.client_names[indices],
        split.capacities[indices],
        split.sample_ids[indices],
        split.date_indices[indices],
        split.dates[indices],
    )
end

function _pjm_cold_start_manifest(
    clients::Vector{PJMBatteryClient},
    train::PJMBatterySplit,
    val::PJMBatterySplit,
    test::PJMBatterySplit,
)
    manifests = PJMBatteryClientManifest[]
    for client in clients
        train_indices = findall(==(client.client_id), train.client_ids)
        val_indices = findall(==(client.client_id), val.client_ids)
        test_indices = findall(==(client.client_id), test.client_ids)
        push!(
            manifests,
            PJMBatteryClientManifest(
                client.client_id,
                client.client_name,
                client.capacity,
                train.sample_ids[train_indices],
                val.sample_ids[val_indices],
                test.sample_ids[test_indices],
            ),
        )
    end
    return PJMBatteryPartitionManifest(
        sort(unique(train.date_indices)),
        sort(unique(val.date_indices)),
        sort(unique(test.date_indices)),
        manifests,
    )
end

function _pjm_cold_start_metadata(
    source::PJMBatteryDataset,
    clients::Vector{PJMBatteryClient},
    train::PJMBatterySplit,
    val::PJMBatterySplit,
    test::PJMBatterySplit,
)
    return PJMBatteryDatasetMetadata(
        source.metadata.raw_path,
        source.metadata.combined_path,
        source.metadata.panel_path,
        length(clients),
        source.metadata.n_dates,
        length(train.sample_ids) + length(val.sample_ids) + length(test.sample_ids),
        copy(source.metadata.feature_names),
        copy(source.metadata.target_names),
        sort(unique(train.date_indices)),
        sort(unique(val.date_indices)),
        sort(unique(test.date_indices)),
        _pjm_client_counts(train, length(clients)),
        _pjm_client_counts(val, length(clients)),
        _pjm_client_counts(test, length(clients)),
        copy(source.metadata.missing_dates),
    )
end

function _pjm_cold_start_donor_dataset(
    source::PJMBatteryDataset,
    target_client_id::Int,
)
    1 <= target_client_id <= length(source.clients) ||
        throw(ArgumentError("invalid target client id $target_client_id"))

    train_indices = findall(!=(target_client_id), source.train.client_ids)
    val_indices = findall(!=(target_client_id), source.val.client_ids)
    test_indices = collect(eachindex(source.test.client_ids))
    train = _pjm_cold_start_subset_split(source.train, train_indices)
    val = _pjm_cold_start_subset_split(source.val, val_indices)
    test = _pjm_cold_start_subset_split(source.test, test_indices)
    clients = copy(source.clients)
    manifest = _pjm_cold_start_manifest(clients, train, val, test)
    metadata = _pjm_cold_start_metadata(source, clients, train, val, test)

    metadata.train_counts[target_client_id] == 0 ||
        error("target client leaked into donor training data")
    metadata.val_counts[target_client_id] == 0 ||
        error("target client leaked into donor validation data")
    all(
        client_id == target_client_id || metadata.train_counts[client_id] ==
                                         length(source.metadata.train_date_indices) for
        client_id in eachindex(clients)
    ) || error("a donor client does not retain the full training history")

    return PJMBatteryDataset(
        source.config,
        source.panel,
        clients,
        train,
        val,
        test,
        manifest,
        metadata,
    )
end

function _pjm_cold_start_target_dataset(
    source::PJMBatteryDataset,
    target_client_id::Int,
    train_days_per_client::Int,
)
    1 <= target_client_id <= length(source.clients) ||
        throw(ArgumentError("invalid target client id $target_client_id"))
    available_dates = sort(unique(source.train.date_indices))
    0 < train_days_per_client <= length(available_dates) || throw(
        ArgumentError(
            "target train-day budget must be in 1:$(length(available_dates)), got $train_days_per_client",
        ),
    )
    selected_dates = available_dates[(end - train_days_per_client + 1):end]
    selected_set = Set(selected_dates)
    train_indices = findall(
        idx -> source.train.client_ids[idx] == target_client_id &&
               source.train.date_indices[idx] in selected_set,
        eachindex(source.train.client_ids),
    )
    val_indices = findall(==(target_client_id), source.val.client_ids)
    test_indices = findall(==(target_client_id), source.test.client_ids)

    train = _pjm_cold_start_subset_split(
        source.train,
        train_indices;
        reindexed_client_id=1,
    )
    val = _pjm_cold_start_subset_split(
        source.val,
        val_indices;
        reindexed_client_id=1,
    )
    test = _pjm_cold_start_subset_split(
        source.test,
        test_indices;
        reindexed_client_id=1,
    )
    original_client = source.clients[target_client_id]
    clients = [
        PJMBatteryClient(1, original_client.client_name, original_client.capacity),
    ]
    panel = copy(source.panel[source.panel.client_id .== target_client_id, :])
    panel.client_id .= 1
    manifest = _pjm_cold_start_manifest(clients, train, val, test)
    metadata = _pjm_cold_start_metadata(source, clients, train, val, test)

    metadata.train_counts == [train_days_per_client] ||
        error("target dataset does not contain exactly the requested training-day budget")
    metadata.val_counts == [length(source.metadata.val_date_indices)] ||
        error("target validation split changed")
    metadata.test_counts == [length(source.metadata.test_date_indices)] ||
        error("target test split changed")
    isempty(intersect(Set(selected_dates), Set(metadata.val_date_indices))) ||
        error("target train and validation dates overlap")
    isempty(intersect(Set(selected_dates), Set(metadata.test_date_indices))) ||
        error("target train and test dates overlap")

    return (
        dataset=PJMBatteryDataset(
            source.config,
            panel,
            clients,
            train,
            val,
            test,
            manifest,
            metadata,
        ),
        selected_dates=selected_dates,
    )
end

function _pjm_cold_start_fixed_target_reference(
    source::PJMBatteryDataset,
    target_client_id::Int,
)
    n_reference_days = length(source.metadata.train_date_indices)
    full_view =
        _pjm_cold_start_target_dataset(source, target_client_id, n_reference_days)
    client_data = only(prepare_pjm_battery_client_data(full_view.dataset))
    return (
        dataset=full_view.dataset,
        client_data=client_data,
        normalization_reference_days=n_reference_days,
    )
end

function _pjm_cold_start_budget_client_data(
    fixed_reference,
    target_dataset::PJMBatteryDataset,
)
    reference_dataset = fixed_reference.dataset
    reference_data = fixed_reference.client_data
    reference_column_by_sample_id = Dict(
        sample_id => column
        for (column, sample_id) in enumerate(reference_dataset.train.sample_ids)
    )
    train_columns = [
        get(reference_column_by_sample_id, sample_id, 0)
        for sample_id in target_dataset.train.sample_ids
    ]
    all(>(0), train_columns) ||
        error("a budgeted target sample is missing from the fixed target reference")
    reference_dataset.train.theta_true[:, train_columns] ==
    target_dataset.train.theta_true ||
        error("budgeted target labels do not match the fixed target reference")
    reference_dataset.val.sample_ids == target_dataset.val.sample_ids ||
        error("target validation sample ids changed across budgets")
    reference_dataset.test.sample_ids == target_dataset.test.sample_ids ||
        error("target test sample ids changed across budgets")
    reference_dataset.val.theta_true == target_dataset.val.theta_true ||
        error("target validation labels changed across budgets")
    reference_dataset.test.theta_true == target_dataset.test.theta_true ||
        error("target test labels changed across budgets")

    return PJMBatteryClientData(
        1,
        reference_data.client_name,
        reference_data.instance,
        reference_data.normalization,
        reference_data.train_x[:, train_columns],
        reference_data.train_theta_true[:, train_columns],
        reference_data.train_y_true[:, train_columns],
        reference_data.val_x,
        reference_data.val_theta_true,
        reference_data.val_y_true,
        reference_data.test_x,
        reference_data.test_theta_true,
        reference_data.test_y_true,
    )
end

function _pjm_cold_start_local_pjm_battery(
    client_data::Vector{PJMBatteryClientData};
    objective::PJMBatteryObjective=PJMRSPOPlusObjective(),
    config::PJMBatteryTrainingConfig=PJMBatteryTrainingConfig(),
    model=nothing,
)
    length(client_data) == 1 ||
        throw(ArgumentError("cold-start local training expects exactly one target client"))
    _validate_pjm_training_config(config, objective)

    rng = MersenneTwister(config.seed)
    model_rng = MersenneTwister(config.seed)
    base_model =
        isnothing(model) ? build_pjm_battery_model(
            length(PJM_FEATURE_COLUMNS),
            length(PJM_TARGET_COLUMNS);
            hidden_dim=config.hidden_dim,
            rng=model_rng,
        ) : Flux.f64(deepcopy(model))
    local_model = Flux.f64(deepcopy(base_model))
    client = only(client_data)
    client_lambda0 =
        config.per_client_lambda && _pjm_objective_uses_lambda(objective) ?
        only(compute_pjm_per_client_lambda0(client_data; fallback=config.lambda0)) :
        config.lambda0
    client_lr0 =
        config.per_client_lr && _pjm_objective_uses_lambda(objective) ?
        config.lr_lambda_alpha * client_lambda0 / 4 : config.lr0
    opt_state = Flux.setup(_pjm_optimizer_rule(config, client_lr0), local_model)
    objective_state = _initial_pjm_objective_state(objective, client, config)
    validation_monitor =
        _pjm_objective_tracks_bound(objective) ?
        build_pjm_validation_monitor(client_data, config; rng=rng, client_ids=[1]) :
        PJMBatteryValidationMonitor(PJMBatteryValidationClientData[])

    frozen = Ref(false)
    lambda_sched =
        _pjm_objective_uses_lambda(objective) ?
        create_inverse_time_scheduler(
            client_lambda0,
            config.kappa_lambda;
            frozen=frozen,
        ) : nothing
    lr_sched =
        create_inverse_time_scheduler(client_lr0, config.kappa_lr; frozen=frozen)
    round_losses = Float64[]
    lambda_values = Float64[]
    lr_values = Float64[]
    bound_values = Float64[]
    frozen_after_round = nothing

    for round in 1:config.rounds
        if _pjm_objective_tracks_bound(objective) &&
           !isnothing(frozen_after_round) &&
           round > frozen_after_round + config.stop_after_freeze_rounds
            continue
        end

        lambda =
            _pjm_objective_uses_lambda(objective) ?
            next_schedule_value!(lambda_sched) : NaN
        lr = next_schedule_value!(lr_sched)
        push!(lambda_values, lambda)
        push!(lr_values, lr)
        Optimisers.adjust!(opt_state, lr)
        round_loss, objective_state = train_pjm_client_model!(
            local_model,
            opt_state,
            client,
            objective,
            config,
            rng;
            state=objective_state,
            lambda=lambda,
        )
        push!(round_losses, round_loss)

        bound = _compute_pjm_objective_bound(
            objective,
            local_model,
            validation_monitor,
            config;
            lambda=lambda,
        )
        push!(bound_values, bound)
        if _pjm_objective_tracks_bound(objective) &&
           !frozen[] &&
           bound <= config.freeze_tau
            frozen[] = true
            frozen_after_round = round
        end
    end

    per_client_lambda0 =
        config.per_client_lambda ? [client_lambda0] : nothing
    return PJMBatteryLocalTrainingResult(
        _pjm_objective_method(objective, :local),
        [
            PJMBatteryLocalClientTrainingResult(
                1,
                local_model,
                round_losses,
                PJMBatterySchedulerTrace(
                    lambda_values,
                    lr_values,
                    bound_values,
                    frozen_after_round,
                    per_client_lambda0,
                ),
            ),
        ],
    )
end

function _pjm_cold_start_evaluate_warm_model(
    model,
    target_data::PJMBatteryClientData,
    trace::PJMBatterySchedulerTrace,
)
    theta_pred = model(target_data.test_x)
    return _evaluate_pjm_client_metrics(
        target_data,
        theta_pred,
        trace;
        include_regularized_decision_gap=false,
    )
end

function _pjm_cold_start_local_evaluation(
    result::PJMBatteryLocalTrainingResult,
    target_data::PJMBatteryClientData,
)
    return evaluate_local_pjm_battery(
        result,
        [target_data];
        include_regularized_decision_gap=false,
    ).clients[1]
end

function _pjm_cold_start_float_matrix_sha256(matrix::Matrix{Float64})
    return bytes2hex(sha256(reinterpret(UInt8, vec(matrix))))
end

function _pjm_cold_start_vector_sha256(values::AbstractVector)
    return bytes2hex(sha256(join(string.(values), '\0')))
end

function _pjm_cold_start_round_count(result::PJMBatteryLocalTrainingResult)
    return length(result.clients[1].round_losses)
end

function run_single_pjm_cold_start_target_seed(;
    seed::Int,
    target_client_id::Int,
    train_days::Vector{Int},
    split_fractions::NTuple{3,Float64},
    data_root::String,
)
    source = load_pjm_battery_dataset(
        PJMBatteryDatasetConfig(
            data_root=data_root,
            split_fractions=split_fractions,
        ),
    )
    target_client = source.clients[target_client_id]
    donor_dataset = _pjm_cold_start_donor_dataset(source, target_client_id)
    objective = PJMRSPOPlusObjective()
    shared_init_model = build_pjm_battery_model(
        length(PJM_FEATURE_COLUMNS),
        length(PJM_TARGET_COLUMNS);
        hidden_dim=64,
        rng=Random.MersenneTwister(seed),
    )
    warm_result = fed_pjm_battery(
        donor_dataset;
        objective=objective,
        config=_pjm_budget_warm_config(seed),
        model=shared_init_model,
    )
    fixed_reference =
        _pjm_cold_start_fixed_target_reference(source, target_client_id)
    fixed_target_data = fixed_reference.client_data
    warm_eval = _pjm_cold_start_evaluate_warm_model(
        warm_result.model,
        fixed_target_data,
        warm_result.trace,
    )
    test_sample_ids_sha256 = _pjm_cold_start_vector_sha256(
        fixed_reference.dataset.test.sample_ids,
    )
    test_x_sha256 =
        _pjm_cold_start_float_matrix_sha256(fixed_target_data.test_x)
    test_theta_sha256 =
        _pjm_cold_start_float_matrix_sha256(fixed_target_data.test_theta_true)
    normalization_sha256 = _pjm_cold_start_float_matrix_sha256(
        hcat(
            fixed_target_data.normalization.mean,
            fixed_target_data.normalization.std,
        ),
    )

    rows = NamedTuple[]
    for budget in train_days
        target_view = _pjm_cold_start_target_dataset(source, target_client_id, budget)
        target_dataset = target_view.dataset
        target_data =
            _pjm_cold_start_budget_client_data(fixed_reference, target_dataset)
        target_data.val_x === fixed_target_data.val_x ||
            error("validation tensor was not reused across target budgets")
        target_data.test_x === fixed_target_data.test_x ||
            error("test tensor was not reused across target budgets")

        local15_result = _pjm_cold_start_local_pjm_battery(
            [target_data];
            objective=objective,
            config=_pjm_budget_personal_config(seed),
            model=shared_init_model,
        )
        local15_eval = _pjm_cold_start_local_evaluation(local15_result, target_data)

        local45_result = _pjm_cold_start_local_pjm_battery(
            [target_data];
            objective=objective,
            config=_pjm_budget_local_config(seed),
            model=shared_init_model,
        )
        local45_eval = _pjm_cold_start_local_evaluation(local45_result, target_data)

        personalized_result = _pjm_cold_start_local_pjm_battery(
            [target_data];
            objective=objective,
            config=_pjm_budget_personal_config(seed),
            model=warm_result.model,
        )
        personalized_eval =
            _pjm_cold_start_local_evaluation(personalized_result, target_data)

        push!(
            rows,
            (
                train_days_per_target=budget,
                seed=seed,
                target_client_id=target_client_id,
                target_client_name=target_client.client_name,
                target_capacity=target_client.capacity,
                n_donor_clients=length(source.clients) - 1,
                donor_train_days_per_client=length(source.metadata.train_date_indices),
                target_date_selection="last_n_before_validation",
                selected_target_train_date_indices=join(target_view.selected_dates, ";"),
                target_normalization="fixed_full_prevalidation_target_features",
                target_normalization_reference_days=
                    fixed_reference.normalization_reference_days,
                normalization_sha256=normalization_sha256,
                target_validation_days=length(source.metadata.val_date_indices),
                target_test_days=length(source.metadata.test_date_indices),
                test_sample_ids_sha256=test_sample_ids_sha256,
                test_x_sha256=test_x_sha256,
                test_theta_sha256=test_theta_sha256,
                local15_absolute_regret=local15_eval.absolute_regret,
                local45_absolute_regret=local45_eval.absolute_regret,
                warm_only_absolute_regret=warm_eval.absolute_regret,
                fed_personalized_absolute_regret=personalized_eval.absolute_regret,
                fed_gain_vs_local15=
                    local15_eval.absolute_regret - personalized_eval.absolute_regret,
                fed_gain_vs_local45=
                    local45_eval.absolute_regret - personalized_eval.absolute_regret,
                local15_mse=local15_eval.mse,
                local45_mse=local45_eval.mse,
                warm_only_mse=warm_eval.mse,
                fed_personalized_mse=personalized_eval.mse,
                local15_rounds_executed=_pjm_cold_start_round_count(local15_result),
                local45_rounds_executed=_pjm_cold_start_round_count(local45_result),
                fed_personalization_rounds_executed=
                    _pjm_cold_start_round_count(personalized_result),
                local15_freeze_round=something(local15_eval.frozen_after_round, ""),
                local45_freeze_round=something(local45_eval.frozen_after_round, ""),
                fed_personalization_freeze_round=
                    something(personalized_eval.frozen_after_round, ""),
            ),
        )
    end
    return rows
end
