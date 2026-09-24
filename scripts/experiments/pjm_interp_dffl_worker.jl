using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Dates
using Random
using Statistics

include(joinpath(@__DIR__, "..", "..", "src", "models", "knapsack_oracles.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "models", "battery_oracles.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "models", "RSPOPlusLoss.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "models", "scheduling.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "datagen", "pjm_battery.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "models", "pjm_battery_training.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "models", "pjm_interp_dffl.jl"))

function _pjm_interp_split_date_bounds(split::PJMBatterySplit)
    dates = sort(unique(split.dates))
    return (first=first(dates), last=last(dates), count=length(dates))
end

function run_single_pjm_interp_dffl_seed(;
    seed::Int,
    split_fractions::NTuple{3,Float64},
    data_root::String,
    lambda_grid::Vector{Float64},
)
    dataset_config = PJMBatteryDatasetConfig(
        data_root=data_root,
        split_fractions=split_fractions,
    )
    dataset = load_pjm_battery_dataset(dataset_config)
    config = PJMInterpDFFLConfig(seed=seed, lambda_grid=lambda_grid)

    started_at = Dates.now()
    train_started = time()
    result = train_pjm_interp_dffl(dataset; config=config)
    training_seconds = time() - train_started

    evaluation_started = time()
    evaluations = evaluate_all_pjm_interp_dffl(result)
    evaluation_seconds = time() - evaluation_started
    finished_at = Dates.now()

    seed_metrics = NamedTuple[]
    client_metrics = NamedTuple[]
    for selector in PJM_INTERP_DFFL_SELECTORS
        evaluation = evaluations[selector]
        aggregate = evaluation.aggregate
        push!(
            seed_metrics,
            (
                seed=seed,
                selector=string(selector),
                aggregate_mse_mean=aggregate.mse.mean,
                aggregate_mse_std_clients=aggregate.mse.std,
                aggregate_absolute_regret_mean=aggregate.absolute_regret.mean,
                aggregate_absolute_regret_std_clients=aggregate.absolute_regret.std,
                aggregate_relative_regret_mean=aggregate.relative_regret.mean,
                aggregate_relative_regret_std_clients=aggregate.relative_regret.std,
                n_clients=aggregate.n_clients,
                total_test_samples=aggregate.total_test_samples,
            ),
        )
        for client in evaluation.clients
            push!(
                client_metrics,
                (
                    seed=seed,
                    selector=string(selector),
                    client_id=client.client_id,
                    client_name=client.client_name,
                    n_test_samples=client.n_test_samples,
                    lambda=client.lambda,
                    mse=client.mse,
                    absolute_regret=client.absolute_regret,
                    relative_regret=client.relative_regret,
                ),
            )
        end
    end

    lambda_selections = NamedTuple[]
    calibration_curves = NamedTuple[]
    for client in result.client_data
        calibration = result.calibrations[client.client_id]
        push!(
            lambda_selections,
            (
                seed=seed,
                client_id=client.client_id,
                client_name=client.client_name,
                n_validation_samples=size(client.val_x, 2),
                lambda_spo_plus=calibration.lambda_spo_plus,
                lambda_mse=calibration.lambda_mse,
                best_spo_plus_loss=minimum(calibration.spo_plus_losses),
                best_mse_loss=minimum(calibration.mse_losses),
            ),
        )
        for (idx, lambda) in pairs(config.lambda_grid)
            push!(
                calibration_curves,
                (
                    seed=seed,
                    client_id=client.client_id,
                    client_name=client.client_name,
                    lambda=lambda,
                    spo_plus_loss=calibration.spo_plus_losses[idx],
                    mse_loss=calibration.mse_losses[idx],
                    selected_by_spo_plus=lambda == calibration.lambda_spo_plus,
                    selected_by_mse=lambda == calibration.lambda_mse,
                ),
            )
        end
    end

    endpoint_training = NamedTuple[]
    for round in eachindex(result.federated_endpoint.round_losses)
        push!(
            endpoint_training,
            (
                seed=seed,
                endpoint="federated",
                client_id=0,
                round=round,
                loss=result.federated_endpoint.round_losses[round],
                selected_clients=join(
                    result.federated_endpoint.selected_clients[round],
                    ";",
                ),
            ),
        )
    end
    for client in result.local_endpoint.clients
        for round in eachindex(client.round_losses)
            push!(
                endpoint_training,
                (
                    seed=seed,
                    endpoint="local",
                    client_id=client.client_id,
                    round=round,
                    loss=client.round_losses[round],
                    selected_clients=string(client.client_id),
                ),
            )
        end
    end

    train_dates = _pjm_interp_split_date_bounds(dataset.train)
    val_dates = _pjm_interp_split_date_bounds(dataset.val)
    test_dates = _pjm_interp_split_date_bounds(dataset.test)
    metadata = (
        protocol=string(PJM_INTERP_DFFL_PROTOCOL),
        seed=seed,
        split_train=split_fractions[1],
        split_val=split_fractions[2],
        split_test=split_fractions[3],
        n_clients=length(dataset.clients),
        n_dates=dataset.metadata.n_dates,
        total_samples=dataset.metadata.n_samples,
        train_dates=train_dates.count,
        validation_dates=val_dates.count,
        test_dates=test_dates.count,
        train_samples_per_client=join(dataset.metadata.train_counts, ";"),
        validation_samples_per_client=join(dataset.metadata.val_counts, ";"),
        test_samples_per_client=join(dataset.metadata.test_counts, ";"),
        train_first_date=string(train_dates.first),
        train_last_date=string(train_dates.last),
        validation_first_date=string(val_dates.first),
        validation_last_date=string(val_dates.last),
        test_first_date=string(test_dates.first),
        test_last_date=string(test_dates.last),
        hidden_dim=config.hidden_dim,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        clip_norm=config.clip_norm,
        spo_alpha=config.spo_alpha,
        federated_rounds=config.federated_rounds,
        federated_local_epochs=config.federated_local_epochs,
        federated_client_fraction=config.federated_client_fraction,
        local_rounds=config.local_rounds,
        local_epochs=config.local_epochs,
        lambda_grid=join(config.lambda_grid, ";"),
        lambda_grid_size=length(config.lambda_grid),
        initialization_seed=seed,
        federated_training_seed=result.federated_config.seed,
        local_training_seed=result.local_config.seed,
        training_seconds=training_seconds,
        evaluation_seconds=evaluation_seconds,
        total_seconds=training_seconds + evaluation_seconds,
        started_at=string(started_at),
        finished_at=string(finished_at),
    )

    return (
        seed_metrics=seed_metrics,
        client_metrics=client_metrics,
        lambda_selections=lambda_selections,
        calibration_curves=calibration_curves,
        endpoint_training=endpoint_training,
        metadata=metadata,
    )
end
