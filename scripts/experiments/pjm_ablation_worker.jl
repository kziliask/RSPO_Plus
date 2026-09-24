using Pkg
using Random
using Statistics

Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "..", "..", "src", "models", "knapsack_oracles.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "models", "battery_oracles.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "models", "RSPOPlusLoss.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "models", "scheduling.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "datagen", "pjm_battery.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "models", "pjm_battery_training.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "experiments", "pjm_battery.jl"))

const PJM_ABLATION_ROW_ORDER = [
    :mse,
    :spo_plus,
    :rspo_plus,
    :rspo_plus_personalization,
    :rspo_plus_personalization_per_client,
    :rspo_plus_personalization_per_client_scheduling,
]

const PJM_ABLATION_ROW_LABEL = Dict(
    :mse => "MSE",
    :spo_plus => "SPO+",
    :rspo_plus => "RSPO+",
    :rspo_plus_personalization => "RSPO+ w/ personalization",
    :rspo_plus_personalization_per_client => "RSPO+ w/ personalization + per-client lambda/lr",
    :rspo_plus_personalization_per_client_scheduling => "RSPO+ w/ personalization + per-client lambda/lr + scheduling",
)

function build_pjm_ablation_warm_config(;
    seed::Int,
    lambda0::Float64,
    per_client_lambda::Bool,
    lr0::Float64,
    per_client_lr::Bool,
    lr_lambda_alpha::Float64,
)
    return PJMBatteryTrainingConfig(
        seed=seed,
        hidden_dim=64,
        rounds=10,
        local_epochs=3,
        batch_size=64,
        client_fraction=0.4,
        validation_client_fraction=0.4,
        validation_max_samples_per_client=8,
        lambda0=lambda0,
        kappa_lambda=0.0,
        per_client_lambda=per_client_lambda,
        lr0=lr0,
        kappa_lr=0.0,
        per_client_lr=per_client_lr,
        lr_lambda_alpha=lr_lambda_alpha,
        clip_norm=1.0,
        freeze_tau=1.0,
        freeze_eps=1e-8,
        stop_after_freeze_rounds=3,
        shuffle_batches=true,
        use_warm_start=true,
    )
end

function build_pjm_ablation_personalization_config(;
    seed::Int,
    scheduled::Bool,
    personal_kappa::Float64,
    personal_kappa_lr::Float64,
    personal_lambda0::Float64,
    personal_freeze_tau::Float64,
    personal_lr0::Float64,
    personal_lr_lambda_alpha::Float64,
)
    return PJMBatteryTrainingConfig(
        seed=seed,
        hidden_dim=64,
        rounds=15,
        local_epochs=1,
        batch_size=64,
        client_fraction=1.0,
        validation_client_fraction=1.0,
        validation_max_samples_per_client=8,
        lambda0=personal_lambda0,
        kappa_lambda=scheduled ? personal_kappa : 0.0,
        per_client_lambda=true,
        lr0=personal_lr0,
        kappa_lr=scheduled ? personal_kappa_lr : 0.0,
        per_client_lr=personal_lr_lambda_alpha > 0,
        lr_lambda_alpha=personal_lr_lambda_alpha,
        clip_norm=1.0,
        freeze_tau=scheduled ? personal_freeze_tau : NaN,
        freeze_eps=1e-8,
        stop_after_freeze_rounds=3,
        shuffle_batches=true,
        use_warm_start=true,
    )
end

function build_pjm_ablation_calibrated_config(;
    seed::Int,
    calibrated_lambda0::Float64,
    calibrated_lr0::Float64,
    rounds::Int=15,
)
    return PJMBatteryTrainingConfig(
        seed=seed,
        hidden_dim=64,
        rounds=rounds,
        local_epochs=1,
        batch_size=64,
        client_fraction=1.0,
        validation_client_fraction=1.0,
        validation_max_samples_per_client=8,
        lambda0=calibrated_lambda0,
        kappa_lambda=0.0,
        per_client_lambda=false,
        lr0=calibrated_lr0,
        kappa_lr=0.0,
        per_client_lr=false,
        lr_lambda_alpha=0.0,
        clip_norm=1.0,
        freeze_tau=NaN,
        freeze_eps=1e-8,
        stop_after_freeze_rounds=3,
        shuffle_batches=true,
        use_warm_start=true,
    )
end

function pjm_ablation_config_id(method_key::Symbol, config::PJMBatteryTrainingConfig)
    return string(method_key, "__", _pjm_config_slug(config))
end

function _pjm_ablation_row(
    method_key::Symbol,
    evaluation::PJMBatteryEvaluationResult,
    config::PJMBatteryTrainingConfig;
    seed::Int,
    split_fractions::NTuple{3,Float64},
    objective_name::Symbol,
    phase::Symbol,
    uses_personalization::Bool,
    uses_scheduling::Bool,
    export_dir::AbstractString,
    warm_start_export_dir::AbstractString="",
)
    aggregate = evaluation.aggregate
    scheduler = aggregate.scheduler

    return (
        method_key=string(method_key),
        method_label=PJM_ABLATION_ROW_LABEL[method_key],
        seed=seed,
        objective=string(objective_name),
        phase=string(phase),
        uses_personalization=uses_personalization,
        uses_scheduling=uses_scheduling,
        split_train=split_fractions[1],
        split_val=split_fractions[2],
        split_test=split_fractions[3],
        training_seed=config.seed,
        rounds=config.rounds,
        local_epochs=config.local_epochs,
        batch_size=config.batch_size,
        client_fraction=config.client_fraction,
        validation_client_fraction=config.validation_client_fraction,
        validation_max_samples_per_client=config.validation_max_samples_per_client,
        lambda0=config.lambda0,
        kappa_lambda=config.kappa_lambda,
        per_client_lambda=config.per_client_lambda,
        lr0=config.lr0,
        kappa_lr=config.kappa_lr,
        per_client_lr=config.per_client_lr,
        lr_lambda_alpha=config.lr_lambda_alpha,
        freeze_tau=config.freeze_tau,
        stop_after_freeze_rounds=config.stop_after_freeze_rounds,
        aggregate_mse_mean=aggregate.mse.mean,
        aggregate_mse_std_clients=aggregate.mse.std,
        aggregate_absolute_regret_mean=aggregate.absolute_regret.mean,
        aggregate_absolute_regret_std_clients=aggregate.absolute_regret.std,
        aggregate_relative_regret_mean=aggregate.relative_regret.mean,
        aggregate_relative_regret_std_clients=aggregate.relative_regret.std,
        aggregate_reg_decision_gap_mean=aggregate.regularized_decision_gap.mean,
        aggregate_reg_decision_gap_std_clients=aggregate.regularized_decision_gap.std,
        final_lambda_mean=scheduler.final_lambda.mean,
        final_lambda_std_clients=scheduler.final_lambda.std,
        final_lr_mean=scheduler.final_lr.mean,
        final_lr_std_clients=scheduler.final_lr.std,
        final_bound_mean=scheduler.final_bound.mean,
        final_bound_std_clients=scheduler.final_bound.std,
        freeze_round_mean=scheduler.freeze_round.mean,
        freeze_round_std_clients=scheduler.freeze_round.std,
        frozen_fraction=scheduler.frozen_fraction,
        n_clients=aggregate.n_clients,
        total_test_samples=aggregate.total_test_samples,
        export_dir=String(export_dir),
        warm_start_export_dir=String(warm_start_export_dir),
    )
end

function run_single_pjm_ablation_seed(;
    seed::Int,
    split_fractions::NTuple{3,Float64},
    data_root::String,
    global_lambda0::Float64,
    global_lr0::Float64,
    global_lr_lambda_alpha::Float64,
    personal_kappa::Float64,
    personal_kappa_lr::Float64,
    personal_lambda0::Float64,
    personal_freeze_tau::Float64,
    personal_lr0::Float64,
    personal_lr_lambda_alpha::Float64,
    spo_alpha::Float64,
    export_root::String,
    fixed_demand::Union{Nothing,Float64}=nothing,
)
    dataset_config = PJMBatteryDatasetConfig(
        data_root=data_root,
        split_fractions=split_fractions,
        fixed_demand=fixed_demand,
    )
    dataset = load_pjm_battery_dataset(dataset_config)
    shared_init_model = build_pjm_battery_model(
        length(PJM_FEATURE_COLUMNS),
        length(PJM_TARGET_COLUMNS);
        hidden_dim=64,
        rng=Random.MersenneTwister(seed),
    )

    rows = NamedTuple[]

    for (method_key, objective_name, objective) in (
        (:mse, :mse, PJMMSEObjective()),
        (:spo_plus, :spo_plus, PJMSPOPlusObjective(; α=spo_alpha)),
    )
        warm_config = build_pjm_ablation_warm_config(
            seed=seed,
            lambda0=NaN,
            per_client_lambda=false,
            lr0=global_lr0,
            per_client_lr=false,
            lr_lambda_alpha=0.0,
        )
        result = fed_pjm_battery(
            dataset;
            objective=objective,
            config=warm_config,
            model=shared_init_model,
        )
        evaluation = evaluate_fed_pjm_battery(result)
        exported = export_pjm_battery_run(
            result,
            dataset;
            config=warm_config,
            evaluation=evaluation,
            root=export_root,
            config_id=pjm_ablation_config_id(method_key, warm_config),
            seed_tag=string("seed", seed),
        )
        push!(
            rows,
            _pjm_ablation_row(
                method_key,
                evaluation,
                warm_config;
                seed=seed,
                split_fractions=split_fractions,
                objective_name=objective_name,
                phase=:fed,
                uses_personalization=false,
                uses_scheduling=false,
                export_dir=exported.dir,
            ),
        )
    end

    rspo_objective = PJMRSPOPlusObjective()
    rspo_per_client_warm_config = build_pjm_ablation_warm_config(
        seed=seed,
        lambda0=global_lambda0,
        per_client_lambda=true,
        lr0=global_lr0,
        per_client_lr=global_lr_lambda_alpha > 0,
        lr_lambda_alpha=global_lr_lambda_alpha,
    )
    rspo_per_client_warm_result = fed_pjm_battery(
        dataset;
        objective=rspo_objective,
        config=rspo_per_client_warm_config,
        model=shared_init_model,
    )
    rspo_per_client_warm_evaluation = evaluate_fed_pjm_battery(rspo_per_client_warm_result)
    rspo_per_client_warm_export = export_pjm_battery_run(
        rspo_per_client_warm_result,
        dataset;
        config=rspo_per_client_warm_config,
        evaluation=rspo_per_client_warm_evaluation,
        root=export_root,
        config_id=pjm_ablation_config_id(
            :rspo_plus_personalization_per_client_warm_start,
            rspo_per_client_warm_config,
        ),
        seed_tag=string("seed", seed),
    )
    per_client_lambda0s = let pcl = rspo_per_client_warm_result.trace.per_client_lambda0
        isnothing(pcl) && error("expected RSPO+ warm-start to expose per-client lambda0s")
        copy(pcl)
    end
    per_client_lr0s = global_lr_lambda_alpha > 0 ? [
        global_lr_lambda_alpha * lambda0 / 4 for lambda0 in per_client_lambda0s
    ] : fill(global_lr0, length(per_client_lambda0s))
    mean_lambda0 = mean(per_client_lambda0s)
    mean_lr0 = mean(per_client_lr0s)

    rspo_shared_mean_warm_config = build_pjm_ablation_warm_config(
        seed=seed,
        lambda0=mean_lambda0,
        per_client_lambda=false,
        lr0=mean_lr0,
        per_client_lr=false,
        lr_lambda_alpha=0.0,
    )
    rspo_shared_mean_warm_result = fed_pjm_battery(
        dataset;
        objective=rspo_objective,
        config=rspo_shared_mean_warm_config,
        model=shared_init_model,
    )
    rspo_shared_mean_warm_evaluation = evaluate_fed_pjm_battery(rspo_shared_mean_warm_result)
    rspo_shared_mean_warm_export = export_pjm_battery_run(
        rspo_shared_mean_warm_result,
        dataset;
        config=rspo_shared_mean_warm_config,
        evaluation=rspo_shared_mean_warm_evaluation,
        root=export_root,
        config_id=pjm_ablation_config_id(:rspo_plus, rspo_shared_mean_warm_config),
        seed_tag=string("seed", seed),
    )
    push!(
        rows,
        _pjm_ablation_row(
            :rspo_plus,
            rspo_shared_mean_warm_evaluation,
            rspo_shared_mean_warm_config;
            seed=seed,
            split_fractions=split_fractions,
            objective_name=:rspo_plus,
            phase=:fed,
            uses_personalization=false,
            uses_scheduling=false,
            export_dir=rspo_shared_mean_warm_export.dir,
        ),
    )

    personalization_seed = seed + 1
    shared_mean_personalization_config = build_pjm_ablation_calibrated_config(
        seed=personalization_seed,
        calibrated_lambda0=mean_lambda0,
        calibrated_lr0=mean_lr0,
    )
    shared_mean_personalization_result = local_pjm_battery(
        dataset;
        objective=rspo_objective,
        config=shared_mean_personalization_config,
        model=rspo_shared_mean_warm_result.model,
    )
    shared_mean_personalization_evaluation =
        evaluate_local_pjm_battery(shared_mean_personalization_result, dataset)
    shared_mean_personalization_export = export_pjm_battery_run(
        shared_mean_personalization_result,
        dataset;
        config=shared_mean_personalization_config,
        evaluation=shared_mean_personalization_evaluation,
        root=export_root,
        config_id=pjm_ablation_config_id(:rspo_plus_personalization, shared_mean_personalization_config),
        seed_tag=string("seed", seed),
    )
    push!(
        rows,
        _pjm_ablation_row(
            :rspo_plus_personalization,
            shared_mean_personalization_evaluation,
            shared_mean_personalization_config;
            seed=seed,
            split_fractions=split_fractions,
            objective_name=:rspo_plus,
            phase=:local,
            uses_personalization=true,
            uses_scheduling=false,
            export_dir=shared_mean_personalization_export.dir,
            warm_start_export_dir=rspo_shared_mean_warm_export.dir,
        ),
    )

    per_client_personalization_config = build_pjm_ablation_personalization_config(
        seed=personalization_seed,
        scheduled=false,
        personal_kappa=personal_kappa,
        personal_kappa_lr=personal_kappa_lr,
        personal_lambda0=personal_lambda0,
        personal_freeze_tau=personal_freeze_tau,
        personal_lr0=personal_lr0,
        personal_lr_lambda_alpha=personal_lr_lambda_alpha,
    )
    per_client_personalization_result = local_pjm_battery(
        dataset;
        objective=rspo_objective,
        config=per_client_personalization_config,
        model=rspo_per_client_warm_result.model,
    )
    per_client_personalization_evaluation =
        evaluate_local_pjm_battery(per_client_personalization_result, dataset)
    per_client_personalization_export = export_pjm_battery_run(
        per_client_personalization_result,
        dataset;
        config=per_client_personalization_config,
        evaluation=per_client_personalization_evaluation,
        root=export_root,
        config_id=pjm_ablation_config_id(
            :rspo_plus_personalization_per_client,
            per_client_personalization_config,
        ),
        seed_tag=string("seed", seed),
    )
    push!(
        rows,
        _pjm_ablation_row(
            :rspo_plus_personalization_per_client,
            per_client_personalization_evaluation,
            per_client_personalization_config;
            seed=seed,
            split_fractions=split_fractions,
            objective_name=:rspo_plus,
            phase=:local,
            uses_personalization=true,
            uses_scheduling=false,
            export_dir=per_client_personalization_export.dir,
            warm_start_export_dir=rspo_per_client_warm_export.dir,
        ),
    )

    per_client_schedule_config = build_pjm_ablation_personalization_config(
        seed=personalization_seed,
        scheduled=true,
        personal_kappa=personal_kappa,
        personal_kappa_lr=personal_kappa_lr,
        personal_lambda0=personal_lambda0,
        personal_freeze_tau=personal_freeze_tau,
        personal_lr0=personal_lr0,
        personal_lr_lambda_alpha=personal_lr_lambda_alpha,
    )
    per_client_schedule_result = local_pjm_battery(
        dataset;
        objective=rspo_objective,
        config=per_client_schedule_config,
        model=rspo_per_client_warm_result.model,
    )
    per_client_schedule_evaluation = evaluate_local_pjm_battery(per_client_schedule_result, dataset)
    per_client_schedule_export = export_pjm_battery_run(
        per_client_schedule_result,
        dataset;
        config=per_client_schedule_config,
        evaluation=per_client_schedule_evaluation,
        root=export_root,
        config_id=pjm_ablation_config_id(
            :rspo_plus_personalization_per_client_scheduling,
            per_client_schedule_config,
        ),
        seed_tag=string("seed", seed),
    )
    push!(
        rows,
        _pjm_ablation_row(
            :rspo_plus_personalization_per_client_scheduling,
            per_client_schedule_evaluation,
            per_client_schedule_config;
            seed=seed,
            split_fractions=split_fractions,
            objective_name=:rspo_plus,
            phase=:local,
            uses_personalization=true,
            uses_scheduling=true,
            export_dir=per_client_schedule_export.dir,
            warm_start_export_dir=rspo_per_client_warm_export.dir,
        ),
    )

    return rows
end
