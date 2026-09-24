#!/usr/bin/env julia
using Pkg
const PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(PROJECT_ROOT)

using Distributed
using Dates

function parse_int_list(text::AbstractString)
    return [parse(Int, strip(token)) for token in split(text, ',') if !isempty(strip(token))]
end

function parse_split_fractions(text::AbstractString)
    values = [parse(Float64, strip(token)) for token in split(text, ',') if !isempty(strip(token))]
    length(values) == 3 || error("expected three comma-separated split fractions, got `$text`")
    return (values[1], values[2], values[3])
end

function parse_cli(args::Vector{String})
    opts = Dict{Symbol,Any}(
        :global_lambda0 => 2.0,
        :global_lr0 => 1e-3,
        :global_lr_lambda_alpha => 0.01,
        :objective => "all",
        :spo_alpha => 2.0,
        :perturbed_nb_samples => 10,
        :perturbed_epsilon => 1.0,
        :perturbed_threaded => false,
        :perturbed_seed => nothing,
        :personal_kappa => 1.0,
        :personal_kappa_lr => 0.1,
        :personal_lambda0 => 55.0,
        :personal_freeze_tau => 0.03,
        :personal_lr0 => 1e-3,
        :personal_lr_lambda_alpha => 0.001,
        :seed_values => [42, 69, 1993, 2000, 2026],
        :split => (0.5, 0.1, 0.4),
        :data_root => joinpath("src", "data", "exp3"),
        :nprocs => 0,
        :outfile => joinpath("results", "experiment3", "pjm_battery_results.csv"),
    )

    idx = 1
    while idx <= length(args)
        flag = args[idx]
        if flag == "--nprocs"
            opts[:nprocs] = parse(Int, args[idx + 1])
            idx += 2
        elseif flag == "--objective"
            opts[:objective] = args[idx + 1]
            idx += 2
        elseif flag == "--spo-alpha"
            opts[:spo_alpha] = parse(Float64, args[idx + 1])
            idx += 2
        elseif flag == "--perturbed-nb-samples"
            opts[:perturbed_nb_samples] = parse(Int, args[idx + 1])
            idx += 2
        elseif flag == "--perturbed-epsilon"
            opts[:perturbed_epsilon] = parse(Float64, args[idx + 1])
            idx += 2
        elseif flag == "--perturbed-threaded"
            opts[:perturbed_threaded] = true
            idx += 1
        elseif flag == "--perturbed-seed"
            opts[:perturbed_seed] = parse(Int, args[idx + 1])
            idx += 2
        elseif flag == "--global-lambda0"
            opts[:global_lambda0] = parse(Float64, args[idx + 1])
            idx += 2
        elseif flag == "--global-lr0"
            opts[:global_lr0] = parse(Float64, args[idx + 1])
            idx += 2
        elseif flag == "--global-lr-lambda-alpha"
            opts[:global_lr_lambda_alpha] = parse(Float64, args[idx + 1])
            idx += 2
        elseif flag == "--personal-kappa"
            opts[:personal_kappa] = parse(Float64, args[idx + 1])
            idx += 2
        elseif flag == "--personal-kappa-lr"
            opts[:personal_kappa_lr] = parse(Float64, args[idx + 1])
            idx += 2
        elseif flag == "--personal-lambda0"
            opts[:personal_lambda0] = parse(Float64, args[idx + 1])
            idx += 2
        elseif flag == "--personal-freeze-tau"
            opts[:personal_freeze_tau] = parse(Float64, args[idx + 1])
            idx += 2
        elseif flag == "--personal-lr0"
            opts[:personal_lr0] = parse(Float64, args[idx + 1])
            idx += 2
        elseif flag == "--personal-lr-lambda-alpha"
            opts[:personal_lr_lambda_alpha] = parse(Float64, args[idx + 1])
            idx += 2
        elseif flag == "--seed-values"
            opts[:seed_values] = parse_int_list(args[idx + 1])
            idx += 2
        elseif flag == "--split"
            opts[:split] = parse_split_fractions(args[idx + 1])
            idx += 2
        elseif flag == "--data-root"
            opts[:data_root] = args[idx + 1]
            idx += 2
        elseif flag == "--outfile"
            opts[:outfile] = args[idx + 1]
            idx += 2
        else
            error("unknown flag `$flag`")
        end
    end

    return opts
end

function parse_objectives(spec::AbstractString)
    tokens = [canonical_objective_token(Symbol(strip(token))) for token in split(spec, ',') if !isempty(strip(token))]
    isempty(tokens) && error("at least one objective must be provided")
    :all in tokens &&
        return [:rspo_plus, :spo_plus, :perturbed_fyl_mult, :dpo_perturbed_mse_mult, :mse]

    valid = Set([:rspo_plus, :spo_plus, :perturbed_fyl_mult, :dpo_perturbed_mse_mult, :mse])
    for token in tokens
        token in valid || error("unknown objective `$token`")
    end
    return unique(tokens)
end

function canonical_objective_token(token::Symbol)
    if token === :pfyl || token === :perturbed_fyl
        return :perturbed_fyl_mult
    elseif token === :dpo || token === :perturbed_mse
        return :dpo_perturbed_mse_mult
    end
    return token
end

cli = parse_cli(ARGS)

desired_workers = cli[:nprocs] == 0 ? Sys.CPU_THREADS : cli[:nprocs]
current_workers = nworkers()
if current_workers < desired_workers
    addprocs(desired_workers - current_workers)
end

println("Workers active: ", nworkers(), "  (pids ", workers(), ")")

@everywhere begin
    const PROJECT_ROOT = $PROJECT_ROOT
    using Pkg
    Pkg.activate(PROJECT_ROOT)

    include(joinpath(PROJECT_ROOT, "src", "models", "knapsack_oracles.jl"))
    include(joinpath(PROJECT_ROOT, "src", "models", "battery_oracles.jl"))
    include(joinpath(PROJECT_ROOT, "src", "models", "RSPOPlusLoss.jl"))
    include(joinpath(PROJECT_ROOT, "src", "models", "scheduling.jl"))
    include(joinpath(PROJECT_ROOT, "src", "datagen", "pjm_battery.jl"))
    include(joinpath(PROJECT_ROOT, "src", "models", "pjm_battery_training.jl"))
    include(joinpath(PROJECT_ROOT, "src", "experiments", "pjm_battery.jl"))

    function run_single_pjm_experiment(;
        objective_name::Symbol,
        split_fractions::NTuple{3,Float64},
        data_root::String,
        global_lambda0::Float64,
        global_lr0::Float64,
        global_lr_lambda_alpha::Float64=0.01,
        personal_kappa::Float64,
        personal_kappa_lr::Float64=personal_kappa,
        personal_lambda0::Float64,
        personal_freeze_tau::Float64,
        personal_lr0::Float64,
        personal_lr_lambda_alpha::Float64=0.0,
        spo_alpha::Float64=2.0,
        perturbed_nb_samples::Int=10,
        perturbed_epsilon::Float64=1.0,
        perturbed_threaded::Bool=false,
        perturbed_seed=nothing,
        seed::Int=42,
    )
        objective = pjm_objective_from_name(
            objective_name;
            spo_alpha=spo_alpha,
            perturbed_nb_samples=perturbed_nb_samples,
            perturbed_epsilon=perturbed_epsilon,
            perturbed_threaded=perturbed_threaded,
            perturbed_seed=perturbed_seed,
        )
        objective_label = _pjm_objective_name(objective)
        uses_lambda = _pjm_objective_uses_lambda(objective)
        use_global_per_client_lr = uses_lambda && global_lr_lambda_alpha > 0
        dataset_config = PJMBatteryDatasetConfig(
            data_root=data_root,
            split_fractions=split_fractions,
        )
        dataset = load_pjm_battery_dataset(dataset_config)

        warm_config = PJMBatteryTrainingConfig(
            seed=seed,
            hidden_dim=64,
            rounds=10,
            local_epochs=3,
            batch_size=64,
            client_fraction=0.4,
            validation_client_fraction=0.4,
            validation_max_samples_per_client=8,
            lambda0=uses_lambda ? global_lambda0 : NaN,
            kappa_lambda=0.0,
            per_client_lambda=true,
            lr0=global_lr0,
            kappa_lr=0.0,
            per_client_lr=use_global_per_client_lr,
            lr_lambda_alpha=global_lr_lambda_alpha,
            clip_norm=1.0,
            freeze_tau=1.0,
            freeze_eps=1e-8,
            stop_after_freeze_rounds=3,
            shuffle_batches=true,
            use_warm_start=true,
        )

        personal_config = PJMBatteryTrainingConfig(
            seed=seed + 1,
            hidden_dim=64,
            rounds=10,
            local_epochs=1,
            batch_size=64,
            client_fraction=1.0,
            validation_client_fraction=1.0,
            validation_max_samples_per_client=8,
            lambda0=uses_lambda ? personal_lambda0 : NaN,
            kappa_lambda=uses_lambda ? personal_kappa : 0.0,
            per_client_lambda=true,
            lr0=personal_lr0,
            kappa_lr=uses_lambda ? personal_kappa_lr : 0.0,
            per_client_lr=uses_lambda && personal_lr_lambda_alpha > 0,
            lr_lambda_alpha=personal_lr_lambda_alpha,
            clip_norm=1.0,
            freeze_tau=uses_lambda ? personal_freeze_tau : 0.0,
            freeze_eps=1e-8,
            stop_after_freeze_rounds=3,
            shuffle_batches=true,
            use_warm_start=true,
        )

        warm_result = fed_pjm_battery(dataset; objective=objective, config=warm_config)
        warm_eval = evaluate_fed_pjm_battery(warm_result)
        personal_result = local_pjm_battery(
            dataset;
            objective=objective,
            config=personal_config,
            model=warm_result.model,
        )
        personal_eval = evaluate_local_pjm_battery(personal_result, dataset)

        warm_aggregate = warm_eval.aggregate
        personal_aggregate = personal_eval.aggregate
        warm_scheduler = warm_aggregate.scheduler
        personal_scheduler = personal_aggregate.scheduler
        warm_clients = Dict(client.client_id => client for client in warm_eval.clients)
        personal_clients = Dict(client.client_id => client for client in personal_eval.clients)
        personal_training_clients =
            Dict(client.client_id => client for client in personal_result.clients)
        sample_lookup = Dict(
            client.client_id => (
                train=count(==(client.client_id), dataset.train.client_ids),
                val=count(==(client.client_id), dataset.val.client_ids),
                test=count(==(client.client_id), dataset.test.client_ids),
                capacity=dataset.clients[client.client_id].capacity,
                client_name=dataset.clients[client.client_id].client_name,
            ) for client in dataset.clients
        )

        rows = Vector{NamedTuple}(undef, length(dataset.clients))
        for (idx, client) in enumerate(dataset.clients)
            warm_client = warm_clients[client.client_id]
            personal_client = personal_clients[client.client_id]
            personal_training_client = personal_training_clients[client.client_id]
            counts = sample_lookup[client.client_id]
            rows[idx] = (
                objective=objective_label,
                global_method=warm_eval.method,
                personal_method=personal_eval.method,
                seed=seed,
                split_train=split_fractions[1],
                split_val=split_fractions[2],
                split_test=split_fractions[3],
                perturbed_nb_samples=perturbed_nb_samples,
                perturbed_epsilon=perturbed_epsilon,
                perturbed_threaded=perturbed_threaded,
                perturbed_seed=something(perturbed_seed, ""),
                client_id=client.client_id,
                client_name=client.client_name,
                capacity=client.capacity,
                n_dates=dataset.metadata.n_dates,
                train_date_count=length(dataset.metadata.train_date_indices),
                val_date_count=length(dataset.metadata.val_date_indices),
                test_date_count=length(dataset.metadata.test_date_indices),
                client_n_train_samples=counts.train,
                client_n_val_samples=counts.val,
                client_n_test_samples=counts.test,
                global_mse_mean=warm_aggregate.mse.mean,
                global_absolute_regret_mean=warm_aggregate.absolute_regret.mean,
                global_relative_regret_mean=warm_aggregate.relative_regret.mean,
                global_reg_decision_gap_mean=warm_aggregate.regularized_decision_gap.mean,
                global_final_lambda_mean=warm_scheduler.final_lambda.mean,
                global_final_lr_mean=warm_scheduler.final_lr.mean,
                global_final_bound_mean=warm_scheduler.final_bound.mean,
                global_freeze_round_mean=warm_scheduler.freeze_round.mean,
                personal_mse_mean=personal_aggregate.mse.mean,
                personal_absolute_regret_mean=personal_aggregate.absolute_regret.mean,
                personal_relative_regret_mean=personal_aggregate.relative_regret.mean,
                personal_reg_decision_gap_mean=personal_aggregate.regularized_decision_gap.mean,
                personal_final_lambda_mean=personal_scheduler.final_lambda.mean,
                personal_final_lr_mean=personal_scheduler.final_lr.mean,
                personal_final_bound_mean=personal_scheduler.final_bound.mean,
                personal_freeze_round_mean=personal_scheduler.freeze_round.mean,
                global_client_mse=warm_client.mse,
                global_client_absolute_regret=warm_client.absolute_regret,
                global_client_relative_regret=warm_client.relative_regret,
                global_client_reg_decision_gap=warm_client.regularized_decision_gap,
                personal_client_mse=personal_client.mse,
                personal_client_absolute_regret=personal_client.absolute_regret,
                personal_client_relative_regret=personal_client.relative_regret,
                personal_client_reg_decision_gap=personal_client.regularized_decision_gap,
                client_final_loss=personal_training_client.round_losses[end],
                client_round_losses=join(personal_training_client.round_losses, ";"),
                client_lambda_values=join(personal_training_client.trace.lambda_values, ";"),
                client_lr_values=join(personal_training_client.trace.lr_values, ";"),
                client_bound_values=join(personal_training_client.trace.bound_values, ";"),
                client_frozen_after_round=something(
                    personal_training_client.trace.frozen_after_round,
                    "",
                ),
            )
        end

        return rows
    end
end

objectives = parse_objectives(cli[:objective])
grid = [(objective=objective, seed=seed) for objective in objectives for seed in cli[:seed_values]]

println("Experiment grid size: ", length(grid))

const _split = cli[:split]
const _data_root = String(cli[:data_root])
const _global_lambda0 = Float64(cli[:global_lambda0])
const _global_lr0 = Float64(cli[:global_lr0])
const _global_lr_lambda_alpha = Float64(cli[:global_lr_lambda_alpha])
const _spo_alpha = Float64(cli[:spo_alpha])
const _perturbed_nb_samples = Int(cli[:perturbed_nb_samples])
const _perturbed_epsilon = Float64(cli[:perturbed_epsilon])
const _perturbed_threaded = Bool(cli[:perturbed_threaded])
const _perturbed_seed = cli[:perturbed_seed]
const _personal_kappa = Float64(cli[:personal_kappa])
const _personal_kappa_lr = Float64(cli[:personal_kappa_lr])
const _personal_lambda0 = Float64(cli[:personal_lambda0])
const _personal_freeze_tau = Float64(cli[:personal_freeze_tau])
const _personal_lr0 = Float64(cli[:personal_lr0])
const _personal_lr_lambda_alpha = Float64(cli[:personal_lr_lambda_alpha])

# Prepare derived CSVs once before workers read the PJM dataset.
ensure_pjm_derived_csvs(PJMBatteryDatasetConfig(data_root=_data_root, split_fractions=_split))

t0 = time()

all_rows = pmap(grid; on_error=ex -> ex) do cfg
    try
        rows = run_single_pjm_experiment(
            objective_name=cfg.objective,
            split_fractions=_split,
            data_root=_data_root,
            global_lambda0=_global_lambda0,
            global_lr0=_global_lr0,
            global_lr_lambda_alpha=_global_lr_lambda_alpha,
            personal_kappa=_personal_kappa,
            personal_kappa_lr=_personal_kappa_lr,
            personal_lambda0=_personal_lambda0,
            personal_freeze_tau=_personal_freeze_tau,
            personal_lr0=_personal_lr0,
            personal_lr_lambda_alpha=_personal_lr_lambda_alpha,
            spo_alpha=_spo_alpha,
            perturbed_nb_samples=_perturbed_nb_samples,
            perturbed_epsilon=_perturbed_epsilon,
            perturbed_threaded=_perturbed_threaded,
            perturbed_seed=_perturbed_seed,
            seed=cfg.seed,
        )
        println("  ✓ done objective=$(cfg.objective) seed=$(cfg.seed) (worker $(myid()))")
        return rows
    catch ex
        @warn "Experiment failed" cfg exception=(ex, catch_backtrace())
        return ex
    end
end

elapsed = round(time() - t0; digits=1)
println("\nAll experiments finished in $(elapsed) s")

successes = Vector{NamedTuple}()
n_failed = 0
for result in all_rows
    if result isa Exception
        global n_failed += 1
    else
        append!(successes, result)
    end
end

isempty(successes) && error("all experiments failed — no CSV written")

function _csv_escape(value)
    text = string(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

outfile = cli[:outfile]
mkpath(dirname(outfile))
header = keys(successes[1])

open(outfile, "w") do io
    println(io, join(header, ","))
    for row in successes
        println(io, join((_csv_escape(row[key]) for key in header), ","))
    end
end

println("\nCSV written → $(outfile)")
println("  rows          = ", length(successes))
println("  columns       = ", length(header))
println("  failed exprs  = ", n_failed)
println("  timestamp     = ", Dates.now())
