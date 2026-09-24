#!/usr/bin/env julia
"""
    synth_run.jl

Distributed experiment runner for the synthetic knapsack benchmark.

Usage
─────
    julia --project=. scripts/experiments/synth_run.jl
    julia --project=. scripts/experiments/synth_run.jl --objective rspo_plus,spo_plus,mse
    julia --project=. scripts/experiments/synth_run.jl --nprocs 8

Results are written to `results/experiment1/synthetic_knapsack_results.csv` by default.
"""

using Pkg
const PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(PROJECT_ROOT)

using Distributed
using Dates

function parse_int_list(s::AbstractString)
    return [parse(Int, strip(t)) for t in split(s, ',') if !isempty(strip(t))]
end

function parse_cli(args::Vector{String})
    kw = Dict{Symbol,Any}(
        :global_lambda0 => 3.0,
        :global_lr0 => 0.001,
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
        :personal_freeze_tau => 0.01,
        :personal_lr0 => 0.001,
        :personal_lr_lambda_alpha => 0.01,
        :rspo_only_personal_schedule => true,
        :seed_values => [42, 69, 1993, 67, 2026],
        :nprocs => 0,
        :outfile => joinpath("results", "experiment1", "synthetic_knapsack_results.csv"),
    )

    i = 1
    while i <= length(args)
        flag = args[i]
        if flag == "--nprocs"
            kw[:nprocs] = parse(Int, args[i + 1])
            i += 2
        elseif flag == "--objective"
            kw[:objective] = args[i + 1]
            i += 2
        elseif flag == "--spo-alpha"
            kw[:spo_alpha] = parse(Float64, args[i + 1])
            i += 2
        elseif flag == "--perturbed-nb-samples"
            kw[:perturbed_nb_samples] = parse(Int, args[i + 1])
            i += 2
        elseif flag == "--perturbed-epsilon"
            kw[:perturbed_epsilon] = parse(Float64, args[i + 1])
            i += 2
        elseif flag == "--perturbed-threaded"
            kw[:perturbed_threaded] = true
            i += 1
        elseif flag == "--perturbed-seed"
            kw[:perturbed_seed] = parse(Int, args[i + 1])
            i += 2
        elseif flag == "--global-lambda0"
            kw[:global_lambda0] = parse(Float64, args[i + 1])
            i += 2
        elseif flag == "--global-lr0"
            kw[:global_lr0] = parse(Float64, args[i + 1])
            i += 2
        elseif flag == "--global-lr-lambda-alpha"
            kw[:global_lr_lambda_alpha] = parse(Float64, args[i + 1])
            i += 2
        elseif flag == "--personal-kappa"
            kw[:personal_kappa] = parse(Float64, args[i + 1])
            i += 2
        elseif flag == "--personal-kappa-lr"
            kw[:personal_kappa_lr] = parse(Float64, args[i + 1])
            i += 2
        elseif flag == "--personal-lambda0"
            kw[:personal_lambda0] = parse(Float64, args[i + 1])
            i += 2
        elseif flag == "--personal-freeze-tau"
            kw[:personal_freeze_tau] = parse(Float64, args[i + 1])
            i += 2
        elseif flag == "--personal-lr0"
            kw[:personal_lr0] = parse(Float64, args[i + 1])
            i += 2
        elseif flag == "--personal-lr-lambda-alpha"
            kw[:personal_lr_lambda_alpha] = parse(Float64, args[i + 1])
            i += 2
        elseif flag == "--rspo-only-personal-schedule"
            kw[:rspo_only_personal_schedule] = true
            i += 1
        elseif flag == "--seed-values"
            kw[:seed_values] = parse_int_list(args[i + 1])
            i += 2
        elseif flag == "--outfile"
            kw[:outfile] = args[i + 1]
            i += 2
        else
            error("Unknown flag: $flag")
        end
    end

    return kw
end

function canonical_objective_token(token::Symbol)
    if token === :pfyl || token === :perturbed_fyl
        return :perturbed_fyl_mult
    elseif token === :dpo || token === :perturbed_mse
        return :dpo_perturbed_mse_mult
    end
    return token
end

function parse_objectives(spec::AbstractString)
    tokens = [
        canonical_objective_token(Symbol(strip(token))) for
        token in split(spec, ',') if !isempty(strip(token))
    ]
    isempty(tokens) && error("At least one objective must be provided.")
    :all in tokens &&
        return [
            :rspo_plus,
            :spo_plus,
            :perturbed_fyl_mult,
            :dpo_perturbed_mse_mult,
            :mse,
            :mse_then_spo_plus,
        ]

    valid = Set([
        :rspo_plus,
        :spo_plus,
        :perturbed_fyl_mult,
        :dpo_perturbed_mse_mult,
        :mse,
        :mse_then_spo_plus,
    ])
    for token in tokens
        token in valid || error(
            "Unknown objective `$token` (expected rspo_plus, spo_plus, perturbed_fyl_mult, dpo_perturbed_mse_mult, mse, mse_then_spo_plus, or all).",
        )
    end
    return unique(tokens)
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
    include(joinpath(PROJECT_ROOT, "src", "models", "RSPOPlusLoss.jl"))
    include(joinpath(PROJECT_ROOT, "src", "models", "scheduling.jl"))
    include(joinpath(PROJECT_ROOT, "src", "datagen", "synthetic_knapsack.jl"))
    include(joinpath(PROJECT_ROOT, "src", "models", "training.jl"))
    include(joinpath(PROJECT_ROOT, "src", "experiments", "synthetic_knapsack.jl"))

    function resolve_training_objective(
        objective_name::Symbol;
        spo_alpha::Float64=2.0,
        perturbed_nb_samples::Int=10,
        perturbed_epsilon::Float64=1.0,
        perturbed_threaded::Bool=false,
        perturbed_seed=nothing,
    )
        return synthetic_knapsack_objective_from_name(
            objective_name;
            spo_alpha=spo_alpha,
            perturbed_nb_samples=perturbed_nb_samples,
            perturbed_epsilon=perturbed_epsilon,
            perturbed_threaded=perturbed_threaded,
            perturbed_seed=perturbed_seed,
        )
    end

    function resolve_phase_objectives(
        objective_name::Symbol;
        spo_alpha::Float64=2.0,
        perturbed_nb_samples::Int=10,
        perturbed_epsilon::Float64=1.0,
        perturbed_threaded::Bool=false,
        perturbed_seed=nothing,
    )
        if objective_name === :mse_then_spo_plus
            return (
                warm_start=MSEObjective(),
                personalization=SPOPlusObjective(; α=spo_alpha),
            )
        end

        objective = resolve_training_objective(
            objective_name;
            spo_alpha=spo_alpha,
            perturbed_nb_samples=perturbed_nb_samples,
            perturbed_epsilon=perturbed_epsilon,
            perturbed_threaded=perturbed_threaded,
            perturbed_seed=perturbed_seed,
        )
        return (warm_start=objective, personalization=objective)
    end

    function experiment_objective_label(
        raw_objective_name::Symbol,
        warm_start_objective::SyntheticKnapsackObjective,
        personalization_objective::SyntheticKnapsackObjective,
    )
        if raw_objective_name === :mse_then_spo_plus
            return :mse_then_spo_plus
        end
        return objective_name(personalization_objective)
    end

    function run_single_experiment(;
        objective_name::Symbol,
        eta_obj::Float64,
        eta_constr::Float64,
        eta_data_dist::Float64,
        data_imbalance::Bool,
        global_lambda0::Float64,
        global_lr0::Float64,
        global_lr_lambda_alpha::Float64=0.01,
        personal_kappa::Float64,
        personal_kappa_lr::Float64=personal_kappa,
        personal_lambda0::Float64,
        personal_freeze_tau::Float64,
        personal_lr0::Float64,
        personal_lr_lambda_alpha::Float64=0.0,
        rspo_only_personal_schedule::Bool=false,
        spo_alpha::Float64=2.0,
        perturbed_nb_samples::Int=10,
        perturbed_epsilon::Float64=1.0,
        perturbed_threaded::Bool=false,
        perturbed_seed=nothing,
        seed::Int=42,
        n_clients::Int=20,
        n_val_total::Int=200,
        n_train_total::Union{Nothing,Int}=nothing,
        n_test_total::Union{Nothing,Int}=nothing,
        validation_max_samples_per_client::Int=8,
        warm_rounds::Int=10,
        personal_rounds::Int=10,
        warm_local_epochs::Int=3,
        personal_local_epochs::Int=1,
        warm_batch_size::Int=64,
        personal_batch_size::Int=64,
    )
        objectives = resolve_phase_objectives(
            objective_name;
            spo_alpha=spo_alpha,
            perturbed_nb_samples=perturbed_nb_samples,
            perturbed_epsilon=perturbed_epsilon,
            perturbed_threaded=perturbed_threaded,
            perturbed_seed=perturbed_seed,
        )
        warm_start_objective = objectives.warm_start
        personalization_objective = objectives.personalization
        objective_label = experiment_objective_label(
            objective_name,
            warm_start_objective,
            personalization_objective,
        )
        warm_start_uses_lambda = objective_uses_lambda(warm_start_objective)
        personalization_uses_lambda = objective_uses_lambda(personalization_objective)
        personalization_method = objective_method(personalization_objective, :local)
        use_global_per_client_lr = warm_start_uses_lambda && global_lr_lambda_alpha > 0
        use_personal_per_client_lr =
            personalization_uses_lambda && personal_lr_lambda_alpha > 0
        uses_personal_schedule =
            personalization_uses_lambda &&
            (personalization_method === :local_rspo_plus || !rspo_only_personal_schedule)

        default_train_size = data_imbalance ? 5500 : 2000
        n_train_size = something(n_train_total, default_train_size)
        n_test_size = something(n_test_total, n_train_size)
        dataset_config = SyntheticKnapsackConfig(
            seed=seed,
            n_clients=n_clients,
            p=8,
            dim=50,
            n_train_total=n_train_size,
            n_val_total=n_val_total,
            n_test_total=n_test_size,
            eta_obj=eta_obj,
            eta_constr=eta_constr,
            eta_data_dist=eta_data_dist,
            data_imbalance=data_imbalance,
            deg=4,
            epsilon_noise=0.0,
            eta_constr_affects_weights=true,
        )

        dataset = generate_synthetic_knapsack_dataset(dataset_config)

        global_config = SyntheticKnapsackTrainingConfig(
            seed=seed + 1,
            hidden_dim=64,
            rounds=warm_rounds,
            local_epochs=warm_local_epochs,
            batch_size=warm_batch_size,
            client_fraction=0.4,
            validation_client_fraction=1.0,
            validation_max_samples_per_client=validation_max_samples_per_client,
            lambda0=warm_start_uses_lambda ? global_lambda0 : NaN,
            kappa_lambda=0.0,
            lr0=global_lr0,
            kappa_lr=0.0,
            per_client_lr=use_global_per_client_lr,
            lr_lambda_alpha=global_lr_lambda_alpha,
            clip_norm=1.0,
            freeze_tau=warm_start_uses_lambda ? 1.0 : NaN,
            freeze_eps=1e-8,
            stop_after_freeze_rounds=3,
            shuffle_batches=true,
            use_warm_start=true,
            per_client_lambda=warm_start_uses_lambda,
        )

        global_result = fed_synthetic_knapsack(
            dataset;
            objective=warm_start_objective,
            config=global_config,
        )
        global_evaluation = evaluate_fed_training(global_result)
        ga = global_evaluation.aggregate
        gs = ga.scheduler
        global_final_loss_value =
            isempty(global_result.round_losses) ? NaN : global_result.round_losses[end]
        global_round_losses_str = join(global_result.round_losses, ";")
        global_client_metrics = Dict(
            client_metric.client_id => client_metric for client_metric in global_evaluation.clients
        )

        personal_config = SyntheticKnapsackTrainingConfig(
            seed=seed + 2,
            hidden_dim=64,
            rounds=personal_rounds,
            local_epochs=personal_local_epochs,
            batch_size=personal_batch_size,
            client_fraction=1.0,
            validation_client_fraction=1.0,
            validation_max_samples_per_client=validation_max_samples_per_client,
            lambda0=personalization_uses_lambda ? personal_lambda0 : NaN,
            kappa_lambda=uses_personal_schedule ? personal_kappa : 0.0,
            lr0=personal_lr0,
            kappa_lr=uses_personal_schedule ? personal_kappa_lr : 0.0,
            per_client_lr=use_personal_per_client_lr,
            lr_lambda_alpha=personal_lr_lambda_alpha,
            clip_norm=1.0,
            freeze_tau=personalization_uses_lambda && uses_personal_schedule ? personal_freeze_tau : NaN,
            freeze_eps=1e-8,
            stop_after_freeze_rounds=3,
            shuffle_batches=true,
            use_warm_start=true,
            per_client_lambda=personalization_uses_lambda,
        )

        personal_result = local_synthetic_knapsack(
            dataset;
            objective=personalization_objective,
            config=personal_config,
            model=global_result.model,
        )
        personal_evaluation = evaluate_local_training(personal_result, dataset)
        pa = personal_evaluation.aggregate
        ps = pa.scheduler
        personal_client_metrics = Dict(
            client_metric.client_id => client_metric for client_metric in personal_evaluation.clients
        )

        rows = Vector{NamedTuple}(undef, dataset_config.n_clients)
        for (i, client) in enumerate(personal_result.clients)
            cid = client.client_id
            global_client = global_client_metrics[cid]
            personal_client = personal_client_metrics[cid]

            rows[i] = (
                objective=objective_label,
                global_method=global_evaluation.method,
                personal_method=personal_evaluation.method,
                seed=seed,
                eta_obj=eta_obj,
                eta_constr=eta_constr,
                eta_data_dist=eta_data_dist,
                data_imbalance=data_imbalance,
                perturbed_nb_samples=perturbed_nb_samples,
                perturbed_epsilon=perturbed_epsilon,
                perturbed_threaded=perturbed_threaded,
                perturbed_seed=something(perturbed_seed, ""),
                client_id=cid,
                objective_pairwise_mean=dataset.metadata.objective_pairwise_mean,
                capacity_pairwise_mean=dataset.metadata.capacity_pairwise_mean,
                feature_mean_pairwise_mean=dataset.metadata.feature_mean_pairwise_mean,
                global_mse_mean=ga.mse.mean,
                global_mse_std=ga.mse.std,
                global_absolute_regret_mean=ga.absolute_regret.mean,
                global_absolute_regret_std=ga.absolute_regret.std,
                global_relative_regret_mean=ga.relative_regret.mean,
                global_relative_regret_std=ga.relative_regret.std,
                global_reg_decision_gap_mean=ga.regularized_decision_gap.mean,
                global_reg_decision_gap_std=ga.regularized_decision_gap.std,
                global_final_lambda_mean=gs.final_lambda.mean,
                global_final_lambda_std=gs.final_lambda.std,
                global_final_lr_mean=gs.final_lr.mean,
                global_final_lr_std=gs.final_lr.std,
                global_final_bound_mean=gs.final_bound.mean,
                global_final_bound_std=gs.final_bound.std,
                global_freeze_round_mean=gs.freeze_round.mean,
                global_freeze_round_std=gs.freeze_round.std,
                global_frozen_fraction=gs.frozen_fraction,
                global_n_clients=ga.n_clients,
                global_total_test_samples=ga.total_test_samples,
                global_final_loss=global_final_loss_value,
                global_round_losses=global_round_losses_str,
                personal_mse_mean=pa.mse.mean,
                personal_mse_std=pa.mse.std,
                personal_absolute_regret_mean=pa.absolute_regret.mean,
                personal_absolute_regret_std=pa.absolute_regret.std,
                personal_relative_regret_mean=pa.relative_regret.mean,
                personal_relative_regret_std=pa.relative_regret.std,
                personal_reg_decision_gap_mean=pa.regularized_decision_gap.mean,
                personal_reg_decision_gap_std=pa.regularized_decision_gap.std,
                personal_final_lambda_mean=ps.final_lambda.mean,
                personal_final_lambda_std=ps.final_lambda.std,
                personal_final_lr_mean=ps.final_lr.mean,
                personal_final_lr_std=ps.final_lr.std,
                personal_final_bound_mean=ps.final_bound.mean,
                personal_final_bound_std=ps.final_bound.std,
                personal_freeze_round_mean=ps.freeze_round.mean,
                personal_freeze_round_std=ps.freeze_round.std,
                personal_frozen_fraction=ps.frozen_fraction,
                personal_n_clients=pa.n_clients,
                personal_total_test_samples=pa.total_test_samples,
                client_n_test_samples=personal_client.n_test_samples,
                global_client_mse=global_client.mse,
                global_client_absolute_regret=global_client.absolute_regret,
                global_client_relative_regret=global_client.relative_regret,
                global_client_reg_decision_gap=global_client.regularized_decision_gap,
                personal_client_mse=personal_client.mse,
                personal_client_absolute_regret=personal_client.absolute_regret,
                personal_client_relative_regret=personal_client.relative_regret,
                personal_client_reg_decision_gap=personal_client.regularized_decision_gap,
                client_final_loss=client.round_losses[end],
                client_round_losses=join(client.round_losses, ";"),
                client_lambda_values=join(client.trace.lambda_values, ";"),
                client_lr_values=join(client.trace.lr_values, ";"),
                client_bound_values=join(client.trace.bound_values, ";"),
                client_frozen_after_round=client.trace.frozen_after_round,
            )
        end

        return rows
    end
end

obj_eta_values = [0.0, 0.5, 1.0]
constr_eta_values = [0.0, 0.5, 1.0]
dist_eta_values = [0.0, 0.5, 1.0]
imbalance_values = [false]
seed_values = cli[:seed_values]
objectives = parse_objectives(cli[:objective])
grid = [
    (
        objective=objective,
        eta_obj=eo,
        eta_constr=ec,
        eta_data_dist=ed,
        data_imbalance=di,
        seed=sd,
    )
    for objective in objectives
    for eo in obj_eta_values
    for ec in constr_eta_values
    for ed in dist_eta_values
    for di in imbalance_values
    for sd in seed_values
]

println("Experiment grid size: ", length(grid))

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
const _rspo_only_personal_schedule = Bool(cli[:rspo_only_personal_schedule])

println("Keyword hyperparameters:")
println("  global_lambda0      = ", _global_lambda0)
println("  global_lr0          = ", _global_lr0)
println("  global_lr_lambda_alpha = ", _global_lr_lambda_alpha)
println("  objective           = ", cli[:objective])
println("  spo_alpha           = ", _spo_alpha)
println("  perturbed_nb_samples = ", _perturbed_nb_samples)
println("  perturbed_epsilon   = ", _perturbed_epsilon)
println("  perturbed_threaded  = ", _perturbed_threaded)
println("  perturbed_seed      = ", _perturbed_seed)
println("  personal_kappa      = ", _personal_kappa)
println("  personal_kappa_lr   = ", _personal_kappa_lr)
println("  personal_lambda0    = ", _personal_lambda0)
println("  personal_freeze_tau = ", _personal_freeze_tau)
println("  personal_lr0        = ", _personal_lr0)
println("  personal_lr_lambda_alpha = ", _personal_lr_lambda_alpha)
println("  rspo_only_personal_schedule = ", _rspo_only_personal_schedule)
println("  seed_values         = ", seed_values)
println()

t0 = time()

all_rows = pmap(grid; on_error=ex -> ex) do cfg
    try
        rows = run_single_experiment(
            ;
            objective_name=cfg.objective,
            eta_obj=cfg.eta_obj,
            eta_constr=cfg.eta_constr,
            eta_data_dist=cfg.eta_data_dist,
            data_imbalance=cfg.data_imbalance,
            global_lambda0=_global_lambda0,
            global_lr0=_global_lr0,
            global_lr_lambda_alpha=_global_lr_lambda_alpha,
            personal_kappa=_personal_kappa,
            personal_kappa_lr=_personal_kappa_lr,
            personal_lambda0=_personal_lambda0,
            personal_freeze_tau=_personal_freeze_tau,
            personal_lr0=_personal_lr0,
            personal_lr_lambda_alpha=_personal_lr_lambda_alpha,
            rspo_only_personal_schedule=_rspo_only_personal_schedule,
            spo_alpha=_spo_alpha,
            perturbed_nb_samples=_perturbed_nb_samples,
            perturbed_epsilon=_perturbed_epsilon,
            perturbed_threaded=_perturbed_threaded,
            perturbed_seed=_perturbed_seed,
            seed=cfg.seed,
        )
        println(
            "  ✓ done  objective=$(cfg.objective)  eta_obj=$(cfg.eta_obj)  eta_constr=$(cfg.eta_constr)  eta_data_dist=$(cfg.eta_data_dist)  imbalance=$(cfg.data_imbalance)  seed=$(cfg.seed)  (worker $(myid()))",
        )
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

for (i, result) in enumerate(all_rows)
    if result isa Exception
        global n_failed += 1
        @warn "Skipping failed experiment $(i): $(result)"
    else
        append!(successes, result)
    end
end

if n_failed > 0
    println("⚠  $(n_failed) / $(length(grid)) experiments failed and were skipped.")
end

isempty(successes) && error("All experiments failed — no CSV written.")

function _csv_escape(x)
    s = string(x)
    if occursin(',', s) || occursin('"', s) || occursin('\n', s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

outfile = cli[:outfile]
mkpath(dirname(outfile))
header = keys(successes[1])

open(outfile, "w") do io
    println(io, join(header, ","))
    for row in successes
        vals = [_csv_escape(row[k]) for k in header]
        println(io, join(vals, ","))
    end
end

println("\nCSV written → $(outfile)")
println("  rows          = ", length(successes))
println("  columns       = ", length(header))
println("  column names  = ", join(header, ", "))
println("  failed exprs  = ", n_failed)
println("  timestamp     = ", Dates.now())
