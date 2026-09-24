#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CSV
using DataFrames
using Dates
using Statistics

include(joinpath(@__DIR__, "..", "..", "src", "baselines", "id_qp.jl"))
using .IDQPBaseline

const PAPER_SYNTHETIC_SEEDS = [42, 69, 1993, 67, 2026]
const PAPER_PJM_SEEDS = [42, 69, 1993, 2000, 2026]
const PAPER_SYNTHETIC_SCENARIOS = [
    ("0,0,0", 0.0, 0.0, 0.0),
    ("0.5,0,0", 0.5, 0.0, 0.0),
    ("1,0,0", 1.0, 0.0, 0.0),
    ("0,0.5,0", 0.0, 0.5, 0.0),
    ("0,1,0", 0.0, 1.0, 0.0),
    ("0,0,0.5", 0.0, 0.0, 0.5),
    ("0,0,1", 0.0, 0.0, 1.0),
]

function _parse_csv_ints(value::AbstractString)
    return [parse(Int, strip(token)) for token in split(value, ',') if !isempty(strip(token))]
end

function _parse_csv_floats(value::AbstractString)
    return [
        parse(Float64, strip(token)) for token in split(value, ',') if !isempty(strip(token))
    ]
end

function _parse_datasets(value::AbstractString)
    datasets = Symbol.(strip.(split(value, ',')))
    all(dataset -> dataset in (:synthetic, :pjm), datasets) || throw(
        ArgumentError("datasets must be selected from synthetic,pjm"),
    )
    return datasets
end

function _parse_scenarios(value::AbstractString)
    requested = Set(strip.(split(value, ';')))
    scenarios = [scenario for scenario in PAPER_SYNTHETIC_SCENARIOS if scenario[1] in requested]
    length(scenarios) == length(requested) || throw(
        ArgumentError(
            "unknown scenario; use semicolon-separated labels such as '0,0,0;1,0,0'",
        ),
    )
    return scenarios
end

function parse_cli(args)
    options = Dict{Symbol,Any}(
        :datasets => [:synthetic, :pjm],
        :synthetic_seeds => copy(PAPER_SYNTHETIC_SEEDS),
        :pjm_seeds => copy(PAPER_PJM_SEEDS),
        :scenarios => copy(PAPER_SYNTHETIC_SCENARIOS),
        :synthetic_epsilon_multipliers => [0.05, 0.1, 0.25],
        :pjm_epsilon_multipliers => [0.01, 0.025, 0.05],
        :outdir => joinpath("results", "additional_baselines", "id_qp"),
        :pjm_data_root => joinpath("src", "data", "exp3"),
        :smoke => false,
        :warm_rounds => 10,
        :personal_rounds => 10,
        :warm_local_epochs => 3,
        :personal_local_epochs => 1,
        :batch_size => 64,
        :validation_samples => 8,
        :synthetic_n_clients => 20,
        :synthetic_train_per_client => 100,
        :synthetic_validation_per_client => 10,
        :synthetic_test_per_client => 100,
    )

    idx = 1
    while idx <= length(args)
        flag = args[idx]
        if flag == "--datasets"
            options[:datasets] = _parse_datasets(args[idx + 1])
            idx += 2
        elseif flag == "--seed-values"
            seeds = _parse_csv_ints(args[idx + 1])
            options[:synthetic_seeds] = seeds
            options[:pjm_seeds] = seeds
            idx += 2
        elseif flag == "--synthetic-seed-values"
            options[:synthetic_seeds] = _parse_csv_ints(args[idx + 1])
            idx += 2
        elseif flag == "--pjm-seed-values"
            options[:pjm_seeds] = _parse_csv_ints(args[idx + 1])
            idx += 2
        elseif flag == "--scenarios"
            options[:scenarios] = _parse_scenarios(args[idx + 1])
            idx += 2
        elseif flag == "--epsilon-multipliers"
            multipliers = _parse_csv_floats(args[idx + 1])
            options[:synthetic_epsilon_multipliers] = multipliers
            options[:pjm_epsilon_multipliers] = multipliers
            idx += 2
        elseif flag == "--synthetic-epsilon-multipliers"
            options[:synthetic_epsilon_multipliers] =
                _parse_csv_floats(args[idx + 1])
            idx += 2
        elseif flag == "--pjm-epsilon-multipliers"
            options[:pjm_epsilon_multipliers] =
                _parse_csv_floats(args[idx + 1])
            idx += 2
        elseif flag == "--outdir"
            options[:outdir] = args[idx + 1]
            idx += 2
        elseif flag == "--pjm-data-root"
            options[:pjm_data_root] = args[idx + 1]
            idx += 2
        elseif flag == "--warm-rounds"
            options[:warm_rounds] = parse(Int, args[idx + 1])
            idx += 2
        elseif flag == "--personal-rounds"
            options[:personal_rounds] = parse(Int, args[idx + 1])
            idx += 2
        elseif flag == "--smoke"
            options[:smoke] = true
            options[:synthetic_seeds] = [42]
            options[:pjm_seeds] = [42]
            options[:scenarios] = [first(PAPER_SYNTHETIC_SCENARIOS)]
            options[:synthetic_epsilon_multipliers] = [1.0]
            options[:pjm_epsilon_multipliers] = [1.0]
            options[:warm_rounds] = 1
            options[:personal_rounds] = 1
            options[:warm_local_epochs] = 1
            options[:personal_local_epochs] = 1
            options[:batch_size] = 4
            options[:validation_samples] = 2
            options[:synthetic_n_clients] = 2
            options[:synthetic_train_per_client] = 4
            options[:synthetic_validation_per_client] = 2
            options[:synthetic_test_per_client] = 4
            idx += 1
        elseif flag in ("--help", "-h")
            println(
                """
                Usage: julia --project=. scripts/experiments/id_qp_run.jl [options]

                  --datasets synthetic,pjm
                  --seed-values LIST (override both datasets)
                  --synthetic-seed-values 42,69,1993,67,2026
                  --pjm-seed-values 42,69,1993,2000,2026
                  --scenarios '0,0,0;1,0,0;0,1,0;0,0,1'
                  --epsilon-multipliers LIST (override both datasets)
                  --synthetic-epsilon-multipliers 0.05,0.1,0.25
                  --pjm-epsilon-multipliers 0.01,0.025,0.05
                  --pjm-data-root PATH
                  --outdir PATH
                  --warm-rounds INT
                  --personal-rounds INT
                  --smoke
                """,
            )
            return nothing
        else
            throw(ArgumentError("unknown flag `$flag`"))
        end
    end
    isempty(options[:datasets]) && error("at least one dataset is required")
    for key in (:synthetic_seeds, :pjm_seeds, :scenarios)
        isempty(options[key]) && error("$key must not be empty")
        length(unique(options[key])) == length(options[key]) || error("$key must be unique")
    end
    options[:pjm_data_root] = abspath(joinpath(@__DIR__, "..", ".."), options[:pjm_data_root])
    return options
end

function _write_rows(path::AbstractString, rows::Vector{<:NamedTuple})
    isempty(rows) && return path
    CSV.write(path, DataFrame(rows))
    return path
end

function _summary_rows(
    result::IDQPRunResult;
    scenario::String,
    eta_obj::Float64=NaN,
    eta_constr::Float64=NaN,
    eta_data::Float64=NaN,
    validation_samples_per_client::Int=8,
)
    tuning = result.tuning
    warm = result.warm_evaluation.aggregate
    personal = result.personal_evaluation.aggregate
    total_runtime = result.total_wall_clock_seconds
    selected_warm_multiplier =
        tuning.epsilon_multipliers[tuning.selected_warm_index]
    selected_personal_multipliers =
        tuning.epsilon_multipliers[tuning.selected_personal_indices]
    shared = (
        dataset=result.dataset,
        scenario=scenario,
        seed=result.seed,
        eta_obj=eta_obj,
        eta_constr=eta_constr,
        eta_data=eta_data,
        validation_samples_per_client=validation_samples_per_client,
        epsilon_grid=join(tuning.epsilon_multipliers, ";"),
        selected_warm_epsilon_multiplier=selected_warm_multiplier,
        selected_personal_epsilon_multipliers=join(selected_personal_multipliers, ";"),
        total_tuning_and_training_seconds=total_runtime,
        solver_failure_count=sum(row.failure_count for row in result.diagnostics) +
                             length(result.failures),
    )
    return [
        merge(
            shared,
            (
                stage=:fed,
                n_clients=warm.n_clients,
                total_test_samples=warm.total_test_samples,
                mean_mse=warm.mse.mean,
                mean_absolute_regret=warm.absolute_regret.mean,
                mean_relative_regret=warm.relative_regret.mean,
            ),
        ),
        merge(
            shared,
            (
                stage=:personalized,
                n_clients=personal.n_clients,
                total_test_samples=personal.total_test_samples,
                mean_mse=personal.mse.mean,
                mean_absolute_regret=personal.absolute_regret.mean,
                mean_relative_regret=personal.relative_regret.mean,
            ),
        ),
    ]
end

function _client_rows(result::IDQPRunResult; scenario::String)
    tuning = result.tuning
    warm_clients = Dict(client.client_id => client for client in result.warm_evaluation.clients)
    personal_clients =
        Dict(client.client_id => client for client in result.personal_evaluation.clients)
    return [
        (
            dataset=result.dataset,
            scenario=scenario,
            seed=result.seed,
            client_id=client_id,
            selected_warm_epsilon_multiplier=
                tuning.epsilon_multipliers[tuning.selected_warm_index],
            selected_personal_epsilon_multiplier=
                tuning.epsilon_multipliers[tuning.selected_personal_indices[client_id]],
            warm_validation_score=tuning.warm_validation_scores[tuning.selected_warm_index],
            personal_validation_score=tuning.personal_validation_scores[
                tuning.selected_personal_indices[client_id],
                client_id,
            ],
            warm_mse=warm_clients[client_id].mse,
            warm_absolute_regret=warm_clients[client_id].absolute_regret,
            warm_relative_regret=warm_clients[client_id].relative_regret,
            personal_mse=personal_clients[client_id].mse,
            personal_absolute_regret=personal_clients[client_id].absolute_regret,
            personal_relative_regret=personal_clients[client_id].relative_regret,
        ) for client_id in sort!(collect(keys(warm_clients)))
    ]
end

function _failure_row(
    dataset::Symbol,
    scenario::String,
    seed::Int,
    phase::Symbol,
    multiplier,
    err,
)
    return (
        dataset=dataset,
        scenario=scenario,
        seed=seed,
        phase=phase,
        epsilon_multiplier=multiplier,
        error_type=string(typeof(err)),
        error_message=sprint(showerror, err),
    )
end

function _persist_outputs(
    outdir::AbstractString,
    summary_rows,
    client_rows,
    timing_rows,
    diagnostic_rows,
    failure_rows,
)
    mkpath(outdir)
    _write_rows(joinpath(outdir, "seed_level_summary.csv"), summary_rows)
    _write_rows(joinpath(outdir, "client_results.csv"), client_rows)
    _write_rows(joinpath(outdir, "candidate_wall_clock.csv"), timing_rows)
    _write_rows(joinpath(outdir, "solver_diagnostics.csv"), diagnostic_rows)
    _write_rows(joinpath(outdir, "run_failures.csv"), failure_rows)
    return outdir
end

function _aggregate_summary(summary_rows)
    isempty(summary_rows) && return NamedTuple[]
    frame = DataFrame(summary_rows)
    rows = NamedTuple[]
    for group in groupby(frame, [:dataset, :scenario, :stage])
        first_row = first(group)
        # Match the conventions used by the reproduced paper pipelines:
        # synthetic Table 2 uses sample std; PJM Table 3 uses population std.
        corrected = String(first_row.dataset) == "synthetic_knapsack" && nrow(group) > 1
        push!(
            rows,
            (
                dataset=first_row.dataset,
                scenario=first_row.scenario,
                stage=first_row.stage,
                mean_absolute_regret=mean(group.mean_absolute_regret),
                std_absolute_regret=std(
                    group.mean_absolute_regret;
                    corrected=corrected,
                ),
                mean_relative_regret=mean(group.mean_relative_regret),
                std_relative_regret=std(
                    group.mean_relative_regret;
                    corrected=corrected,
                ),
                mean_wall_clock_seconds=mean(
                    group.total_tuning_and_training_seconds,
                ),
                total_wall_clock_seconds=sum(
                    group.total_tuning_and_training_seconds,
                ),
                solver_failure_count=sum(group.solver_failure_count),
                n_seeds=nrow(group),
            ),
        )
    end
    return rows
end

function _run_synthetic!(options, collections)
    for (scenario, eta_obj, eta_constr, eta_data) in options[:scenarios]
        for seed in options[:synthetic_seeds]
            println(
                "ID-QP synthetic scenario=$scenario seed=$seed grid=$(options[:synthetic_epsilon_multipliers])",
            )
            config = IDQPBaseline.SyntheticKnapsackConfig(
                seed=seed,
                n_clients=options[:synthetic_n_clients],
                p=8,
                dim=50,
                deg=4,
                epsilon_noise=0.0,
                eta_obj=eta_obj,
                eta_constr=eta_constr,
                eta_constr_affects_weights=true,
                eta_data_dist=eta_data,
                data_imbalance=false,
                n_train_total=
                    options[:synthetic_n_clients] * options[:synthetic_train_per_client],
                n_val_total=
                    options[:synthetic_n_clients] * options[:synthetic_validation_per_client],
                n_test_total=
                    options[:synthetic_n_clients] * options[:synthetic_test_per_client],
            )
            try
                dataset = IDQPBaseline.generate_synthetic_knapsack_dataset(config)
                result = run_synthetic_id_qp(
                    dataset;
                    epsilon_multipliers=options[:synthetic_epsilon_multipliers],
                    validation_max_samples_per_client=options[:validation_samples],
                    warm_rounds=options[:warm_rounds],
                    personal_rounds=options[:personal_rounds],
                    warm_local_epochs=options[:warm_local_epochs],
                    personal_local_epochs=options[:personal_local_epochs],
                    batch_size=options[:batch_size],
                    warm_client_fraction=options[:smoke] ? 1.0 : 0.4,
                )
                append!(
                    collections.summary,
                    _summary_rows(
                        result;
                        scenario=scenario,
                        eta_obj=eta_obj,
                        eta_constr=eta_constr,
                        eta_data=eta_data,
                        validation_samples_per_client=options[:validation_samples],
                    ),
                )
                append!(collections.clients, _client_rows(result; scenario=scenario))
                append!(
                    collections.timings,
                    [merge(row, (scenario=scenario,)) for row in result.stage_timings],
                )
                append!(
                    collections.diagnostics,
                    [merge(row, (scenario=scenario,)) for row in result.diagnostics],
                )
                append!(
                    collections.failures,
                    [merge(row, (scenario=scenario,)) for row in result.failures],
                )
            catch err
                push!(
                    collections.failures,
                    _failure_row(:synthetic_knapsack, scenario, seed, :run, NaN, err),
                )
                showerror(stderr, err)
                println(stderr)
            end
            _persist_outputs(
                options[:outdir],
                collections.summary,
                collections.clients,
                collections.timings,
                collections.diagnostics,
                collections.failures,
            )
        end
    end
end

function _run_pjm!(options, collections)
    dataset_config = IDQPBaseline.PJMBatteryDatasetConfig(
        data_root=options[:pjm_data_root],
        split_fractions=(0.5, 0.1, 0.4),
    )
    dataset = IDQPBaseline.load_pjm_battery_dataset(dataset_config)
    for seed in options[:pjm_seeds]
        println("ID-QP PJM seed=$seed grid=$(options[:pjm_epsilon_multipliers])")
        try
            result = run_pjm_id_qp(
                dataset;
                seed=seed,
                epsilon_multipliers=options[:pjm_epsilon_multipliers],
                validation_max_samples_per_client=options[:validation_samples],
                warm_rounds=options[:warm_rounds],
                personal_rounds=options[:personal_rounds],
                warm_local_epochs=options[:warm_local_epochs],
                personal_local_epochs=options[:personal_local_epochs],
                batch_size=options[:batch_size],
                warm_client_fraction=0.4,
            )
            append!(
                collections.summary,
                _summary_rows(
                    result;
                    scenario="pjm",
                    validation_samples_per_client=options[:validation_samples],
                ),
            )
            append!(collections.clients, _client_rows(result; scenario="pjm"))
            append!(
                collections.timings,
                [merge(row, (scenario="pjm",)) for row in result.stage_timings],
            )
            append!(
                collections.diagnostics,
                [merge(row, (scenario="pjm",)) for row in result.diagnostics],
            )
            append!(
                collections.failures,
                [merge(row, (scenario="pjm",)) for row in result.failures],
            )
        catch err
            push!(collections.failures, _failure_row(:pjm, "pjm", seed, :run, NaN, err))
            showerror(stderr, err)
            println(stderr)
        end
        _persist_outputs(
            options[:outdir],
            collections.summary,
            collections.clients,
            collections.timings,
            collections.diagnostics,
            collections.failures,
        )
    end
end

function main(args=ARGS)
    options = parse_cli(args)
    isnothing(options) && return
    collections = (
        summary=NamedTuple[],
        clients=NamedTuple[],
        timings=NamedTuple[],
        diagnostics=NamedTuple[],
        failures=NamedTuple[],
    )
    started = time()
    :synthetic in options[:datasets] && _run_synthetic!(options, collections)
    :pjm in options[:datasets] && _run_pjm!(options, collections)
    aggregate = _aggregate_summary(collections.summary)
    _write_rows(joinpath(options[:outdir], "aggregate_results.csv"), aggregate)
    failed_runs = count(row -> row.phase == :run, collections.failures)
    failed_runs == 0 || error("$failed_runs ID-QP runs failed; see run_failures.csv")
    println("Completed in $(round(time() - started; digits=1)) seconds")
    println("ID-QP outputs written to $(abspath(options[:outdir]))")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
