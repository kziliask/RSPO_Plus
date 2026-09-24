#!/usr/bin/env julia

using Pkg
const PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(PROJECT_ROOT)

if abspath(PROGRAM_FILE) == abspath(@__FILE__) && any(arg -> arg in ("--help", "-h"), ARGS)
    println("""
PJM leave-one-client-out (LOCO) cold start.
Usage: julia --project=. scripts/experiments/pjm_cold_start_run.jl [options]
  --seed-values 42,69,1993,2000,2026
  --train-days 3,7,14,30,60
  --target-client-ids all    Comma-separated IDs or all
  --split 0.5,0.1,0.4
  --data-root PATH
  --nprocs INT              Total Julia processes; 0=automatic, 1=serial
  --outdir PATH
""")
    exit()
end

using Distributed
using Dates
using Statistics

include(joinpath(@__DIR__, "pjm_cold_start_worker.jl"))

const DEFAULT_PJM_COLD_START_OUTDIR =
    joinpath("results", "experiment3", "pjm_cold_start")

function _parse_pjm_cold_start_int_list(text::AbstractString)
    return [parse(Int, strip(token)) for token in split(text, ',') if !isempty(strip(token))]
end

function _parse_pjm_cold_start_split(text::AbstractString)
    values = [
        parse(Float64, strip(token)) for token in split(text, ',') if !isempty(strip(token))
    ]
    length(values) == 3 || error("expected three comma-separated split fractions")
    return (values[1], values[2], values[3])
end

function parse_pjm_cold_start_cli(args::Vector{String})
    opts = Dict{Symbol,Any}(
        :train_days => [3, 7, 14, 30, 60],
        :seed_values => [42, 69, 1993, 2000, 2026],
        :target_client_ids => nothing,
        :split => (0.5, 0.1, 0.4),
        :data_root => joinpath("src", "data", "exp3"),
        :nprocs => 0,
        :outdir => DEFAULT_PJM_COLD_START_OUTDIR,
    )

    idx = 1
    while idx <= length(args)
        flag = args[idx]
        if flag == "--train-days"
            opts[:train_days] = _parse_pjm_cold_start_int_list(args[idx + 1])
            idx += 2
        elseif flag == "--seed-values"
            opts[:seed_values] = _parse_pjm_cold_start_int_list(args[idx + 1])
            idx += 2
        elseif flag == "--target-client-ids"
            spec = lowercase(strip(args[idx + 1]))
            opts[:target_client_ids] =
                spec == "all" ? nothing : _parse_pjm_cold_start_int_list(spec)
            idx += 2
        elseif flag == "--split"
            opts[:split] = _parse_pjm_cold_start_split(args[idx + 1])
            idx += 2
        elseif flag == "--data-root"
            opts[:data_root] = args[idx + 1]
            idx += 2
        elseif flag == "--nprocs"
            opts[:nprocs] = parse(Int, args[idx + 1])
            idx += 2
        elseif flag == "--outdir"
            opts[:outdir] = args[idx + 1]
            idx += 2
        else
            error("unknown flag `$flag`")
        end
    end

    isempty(opts[:train_days]) && error("at least one target train-day budget is required")
    isempty(opts[:seed_values]) && error("at least one seed is required")
    all(>(0), opts[:train_days]) || error("all target train-day budgets must be positive")
    length(unique(opts[:train_days])) == length(opts[:train_days]) ||
        error("target train-day budgets must be unique")
    length(unique(opts[:seed_values])) == length(opts[:seed_values]) ||
        error("seed values must be unique")
    if !isnothing(opts[:target_client_ids])
        isempty(opts[:target_client_ids]) && error("at least one target client is required")
        length(unique(opts[:target_client_ids])) == length(opts[:target_client_ids]) ||
            error("target client ids must be unique")
    end
    return opts
end

function _pjm_cold_start_csv_escape(value)
    rendered = string(value)
    if occursin(',', rendered) || occursin('"', rendered) || occursin('\n', rendered)
        return "\"" * replace(rendered, "\"" => "\"\"") * "\""
    end
    return rendered
end

function _write_pjm_cold_start_csv(path::AbstractString, rows::Vector{<:NamedTuple})
    isempty(rows) && error("cannot write empty CSV to `$path`")
    mkpath(dirname(path))
    header = collect(keys(rows[1]))
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join((_pjm_cold_start_csv_escape(row[key]) for key in header), ","))
        end
    end
    return path
end

function summarize_pjm_cold_start_by_seed(
    rows::Vector{<:NamedTuple},
    train_days::Vector{Int},
    seed_values::Vector{Int},
)
    summaries = NamedTuple[]
    for budget in train_days, seed in seed_values
        group = filter(
            row -> row.train_days_per_target == budget && row.seed == seed,
            rows,
        )
        isempty(group) && continue
        push!(
            summaries,
            (
                train_days_per_target=budget,
                seed=seed,
                n_target_clients=length(group),
                local15_absolute_regret_mean=
                    mean(row.local15_absolute_regret for row in group),
                local45_absolute_regret_mean=
                    mean(row.local45_absolute_regret for row in group),
                warm_only_absolute_regret_mean=
                    mean(row.warm_only_absolute_regret for row in group),
                fed_personalized_absolute_regret_mean=
                    mean(row.fed_personalized_absolute_regret for row in group),
                fed_gain_vs_local15_mean=mean(row.fed_gain_vs_local15 for row in group),
                fed_gain_vs_local45_mean=mean(row.fed_gain_vs_local45 for row in group),
                fed_win_fraction_vs_local15=
                    count(row -> row.fed_gain_vs_local15 > 0, group) / length(group),
                fed_win_fraction_vs_local45=
                    count(row -> row.fed_gain_vs_local45 > 0, group) / length(group),
            ),
        )
    end
    return summaries
end

function summarize_pjm_cold_start_across_seeds(
    raw_rows::Vector{<:NamedTuple},
    seed_rows::Vector{<:NamedTuple},
    train_days::Vector{Int},
)
    summaries = NamedTuple[]
    for budget in train_days
        group = filter(row -> row.train_days_per_target == budget, seed_rows)
        raw_group = filter(row -> row.train_days_per_target == budget, raw_rows)
        isempty(group) && continue
        local15 = [row.local15_absolute_regret_mean for row in group]
        local45 = [row.local45_absolute_regret_mean for row in group]
        warm = [row.warm_only_absolute_regret_mean for row in group]
        fed = [row.fed_personalized_absolute_regret_mean for row in group]
        gain15 = [row.fed_gain_vs_local15_mean for row in group]
        gain45 = [row.fed_gain_vs_local45_mean for row in group]
        push!(
            summaries,
            (
                train_days_per_target=budget,
                local15_mean=mean(local15),
                local15_std=std(local15; corrected=false),
                local45_mean=mean(local45),
                local45_std=std(local45; corrected=false),
                warm_only_mean=mean(warm),
                warm_only_std=std(warm; corrected=false),
                fed_personalized_mean=mean(fed),
                fed_personalized_std=std(fed; corrected=false),
                fed_gain_vs_local15_mean=mean(gain15),
                fed_gain_vs_local15_std=std(gain15; corrected=false),
                fed_gain_vs_local45_mean=mean(gain45),
                fed_gain_vs_local45_std=std(gain45; corrected=false),
                target_seed_win_fraction_vs_local15=
                    count(row -> row.fed_gain_vs_local15 > 0, raw_group) /
                    length(raw_group),
                target_seed_win_fraction_vs_local45=
                    count(row -> row.fed_gain_vs_local45 > 0, raw_group) /
                    length(raw_group),
                n_target_clients=
                    length(unique([row.target_client_id for row in raw_group])),
                n_seeds=length(group),
                n_target_seed_pairs=length(raw_group),
            ),
        )
    end
    return summaries
end

function audit_pjm_cold_start_rows(
    rows::Vector{<:NamedTuple},
    train_days::Vector{Int},
    seed_values::Vector{Int},
    target_client_ids::Vector{Int},
)
    expected_rows =
        length(train_days) * length(seed_values) * length(target_client_ids)
    length(rows) == expected_rows ||
        error("expected $expected_rows rows, found $(length(rows))")
    keys_seen = Set(
        (
            row.train_days_per_target,
            row.seed,
            row.target_client_id,
        ) for row in rows
    )
    length(keys_seen) == expected_rows ||
        error("target-day/seed/client rows are not unique and complete")

    metric_names = (
        :local15_absolute_regret,
        :local45_absolute_regret,
        :warm_only_absolute_regret,
        :fed_personalized_absolute_regret,
        :local15_mse,
        :local45_mse,
        :warm_only_mse,
        :fed_personalized_mse,
    )
    all(
        row -> all(name -> isfinite(getproperty(row, name)), metric_names),
        rows,
    ) || error("at least one reported metric is non-finite")
    all(
        row ->
            row.target_normalization ==
            "fixed_full_prevalidation_target_features",
        rows,
    ) || error("unexpected target normalization protocol")
    all(
        row ->
            row.target_normalization_reference_days ==
            row.donor_train_days_per_client,
        rows,
    ) || error("normalization reference does not cover the full training split")

    donor_invariance_groups = Dict{Tuple{Int,Int},Vector{NamedTuple}}()
    target_groups = Dict{Int,Vector{NamedTuple}}()
    for row in rows
        push!(
            get!(
                donor_invariance_groups,
                (row.seed, row.target_client_id),
                NamedTuple[],
            ),
            row,
        )
        push!(
            get!(target_groups, row.target_client_id, NamedTuple[]),
            row,
        )
    end
    for ((seed, target_client_id), group) in donor_invariance_groups
        length(group) == length(train_days) ||
            error("incomplete budget group for seed=$seed target=$target_client_id")
        length(unique(row.warm_only_absolute_regret for row in group)) == 1 ||
            error("warm-only regret changes across budgets for seed=$seed target=$target_client_id")
        length(unique(row.warm_only_mse for row in group)) == 1 ||
            error("warm-only MSE changes across budgets for seed=$seed target=$target_client_id")
    end
    for (target_client_id, group) in target_groups
        for field in (
            :normalization_sha256,
            :test_sample_ids_sha256,
            :test_x_sha256,
            :test_theta_sha256,
            :target_validation_days,
            :target_test_days,
        )
            length(unique(getproperty(row, field) for row in group)) == 1 ||
                error("$field changes across runs for target=$target_client_id")
        end
    end

    return [
        (
            check="row_count_and_unique_grid",
            passed=true,
            detail="$expected_rows/$expected_rows unique target-budget-seed rows",
        ),
        (
            check="finite_metrics",
            passed=true,
            detail="all eight regret/MSE metrics are finite",
        ),
        (
            check="fixed_target_normalization",
            passed=true,
            detail="one full-prevalidation feature normalization per target",
        ),
        (
            check="fixed_test_dataset",
            passed=true,
            detail="sample-id, normalized-X, and theta hashes are invariant per target",
        ),
        (
            check="donor_only_invariance",
            passed=true,
            detail="warm-only regret and MSE are exactly invariant across budgets for every target/seed",
        ),
    ]
end

function main(args::Vector{String})
    cli = parse_pjm_cold_start_cli(args)
    data_root = abspath(PROJECT_ROOT, String(cli[:data_root]))
    split_fractions = cli[:split]
    dataset = load_pjm_battery_dataset(PJMBatteryDatasetConfig(
        data_root=data_root, split_fractions=split_fractions,
    ))
    n_clients = length(dataset.clients)
    target_client_ids =
        isnothing(cli[:target_client_ids]) ? collect(1:n_clients) :
        cli[:target_client_ids]
    all(client_id -> 1 <= client_id <= n_clients, target_client_ids) ||
        error("target client ids must be in 1:$n_clients")

    grid = [
        (target_client_id=target_client_id, seed=seed)
        for target_client_id in target_client_ids
        for seed in cli[:seed_values]
    ]
    cli[:nprocs] >= 0 || error("nprocs must be nonnegative")
    desired_processes = cli[:nprocs] == 0 ? min(length(grid), Sys.CPU_THREADS) : cli[:nprocs]
    external_workers = filter(!=(1), workers())
    if length(external_workers) < desired_processes - 1
        addprocs(desired_processes - 1 - length(external_workers); exeflags=`--project=$PROJECT_ROOT`)
    end
    @everywhere filter(!=(1), workers()) include(joinpath(@__DIR__, "pjm_cold_start_worker.jl"))
    println("Target/seed configurations: ", length(grid))

    train_days = Int.(cli[:train_days])
    started = time()
    results = pmap(grid; on_error=identity) do cfg
        try
            runner = Core.eval(Main, :run_single_pjm_cold_start_target_seed)
            rows = Base.invokelatest(
                runner;
                seed=cfg.seed,
                target_client_id=cfg.target_client_id,
                train_days=train_days,
                split_fractions=split_fractions,
                data_root=data_root,
            )
            println(
                "  done target=$(cfg.target_client_id) seed=$(cfg.seed) (worker $(myid()))",
            )
            rows
        catch ex
            @warn "PJM cold-start experiment failed" cfg exception=(ex, catch_backtrace())
            ex
        end
    end
    elapsed_seconds = time() - started

    raw_rows = NamedTuple[]
    failures = Pair{NamedTuple,Exception}[]
    for (cfg, result) in zip(grid, results)
        if result isa Exception
            push!(failures, cfg => result)
        else
            append!(raw_rows, result)
        end
    end
    isempty(failures) || error(
        "$(length(failures)) of $(length(grid)) target/seed configurations failed; first failure=$(first(failures))",
    )
    sort!(raw_rows; by=row -> (row.train_days_per_target, row.seed, row.target_client_id))
    seed_rows = summarize_pjm_cold_start_by_seed(
        raw_rows,
        train_days,
        Int.(cli[:seed_values]),
    )
    summary_rows =
        summarize_pjm_cold_start_across_seeds(raw_rows, seed_rows, train_days)
    integrity_rows = audit_pjm_cold_start_rows(
        raw_rows,
        train_days,
        Int.(cli[:seed_values]),
        Int.(target_client_ids),
    )

    outdir = String(cli[:outdir])
    raw_path = _write_pjm_cold_start_csv(joinpath(outdir, "target_seed_metrics.csv"), raw_rows)
    seed_path = _write_pjm_cold_start_csv(joinpath(outdir, "seed_summary.csv"), seed_rows)
    summary_path = _write_pjm_cold_start_csv(joinpath(outdir, "summary.csv"), summary_rows)
    integrity_path = _write_pjm_cold_start_csv(
        joinpath(outdir, "integrity_checks.csv"),
        integrity_rows,
    )
    metadata_path = _write_pjm_cold_start_csv(
        joinpath(outdir, "run_metadata.csv"),
        [
            (
                generated_at=Dates.now(),
                elapsed_seconds=elapsed_seconds,
                parallel_workers=length([pid for pid in workers() if pid != 1]),
                target_clients=length(target_client_ids),
                seeds=join(Int.(cli[:seed_values]), ";"),
                train_day_budgets=join(train_days, ";"),
                target_normalization="fixed_full_prevalidation_target_features",
                normalization_uses_target_labels=false,
                validation_and_test_tensors_reused=true,
                donor_only_required_invariant=true,
            ),
        ],
    )
    println("Finished in ", round(elapsed_seconds; digits=1), " seconds")
    println("Raw target/seed metrics: ", raw_path)
    println("Seed summary: ", seed_path)
    println("Summary: ", summary_path)
    println("Integrity checks: ", integrity_path)
    println("Run metadata: ", metadata_path)
    println("Timestamp: ", Dates.now())
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
