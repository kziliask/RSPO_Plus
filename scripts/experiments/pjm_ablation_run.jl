#!/usr/bin/env julia

using Pkg
const PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(PROJECT_ROOT)

if abspath(PROGRAM_FILE) == abspath(@__FILE__) && any(arg -> arg in ("--help", "-h"), ARGS)
    println("""
PJM component and fixed-demand ablations.
Usage: julia --project=. scripts/experiments/pjm_ablation_run.jl [options]
  --fixed-demand FLOAT        Shared dispatch demand in [0,24]; omit for original demands
  --seed-values 42,69,1993,2000,2026
  --split 0.5,0.1,0.4
  --data-root PATH
  --nprocs INT                 Total Julia processes; 0=automatic, 1=serial
  --outdir PATH
  --spo-alpha FLOAT
  --global-lambda0 FLOAT
  --global-lr0 FLOAT
  --global-lr-lambda-alpha FLOAT
  --personal-kappa FLOAT
  --personal-kappa-lr FLOAT
  --personal-lambda0 FLOAT
  --personal-freeze-tau FLOAT
  --personal-lr0 FLOAT
  --personal-lr-lambda-alpha FLOAT
""")
    exit()
end

using Distributed
using Dates
using Statistics

include(joinpath(@__DIR__, "pjm_ablation_worker.jl"))

const DEFAULT_OUTDIR = joinpath("results", "experiment3", "pjm_ablation")

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
        :spo_alpha => 2.0,
        :personal_kappa => 1.0,
        :personal_kappa_lr => 0.1,
        :personal_lambda0 => 55.0,
        :personal_freeze_tau => 0.03,
        :personal_lr0 => 1e-3,
        :personal_lr_lambda_alpha => 0.001,
        :seed_values => [42, 69, 1993, 2000, 2026],
        :split => (0.5, 0.1, 0.4),
        :data_root => joinpath("src", "data", "exp3"),
        :fixed_demand => nothing,
        :nprocs => 0,
        :outdir => DEFAULT_OUTDIR,
    )

    idx = 1
    while idx <= length(args)
        flag = args[idx]
        if flag == "--nprocs"
            opts[:nprocs] = parse(Int, args[idx + 1])
            idx += 2
        elseif flag == "--spo-alpha"
            opts[:spo_alpha] = parse(Float64, args[idx + 1])
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
        elseif flag == "--fixed-demand"
            opts[:fixed_demand] = parse(Float64, args[idx + 1])
            idx += 2
        elseif flag == "--outdir"
            opts[:outdir] = args[idx + 1]
            idx += 2
        else
            error("unknown flag `$flag`")
        end
    end

    outdir = String(opts[:outdir])
    opts[:raw_outfile] = joinpath(outdir, "pjm_ablation_seed_metrics.csv")
    opts[:summary_outfile] = joinpath(outdir, "pjm_ablation_summary.csv")
    opts[:export_root] = joinpath(outdir, "runs")
    return opts
end

function _csv_escape(value)
    text = string(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function write_namedtuple_csv(path::AbstractString, rows::Vector{<:NamedTuple})
    isempty(rows) && error("cannot write empty CSV to `$path`")
    mkpath(dirname(path))
    header = collect(keys(rows[1]))
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join((_csv_escape(row[key]) for key in header), ","))
        end
    end
    return path
end

const PJM_ABLATION_METHOD_ORDER = [
    "mse",
    "spo_plus",
    "rspo_plus",
    "rspo_plus_personalization",
    "rspo_plus_personalization_per_client",
    "rspo_plus_personalization_per_client_scheduling",
]

function summarize_ablation_rows(rows::Vector{<:NamedTuple})
    grouped = Dict{String,Vector{NamedTuple}}()
    for row in rows
        push!(get!(grouped, row.method_key, NamedTuple[]), row)
    end

    summary_rows = NamedTuple[]
    for method_key in PJM_ABLATION_METHOD_ORDER
        method_rows = get(grouped, method_key, NamedTuple[])
        isempty(method_rows) && continue

        values = Float64[row.aggregate_absolute_regret_mean for row in method_rows]
        mean_value = mean(values)
        std_value = length(values) == 1 ? 0.0 : std(values; corrected=false)
        push!(
            summary_rows,
            (
                method_key=method_key,
                method_label=method_rows[1].method_label,
                mean_absolute_regret_mean=mean_value,
                mean_absolute_regret_std=std_value,
                n_seeds=length(values),
                seed_values=join(sort([row.seed for row in method_rows]), ";"),
            ),
        )
    end
    return summary_rows
end

cli = parse_cli(ARGS)

seed_values = cli[:seed_values]
isempty(seed_values) && error("at least one seed is required")
length(unique(seed_values)) == length(seed_values) || error("seed values must be unique")
cli[:nprocs] >= 0 || error("nprocs must be nonnegative")
cli[:data_root] = abspath(PROJECT_ROOT, String(cli[:data_root]))
# Prepare derived CSVs once before parallel workers load the dataset.
load_pjm_battery_dataset(PJMBatteryDatasetConfig(
    data_root=cli[:data_root], split_fractions=cli[:split],
    fixed_demand=cli[:fixed_demand],
))
desired_processes = cli[:nprocs] == 0 ? min(length(seed_values), Sys.CPU_THREADS) : cli[:nprocs]
external_workers = filter(!=(1), workers())
if length(external_workers) < desired_processes - 1
    addprocs(desired_processes - 1 - length(external_workers); exeflags=`--project=$PROJECT_ROOT`)
end
@everywhere filter(!=(1), workers()) include(joinpath(@__DIR__, "pjm_ablation_worker.jl"))

const _split = cli[:split]
const _data_root = String(cli[:data_root])
const _fixed_demand = cli[:fixed_demand]
const _global_lambda0 = Float64(cli[:global_lambda0])
const _global_lr0 = Float64(cli[:global_lr0])
const _global_lr_lambda_alpha = Float64(cli[:global_lr_lambda_alpha])
const _spo_alpha = Float64(cli[:spo_alpha])
const _personal_kappa = Float64(cli[:personal_kappa])
const _personal_kappa_lr = Float64(cli[:personal_kappa_lr])
const _personal_lambda0 = Float64(cli[:personal_lambda0])
const _personal_freeze_tau = Float64(cli[:personal_freeze_tau])
const _personal_lr0 = Float64(cli[:personal_lr0])
const _personal_lr_lambda_alpha = Float64(cli[:personal_lr_lambda_alpha])
const _export_root = String(cli[:export_root])

t0 = time()

all_rows = pmap(seed_values; on_error = ex -> ex) do seed
    try
        rows = run_single_pjm_ablation_seed(
            seed=seed,
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
            export_root=_export_root,
            fixed_demand=_fixed_demand,
        )
        println("  ✓ done seed=$(seed) (worker $(myid()))")
        return rows
    catch ex
        @warn "PJM ablation failed" seed exception=(ex, catch_backtrace())
        return ex
    end
end

elapsed = round(time() - t0; digits=1)
println("\nAll ablation runs finished in $(elapsed) s")

successes = Vector{NamedTuple}()
n_failed = 0
for result in all_rows
    if result isa Exception
        global n_failed += 1
    else
        append!(successes, [merge(row, (fixed_demand=_fixed_demand,)) for row in result])
    end
end

isempty(successes) && error("all ablation runs failed")
n_failed == 0 || error("$n_failed PJM ablation seeds failed")

summary_rows = summarize_ablation_rows(successes)

raw_outfile = cli[:raw_outfile]
summary_outfile = cli[:summary_outfile]

write_namedtuple_csv(raw_outfile, successes)
write_namedtuple_csv(summary_outfile, summary_rows)

println("\nArtifacts written:")
println("  raw CSV      = ", raw_outfile)
println("  summary CSV  = ", summary_outfile)
println("  export root  = ", _export_root)
println("  rows         = ", length(successes))
println("  methods      = ", length(summary_rows))
println("  failed seeds = ", n_failed)
println("  timestamp    = ", Dates.now())
