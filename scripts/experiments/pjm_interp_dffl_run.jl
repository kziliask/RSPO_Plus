#!/usr/bin/env julia

using Pkg
const PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(PROJECT_ROOT)

if abspath(PROGRAM_FILE) == abspath(@__FILE__) && any(arg -> arg in ("--help", "-h"), ARGS)
    println("""
PJM Interp-DFFL baseline.
Usage: julia --project=. scripts/experiments/pjm_interp_dffl_run.jl [options]
  --seed-values 42,69,1993,2000,2026
  --lambda-grid LIST           Comma-separated weights; default grid is 0:0.05:1
  --split 0.5,0.1,0.4
  --data-root PATH
  --nprocs INT                 Total Julia processes; 0=automatic, 1=serial
  --outdir PATH
""")
    exit()
end

using Distributed
using Dates
using Printf
using Statistics

include(joinpath(@__DIR__, "pjm_interp_dffl_worker.jl"))

const DEFAULT_PJM_INTERP_OUTDIR =
    joinpath("results", "experiment3", "pjm_interp_dffl_matched")
const DEFAULT_PJM_INTERP_SEEDS = [42, 69, 1993, 2000, 2026]
function parse_pjm_interp_int_list(text::AbstractString)
    return [parse(Int, strip(token)) for token in split(text, ',') if !isempty(strip(token))]
end

function parse_pjm_interp_float_list(text::AbstractString)
    return [
        parse(Float64, strip(token)) for token in split(text, ',') if !isempty(strip(token))
    ]
end

function parse_pjm_interp_split(text::AbstractString)
    values = parse_pjm_interp_float_list(text)
    length(values) == 3 || error("expected three comma-separated split fractions")
    return (values[1], values[2], values[3])
end

function parse_pjm_interp_cli(args::Vector{String})
    options = Dict{Symbol,Any}(
        :seed_values => copy(DEFAULT_PJM_INTERP_SEEDS),
        :split => (0.5, 0.1, 0.4),
        :data_root => joinpath("src", "data", "exp3"),
        :lambda_grid => collect(0.0:0.05:1.0),
        :nprocs => 0,
        :outdir => DEFAULT_PJM_INTERP_OUTDIR,
    )
    idx = 1
    while idx <= length(args)
        flag = args[idx]
        if flag == "--seed-values"
            options[:seed_values] = parse_pjm_interp_int_list(args[idx + 1])
        elseif flag == "--split"
            options[:split] = parse_pjm_interp_split(args[idx + 1])
        elseif flag == "--data-root"
            options[:data_root] = args[idx + 1]
        elseif flag == "--lambda-grid"
            options[:lambda_grid] = parse_pjm_interp_float_list(args[idx + 1])
        elseif flag == "--nprocs"
            options[:nprocs] = parse(Int, args[idx + 1])
        elseif flag == "--outdir"
            options[:outdir] = args[idx + 1]
        else
            error("unknown flag `$flag`")
        end
        idx += 2
    end
    return options
end

function _pjm_interp_csv_escape(value)
    text = string(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function write_pjm_interp_csv(path::AbstractString, rows::Vector{<:NamedTuple})
    isempty(rows) && error("cannot write empty CSV to `$path`")
    mkpath(dirname(path))
    header = collect(keys(rows[1]))
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(
                io,
                join((_pjm_interp_csv_escape(getfield(row, key)) for key in header), ","),
            )
        end
    end
    return path
end

function summarize_pjm_interp(seed_metrics::Vector{<:NamedTuple})
    rows = NamedTuple[]
    for selector in (
        "local_endpoint",
        "federated_endpoint",
        "calibrated_spo",
        "calibrated_mse",
    )
        selected = [row for row in seed_metrics if row.selector == selector]
        isempty(selected) && continue
        regrets = [row.aggregate_absolute_regret_mean for row in selected]
        mses = [row.aggregate_mse_mean for row in selected]
        relative_regrets = [row.aggregate_relative_regret_mean for row in selected]
        absolute_regret_population_std =
            length(regrets) == 1 ? 0.0 : std(regrets; corrected=false)
        absolute_regret_sample_std =
            length(regrets) == 1 ? 0.0 : std(regrets; corrected=true)
        push!(
            rows,
            (
                selector=selector,
                mean_absolute_regret=mean(regrets),
                std_absolute_regret=absolute_regret_population_std,
                std_absolute_regret_population=absolute_regret_population_std,
                std_absolute_regret_sample=absolute_regret_sample_std,
                mean_mse=mean(mses),
                std_mse=length(mses) == 1 ? 0.0 : std(mses; corrected=false),
                mean_relative_regret=mean(relative_regrets),
                std_relative_regret=length(relative_regrets) == 1 ? 0.0 :
                                    std(relative_regrets; corrected=false),
                n_seeds=length(selected),
                seed_values=join(sort([row.seed for row in selected]), ";"),
            ),
        )
    end
    return rows
end

options = parse_pjm_interp_cli(ARGS)
seed_values = options[:seed_values]
isempty(seed_values) && error("at least one seed is required")
length(unique(seed_values)) == length(seed_values) || error("seed values must be unique")
options[:nprocs] >= 0 || error("nprocs must be nonnegative")
options[:data_root] = abspath(PROJECT_ROOT, String(options[:data_root]))
# Prepare derived CSVs once before parallel workers load the dataset.
load_pjm_battery_dataset(PJMBatteryDatasetConfig(
    data_root=options[:data_root], split_fractions=options[:split],
))
desired_processes = options[:nprocs] == 0 ? min(length(seed_values), Sys.CPU_THREADS) : options[:nprocs]
external_workers = filter(!=(1), workers())
if length(external_workers) < desired_processes - 1
    addprocs(desired_processes - 1 - length(external_workers); exeflags=`--project=$PROJECT_ROOT`)
end
@everywhere filter(!=(1), workers()) include(joinpath(@__DIR__, "pjm_interp_dffl_worker.jl"))

const _pjm_interp_split = options[:split]
const _pjm_interp_data_root = String(options[:data_root])
const _pjm_interp_lambda_grid = Float64.(options[:lambda_grid])

started = time()
raw_results = pmap(seed_values; on_error = ex -> ex) do seed
    try
        result = run_single_pjm_interp_dffl_seed(
            seed=seed,
            split_fractions=_pjm_interp_split,
            data_root=_pjm_interp_data_root,
            lambda_grid=_pjm_interp_lambda_grid,
        )
        println("  done seed=$(seed) (worker $(myid()))")
        result
    catch ex
        @warn "PJM Interp-DFFL failed" seed exception=(ex, catch_backtrace())
        ex
    end
end
elapsed = time() - started

successes = [result for result in raw_results if !(result isa Exception)]
failures = [result for result in raw_results if result isa Exception]
isempty(successes) && error("all PJM Interp-DFFL runs failed")
isempty(failures) || error("$(length(failures)) PJM Interp-DFFL seed runs failed")

seed_metrics = reduce(vcat, [result.seed_metrics for result in successes])
client_metrics = reduce(vcat, [result.client_metrics for result in successes])
lambda_selections = reduce(vcat, [result.lambda_selections for result in successes])
calibration_curves = reduce(vcat, [result.calibration_curves for result in successes])
endpoint_training = reduce(vcat, [result.endpoint_training for result in successes])
metadata_rows = [result.metadata for result in successes]
summary_rows = summarize_pjm_interp(seed_metrics)

outdir = String(options[:outdir])
mkpath(outdir)
write_pjm_interp_csv(joinpath(outdir, "seed_metrics.csv"), seed_metrics)
write_pjm_interp_csv(joinpath(outdir, "client_metrics.csv"), client_metrics)
write_pjm_interp_csv(joinpath(outdir, "selected_lambdas.csv"), lambda_selections)
write_pjm_interp_csv(joinpath(outdir, "calibration_curves.csv"), calibration_curves)
write_pjm_interp_csv(joinpath(outdir, "endpoint_training.csv"), endpoint_training)
write_pjm_interp_csv(joinpath(outdir, "run_metadata.csv"), metadata_rows)
write_pjm_interp_csv(joinpath(outdir, "aggregate_summary.csv"), summary_rows)
println("\nArtifacts written to ", outdir)
println("  seeds        = ", join(seed_values, ", "))
println("  elapsed      = ", round(elapsed; digits=1), " seconds")
for row in summary_rows
    println(
        "  ",
        rpad(row.selector, 20),
        " absolute regret = ",
        @sprintf("%.6f +/- %.6f", row.mean_absolute_regret, row.std_absolute_regret),
    )
end
println("  timestamp    = ", Dates.now())
