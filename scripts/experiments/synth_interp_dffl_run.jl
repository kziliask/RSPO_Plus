#!/usr/bin/env julia

"""
Run the Interp-DFFL baseline for the paper's synthetic fractional-knapsack
case study.

Shared settings follow the FedRSPO+ paper: a 64-unit shallow network, batch
size 64, Adam at 1e-3, gradient clipping at norm 1, 20 clients, 100 training
and 10 validation observations per client, and the five paper seeds.

Interp-DFFL settings:

- deterministic client-wise 80/20 fit/calibration split of the training rows;
- independently trained local SPO+ models (50 epochs);
- one sample-size-weighted FedAvg SPO+ model (50 rounds, 5 local epochs,
  20% client participation);
- prediction-space interpolation over 0:0.05:1;
- separate per-client SPO+ and MSE selectors;
- 1,000 independent test observations per client.

The generated Julia artifact contains both endpoint models, all per-client
weights, preprocessing state, client optimization parameters, and split
metadata. CSV exports contain complete calibration curves and test metrics.
"""

using Dates
using Pkg

Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "..", "..", "src", "models", "knapsack_oracles.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "models", "RSPOPlusLoss.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "models", "scheduling.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "datagen", "synthetic_knapsack.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "models", "training.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "models", "interp_dffl.jl"))

function _parse_int_values(value::AbstractString)
    return [parse(Int, strip(token)) for token in split(value, ',') if !isempty(strip(token))]
end

function _parse_float_values(value::AbstractString)
    return [
        parse(Float64, strip(token)) for token in split(value, ',') if !isempty(strip(token))
    ]
end

function _interp_cli(args::Vector{String})
    options = Dict{Symbol,Any}(
        :seed_values => [42, 69, 1993, 67, 2026],
        :eta_obj_values => [0.0, 0.5, 1.0],
        :eta_constr_values => [0.0, 0.5, 1.0],
        :eta_data_values => [0.0, 0.5, 1.0],
        :n_clients => 20,
        :train_per_client => 100,
        :validation_per_client => 10,
        :test_per_client => 1_000,
        :fit_fraction => 0.8,
        :hidden_dim => 64,
        :local_epochs => 50,
        :federated_rounds => 50,
        :federated_local_epochs => 5,
        :batch_size => 64,
        :client_fraction => 0.2,
        :learning_rate => 1e-3,
        :clip_norm => 1.0,
        :output_norm_bound => 20.0,
        :artifact_root => joinpath(
            "results",
            "experiment1",
            "synthetic_knapsack",
            "interp_dffl",
        ),
        :summary_file => joinpath(
            "results",
            "experiment1",
            "synthetic_knapsack",
            "interp_dffl_summary.csv",
        ),
    )

    idx = 1
    while idx <= length(args)
        flag = args[idx]
        if flag == "--seed-values"
            options[:seed_values] = _parse_int_values(args[idx + 1])
            idx += 2
        elseif flag == "--eta-obj-values"
            options[:eta_obj_values] = _parse_float_values(args[idx + 1])
            idx += 2
        elseif flag == "--eta-constr-values"
            options[:eta_constr_values] = _parse_float_values(args[idx + 1])
            idx += 2
        elseif flag == "--eta-data-values"
            options[:eta_data_values] = _parse_float_values(args[idx + 1])
            idx += 2
        elseif flag == "--n-clients"
            options[:n_clients] = parse(Int, args[idx + 1])
            idx += 2
        elseif flag == "--train-per-client"
            options[:train_per_client] = parse(Int, args[idx + 1])
            idx += 2
        elseif flag == "--validation-per-client"
            options[:validation_per_client] = parse(Int, args[idx + 1])
            idx += 2
        elseif flag == "--test-per-client"
            options[:test_per_client] = parse(Int, args[idx + 1])
            idx += 2
        elseif flag == "--fit-fraction"
            options[:fit_fraction] = parse(Float64, args[idx + 1])
            idx += 2
        elseif flag == "--hidden-dim"
            options[:hidden_dim] = parse(Int, args[idx + 1])
            idx += 2
        elseif flag == "--local-epochs"
            options[:local_epochs] = parse(Int, args[idx + 1])
            idx += 2
        elseif flag == "--federated-rounds"
            options[:federated_rounds] = parse(Int, args[idx + 1])
            idx += 2
        elseif flag == "--federated-local-epochs"
            options[:federated_local_epochs] = parse(Int, args[idx + 1])
            idx += 2
        elseif flag == "--batch-size"
            options[:batch_size] = parse(Int, args[idx + 1])
            idx += 2
        elseif flag == "--client-fraction"
            options[:client_fraction] = parse(Float64, args[idx + 1])
            idx += 2
        elseif flag == "--learning-rate"
            options[:learning_rate] = parse(Float64, args[idx + 1])
            idx += 2
        elseif flag == "--clip-norm"
            options[:clip_norm] = parse(Float64, args[idx + 1])
            idx += 2
        elseif flag == "--output-norm-bound"
            options[:output_norm_bound] = parse(Float64, args[idx + 1])
            idx += 2
        elseif flag == "--artifact-root"
            options[:artifact_root] = args[idx + 1]
            idx += 2
        elseif flag == "--summary-file"
            options[:summary_file] = args[idx + 1]
            idx += 2
        elseif flag == "--smoke"
            options[:seed_values] = [42]
            options[:eta_obj_values] = [0.0]
            options[:eta_constr_values] = [0.0]
            options[:eta_data_values] = [0.0]
            options[:n_clients] = 2
            options[:train_per_client] = 5
            options[:validation_per_client] = 0
            options[:test_per_client] = 5
            options[:hidden_dim] = 4
            options[:local_epochs] = 1
            options[:federated_rounds] = 1
            options[:federated_local_epochs] = 1
            options[:batch_size] = 4
            options[:client_fraction] = 1.0
            idx += 1
        elseif flag in ("--help", "-h")
            println(
                """
                Usage: julia --project=. scripts/experiments/synth_interp_dffl_run.jl [options]

                  --seed-values CSV
                  --eta-obj-values CSV
                  --eta-constr-values CSV
                  --eta-data-values CSV
                  --n-clients INT
                  --train-per-client INT
                  --validation-per-client INT
                  --test-per-client INT
                  --fit-fraction FLOAT
                  --hidden-dim INT
                  --local-epochs INT
                  --federated-rounds INT
                  --federated-local-epochs INT
                  --batch-size INT
                  --client-fraction FLOAT
                  --learning-rate FLOAT
                  --clip-norm FLOAT
                  --output-norm-bound FLOAT
                  --artifact-root PATH
                  --summary-file PATH
                  --smoke
                """,
            )
            return nothing
        else
            throw(ArgumentError("unknown flag `$flag`; use --help for supported options"))
        end
    end
    return options
end

function _interp_summary_rows(
    result::InterpDFFLTrainingResult,
    evaluations,
    dataset::SyntheticKnapsackDataset,
    export_dir::AbstractString,
)
    rows = NamedTuple[]
    for evaluation in (evaluations.spo_plus, evaluations.mse)
        aggregate = evaluation.aggregate
        calibration_by_client = result.calibrations
        for client in evaluation.clients
            calibration = calibration_by_client[client.client_id]
            push!(
                rows,
                (
                    method="interp_dffl",
                    selector=evaluation.selector,
                    seed=dataset.config.seed,
                    eta_obj=dataset.config.eta_obj,
                    eta_constr=dataset.config.eta_constr,
                    eta_data_dist=dataset.config.eta_data_dist,
                    client_id=client.client_id,
                    fit_count=length(
                        result.split_metadata.fit_sample_ids[client.client_id],
                    ),
                    calibration_count=length(
                        result.split_metadata.calibration_sample_ids[client.client_id],
                    ),
                    test_count=client.n_test_samples,
                    selected_lambda=client.lambda,
                    lambda_spo_plus=calibration.lambda_spo_plus,
                    lambda_mse=calibration.lambda_mse,
                    client_mse=client.mse,
                    client_absolute_regret=client.absolute_regret,
                    client_relative_regret=client.relative_regret,
                    macro_mse=aggregate.mse.mean,
                    macro_absolute_regret=aggregate.absolute_regret.mean,
                    macro_relative_regret=aggregate.relative_regret.mean,
                    objective_pairwise_mean=dataset.metadata.objective_pairwise_mean,
                    capacity_pairwise_mean=dataset.metadata.capacity_pairwise_mean,
                    feature_mean_pairwise_mean=dataset.metadata.feature_mean_pairwise_mean,
                    artifact_dir=export_dir,
                ),
            )
        end
    end
    return rows
end

function _write_interp_summary(path::AbstractString, rows::Vector{<:NamedTuple})
    isempty(rows) && throw(ArgumentError("cannot write an empty Interp-DFFL summary"))
    mkpath(dirname(path))
    header = collect(keys(first(rows)))
    table_rows = ([getproperty(row, column) for column in header] for row in rows)
    return _write_interp_csv(path, string.(header), table_rows)
end

function main(args::Vector{String}=ARGS)
    options = _interp_cli(args)
    isnothing(options) && return nothing
    summary_rows = NamedTuple[]
    started_at = time()

    grid = collect(
        Iterators.product(
            options[:seed_values],
            options[:eta_obj_values],
            options[:eta_constr_values],
            options[:eta_data_values],
        ),
    )
    println("Running $(length(grid)) Interp-DFFL synthetic configuration(s)")

    for (run_idx, (seed, eta_obj, eta_constr, eta_data)) in enumerate(grid)
        println(
            "[$run_idx/$(length(grid))] seed=$seed eta=($eta_obj,$eta_constr,$eta_data)",
        )
        n_clients = options[:n_clients]
        dataset_config = SyntheticKnapsackConfig(
            seed=seed,
            n_clients=n_clients,
            p=8,
            dim=50,
            deg=4,
            epsilon_noise=0.0,
            eta_obj=eta_obj,
            eta_constr=eta_constr,
            eta_constr_affects_weights=true,
            eta_data_dist=eta_data,
            data_imbalance=false,
            n_train_total=n_clients * options[:train_per_client],
            n_val_total=n_clients * options[:validation_per_client],
            n_test_total=n_clients * options[:test_per_client],
            capacity_ratio=0.6,
        )
        dataset = generate_synthetic_knapsack_dataset(dataset_config)
        interp_config = InterpDFFLConfig(
            seed=seed,
            fit_fraction=options[:fit_fraction],
            hidden_dim=options[:hidden_dim],
            local_epochs=options[:local_epochs],
            federated_rounds=options[:federated_rounds],
            federated_local_epochs=options[:federated_local_epochs],
            batch_size=options[:batch_size],
            client_fraction=options[:client_fraction],
            learning_rate=options[:learning_rate],
            clip_norm=options[:clip_norm],
            output_norm_bound=options[:output_norm_bound],
        )

        result = train_interp_dffl(dataset; config=interp_config)
        run_id = join(
            (
                "seed$seed",
                "obj$eta_obj",
                "constr$eta_constr",
                "data$eta_data",
            ),
            "_",
        )
        exported = export_interp_dffl_run(
            result;
            root=options[:artifact_root],
            run_id=run_id,
        )
        append!(
            summary_rows,
            _interp_summary_rows(
                result,
                exported.evaluations,
                dataset,
                exported.dir,
            ),
        )
    end

    summary_path = _write_interp_summary(options[:summary_file], summary_rows)
    println(
        "Completed in $(round(time() - started_at; digits=1))s; summary: $summary_path",
    )
    return summary_path
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
