#!/usr/bin/env julia
using Pkg
const PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(PROJECT_ROOT)
using Dates
include(joinpath(PROJECT_ROOT, "src", "models", "RSPOPlusLoss.jl"))
include(joinpath(PROJECT_ROOT, "src", "models", "scheduling.jl"))
include(joinpath(PROJECT_ROOT, "src", "datagen", "warcraft_dataset.jl"))
include(joinpath(PROJECT_ROOT, "src", "models", "warcraft_oracles.jl"))
include(joinpath(PROJECT_ROOT, "src", "models", "warcraft_training.jl"))
include(joinpath(PROJECT_ROOT, "src", "experiments", "warcraft.jl"))

function parse_cli(args::Vector{String})
    opts = Dict{Symbol,Any}(
        :mode => "two_stage",
        :method => "rspo_plus",
        :seed => 42,
        :n_clients => 4,
        :client_layout => "border_antipodes",
        :objective_heterogeneity => false,
        :terrain_heterogeneity_scale => 0.0,
        :warm_rounds => 10,
        :personal_rounds => 10,
        :warm_local_epochs => 3,
        :personal_local_epochs => 2,
        :warm_batch_size => 256,
        :personal_batch_size => 256,
        :warm_lambda0 => 2.0,
        :warm_lr0 => 1e-3,
        :personal_lambda0 => 2.0,
        :personal_kappa => 1.0,
        :personal_kappa_lr => 0.1,
        :personal_freeze_tau => 0.01,
        :personal_lr0 => 1e-3,
        :personal_lr_lambda_alpha => 0.001,
        :spo_alpha => 2.0,
        :perturbed_nb_samples => 10,
        :perturbed_epsilon => 1.0,
        :perturbed_threaded => false,
        :perturbed_seed => nothing,
        :diffopt_tau => 1e-3,
        :root => joinpath("results", "experiment2", "warcraft"),
    )

    idx = 1
    while idx <= length(args)
        flag = args[idx]
        if flag == "--mode"
            opts[:mode] = args[idx + 1]
            idx += 2
        elseif flag == "--method"
            opts[:method] = args[idx + 1]
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
        elseif flag == "--diffopt-tau"
            opts[:diffopt_tau] = parse(Float64, args[idx + 1])
            idx += 2
        elseif flag == "--seed"
            opts[:seed] = parse(Int, args[idx + 1])
            idx += 2
        elseif flag == "--n-clients"
            opts[:n_clients] = parse(Int, args[idx + 1])
            idx += 2
        elseif flag == "--client-layout"
            opts[:client_layout] = args[idx + 1]
            idx += 2
        elseif flag == "--objective-heterogeneity"
            opts[:objective_heterogeneity] = true
            idx += 1
        elseif flag == "--terrain-heterogeneity-scale"
            opts[:terrain_heterogeneity_scale] = parse(Float64, args[idx + 1])
            idx += 2
        elseif flag == "--rounds"
            rounds = parse(Int, args[idx + 1])
            opts[:warm_rounds] = rounds
            opts[:personal_rounds] = rounds
            idx += 2
        elseif flag == "--batch-size"
            batch_size = parse(Int, args[idx + 1])
            opts[:warm_batch_size] = batch_size
            opts[:personal_batch_size] = batch_size
            idx += 2
        elseif flag == "--warm-rounds"
            opts[:warm_rounds] = parse(Int, args[idx + 1])
            idx += 2
        elseif flag == "--personal-rounds"
            opts[:personal_rounds] = parse(Int, args[idx + 1])
            idx += 2
        elseif flag == "--warm-local-epochs"
            opts[:warm_local_epochs] = parse(Int, args[idx + 1])
            idx += 2
        elseif flag == "--personal-local-epochs"
            opts[:personal_local_epochs] = parse(Int, args[idx + 1])
            idx += 2
        elseif flag == "--warm-batch-size"
            opts[:warm_batch_size] = parse(Int, args[idx + 1])
            idx += 2
        elseif flag == "--personal-batch-size"
            opts[:personal_batch_size] = parse(Int, args[idx + 1])
            idx += 2
        elseif flag == "--warm-lambda0"
            opts[:warm_lambda0] = parse(Float64, args[idx + 1])
            idx += 2
        elseif flag == "--warm-lr0"
            opts[:warm_lr0] = parse(Float64, args[idx + 1])
            idx += 2
        elseif flag == "--personal-lambda0"
            opts[:personal_lambda0] = parse(Float64, args[idx + 1])
            idx += 2
        elseif flag == "--personal-kappa"
            opts[:personal_kappa] = parse(Float64, args[idx + 1])
            idx += 2
        elseif flag == "--personal-kappa-lr"
            opts[:personal_kappa_lr] = parse(Float64, args[idx + 1])
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
        elseif flag == "--root"
            opts[:root] = args[idx + 1]
            idx += 2
        else
            error("unknown flag `$flag`")
        end
    end

    if isnothing(opts[:personal_kappa_lr])
        opts[:personal_kappa_lr] = opts[:personal_kappa]
    end

    return opts
end

function main(args::Vector{String})
    opts = parse_cli(args)
    use_personal_per_client_lr = opts[:personal_lr_lambda_alpha] > 0
    objective = warcraft_objective_from_name(
        Symbol(opts[:method]);
        spo_alpha=opts[:spo_alpha],
        perturbed_nb_samples=opts[:perturbed_nb_samples],
        perturbed_epsilon=opts[:perturbed_epsilon],
        perturbed_threaded=opts[:perturbed_threaded],
        perturbed_seed=opts[:perturbed_seed],
        diffopt_tau=opts[:diffopt_tau],
    )
    dataset_config = WarcraftDatasetConfig(
        seed=opts[:seed],
        n_clients=opts[:n_clients],
        client_layout=Symbol(opts[:client_layout]),
        objective_heterogeneity=opts[:objective_heterogeneity],
        terrain_heterogeneity_scale=opts[:terrain_heterogeneity_scale],
    )
    warm_start_config = WarcraftTrainingConfig(
        seed=opts[:seed],
        rounds=opts[:warm_rounds],
        batch_size=opts[:warm_batch_size],
        lambda0=opts[:warm_lambda0],
        kappa_lambda=0.0,
        per_client_lambda=true,
        lr0=opts[:warm_lr0],
        kappa_lr=0.0,
        local_epochs=opts[:warm_local_epochs],
        per_client_lr=true,
        lr_lambda_alpha=opts[:personal_lr_lambda_alpha],
    )
    personalization_config = WarcraftTrainingConfig(
        seed=opts[:seed] + 1,
        rounds=opts[:personal_rounds],
        batch_size=opts[:personal_batch_size],
        client_fraction=1.0,
        lambda0=opts[:personal_lambda0],
        kappa_lambda=opts[:personal_kappa],
        per_client_lambda=true,
        lr0=opts[:personal_lr0],
        kappa_lr=opts[:personal_kappa_lr],
        per_client_lr=use_personal_per_client_lr,
        lr_lambda_alpha=opts[:personal_lr_lambda_alpha],
        freeze_tau=opts[:personal_freeze_tau],
        local_epochs=opts[:personal_local_epochs],
    )

    println("[$(Dates.now())] loading Warcraft dataset...")
    dataset = load_warcraft_dataset(dataset_config)
    println("benchmark train/test sizes: ",
        length(dataset.benchmark_train.samples), " / ", length(dataset.benchmark_test.samples))
    println(
        "dataset config: n_clients=$(dataset.config.n_clients), client_layout=$(dataset.config.client_layout), objective_heterogeneity=$(dataset.config.objective_heterogeneity), terrain_heterogeneity_scale=$(dataset.config.terrain_heterogeneity_scale)",
    )
    println("method: $(objective_name(objective))")
    println(
        "warm start config: rounds=$(warm_start_config.rounds), local_epochs=$(warm_start_config.local_epochs), batch_size=$(warm_start_config.batch_size), seed=$(warm_start_config.seed), lambda0=$(warm_start_config.lambda0), lr0=$(warm_start_config.lr0), per_client_lambda=$(warm_start_config.per_client_lambda), per_client_lr=$(warm_start_config.per_client_lr), lr_lambda_alpha=$(warm_start_config.lr_lambda_alpha)",
            )
    println(
        "personalization config: rounds=$(personalization_config.rounds), local_epochs=$(personalization_config.local_epochs), batch_size=$(personalization_config.batch_size), seed=$(personalization_config.seed), lambda0=$(personalization_config.lambda0), kappa_lambda=$(personalization_config.kappa_lambda), kappa_lr=$(personalization_config.kappa_lr), lr0=$(personalization_config.lr0), per_client_lambda=$(personalization_config.per_client_lambda), per_client_lr=$(personalization_config.per_client_lr), lr_lambda_alpha=$(personalization_config.lr_lambda_alpha), freeze_tau=$(personalization_config.freeze_tau)",
    )

    if opts[:mode] == "fed"
        println("[$(Dates.now())] running fed $(objective_name(objective)) warcraft...")
        result = fed_warcraft(dataset; objective=objective, config=warm_start_config)
        exported = export_warcraft_run(result, dataset; config=warm_start_config, root=opts[:root])
        println("exported to ", exported.dir)
    elseif opts[:mode] == "local"
        println("[$(Dates.now())] running local $(objective_name(objective)) warcraft...")
        result = local_warcraft(dataset; objective=objective, config=personalization_config)
        exported = export_warcraft_run(
            result,
            dataset;
            config=personalization_config,
            root=opts[:root],
        )
        println("exported to ", exported.dir)
    elseif opts[:mode] == "two_stage"
        println("[$(Dates.now())] running two-stage $(objective_name(objective)) Warcraft experiment...")
        result = run_warcraft_two_stage(
            dataset;
            objective=objective,
            warm_start_config=warm_start_config,
            personalization_config=personalization_config,
        )
        exported = export_warcraft_two_stage_run(
            result,
            dataset;
            warm_start_config=warm_start_config,
            personalization_config=personalization_config,
            root=opts[:root],
        )
        println("warm-start exported to ", exported.warm_start.dir)
        println("personalization exported to ", exported.personalization.dir)
    else
        error("mode must be `fed`, `local`, or `two_stage`")
    end
end

main(ARGS)
