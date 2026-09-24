using Flux
using LinearAlgebra
using Metalhead
using Optimisers
using Random
using Statistics

if !isdefined(Main, :SchedulerTrace)
    struct SchedulerTrace
        lambda_values::Vector{Float64}
        lr_values::Vector{Float64}
        bound_values::Vector{Float64}
        frozen_after_round::Union{Nothing,Int}
        per_client_lambda0::Union{Nothing,Vector{Float64}}
    end

    SchedulerTrace(lv, lr, bv, far) = SchedulerTrace(lv, lr, bv, far, nothing)
end

Base.@kwdef struct WarcraftTrainingConfig
    seed::Int = 42
    rounds::Int = 50
    local_epochs::Int = 3
    batch_size::Int = 32
    client_fraction::Float64 = 0.5
    validation_client_fraction::Float64 = 0.25
    validation_max_samples_per_client::Int = 64
    lambda0::Float64 = 2.0
    kappa_lambda::Float64 = 0.05
    per_client_lambda::Bool = true
    lr0::Float64 = 1e-3
    kappa_lr::Float64 = 0.05
    per_client_lr::Bool = true
    lr_lambda_alpha::Float64 = 0.01
    clip_norm::Float64 = 1.0
    freeze_tau::Float64 = 0.03
    freeze_eps::Float64 = 1e-8
    stop_after_freeze_rounds::Int = 3
    shuffle_batches::Bool = true
    use_warm_start::Bool = true
    pretrain_backbone::Bool = false
end

struct WarcraftSample
    sample_id::Int
    image::AbstractArray{Float32,3}
    raw_y_true::AbstractMatrix{Float32}
    raw_weights::AbstractMatrix{Float32}
    theta_true::Matrix{Float64}
    y_true::Matrix{Float64}
    instance::WarcraftInstance
end

struct WarcraftClientData
    client_id::Int
    reference_instance::WarcraftInstance
    train::Vector{WarcraftSample}
    val::Vector{WarcraftSample}
    test::Vector{WarcraftSample}
end

struct WarcraftValidationClientData
    client_id::Int
    samples::Vector{WarcraftSample}
    z_star::Vector{Float64}
end

struct WarcraftValidationMonitor
    clients::Vector{WarcraftValidationClientData}
end

abstract type WarcraftObjective end

struct WarcraftRSPOPlusObjective <: WarcraftObjective end

struct WarcraftInferOptObjective{B,T} <: WarcraftObjective
    name::Symbol
    loss_builder::B
    target_getter::T
end

struct WarcraftPerturbedMSEObjective{B} <: WarcraftObjective
    name::Symbol
    layer_builder::B
end

struct WarcraftDiffOptObjective <: WarcraftObjective
    tau::Float64
end

struct WarcraftMSEObjective <: WarcraftObjective end

struct WarcraftFedTrainingResult
    method::Symbol
    model
    client_data::Vector{WarcraftClientData}
    round_losses::Vector{Float64}
    selected_clients::Vector{Vector{Int}}
    trace::SchedulerTrace
end

struct WarcraftLocalClientTrainingResult
    client_id::Int
    model
    round_losses::Vector{Float64}
    trace::SchedulerTrace
end

struct WarcraftLocalTrainingResult
    method::Symbol
    clients::Vector{WarcraftLocalClientTrainingResult}
end

struct WarcraftTwoStageTrainingResult
    warm_start::WarcraftFedTrainingResult
    personalization::WarcraftLocalTrainingResult
end

objective_name(::WarcraftRSPOPlusObjective) = :rspo_plus
objective_name(objective::WarcraftInferOptObjective) = objective.name
objective_name(objective::WarcraftPerturbedMSEObjective) = objective.name
objective_name(::WarcraftDiffOptObjective) = :diffopt
objective_name(::WarcraftMSEObjective) = :mse
objective_uses_lambda(::WarcraftObjective) = false
objective_uses_lambda(::WarcraftRSPOPlusObjective) = true
objective_tracks_bound(::WarcraftObjective) = false
objective_tracks_bound(::WarcraftRSPOPlusObjective) = true

function objective_method(objective::WarcraftObjective, phase::Symbol)
    phase in (:fed, :local, :centralized) ||
        throw(ArgumentError("phase must be :fed, :local, or :centralized, got $phase"))
    return Symbol(phase, "_", objective_name(objective), "_warcraft")
end

function _validate_warcraft_training_config(config::WarcraftTrainingConfig)
    config.rounds > 0 || throw(ArgumentError("rounds must be positive"))
    config.local_epochs > 0 || throw(ArgumentError("local_epochs must be positive"))
    config.batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    config.client_fraction > 0 || throw(ArgumentError("client_fraction must be positive"))
    config.validation_client_fraction > 0 ||
        throw(ArgumentError("validation_client_fraction must be positive"))
    config.per_client_lambda || config.lambda0 > 0 ||
        throw(ArgumentError("lambda0 must be positive (or set per_client_lambda=true)"))
    config.kappa_lambda >= 0 || throw(ArgumentError("kappa_lambda must be nonnegative"))
    config.lr0 > 0 || throw(ArgumentError("lr0 must be positive"))
    config.kappa_lr >= 0 || throw(ArgumentError("kappa_lr must be nonnegative"))
    config.lr_lambda_alpha >= 0 ||
        throw(ArgumentError("lr_lambda_alpha must be nonnegative"))
    config.clip_norm > 0 || throw(ArgumentError("clip_norm must be positive"))
    config.freeze_eps > 0 || throw(ArgumentError("freeze_eps must be positive"))
    config.stop_after_freeze_rounds >= 0 ||
        throw(ArgumentError("stop_after_freeze_rounds must be nonnegative"))
    return config
end

function _resolve_warcraft_weights(
    raw_weights::AbstractMatrix{<:Real},
    terrain_values::Vector{Float32},
    terrain_multipliers::Vector{Float64},
)
    weights = Matrix{Float64}(raw_weights)
    if isempty(terrain_values) || isempty(terrain_multipliers)
        return weights
    end
    length(terrain_values) == length(terrain_multipliers) || throw(
        ArgumentError("terrain multiplier length does not match terrain values"),
    )
    for idx in eachindex(terrain_values)
        terrain_value = Float64(terrain_values[idx])
        multiplier = terrain_multipliers[idx]
        weights[weights .== terrain_value] .*= multiplier
    end
    return weights
end

function _make_warcraft_sample(
    raw_sample::WarcraftRawSample,
    manifest::WarcraftClientManifest,
    terrain_values::Vector{Float32},
)
    resolved_weights = _resolve_warcraft_weights(
        raw_sample.true_weights,
        terrain_values,
        manifest.terrain_multipliers,
    )
    instance = WarcraftInstance(
        resolved_weights,
        manifest.source,
        manifest.sink;
        endpoint_kind=manifest.endpoint_kind,
        terrain_multipliers=manifest.terrain_multipliers,
    )
    y_true = solve_warcraft_path(resolved_weights; instance=instance)
    return WarcraftSample(
        raw_sample.sample_id,
        raw_sample.image,
        raw_sample.y_true,
        raw_sample.true_weights,
        resolved_weights,
        y_true,
        instance,
    )
end

function _reference_instance(
    train::Vector{WarcraftSample},
    val::Vector{WarcraftSample},
    test::Vector{WarcraftSample},
)
    for collection in (train, val, test)
        isempty(collection) || return first(collection).instance
    end
    throw(ArgumentError("a Warcraft client must contain at least one sample"))
end

function prepare_warcraft_client_data(dataset::WarcraftDataset)
    t0 = time()
    terrain_values = dataset.metadata.terrain_values
    n_clients = dataset.config.n_clients
    n_test = length(dataset.benchmark_test.samples)
    println("  preparing client data: $(n_clients) clients, $(length(dataset.benchmark_train.samples)) train+val samples, $(n_test) test samples")
    clients = Vector{WarcraftClientData}(undef, n_clients)
    for manifest in dataset.partition_manifest.clients
        tc = time()
        train = [
            _make_warcraft_sample(sample, manifest, terrain_values) for
            sample in dataset.benchmark_train.samples[manifest.train_indices]
        ]
        val = [
            _make_warcraft_sample(sample, manifest, terrain_values) for
            sample in dataset.benchmark_train.samples[manifest.val_indices]
        ]
        test = [
            _make_warcraft_sample(sample, manifest, terrain_values) for
            sample in dataset.benchmark_test.samples
        ]
        clients[manifest.client_id] = WarcraftClientData(
            manifest.client_id,
            _reference_instance(train, val, test),
            train,
            val,
            test,
        )
        println("    client $(manifest.client_id)/$(n_clients): $(length(train)) train, $(length(val)) val, $(length(test)) test samples [$(round(time() - tc; digits=1))s]")
    end
    println("  client data ready [$(round(time() - t0; digits=1))s total]")
    return clients
end

function prepare_centralized_warcraft_client_data(dataset::WarcraftDataset)
    t0 = time()
    terrain_values = dataset.metadata.terrain_values
    manifests = dataset.partition_manifest.clients
    pooled_train_indices = sort!(reduce(vcat, [manifest.train_indices for manifest in manifests]; init=Int[]))
    n_clients = length(manifests)
    println(
        "  preparing centralized client data: $(n_clients) target clients, $(length(pooled_train_indices)) pooled train samples",
    )
    clients = Vector{WarcraftClientData}(undef, n_clients)

    for target_manifest in manifests
        tc = time()
        train = [
            _make_warcraft_sample(sample, target_manifest, terrain_values) for
            sample in dataset.benchmark_train.samples[pooled_train_indices]
        ]
        val = [
            _make_warcraft_sample(sample, target_manifest, terrain_values) for
            sample in dataset.benchmark_train.samples[target_manifest.val_indices]
        ]
        test = [
            _make_warcraft_sample(sample, target_manifest, terrain_values) for
            sample in dataset.benchmark_test.samples
        ]
        clients[target_manifest.client_id] = WarcraftClientData(
            target_manifest.client_id,
            _reference_instance(train, val, test),
            train,
            val,
            test,
        )
        println(
            "    centralized client $(target_manifest.client_id)/$(n_clients): $(length(train)) pooled train, $(length(val)) val, $(length(test)) test [$(round(time() - tc; digits=1))s]",
        )
    end

    println("  centralized client data ready [$(round(time() - t0; digits=1))s total]")
    return clients
end

function _sample_subset(
    rng::AbstractRNG,
    candidates::AbstractVector{Int},
    subset_spec::Real,
)
    isempty(candidates) && return Int[]
    total = length(candidates)
    n_select =
        subset_spec <= 1 ? clamp(ceil(Int, subset_spec * total), 1, total) :
        clamp(round(Int, subset_spec), 1, total)
    order = randperm(rng, total)[1:n_select]
    return sort(candidates[order])
end

function _batch_index_sets(
    rng::AbstractRNG,
    n_samples::Int,
    batch_size::Int;
    shuffle::Bool=true,
)
    n_samples == 0 && return Vector{Vector{Int}}()
    order = shuffle ? randperm(rng, n_samples) : collect(1:n_samples)
    batches = Vector{Vector{Int}}()
    for start_idx in 1:batch_size:n_samples
        stop_idx = min(start_idx + batch_size - 1, n_samples)
        push!(batches, collect(order[start_idx:stop_idx]))
    end
    return batches
end

function _batch_images(samples::Vector{WarcraftSample}, indices::AbstractVector{Int})
    isempty(indices) && return zeros(Float32, 96, 96, 3, 0)
    height, width, channels = size(samples[first(indices)].image)
    x = Array{Float32}(undef, height, width, channels, length(indices))
    for (batch_pos, sample_idx) in enumerate(indices)
        x[:, :, :, batch_pos] .= samples[sample_idx].image
    end
    return x
end

function _single_sample_prediction(model, sample::WarcraftSample)
    batch = Array{Float32}(undef, size(sample.image)..., 1)
    batch[:, :, :, 1] .= sample.image
    prediction = max.(model(batch), zero(Float32))
    return Array{Float64}(dropdims(prediction; dims=3))
end

function _optimizer_rule(config::WarcraftTrainingConfig, lr::Real)
    return Optimisers.OptimiserChain(Optimisers.ClipNorm(config.clip_norm), Optimisers.Adam(lr))
end

function build_warcraft_model(
    dataset::WarcraftDataset;
    pretrain::Bool=false,
    rng::AbstractRNG=Random.default_rng(),
)
    t0 = time()
    grid_dim = dataset.config.grid_dim
    backbone = Metalhead.backbone(Metalhead.ResNet(18; pretrain=pretrain))[1:2]
    pool = Flux.AdaptiveMaxPool((grid_dim, grid_dim))
    model = Flux.Chain(
        backbone,
        pool,
        x -> dropdims(mean(x; dims=3), dims=3),
    )
    result = Flux.f32(model)
    n_params = sum(length, Flux.trainables(result); init=0)
    println(
        "  model built: CombResNet18-style ResNet18[1:2] -> AdaptiveMaxPool($(grid_dim)x$(grid_dim)) -> channel mean, $(n_params) params [$(round(time() - t0; digits=1))s]",
    )
    return result
end

function build_warcraft_validation_monitor(
    client_data::Vector{WarcraftClientData},
    config::WarcraftTrainingConfig;
    rng::AbstractRNG=MersenneTwister(config.seed),
    client_ids::Union{Nothing,AbstractVector{Int}}=nothing,
)
    if isnothing(client_ids)
        available = [client.client_id for client in client_data if !isempty(client.val)]
        selected_client_ids = _sample_subset(rng, available, config.validation_client_fraction)
    else
        selected_client_ids = collect(client_ids)
    end

    monitor_clients = WarcraftValidationClientData[]
    for client_id in selected_client_ids
        client = client_data[client_id]
        isempty(client.val) && continue
        selected =
            length(client.val) <= config.validation_max_samples_per_client ?
            client.val :
            client.val[
                sort(randperm(rng, length(client.val))[1:config.validation_max_samples_per_client])
            ]
        z_star = [warcraft_path_cost(sample.theta_true, sample.y_true) for sample in selected]
        push!(monitor_clients, WarcraftValidationClientData(client_id, selected, z_star))
    end

    return WarcraftValidationMonitor(monitor_clients)
end

function _mean_warcraft_rspo_loss(
    samples::Vector{WarcraftSample},
    predictions,
    cache::WarcraftProjectionCache,
    lambda::Real,
    config::WarcraftTrainingConfig,
)
    loss_layer = RSPOPlusLoss(projection_optimizer; lambda=lambda)
    losses = Float64[]
    for idx in eachindex(samples)
        sample = samples[idx]
        prediction = view(predictions, :, :, idx)
        v_opt = projection_optimizer(
            sample.theta_true;
            instance=sample.instance,
            cache=cache,
            lambda=lambda,
            use_warm_start=config.use_warm_start,
        )
        loss_value = loss_layer(
            prediction,
            sample.theta_true;
            instance=sample.instance,
            cache=cache,
            lambda=lambda,
            use_warm_start=config.use_warm_start,
            v_opt=v_opt,
        )
        push!(losses, float(loss_value))
    end
    return isempty(losses) ? NaN : mean(losses)
end

"""
    compute_warcraft_freeze_bound(model, monitor; lambda, lambdas, freeze_eps, use_warm_start)

Compute the global regularization-bias ratio across all validation samples:

    mean_i[c_i^T w_{it}^{*,reg} - z_i^*] / (mean_i[|z_i^*|] + ε)

where `c_i` is the ground-truth cost, `w_{it}^{*,reg}` is the optimal regularized
decision at the current lambda, and `z_i^*` is the unregularized optimal objective.
The mean is taken over the union of all validation samples in `monitor`, so this is
a single ratio of averages rather than an average of per-sample or per-client ratios.
"""
function compute_warcraft_freeze_bound(
    model,
    monitor::WarcraftValidationMonitor;
    lambda::Union{Nothing,Real}=nothing,
    lambdas::Union{Nothing,AbstractVector{<:Real}}=nothing,
    freeze_eps::Real=1e-8,
    use_warm_start::Bool=true,
)
    isempty(monitor.clients) && return Inf

    t_bound = time()
    numerator_sum = 0.0
    denominator_sum = 0.0
    n_validation_samples = 0

    for client in monitor.clients
        lam = _resolve_warcraft_bound_lambda(
            client.client_id;
            lambda=lambda,
            lambdas=lambdas,
        )
        cache = init_warcraft_projection_cache(first(client.samples).instance; lambda=lam)

        for (sample_idx, sample) in enumerate(client.samples)
            w_star_reg = projection_optimizer(
                sample.theta_true;
                instance=sample.instance,
                cache=cache,
                lambda=lam,
                use_warm_start=use_warm_start,
            )
            # c_i^T * w_{it}^{*,reg}
            reg_objective = warcraft_path_cost(sample.theta_true, w_star_reg)
            z_star = client.z_star[sample_idx]
            numerator_sum += reg_objective - z_star
            denominator_sum += abs(z_star)
            n_validation_samples += 1
        end
    end

    n_validation_samples == 0 && return Inf
    mean_bias = numerator_sum / n_validation_samples
    mean_abs_z_star = denominator_sum / n_validation_samples
    result = mean_bias / (mean_abs_z_star + freeze_eps)
    println("      freeze bound: $(round(result; sigdigits=4)) ($(n_validation_samples) val samples) [$(round(time() - t_bound; digits=2))s]")
    return result
end

_initial_client_objective_state(::WarcraftObjective, client, config) = nothing

function _require_warcraft_lambda(lambda::Real)
    isfinite(lambda) && lambda > 0 ||
        throw(ArgumentError("lambda must be positive and finite, got $lambda"))
    return float(lambda)
end

function _resolve_warcraft_bound_lambda(
    client_id::Integer;
    lambda::Union{Nothing,Real}=nothing,
    lambdas::Union{Nothing,AbstractVector{<:Real}}=nothing,
)
    if !isnothing(lambda) && !isnothing(lambdas)
        throw(ArgumentError("pass either `lambda` or `lambdas`, not both"))
    elseif !isnothing(lambdas)
        return _require_warcraft_lambda(lambdas[client_id])
    elseif !isnothing(lambda)
        return _require_warcraft_lambda(lambda)
    end
    throw(UndefKeywordError(:lambda))
end

function _initial_client_objective_state(
    ::WarcraftRSPOPlusObjective,
    client::WarcraftClientData,
    config::WarcraftTrainingConfig,
)
    return init_warcraft_projection_cache(client.reference_instance; lambda=config.lambda0)
end

function _initial_client_objective_state(
    objective::WarcraftDiffOptObjective,
    client::WarcraftClientData,
    config::WarcraftTrainingConfig,
)
    return init_warcraft_diffopt_layer(client.reference_instance; tau=objective.tau)
end

function compute_warcraft_client_c_norm(client::WarcraftClientData)
    isempty(client.train) && return NaN
    return mean(norm(vec(s.theta_true)) for s in client.train)
end

function compute_warcraft_feasible_radius(instance::WarcraftInstance)
    return _warcraft_projection_feasible_radius(instance)
end

function compute_warcraft_per_client_lambda0(
    client_data::AbstractVector{WarcraftClientData};
    fallback::Float64=1.0,
)
    return [let
        c_norm = compute_warcraft_client_c_norm(c)
        radius = compute_warcraft_feasible_radius(c.reference_instance)
        (isnan(c_norm) || radius <= 0) ? fallback : max(c_norm / radius, 1e-6)
    end for c in client_data]
end

function _warcraft_initial_client_lambda0s(
    objective::WarcraftObjective,
    client_data::AbstractVector{WarcraftClientData},
    config::WarcraftTrainingConfig,
)
    if config.per_client_lambda && objective_uses_lambda(objective)
        return compute_warcraft_per_client_lambda0(client_data; fallback=config.lambda0)
    end
    return fill(config.lambda0, length(client_data))
end

function _warcraft_initial_client_lr0s(
    objective::WarcraftObjective,
    client_data::AbstractVector{WarcraftClientData},
    config::WarcraftTrainingConfig,
    client_lambda0s::AbstractVector{<:Real},
)
    length(client_lambda0s) == length(client_data) || throw(
        ArgumentError("client_lambda0s length must match client_data length"),
    )
    if config.per_client_lr && objective_uses_lambda(objective)
        return [config.lr_lambda_alpha * client_lambda0s[i] / 4 for i in eachindex(client_data)]
    end
    return fill(config.lr0, length(client_data))
end

function _prepare_client_training_state(
    ::WarcraftRSPOPlusObjective,
    client::WarcraftClientData,
    config::WarcraftTrainingConfig,
    state,
    lambda::Real,
)
    tp = time()
    local_cache =
        isnothing(state) ? init_warcraft_projection_cache(client.reference_instance; lambda=lambda) :
        state
    v_opt_train = [
        projection_optimizer(
            sample.theta_true;
            instance=sample.instance,
            cache=local_cache,
            lambda=lambda,
            use_warm_start=config.use_warm_start,
        ) for sample in client.train
    ]
    println("      projection precompute ($(length(client.train)) samples): $(round(time() - tp; digits=2))s")
    return (cache=local_cache, v_opt_train=v_opt_train)
end

function _prepare_client_training_state(
    objective::WarcraftDiffOptObjective,
    client::WarcraftClientData,
    config::WarcraftTrainingConfig,
    state,
    lambda::Real,
)
    return isnothing(state) ? init_warcraft_diffopt_layer(client.reference_instance; tau=objective.tau) : state
end

function _persist_client_training_state(::WarcraftRSPOPlusObjective, state)
    return state.cache
end

_persist_client_training_state(::WarcraftDiffOptObjective, state) = state

_prepare_client_training_state(::WarcraftObjective, client, config, state, lambda) = state
_persist_client_training_state(::WarcraftObjective, state) = state

# ── loss layer builders ──────────────────────────────────────────────────────

function _build_warcraft_loss_layer(
    ::WarcraftRSPOPlusObjective,
    client::WarcraftClientData;
    lambda::Real,
    config::WarcraftTrainingConfig,
    state=nothing,
)
    return RSPOPlusLoss(projection_optimizer; lambda=lambda)
end

function _build_warcraft_loss_layer(
    objective::WarcraftInferOptObjective,
    client::WarcraftClientData;
    lambda::Real,
    config::WarcraftTrainingConfig,
    state=nothing,
)
    return objective.loss_builder(client)
end

function _build_warcraft_loss_layer(
    objective::WarcraftPerturbedMSEObjective,
    client::WarcraftClientData;
    lambda::Real,
    config::WarcraftTrainingConfig,
    state=nothing,
)
    return objective.layer_builder(client)
end

function _build_warcraft_loss_layer(
    objective::WarcraftDiffOptObjective,
    client::WarcraftClientData;
    lambda::Real,
    config::WarcraftTrainingConfig,
    state=nothing,
)
    return isnothing(state) ? init_warcraft_diffopt_layer(client.reference_instance; tau=objective.tau) : state
end

function _build_warcraft_loss_layer(
    ::WarcraftMSEObjective,
    client::WarcraftClientData;
    lambda::Real,
    config::WarcraftTrainingConfig,
    state=nothing,
)
    return nothing
end

# ── per-batch training loss dispatch ─────────────────────────────────────────

function _warcraft_batch_training_loss(
    ::WarcraftRSPOPlusObjective,
    loss_layer,
    predictions,
    batch_indices,
    client::WarcraftClientData,
    state,
    config::WarcraftTrainingConfig;
    lambda::Real,
)
    total = zero(eltype(predictions))
    for (batch_pos, sample_idx) in enumerate(batch_indices)
        sample = client.train[sample_idx]
        prediction = view(predictions, :, :, batch_pos)
        total += loss_layer(
            prediction,
            sample.theta_true;
            instance=sample.instance,
            cache=state.cache,
            lambda=lambda,
            use_warm_start=config.use_warm_start,
            v_opt=state.v_opt_train[sample_idx],
        )
    end
    return total / length(batch_indices)
end

function _warcraft_batch_training_loss(
    objective::WarcraftInferOptObjective,
    loss_layer,
    predictions,
    batch_indices,
    client::WarcraftClientData,
    state,
    config::WarcraftTrainingConfig;
    lambda::Real,
)
    total = zero(eltype(predictions))
    for (batch_pos, sample_idx) in enumerate(batch_indices)
        sample = client.train[sample_idx]
        prediction = view(predictions, :, :, batch_pos)
        total += loss_layer(prediction, objective.target_getter(sample))
    end
    return total / length(batch_indices)
end

function _warcraft_batch_training_loss(
    ::WarcraftMSEObjective,
    loss_layer,
    predictions,
    batch_indices,
    client::WarcraftClientData,
    state,
    config::WarcraftTrainingConfig;
    lambda::Real,
)
    total = zero(eltype(predictions))
    for (batch_pos, sample_idx) in enumerate(batch_indices)
        sample = client.train[sample_idx]
        prediction = view(predictions, :, :, batch_pos)
        total += sum(abs2, prediction .- Float32.(sample.theta_true))
    end
    return total / (length(batch_indices) * length(client.train[first(batch_indices)].theta_true))
end

function _warcraft_batch_training_loss(
    ::WarcraftPerturbedMSEObjective,
    loss_layer,
    predictions,
    batch_indices,
    client::WarcraftClientData,
    state,
    config::WarcraftTrainingConfig;
    lambda::Real,
)
    total = zero(eltype(predictions))
    for (batch_pos, sample_idx) in enumerate(batch_indices)
        sample = client.train[sample_idx]
        prediction = view(predictions, :, :, batch_pos)
        y_hat = loss_layer(prediction)
        total += sum(abs2, y_hat .- sample.y_true)
    end
    return total / (length(batch_indices) * length(client.train[first(batch_indices)].y_true))
end

function _warcraft_batch_training_loss(
    ::WarcraftDiffOptObjective,
    loss_layer,
    predictions,
    batch_indices,
    client::WarcraftClientData,
    state,
    config::WarcraftTrainingConfig;
    lambda::Real,
)
    total = zero(eltype(predictions))
    for (batch_pos, sample_idx) in enumerate(batch_indices)
        sample = client.train[sample_idx]
        prediction = view(predictions, :, :, batch_pos)
        y_hat = loss_layer(prediction)
        total += sum(abs2, y_hat .- sample.y_true)
    end
    return total / (length(batch_indices) * length(client.train[first(batch_indices)].y_true))
end

function _compute_objective_bound(
    ::WarcraftObjective,
    model,
    monitor::WarcraftValidationMonitor,
    config::WarcraftTrainingConfig;
    kwargs...,
)
    return NaN
end

function _compute_objective_bound(
    ::WarcraftRSPOPlusObjective,
    model,
    monitor::WarcraftValidationMonitor,
    config::WarcraftTrainingConfig;
    lambda::Union{Nothing,Real}=nothing,
    lambdas::Union{Nothing,AbstractVector{<:Real}}=nothing,
)
    if !isnothing(lambda) && !isnothing(lambdas)
        throw(ArgumentError("pass either `lambda` or `lambdas`, not both"))
    elseif !isnothing(lambdas)
        return compute_warcraft_freeze_bound(
            model,
            monitor;
            lambdas=lambdas,
            freeze_eps=config.freeze_eps,
            use_warm_start=config.use_warm_start,
        )
    elseif !isnothing(lambda)
        return compute_warcraft_freeze_bound(
            model,
            monitor;
            lambda=lambda,
            freeze_eps=config.freeze_eps,
            use_warm_start=config.use_warm_start,
        )
    end
    throw(UndefKeywordError(:lambda))
end

function train_warcraft_client_model!(
    model,
    opt_state,
    client::WarcraftClientData,
    objective::WarcraftObjective,
    config::WarcraftTrainingConfig,
    rng::AbstractRNG;
    state=nothing,
    lambda::Real=config.lambda0,
)
    n_samples = length(client.train)
    n_samples == 0 && return 0.0, state

    objective_state = _prepare_client_training_state(objective, client, config, state, lambda)
    loss_layer = _build_warcraft_loss_layer(
        objective,
        client;
        lambda=lambda,
        config=config,
        state=objective_state,
    )
    total_loss = 0.0
    total_samples = 0
    n_batches = ceil(Int, n_samples / config.batch_size)

    for epoch in 1:config.local_epochs
        t_epoch = time()
        batch_num = 0
        for batch_indices in _batch_index_sets(
            rng,
            n_samples,
            config.batch_size;
            shuffle=config.shuffle_batches,
        )
            batch_num += 1
            t_batch = time()
            x_batch = _batch_images(client.train, batch_indices)

            batch_loss, grads = Flux.withgradient(model) do m
                predictions = max.(m(x_batch), zero(Float32))
                _warcraft_batch_training_loss(
                    objective,
                    loss_layer,
                    predictions,
                    batch_indices,
                    client,
                    objective_state,
                    config;
                    lambda=lambda,
                )
            end
            Flux.update!(opt_state, model, grads[1])

            total_loss += float(batch_loss) * length(batch_indices)
            total_samples += length(batch_indices)
            println("        batch $(batch_num)/$(n_batches): loss=$(round(float(batch_loss); sigdigits=4)) [$(round(time() - t_batch; digits=2))s]")
        end
        avg_loss = total_samples > 0 ? total_loss / total_samples : NaN
        println("      epoch $(epoch)/$(config.local_epochs): avg_loss=$(round(avg_loss; sigdigits=4)) [$(round(time() - t_epoch; digits=1))s]")
    end

    return total_loss / total_samples, _persist_client_training_state(objective, objective_state)
end

function fed_warcraft(
    dataset::WarcraftDataset;
    objective::WarcraftObjective=WarcraftRSPOPlusObjective(),
    config::WarcraftTrainingConfig=WarcraftTrainingConfig(),
    model=nothing,
)
    _validate_warcraft_training_config(config)

    rng = MersenneTwister(config.seed)
    model_rng = MersenneTwister(config.seed)
    client_data = prepare_warcraft_client_data(dataset)
    validation_monitor =
        objective_tracks_bound(objective) ? build_warcraft_validation_monitor(
            client_data,
            config;
            rng=rng,
        ) : WarcraftValidationMonitor(WarcraftValidationClientData[])

    global_model =
        isnothing(model) ? build_warcraft_model(dataset; pretrain=config.pretrain_backbone, rng=model_rng) :
        Flux.f32(deepcopy(model))
    flat_global, rebuild = Flux.destructure(global_model)

    frozen = Ref(false)
    client_lambda0s = _warcraft_initial_client_lambda0s(objective, client_data, config)
    lambda_scheds = [
        objective_uses_lambda(objective) ?
        create_inverse_time_scheduler(client_lambda0s[i], config.kappa_lambda; frozen=frozen) :
        nothing for
        i in eachindex(client_data)
    ]
    client_lr0s = _warcraft_initial_client_lr0s(objective, client_data, config, client_lambda0s)
    lr_scheds = [
        create_inverse_time_scheduler(client_lr0s[i], config.kappa_lr; frozen=frozen)
        for i in eachindex(client_data)
    ]

    round_losses = Float64[]
    selected_clients = Vector{Vector{Int}}()
    lambda_values = Float64[]
    lr_values = Float64[]
    bound_values = Float64[]
    objective_states = [
        _initial_client_objective_state(objective, client, config) for client in client_data
    ]
    frozen_after_round = nothing

    n_params = length(flat_global)
    println("[fed_warcraft] $(objective_name(objective)) | $(config.rounds) rounds, $(length(client_data)) clients, $(n_params) params")

    for round in 1:config.rounds
        t_round = time()
        lambdas = [
            isnothing(lambda_scheds[i]) ? NaN : next_schedule_value!(lambda_scheds[i]) for
            i in eachindex(client_data)
        ]
        lrs = [next_schedule_value!(lr_scheds[i]) for i in eachindex(lr_scheds)]
        push!(lambda_values, mean(lambdas))
        push!(lr_values, mean(lrs))

        candidate_ids = [client.client_id for client in client_data if !isempty(client.train)]
        active_client_ids = _sample_subset(rng, candidate_ids, config.client_fraction)
        push!(selected_clients, active_client_ids)
        println("  round $(round)/$(config.rounds): mean_lr=$(Base.round(mean(lrs); sigdigits=3)), mean_lambda=$(Base.round(mean(lambdas); sigdigits=3)), clients=$(active_client_ids)")
        client_deltas = Vector{Vector{Float32}}()
        client_sizes = Int[]
        client_losses = Float64[]

        for (ci, client_id) in enumerate(active_client_ids)
            t_client = time()
            println("    client $(client_id) ($(ci)/$(length(active_client_ids))): $(length(client_data[client_id].train)) train samples, lr=$(Base.round(lrs[client_id]; sigdigits=3)), lambda=$(Base.round(lambdas[client_id]; sigdigits=3))")
            local_model = rebuild(copy(flat_global))
            opt_state = Flux.setup(_optimizer_rule(config, lrs[client_id]), local_model)
            round_loss, objective_states[client_id] = train_warcraft_client_model!(
                local_model,
                opt_state,
                client_data[client_id],
                objective,
                config,
                rng;
                state=objective_states[client_id],
                lambda=lambdas[client_id],
            )
            flat_local, _ = Flux.destructure(local_model)
            push!(client_deltas, flat_local .- flat_global)
            push!(client_sizes, length(client_data[client_id].train))
            push!(client_losses, round_loss)
            println("    client $(client_id) done: loss=$(Base.round(round_loss; sigdigits=4)) [$(Base.round(time() - t_client; digits=1))s]")
        end

        if isempty(client_deltas)
            push!(round_losses, NaN)
        else
            weights = client_sizes ./ sum(client_sizes)
            average_delta = zeros(Float32, length(flat_global))
            for (delta, weight) in zip(client_deltas, weights)
                average_delta .+= weight .* delta
            end
            flat_global .+= average_delta
            push!(round_losses, sum(weights .* client_losses))
        end

        global_model = rebuild(copy(flat_global))
        t_bound = time()
        bound = _compute_objective_bound(objective, global_model, validation_monitor, config; lambdas=lambdas)
        push!(bound_values, bound)
        if objective_tracks_bound(objective) && !frozen[] && bound <= config.freeze_tau
            frozen[] = true
            frozen_after_round = round
        end
        println("  round $(round)/$(config.rounds) done: avg_loss=$(Base.round(last(round_losses); sigdigits=4)), bound=$(Base.round(bound; sigdigits=4)), frozen=$(frozen[]) [$(Base.round(time() - t_round; digits=1))s]")
    end

    pcl = config.per_client_lambda && objective_uses_lambda(objective) ? client_lambda0s : nothing
    return WarcraftFedTrainingResult(
        objective_method(objective, :fed),
        global_model,
        client_data,
        round_losses,
        selected_clients,
        SchedulerTrace(lambda_values, lr_values, bound_values, frozen_after_round, pcl),
    )
end

function local_warcraft(
    dataset::WarcraftDataset;
    objective::WarcraftObjective=WarcraftRSPOPlusObjective(),
    config::WarcraftTrainingConfig=WarcraftTrainingConfig(),
    model=nothing,
)
    _validate_warcraft_training_config(config)

    rng = MersenneTwister(config.seed)
    model_rng = MersenneTwister(config.seed)
    client_data = prepare_warcraft_client_data(dataset)
    base_model =
        isnothing(model) ? build_warcraft_model(dataset; pretrain=config.pretrain_backbone, rng=model_rng) :
        Flux.f32(deepcopy(model))
    flat_init, rebuild = Flux.destructure(base_model)

    local_models = [rebuild(copy(flat_init)) for _ in eachindex(client_data)]
    client_lambda0s = _warcraft_initial_client_lambda0s(objective, client_data, config)
    client_lr0s = _warcraft_initial_client_lr0s(objective, client_data, config, client_lambda0s)
    opt_states = [
        Flux.setup(_optimizer_rule(config, client_lr0s[idx]), local_models[idx]) for
        idx in eachindex(client_data)
    ]
    objective_states = [
        _initial_client_objective_state(objective, client, config) for client in client_data
    ]
    validation_monitors = [
        objective_tracks_bound(objective) ? build_warcraft_validation_monitor(
            client_data,
            config;
            rng=rng,
            client_ids=[client_id],
        ) : WarcraftValidationMonitor(WarcraftValidationClientData[]) for
        client_id in eachindex(client_data)
    ]

    frozen_flags = [Ref(false) for _ in eachindex(client_data)]
    lambda_scheds = [
        objective_uses_lambda(objective) ?
        create_inverse_time_scheduler(client_lambda0s[idx], config.kappa_lambda; frozen=frozen_flags[idx]) :
        nothing for
        idx in eachindex(client_data)
    ]
    lr_scheds = [
        create_inverse_time_scheduler(client_lr0s[idx], config.kappa_lr; frozen=frozen_flags[idx]) for
        idx in eachindex(client_data)
    ]

    round_losses = [Float64[] for _ in eachindex(client_data)]
    lambda_values = [Float64[] for _ in eachindex(client_data)]
    lr_values = [Float64[] for _ in eachindex(client_data)]
    bound_values = [Float64[] for _ in eachindex(client_data)]
    frozen_after_round = Vector{Union{Nothing,Int}}(fill(nothing, length(client_data)))

    println("[local_warcraft] $(objective_name(objective)) | $(config.rounds) rounds, $(length(client_data)) clients")

    for round in 1:config.rounds
        t_round = time()
        n_active = 0
        for client_id in eachindex(client_data)
            freeze_round = frozen_after_round[client_id]
            if objective_tracks_bound(objective) &&
               !isnothing(freeze_round) &&
               round > freeze_round + config.stop_after_freeze_rounds
                continue
            end
            n_active += 1

            lambda = isnothing(lambda_scheds[client_id]) ? NaN :
                next_schedule_value!(lambda_scheds[client_id])
            lr = next_schedule_value!(lr_scheds[client_id])
            push!(lambda_values[client_id], lambda)
            push!(lr_values[client_id], lr)

            t_client = time()
            println("    client $(client_id)/$(length(client_data)): lr=$(Base.round(lr; sigdigits=3)), lambda=$(Base.round(lambda; sigdigits=3))")
            Optimisers.adjust!(opt_states[client_id], lr)
            round_loss, objective_states[client_id] = train_warcraft_client_model!(
                local_models[client_id],
                opt_states[client_id],
                client_data[client_id],
                objective,
                config,
                rng;
                state=objective_states[client_id],
                lambda=lambda,
            )
            push!(round_losses[client_id], round_loss)

            bound = _compute_objective_bound(
                objective,
                local_models[client_id],
                validation_monitors[client_id],
                config;
                lambda=lambda,
            )
            push!(bound_values[client_id], bound)

            if objective_tracks_bound(objective) &&
               !frozen_flags[client_id][] &&
               bound <= config.freeze_tau
                frozen_flags[client_id][] = true
                frozen_after_round[client_id] = round
            end
            println("    client $(client_id) done: loss=$(Base.round(round_loss; sigdigits=4)), bound=$(Base.round(bound; sigdigits=4)) [$(Base.round(time() - t_client; digits=1))s]")
        end
        println("  round $(round)/$(config.rounds): $(n_active) active clients [$(Base.round(time() - t_round; digits=1))s]")
    end

    pcl = config.per_client_lambda && objective_uses_lambda(objective) ? client_lambda0s : nothing
    return WarcraftLocalTrainingResult(
        objective_method(objective, :local),
        [
            WarcraftLocalClientTrainingResult(
                client_id,
                local_models[client_id],
                round_losses[client_id],
                SchedulerTrace(
                    lambda_values[client_id],
                    lr_values[client_id],
                    bound_values[client_id],
                    frozen_after_round[client_id],
                    isnothing(pcl) ? nothing : [pcl[client_id]],
                ),
            ) for client_id in eachindex(client_data)
        ],
    )
end

function centralized_warcraft(
    dataset::WarcraftDataset;
    objective::Union{Nothing,WarcraftObjective}=nothing,
    config::WarcraftTrainingConfig=WarcraftTrainingConfig(),
    model=nothing,
)
    resolved_objective = isnothing(objective) ? WarcraftSPOPlusObjective() : objective
    objective_uses_lambda(resolved_objective) && throw(
        ArgumentError(
            "centralized training is only implemented for objectives without lambda schedules; got $(objective_name(resolved_objective))",
        ),
    )
    _validate_warcraft_training_config(config)

    rng = MersenneTwister(config.seed)
    model_rng = MersenneTwister(config.seed)
    client_data = prepare_centralized_warcraft_client_data(dataset)
    base_model =
        isnothing(model) ? build_warcraft_model(
            dataset;
            pretrain=config.pretrain_backbone,
            rng=model_rng,
        ) : Flux.f32(deepcopy(model))
    flat_init, rebuild = Flux.destructure(base_model)

    local_models = [rebuild(copy(flat_init)) for _ in eachindex(client_data)]
    opt_states = [
        Flux.setup(_optimizer_rule(config, config.lr0), local_models[client_id]) for
        client_id in eachindex(client_data)
    ]
    lr_scheds = [
        create_inverse_time_scheduler(config.lr0, config.kappa_lr) for _ in eachindex(client_data)
    ]
    objective_states = [
        _initial_client_objective_state(resolved_objective, client, config) for client in client_data
    ]
    round_losses = [Float64[] for _ in eachindex(client_data)]
    lambda_values = [Float64[] for _ in eachindex(client_data)]
    lr_values = [Float64[] for _ in eachindex(client_data)]
    bound_values = [Float64[] for _ in eachindex(client_data)]

    println(
        "[centralized_warcraft] $(objective_name(resolved_objective)) | $(config.rounds) epochs, $(length(client_data)) clients",
    )

    for round in 1:config.rounds
        t_round = time()
        for client_id in eachindex(client_data)
            lr = next_schedule_value!(lr_scheds[client_id])
            client = client_data[client_id]
            push!(lambda_values[client_id], NaN)
            push!(lr_values[client_id], lr)
            push!(bound_values[client_id], NaN)
            println(
                "  epoch $(round)/$(config.rounds): client $(client_id) ($(length(client.train)) train samples), lr=$(Base.round(lr; sigdigits=3))",
            )
            Optimisers.adjust!(opt_states[client_id], lr)
            round_loss, objective_states[client_id] = train_warcraft_client_model!(
                local_models[client_id],
                opt_states[client_id],
                client,
                resolved_objective,
                config,
                rng;
                state=objective_states[client_id],
                lambda=NaN,
            )
            push!(round_losses[client_id], round_loss)
        end

        epoch_loss = mean(last(values) for values in round_losses if !isempty(values))
        println(
            "  epoch $(round)/$(config.rounds) done: avg_loss=$(Base.round(epoch_loss; sigdigits=4)) [$(Base.round(time() - t_round; digits=1))s]",
        )
    end

    return WarcraftLocalTrainingResult(
        objective_method(resolved_objective, :centralized),
        [
            WarcraftLocalClientTrainingResult(
                client_id,
                local_models[client_id],
                round_losses[client_id],
                SchedulerTrace(
                    lambda_values[client_id],
                    lr_values[client_id],
                    bound_values[client_id],
                    nothing,
                    nothing,
                ),
            ) for client_id in eachindex(client_data)
        ],
    )
end

function fed_rspo_plus_warcraft(
    dataset::WarcraftDataset;
    config::WarcraftTrainingConfig=WarcraftTrainingConfig(),
    model=nothing,
)
    return fed_warcraft(
        dataset;
        objective=WarcraftRSPOPlusObjective(),
        config=config,
        model=model,
    )
end

function local_rspo_plus_warcraft(
    dataset::WarcraftDataset;
    config::WarcraftTrainingConfig=WarcraftTrainingConfig(),
    model=nothing,
)
    return local_warcraft(
        dataset;
        objective=WarcraftRSPOPlusObjective(),
        config=config,
        model=model,
    )
end

function run_warcraft_two_stage(
    dataset::WarcraftDataset;
    objective::WarcraftObjective=WarcraftRSPOPlusObjective(),
    warm_start_config::WarcraftTrainingConfig=WarcraftTrainingConfig(),
    personalization_config::WarcraftTrainingConfig=WarcraftTrainingConfig(),
    model=nothing,
)
    t_total = time()
    println("=== TWO-STAGE WARCRAFT: $(objective_name(objective)) ===")
    println("--- Stage 1: Federated warm start ($(warm_start_config.rounds) rounds) ---")
    warm_start = fed_warcraft(
        dataset;
        objective=objective,
        config=warm_start_config,
        model=model,
    )
    println("--- Stage 1 complete [$(round(time() - t_total; digits=1))s elapsed] ---")
    t_stage2 = time()
    println("--- Stage 2: Local personalization ($(personalization_config.rounds) rounds) ---")
    personalization = local_warcraft(
        dataset;
        objective=objective,
        config=personalization_config,
        model=warm_start.model,
    )
    println("--- Stage 2 complete [$(round(time() - t_stage2; digits=1))s] ---")
    println("=== TWO-STAGE COMPLETE [$(round(time() - t_total; digits=1))s total] ===")
    return WarcraftTwoStageTrainingResult(warm_start, personalization)
end

function run_rspo_plus_warcraft_experiment(
    dataset::WarcraftDataset;
    warm_start_config::WarcraftTrainingConfig=WarcraftTrainingConfig(),
    personalization_config::WarcraftTrainingConfig=WarcraftTrainingConfig(),
    model=nothing,
)
    return run_warcraft_two_stage(
        dataset;
        objective=WarcraftRSPOPlusObjective(),
        warm_start_config=warm_start_config,
        personalization_config=personalization_config,
        model=model,
    )
end

# ── SPO+ objective factory ──────────────────────────────────────────────────

function WarcraftSPOPlusObjective(; α::Real=2.0)
    alpha = float(α)
    return WarcraftInferOptObjective(
        :spo_plus,
        client -> InferOpt.SPOPlusLoss(warcraft_linear_maximizer(; instance=client.reference_instance); α=alpha),
        sample -> sample.theta_true,
    )
end

function _warcraft_perturbed_layer_builder(;
    nb_samples::Integer=10,
    epsilon::Real=1.0,
    threaded::Bool=false,
    seed=nothing,
)
    nb_samples > 0 || throw(ArgumentError("nb_samples must be positive, got $nb_samples"))
    isfinite(epsilon) && epsilon > 0 ||
        throw(ArgumentError("epsilon must be positive and finite, got $epsilon"))
    nb_samples_int = Int(nb_samples)
    epsilon_float = float(epsilon)
    return client -> InferOpt.PerturbedAdditive(
        warcraft_linear_maximizer(; instance=client.reference_instance);
        ε=epsilon_float,
        nb_samples=nb_samples_int,
        threaded=threaded,
        seed=seed,
    )
end

function WarcraftPerturbedFYLObjective(;
    nb_samples::Integer=10,
    epsilon::Real=1.0,
    threaded::Bool=false,
    seed=nothing,
)
    perturbed_builder = _warcraft_perturbed_layer_builder(
        ;
        nb_samples=nb_samples,
        epsilon=epsilon,
        threaded=threaded,
        seed=seed,
    )
    return WarcraftInferOptObjective(
        :perturbed_fyl_mult,
        client -> InferOpt.FenchelYoungLoss(perturbed_builder(client)),
        sample -> sample.y_true,
    )
end

function fed_spo_plus_warcraft(
    dataset::WarcraftDataset;
    config::WarcraftTrainingConfig=WarcraftTrainingConfig(),
    model=nothing,
    α::Real=2.0,
)
    return fed_warcraft(dataset; objective=WarcraftSPOPlusObjective(; α=α), config=config, model=model)
end

function local_spo_plus_warcraft(
    dataset::WarcraftDataset;
    config::WarcraftTrainingConfig=WarcraftTrainingConfig(),
    model=nothing,
    α::Real=2.0,
)
    return local_warcraft(dataset; objective=WarcraftSPOPlusObjective(; α=α), config=config, model=model)
end

function centralized_spo_plus_warcraft(
    dataset::WarcraftDataset;
    config::WarcraftTrainingConfig=WarcraftTrainingConfig(),
    model=nothing,
    α::Real=2.0,
)
    return centralized_warcraft(
        dataset;
        objective=WarcraftSPOPlusObjective(; α=α),
        config=config,
        model=model,
    )
end

function fed_perturbed_fyl_warcraft(
    dataset::WarcraftDataset;
    config::WarcraftTrainingConfig=WarcraftTrainingConfig(),
    model=nothing,
    nb_samples::Integer=10,
    epsilon::Real=1.0,
    threaded::Bool=false,
    seed=nothing,
)
    return fed_warcraft(
        dataset;
        objective=WarcraftPerturbedFYLObjective(
            ;
            nb_samples=nb_samples,
            epsilon=epsilon,
            threaded=threaded,
            seed=seed,
        ),
        config=config,
        model=model,
    )
end

function local_perturbed_fyl_warcraft(
    dataset::WarcraftDataset;
    config::WarcraftTrainingConfig=WarcraftTrainingConfig(),
    model=nothing,
    nb_samples::Integer=10,
    epsilon::Real=1.0,
    threaded::Bool=false,
    seed=nothing,
)
    return local_warcraft(
        dataset;
        objective=WarcraftPerturbedFYLObjective(
            ;
            nb_samples=nb_samples,
            epsilon=epsilon,
            threaded=threaded,
            seed=seed,
        ),
        config=config,
        model=model,
    )
end

# ── Perturbed decision-space MSE objective factory ──────────────────────────

function WarcraftPerturbedMSEObjective(;
    nb_samples::Integer=10,
    epsilon::Real=1.0,
    threaded::Bool=false,
    seed=nothing,
)
    return WarcraftPerturbedMSEObjective(
        :dpo_perturbed_mse_mult,
        _warcraft_perturbed_layer_builder(
            ;
            nb_samples=nb_samples,
            epsilon=epsilon,
            threaded=threaded,
            seed=seed,
        ),
    )
end

function fed_perturbed_mse_warcraft(
    dataset::WarcraftDataset;
    config::WarcraftTrainingConfig=WarcraftTrainingConfig(),
    model=nothing,
    nb_samples::Integer=10,
    epsilon::Real=1.0,
    threaded::Bool=false,
    seed=nothing,
)
    return fed_warcraft(
        dataset;
        objective=WarcraftPerturbedMSEObjective(
            ;
            nb_samples=nb_samples,
            epsilon=epsilon,
            threaded=threaded,
            seed=seed,
        ),
        config=config,
        model=model,
    )
end

function local_perturbed_mse_warcraft(
    dataset::WarcraftDataset;
    config::WarcraftTrainingConfig=WarcraftTrainingConfig(),
    model=nothing,
    nb_samples::Integer=10,
    epsilon::Real=1.0,
    threaded::Bool=false,
    seed=nothing,
)
    return local_warcraft(
        dataset;
        objective=WarcraftPerturbedMSEObjective(
            ;
            nb_samples=nb_samples,
            epsilon=epsilon,
            threaded=threaded,
            seed=seed,
        ),
        config=config,
        model=model,
    )
end

function WarcraftDiffOptObjective(; tau::Real=1e-3)
    isfinite(tau) && tau > 0 ||
        throw(ArgumentError("tau must be positive and finite, got $tau"))
    return WarcraftDiffOptObjective(float(tau))
end

function fed_diffopt_warcraft(
    dataset::WarcraftDataset;
    config::WarcraftTrainingConfig=WarcraftTrainingConfig(),
    model=nothing,
    tau::Real=1e-3,
)
    return fed_warcraft(
        dataset;
        objective=WarcraftDiffOptObjective(; tau=tau),
        config=config,
        model=model,
    )
end

function local_diffopt_warcraft(
    dataset::WarcraftDataset;
    config::WarcraftTrainingConfig=WarcraftTrainingConfig(),
    model=nothing,
    tau::Real=1e-3,
)
    return local_warcraft(
        dataset;
        objective=WarcraftDiffOptObjective(; tau=tau),
        config=config,
        model=model,
    )
end

# ── MSE objective factory ────────────────────────────────────────────────────

function fed_mse_warcraft(
    dataset::WarcraftDataset;
    config::WarcraftTrainingConfig=WarcraftTrainingConfig(),
    model=nothing,
)
    return fed_warcraft(dataset; objective=WarcraftMSEObjective(), config=config, model=model)
end

function local_mse_warcraft(
    dataset::WarcraftDataset;
    config::WarcraftTrainingConfig=WarcraftTrainingConfig(),
    model=nothing,
)
    return local_warcraft(dataset; objective=WarcraftMSEObjective(), config=config, model=model)
end

# ── objective lookup by name ─────────────────────────────────────────────────

function warcraft_objective_from_name(
    name::Symbol;
    spo_alpha::Real=2.0,
    perturbed_nb_samples::Integer=10,
    perturbed_epsilon::Real=1.0,
    perturbed_threaded::Bool=false,
    perturbed_seed=nothing,
    diffopt_tau::Real=1e-3,
)
    if name === :rspo_plus
        return WarcraftRSPOPlusObjective()
    elseif name === :spo_plus
        return WarcraftSPOPlusObjective(; α=spo_alpha)
    elseif name === :perturbed_fyl || name === :perturbed_fyl_mult
        return WarcraftPerturbedFYLObjective(
            ;
            nb_samples=perturbed_nb_samples,
            epsilon=perturbed_epsilon,
            threaded=perturbed_threaded,
            seed=perturbed_seed,
        )
    elseif name === :perturbed_mse || name === :dpo_perturbed_mse_mult
        return WarcraftPerturbedMSEObjective(
            ;
            nb_samples=perturbed_nb_samples,
            epsilon=perturbed_epsilon,
            threaded=perturbed_threaded,
            seed=perturbed_seed,
        )
    elseif name === :diffopt
        return WarcraftDiffOptObjective(; tau=diffopt_tau)
    elseif name === :mse
        return WarcraftMSEObjective()
    end
    throw(
        ArgumentError(
            "unknown warcraft objective `$name` (expected rspo_plus, spo_plus, perturbed_fyl, perturbed_mse, diffopt, or mse)",
        ),
    )
end
