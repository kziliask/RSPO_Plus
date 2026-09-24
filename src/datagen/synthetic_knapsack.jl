using Distributions
using LinearAlgebra
using Random
using Statistics

Base.@kwdef struct SyntheticKnapsackConfig
    seed::Int = 42
    n_clients::Int = 20
    p::Int = 8
    dim::Int = 50
    deg::Int = 2
    epsilon_noise::Float64 = 0.0
    eta_obj::Float64 = 0.0
    eta_constr::Float64 = 0.0
    eta_constr_affects_weights::Bool = false
    eta_data_dist::Float64 = 0.0
    data_imbalance::Bool = false
    n_train_total::Int = 5500
    n_val_total::Int = 1000
    n_test_total::Int = 20_000
    capacity_ratio::Float64 = 0.6
    obj_low::Float64 = 0.5
    obj_high::Float64 = 1.5
    weight_low::Float64 = 0.5
    weight_high::Float64 = 1.5
end

struct SyntheticKnapsackClient
    client_id::Int
    objective_basis::Matrix{Float64}
    weights::Vector{Float64}
    capacity::Float64
    feature_mean::Vector{Float64}
    feature_scale::Vector{Float64}
end

struct SyntheticKnapsackSplit
    x::Matrix{Float64}
    theta_true::Matrix{Float64}
    client_ids::Vector{Int}
    sample_ids::Vector{Int}
end

struct SyntheticKnapsackMetadata
    objective_pairwise_mean::Float64
    capacity_pairwise_mean::Float64
    feature_mean_pairwise_mean::Float64
    train_counts::Vector{Int}
    val_counts::Vector{Int}
    test_counts::Vector{Int}
end

struct SyntheticKnapsackDataset
    config::SyntheticKnapsackConfig
    clients::Vector{SyntheticKnapsackClient}
    true_weight_matrix::Matrix{Float64}
    train::SyntheticKnapsackSplit
    val::SyntheticKnapsackSplit
    test::SyntheticKnapsackSplit
    metadata::SyntheticKnapsackMetadata
end

struct FeatureNormalizationStats{T<:AbstractFloat,V<:AbstractVector{T}}
    mean::V
    std::V
end

const _RNG_OFFSETS = (
    ground_truth = 1_000,
    problem_base = 2_000,
    objective = 3_000,
    constraint = 4_000,
    constraint_weights = 4_500,
    data_shift = 5_000,
    split_train = 10_000,
    split_val = 20_000,
    split_test = 30_000,
)

function _component_rng(seed::Int, offset::Int)
    return MersenneTwister(seed + offset)
end

function _split_rng(seed::Int, split::Symbol, client_id::Int)
    split_offset =
        if split === :train
            _RNG_OFFSETS.split_train
        elseif split === :val
            _RNG_OFFSETS.split_val
        elseif split === :test
            _RNG_OFFSETS.split_test
        else
            throw(ArgumentError("unsupported split $split"))
        end
    return _component_rng(seed, split_offset + 1_000 * client_id)
end

function _uniform_vec(rng::AbstractRNG, n::Int, low::Real, high::Real)
    return low .+ (high - low) .* rand(rng, n)
end

function _random_rotation_matrix(rng::AbstractRNG, dim::Int, scale::Float64)
    dim <= 1 && return Matrix{Float64}(I, dim, dim)
    scale == 0.0 && return Matrix{Float64}(I, dim, dim)

    raw = randn(rng, dim, dim)
    generator = raw .- raw'
    generator_norm = norm(generator)
    generator_norm == 0.0 && return Matrix{Float64}(I, dim, dim)

    return exp(scale .* generator ./ generator_norm)
end

function _rotate_objective_basis(
    rng::AbstractRNG,
    objective_basis::AbstractMatrix{<:Real},
    eta_obj::Float64,
)
    eta_obj == 0.0 && return Matrix{Float64}(objective_basis)
    rotation = _random_rotation_matrix(rng, size(objective_basis, 2), eta_obj)
    return Matrix{Float64}(objective_basis * rotation)
end

"""
    _beta_multiplicative_factors(rng, dims, c; min_factor=1e-9)

Sample a multiplicative perturbation matrix from Beta(1/c, 1/c) scaled to [0, 2].
The mean factor is always 1 (perturbation is unbiased relative to the base value).
Spread grows with c: c<1 gives a bell curve, c=1 gives Uniform[0,2], c>1 inverts
to a U-shaped (bimodal at 0 and 2). c=0 short-circuits to all-ones (no perturbation).
"""
function _beta_multiplicative_factors(
    rng::AbstractRNG,
    dims::NTuple{N,Int} where {N},
    c::Float64;
    min_factor::Float64 = 1e-9,
)
    c == 0.0 && return ones(dims...)
    α = 1.0 / c
    u = rand(rng, Beta(α, α), dims...)
    return max.(2.0 .* u, min_factor)
end

function _validate_config(config::SyntheticKnapsackConfig)
    config.n_clients > 0 || throw(ArgumentError("n_clients must be positive"))
    config.p > 0 || throw(ArgumentError("p must be positive"))
    config.dim > 0 || throw(ArgumentError("dim must be positive"))
    config.deg > 0 || throw(ArgumentError("deg must be positive"))
    config.n_train_total >= 0 || throw(ArgumentError("n_train_total must be nonnegative"))
    config.n_val_total >= 0 || throw(ArgumentError("n_val_total must be nonnegative"))
    config.n_test_total >= 0 || throw(ArgumentError("n_test_total must be nonnegative"))
    config.capacity_ratio > 0 || throw(ArgumentError("capacity_ratio must be positive"))
    config.obj_low < config.obj_high ||
        throw(ArgumentError("obj_low must be smaller than obj_high"))
    config.weight_low > 0 || throw(ArgumentError("weight_low must be positive"))
    config.weight_low < config.weight_high ||
        throw(ArgumentError("weight_low must be smaller than weight_high"))
    config.epsilon_noise >= 0 || throw(ArgumentError("epsilon_noise must be nonnegative"))
    return config
end

function _validate_constraint_parameters(
    weights::AbstractVector{<:Real},
    capacity::Real;
    client_id::Int,
)
    all(>(0.0), weights) ||
        throw(ArgumentError("client $client_id has a nonpositive item weight"))
    capacity > 0 || throw(ArgumentError("client $client_id has a nonpositive capacity"))
    return nothing
end

function _generate_client_constraint_parameters(
    config::SyntheticKnapsackConfig,
    weights_base::AbstractVector{<:Real},
    capacity_base::Real,
    capacity_factor::Real;
    weight_factor::Union{Nothing,AbstractVector{<:Real}}=nothing,
    client_id::Int,
)
    weights =
        if config.eta_constr_affects_weights
            isnothing(weight_factor) &&
                throw(ArgumentError("weight factor must be provided when enabled"))
            weights_base .* weight_factor
        else
            weights_base
        end
    capacity = capacity_base * capacity_factor
    _validate_constraint_parameters(weights, capacity; client_id=client_id)
    return collect(weights), capacity
end

function generate_ground_truth_matrix(config::SyntheticKnapsackConfig)
    _validate_config(config)
    rng = _component_rng(config.seed, _RNG_OFFSETS.ground_truth)
    basis = Float64.(rand(rng, config.dim, config.p) .< 0.5)
    row_scales = _uniform_vec(rng, config.dim, config.obj_low, config.obj_high)
    return basis .* row_scales
end

function generate_synthetic_knapsack_clients(
    config::SyntheticKnapsackConfig,
    true_weight_matrix::AbstractMatrix{<:Real},
)
    _validate_config(config)

    base_rng = _component_rng(config.seed, _RNG_OFFSETS.problem_base)
    obj_rng = _component_rng(config.seed, _RNG_OFFSETS.objective)
    constr_rng = _component_rng(config.seed, _RNG_OFFSETS.constraint)
    weight_rng = _component_rng(config.seed, _RNG_OFFSETS.constraint_weights)
    data_rng = _component_rng(config.seed, _RNG_OFFSETS.data_shift)

    weights_base = _uniform_vec(base_rng, config.dim, config.weight_low, config.weight_high)
    capacity_base = config.capacity_ratio * config.dim

    capacity_factors = _beta_multiplicative_factors(
        constr_rng, (config.n_clients,), config.eta_constr
    )
    weight_factors =
        config.eta_constr_affects_weights ?
        _beta_multiplicative_factors(
            weight_rng, (config.dim, config.n_clients), config.eta_constr
        ) : nothing
    feature_mean_noise = randn(data_rng, config.p, config.n_clients)
    feature_scale_factors = _beta_multiplicative_factors(
        data_rng, (config.p, config.n_clients), config.eta_data_dist
    )

    clients = Vector{SyntheticKnapsackClient}(undef, config.n_clients)
    for client_id in 1:config.n_clients
        objective_basis = _rotate_objective_basis(obj_rng, true_weight_matrix, config.eta_obj)
        weights, capacity = _generate_client_constraint_parameters(
            config,
            weights_base,
            capacity_base,
            capacity_factors[client_id];
            weight_factor=
                isnothing(weight_factors) ? nothing : @view(weight_factors[:, client_id]),
            client_id=client_id,
        )
        feature_mean = config.eta_data_dist .* feature_mean_noise[:, client_id]
        feature_scale = feature_scale_factors[:, client_id]
        clients[client_id] = SyntheticKnapsackClient(
            client_id,
            objective_basis,
            weights,
            capacity,
            collect(feature_mean),
            collect(feature_scale),
        )
    end

    return clients
end

function allocate_client_counts(
    total::Int;
    n_clients::Int,
    data_imbalance::Bool=false,
)
    total >= 0 || throw(ArgumentError("total must be nonnegative"))
    n_clients > 0 || throw(ArgumentError("n_clients must be positive"))

    if !data_imbalance
        counts = fill(div(total, n_clients), n_clients)
        for i in 1:rem(total, n_clients)
            counts[i] += 1
        end
        return counts
    end

    n_large = fld(n_clients, 2)
    n_small = n_clients - n_large
    weights = vcat(fill(10.0, n_large), fill(1.0, n_small))
    raw_counts = total .* weights ./ sum(weights)
    counts = floor.(Int, raw_counts)

    remainder = total - sum(counts)
    order = sortperm(raw_counts .- counts; rev=true)
    for idx in order[1:remainder]
        counts[idx] += 1
    end
    return counts
end

function generate_true_objectives(
    rng::AbstractRNG,
    x::Matrix{Float64},
    objective_basis::AbstractMatrix{<:Real},
    config::SyntheticKnapsackConfig,
)
    activations = objective_basis * x ./ sqrt(config.p)
    signal = -(1 .+ (1 .+ activations) .^ config.deg)

    if config.epsilon_noise == 0.0
        return signal ./ mean(abs.(signal))
    end

    low = 1 - config.epsilon_noise
    high = 1 + config.epsilon_noise
    # `rand(rng, size(signal))` samples from the size tuple itself, so splat dims here.
    noise = low .+ (high - low) .* rand(rng, size(signal)...)
    return (signal .* noise) ./ mean(abs.(signal .* noise))
end

function generate_synthetic_knapsack_split(
    config::SyntheticKnapsackConfig,
    clients::Vector{SyntheticKnapsackClient},
    split::Symbol,
    total::Int,
)
    counts = allocate_client_counts(
        total;
        n_clients=config.n_clients,
        data_imbalance=config.data_imbalance,
    )

    x = Matrix{Float64}(undef, config.p, total)
    theta_true = Matrix{Float64}(undef, config.dim, total)
    client_ids = Vector{Int}(undef, total)
    sample_ids = collect(1:total)

    cursor = 1
    for client in clients
        count = counts[client.client_id]
        count == 0 && continue

        rng = _split_rng(config.seed, split, client.client_id)
        x_client =
            client.feature_mean .+ client.feature_scale .* randn(rng, config.p, count)
        theta_client = generate_true_objectives(rng, x_client, client.objective_basis, config)

        next_cursor = cursor + count - 1
        x[:, cursor:next_cursor] .= x_client
        theta_true[:, cursor:next_cursor] .= theta_client
        client_ids[cursor:next_cursor] .= client.client_id
        cursor = next_cursor + 1
    end

    return SyntheticKnapsackSplit(x, theta_true, client_ids, sample_ids)
end

function _mean_pairwise_vector_distance(vectors::Vector{Vector{Float64}})
    n = length(vectors)
    n <= 1 && return 0.0

    total = 0.0
    pairs = 0
    for i in 1:(n - 1), j in (i + 1):n
        total += norm(vectors[i] .- vectors[j])
        pairs += 1
    end
    return total / pairs
end

function _mean_pairwise_scalar_distance(values::Vector{Float64})
    n = length(values)
    n <= 1 && return 0.0

    total = 0.0
    pairs = 0
    for i in 1:(n - 1), j in (i + 1):n
        total += abs(values[i] - values[j])
        pairs += 1
    end
    return total / pairs
end

function _mean_pairwise_matrix_distance(matrices::AbstractVector{<:AbstractMatrix{<:Real}})
    n = length(matrices)
    n <= 1 && return 0.0

    total = 0.0
    pairs = 0
    for i in 1:(n - 1), j in (i + 1):n
        total += norm(matrices[i] .- matrices[j])
        pairs += 1
    end
    return total / pairs
end

count_samples_per_client(split::SyntheticKnapsackSplit, n_clients::Int) = [
    count(==(client_id), split.client_ids) for client_id in 1:n_clients
]

function client_split_indices(split::SyntheticKnapsackSplit, client_id::Int)
    return findall(==(client_id), split.client_ids)
end

function fit_feature_normalization_stats(x::AbstractMatrix{<:Real})
    mean_vec = vec(mean(x; dims=2))
    std_vec = vec(std(x; dims=2) .+ 1e-8)
    return FeatureNormalizationStats(mean_vec, std_vec)
end

function apply_feature_normalization(
    x::AbstractMatrix{<:Real},
    stats::FeatureNormalizationStats,
)
    return (x .- stats.mean) ./ stats.std
end

function fit_client_normalization_stats(
    split::SyntheticKnapsackSplit,
    client_id::Int,
)
    indices = client_split_indices(split, client_id)
    if isempty(indices)
        feature_dim = size(split.x, 1)
        return FeatureNormalizationStats(zeros(feature_dim), ones(feature_dim))
    end
    return fit_feature_normalization_stats(split.x[:, indices])
end

function fit_client_normalization_stats(
    split::SyntheticKnapsackSplit,
    client_ids::AbstractVector{Int},
)
    return [fit_client_normalization_stats(split, client_id) for client_id in client_ids]
end

function fit_all_client_normalization_stats(
    split::SyntheticKnapsackSplit,
    n_clients::Int,
)
    return fit_client_normalization_stats(split, collect(1:n_clients))
end

function summarize_synthetic_knapsack_dataset(
    clients::Vector{SyntheticKnapsackClient},
    train::SyntheticKnapsackSplit,
    val::SyntheticKnapsackSplit,
    test::SyntheticKnapsackSplit,
)
    objective_pairwise_mean =
        _mean_pairwise_matrix_distance([client.objective_basis for client in clients])
    capacity_pairwise_mean =
        _mean_pairwise_scalar_distance([client.capacity for client in clients])
    feature_mean_pairwise_mean =
        _mean_pairwise_vector_distance([client.feature_mean for client in clients])

    train_counts = count_samples_per_client(train, length(clients))
    val_counts = count_samples_per_client(val, length(clients))
    test_counts = count_samples_per_client(test, length(clients))

    return SyntheticKnapsackMetadata(
        objective_pairwise_mean,
        capacity_pairwise_mean,
        feature_mean_pairwise_mean,
        train_counts,
        val_counts,
        test_counts,
    )
end

function generate_synthetic_knapsack_dataset(config::SyntheticKnapsackConfig)
    _validate_config(config)

    true_weight_matrix = generate_ground_truth_matrix(config)
    clients = generate_synthetic_knapsack_clients(config, true_weight_matrix)
    train = generate_synthetic_knapsack_split(config, clients, :train, config.n_train_total)
    val = generate_synthetic_knapsack_split(config, clients, :val, config.n_val_total)
    test = generate_synthetic_knapsack_split(config, clients, :test, config.n_test_total)
    metadata = summarize_synthetic_knapsack_dataset(clients, train, val, test)

    return SyntheticKnapsackDataset(config, clients, true_weight_matrix, train, val, test, metadata)
end

function get_client_split_data(
    split::SyntheticKnapsackSplit,
    clients::Vector{SyntheticKnapsackClient},
    client_id::Int;
    normalize_x::Bool=true,
    normalization_stats::Union{Nothing,FeatureNormalizationStats}=nothing,
)
    indices = client_split_indices(split, client_id)
    client = clients[client_id]
    x = split.x[:, indices]
    theta_true = split.theta_true[:, indices]

    if normalize_x && !isempty(indices)
        stats =
            isnothing(normalization_stats) ? fit_feature_normalization_stats(x) :
            normalization_stats
        x = apply_feature_normalization(x, stats)
    end

    return client.weights, client.capacity, x, theta_true
end

function get_client_split_data(
    dataset::SyntheticKnapsackDataset,
    split_name::Symbol,
    client_id::Int;
    normalize_x::Bool=true,
    normalization_split::Union{Nothing,Symbol}=:train,
)
    split = getproperty(dataset, split_name)
    stats =
        if normalize_x && !isnothing(normalization_split)
            reference_split = getproperty(dataset, normalization_split)
            fit_client_normalization_stats(reference_split, client_id)
        else
            nothing
        end
    return get_client_split_data(
        split,
        dataset.clients,
        client_id;
        normalize_x=normalize_x,
        normalization_stats=stats,
    )
end

function _config_slug(config::SyntheticKnapsackConfig)
    return join(
        [
            "seed$(config.seed)",
            "obj$(config.eta_obj)",
            "constr$(config.eta_constr)",
            "constrweights$(config.eta_constr_affects_weights)",
            "data$(config.eta_data_dist)",
            "imbalance$(config.data_imbalance)",
        ],
        "_",
    )
end

function _split_header(config::SyntheticKnapsackConfig)
    return vcat(
        ["sample_id", "client_id"],
        ["x$i" for i in 1:config.p],
        ["cost_prod$i" for i in 1:config.dim],
    )
end

function _client_header(config::SyntheticKnapsackConfig)
    return vcat(
        ["client_id", "capacity"],
        ["weights$i" for i in 1:config.dim],
        ["feature_mean$i" for i in 1:config.p],
        ["feature_scale$i" for i in 1:config.p],
    )
end

function _write_csv_rows(path::AbstractString, header::Vector{<:AbstractString}, rows)
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(row, ","))
        end
    end
    return path
end

function export_synthetic_knapsack_split(
    split::SyntheticKnapsackSplit,
    config::SyntheticKnapsackConfig,
    path::AbstractString,
)
    header = _split_header(config)
    rows = (
        vcat(
            string.((
                split.sample_ids[idx],
                split.client_ids[idx],
            )),
            string.(split.x[:, idx]),
            string.(split.theta_true[:, idx]),
        ) for idx in eachindex(split.sample_ids)
    )
    return _write_csv_rows(path, header, rows)
end

function export_synthetic_knapsack_clients(
    clients::Vector{SyntheticKnapsackClient},
    config::SyntheticKnapsackConfig,
    path::AbstractString,
)
    header = _client_header(config)
    rows = (
        vcat(
            string.((client.client_id, client.capacity)),
            string.(client.weights),
            string.(client.feature_mean),
            string.(client.feature_scale),
        ) for client in clients
    )
    return _write_csv_rows(path, header, rows)
end

function export_synthetic_knapsack_metadata(
    metadata::SyntheticKnapsackMetadata,
    path::AbstractString,
)
    rows = [
        ("objective_pairwise_mean", metadata.objective_pairwise_mean),
        ("capacity_pairwise_mean", metadata.capacity_pairwise_mean),
        ("feature_mean_pairwise_mean", metadata.feature_mean_pairwise_mean),
    ]
    return _write_csv_rows(path, ["metric", "value"], (string.(row) for row in rows))
end

function export_synthetic_knapsack_sample_counts(
    metadata::SyntheticKnapsackMetadata,
    path::AbstractString,
)
    rows = String[]
    open(path, "w") do io
        println(io, "split,client_id,count")
        for (split_name, counts) in (
            ("train", metadata.train_counts),
            ("val", metadata.val_counts),
            ("test", metadata.test_counts),
        )
            for (client_id, count_value) in enumerate(counts)
                println(io, string(split_name, ",", client_id, ",", count_value))
            end
        end
    end
    return path
end

function export_synthetic_knapsack_dataset(
    dataset::SyntheticKnapsackDataset;
    root::AbstractString=joinpath("src", "data", "synthetic_knapsack"),
    prefix::Union{Nothing,AbstractString}=nothing,
)
    export_dir = joinpath(root, something(prefix, _config_slug(dataset.config)))
    mkpath(export_dir)

    paths = (
        train=export_synthetic_knapsack_split(
            dataset.train,
            dataset.config,
            joinpath(export_dir, "train.csv"),
        ),
        val=export_synthetic_knapsack_split(
            dataset.val,
            dataset.config,
            joinpath(export_dir, "val.csv"),
        ),
        test=export_synthetic_knapsack_split(
            dataset.test,
            dataset.config,
            joinpath(export_dir, "test.csv"),
        ),
        clients=export_synthetic_knapsack_clients(
            dataset.clients,
            dataset.config,
            joinpath(export_dir, "clients.csv"),
        ),
        metadata=export_synthetic_knapsack_metadata(
            dataset.metadata,
            joinpath(export_dir, "metadata.csv"),
        ),
        sample_counts=export_synthetic_knapsack_sample_counts(
            dataset.metadata,
            joinpath(export_dir, "sample_counts.csv"),
        ),
    )

    return (; dir=export_dir, paths...)
end
