using JSON3
using NPZ
using Random
using Statistics

Base.@kwdef struct WarcraftDatasetConfig
    seed::Int = 42
    data_root::String = joinpath("src", "data", "warcraft_shortest_path_oneskin")
    grid_dim::Int = 12
    n_clients::Int = 20
    benchmark_train_fraction::Float64 = 0.8
    client_validation_fraction::Float64 = 0.2
    endpoint_heterogeneity::Bool = true
    client_layout::Symbol = :corner_cycle
    objective_heterogeneity::Bool = false
    terrain_heterogeneity_scale::Float64 = 0.0
end

struct WarcraftRawSample{TI<:AbstractArray{Float32,3},TM<:AbstractMatrix{Float32}}
    sample_id::Int
    image::TI
    y_true::TM
    true_weights::TM
end

struct WarcraftSplit{S}
    name::Symbol
    sample_ids::Vector{Int}
    images::Array{Float32,4}
    y_true::Array{Float32,3}
    true_weights::Array{Float32,3}
    samples::Vector{S}
end

struct WarcraftClientManifest
    client_id::Int
    endpoint_kind::Symbol
    source::Int
    sink::Int
    train_indices::Vector{Int}
    val_indices::Vector{Int}
    terrain_multipliers::Vector{Float64}
end

struct WarcraftPartitionManifest
    benchmark_train_sample_ids::Vector{Int}
    benchmark_test_sample_ids::Vector{Int}
    clients::Vector{WarcraftClientManifest}
end

struct WarcraftDatasetMetadata
    data_root::String
    grid_dim::Int
    pixels_per_cell::Int
    terrain_values::Vector{Float32}
    raw_train_count::Int
    raw_val_count::Int
    raw_test_count::Int
    benchmark_train_count::Int
    benchmark_test_count::Int
    val_files_present::Bool
    test_files_present::Bool
    objective_heterogeneity::Bool
    endpoint_heterogeneity::Bool
    terrain_heterogeneity_scale::Float64
end

struct WarcraftDataset
    config::WarcraftDatasetConfig
    benchmark_train::WarcraftSplit
    benchmark_test::WarcraftSplit
    partition_manifest::WarcraftPartitionManifest
    metadata::WarcraftDatasetMetadata
end

function _validate_warcraft_dataset_config(config::WarcraftDatasetConfig)
    config.grid_dim == 12 || throw(
        ArgumentError("v1 Warcraft loader only supports grid_dim=12, got $(config.grid_dim)"),
    )
    config.n_clients > 0 || throw(ArgumentError("n_clients must be positive"))
    config.client_layout in (:corner_cycle, :border_antipodes) || throw(
        ArgumentError(
            "client_layout must be :corner_cycle or :border_antipodes, got $(config.client_layout)",
        ),
    )
    0 < config.benchmark_train_fraction < 1 ||
        throw(ArgumentError("benchmark_train_fraction must lie in (0, 1)"))
    0 <= config.client_validation_fraction < 1 ||
        throw(ArgumentError("client_validation_fraction must lie in [0, 1)"))
    config.terrain_heterogeneity_scale >= 0 ||
        throw(ArgumentError("terrain_heterogeneity_scale must be nonnegative"))
    if config.objective_heterogeneity && config.terrain_heterogeneity_scale <= 0
        throw(
            ArgumentError(
                "terrain_heterogeneity_scale must be strictly positive when objective_heterogeneity=true",
            ),
        )
    end
    if config.client_layout === :border_antipodes
        max_clients = 4 * config.grid_dim - 4
        config.n_clients <= max_clients || throw(
            ArgumentError(
                "border_antipodes supports at most $max_clients clients for grid_dim=$(config.grid_dim), got $(config.n_clients)",
            ),
        )
    end
    return config
end

_warcraft_grid_root(config::WarcraftDatasetConfig) =
    joinpath(config.data_root, string(config.grid_dim, "x", config.grid_dim))

function _require_warcraft_file(path::AbstractString)
    isfile(path) || throw(
        ArgumentError(
            "expected Warcraft dataset file at `$path`. This minimal repo does not ship the Warcraft dataset. Download the external one-skin 12x12 files and place them under `src/data/warcraft_shortest_path_oneskin/12x12/`.",
        ),
    )
    return path
end

function _warcraft_info(config::WarcraftDatasetConfig)
    info_path = _require_warcraft_file(joinpath(_warcraft_grid_root(config), "info.json"))
    return JSON3.read(read(info_path, String))
end

function _load_warcraft_npy(path::AbstractString)
    return NPZ.npzread(_require_warcraft_file(path))
end

function _load_warcraft_train_arrays(config::WarcraftDatasetConfig)
    root = _warcraft_grid_root(config)
    raw_images = _load_warcraft_npy(joinpath(root, "train_maps.npy"))
    raw_paths = _load_warcraft_npy(joinpath(root, "train_shortest_paths.npy"))
    raw_weights = _load_warcraft_npy(joinpath(root, "train_vertex_weights.npy"))

    images = Float32.(permutedims(raw_images, (2, 3, 4, 1))) ./ 255.0f0
    channel_mean = mean(images; dims=(1, 2, 4))
    channel_std = std(images; dims=(1, 2, 4), corrected=false)
    images = (images .- channel_mean) ./ channel_std
    y_true = Float32.(permutedims(raw_paths, (2, 3, 1)))
    true_weights = Float32.(permutedims(raw_weights, (2, 3, 1)))
    return images, y_true, true_weights
end

function _make_warcraft_split(
    name::Symbol,
    sample_ids::Vector{Int},
    images::Array{Float32,4},
    y_true::Array{Float32,3},
    true_weights::Array{Float32,3},
)
    n_samples = length(sample_ids)
    size(images, 4) == n_samples ||
        throw(ArgumentError("image batch size does not match sample_ids length"))
    size(y_true, 3) == n_samples ||
        throw(ArgumentError("path batch size does not match sample_ids length"))
    size(true_weights, 3) == n_samples ||
        throw(ArgumentError("weight batch size does not match sample_ids length"))

    samples = [
        WarcraftRawSample(
            sample_ids[idx],
            view(images, :, :, :, idx),
            view(y_true, :, :, idx),
            view(true_weights, :, :, idx),
        ) for idx in 1:n_samples
    ]
    return WarcraftSplit(name, sample_ids, images, y_true, true_weights, samples)
end

function _subset_warcraft_split(
    name::Symbol,
    sample_ids::AbstractVector{Int},
    images::Array{Float32,4},
    y_true::Array{Float32,3},
    true_weights::Array{Float32,3},
    indices::AbstractVector{Int},
)
    return _make_warcraft_split(
        name,
        collect(sample_ids[indices]),
        images[:, :, :, indices],
        y_true[:, :, indices],
        true_weights[:, :, indices],
    )
end

function _warcraft_coord_to_index(grid_dim::Int, row::Int, col::Int)
    return row + (col - 1) * grid_dim
end

function _endpoint_source_sink(endpoint_kind::Symbol, grid_dim::Int)
    if endpoint_kind === :tl_br
        return 1, grid_dim * grid_dim
    elseif endpoint_kind === :tr_bl
        return (
            _warcraft_coord_to_index(grid_dim, 1, grid_dim),
            _warcraft_coord_to_index(grid_dim, grid_dim, 1),
        )
    elseif endpoint_kind === :bl_tr
        return (
            _warcraft_coord_to_index(grid_dim, grid_dim, 1),
            _warcraft_coord_to_index(grid_dim, 1, grid_dim),
        )
    elseif endpoint_kind === :br_tl
        return grid_dim * grid_dim, 1
    end
    throw(ArgumentError("unsupported endpoint_kind `$endpoint_kind`"))
end

function _endpoint_kind_from_coords(source_row::Int, source_col::Int, sink_row::Int, sink_col::Int)
    if source_row <= sink_row && source_col <= sink_col
        return :tl_br
    elseif source_row <= sink_row && source_col > sink_col
        return :tr_bl
    elseif source_row > sink_row && source_col <= sink_col
        return :bl_tr
    end
    return :br_tl
end

function _endpoint_palette(config::WarcraftDatasetConfig)
    return config.endpoint_heterogeneity ? [:tl_br, :br_tl, :tr_bl, :bl_tr] : [:tl_br]
end

function _clockwise_border_coords(grid_dim::Int)
    coords = Tuple{Int,Int}[]
    append!(coords, ((1, col) for col in 1:grid_dim))
    append!(coords, ((row, grid_dim) for row in 2:grid_dim))
    append!(coords, ((grid_dim, col) for col in (grid_dim - 1):-1:1))
    append!(coords, ((row, 1) for row in (grid_dim - 1):-1:2))
    return coords
end

_antipode_coord(grid_dim::Int, row::Int, col::Int) = (grid_dim - row + 1, grid_dim - col + 1)

function _border_antipode_client_specs(grid_dim::Int)
    specs = NamedTuple{(:endpoint_kind, :source, :sink),Tuple{Symbol,Int,Int}}[]
    for (source_row, source_col) in _clockwise_border_coords(grid_dim)
        sink_row, sink_col = _antipode_coord(grid_dim, source_row, source_col)
        endpoint_kind = _endpoint_kind_from_coords(source_row, source_col, sink_row, sink_col)
        push!(
            specs,
            (
                endpoint_kind=endpoint_kind,
                source=_warcraft_coord_to_index(grid_dim, source_row, source_col),
                sink=_warcraft_coord_to_index(grid_dim, sink_row, sink_col),
            ),
        )
    end
    return specs
end

function _client_endpoint_specs(config::WarcraftDatasetConfig)
    if config.client_layout === :border_antipodes
        return _border_antipode_client_specs(config.grid_dim)[1:config.n_clients]
    end

    endpoint_palette = _endpoint_palette(config)
    return [
        let endpoint_kind = endpoint_palette[mod1(client_id, length(endpoint_palette))]
            source, sink = _endpoint_source_sink(endpoint_kind, config.grid_dim)
            (endpoint_kind=endpoint_kind, source=source, sink=sink)
        end for client_id in 1:config.n_clients
    ]
end

function _terrain_multipliers(
    config::WarcraftDatasetConfig,
    terrain_values::Vector{Float32},
    client_id::Int,
)
    if !config.objective_heterogeneity || config.terrain_heterogeneity_scale == 0
        return ones(Float64, length(terrain_values))
    end

    rng = MersenneTwister(config.seed + 10_000 * client_id)
    return exp.(config.terrain_heterogeneity_scale .* randn(rng, length(terrain_values)))
end

function _client_validation_count(total::Int, validation_fraction::Float64)
    total <= 1 && return 0
    desired = round(Int, validation_fraction * total)
    return clamp(desired, 1, total - 1)
end

function _build_warcraft_partition_manifest(
    config::WarcraftDatasetConfig,
    benchmark_train::WarcraftSplit,
    benchmark_test::WarcraftSplit,
    terrain_values::Vector{Float32},
)
    rng = MersenneTwister(config.seed)
    shuffled_indices = randperm(rng, length(benchmark_train.samples))
    client_buckets = [Int[] for _ in 1:config.n_clients]
    for (position, sample_idx) in enumerate(shuffled_indices)
        client_id = mod1(position, config.n_clients)
        push!(client_buckets[client_id], sample_idx)
    end

    endpoint_specs = _client_endpoint_specs(config)
    manifests = Vector{WarcraftClientManifest}(undef, config.n_clients)
    for client_id in 1:config.n_clients
        assigned = sort(client_buckets[client_id])
        client_rng = MersenneTwister(config.seed + 1_000 * client_id)
        shuffled = isempty(assigned) ? Int[] : assigned[randperm(client_rng, length(assigned))]
        n_val = _client_validation_count(length(shuffled), config.client_validation_fraction)
        val_indices = sort(shuffled[1:n_val])
        train_indices_client = sort(shuffled[(n_val + 1):end])
        endpoint_spec = endpoint_specs[client_id]
        manifests[client_id] = WarcraftClientManifest(
            client_id,
            endpoint_spec.endpoint_kind,
            endpoint_spec.source,
            endpoint_spec.sink,
            train_indices_client,
            val_indices,
            _terrain_multipliers(config, terrain_values, client_id),
        )
    end

    return WarcraftPartitionManifest(
        copy(benchmark_train.sample_ids),
        copy(benchmark_test.sample_ids),
        manifests,
    )
end

function load_warcraft_dataset(config::WarcraftDatasetConfig=WarcraftDatasetConfig())
    _validate_warcraft_dataset_config(config)
    info = _warcraft_info(config)
    images, y_true, true_weights = _load_warcraft_train_arrays(config)
    n_samples = size(images, 4)
    split_at = floor(Int, config.benchmark_train_fraction * n_samples)
    split_at = clamp(split_at, 1, n_samples - 1)
    sample_ids = collect(1:n_samples)
    terrain_values = sort(unique(vec(true_weights)))

    benchmark_train = _subset_warcraft_split(
        :benchmark_train,
        sample_ids,
        images,
        y_true,
        true_weights,
        1:split_at,
    )
    benchmark_test = _subset_warcraft_split(
        :benchmark_test,
        sample_ids,
        images,
        y_true,
        true_weights,
        (split_at + 1):n_samples,
    )

    root = _warcraft_grid_root(config)
    effective_endpoint_heterogeneity =
        config.client_layout === :border_antipodes ? true : config.endpoint_heterogeneity
    metadata = WarcraftDatasetMetadata(
        config.data_root,
        config.grid_dim,
        Int(info.dataset_params.pixels_per_cell),
        terrain_values,
        Int(info.dataset_params.train_set_params.num_examples),
        Int(info.dataset_params.val_set_params.num_examples),
        Int(info.dataset_params.test_set_params.num_examples),
        length(benchmark_train.samples),
        length(benchmark_test.samples),
        all(
            isfile,
            [
                joinpath(root, "val_maps.npy"),
                joinpath(root, "val_shortest_paths.npy"),
                joinpath(root, "val_vertex_weights.npy"),
            ],
        ),
        all(
            isfile,
            [
                joinpath(root, "test_maps.npy"),
                joinpath(root, "test_shortest_paths.npy"),
                joinpath(root, "test_vertex_weights.npy"),
            ],
        ),
        config.objective_heterogeneity,
        effective_endpoint_heterogeneity,
        config.terrain_heterogeneity_scale,
    )

    partition_manifest =
        _build_warcraft_partition_manifest(config, benchmark_train, benchmark_test, terrain_values)

    return WarcraftDataset(config, benchmark_train, benchmark_test, partition_manifest, metadata)
end

function _warcraft_csv_escape(value)
    text = string(value)
    needs_quotes =
        occursin(",", text) ||
        occursin("\"", text) ||
        occursin("\n", text) ||
        occursin("\r", text)
    return needs_quotes ? string("\"", replace(text, "\"" => "\"\""), "\"") : text
end

function _write_warcraft_csv_rows(
    path::AbstractString,
    header::Vector{<:AbstractString},
    rows,
)
    open(path, "w") do io
        println(io, join((_warcraft_csv_escape(value) for value in header), ","))
        for row in rows
            println(io, join((_warcraft_csv_escape(value) for value in row), ","))
        end
    end
    return path
end

function export_warcraft_dataset_metadata(
    metadata::WarcraftDatasetMetadata,
    path::AbstractString,
)
    rows = vcat(
        [
            ("data_root", metadata.data_root),
            ("grid_dim", metadata.grid_dim),
            ("pixels_per_cell", metadata.pixels_per_cell),
            ("raw_train_count", metadata.raw_train_count),
            ("raw_val_count", metadata.raw_val_count),
            ("raw_test_count", metadata.raw_test_count),
            ("benchmark_train_count", metadata.benchmark_train_count),
            ("benchmark_test_count", metadata.benchmark_test_count),
            ("val_files_present", metadata.val_files_present),
            ("test_files_present", metadata.test_files_present),
            ("objective_heterogeneity", metadata.objective_heterogeneity),
            ("endpoint_heterogeneity", metadata.endpoint_heterogeneity),
            ("terrain_heterogeneity_scale", metadata.terrain_heterogeneity_scale),
        ],
        [("terrain_value_$(idx)", metadata.terrain_values[idx]) for idx in eachindex(metadata.terrain_values)],
    )
    return _write_warcraft_csv_rows(path, ["metric", "value"], (string.(row) for row in rows))
end

function export_warcraft_partition_manifest(
    manifest::WarcraftPartitionManifest,
    terrain_values::Vector{Float32},
    path::AbstractString,
)
    header = [
        "client_id",
        "endpoint_kind",
        "source",
        "sink",
        "split",
        "benchmark_train_index",
        "sample_id",
        "terrain_multipliers",
    ]
    open(path, "w") do io
        println(io, join(header, ","))
        for client in manifest.clients
            multipliers = join(
                [
                    string(Float64(terrain_values[idx]), ":", client.terrain_multipliers[idx]) for
                    idx in eachindex(terrain_values)
                ],
                "|",
            )
            for (split_name, indices) in (("train", client.train_indices), ("val", client.val_indices))
                for split_idx in indices
                    sample_id = manifest.benchmark_train_sample_ids[split_idx]
                    println(
                        io,
                        join(
                            [
                                string(client.client_id),
                                string(client.endpoint_kind),
                                string(client.source),
                                string(client.sink),
                                split_name,
                                string(split_idx),
                                string(sample_id),
                                multipliers,
                            ],
                            ",",
                        ),
                    )
                end
            end
        end
    end
    return path
end
