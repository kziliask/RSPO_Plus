using CSV
using DataFrames
using Dates
using Statistics

const PJM_FEATURE_COLUMNS = [
    "temp_c_mean",
    "temp_c_min",
    "temp_c_max",
    "humidity_mean",
    "wind_speed_mean",
    "solar_irradiance_mean",
    "solar_irradiance_max",
    "day_of_week",
]
const PJM_TARGET_COLUMNS = ["c_$(hour)" for hour in 0:23]
const PJM_TOTAL_LMP_COLUMNS = ["total_lmp_$(hour)" for hour in 0:23]
const PJM_CONGESTION_COLUMNS = ["congestion_$(hour)" for hour in 0:23]

Base.@kwdef struct PJMBatteryDatasetConfig
    data_root::String = joinpath("src", "data", "exp3")
    raw_filename::String = "pjm_data.csv"
    combined_filename::String = "pjm_data_combined.csv"
    panel_filename::String = "pjm_fl_panel.csv"
    split_fractions::NTuple{3,Float64} = (0.5, 0.1, 0.4)
    fixed_demand::Union{Nothing,Float64} = nothing
end

struct PJMBatteryClient
    client_id::Int
    client_name::String
    capacity::Float64
end

struct PJMBatterySplit
    name::Symbol
    x::Matrix{Float64}
    theta_true::Matrix{Float64}
    client_ids::Vector{Int}
    client_names::Vector{String}
    capacities::Vector{Float64}
    sample_ids::Vector{Int}
    date_indices::Vector{Int}
    dates::Vector{Date}
end

struct PJMBatteryClientManifest
    client_id::Int
    client_name::String
    capacity::Float64
    train_sample_ids::Vector{Int}
    val_sample_ids::Vector{Int}
    test_sample_ids::Vector{Int}
end

struct PJMBatteryPartitionManifest
    train_date_indices::Vector{Int}
    val_date_indices::Vector{Int}
    test_date_indices::Vector{Int}
    clients::Vector{PJMBatteryClientManifest}
end

struct PJMBatteryDatasetMetadata
    raw_path::String
    combined_path::String
    panel_path::String
    n_clients::Int
    n_dates::Int
    n_samples::Int
    feature_names::Vector{String}
    target_names::Vector{String}
    train_date_indices::Vector{Int}
    val_date_indices::Vector{Int}
    test_date_indices::Vector{Int}
    train_counts::Vector{Int}
    val_counts::Vector{Int}
    test_counts::Vector{Int}
    missing_dates::Vector{Date}
end

struct PJMBatteryDataset
    config::PJMBatteryDatasetConfig
    panel::DataFrame
    clients::Vector{PJMBatteryClient}
    train::PJMBatterySplit
    val::PJMBatterySplit
    test::PJMBatterySplit
    partition_manifest::PJMBatteryPartitionManifest
    metadata::PJMBatteryDatasetMetadata
end

function _validate_pjm_dataset_config(config::PJMBatteryDatasetConfig)
    all(fraction -> fraction > 0, config.split_fractions) || throw(
        ArgumentError("all split fractions must be positive, got $(config.split_fractions)"),
    )
    abs(sum(config.split_fractions) - 1.0) <= 1e-8 || throw(
        ArgumentError("split fractions must sum to 1, got $(config.split_fractions)"),
    )
    if !isnothing(config.fixed_demand)
        0.0 <= config.fixed_demand <= 24.0 || throw(
            ArgumentError(
                "fixed demand must be feasible for the 24-hour unit-box dispatch, got $(config.fixed_demand)",
            ),
        )
    end
    return config
end

_pjm_raw_path(config::PJMBatteryDatasetConfig) = joinpath(config.data_root, config.raw_filename)
_pjm_combined_path(config::PJMBatteryDatasetConfig) =
    joinpath(config.data_root, config.combined_filename)
_pjm_panel_path(config::PJMBatteryDatasetConfig) = joinpath(config.data_root, config.panel_filename)

function _require_pjm_file(path::AbstractString)
    isfile(path) || throw(ArgumentError("expected PJM dataset file at `$path`"))
    return path
end

function _pjm_required_columns()
    return vcat(
        ["client_id", "date", "bess", "client_name"],
        PJM_TOTAL_LMP_COLUMNS,
        PJM_CONGESTION_COLUMNS,
        PJM_FEATURE_COLUMNS,
    )
end

function _validate_pjm_frame_columns(frame::DataFrame)
    missing = [column for column in _pjm_required_columns() if !(column in names(frame))]
    isempty(missing) || throw(ArgumentError("PJM data is missing required columns $(missing)"))
    return frame
end

function _sorted_unique_dates(dates::AbstractVector{Date})
    return sort(unique(dates))
end

function _pjm_missing_dates(sorted_dates::Vector{Date})
    missing_dates = Date[]
    for (left, right) in zip(sorted_dates, sorted_dates[2:end])
        cursor = left + Day(1)
        while cursor < right
            push!(missing_dates, cursor)
            cursor += Day(1)
        end
    end
    return missing_dates
end

function _pjm_escape_csv(value)
    text = string(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function _write_pjm_csv_rows(
    path::AbstractString,
    header::AbstractVector{<:AbstractString},
    rows,
)
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join((_pjm_escape_csv(value) for value in row), ","))
        end
    end
    return path
end

function _pjm_numeric_matrix(frame::DataFrame, columns::Vector{String})
    matrix = Matrix{Float64}(frame[:, columns])
    return permutedims(matrix)
end

function _make_pjm_split(name::Symbol, frame::DataFrame)
    return PJMBatterySplit(
        name,
        _pjm_numeric_matrix(frame, PJM_FEATURE_COLUMNS),
        _pjm_numeric_matrix(frame, PJM_TARGET_COLUMNS),
        Int.(frame.client_id),
        String.(frame.client_name),
        Float64.(frame.bess),
        Int.(frame.sample_id),
        Int.(frame.date_index),
        Date.(frame.date),
    )
end

function _pjm_client_counts(split::PJMBatterySplit, n_clients::Int)
    return [count(==(client_id), split.client_ids) for client_id in 1:n_clients]
end

function _pjm_split_counts(total::Int, fractions::NTuple{3,Float64})
    train_count = floor(Int, total * fractions[1])
    val_count = floor(Int, total * fractions[2])
    test_count = total - train_count - val_count
    min(train_count, val_count, test_count) > 0 || throw(
        ArgumentError("split fractions $(fractions) produce an empty split for $total dates"),
    )
    return train_count, val_count, test_count
end

function generate_pjm_derived_csvs(
    config::PJMBatteryDatasetConfig=PJMBatteryDatasetConfig();
    force::Bool=true,
)
    _validate_pjm_dataset_config(config)
    raw_path = _require_pjm_file(_pjm_raw_path(config))
    combined_path = _pjm_combined_path(config)
    panel_path = _pjm_panel_path(config)

    if !force && isfile(combined_path) && isfile(panel_path)
        return (; raw_path, combined_path, panel_path)
    end

    raw = CSV.read(raw_path, DataFrame)
    _validate_pjm_frame_columns(raw)
    raw.date = Date.(raw.date)
    sort!(raw, [:date, :client_id])

    for hour in 0:23
        combined_column = Symbol("c_$(hour)")
        raw[!, combined_column] =
            Float64.(raw[!, Symbol("total_lmp_$(hour)")]) .+
            Float64.(raw[!, Symbol("congestion_$(hour)")])
    end

    CSV.write(combined_path, raw)

    unique_dates = _sorted_unique_dates(raw.date)
    date_to_index = Dict(date => idx for (idx, date) in enumerate(unique_dates))
    panel = DataFrame(
        sample_id=collect(1:nrow(raw)),
        date_index=[date_to_index[date] for date in raw.date],
        client_id=Int.(raw.client_id),
        client_name=String.(raw.client_name),
        date=raw.date,
        bess=Float64.(raw.bess),
    )
    for column in PJM_FEATURE_COLUMNS
        panel[!, Symbol(column)] = Float64.(raw[!, Symbol(column)])
    end
    for column in PJM_TARGET_COLUMNS
        panel[!, Symbol(column)] = Float64.(raw[!, Symbol(column)])
    end

    CSV.write(panel_path, panel)
    return (; raw_path, combined_path, panel_path)
end

function ensure_pjm_derived_csvs(
    config::PJMBatteryDatasetConfig=PJMBatteryDatasetConfig();
    force::Bool=false,
)
    if force || !isfile(_pjm_combined_path(config)) || !isfile(_pjm_panel_path(config))
        return generate_pjm_derived_csvs(config; force=true)
    end
    return (
        raw_path=_pjm_raw_path(config),
        combined_path=_pjm_combined_path(config),
        panel_path=_pjm_panel_path(config),
    )
end

function _build_pjm_clients(panel::DataFrame)
    grouped = groupby(panel, :client_id; sort=true)
    clients = Vector{PJMBatteryClient}(undef, length(grouped))
    for (idx, group) in enumerate(grouped)
        capacities = unique(Float64.(group.bess))
        length(capacities) == 1 || throw(
            ArgumentError("client $(first(group.client_id)) has multiple bess values $(capacities)"),
        )
        names = unique(String.(group.client_name))
        length(names) == 1 || throw(
            ArgumentError("client $(first(group.client_id)) has multiple names $(names)"),
        )
        clients[idx] = PJMBatteryClient(first(group.client_id), only(names), only(capacities))
    end
    return clients
end

function _build_pjm_partition_manifest(
    panel::DataFrame,
    clients::Vector{PJMBatteryClient},
    train_dates::Vector{Int},
    val_dates::Vector{Int},
    test_dates::Vector{Int},
)
    train_set = Set(train_dates)
    val_set = Set(val_dates)
    test_set = Set(test_dates)
    manifests = Vector{PJMBatteryClientManifest}(undef, length(clients))

    for (idx, client) in enumerate(clients)
        client_rows = panel[panel.client_id .== client.client_id, :]
        manifests[idx] = PJMBatteryClientManifest(
            client.client_id,
            client.client_name,
            client.capacity,
            Int.(client_rows.sample_id[in.(client_rows.date_index, Ref(train_set))]),
            Int.(client_rows.sample_id[in.(client_rows.date_index, Ref(val_set))]),
            Int.(client_rows.sample_id[in.(client_rows.date_index, Ref(test_set))]),
        )
    end

    return PJMBatteryPartitionManifest(train_dates, val_dates, test_dates, manifests)
end

function _build_pjm_metadata(
    config::PJMBatteryDatasetConfig,
    raw_path::String,
    combined_path::String,
    panel_path::String,
    clients::Vector{PJMBatteryClient},
    panel::DataFrame,
    train::PJMBatterySplit,
    val::PJMBatterySplit,
    test::PJMBatterySplit,
    unique_dates::Vector{Date},
)
    return PJMBatteryDatasetMetadata(
        raw_path,
        combined_path,
        panel_path,
        length(clients),
        length(unique_dates),
        nrow(panel),
        copy(PJM_FEATURE_COLUMNS),
        copy(PJM_TARGET_COLUMNS),
        sort(unique(train.date_indices)),
        sort(unique(val.date_indices)),
        sort(unique(test.date_indices)),
        _pjm_client_counts(train, length(clients)),
        _pjm_client_counts(val, length(clients)),
        _pjm_client_counts(test, length(clients)),
        _pjm_missing_dates(unique_dates),
    )
end

function load_pjm_battery_dataset(
    config::PJMBatteryDatasetConfig=PJMBatteryDatasetConfig();
    regenerate::Bool=false,
)
    _validate_pjm_dataset_config(config)
    paths = ensure_pjm_derived_csvs(config; force=regenerate)
    panel = CSV.read(_require_pjm_file(paths.panel_path), DataFrame)
    panel.date = Date.(panel.date)
    if !isnothing(config.fixed_demand)
        panel.bess = fill(config.fixed_demand, nrow(panel))
    end
    sort!(panel, [:date_index, :client_id])

    unique_dates = _sorted_unique_dates(Date.(panel.date))
    train_date_count, val_date_count, test_date_count = _pjm_split_counts(
        length(unique_dates),
        config.split_fractions,
    )

    train_dates = collect(1:train_date_count)
    val_dates = collect((train_date_count + 1):(train_date_count + val_date_count))
    test_dates = collect((train_date_count + val_date_count + 1):length(unique_dates))

    train = _make_pjm_split(
        :train,
        panel[in.(panel.date_index, Ref(Set(train_dates))), :],
    )
    val = _make_pjm_split(
        :val,
        panel[in.(panel.date_index, Ref(Set(val_dates))), :],
    )
    test = _make_pjm_split(
        :test,
        panel[in.(panel.date_index, Ref(Set(test_dates))), :],
    )

    clients = _build_pjm_clients(panel)
    partition_manifest = _build_pjm_partition_manifest(panel, clients, train_dates, val_dates, test_dates)
    metadata = _build_pjm_metadata(
        config,
        paths.raw_path,
        paths.combined_path,
        paths.panel_path,
        clients,
        panel,
        train,
        val,
        test,
        unique_dates,
    )

    return PJMBatteryDataset(config, panel, clients, train, val, test, partition_manifest, metadata)
end

function _pjm_sample_lookup(dataset::PJMBatteryDataset)
    return Dict(
        Int(row.sample_id) => (
            client_id=Int(row.client_id),
            client_name=String(row.client_name),
            capacity=Float64(row.bess),
            date_index=Int(row.date_index),
            date=Date(row.date),
        ) for row in eachrow(dataset.panel)
    )
end

function export_pjm_dataset_metadata(
    metadata::PJMBatteryDatasetMetadata,
    path::AbstractString,
)
    rows = vcat(
        [
            ("raw_path", metadata.raw_path),
            ("combined_path", metadata.combined_path),
            ("panel_path", metadata.panel_path),
            ("n_clients", metadata.n_clients),
            ("n_dates", metadata.n_dates),
            ("n_samples", metadata.n_samples),
            ("train_date_count", length(metadata.train_date_indices)),
            ("val_date_count", length(metadata.val_date_indices)),
            ("test_date_count", length(metadata.test_date_indices)),
            ("train_start_date_index", first(metadata.train_date_indices)),
            ("train_end_date_index", last(metadata.train_date_indices)),
            ("val_start_date_index", first(metadata.val_date_indices)),
            ("val_end_date_index", last(metadata.val_date_indices)),
            ("test_start_date_index", first(metadata.test_date_indices)),
            ("test_end_date_index", last(metadata.test_date_indices)),
        ],
        [("feature_$(idx)", metadata.feature_names[idx]) for idx in eachindex(metadata.feature_names)],
        [("target_$(idx)", metadata.target_names[idx]) for idx in eachindex(metadata.target_names)],
        [("train_count_client_$(idx)", metadata.train_counts[idx]) for idx in eachindex(metadata.train_counts)],
        [("val_count_client_$(idx)", metadata.val_counts[idx]) for idx in eachindex(metadata.val_counts)],
        [("test_count_client_$(idx)", metadata.test_counts[idx]) for idx in eachindex(metadata.test_counts)],
        [("missing_date_$(idx)", metadata.missing_dates[idx]) for idx in eachindex(metadata.missing_dates)],
    )
    return _write_pjm_csv_rows(path, ["metric", "value"], (string.(row) for row in rows))
end

function export_pjm_partition_manifest(
    dataset::PJMBatteryDataset,
    path::AbstractString,
)
    sample_lookup = _pjm_sample_lookup(dataset)
    header = ["client_id", "client_name", "capacity", "split", "sample_id", "date_index", "date"]
    rows = (
        let
            sample = sample_lookup[sample_id]
            [
                string(client.client_id),
                client.client_name,
                string(client.capacity),
                split_name,
                string(sample_id),
                string(sample.date_index),
                string(sample.date),
            ]
        end for client in dataset.partition_manifest.clients for
        (split_name, sample_ids) in (
            ("train", client.train_sample_ids),
            ("val", client.val_sample_ids),
            ("test", client.test_sample_ids),
        ) for sample_id in sample_ids
    )
    return _write_pjm_csv_rows(path, header, rows)
end
