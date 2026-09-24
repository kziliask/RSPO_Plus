using ChainRulesCore
using JuMP
using Gurobi
import DiffOpt
import InferOpt
using LinearAlgebra

const WARCRAFT_BENCHMARK_HARD_SOLVER = :dijkstra

struct WarcraftInstance
    resolved_weights::Matrix{Float64}
    source::Int
    sink::Int
    endpoint_kind::Symbol
    terrain_multipliers::Vector{Float64}
end

function WarcraftInstance(
    resolved_weights::AbstractMatrix{<:Real},
    source::Int,
    sink::Int;
    endpoint_kind::Symbol=:tl_br,
    terrain_multipliers::AbstractVector{<:Real}=Float64[],
)
    size(resolved_weights, 1) == size(resolved_weights, 2) ||
        throw(ArgumentError("Warcraft grids must be square"))
    return WarcraftInstance(
        Matrix{Float64}(resolved_weights),
        source,
        sink,
        endpoint_kind,
        Float64.(terrain_multipliers),
    )
end

mutable struct WarcraftProjectionCache
    jump_model::Model
    x::Vector{VariableRef}
    e::Vector{VariableRef}
    lambda::Float64
    grid_dim::Int
    canonical_source::Int
    canonical_sink::Int
    edges::Vector{Tuple{Int,Int}}
    last_vertex_solution::Vector{Float64}
end

mutable struct WarcraftDiffOptLayer
    model::Model
    e::Vector{VariableRef}
    theta_params::Vector{VariableRef}
    incoming_edges::Vector{Vector{Int}}
    edges::Vector{Tuple{Int,Int}}
    grid_dim::Int
    canonical_source::Int
    canonical_sink::Int
    endpoint_kind::Symbol
    tau::Float64
    last_edge_solution::Vector{Float64}
end

struct WarcraftDiffOptLinearSolver end

function DiffOpt.QuadraticProgram.solve_system(
    ::WarcraftDiffOptLinearSolver,
    LHS,
    RHS,
    iterative::Bool,
)
    return DiffOpt.QuadraticProgram.IterativeSolvers.lsqr(LHS, RHS)
end

function JuMP.MOI.Utilities.map_indices(
    variable_map::AbstractDict{T,T},
    solver::WarcraftDiffOptLinearSolver,
) where {T<:Union{JuMP.MOI.VariableIndex,JuMP.MOI.ConstraintIndex}}
    return solver
end

warcraft_grid_dim(instance::WarcraftInstance) = size(instance.resolved_weights, 1)

function warcraft_index_to_coord(grid_dim::Int, index::Integer)
    1 <= index <= grid_dim * grid_dim ||
        throw(ArgumentError("index $index is out of range for grid_dim=$grid_dim"))
    int_index = Int(index)
    row = mod1(int_index, grid_dim)
    col = fld(int_index - row, grid_dim) + 1
    return row, col
end

function warcraft_coord_to_index(grid_dim::Int, row::Integer, col::Integer)
    1 <= row <= grid_dim || throw(ArgumentError("row $row is out of range for grid_dim=$grid_dim"))
    1 <= col <= grid_dim || throw(ArgumentError("col $col is out of range for grid_dim=$grid_dim"))
    return Int(row) + (Int(col) - 1) * grid_dim
end

function _to_warcraft_matrix(theta::AbstractVector{<:Real}, grid_dim::Int)
    length(theta) == grid_dim * grid_dim ||
        throw(ArgumentError("expected vector of length $(grid_dim * grid_dim), got $(length(theta))"))
    return reshape(Float64.(theta), grid_dim, grid_dim)
end

function _to_warcraft_matrix(theta::AbstractMatrix{<:Real}, grid_dim::Int)
    size(theta) == (grid_dim, grid_dim) ||
        throw(ArgumentError("expected matrix shape ($grid_dim, $grid_dim), got $(size(theta))"))
    return Matrix{Float64}(theta)
end

function _to_warcraft_matrix(theta::AbstractArray{<:Real,3}, grid_dim::Int)
    size(theta, 1) == grid_dim && size(theta, 2) == grid_dim || throw(
        ArgumentError("expected tensor with first dims ($grid_dim, $grid_dim), got $(size(theta))"),
    )
    return Array{Float64,3}(theta)
end

function _to_canonical_orientation(values::AbstractMatrix{<:Real}, endpoint_kind::Symbol)
    if endpoint_kind === :tl_br
        return Matrix{Float64}(values)
    elseif endpoint_kind === :tr_bl
        return reverse(Float64.(values); dims=2)
    elseif endpoint_kind === :bl_tr
        return reverse(Float64.(values); dims=1)
    elseif endpoint_kind === :br_tl
        return reverse(reverse(Float64.(values); dims=1); dims=2)
    end
    throw(ArgumentError("unsupported endpoint_kind `$endpoint_kind`"))
end

function _from_canonical_orientation(values::AbstractMatrix{<:Real}, endpoint_kind::Symbol)
    return _to_canonical_orientation(values, endpoint_kind)
end

function _to_canonical_coord(
    grid_dim::Int,
    row::Int,
    col::Int,
    endpoint_kind::Symbol,
)
    if endpoint_kind === :tl_br
        return row, col
    elseif endpoint_kind === :tr_bl
        return row, grid_dim - col + 1
    elseif endpoint_kind === :bl_tr
        return grid_dim - row + 1, col
    elseif endpoint_kind === :br_tl
        return grid_dim - row + 1, grid_dim - col + 1
    end
    throw(ArgumentError("unsupported endpoint_kind `$endpoint_kind`"))
end

function _canonical_source_sink(instance::WarcraftInstance)
    grid_dim = warcraft_grid_dim(instance)
    source_row, source_col = warcraft_index_to_coord(grid_dim, instance.source)
    sink_row, sink_col = warcraft_index_to_coord(grid_dim, instance.sink)
    canonical_source_row, canonical_source_col =
        _to_canonical_coord(grid_dim, source_row, source_col, instance.endpoint_kind)
    canonical_sink_row, canonical_sink_col =
        _to_canonical_coord(grid_dim, sink_row, sink_col, instance.endpoint_kind)
    return (
        warcraft_coord_to_index(grid_dim, canonical_source_row, canonical_source_col),
        warcraft_coord_to_index(grid_dim, canonical_sink_row, canonical_sink_col),
    )
end

function _canonical_tie_rank(grid_dim::Int, vertex::Integer, endpoint_kind::Symbol)
    row, col = warcraft_index_to_coord(grid_dim, vertex)
    original_row, original_col = _to_canonical_coord(grid_dim, row, col, endpoint_kind)
    return warcraft_coord_to_index(grid_dim, original_row, original_col)
end

function _canonical_benchmark_neighbor_coords(grid_dim::Int, row::Int, col::Int)
    return (
        (candidate_row, candidate_col) for
        (candidate_row, candidate_col) in (
            (row - 1, col - 1),
            (row - 1, col),
            (row - 1, col + 1),
            (row, col - 1),
            (row, col + 1),
            (row + 1, col - 1),
            (row + 1, col),
            (row + 1, col + 1),
        ) if 1 <= candidate_row <= grid_dim && 1 <= candidate_col <= grid_dim
    )
end

function _canonical_acyclic_neighbor_coords(grid_dim::Int, row::Int, col::Int)
    return (
        (candidate_row, candidate_col) for
        (candidate_row, candidate_col) in (
            (row, col + 1),
            (row + 1, col),
            (row + 1, col + 1),
        ) if 1 <= candidate_row <= grid_dim && 1 <= candidate_col <= grid_dim
    )
end

_warcraft_float_costs(theta::AbstractMatrix{<:Real}) = Float64.(theta)

function _validate_warcraft_hard_solver(solver::Symbol)
    solver in (:acyclic_dp, :bellman_ford, :dijkstra) || throw(
        ArgumentError(
            "solver must be :acyclic_dp, :bellman_ford, or :dijkstra, got $solver",
        ),
    )
    return solver
end

function _reconstruct_warcraft_path(
    predecessor::AbstractVector{<:Integer},
    sink::Int,
    grid_dim::Int,
)
    predecessor[sink] == 0 && throw(ArgumentError("no Warcraft path reaches sink $sink"))
    path = zeros(Float64, grid_dim, grid_dim)
    vertices = Int[]
    vertex = sink
    while true
        row, col = warcraft_index_to_coord(grid_dim, vertex)
        path[row, col] = 1.0
        push!(vertices, vertex)
        predecessor[vertex] == -1 && break
        vertex = Int(predecessor[vertex])
    end
    reverse!(vertices)
    return (path=path, vertices=vertices)
end

function _canonical_shortest_path_solution(
    theta::AbstractMatrix{<:Real};
    source::Int,
    sink::Int,
    endpoint_kind::Symbol=:tl_br,
    solver::Symbol=WARCRAFT_BENCHMARK_HARD_SOLVER,
)
    grid_dim = size(theta, 1)
    size(theta, 2) == grid_dim || throw(ArgumentError("theta must be square"))
    1 <= source <= grid_dim * grid_dim || throw(
        ArgumentError("source $source is out of range for grid_dim=$grid_dim"),
    )
    1 <= sink <= grid_dim * grid_dim || throw(
        ArgumentError("sink $sink is out of range for grid_dim=$grid_dim"),
    )

    solver = _validate_warcraft_hard_solver(solver)
    if solver === :acyclic_dp
        return _canonical_acyclic_shortest_path_solution(
            theta;
            source=source,
            sink=sink,
            endpoint_kind=endpoint_kind,
        )
    elseif solver === :bellman_ford
        return _canonical_bellman_ford_shortest_path_solution(
            theta;
            source=source,
            sink=sink,
            endpoint_kind=endpoint_kind,
        )
    end

    return _canonical_dijkstra_shortest_path_solution(
        theta;
        source=source,
        sink=sink,
        endpoint_kind=endpoint_kind,
    )
end

function _benchmark_priority_key(grid_dim::Int, vertex::Int)
    row, col = warcraft_index_to_coord(grid_dim, vertex)
    return (row, col)
end

function _queue_entry_lt(
    left::Tuple{Float64,Int},
    right::Tuple{Float64,Int},
    grid_dim::Int,
)
    left_cost, left_vertex = left
    right_cost, right_vertex = right
    if left_cost < right_cost
        return true
    elseif right_cost < left_cost
        return false
    end
    return _benchmark_priority_key(grid_dim, left_vertex) <
           _benchmark_priority_key(grid_dim, right_vertex)
end

function _pop_benchmark_queue!(
    queue::Vector{Tuple{Float64,Int}},
    grid_dim::Int,
)
    best_idx = firstindex(queue)
    best_entry = queue[best_idx]
    for idx in (best_idx + 1):lastindex(queue)
        entry = queue[idx]
        if _queue_entry_lt(entry, best_entry, grid_dim)
            best_idx = idx
            best_entry = entry
        end
    end
    deleteat!(queue, best_idx)
    return best_entry
end

function _canonical_dijkstra_shortest_path_solution(
    theta::AbstractMatrix{<:Real};
    source::Int,
    sink::Int,
    endpoint_kind::Symbol=:tl_br,
)
    grid_dim = size(theta, 1)
    size(theta, 2) == grid_dim || throw(ArgumentError("theta must be square"))
    1 <= source <= grid_dim * grid_dim || throw(
        ArgumentError("source $source is out of range for grid_dim=$grid_dim"),
    )
    1 <= sink <= grid_dim * grid_dim || throw(
        ArgumentError("sink $sink is out of range for grid_dim=$grid_dim"),
    )

    costs = fill(1.0e10, grid_dim, grid_dim)
    num_paths = zeros(Float64, grid_dim, grid_dim)
    predecessor = fill(0, grid_dim * grid_dim)
    certain = falses(grid_dim, grid_dim)
    queue = Tuple{Float64,Int}[]

    source_row, source_col = warcraft_index_to_coord(grid_dim, source)
    costs[source_row, source_col] = float(theta[source_row, source_col])
    num_paths[source_row, source_col] = 1.0
    predecessor[source] = -1
    push!(queue, (costs[source_row, source_col], source))

    while !isempty(queue)
        _, current_vertex = _pop_benchmark_queue!(queue, grid_dim)
        current_row, current_col = warcraft_index_to_coord(grid_dim, current_vertex)

        for (candidate_row, candidate_col) in
            _canonical_benchmark_neighbor_coords(grid_dim, current_row, current_col)
            certain[candidate_row, candidate_col] && continue
            candidate_cost =
                float(theta[candidate_row, candidate_col]) + costs[current_row, current_col]
            if candidate_cost < costs[candidate_row, candidate_col]
                costs[candidate_row, candidate_col] = candidate_cost
                candidate_vertex =
                    warcraft_coord_to_index(grid_dim, candidate_row, candidate_col)
                push!(queue, (candidate_cost, candidate_vertex))
                predecessor[candidate_vertex] = current_vertex
                num_paths[candidate_row, candidate_col] = num_paths[current_row, current_col]
            elseif candidate_cost == costs[candidate_row, candidate_col]
                num_paths[candidate_row, candidate_col] += num_paths[current_row, current_col]
            end
        end

        certain[current_row, current_col] = true
    end

    return _reconstruct_warcraft_path(predecessor, sink, grid_dim)
end

function _canonical_bellman_ford_shortest_path_solution(
    theta::AbstractMatrix{<:Real};
    source::Int,
    sink::Int,
    endpoint_kind::Symbol=:tl_br,
)
    grid_dim = size(theta, 1)
    size(theta, 2) == grid_dim || throw(ArgumentError("theta must be square"))
    1 <= source <= grid_dim * grid_dim || throw(
        ArgumentError("source $source is out of range for grid_dim=$grid_dim"),
    )
    1 <= sink <= grid_dim * grid_dim || throw(
        ArgumentError("sink $sink is out of range for grid_dim=$grid_dim"),
    )

    costs = _warcraft_float_costs(theta)
    n_vertices = grid_dim * grid_dim
    distances = fill(Inf, n_vertices)
    predecessor = fill(0, n_vertices)
    edges = _canonical_warcraft_edges(grid_dim; acyclic=false)

    source_row, source_col = warcraft_index_to_coord(grid_dim, source)
    distances[source] = costs[source_row, source_col]
    predecessor[source] = -1

    for _ in 1:(n_vertices - 1)
        updated = false
        for (src, dst) in edges
            isfinite(distances[src]) || continue
            dst_row, dst_col = warcraft_index_to_coord(grid_dim, dst)
            candidate_distance = distances[src] + costs[dst_row, dst_col]
            if candidate_distance < distances[dst] - 1e-12 || (
                abs(candidate_distance - distances[dst]) <= 1e-12 &&
                (predecessor[dst] <= 0 || _canonical_tie_rank(grid_dim, src, endpoint_kind) <
                 _canonical_tie_rank(grid_dim, predecessor[dst], endpoint_kind))
            )
                distances[dst] = candidate_distance
                predecessor[dst] = src
                updated = true
            end
        end
        updated || break
    end

    for (src, dst) in edges
        isfinite(distances[src]) || continue
        dst_row, dst_col = warcraft_index_to_coord(grid_dim, dst)
        candidate_distance = distances[src] + costs[dst_row, dst_col]
        if candidate_distance < distances[dst] - 1e-12
            throw(
                ArgumentError(
                    "reachable negative cycle detected in Bellman-Ford Warcraft path backend",
                ),
            )
        end
    end

    return _reconstruct_warcraft_path(predecessor, sink, grid_dim)
end

function _canonical_acyclic_shortest_path_solution(
    theta::AbstractMatrix{<:Real};
    source::Int,
    sink::Int,
    endpoint_kind::Symbol=:tl_br,
)
    grid_dim = size(theta, 1)
    size(theta, 2) == grid_dim || throw(ArgumentError("theta must be square"))
    1 <= source <= grid_dim * grid_dim || throw(
        ArgumentError("source $source is out of range for grid_dim=$grid_dim"),
    )
    1 <= sink <= grid_dim * grid_dim || throw(
        ArgumentError("sink $sink is out of range for grid_dim=$grid_dim"),
    )

    costs = _warcraft_float_costs(theta)
    n_vertices = grid_dim * grid_dim
    distances = fill(Inf, n_vertices)
    predecessor = fill(0, n_vertices)

    source_row, source_col = warcraft_index_to_coord(grid_dim, source)
    sink_row, sink_col = warcraft_index_to_coord(grid_dim, sink)
    if sink_row < source_row || sink_col < source_col
        throw(
            ArgumentError(
                "acyclic Warcraft path backend requires sink to be southeast of source " *
                "after canonicalization, got source=($source_row, $source_col) and " *
                "sink=($sink_row, $sink_col)",
            ),
        )
    end

    distances[source] = costs[source_row, source_col]
    predecessor[source] = -1

    for col in source_col:grid_dim
        row_start = col == source_col ? source_row : 1
        for row in row_start:grid_dim
            current_vertex = warcraft_coord_to_index(grid_dim, row, col)
            isfinite(distances[current_vertex]) || continue
            current_distance = distances[current_vertex]

            for (candidate_row, candidate_col) in
                _canonical_acyclic_neighbor_coords(grid_dim, row, col)
                neighbor = warcraft_coord_to_index(grid_dim, candidate_row, candidate_col)
                candidate_distance = current_distance + costs[candidate_row, candidate_col]
                if candidate_distance < distances[neighbor] - 1e-12 || (
                    abs(candidate_distance - distances[neighbor]) <= 1e-12 &&
                    (
                        predecessor[neighbor] == 0 ||
                        _canonical_tie_rank(grid_dim, current_vertex, endpoint_kind) <
                        _canonical_tie_rank(grid_dim, predecessor[neighbor], endpoint_kind)
                    )
                )
                    distances[neighbor] = candidate_distance
                    predecessor[neighbor] = current_vertex
                end
            end
        end
    end

    return _reconstruct_warcraft_path(predecessor, sink, grid_dim)
end

function _canonical_shortest_path(
    theta::AbstractMatrix{<:Real};
    source::Int,
    sink::Int,
    endpoint_kind::Symbol=:tl_br,
    solver::Symbol=WARCRAFT_BENCHMARK_HARD_SOLVER,
)
    return _canonical_shortest_path_solution(
        theta;
        source=source,
        sink=sink,
        endpoint_kind=endpoint_kind,
        solver=solver,
    ).path
end

function solve_warcraft_path(
    theta;
    instance::WarcraftInstance,
    solver::Symbol=WARCRAFT_BENCHMARK_HARD_SOLVER,
)
    grid_dim = warcraft_grid_dim(instance)
    theta_matrix = _to_warcraft_matrix(theta, grid_dim)
    canonical_theta = _to_canonical_orientation(theta_matrix, instance.endpoint_kind)
    canonical_source, canonical_sink = _canonical_source_sink(instance)
    canonical_path = _canonical_shortest_path(
        canonical_theta;
        source=canonical_source,
        sink=canonical_sink,
        endpoint_kind=instance.endpoint_kind,
        solver=solver,
    )
    return _from_canonical_orientation(canonical_path, instance.endpoint_kind)
end

warcraft_path_cost(weights::AbstractMatrix{<:Real}, path::AbstractMatrix{<:Real}) =
    sum(Float64.(weights) .* Float64.(path))

function warcraft_linear_maximizer(
    ;
    instance::WarcraftInstance,
    solver::Symbol=WARCRAFT_BENCHMARK_HARD_SOLVER,
)
    return InferOpt.LinearMaximizer(
        θ -> solve_warcraft_path(θ; instance=instance, solver=solver);
        g=y -> -y,
    )
end

function _canonical_warcraft_edges(grid_dim::Int; acyclic::Bool=false)
    neighbor_coords = acyclic ? _canonical_acyclic_neighbor_coords : _canonical_benchmark_neighbor_coords
    edges = Tuple{Int,Int}[]
    for col in 1:grid_dim
        for row in 1:grid_dim
            src = warcraft_coord_to_index(grid_dim, row, col)
            for (neighbor_row, neighbor_col) in neighbor_coords(grid_dim, row, col)
                push!(edges, (src, warcraft_coord_to_index(grid_dim, neighbor_row, neighbor_col)))
            end
        end
    end
    return edges
end

const GUROBI_ENV = Ref{Gurobi.Env}()

function get_gurobi_env()
    if !isassigned(GUROBI_ENV)
        GUROBI_ENV[] = Gurobi.Env()
    end
    return GUROBI_ENV[]
end

function _build_warcraft_jump_model(grid_dim::Int, source::Int, sink::Int, edges::Vector{Tuple{Int,Int}}, lambda::Real, theta::AbstractMatrix{<:Real})
    n = grid_dim * grid_dim
    n_edges = length(edges)

    incoming = [Int[] for _ in 1:n]
    outgoing = [Int[] for _ in 1:n]
    for (idx, (src, dst)) in enumerate(edges)
        push!(outgoing[src], idx)
        push!(incoming[dst], idx)
    end

    model = Model(() -> Gurobi.Optimizer(get_gurobi_env()))
    set_silent(model)

    @variable(model, 0 <= x[1:n] <= 1)
    @variable(model, 0 <= e[1:n_edges] <= 1)

    # Source is on the path
    @constraint(model, x[source] == 1)

    # Flow in: for non-source vertices, x[v] = sum of incoming edge flows
    for v in 1:n
        v == source || @constraint(model, x[v] == sum(e[idx] for idx in incoming[v]; init=0.0))
    end

    # Flow out: for non-sink vertices, x[v] = sum of outgoing edge flows
    for v in 1:n
        v == sink || @constraint(model, x[v] == sum(e[idx] for idx in outgoing[v]; init=0.0))
    end

    # No incoming to source
    if !isempty(incoming[source])
        @constraint(model, sum(e[idx] for idx in incoming[source]) == 0)
    end

    # No outgoing from sink
    if !isempty(outgoing[sink])
        @constraint(model, sum(e[idx] for idx in outgoing[sink]) == 0)
    end

    # Objective: min theta'*x + (lambda/2)*||x||^2
    theta_vec = vec(Float64.(theta))
    @objective(model, Min,
        sum(theta_vec[v] * x[v] for v in 1:n) +
        (lambda / 2) * sum(x[v]^2 for v in 1:n)
    )

    return model, x, e
end

function _build_warcraft_diffopt_model(
    grid_dim::Int,
    source::Int,
    sink::Int,
    edges::Vector{Tuple{Int,Int}},
    tau::Real,
    theta::AbstractVector{<:Real},
)
    n = grid_dim * grid_dim
    n_edges = length(edges)
    length(theta) == n ||
        throw(ArgumentError("expected theta vector of length $n, got $(length(theta))"))

    incoming = [Int[] for _ in 1:n]
    outgoing = [Int[] for _ in 1:n]
    for (idx, (src, dst)) in enumerate(edges)
        push!(outgoing[src], idx)
        push!(incoming[dst], idx)
    end

    model = DiffOpt.quadratic_diff_model(() -> Gurobi.Optimizer(get_gurobi_env()))
    set_silent(model)

    @variable(model, 0 <= e[1:n_edges] <= 1)
    @variable(model, theta_param[v = 1:n] in Parameter(theta[v]))

    for v in 1:n
        incoming_flow = sum(e[idx] for idx in incoming[v]; init=0.0)
        outgoing_flow = sum(e[idx] for idx in outgoing[v]; init=0.0)
        if v == source
            @constraint(model, outgoing_flow == 1)
            isempty(incoming[v]) || @constraint(model, incoming_flow == 0)
        elseif v == sink
            isempty(outgoing[v]) || @constraint(model, outgoing_flow == 0)
            @constraint(model, incoming_flow == 1)
        else
            @constraint(model, outgoing_flow == incoming_flow)
            @constraint(model, incoming_flow <= 1)
            @constraint(model, outgoing_flow <= 1)
        end
    end

    tau_float = float(tau)
    edge_ridge = max(tau_float * 1e-2, 1e-6)
    @objective(
        model,
        Min,
        sum(
            theta_param[v] * sum(e[idx] for idx in incoming[v]; init=0.0) for
            v in 1:n if v != source
        ) +
        (tau_float / 2) * sum(
            (sum(e[idx] for idx in incoming[v]; init=0.0))^2 for v in 1:n if v != source
        ) +
        (edge_ridge / 2) * sum(e[idx]^2 for idx in 1:n_edges),
    )

    return model, e, theta_param, incoming
end

function _warcraft_diffopt_vertex_solution(layer::WarcraftDiffOptLayer, edge_solution::AbstractVector{<:Real})
    length(edge_solution) == length(layer.e) || throw(
        ArgumentError("expected edge solution of length $(length(layer.e)), got $(length(edge_solution))"),
    )
    vertex_solution = zeros(Float64, layer.grid_dim * layer.grid_dim)
    vertex_solution[layer.canonical_source] = 1.0
    for v in 1:length(layer.incoming_edges)
        v == layer.canonical_source && continue
        vertex_solution[v] = sum(edge_solution[idx] for idx in layer.incoming_edges[v]; init=0.0)
    end
    return vertex_solution
end

function init_warcraft_diffopt_layer(
    instance::WarcraftInstance;
    tau::Real=1e-3,
)
    isfinite(tau) && tau > 0 ||
        throw(ArgumentError("tau must be positive and finite, got $tau"))
    grid_dim = warcraft_grid_dim(instance)
    canonical_source, canonical_sink = _canonical_source_sink(instance)
    edges = _canonical_warcraft_edges(grid_dim; acyclic=true)
    n = grid_dim * grid_dim
    model, e, theta_params, incoming = _build_warcraft_diffopt_model(
        grid_dim,
        canonical_source,
        canonical_sink,
        edges,
        tau,
        zeros(n),
    )
    return WarcraftDiffOptLayer(
        model,
        e,
        theta_params,
        incoming,
        edges,
        grid_dim,
        canonical_source,
        canonical_sink,
        instance.endpoint_kind,
        float(tau),
        zeros(Float64, length(edges)),
    )
end

function _warcraft_configure_diffopt_linear_solver!(layer::WarcraftDiffOptLayer)
    backend = JuMP.backend(layer.model)
    backend.diff = nothing
    backend.index_map = nothing
    diff_model = DiffOpt._diff(backend)
    JuMP.MOI.set(
        diff_model,
        DiffOpt.QuadraticProgram.LinearAlgebraSolver(),
        WarcraftDiffOptLinearSolver(),
    )
    return diff_model
end

function (layer::WarcraftDiffOptLayer)(theta_pred::AbstractMatrix{<:Real})
    theta_matrix = _to_warcraft_matrix(theta_pred, layer.grid_dim)
    canonical_theta = _to_canonical_orientation(theta_matrix, layer.endpoint_kind)
    theta_vec = vec(_warcraft_float_costs(canonical_theta))

    for idx in eachindex(layer.theta_params)
        set_parameter_value(layer.theta_params[idx], theta_vec[idx])
    end

    if any(!iszero, layer.last_edge_solution)
        for idx in eachindex(layer.e)
            set_start_value(layer.e[idx], layer.last_edge_solution[idx])
        end
    end

    optimize!(layer.model)
    status = termination_status(layer.model)
    status in (OPTIMAL, LOCALLY_SOLVED) ||
        error("DiffOpt Warcraft layer failed with status $status")

    edge_solution = Float64.(value.(layer.e))
    layer.last_edge_solution .= edge_solution
    vertex_solution = _warcraft_diffopt_vertex_solution(layer, edge_solution)
    canonical_path = reshape(vertex_solution, layer.grid_dim, layer.grid_dim)
    return _from_canonical_orientation(canonical_path, layer.endpoint_kind)
end

function ChainRulesCore.rrule(
    layer::WarcraftDiffOptLayer,
    theta_pred::AbstractMatrix{<:Real},
)
    relaxed_path = layer(theta_pred)

    function pullback(drelaxed_path)
        drelaxed_path = ChainRulesCore.unthunk(drelaxed_path)
        canonical_seed = _to_canonical_orientation(
            _to_warcraft_matrix(drelaxed_path, layer.grid_dim),
            layer.endpoint_kind,
        )
        DiffOpt.empty_input_sensitivities!(layer.model)
        seed_vec = vec(Float64.(canonical_seed))
        edge_seed = zeros(Float64, length(layer.e))
        for (edge_idx, (_, dst)) in enumerate(layer.edges)
            edge_seed[edge_idx] += seed_vec[dst]
        end
        for idx in eachindex(layer.e)
            DiffOpt.set_reverse_variable(layer.model, layer.e[idx], edge_seed[idx])
        end
        _warcraft_configure_diffopt_linear_solver!(layer)
        DiffOpt.reverse_differentiate!(layer.model)
        canonical_grad = reshape(
            Float64.([
                DiffOpt.get_reverse_parameter(layer.model, theta_param) for
                theta_param in layer.theta_params
            ]),
            layer.grid_dim,
            layer.grid_dim,
        )
        theta_grad = _from_canonical_orientation(canonical_grad, layer.endpoint_kind)
        return ChainRulesCore.NoTangent(), eltype(theta_pred).(theta_grad)
    end

    return relaxed_path, pullback
end

function init_warcraft_projection_cache(instance::WarcraftInstance; lambda::Real)
    lambda > 0 || throw(ArgumentError("lambda must be positive, got $lambda"))
    grid_dim = warcraft_grid_dim(instance)
    canonical_source, canonical_sink = _canonical_source_sink(instance)
    edges = _canonical_warcraft_edges(grid_dim; acyclic=false)
    n = grid_dim * grid_dim

    jump_model, x, e = _build_warcraft_jump_model(
        grid_dim, canonical_source, canonical_sink, edges,
        lambda, zeros(grid_dim, grid_dim),
    )

    return WarcraftProjectionCache(
        jump_model, x, e,
        float(lambda), grid_dim,
        canonical_source, canonical_sink,
        edges, zeros(Float64, n),
    )
end

function _warcraft_cache_matches_instance(
    cache::WarcraftProjectionCache,
    instance::WarcraftInstance,
)
    canonical_source, canonical_sink = _canonical_source_sink(instance)
    return (
        cache.grid_dim == warcraft_grid_dim(instance) &&
        cache.canonical_source == canonical_source &&
        cache.canonical_sink == canonical_sink
    )
end

function _rebuild_warcraft_projection_cache!(
    cache::WarcraftProjectionCache,
    instance::WarcraftInstance;
    lambda::Real,
)
    grid_dim = warcraft_grid_dim(instance)
    canonical_source, canonical_sink = _canonical_source_sink(instance)
    edges = _canonical_warcraft_edges(grid_dim; acyclic=false)
    n = grid_dim * grid_dim

    jump_model, x, e = _build_warcraft_jump_model(
        grid_dim, canonical_source, canonical_sink, edges,
        lambda, zeros(grid_dim, grid_dim),
    )

    cache.jump_model = jump_model
    cache.x = x
    cache.e = e
    cache.lambda = float(lambda)
    cache.grid_dim = grid_dim
    cache.canonical_source = canonical_source
    cache.canonical_sink = canonical_sink
    cache.edges = edges
    cache.last_vertex_solution = zeros(Float64, n)
    return cache
end

function reset_warcraft_projection_cache!(cache::WarcraftProjectionCache)
    fill!(cache.last_vertex_solution, 0.0)
    return cache
end

function _warcraft_projection_feasible_radius(instance::WarcraftInstance)
    grid_dim = warcraft_grid_dim(instance)
    canonical_source, canonical_sink = _canonical_source_sink(instance)
    edges = _canonical_warcraft_edges(grid_dim; acyclic=false)
    model, x, _ = _build_warcraft_jump_model(
        grid_dim,
        canonical_source,
        canonical_sink,
        edges,
        1.0,
        zeros(grid_dim, grid_dim),
    )

    set_objective_sense(model, JuMP.MOI.MAX_SENSE)
    set_optimizer_attribute(model, "NonConvex", 2)
    for v in eachindex(x)
        set_objective_coefficient(model, x[v], 0.0)
        set_objective_coefficient(model, x[v], x[v], 1.0)
    end

    optimize!(model)
    status = termination_status(model)
    status in (OPTIMAL, LOCALLY_SOLVED) ||
        error("Warcraft projection-radius solve failed with status $status")
    return sqrt(max(objective_value(model), 0.0))
end

function solve_warcraft_projection(
    theta::AbstractMatrix{<:Real};
    instance::WarcraftInstance,
    cache::Union{Nothing,WarcraftProjectionCache}=nothing,
    lambda::Real,
    use_warm_start::Bool=true,
)
    lambda > 0 || throw(ArgumentError("lambda must be positive, got $lambda"))
    grid_dim = warcraft_grid_dim(instance)
    theta_matrix = _to_warcraft_matrix(theta, grid_dim)
    canonical_theta = _to_canonical_orientation(theta_matrix, instance.endpoint_kind)
    theta_costs = _warcraft_float_costs(canonical_theta)
    theta_vec = vec(theta_costs)
    local_cache =
        isnothing(cache) ? init_warcraft_projection_cache(instance; lambda=lambda) : cache
    if !_warcraft_cache_matches_instance(local_cache, instance)
        _rebuild_warcraft_projection_cache!(local_cache, instance; lambda=lambda)
    end

    n = grid_dim * grid_dim
    x = local_cache.x
    model = local_cache.jump_model

    # Update linear objective coefficients (theta)
    for v in 1:n
        set_objective_coefficient(model, x[v], theta_vec[v])
    end

    # Update quadratic objective coefficients (lambda) if changed.
    # JuMP's quadratic objective coefficient is the literal coefficient on x[v]^2,
    # so this must match the (lambda / 2) term used when the model is built.
    if local_cache.lambda != float(lambda)
        for v in 1:n
            set_objective_coefficient(model, x[v], x[v], float(lambda) / 2)
        end
        local_cache.lambda = float(lambda)
    end

    # Warm-start from previous solution
    if use_warm_start && any(!iszero, local_cache.last_vertex_solution)
        for v in 1:n
            set_start_value(x[v], local_cache.last_vertex_solution[v])
        end
    end

    optimize!(model)

    status = termination_status(model)
    if status == OPTIMAL || status == LOCALLY_SOLVED
        vertex_sol = value.(x)
        canonical_path = reshape(vertex_sol, grid_dim, grid_dim)
        obj_val = objective_value(model)
        exit_flag = 1
        info = (status=status,)
        local_cache.last_vertex_solution .= vertex_sol
    else
        fallback = _canonical_shortest_path_solution(
            theta_costs;
            source=local_cache.canonical_source,
            sink=local_cache.canonical_sink,
            endpoint_kind=instance.endpoint_kind,
            solver=WARCRAFT_BENCHMARK_HARD_SOLVER,
        )
        canonical_path = fallback.path
        obj_val = dot(theta_vec, vec(fallback.path)) +
                  (float(lambda) / 2) * sum(abs2, fallback.path)
        exit_flag = 0
        info = (status=status,)
        local_cache.last_vertex_solution .= vec(fallback.path)
    end

    return (
        decision=_from_canonical_orientation(canonical_path, instance.endpoint_kind),
        objective_value=obj_val,
        exit_flag=exit_flag,
        info=info,
        cache=local_cache,
    )
end

function projection_optimizer(
    theta::AbstractMatrix{<:Real};
    instance::WarcraftInstance,
    cache::Union{Nothing,WarcraftProjectionCache}=nothing,
    lambda::Real,
    use_warm_start::Bool=true,
)
    return solve_warcraft_projection(
        theta;
        instance=instance,
        cache=cache,
        lambda=lambda,
        use_warm_start=use_warm_start,
    ).decision
end

function projection_optimizer(
    theta::AbstractVector{<:Real};
    instance::WarcraftInstance,
    cache::Union{Nothing,WarcraftProjectionCache}=nothing,
    lambda::Real,
    use_warm_start::Bool=true,
)
    grid_dim = warcraft_grid_dim(instance)
    return projection_optimizer(
        reshape(theta, grid_dim, grid_dim);
        instance=instance,
        cache=cache,
        lambda=lambda,
        use_warm_start=use_warm_start,
    )
end
