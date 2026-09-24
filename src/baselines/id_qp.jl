module IDQPBaseline

using CSV
using ChainRulesCore
using DataFrames
import DiffOpt
using Flux
using Gurobi
using JuMP
using LinearAlgebra
using Random
using Statistics

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

# Load the existing data, training, and evaluation implementation inside this
# module. ID-QP therefore extends the established APIs only when this standalone
# file is included and does not alter the default code paths.
include(joinpath(PROJECT_ROOT, "src", "models", "knapsack_oracles.jl"))
include(joinpath(PROJECT_ROOT, "src", "models", "battery_oracles.jl"))
include(joinpath(PROJECT_ROOT, "src", "models", "RSPOPlusLoss.jl"))
include(joinpath(PROJECT_ROOT, "src", "models", "scheduling.jl"))
include(joinpath(PROJECT_ROOT, "src", "datagen", "synthetic_knapsack.jl"))
include(joinpath(PROJECT_ROOT, "src", "models", "training.jl"))
include(joinpath(PROJECT_ROOT, "src", "experiments", "synthetic_knapsack.jl"))
include(joinpath(PROJECT_ROOT, "src", "datagen", "pjm_battery.jl"))
include(joinpath(PROJECT_ROOT, "src", "models", "pjm_battery_training.jl"))
include(joinpath(PROJECT_ROOT, "src", "experiments", "pjm_battery.jl"))

export IDQPDiagnostics,
       IDQPFailure,
       IDQPLayer,
       IDQPObjective,
       PJMIDQPObjective,
       IDQPRunResult,
       IDQPTuningResult,
       build_id_qp_layer,
       diagnostics_row,
       id_qp_loss,
       run_pjm_id_qp,
       run_synthetic_id_qp,
       synthetic_validation_metrics,
       pjm_validation_metrics

const IDQP_MOI = JuMP.MOI
const IDQP_SUCCESS_STATUSES = (IDQP_MOI.OPTIMAL, IDQP_MOI.LOCALLY_SOLVED, IDQP_MOI.ALMOST_OPTIMAL)

struct IDQPFailure
    stage::Symbol
    status::String
    message::String
end

Base.@kwdef mutable struct IDQPDiagnostics
    forward_solves::Int = 0
    backward_resolves::Int = 0
    differentiations::Int = 0
    forward_solve_seconds::Float64 = 0.0
    backward_resolve_seconds::Float64 = 0.0
    differentiation_seconds::Float64 = 0.0
    nonfinite_solution_count::Int = 0
    nonfinite_gradient_count::Int = 0
    failures::Vector{IDQPFailure} = IDQPFailure[]
end

struct IDQPSolverError <: Exception
    stage::Symbol
    status::String
    message::String
end

function Base.showerror(io::IO, err::IDQPSolverError)
    print(io, "ID-QP $(err.stage) failed [$(err.status)]: $(err.message)")
end

struct IDQPLeastSquaresSolver end

function DiffOpt.QuadraticProgram.solve_system(
    ::IDQPLeastSquaresSolver,
    lhs,
    rhs,
    iterative::Bool,
)
    return DiffOpt.QuadraticProgram.IterativeSolvers.lsqr(lhs, rhs)
end

function JuMP.MOI.Utilities.map_indices(
    variable_map::AbstractDict{T,T},
    solver::IDQPLeastSquaresSolver,
) where {T<:Union{JuMP.MOI.VariableIndex,JuMP.MOI.ConstraintIndex}}
    return solver
end

mutable struct IDQPLayer{I}
    model::Model
    w::Vector{VariableRef}
    theta_params::Vector{VariableRef}
    instance::I
    epsilon::Float64
    last_solution::Vector{Float64}
    diagnostics::IDQPDiagnostics
end

function _record_failure!(
    diagnostics::IDQPDiagnostics,
    stage::Symbol,
    status,
    message,
)
    failure = IDQPFailure(stage, string(status), string(message))
    push!(diagnostics.failures, failure)
    return failure
end

function _id_qp_model(dim::Int, epsilon::Real)
    model = DiffOpt.quadratic_diff_model(
        () -> Gurobi.Optimizer(get_knapsack_gurobi_env()),
    )
    set_silent(model)
    @variable(model, 0 <= w[1:dim] <= 1)
    @variable(model, theta_param[idx = 1:dim] in Parameter(0.0))
    @objective(
        model,
        Min,
        sum(theta_param[idx] * w[idx] for idx in 1:dim) +
        (float(epsilon) / 2) * sum(w[idx]^2 for idx in 1:dim),
    )
    return model, w, theta_param
end

function build_id_qp_layer(instance::KnapsackInstance; epsilon::Real)
    isfinite(epsilon) && epsilon > 0 ||
        throw(ArgumentError("epsilon must be positive and finite, got $epsilon"))
    dim = length(instance.weights)
    model, w, theta_params = _id_qp_model(dim, epsilon)
    @constraint(
        model,
        sum(instance.weights[idx] * w[idx] for idx in 1:dim) <= instance.capacity,
    )
    return IDQPLayer(
        model,
        w,
        theta_params,
        instance,
        float(epsilon),
        zeros(Float64, dim),
        IDQPDiagnostics(),
    )
end

function build_id_qp_layer(instance::BatteryDispatchInstance; epsilon::Real)
    isfinite(epsilon) && epsilon > 0 ||
        throw(ArgumentError("epsilon must be positive and finite, got $epsilon"))
    model, w, theta_params = _id_qp_model(instance.horizon, epsilon)
    @constraint(model, sum(w) == instance.capacity)
    return IDQPLayer(
        model,
        w,
        theta_params,
        instance,
        float(epsilon),
        zeros(Float64, instance.horizon),
        IDQPDiagnostics(),
    )
end

function _configure_id_qp_linear_solver!(layer::IDQPLayer)
    backend = JuMP.backend(layer.model)
    backend.diff = nothing
    backend.index_map = nothing
    diff_model = DiffOpt._diff(backend)
    IDQP_MOI.set(
        diff_model,
        DiffOpt.QuadraticProgram.LinearAlgebraSolver(),
        IDQPLeastSquaresSolver(),
    )
    return diff_model
end

function _set_id_qp_parameters!(
    layer::IDQPLayer,
    theta_pred::AbstractVector{<:Real},
)
    length(theta_pred) == length(layer.theta_params) || throw(
        ArgumentError(
            "prediction length $(length(theta_pred)) does not match ID-QP dimension $(length(layer.theta_params))",
        ),
    )
    all(isfinite, theta_pred) || throw(
        ArgumentError("ID-QP prediction contains a non-finite coefficient"),
    )
    for idx in eachindex(layer.theta_params)
        set_parameter_value(layer.theta_params[idx], Float64(theta_pred[idx]))
        set_start_value(layer.w[idx], layer.last_solution[idx])
    end
    return layer
end

function _solve_id_qp!(
    layer::IDQPLayer,
    theta_pred::AbstractVector{<:Real};
    stage::Symbol,
)
    _set_id_qp_parameters!(layer, theta_pred)
    elapsed = @elapsed optimize!(layer.model)
    if stage === :forward
        layer.diagnostics.forward_solves += 1
        layer.diagnostics.forward_solve_seconds += elapsed
    elseif stage === :backward_resolve
        layer.diagnostics.backward_resolves += 1
        layer.diagnostics.backward_resolve_seconds += elapsed
    else
        throw(ArgumentError("unknown ID-QP solve stage $stage"))
    end

    status = termination_status(layer.model)
    if !(status in IDQP_SUCCESS_STATUSES)
        message = "optimizer did not return a usable solution"
        _record_failure!(layer.diagnostics, stage, status, message)
        throw(IDQPSolverError(stage, string(status), message))
    end

    solution = Float64.(value.(layer.w))
    if any(!isfinite, solution)
        layer.diagnostics.nonfinite_solution_count += 1
        message = "optimizer returned a non-finite decision"
        _record_failure!(layer.diagnostics, stage, status, message)
        throw(IDQPSolverError(stage, string(status), message))
    end
    layer.last_solution .= solution
    return solution
end

function (layer::IDQPLayer)(theta_pred::AbstractVector{<:Real})
    return _solve_id_qp!(layer, theta_pred; stage=:forward)
end

function ChainRulesCore.rrule(
    layer::IDQPLayer,
    theta_pred::AbstractVector{<:Real},
)
    decision = layer(theta_pred)
    captured_theta = Float64.(theta_pred)

    function id_qp_pullback(ddecision)
        ddecision = ChainRulesCore.unthunk(ddecision)
        if ddecision isa ChainRulesCore.AbstractZero
            return ChainRulesCore.NoTangent(), zeros(eltype(theta_pred), size(theta_pred))
        end

        # A layer is shared by all samples from a client. Re-solving the captured
        # sample here is essential: the final forward solve in a batch may have a
        # different active set from this pullback's sample.
        _solve_id_qp!(layer, captured_theta; stage=:backward_resolve)
        seed = Float64.(ddecision)
        length(seed) == length(layer.w) || throw(
            ArgumentError(
                "ID-QP pullback seed length $(length(seed)) does not match decision dimension $(length(layer.w))",
            ),
        )
        gradient = _differentiate_current_id_qp!(layer, seed)
        return ChainRulesCore.NoTangent(), eltype(theta_pred).(gradient)
    end

    return decision, id_qp_pullback
end

function _differentiate_current_id_qp!(
    layer::IDQPLayer,
    decision_seed::AbstractVector{<:Real},
)
    length(decision_seed) == length(layer.w) || throw(
        ArgumentError(
            "ID-QP seed length $(length(decision_seed)) does not match decision dimension $(length(layer.w))",
        ),
    )
    DiffOpt.empty_input_sensitivities!(layer.model)
    for idx in eachindex(layer.w)
        DiffOpt.set_reverse_variable(
            layer.model,
            layer.w[idx],
            Float64(decision_seed[idx]),
        )
    end
    elapsed = @elapsed begin
        try
            _configure_id_qp_linear_solver!(layer)
            DiffOpt.reverse_differentiate!(layer.model)
        catch err
            message = sprint(showerror, err)
            _record_failure!(layer.diagnostics, :differentiate, "EXCEPTION", message)
            throw(IDQPSolverError(:differentiate, "EXCEPTION", message))
        end
    end
    layer.diagnostics.differentiations += 1
    layer.diagnostics.differentiation_seconds += elapsed

    gradient = Float64.([
        DiffOpt.get_reverse_parameter(layer.model, theta_param) for
        theta_param in layer.theta_params
    ])
    if any(!isfinite, gradient)
        layer.diagnostics.nonfinite_gradient_count += 1
        message = "implicit differentiation returned a non-finite gradient"
        _record_failure!(layer.diagnostics, :differentiate, "NONFINITE", message)
        throw(IDQPSolverError(:differentiate, "NONFINITE", message))
    end
    return gradient
end

"""
    id_qp_loss(layer, theta_pred, theta_true)

Compute `theta_true' * w_epsilon(theta_pred)`, where `w_epsilon` is the
client-local strongly convex QP solution. The custom rule above differentiates
this true downstream objective through the local KKT system.
"""
function id_qp_loss(
    layer::IDQPLayer,
    theta_pred::AbstractVector{<:Real},
    theta_true::AbstractVector{<:Real},
)
    length(theta_pred) == length(theta_true) ||
        throw(ArgumentError("prediction and truth lengths must match"))
    return dot(theta_true, layer(theta_pred))
end

mutable struct IDQPObjective <: SyntheticKnapsackObjective
    epsilon_multiplier::Float64
    layers::Dict{Int,IDQPLayer}
end

function IDQPObjective(; epsilon_multiplier::Real=1.0)
    isfinite(epsilon_multiplier) && epsilon_multiplier > 0 || throw(
        ArgumentError("epsilon multiplier must be positive and finite"),
    )
    return IDQPObjective(float(epsilon_multiplier), Dict{Int,IDQPLayer}())
end

objective_name(::IDQPObjective) = :id_qp
objective_uses_lambda(::IDQPObjective) = true
objective_tracks_bound(::IDQPObjective) = false

function _initial_client_objective_state(
    objective::IDQPObjective,
    client::SyntheticKnapsackClientData,
    config::SyntheticKnapsackTrainingConfig,
)
    base_epsilon = if config.per_client_lambda
        only(compute_knapsack_per_client_lambda0([client]; fallback=config.lambda0))
    else
        config.lambda0
    end
    layer = build_id_qp_layer(
        client.instance;
        epsilon=objective.epsilon_multiplier * _require_objective_lambda(base_epsilon),
    )
    objective.layers[client.client_id] = layer
    return layer
end

function _build_loss_layer(
    ::IDQPObjective,
    client::SyntheticKnapsackClientData;
    lambda::Real,
    config::SyntheticKnapsackTrainingConfig,
)
    return nothing
end

function _prepare_client_training_state(
    objective::IDQPObjective,
    client::SyntheticKnapsackClientData,
    config::SyntheticKnapsackTrainingConfig,
    state,
    lambda::Real,
)
    epsilon = objective.epsilon_multiplier * _require_objective_lambda(lambda)
    isapprox(state.epsilon, epsilon; atol=0.0, rtol=1e-12) || throw(
        ArgumentError(
            "ID-QP epsilon changed within a fixed-epsilon run: $(state.epsilon) -> $epsilon",
        ),
    )
    return state
end

_persist_client_training_state(::IDQPObjective, state) = state

function _batch_training_loss(
    ::IDQPObjective,
    loss_layer,
    theta_pred,
    theta_batch,
    batch_indices,
    client::SyntheticKnapsackClientData,
    state,
    config::SyntheticKnapsackTrainingConfig;
    lambda::Real,
)
    total = zero(eltype(theta_pred))
    for batch_pos in eachindex(batch_indices)
        total += id_qp_loss(
            state,
            view(theta_pred, :, batch_pos),
            view(theta_batch, :, batch_pos),
        )
    end
    return total / length(batch_indices)
end

function train_client_model!(
    model,
    opt_state,
    client::SyntheticKnapsackClientData,
    objective::IDQPObjective,
    config::SyntheticKnapsackTrainingConfig,
    rng::AbstractRNG;
    state=nothing,
    lambda::Real=NaN,
)
    n_samples = size(client.train_x, 2)
    n_samples == 0 && return 0.0, state
    layer = _prepare_client_training_state(objective, client, config, state, lambda)
    total_loss = 0.0
    total_samples = 0

    for _ in 1:config.local_epochs
        for batch_indices in _batch_index_sets(
            rng,
            n_samples,
            config.batch_size;
            shuffle=config.shuffle_batches,
        )
            x_batch = view(client.train_x, :, batch_indices)
            theta_batch = view(client.train_theta_true, :, batch_indices)
            theta_pred = model(x_batch)
            output_gradient = zeros(Float64, size(theta_pred))
            batch_loss = 0.0

            # Solve and differentiate each sample before reusing the client's
            # cached layer. This keeps the KKT active set sample-correct without
            # a second QP solve in the pullback.
            for batch_pos in eachindex(batch_indices)
                prediction = view(theta_pred, :, batch_pos)
                truth = view(theta_batch, :, batch_pos)
                decision = layer(prediction)
                batch_loss += dot(truth, decision)
                output_gradient[:, batch_pos] .=
                    _differentiate_current_id_qp!(layer, truth)
            end

            _, grads = Flux.withgradient(model) do m
                return sum(m(x_batch) .* output_gradient) / length(batch_indices)
            end
            Flux.update!(opt_state, model, grads[1])

            total_loss += batch_loss
            total_samples += length(batch_indices)
        end
    end
    return total_loss / total_samples, layer
end

mutable struct PJMIDQPObjective <: PJMBatteryObjective
    epsilon_multiplier::Float64
    layers::Dict{Int,IDQPLayer}
end

function PJMIDQPObjective(; epsilon_multiplier::Real=1.0)
    isfinite(epsilon_multiplier) && epsilon_multiplier > 0 || throw(
        ArgumentError("epsilon multiplier must be positive and finite"),
    )
    return PJMIDQPObjective(float(epsilon_multiplier), Dict{Int,IDQPLayer}())
end

_pjm_objective_name(::PJMIDQPObjective) = :id_qp
_pjm_objective_uses_lambda(::PJMIDQPObjective) = true
_pjm_objective_tracks_bound(::PJMIDQPObjective) = false

function _initial_pjm_objective_state(
    objective::PJMIDQPObjective,
    client::PJMBatteryClientData,
    config::PJMBatteryTrainingConfig,
)
    base_epsilon = if config.per_client_lambda
        only(compute_pjm_per_client_lambda0([client]; fallback=config.lambda0))
    else
        config.lambda0
    end
    layer = build_id_qp_layer(
        client.instance;
        epsilon=objective.epsilon_multiplier * _pjm_require_objective_lambda(base_epsilon),
    )
    objective.layers[client.client_id] = layer
    return layer
end

function _build_pjm_loss_layer(
    ::PJMIDQPObjective,
    client::PJMBatteryClientData;
    lambda::Real,
    config::PJMBatteryTrainingConfig,
)
    return nothing
end

function _prepare_pjm_client_training_state(
    objective::PJMIDQPObjective,
    client::PJMBatteryClientData,
    config::PJMBatteryTrainingConfig,
    state,
    lambda::Real,
)
    epsilon = objective.epsilon_multiplier * _pjm_require_objective_lambda(lambda)
    isapprox(state.epsilon, epsilon; atol=0.0, rtol=1e-12) || throw(
        ArgumentError(
            "PJM ID-QP epsilon changed within a fixed-epsilon run: $(state.epsilon) -> $epsilon",
        ),
    )
    return state
end

_persist_pjm_client_training_state(::PJMIDQPObjective, state) = state

function _pjm_batch_training_loss(
    ::PJMIDQPObjective,
    loss_layer,
    theta_pred,
    theta_batch,
    batch_indices,
    client::PJMBatteryClientData,
    state,
    config::PJMBatteryTrainingConfig;
    lambda::Real,
)
    total = zero(eltype(theta_pred))
    for batch_pos in eachindex(batch_indices)
        total += id_qp_loss(
            state,
            view(theta_pred, :, batch_pos),
            view(theta_batch, :, batch_pos),
        )
    end
    return total / length(batch_indices)
end

function train_pjm_client_model!(
    model,
    opt_state,
    client::PJMBatteryClientData,
    objective::PJMIDQPObjective,
    config::PJMBatteryTrainingConfig,
    rng::AbstractRNG;
    state=nothing,
    lambda::Real=NaN,
    prox_reference=nothing,
    prox_mu::Real=config.prox_mu,
)
    n_samples = size(client.train_x, 2)
    n_samples == 0 && return 0.0, state
    layer = _prepare_pjm_client_training_state(objective, client, config, state, lambda)
    total_loss = 0.0
    total_samples = 0

    for _ in 1:config.local_epochs
        for batch_indices in _pjm_batch_index_sets(
            rng,
            n_samples,
            config.batch_size;
            shuffle=config.shuffle_batches,
        )
            x_batch = view(client.train_x, :, batch_indices)
            theta_batch = view(client.train_theta_true, :, batch_indices)
            theta_pred = model(x_batch)
            output_gradient = zeros(Float64, size(theta_pred))
            base_loss = 0.0
            for batch_pos in eachindex(batch_indices)
                prediction = view(theta_pred, :, batch_pos)
                truth = view(theta_batch, :, batch_pos)
                decision = layer(prediction)
                base_loss += dot(truth, decision)
                output_gradient[:, batch_pos] .=
                    _differentiate_current_id_qp!(layer, truth)
            end

            proximal_loss = _pjm_prox_penalty(model, prox_reference, prox_mu)
            _, grads = Flux.withgradient(model) do m
                return sum(m(x_batch) .* output_gradient) / length(batch_indices) +
                       _pjm_prox_penalty(m, prox_reference, prox_mu)
            end
            Flux.update!(opt_state, model, grads[1])

            total_loss += base_loss + proximal_loss * length(batch_indices)
            total_samples += length(batch_indices)
        end
    end
    return total_loss / total_samples, layer
end

function _validation_regret(
    model,
    x::AbstractMatrix{<:Real},
    theta_true::AbstractMatrix{<:Real},
    solve_lp;
    max_samples::Int,
)
    n_samples = min(max_samples, size(x, 2), size(theta_true, 2))
    n_samples > 0 || return (absolute_regret=NaN, relative_regret=NaN, n=0)
    predictions = model(view(x, :, 1:n_samples))
    regrets = Float64[]
    true_objectives = Float64[]
    for sample_idx in 1:n_samples
        truth = view(theta_true, :, sample_idx)
        predicted_decision, _ = solve_lp(view(predictions, :, sample_idx))
        _, true_objective = solve_lp(truth)
        push!(regrets, dot(truth, predicted_decision) - true_objective)
        push!(true_objectives, true_objective)
    end
    absolute = mean(regrets)
    denominator = mean(abs.(true_objectives))
    relative = denominator > 0 ? absolute / denominator : NaN
    return (absolute_regret=absolute, relative_regret=relative, n=n_samples)
end

function synthetic_validation_metrics(
    model,
    client::SyntheticKnapsackClientData;
    max_samples::Int=8,
)
    return _validation_regret(
        model,
        client.val_x,
        client.val_theta_true,
        theta -> solve_fractional_knapsack(
            theta;
            instance=client.instance,
            sense=:min,
        );
        max_samples=max_samples,
    )
end

function pjm_validation_metrics(
    model,
    client::PJMBatteryClientData;
    max_samples::Int=8,
)
    return _validation_regret(
        model,
        client.val_x,
        client.val_theta_true,
        theta -> solve_battery_dispatch(
            theta;
            instance=client.instance,
            sense=:min,
        );
        max_samples=max_samples,
    )
end

function _mean_finite(values)
    isempty(values) &&
        throw(ArgumentError("cannot validate an ID-QP candidate without scores"))
    all(isfinite, values) || throw(
        ArgumentError(
            "ID-QP candidate produced a non-finite validation score; the candidate is rejected and reported",
        ),
    )
    return mean(Float64.(values))
end

function _select_best_candidate(scores::AbstractVector{<:Real})
    isempty(scores) && throw(ArgumentError("candidate score vector is empty"))
    all(!isfinite, scores) &&
        throw(ArgumentError("all ID-QP epsilon candidates failed validation"))
    return argmin([isfinite(score) ? score : Inf for score in scores])
end

struct IDQPTuningResult
    epsilon_multipliers::Vector{Float64}
    warm_validation_scores::Vector{Float64}
    selected_warm_index::Int
    personal_validation_scores::Matrix{Float64}
    selected_personal_indices::Vector{Int}
end

struct IDQPRunResult
    dataset::Symbol
    seed::Int
    warm_result
    personal_result
    warm_evaluation
    personal_evaluation
    tuning::IDQPTuningResult
    stage_timings::Vector{NamedTuple}
    diagnostics::Vector{NamedTuple}
    failures::Vector{NamedTuple}
    total_wall_clock_seconds::Float64
end

struct IDQPTuningError <: Exception
    phase::Symbol
    failures::Vector{NamedTuple}
end

function Base.showerror(io::IO, err::IDQPTuningError)
    print(io, "all ID-QP $(err.phase) epsilon candidates failed")
    for failure in err.failures
        failure.phase === err.phase || continue
        print(
            io,
            "\n  epsilon_multiplier=$(failure.epsilon_multiplier): $(failure.error_message)",
        )
    end
end

function _require_successful_candidate!(
    scores::AbstractArray{<:Real},
    phase::Symbol,
    failures::Vector{NamedTuple},
)
    all(!isfinite, scores) && throw(IDQPTuningError(phase, copy(failures)))
    return scores
end

function diagnostics_row(
    diagnostics::IDQPDiagnostics;
    dataset::Symbol,
    seed::Int,
    phase::Symbol,
    epsilon_multiplier::Real,
    client_id::Int,
    selected::Bool,
)
    return (
        dataset=dataset,
        seed=seed,
        phase=phase,
        epsilon_multiplier=float(epsilon_multiplier),
        client_id=client_id,
        selected=selected,
        forward_solves=diagnostics.forward_solves,
        backward_resolves=diagnostics.backward_resolves,
        differentiations=diagnostics.differentiations,
        forward_solve_seconds=diagnostics.forward_solve_seconds,
        backward_resolve_seconds=diagnostics.backward_resolve_seconds,
        differentiation_seconds=diagnostics.differentiation_seconds,
        nonfinite_solution_count=diagnostics.nonfinite_solution_count,
        nonfinite_gradient_count=diagnostics.nonfinite_gradient_count,
        failure_count=length(diagnostics.failures),
        failure_stages=join(string.(getfield.(diagnostics.failures, :stage)), ";"),
        failure_statuses=join(getfield.(diagnostics.failures, :status), ";"),
        failure_messages=join(getfield.(diagnostics.failures, :message), " | "),
    )
end

function _collect_objective_diagnostics(
    rows::Vector{NamedTuple},
    objective,
    dataset::Symbol,
    seed::Int,
    phase::Symbol,
    multiplier::Real,
    selected_clients::AbstractVector{Int},
)
    selected_set = Set(selected_clients)
    for client_id in sort!(collect(keys(objective.layers)))
        layer = objective.layers[client_id]
        push!(
            rows,
            diagnostics_row(
                layer.diagnostics;
                dataset=dataset,
                seed=seed,
                phase=phase,
                epsilon_multiplier=multiplier,
                client_id=client_id,
                selected=client_id in selected_set,
            ),
        )
    end
    return rows
end

function _failure_row(
    dataset::Symbol,
    seed::Int,
    phase::Symbol,
    multiplier::Real,
    err,
)
    return (
        dataset=dataset,
        seed=seed,
        phase=phase,
        epsilon_multiplier=float(multiplier),
        error_type=string(typeof(err)),
        error_message=sprint(showerror, err),
    )
end

function _synthetic_configs(
    seed::Int;
    warm_rounds::Int,
    personal_rounds::Int,
    warm_local_epochs::Int,
    personal_local_epochs::Int,
    batch_size::Int,
    warm_client_fraction::Float64,
    validation_max_samples_per_client::Int,
)
    common = (
        hidden_dim=64,
        batch_size=batch_size,
        validation_client_fraction=1.0,
        validation_max_samples_per_client=validation_max_samples_per_client,
        lambda0=55.0,
        kappa_lambda=0.0,
        per_client_lambda=true,
        lr0=1e-3,
        kappa_lr=0.0,
        per_client_lr=false,
        lr_lambda_alpha=0.0,
        clip_norm=1.0,
        freeze_tau=NaN,
        freeze_eps=1e-8,
        stop_after_freeze_rounds=personal_rounds,
        shuffle_batches=true,
        use_warm_start=true,
    )
    warm = SyntheticKnapsackTrainingConfig(
        ;
        common...,
        seed=seed + 1,
        rounds=warm_rounds,
        local_epochs=warm_local_epochs,
        client_fraction=warm_client_fraction,
    )
    personal = SyntheticKnapsackTrainingConfig(
        ;
        common...,
        seed=seed + 2,
        rounds=personal_rounds,
        local_epochs=personal_local_epochs,
        client_fraction=1.0,
    )
    return (warm=warm, personal=personal)
end

function run_synthetic_id_qp(
    dataset::SyntheticKnapsackDataset;
    epsilon_multipliers::AbstractVector{<:Real}=[0.05, 0.1, 0.25],
    validation_max_samples_per_client::Int=8,
    warm_rounds::Int=10,
    personal_rounds::Int=10,
    warm_local_epochs::Int=3,
    personal_local_epochs::Int=1,
    batch_size::Int=64,
    warm_client_fraction::Float64=0.4,
)
    run_started = time()
    multipliers = Float64.(epsilon_multipliers)
    isempty(multipliers) && throw(ArgumentError("epsilon multiplier grid is empty"))
    all(value -> isfinite(value) && value > 0, multipliers) ||
        throw(ArgumentError("epsilon multipliers must be positive and finite"))

    seed = dataset.config.seed
    client_data = prepare_synthetic_knapsack_client_data(dataset)
    configs = _synthetic_configs(
        seed;
        warm_rounds=warm_rounds,
        personal_rounds=personal_rounds,
        warm_local_epochs=warm_local_epochs,
        personal_local_epochs=personal_local_epochs,
        batch_size=batch_size,
        warm_client_fraction=warm_client_fraction,
        validation_max_samples_per_client=validation_max_samples_per_client,
    )
    warm_results = Any[nothing for _ in multipliers]
    warm_objectives = Any[nothing for _ in multipliers]
    warm_scores = fill(Inf, length(multipliers))
    timings = NamedTuple[]
    diagnostics = NamedTuple[]
    failures = NamedTuple[]

    for (candidate_idx, multiplier) in enumerate(multipliers)
        objective = IDQPObjective(epsilon_multiplier=multiplier)
        warm_objectives[candidate_idx] = objective
        started = time()
        try
            result = fed_synthetic_knapsack(
                dataset;
                objective=objective,
                config=configs.warm,
            )
            warm_results[candidate_idx] = result
            client_scores = [
                synthetic_validation_metrics(
                    result.model,
                    client;
                    max_samples=validation_max_samples_per_client,
                ).relative_regret for client in client_data
            ]
            warm_scores[candidate_idx] = _mean_finite(client_scores)
            elapsed = time() - started
            push!(timings, (
                dataset=:synthetic_knapsack,
                seed=seed,
                phase=:warm_start,
                epsilon_multiplier=multiplier,
                selected=false,
                wall_clock_seconds=elapsed,
                validation_score=warm_scores[candidate_idx],
            ))
        catch err
            push!(failures, _failure_row(
                :synthetic_knapsack,
                seed,
                :warm_start,
                multiplier,
                err,
            ))
            @error "ID-QP synthetic warm-start candidate failed" seed multiplier exception=(
                err,
                catch_backtrace(),
            )
        end
    end
    _require_successful_candidate!(warm_scores, :warm_start, failures)
    selected_warm_idx = _select_best_candidate(warm_scores)
    selected_warm = warm_results[selected_warm_idx]
    for idx in eachindex(timings)
        row = timings[idx]
        if row.phase === :warm_start &&
           row.epsilon_multiplier == multipliers[selected_warm_idx]
            timings[idx] = merge(row, (selected=true,))
        end
    end

    personal_results = Any[nothing for _ in multipliers]
    personal_objectives = Any[nothing for _ in multipliers]
    personal_scores = fill(Inf, length(multipliers), length(client_data))
    for (candidate_idx, multiplier) in enumerate(multipliers)
        objective = IDQPObjective(epsilon_multiplier=multiplier)
        personal_objectives[candidate_idx] = objective
        started = time()
        try
            result = local_synthetic_knapsack(
                dataset;
                objective=objective,
                config=configs.personal,
                model=selected_warm.model,
            )
            personal_results[candidate_idx] = result
            for client in client_data
                personal_client = result.clients[client.client_id]
                personal_scores[candidate_idx, client.client_id] =
                    synthetic_validation_metrics(
                        personal_client.model,
                        client;
                        max_samples=validation_max_samples_per_client,
                    ).relative_regret
            end
            elapsed = time() - started
            push!(timings, (
                dataset=:synthetic_knapsack,
                seed=seed,
                phase=:personalization,
                epsilon_multiplier=multiplier,
                selected=false,
                wall_clock_seconds=elapsed,
                validation_score=_mean_finite(view(personal_scores, candidate_idx, :)),
            ))
        catch err
            push!(failures, _failure_row(
                :synthetic_knapsack,
                seed,
                :personalization,
                multiplier,
                err,
            ))
            @error "ID-QP synthetic personalization candidate failed" seed multiplier exception=(
                err,
                catch_backtrace(),
            )
        end
    end

    for client_id in eachindex(client_data)
        _require_successful_candidate!(
            view(personal_scores, :, client_id),
            :personalization,
            failures,
        )
    end
    selected_personal_indices = [
        _select_best_candidate(view(personal_scores, :, client_id)) for
        client_id in eachindex(client_data)
    ]
    selected_clients = [
        personal_results[selected_personal_indices[client_id]].clients[client_id] for
        client_id in eachindex(client_data)
    ]
    combined_personal = LocalTrainingResult(:local_id_qp, selected_clients)
    selected_personal_candidate_indices = Set(selected_personal_indices)
    for idx in eachindex(timings)
        row = timings[idx]
        if row.phase === :personalization
            candidate_idx = findfirst(==(row.epsilon_multiplier), multipliers)
            if !isnothing(candidate_idx) && candidate_idx in selected_personal_candidate_indices
                timings[idx] = merge(row, (selected=true,))
            end
        end
    end

    for (candidate_idx, objective) in enumerate(warm_objectives)
        isnothing(objective) && continue
        selected_ids =
            candidate_idx == selected_warm_idx ? collect(eachindex(client_data)) : Int[]
        _collect_objective_diagnostics(
            diagnostics,
            objective,
            :synthetic_knapsack,
            seed,
            :warm_start,
            multipliers[candidate_idx],
            selected_ids,
        )
    end
    for (candidate_idx, objective) in enumerate(personal_objectives)
        isnothing(objective) && continue
        selected_ids = [
            client_id for client_id in eachindex(client_data) if
            selected_personal_indices[client_id] == candidate_idx
        ]
        _collect_objective_diagnostics(
            diagnostics,
            objective,
            :synthetic_knapsack,
            seed,
            :personalization,
            multipliers[candidate_idx],
            selected_ids,
        )
    end

    warm_eval = evaluate_fed_training(
        selected_warm;
        include_regularized_decision_gap=false,
    )
    personal_eval = evaluate_local_training(
        combined_personal,
        client_data;
        include_regularized_decision_gap=false,
    )
    tuning = IDQPTuningResult(
        multipliers,
        warm_scores,
        selected_warm_idx,
        personal_scores,
        selected_personal_indices,
    )
    return IDQPRunResult(
        :synthetic_knapsack,
        seed,
        selected_warm,
        combined_personal,
        warm_eval,
        personal_eval,
        tuning,
        timings,
        diagnostics,
        failures,
        time() - run_started,
    )
end

function _pjm_configs(
    seed::Int;
    warm_rounds::Int,
    personal_rounds::Int,
    warm_local_epochs::Int,
    personal_local_epochs::Int,
    batch_size::Int,
    warm_client_fraction::Float64,
    validation_max_samples_per_client::Int,
)
    common = (
        hidden_dim=64,
        batch_size=batch_size,
        validation_client_fraction=1.0,
        validation_max_samples_per_client=validation_max_samples_per_client,
        lambda0=10.0,
        kappa_lambda=0.0,
        per_client_lambda=true,
        lr0=1e-3,
        kappa_lr=0.0,
        per_client_lr=false,
        lr_lambda_alpha=0.0,
        clip_norm=1.0,
        freeze_tau=NaN,
        freeze_eps=1e-8,
        stop_after_freeze_rounds=personal_rounds,
        shuffle_batches=true,
        use_warm_start=true,
    )
    warm = PJMBatteryTrainingConfig(
        ;
        common...,
        seed=seed,
        rounds=warm_rounds,
        local_epochs=warm_local_epochs,
        client_fraction=warm_client_fraction,
    )
    personal = PJMBatteryTrainingConfig(
        ;
        common...,
        seed=seed + 1,
        rounds=personal_rounds,
        local_epochs=personal_local_epochs,
        client_fraction=1.0,
    )
    return (warm=warm, personal=personal)
end

function run_pjm_id_qp(
    dataset::PJMBatteryDataset;
    seed::Int=42,
    epsilon_multipliers::AbstractVector{<:Real}=[0.01, 0.025, 0.05],
    validation_max_samples_per_client::Int=8,
    warm_rounds::Int=10,
    personal_rounds::Int=10,
    warm_local_epochs::Int=3,
    personal_local_epochs::Int=1,
    batch_size::Int=64,
    warm_client_fraction::Float64=0.4,
)
    run_started = time()
    multipliers = Float64.(epsilon_multipliers)
    isempty(multipliers) && throw(ArgumentError("epsilon multiplier grid is empty"))
    all(value -> isfinite(value) && value > 0, multipliers) ||
        throw(ArgumentError("epsilon multipliers must be positive and finite"))

    client_data = prepare_pjm_battery_client_data(dataset)
    configs = _pjm_configs(
        seed;
        warm_rounds=warm_rounds,
        personal_rounds=personal_rounds,
        warm_local_epochs=warm_local_epochs,
        personal_local_epochs=personal_local_epochs,
        batch_size=batch_size,
        warm_client_fraction=warm_client_fraction,
        validation_max_samples_per_client=validation_max_samples_per_client,
    )
    warm_results = Any[nothing for _ in multipliers]
    warm_objectives = Any[nothing for _ in multipliers]
    warm_scores = fill(Inf, length(multipliers))
    timings = NamedTuple[]
    diagnostics = NamedTuple[]
    failures = NamedTuple[]

    for (candidate_idx, multiplier) in enumerate(multipliers)
        objective = PJMIDQPObjective(epsilon_multiplier=multiplier)
        warm_objectives[candidate_idx] = objective
        started = time()
        try
            result = fed_pjm_battery(
                dataset;
                objective=objective,
                config=configs.warm,
            )
            warm_results[candidate_idx] = result
            client_scores = [
                pjm_validation_metrics(
                    result.model,
                    client;
                    max_samples=validation_max_samples_per_client,
                ).absolute_regret for client in client_data
            ]
            warm_scores[candidate_idx] = _mean_finite(client_scores)
            elapsed = time() - started
            push!(timings, (
                dataset=:pjm,
                seed=seed,
                phase=:warm_start,
                epsilon_multiplier=multiplier,
                selected=false,
                wall_clock_seconds=elapsed,
                validation_score=warm_scores[candidate_idx],
            ))
        catch err
            push!(failures, _failure_row(:pjm, seed, :warm_start, multiplier, err))
            @error "ID-QP PJM warm-start candidate failed" seed multiplier exception=(
                err,
                catch_backtrace(),
            )
        end
    end
    _require_successful_candidate!(warm_scores, :warm_start, failures)
    selected_warm_idx = _select_best_candidate(warm_scores)
    selected_warm = warm_results[selected_warm_idx]
    for idx in eachindex(timings)
        row = timings[idx]
        if row.phase === :warm_start &&
           row.epsilon_multiplier == multipliers[selected_warm_idx]
            timings[idx] = merge(row, (selected=true,))
        end
    end

    personal_results = Any[nothing for _ in multipliers]
    personal_objectives = Any[nothing for _ in multipliers]
    personal_scores = fill(Inf, length(multipliers), length(client_data))
    for (candidate_idx, multiplier) in enumerate(multipliers)
        objective = PJMIDQPObjective(epsilon_multiplier=multiplier)
        personal_objectives[candidate_idx] = objective
        started = time()
        try
            result = local_pjm_battery(
                dataset;
                objective=objective,
                config=configs.personal,
                model=selected_warm.model,
            )
            personal_results[candidate_idx] = result
            for client in client_data
                personal_client = result.clients[client.client_id]
                personal_scores[candidate_idx, client.client_id] =
                    pjm_validation_metrics(
                        personal_client.model,
                        client;
                        max_samples=validation_max_samples_per_client,
                    ).absolute_regret
            end
            elapsed = time() - started
            push!(timings, (
                dataset=:pjm,
                seed=seed,
                phase=:personalization,
                epsilon_multiplier=multiplier,
                selected=false,
                wall_clock_seconds=elapsed,
                validation_score=_mean_finite(view(personal_scores, candidate_idx, :)),
            ))
        catch err
            push!(failures, _failure_row(:pjm, seed, :personalization, multiplier, err))
            @error "ID-QP PJM personalization candidate failed" seed multiplier exception=(
                err,
                catch_backtrace(),
            )
        end
    end

    for client_id in eachindex(client_data)
        _require_successful_candidate!(
            view(personal_scores, :, client_id),
            :personalization,
            failures,
        )
    end
    selected_personal_indices = [
        _select_best_candidate(view(personal_scores, :, client_id)) for
        client_id in eachindex(client_data)
    ]
    selected_clients = [
        personal_results[selected_personal_indices[client_id]].clients[client_id] for
        client_id in eachindex(client_data)
    ]
    combined_personal = PJMBatteryLocalTrainingResult(:local_id_qp, selected_clients)
    selected_personal_candidate_indices = Set(selected_personal_indices)
    for idx in eachindex(timings)
        row = timings[idx]
        if row.phase === :personalization
            candidate_idx = findfirst(==(row.epsilon_multiplier), multipliers)
            if !isnothing(candidate_idx) && candidate_idx in selected_personal_candidate_indices
                timings[idx] = merge(row, (selected=true,))
            end
        end
    end

    for (candidate_idx, objective) in enumerate(warm_objectives)
        isnothing(objective) && continue
        selected_ids =
            candidate_idx == selected_warm_idx ? collect(eachindex(client_data)) : Int[]
        _collect_objective_diagnostics(
            diagnostics,
            objective,
            :pjm,
            seed,
            :warm_start,
            multipliers[candidate_idx],
            selected_ids,
        )
    end
    for (candidate_idx, objective) in enumerate(personal_objectives)
        isnothing(objective) && continue
        selected_ids = [
            client_id for client_id in eachindex(client_data) if
            selected_personal_indices[client_id] == candidate_idx
        ]
        _collect_objective_diagnostics(
            diagnostics,
            objective,
            :pjm,
            seed,
            :personalization,
            multipliers[candidate_idx],
            selected_ids,
        )
    end

    warm_eval = evaluate_fed_pjm_battery(
        selected_warm;
        include_regularized_decision_gap=false,
    )
    personal_eval = evaluate_local_pjm_battery(
        combined_personal,
        client_data;
        include_regularized_decision_gap=false,
    )
    tuning = IDQPTuningResult(
        multipliers,
        warm_scores,
        selected_warm_idx,
        personal_scores,
        selected_personal_indices,
    )
    return IDQPRunResult(
        :pjm,
        seed,
        selected_warm,
        combined_personal,
        warm_eval,
        personal_eval,
        tuning,
        timings,
        diagnostics,
        failures,
        time() - run_started,
    )
end

end # module
