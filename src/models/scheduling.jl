using ParameterSchedulers

"""
    InverseTimeScheduler(start, kappa)

Inverse-time schedule

`value_t = start / (1 + kappa * (t - 1))`

with 1-based steps so the first emitted value is exactly `start`.
"""
struct InverseTimeScheduler{T} <: ParameterSchedulers.AbstractSchedule{false}
    start::T
    kappa::T
end

function InverseTimeScheduler(start::Real, kappa::Real)
    start > 0 || throw(ArgumentError("start must be positive"))
    kappa >= 0 || throw(ArgumentError("kappa must be nonnegative"))
    T = promote_type(Float64, typeof(float(start)), typeof(float(kappa)))
    return InverseTimeScheduler{T}(T(start), T(kappa))
end

(s::InverseTimeScheduler)(step::Integer) = s.start / (one(s.start) + s.kappa * (step - 1))

Base.eltype(::Type{<:InverseTimeScheduler{T}}) where {T} = T

"""
    GatedScheduler(schedule; frozen=Ref(false))

Small stateful scheduler wrapper used by the Phase 3 training loops. The
wrapped schedule advances only while `frozen[] == false`.
"""
mutable struct GatedScheduler{S,T}
    schedule::S
    step::Int
    value::T
    frozen::Base.RefValue{Bool}
end

function GatedScheduler(schedule::S; frozen::Base.RefValue{Bool}=Ref(false)) where {S}
    return GatedScheduler{S,eltype(S)}(schedule, 0, schedule(1), frozen)
end

function next_schedule_value!(scheduler::GatedScheduler)
    if scheduler.step == 0
        scheduler.step = 1
        scheduler.value = scheduler.schedule(1)
    elseif !scheduler.frozen[]
        scheduler.step += 1
        scheduler.value = scheduler.schedule(scheduler.step)
    end
    return scheduler.value
end

current_schedule_value(scheduler::GatedScheduler) = scheduler.value

function reset_schedule!(scheduler::GatedScheduler)
    scheduler.step = 0
    scheduler.value = scheduler.schedule(1)
    return scheduler
end

function create_inverse_time_scheduler(
    start::Real,
    kappa::Real;
    frozen::Base.RefValue{Bool}=Ref(false),
)
    return GatedScheduler(InverseTimeScheduler(start, kappa); frozen=frozen)
end
