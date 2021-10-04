abstract type AbstractEvent end


mutable struct Input <: AbstractEvent
    time::Float64
    population::String
    mean::Float64
    std::Float64
end


mutable struct End <: AbstractEvent
    time::Float64
end
