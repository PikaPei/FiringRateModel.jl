@with_kw mutable struct Population{T<:Real}
    name::String
    N::Int = 1
    tau::T = 1
end


Population(name; kwargs...) = Population(name=name; kwargs...)
