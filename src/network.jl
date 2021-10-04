using DataStructures
using Parameters


include("population.jl")
include("synapse.jl")
include("event.jl")


mutable struct Network
    neuron::OrderedDict{String,Population}
    synapse::OrderedDict{String,Vector{Synapse}}
    event::Vector{Input}
    event_end::End

    Network() = new(OrderedDict{String,Population}(),
                    OrderedDict{String,Vector{Synapse}}(),
                    Vector{Input}(),
                    End(0.0))
end

function Base.show(io::IO, net::Network)
    println(io, keys(net.neuron))
end


function reset_event(net::Network)
    empty!(net.event)
    net.event_end = End(0.0)
end


function add_neuron(net::Network, name; kwargs...)
    neuron = Population(name; kwargs...)
    net.neuron[name] = neuron
end


function set_neuron_param_all(net::Network, N::Int, tau::Real)
    for neu in values(net.neuron)
        neu.N = N
        neu.tau = tau
    end
end


function add_synapse(net::Network, pre::String, post::String, weight::Float64)
    haskey(net.neuron, post) || error("[$(post)] cannot be found in the network.")
    haskey(net.synapse, post) || begin net.synapse[post] = Vector{Synapse}() end
    push!(net.synapse[post], Synapse(pre, post, weight))
end


function add_event(net::Network, time, type::Type{Input}, population::String, args...)
    haskey(net.neuron, population) || error("[$(population)] cannot be found in the network.")
    event = Input(time, population, args...)
    push!(net.event, event)
end


function add_event(net::Network, time, type::Type{End})
    event = End(time)
    net.event_end = event
end
