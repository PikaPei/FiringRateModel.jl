using DifferentialEquations
using ModelingToolkit
using DataStructures
using Random

include("network.jl")
include("activation.jl")


# Random.seed!(1234)

@parameters t
D = Differential(t)


mutable struct NetworkIndex
    vars::OrderedDict{String,Vector{Num}}
	ode_idx::OrderedDict{String,Int}
	param_idx::OrderedDict{String,Int}

    NetworkIndex() = new(OrderedDict{String,Vector{Num}}(),
						 OrderedDict{String,Int}(),
						 OrderedDict{String,Int}())
end

network_idx = NetworkIndex()


function gen_variables(net::Network)
    vars = OrderedDict{String,Vector{Num}}()

	param_index = OrderedDict{String,Int}()
	counter = 1

	for neu in values(net.neuron)
        _sym = Symbol(neu.name)
        vars[neu.name] = @variables $(_sym)(t)

		current = neu.name * "_I"
        _sym = Symbol(current)
        vars[current] = @parameters $(_sym)
		param_index[current] = counter
		counter += 1
	end

	network_idx.param_idx = param_index
	network_idx.vars = vars
end


function gen_ode(net::Network)
	ode_idx = OrderedDict{String,Int}()
	counter = 1

	eqs = Equation[]
	for neu in values(net.neuron)
		# synapse
		syns = []
        if haskey(net.synapse, neu.name)
            for syn in values(net.synapse[neu.name])
                push!(syns, network_idx.vars[syn.pre][1] * syn.weight)
            end
        end

		# neuron
		current = neu.name * "_I"
		eq = D(network_idx.vars[neu.name][1]) ~ 1/neu.tau * (
            -network_idx.vars[neu.name][1] + f_sqrt(+(syns..., network_idx.vars[current][1]))
            )

		push!(eqs, eq)
		ode_idx[neu.name] = counter
		counter += 1
	end

	network_idx.ode_idx = ode_idx
	return eqs
end


function callback_event(event::Input)
    condition(u, t, integrator) = (t == event.time)

    affect!(integrator) =
        integrator.p[ network_idx.param_idx[event.population*"_I"] ] = (event.mean + event.std * randn())

    DiscreteCallback(condition, affect!, save_positions=(false,false))
end


# function initialize(var::String, net::Network)
#     var in keys(net.neuron) ? net.neuron[var].rest : 0.0
# end


function gen_tstops(net::Network)
    unique([event.time for event in net.event]) |> sort
end


function gen_problem(net::Network)
	# ODE equations
	gen_variables(net)
    eqs = gen_ode(net)
    sys = ODESystem(eqs)
    f = ODEFunction(sys,
                    [network_idx.vars[ode][1] for ode in keys(network_idx.ode_idx)],
                    [network_idx.vars[p][1] for p in keys(network_idx.param_idx)])

    net.event_end.time != 0.0 || begin @error "You haven't set the end time yet!"; exit(1) end

    # Callback Functions
    event_callbacks = [callback_event(event) for event in net.event]
    cb = CallbackSet(event_callbacks...)

    # u0 = [initialize(ode, net) for ode in keys(network_idx.ode_idx)]
    u0 = repeat([0.0], length(net.neuron))
    tspan = (0.0, net.event_end.time)
    p = repeat([0.0], length(net.neuron))

    prob = ODEProblem(f, u0, tspan, p, callback=cb)
    return (prob, gen_tstops(net))
end
