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

    # # MethodError
    # sol = solve(prob, tstops=tstops)
    # sol
end


function gen_neuron_index(net)
	idx = Dict{String,Int}()
	for (index, neu) in enumerate(values(net.neuron))
		idx[neu.name] = index
	end
	return idx
end


function output_mempot(filename::String, net, sol; dt=0.0, header=true)
	# TODO: output whole table directly
	f = open(filename, "w")
	header == true && println(f, "time", " ", join([neu.name for neu in values(net.neuron)], " "))
	if dt == 0.0
		sol_round = [ [round(mp, digits=3) for mp in u] for u in sol.u ]
		for (index, time) in enumerate(sol.t)
			println(f, time, " ", join(sol_round[index], " "))
		end
	else
		for time in 0.0:dt:net.event_end.time
			sol_round = [round(mp, digits=3) for mp in sol(time)]
			println(f, time, " ", join(sol_round, " "))
		end
	end
	close(f)
end


function output_mempot(filename::String, net, sol, save_idxs; dt=0.0, header=true)
	# TODO: output whole table directly
	f = open(filename, "w")
	header == true && println(f, "time", " ", join([neu.name for neu in values(net.neuron)], " "))
	if dt == 0.0
        sol_round = [ [round(mp, digits=3) for mp in u][save_idxs] for u in sol.u ]
		for (index, time) in enumerate(sol.t)
			println(f, time, " ", join(sol_round[index], " "))
		end
	else
		for time in 0.0:dt:net.event_end.time
            sol_round = [round(mp, digits=3) for mp in sol(time)][save_idxs]
			println(f, time, " ", join(sol_round, " "))
		end
	end
	close(f)
end


function output_spike(filename::String, net; name=false)
	idx = gen_neuron_index(net)
	f = open(filename, "w")
	for spike in spikes
		if name == true
			println(f, spike.second, " ", spike.first, " ", idx[spike.first])
		else
			println(f, spike.second, " ", idx[spike.first])
		end
	end
	close(f)
end
