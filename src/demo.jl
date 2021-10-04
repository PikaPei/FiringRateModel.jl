using Plots
include("simulator.jl")


net = Network()

add_neuron(net, "E", tau=3)
add_neuron(net, "I", tau=5)

add_synapse(net, "E", "E", 1.0)
add_synapse(net, "E", "I", 0.5)
add_synapse(net, "I", "E", -0.5)  # negative sign

add_event(net, 0.1, Input, "I", 0.4, 0.0)
add_event(net, 0.1, Input, "E", 5.0, 0.0)
add_event(net, 15.0, Input, "E", 0.0, 0.0)
add_event(net, 30.0, End)


# solve
prob, tstops = gen_problem(net)
sol = solve(prob, tstops=tstops, saveat=0.1, abstol=1e-9, reltol=1e-9)


# plot
default(fontfamily="Computer Modern",
        framestyle=:box,
        dpi=150)

plot(sol)
xlabel!("Time (s)")
ylabel!("Activity")

mkpath("figures")
savefig("figures/demo.png")
