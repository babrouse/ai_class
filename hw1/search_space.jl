# Import necessary libraries
using LightGraphs
using GraphPlot
using Colors

# Create a new graph with 5 nodes
g = SimpleGraph(5)

# Add edges to the graph
add_edge!(g, 1, 2)
add_edge!(g, 1, 3)
add_edge!(g, 3, 4)
add_edge!(g, 4, 5)

# Use the spring layout for node positions
node_positions = spring_layout(g)

# Draw the graph with custom positions
node_colors = [colorant"lightblue" for i in 1:nv(g)]  # color all nodes lightblue
gplot(g, nodefillc=node_colors, locs=node_positions)
