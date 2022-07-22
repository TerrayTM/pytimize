from pytimize.graphs import DirectedGraph

# Define directed graph with arcs and their capcities
g = DirectedGraph(
    arcs={
        ("s", "c"): 15,
        ("a", "b"): 3,
        ("a", "d"): 4,
        ("c", "b"): 4,
        ("c", "d"): 6,
        ("b", "t"): 10,
        ("d", "t"): 5,
    }
)

# Compute max flow using Preflow Push algorithm
value, flow = g.preflow_push("s", "t")

print("\nPreflow Push ===========================================")
print(f"Max Flow: {value}")
print(f"Flow: {flow}")

# Compute max flow using Ford-Fulkerson algorithm
value, flow = g.ford_fulkerson("s", "t")

print("\nFord Fulkerson =========================================")
print(f"Max Flow: {value}")
print(f"Flow: {flow}")
