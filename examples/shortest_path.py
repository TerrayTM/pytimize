from optimizathon.graphs import UndirectedGraph

g = UndirectedGraph()

g.add_edge("s", "a", 3)
g.add_edge("a", "b", 4)
g.add_edge("b", "t", 1)
g.add_edge("t", "d", 2)
g.add_edge("d", "c", 2)
g.add_edge("c", "b", 2)
g.add_edge("a", "c", 1)
g.add_edge("s", "c", 4)

print("Shortest Path: {}".format(graph.shortest_path("s", "t")))
