from pytimize.graphs import UndirectedGraph
from pytimize.parsers import GraphParser

edges = GraphParser.parse("sa:3 ab:4 bt:1 td:2 dc:2 dc:2 cb:2 ac:1 sc:4")

g = UndirectedGraph(edges)

print("Shortest Path: {}".format(g.shortest_path("s", "t")))
