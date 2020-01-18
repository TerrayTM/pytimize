from ... import UndirectedGraph
from unittest import TestCase, main

class TestShortestPath(TestCase):
    def test_shortest_path(self):
        graph = UndirectedGraph()

        graph.add_edge(("s", "a"), 6)
        graph.add_edge(("a", "t"), 4)
        graph.add_edge("t", "b", 5)
        graph.add_edge("s", "b", 2)
        graph.add_edge("c", "s", 4)
        graph.add_edge("b", "c", 1)
        graph.add_edge("a", "c", 1)
        graph.add_edge("c", "t", 2)

        graph.shortest_path("s", "t")

        graph = UndirectedGraph()

        graph.add_edge("s", "a", 3)
        graph.add_edge("a", "b", 4)
        graph.add_edge("b", "t", 1)
        graph.add_edge("t", "d", 2)
        graph.add_edge("d", "c", 2)
        graph.add_edge("c", "b", 2)
        graph.add_edge("a", "c", 1)
        graph.add_edge("s", "c", 4)

        print(graph.shortest_path("s", "t"))

        graph = UndirectedGraph()

        graph.add_edge("a","c",6)
        graph.add_edge("a","b",4)
        graph.add_edge("b","c",2)
        graph.add_edge("c","g",5)
        graph.add_edge("b","g",3)
        graph.add_edge("c","d",1)
        graph.add_edge("d","g",2)
        graph.add_edge("d","e",5)
        graph.add_edge("e","f",6)
        graph.add_edge("f","d",1)
        graph.add_edge("f","g",4)
        graph.add_edge("g","h",5)
        graph.add_edge("h","i",3)
        graph.add_edge("i","g",2)
        graph.add_edge("j","i",9)
        graph.add_edge("j","k",6)
        graph.add_edge("k","g",4)
        graph.add_edge("k","l",1)
        graph.add_edge("l","b",5)

        print(graph.shortest_path("j", "c")) #["jk", "kg", "gd", "dc"]

        print(graph.shortest_path("g", "e")) #["gd", "de"]
        print(graph.shortest_path("a", "c")) #["gd", "de"]
