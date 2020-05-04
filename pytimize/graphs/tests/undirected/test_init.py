from ... import UndirectedGraph
from unittest import TestCase, main

class TestInit(TestCase):
    def test_init_empty(self) -> None:
        g = UndirectedGraph()
        self.assertEqual(g.graph, {}, "Should initialize with correct graph.")
        self.assertEqual(g.edges, {}, "Should initialize with correct edges.")
        self.assertEqual(g.vertices, {}, "Should initialize with correct vertices.")
        
    def test_init_graph(self) -> None:
        g = UndirectedGraph(graph={
            "a": {"b", "c"},
            "w": {"t", "b"},
            "b": {"z"}
        })
        self.assertEqual(g.graph, {"a": {"c", "b"}, "b": {"w", "z", "a"}, "c": {"a"}, "t": {"w"}, "w": {"b", "t"}, "z": {"b"}}, "Should initialize with correct graph.")
        self.assertEqual(g.edges, {("a", "b"): 0, ("a", "c"): 0, ("b", "w"): 0, ("b", "z"): 0, ("t", "w"): 0}, "Should initialize with correct edges.")
        self.assertEqual(g.vertices, {"a": 0, "b": 0, "c": 0, "t": 0, "w": 0, "z": 0}, "Should initialize with correct vertices.")

        g = UndirectedGraph(graph=g.graph)
        self.assertEqual(g.graph, {"a": {"c", "b"}, "b": {"w", "z", "a"}, "c": {"a"}, "t": {"w"}, "w": {"b", "t"}, "z": {"b"}}, "Should initialize with correct graph.")
        self.assertEqual(g.edges, {("a", "b"): 0, ("a", "c"): 0, ("b", "w"): 0, ("b", "z"): 0, ("t", "w"): 0}, "Should initialize with correct edges.")
        self.assertEqual(g.vertices, {"a": 0, "b": 0, "c": 0, "t": 0, "w": 0, "z": 0}, "Should initialize with correct vertices.")

    def test_init_edges(self) -> None:
        g = UndirectedGraph(edges={
            ("a", "b"): 1, 
            ("a", "c"): 2, 
            ("b", "w"): 3, 
            ("b", "z"): 4, 
            ("t", "w"): 5
        })
        self.assertEqual(g.graph, {"a": {"c", "b"}, "b": {"w", "z", "a"}, "c": {"a"}, "t": {"w"}, "w": {"b", "t"}, "z": {"b"}}, "Should initialize with correct graph.")
        self.assertEqual(g.edges, {("a", "b"): 1, ("a", "c"): 2, ("b", "w"): 3, ("b", "z"): 4, ("t", "w"): 5}, "Should initialize with correct edges.")
        self.assertEqual(g.vertices, {"a": 0, "b": 0, "c": 0, "t": 0, "w": 0, "z": 0}, "Should initialize with correct vertices.")

    def test_init_vertices(self) -> None:
        g = UndirectedGraph(vertices={
            "a": 1, 
            "b": 2, 
            "c": 3, 
            "t": 4, 
            "w": 5, 
            "z": 6
        })
        self.assertEqual(g.graph, {"a": set(), "b": set(), "c": set(), "t": set(), "w": set(), "z": set()}, "Should initialize with correct graph.")
        self.assertEqual(g.edges, {}, "Should initialize with correct edges.")
        self.assertEqual(g.vertices, {"a": 1, "b": 2, "c": 3, "t": 4, "w": 5, "z": 6}, "Should initialize with correct vertices.")

    def test_init_edges_vertices(self) -> None:
        g = UndirectedGraph(edges={
            ("a", "b"): 5
        }, vertices={
            "a": 10,
            "c": 12
        })
        self.assertEqual(g.graph, {"a": {"b"}, "b": {"a"}, "c": set()}, "Should initialize with correct graph.")
        self.assertEqual(g.edges, {("a", "b"): 5}, "Should initialize with correct edges.")
        self.assertEqual(g.vertices, {"a": 10, "b": 0, "c": 12}, "Should initialize with correct vertices.")

    def test_init_graph_vertices(self) -> None:
        g = UndirectedGraph(graph={
            "a": {"b", "d"}
        }, vertices={
            "a": 5,
            "c": 5
        })
        self.assertEqual(g.graph, {"a": {"b", "d"}, "b": {"a"}, "d": {"a"}, "c": set()}, "Should initialize with correct graph.")
        self.assertEqual(g.edges, {("a", "b"): 0, ("a", "d"): 0}, "Should initialize with correct edges.")
        self.assertEqual(g.vertices, {"a": 5, "b": 0, "d": 0, "c": 5}, "Should initialize with correct vertices.")

    def test_init_graph_edges(self) -> None:
        g = UndirectedGraph(graph={
            "a": {"b", "d"}
        }, edges={
            ("a", "b"): 10,
            ("w", "x"): 20
        })
        self.assertEqual(g.graph, {"a": {"b", "d"}, "b": {"a"}, "d": {"a"}, "w": {"x"}, "x": {"w"}}, "Should initialize with correct graph.")
        self.assertEqual(g.edges, {("a", "b"): 10, ("w", "x"): 20, ("a", "d"): 0}, "Should initialize with correct edges.")
        self.assertEqual(g.vertices, {"a": 0, "b": 0, "d": 0, "w": 0, "x": 0}, "Should initialize with correct vertices.")

    def test_init_graph_edges_vertices(self) -> None:
        g = UndirectedGraph(graph={
            "a": {"b"},
            "b": {"a"},
            "y": {"w"}
        }, edges={
            ("a", "b"): 5,
            ("w", "x"): 20
        }, vertices={
            "a": 10,
            "c": 11
        })
        self.assertEqual(g.graph, {"a": {"b"}, "b": {"a"}, "c": set(), "w": {"y", "x"}, "x": {"w"}, "y": {"w"}}, "Should initialize with correct graph.")
        self.assertEqual(g.edges, {("a", "b"): 5, ("w", "x"): 20, ("w", "y"): 0}, "Should initialize with correct edges.")
        self.assertEqual(g.vertices, {"a": 10, "b": 0, "c": 11, "w": 0, "x": 0, "y": 0}, "Should initialize with correct vertices.")

if __name__ == "__main__":
    main()
