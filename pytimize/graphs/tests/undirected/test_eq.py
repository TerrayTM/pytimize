from ... import UndirectedGraph
from unittest import TestCase, main

class TestEq(TestCase):
    def test_eq(self) -> None:
        edges = {
            ("a", "b"): 10,
            ("b", "c"): 20,
            ("l", "m"): 30
        }
        vertices = {
            "a": 10,
            "b": 20,
            "z": 30
        }
        g = UndirectedGraph(edges=edges, vertices=vertices)

        self.assertEqual(g, g.copy(), "Should test equality of graphs.")
        
    def test_empty(self) -> None:
        self.assertEqual(UndirectedGraph(), UndirectedGraph(), "Should compare empty graphs.")

    def test_negative(self) -> None:
        edges_one = {
            ("a", "b"): 10,
            ("b", "c"): 20,
            ("l", "m"): 30
        }
        edges_two = {
            ("a", "b"): 10,
            ("b", "c"): 20,
            ("l", "q"): 30
        }
        one = UndirectedGraph(edges=edges_one)
        two = UndirectedGraph(edges=edges_two)

        self.assertNotEqual(one, two, "Should check different graphs.")

        vertices_one = {
            "a": 2,
            "b": 3
        }
        vertices_two = {
            "a": 1,
            "b": 3
        }
        one = UndirectedGraph(vertices=vertices_one)
        two = UndirectedGraph(vertices=vertices_two)

        self.assertNotEqual(one, two, "Should check different graphs.")

        base = {
            ("a", "b"): 10,
            ("b", "c"): 20,
            ("l", "m"): 30
        }
    
        one = UndirectedGraph(edges=base)
        two = one.copy()

        two.set_vertex_weight("a", 10)

        self.assertNotEqual(one, two, "Should check different graphs.")
      
if __name__ == "__main__":
    main()
