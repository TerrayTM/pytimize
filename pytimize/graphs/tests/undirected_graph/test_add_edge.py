from ... import UndirectedGraph
from unittest import TestCase, main

class TestAddEdge(TestCase):
    def test_add_edge(self) -> None:
        g = UndirectedGraph()

        g.add_edge(("a", "b"), 1)
        g.add_edge(("b", "c"), 2)
        g.add_edge(("e", "f"), 3)
        g.add_edge(("123456", "789456abc"), 3)

        self.assertEqual(g.vertices, ["a", "b", "c", "e", "f", "123456", "789456abc"], "Should add the correct vertices.")
        self.assertTrue(("f", "e", 3) in g.edges, "Should add the correct edges.")
        self.assertTrue(("b", "a", 1) in g.edges, "Should add the correct edges.")
        self.assertTrue(("c", "b", 2) in g.edges, "Should add the correct edges.")
        self.assertTrue(("789456abc", "123456", 3) in g.edges, "Should add the correct edges.")
        self.assertEqual(len(g.edges), 4, "Should add the correct edges.")

    def test_invalid_edge(self) -> None:
        g = UndirectedGraph()

        with self.assertRaises(ValueError, msg="Should throw exception if edge is invalid."):
            g.add_edge(("a", "a"))

        with self.assertRaises(ValueError, msg="Should throw exception if edge is invalid."):
            g.add_edge((2, 5))
        
        with self.assertRaises(ValueError, msg="Should throw exception if edge is invalid."):
            g.add_edge(("", None))

        with self.assertRaises(ValueError, msg="Should throw exception if edge is invalid."):
            g.add_edge(("a", "b"), -100)

    def test_edge_already_exists(self) -> None:
        g = UndirectedGraph([("a", "b", 5)])

        with self.assertRaises(ValueError, msg="Should throw exception if edge already exists."):
            g.add_edge(("a", "b"))

if __name__ == "__main__":
    main()
