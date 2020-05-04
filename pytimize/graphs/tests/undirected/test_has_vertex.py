from ... import UndirectedGraph
from unittest import TestCase, main

class TestHasVertex(TestCase):
    def setUp(self) -> None:
        self.g = UndirectedGraph(edges={
            ("a", "b"): 1,
            ("b", "c"): 2,
            ("e", "f"): 3
        })

    def test_has_vertex(self) -> None:
        self.assertTrue(self.g.has_vertex("a"), "Should check if vertex exists.")
        self.assertTrue(self.g.has_vertex("b"), "Should check if vertex exists.")
        self.assertTrue(self.g.has_vertex("c"), "Should check if vertex exists.")
        self.assertTrue(self.g.has_vertex("e"), "Should check if vertex exists.")
        self.assertTrue(self.g.has_vertex("f"), "Should check if vertex exists.")

    def test_negative(self) -> None:
        self.assertFalse(self.g.has_vertex("w"), "Should check if vertex does not exist.")
        self.assertFalse(self.g.has_vertex("m"), "Should check if vertex does not exist.")
        self.assertFalse(self.g.has_vertex("g"), "Should check if vertex does not exist.")

if __name__ == "__main__":
    main()
