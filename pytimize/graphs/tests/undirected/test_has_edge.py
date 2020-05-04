from ... import UndirectedGraph
from unittest import TestCase, main

class TestHasEdge(TestCase):
    def setUp(self) -> None:
        self.g = UndirectedGraph(edges={
            ("a", "b"): 1,
            ("b", "c"): 2,
            ("e", "f"): 3
        })

    def test_has_edge(self) -> None:
        self.assertTrue(self.g.has_edge(("a", "b")), "Should check if edge exists.")
        self.assertTrue(self.g.has_edge(("b", "a")), "Should check if edge exists.")
        self.assertTrue(self.g.has_edge(("b", "c")), "Should check if edge exists.")
        self.assertTrue(self.g.has_edge(("c", "b")), "Should check if edge exists.")
        self.assertTrue(self.g.has_edge(("e", "f")), "Should check if edge exists.")
        self.assertTrue(self.g.has_edge(("f", "e")), "Should check if edge exists.")
        
    def test_negative(self) -> None:
        self.assertFalse(self.g.has_edge(("z", "z")), "Should check if edge does not exist.")
        self.assertFalse(self.g.has_edge(("b", "w")), "Should check if edge does not exist.")
        self.assertFalse(self.g.has_edge(("b", "f")), "Should check if edge does not exist.")

if __name__ == "__main__":
    main()
