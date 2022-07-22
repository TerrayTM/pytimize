from unittest import TestCase, main

from ... import UndirectedGraph


class TestCopy(TestCase):
    def test_copy(self) -> None:
        edges = {("a", "b"): 10, ("b", "c"): 20, ("l", "m"): 30}
        vertices = {"a": 10, "b": 20, "z": 30}
        g = UndirectedGraph(edges=edges, vertices=vertices)
        copy = g.copy()

        self.assertEqual(copy._edges, g._edges, "Should copy edges.")
        self.assertEqual(copy._vertices, g._vertices, "Should copy vertices.")
        self.assertEqual(copy._graph, g._graph, "Should copy graph.")
        self.assertIsNot(
            copy._edges, g._edges, "Should not be the same dictionary instance."
        )
        self.assertIsNot(
            copy._vertices, g._vertices, "Should not be the same dictionary instance."
        )
        self.assertIsNot(
            copy._graph, g._graph, "Should not be the same dictionary instance."
        )


if __name__ == "__main__":
    main()
