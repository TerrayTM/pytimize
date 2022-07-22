from unittest import TestCase, main

from ... import UndirectedGraph


class TestSetEdgeWeight(TestCase):
    def setUp(self) -> None:
        self.g = UndirectedGraph(edges={("a", "b"): 1, ("b", "c"): 2, ("e", "f"): 3})

    def test_set_edge_weight(self) -> None:
        self.g.set_edge_weight(("a", "b"), 5)

        self.assertEqual(
            self.g.get_edge_weight(("a", "b")), 5, "Should set edge weight properly."
        )

        self.g.set_edge_weight(("b", "a"), 10)

        self.assertEqual(
            self.g.get_edge_weight(("a", "b")), 10, "Should set edge weight properly."
        )

    def test_invalid(self) -> None:
        with self.assertRaises(
            ValueError, msg="Should throw exception if edge is invalid."
        ):
            self.g.set_edge_weight(("w", "s"), 10)


if __name__ == "__main__":
    main()
