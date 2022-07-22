from unittest import TestCase, main

from ... import UndirectedGraph


class TestGetEdgeWeight(TestCase):
    def setUp(self) -> None:
        self.g = UndirectedGraph(edges={("a", "b"): 1, ("b", "c"): 2, ("e", "f"): 3})

    def test_get_weight(self) -> None:
        self.assertEqual(
            self.g.get_edge_weight(("a", "b")), 1, "Should get edge weight."
        )
        self.assertEqual(
            self.g.get_edge_weight(("b", "a")), 1, "Should get edge weight."
        )
        self.assertEqual(
            self.g.get_edge_weight(("b", "c")), 2, "Should get edge weight."
        )
        self.assertEqual(
            self.g.get_edge_weight(("c", "b")), 2, "Should get edge weight."
        )
        self.assertEqual(
            self.g.get_edge_weight(("e", "f")), 3, "Should get edge weight."
        )
        self.assertEqual(
            self.g.get_edge_weight(("f", "e")), 3, "Should get edge weight."
        )

    def test_invalid(self) -> None:
        with self.assertRaises(
            ValueError, msg="Should throw exception if edge is invalid."
        ):
            self.g.get_edge_weight(("b", "g"))

        with self.assertRaises(
            ValueError, msg="Should throw exception if edge is invalid."
        ):
            self.g.get_edge_weight(("b", "b"))


if __name__ == "__main__":
    main()
