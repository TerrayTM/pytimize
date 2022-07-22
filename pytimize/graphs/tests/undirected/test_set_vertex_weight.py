from unittest import TestCase, main

from ... import UndirectedGraph


class TestSetVertexWeight(TestCase):
    def setUp(self) -> None:
        self.g = UndirectedGraph(edges={("a", "b"): 1, ("b", "c"): 2, ("e", "f"): 3})

    def test_set_vertex_weight(self) -> None:
        self.g.set_vertex_weight("a", 5)

        self.assertEqual(
            self.g.get_vertex_weight("a"), 5, "Should set vertex weight properly."
        )

    def test_invalid(self) -> None:
        with self.assertRaises(
            ValueError, msg="Should throw exception if vertex is invalid."
        ):
            self.g.set_vertex_weight("w", 10)


if __name__ == "__main__":
    main()
