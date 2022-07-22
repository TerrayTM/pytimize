from unittest import TestCase, main

from ... import UndirectedGraph


class TestAddEdge(TestCase):
    def test_add_edge(self) -> None:
        g = UndirectedGraph()

        g.add_edge(("a", "b"), 1)
        g.add_edge(("b", "c"), 2)
        g.add_edge(("e", "f"), 3)
        g.add_edge(("abc", "def"), 3)

        self.assertEqual(
            g.vertices,
            {"a": 0, "b": 0, "c": 0, "e": 0, "f": 0, "abc": 0, "def": 0},
            "Should add the correct vertices.",
        )
        self.assertEqual(
            g.edges,
            {("a", "b"): 1, ("b", "c"): 2, ("e", "f"): 3, ("abc", "def"): 3},
            "Should add the correct edges.",
        )
        self.assertEqual(
            g.graph,
            {
                "a": {"b"},
                "b": {"c", "a"},
                "c": {"b"},
                "e": {"f"},
                "f": {"e"},
                "abc": {"def"},
                "def": {"abc"},
            },
            "Should be the correct graph.",
        )

    def test_invalid_edge(self) -> None:
        g = UndirectedGraph()

        with self.assertRaises(
            ValueError, msg="Should throw exception if edge is invalid."
        ):
            g.add_edge(("a", "a"))

        with self.assertRaises(
            ValueError, msg="Should throw exception if edge is invalid."
        ):
            g.add_edge(("", None))

        with self.assertRaises(
            ValueError, msg="Should throw exception if edge is invalid."
        ):
            g.add_edge(("a", "b"), -100)

    def test_edge_already_exists(self) -> None:
        g = UndirectedGraph(edges={("a", "b"): 5})

        with self.assertRaises(
            ValueError, msg="Should throw exception if edge already exists."
        ):
            g.add_edge(("a", "b"))


if __name__ == "__main__":
    main()
