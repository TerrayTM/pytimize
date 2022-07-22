from unittest import TestCase, main

from ... import UndirectedGraph


class TestIsTree(TestCase):
    def test_is_tree(self) -> None:
        g = UndirectedGraph(graph={"a": {"b", "c", "d"}})

        self.assertTrue(g.is_tree(), "Should check graph is a tree.")

        g = UndirectedGraph(
            graph={
                "a": {"b", "c", "d"},
                "b": {"f", "g"},
                "c": {"w"},
                "d": {"j", "k"},
                "j": {"p", "q"},
                "q": {"r"},
                "r": {"s"},
            }
        )

        self.assertTrue(g.is_tree(), "Should check graph is a tree.")

    def test_empty(self) -> None:
        g = UndirectedGraph()

        self.assertFalse(g.is_tree(), "Should handle empty graph.")

    def test_single_vertex(self) -> None:
        g = UndirectedGraph(vertices={"a": 0})

        self.assertTrue(g.is_tree(), "Should handle single vertex.")

    def test_disconnected(self) -> None:
        g = UndirectedGraph(vertices={"a": 0, "b": 0})

        self.assertFalse(g.is_tree(), "Should handle disconnected graph.")

        g = UndirectedGraph(
            edges={("a", "b"): 0, ("b", "c"): 0, ("d", "f"): 0, ("f", "g"): 0}
        )

        self.assertFalse(g.is_tree(), "Should handle disconnected graph.")

        g = UndirectedGraph(graph={"a": {"b", "c", "d"}, "d": {"w"}, "u": set()})

        self.assertFalse(g.is_tree(), "Should handle disconnected graph.")

    def test_cyclic(self) -> None:
        g = UndirectedGraph(graph={"a": {"b", "c"}, "b": {"d"}, "d": {"a"}})

        self.assertFalse(g.is_tree(), "Should handle cyclic graphs.")

        g = UndirectedGraph(
            graph={"a": {"b", "c"}, "c": {"d"}, "d": {"f"}, "f": {"a", "c"}}
        )

        self.assertFalse(g.is_tree(), "Should handle cyclic graphs.")

        g = UndirectedGraph(graph={"a": {"b"}, "b": {"c"}, "c": {"d"}, "d": {"a"}})

        self.assertFalse(g.is_tree(), "Should handle cyclic graphs.")


if __name__ == "__main__":
    main()
