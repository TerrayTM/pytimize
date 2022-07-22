from unittest import TestCase, main

from ... import UndirectedGraph


class TestCut(TestCase):
    def setUp(self) -> None:
        self.g = UndirectedGraph(
            edges={
                ("a", "b"): 1,
                ("b", "c"): 2,
                ("e", "f"): 3,
                ("a", "w"): 4,
                ("a", "s"): 5,
            }
        )

    def test_cut_single(self) -> None:
        self.assertEqual(
            self.g.cut("a"),
            {("a", "b"), ("a", "w"), ("a", "s")},
            "Should compute correct cut.",
        )
        self.assertEqual(
            self.g.cut("b"), {("b", "c"), ("a", "b")}, "Should compute correct cut."
        )
        self.assertEqual(self.g.cut("c"), {("b", "c")}, "Should compute correct cut.")
        self.assertEqual(self.g.cut("e"), {("e", "f")}, "Should compute correct cut.")
        self.assertEqual(self.g.cut("f"), {("e", "f")}, "Should compute correct cut.")
        self.assertEqual(self.g.cut("w"), {("a", "w")}, "Should compute correct cut.")
        self.assertEqual(self.g.cut("s"), {("a", "s")}, "Should compute correct cut.")

    def test_cut_multiple(self) -> None:
        self.assertEqual(
            self.g.cut({"a", "b"}),
            {("a", "w"), ("a", "s"), ("b", "c")},
            "Should compute correct cut.",
        )
        self.assertEqual(
            self.g.cut({"a", "s"}),
            {("a", "w"), ("a", "b")},
            "Should compute correct cut.",
        )
        self.assertEqual(
            self.g.cut({"s", "w", "a"}), {("a", "b")}, "Should compute correct cut."
        )

    def test_cut_invalid(self) -> None:
        self.assertEqual(self.g.cut({"z", "m"}), set(), "Should return empty set.")
        self.assertEqual(self.g.cut({"p", "abc"}), set(), "Should return empty set.")
        self.assertEqual(self.g.cut("z"), set(), "Should return empty set.")

    def test_cut_full(self) -> None:
        self.assertEqual(
            self.g.cut(set(self.g.vertices.keys())), set(), "Should return empty set."
        )

    def test_cut_empty(self) -> None:
        self.assertEqual(self.g.cut({}), set(), "Should return empty set.")


if __name__ == "__main__":
    main()
