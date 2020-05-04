from ... import UndirectedGraph
from unittest import TestCase, main

class TestDelta(TestCase):
    def setUp(self) -> None:
        self.g = UndirectedGraph(edges={
            ("a", "b"): 1,
            ("b", "c"): 2,
            ("e", "f"): 3,
            ("a", "w"): 4,
            ("a", "s"): 5
        })

    def test_delta_single(self) -> None:
        self.assertEqual(self.g.delta("a"), {("a", "b"), ("a", "w"), ("a", "s")}, "Should compute correct delta.")
        self.assertEqual(self.g.delta("b"), {("b", "c"), ("a", "b")}, "Should compute correct delta.")
        self.assertEqual(self.g.delta("c"), {("b", "c")}, "Should compute correct delta.")
        self.assertEqual(self.g.delta("e"), {("e", "f")}, "Should compute correct delta.")
        self.assertEqual(self.g.delta("f"), {("e", "f")}, "Should compute correct delta.")
        self.assertEqual(self.g.delta("w"), {("a", "w")}, "Should compute correct delta.")
        self.assertEqual(self.g.delta("s"), {("a", "s")}, "Should compute correct delta.")

    def test_delta_multiple(self) -> None:
        self.assertEqual(self.g.delta({"a", "b"}), {("a", "w"), ("a", "s"), ("b", "c")}, "Should compute correct delta.")
        self.assertEqual(self.g.delta({"a", "s"}), {("a", "w"), ("a", "b")}, "Should compute correct delta.")
        self.assertEqual(self.g.delta({"s", "w", "a"}), {("a", "b")}, "Should compute correct delta.")

    def test_delta_invalid(self) -> None:
        self.assertEqual(self.g.delta({"z", "m"}), set(), "Should return empty set.")
        self.assertEqual(self.g.delta({"p", "abc"}), set(), "Should return empty set.")
        self.assertEqual(self.g.delta("z"), set(), "Should return empty set.")

    def test_delta_full(self) -> None:
        self.assertEqual(self.g.delta(set(self.g.vertices.keys())), set(), "Should return empty set.")

    def test_delta_empty(self) -> None:
        self.assertEqual(self.g.delta({}), set(), "Should return empty set.")

if __name__ == "__main__":
    main()
