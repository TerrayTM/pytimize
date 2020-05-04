from ... import UndirectedGraph
from unittest import TestCase, main

class TestDegree(TestCase):
    def setUp(self) -> None:
        self.g = UndirectedGraph(edges={
            ("a", "b"): 1,
            ("b", "c"): 2,
            ("e", "f"): 3,
            ("a", "w"): 4,
            ("a", "s"): 5
        }, vertices={
            "u": 10
        })

    def test_degree_single(self) -> None:
        self.assertEqual(self.g.degree("a"), 3, "Should compute correct degree.")
        self.assertEqual(self.g.degree("b"), 2, "Should compute correct degree.")
        self.assertEqual(self.g.degree("c"), 1, "Should compute correct degree.")
        self.assertEqual(self.g.degree("e"), 1, "Should compute correct degree.")
        self.assertEqual(self.g.degree("f"), 1, "Should compute correct degree.")
        self.assertEqual(self.g.degree("w"), 1, "Should compute correct degree.")
        self.assertEqual(self.g.degree("s"), 1, "Should compute correct degree.")
        self.assertEqual(self.g.degree("u"), 0, "Should compute correct degree.")

    def test_degree_multiple(self) -> None:
        self.assertEqual(self.g.degree({"a", "b"}), 3, "Should compute correct degree.")
        self.assertEqual(self.g.degree({"a", "s"}), 2, "Should compute correct degree.")
        self.assertEqual(self.g.degree({"s", "w", "a"}), 1, "Should compute correct degree.")

    def test_degree_invalid(self) -> None:
        self.assertEqual(self.g.degree({"z", "m"}), 0, "Should return 0 degrees.")
        self.assertEqual(self.g.degree({"p", "abc"}), 0, "Should return 0 degrees.")
        self.assertEqual(self.g.degree("z"), 0, "Should return 0 degrees.")

    def test_degree_full(self) -> None:
        self.assertEqual(self.g.degree(set(self.g.vertices.keys())), 0, "Should return 0 degrees.")

    def test_degree_empty(self) -> None:
        self.assertEqual(self.g.degree({}), 0, "Should return 0 degrees.")

if __name__ == "__main__":
    main()
