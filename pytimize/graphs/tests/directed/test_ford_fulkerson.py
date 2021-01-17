from ... import DirectedGraph
from unittest import TestCase, main

class TestFordFulkerson(TestCase):
    def test_ford_fulkerson_one(self) -> None:
        g = DirectedGraph()

        g.add_arc(("s", "b"), 100)
        g.add_arc(("s", "a"), 15)
        g.add_arc(("b", "a"), 10)
        g.add_arc(("a", "t"), 100)
        g.add_arc(("b", "t"), 5)

        flow = g.ford_fulkerson("s", "t")

        self.assertEqual(flow, {
            ("s", "b"): 15,
            ("s", "a"): 15,
            ("b", "a"): 10,
            ("a", "t"): 25,
            ("b", "t"): 5
        }, "Should compute correct flow.")

    def test_ford_fulkerson_two(self) -> None:
        g = DirectedGraph()

        g.add_arc(("a", "b"), 5)
        g.add_arc(("a", "c"), 15)
        g.add_arc(("b", "d"), 5)
        g.add_arc(("b", "e"), 5)
        g.add_arc(("c", "d"), 5)
        g.add_arc(("c", "e"), 5)
        g.add_arc(("d", "f"), 15)
        g.add_arc(("e", "f"), 5)

        flow = g.ford_fulkerson("a", "f")

        self.assertEqual(flow, {
            ("a", "b"): 5,
            ("a", "c"): 10,
            ("b", "d"): 5,
            ("b", "e"): 0,
            ("c", "d"): 5,
            ("c", "e"): 5,
            ("d", "f"): 10,
            ("e", "f"): 5
        }, "Should compute correct flow.")


    def test_ford_fulkerson_three(self) -> None:
        g = DirectedGraph()

        g.add_arc(("s", "a"), 16)
        g.add_arc(("s", "c"), 13)
        g.add_arc(("c", "a"), 4)
        g.add_arc(("a", "b"), 12)
        g.add_arc(("c", "d"), 14)
        g.add_arc(("b", "c"), 9)
        g.add_arc(("d", "b"), 7)
        g.add_arc(("b", "t"), 20)
        g.add_arc(("d", "t"), 4)

        flow = g.ford_fulkerson("s", "t")

        self.assertEqual(flow, {
            ("s", "a"): 12,
            ("s", "c"): 11,
            ("c", "a"): 0,
            ("a", "b"): 12,
            ("c", "d"): 11,
            ("b", "c"): 0,
            ("d", "b"): 7,
            ("b", "t"): 19,
            ("d", "t"): 4
        }, "Should compute correct flow.")

    def test_ford_fulkerson_four(self) -> None:
        g = DirectedGraph()

        g.add_arc(("s", "a"), 5)
        g.add_arc(("s", "c"), 15)
        g.add_arc(("a", "b"), 3)
        g.add_arc(("a", "d"), 4)
        g.add_arc(("c", "b"), 4)
        g.add_arc(("c", "d"), 6)
        g.add_arc(("b", "t"), 10)
        g.add_arc(("d", "t"), 5)

        flow = g.ford_fulkerson("s", "t")

        self.assertEqual(flow, {
            ("s", "a"): 5,
            ("s", "c"): 7, 
            ("a", "b"): 3, 
            ("a", "d"): 2, 
            ("c", "b"): 4, 
            ("c", "d"): 3, 
            ("b", "t"): 7,
            ("d", "t"): 5
        }, "Should compute correct flow.")

    def test_invalid_source_and_sink(self) -> None:
        g = DirectedGraph()

        g.add_arc(("a", "b"), 10)

        with self.assertRaises(ValueError, msg="Should throw exception if invalid source."):
            g.ford_fulkerson("s", "a")

        with self.assertRaises(ValueError, msg="Should throw exception if invalid sink."):
            g.ford_fulkerson("a", "s")
        
        with self.assertRaises(ValueError, msg="Should throw exception if source equals sink."):
            g.ford_fulkerson("s", "s")

if __name__ == "__main__":
    main()
