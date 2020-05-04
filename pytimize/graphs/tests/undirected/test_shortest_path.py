from ... import UndirectedGraph
from ....parsers import GraphParser
from unittest import TestCase, main

class TestShortestPath(TestCase):
    import unittest
    @unittest.skip("Refactor to new graph standard.")
    def test_shortest_path(self) -> None:
        g = UndirectedGraph(GraphParser.parse("sa:6 at:4 tb:5 sb:2 cs:4 bc:1 ac:1 ct:2"))
        
        self.assertEqual(g.shortest_path("s", "t"), [("s", "b"), ("b", "c"), ("c", "t")], "Should compute correct shortest path.")

        g = UndirectedGraph(GraphParser.parse("sa:3 ab:4 bt:1 td:2 dc:2 cb:2 ac:1 sc:5"))
        
        self.assertEqual(g.shortest_path("s", "t"), [("s", "a"), ("a", "c"), ("c", "b"), ("b", "t")], "Should compute correct shortest path.")

        g = UndirectedGraph(GraphParser.parse("ac:7 ab:4 bc:2 cg:5 bg:3 cd:1 dg:2 de:5 ef:6 fd:1 fg:4 gh:5 hi:3 ig:2 ji:9 jk:6 kg:4 kl:1 lb:5"))
        
        self.assertEqual(g.shortest_path("j", "c"), [("j", "k"), ("k", "g"), ("g", "d"), ("d", "c")], "Should compute correct shortest path.")
        self.assertEqual(g.shortest_path("g", "e"), [("g", "d"), ("d", "e")], "Should compute correct shortest path.")
        self.assertEqual(g.shortest_path("a", "c"), [("a", "b"), ("b", "c")], "Should compute correct shortest path.")

if __name__ == "__main__":
    main()
