from ... import UndirectedGraph
from unittest import TestCase, main

class TestGetVertexWeight(TestCase):
    def setUp(self) -> None:
        self.g = UndirectedGraph(edges={
            ("a", "b"): 1,
            ("b", "c"): 2,
            ("e", "f"): 3
        }, vertices={
            "a": 10,
            "b": 20,
            "c": 30,
            "e": 40
        })

    def test_get_weight(self) -> None:
        self.assertEqual(self.g.get_vertex_weight("a"), 10, "Should get vertex weight.")
        self.assertEqual(self.g.get_vertex_weight("b"), 20, "Should get vertex weight.")
        self.assertEqual(self.g.get_vertex_weight("c"), 30, "Should get vertex weight.")
        self.assertEqual(self.g.get_vertex_weight("e"), 40, "Should get vertex weight.")
        self.assertEqual(self.g.get_vertex_weight("f"), 0, "Should get vertex weight.")
       
    def test_invalid(self) -> None:
        with self.assertRaises(ValueError, msg="Should throw exception if vertex is invalid."):
            self.g.get_vertex_weight("w")
        
        with self.assertRaises(ValueError, msg="Should throw exception if vertex is invalid."):
            self.g.get_vertex_weight("vertex")

if __name__ == "__main__":
    main()
