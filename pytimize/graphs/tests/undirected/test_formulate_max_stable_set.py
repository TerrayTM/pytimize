from ... import UndirectedGraph
from unittest import TestCase, main

class TestFormulateMaxStableSet(TestCase):
    def test_formulate_max_stable_set(self) -> None:
        pass # TODO

if __name__ == "__main__":
    g = UndirectedGraph(edges={
        ("a", "b"): 1,
        ("a", "c"): 0,
        ("a", "d"): 2
    }, vertices={
        "a": 10,
        "z": 5
    })

    x, y = g.formulate_max_stable_set()
    print(x)
    print(x.relax().solve())
    print(y)

    # TODO default weight should be 1
