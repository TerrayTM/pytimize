from unittest import TestCase, main


class TestImport(TestCase):
    def test_import_pytimize(self) -> None:
        import pytimize

    def test_import_formulations(self) -> None:
        import pytimize.formulations
        import pytimize.formulations.integer
        import pytimize.formulations.linear
        from pytimize.formulations.integer import (maximize, minimize,
                                                   variables, x)
        from pytimize.formulations.linear import (maximize, minimize,
                                                  variables, x)

    def test_import_graphs(self) -> None:
        import pytimize.graphs
        from pytimize.graphs import DirectedGraph, UndirectedGraph

    def test_import_parsers(self) -> None:
        import pytimize.parsers
        from pytimize.parsers import GraphParser, LinearParser

    def test_import_utilties(self) -> None:
        import pytimize.utilities
        from pytimize.utilities import Comparator, DisjointSet

    def test_import_programs(self) -> None:
        import pytimize.programs
        from pytimize.programs import (IntegerProgram, LinearProgram,
                                       NonlinearProgram, UnconstrainedProgram)
