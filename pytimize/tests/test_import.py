from unittest import TestCase, main

class TestImport(TestCase):
    def test_import_pytimize(self):
        import pytimize

    def test_import_formulations(self):
        import pytimize.formulations
        import pytimize.formulations.integer
        import pytimize.formulations.linear
        from pytimize.formulations.integer import x, variables, maximize, minimize
        from pytimize.formulations.linear import x, variables, maximize, minimize
    
    def test_import_graphs(self):
        import pytimize.graphs
        from pytimize.graphs import UndirectedGraph, DirectedGraph

    def test_import_parsers(self):
        import pytimize.parsers
        from pytimize.parsers import LinearParser, GraphParser
    
    def test_import_utilties(self):
        import pytimize.utilities
        from pytimize.utilities import DisjointSet, Comparator, typecheck

    def test_import_programs(self):
        import pytimize.programs
        from pytimize.programs import LinearProgram, IntegerProgram, NonlinearProgram
