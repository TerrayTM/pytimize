import numpy as np

from ...linear import maximize, minimize, x, variables
from ....programs import LinearProgram
from unittest import TestCase, main

class TestIntegration(TestCase):
    def test_max_program(self):
        p = maximize(x[1] + x[2]).subject_to(
            x[1] >= 3,
            x[2] <= 5
        ).compile()

        self.assertTrue(isinstance(p, LinearProgram))

    def test_named_variables(self):
        pass

if __name__ == "__main__":
    main()
