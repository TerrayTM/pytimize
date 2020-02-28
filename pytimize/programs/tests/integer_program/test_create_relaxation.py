import math
import numpy as np

from ... import IntegerProgram, LinearProgram
from unittest import TestCase, main

class TestCreateRelaxation(TestCase):
    def test_create_relaxation(self) -> None:
        A = np.array([
            [1, 2, 3, 4],
            [3, 5, 7, 9]
        ])
        b = np.array([-9, 7])
        c = np.array([1, 7, 1, 5])
        z = 25

        ip = IntegerProgram(A, b, c, z, "min", ["<=", ">="], [1], [0, 1])
        lp = ip.create_relaxation()

        self.assertTrue(isinstance(lp, LinearProgram), "Should be a linear program.")
        self.assertTrue(np.allclose(lp.A, A), "Should have the same constraint matrix.")
        self.assertTrue(np.allclose(lp.b, b), "Should have the same constraint vector.")
        self.assertTrue(np.allclose(lp.c, c), "Should have the same coefficient vector.")
        self.assertTrue(math.isclose(lp.z, z), "Should have the same constant.")

        # TODO Add cases for inequalities and other data

if __name__ == "__main__":
    main()
