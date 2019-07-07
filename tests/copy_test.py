import sys
import numpy as np

sys.path.append("../optimization")

from main import LinearProgrammingModel
from unittest import TestCase, main
from math import isclose

class TestCopy(TestCase):
    def test_copy(self):
        A = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        b = np.array([6, 8, 10])
        c = np.array([100, 200, 300])
        z = 5
        # Check for objective, free vars, and symbols

        p = LinearProgrammingModel(A, b, c, z)
        copy = p.copy()

        self.assertTrue(np.allclose(p.A, copy.A), "Should be the same coefficient matrix.")
        self.assertTrue(np.allclose(p.b, copy.b), "Should be the same constraints.")
        self.assertTrue(np.allclose(p.c, copy.c), "Should be the same coefficient vector.")
        self.assertTrue(isclose(p.z, copy.z), "Should be the same constant.")

if __name__ == "__main__":
    main()
