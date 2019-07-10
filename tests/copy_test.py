import sys
import numpy as np

sys.path.append("../optimization")
sys.path.append("../optimization/enums")

from main import LinearProgrammingModel
from unittest import TestCase, main
from objective import Objective
from math import isclose

class TestCopy(TestCase):
    def test_copy(self):
        A = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
            [22, 23, 24]
        ])
        b = np.array([6, 8, 10, 12, 14, 16, 18, 20])
        c = np.array([100, 200, 300])
        z = 5

        p = LinearProgrammingModel(A, b, c, z, Objective.min, ['=', '>=', '<=', '=', '>=', '<=', '=', '='])
        copy = p.copy()

        self.assertTrue(np.allclose(p.A, copy.A), "Should be the same coefficient matrix.")
        self.assertTrue(np.allclose(p.b, copy.b), "Should be the same constraints.")
        self.assertTrue(np.allclose(p.c, copy.c), "Should be the same coefficient vector.")
        self.assertTrue(isclose(p.z, copy.z), "Should be the same constant.")
        self.assertTrue(copy.objective == Objective.min, "Should be the same objective.")
        self.assertTrue(copy.inequalities == ['=', '>=', '<=', '=', '>=', '<=', '=', '='], "Should be the same inequalities.")

        #test copy is real and not by reference
        #add case for free_vars

if __name__ == "__main__":
    main()
