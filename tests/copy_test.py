import sys
import numpy as np

sys.path.append("../optimization")
sys.path.append("../optimization/enums")

from main import LinearProgram
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

        p = LinearProgram(A, b, c, z, Objective.min, ["=", ">=", "<=", "=", ">=", "<=", "=", "="], [2, 3])
        
        p._steps.append("one two three")

        copy = p.copy()

        self.assertTrue(np.allclose(p.A, copy.A), "Should be the same coefficient matrix.")
        self.assertTrue(np.allclose(p.b, copy.b), "Should be the same constraint values.")
        self.assertTrue(np.allclose(p.c, copy.c), "Should be the same coefficient vector.")
        self.assertTrue(isclose(p.z, copy.z), "Should be the same constant.")
        self.assertEqual(copy.objective, Objective.min, "Should be the same objective.")
        self.assertFalse(copy.is_sef, "Should be the same SEF state.")
        self.assertEqual(copy.inequalities, ["=", ">=", "<=", "=", ">=", "<=", "=", "="], "Should be the same inequalities.")
        self.assertEqual(copy.free_variables, [2, 3], "Should be the same free variables.")
        self.assertEqual(copy.steps, ["one two three"], "Should be the same steps.")

        copy._A[[1, 2]] = -10000
        copy._b[3] = -100
        copy._c[0] = -99
        copy._z = 123456
        copy._objective = Objective.max
        copy._inequality_indices[0] = "<="
        copy._free_variables.append(0)
        copy._is_sef = True
        copy._steps.append("test")

        self.assertFalse(np.allclose(p.A, copy.A), "Should not copy by reference.")
        self.assertFalse(np.allclose(p.b, copy.b), "Should not copy by reference.")
        self.assertFalse(np.allclose(p.c, copy.c), "Should not copy by reference.")
        self.assertFalse(isclose(p.z, copy.z), "Should not copy by reference.")
        self.assertNotEqual(copy.objective, p.objective, "Should not copy by reference.")
        self.assertNotEqual(copy.inequalities, p.inequalities, "Should not copy by reference.")
        self.assertNotEqual(copy.free_variables, p.free_variables, "Should not copy by reference.")
        self.assertNotEqual(copy.is_sef, p.is_sef, "Should not copy by reference.")
        self.assertNotEqual(copy.steps, p.steps, "Should not copy by reference.")

if __name__ == "__main__":
    main()
