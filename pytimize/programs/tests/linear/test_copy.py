import numpy as np
import math

from ... import LinearProgram
from unittest import TestCase, main

class TestCopy(TestCase):
    def test_copy(self) -> None:
        A = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        b = np.array([6, 8, 10])
        c = np.array([100, 200, 300])
        z = 5

        p = LinearProgram(A, b, c, z, "min", ["=", ">=", "<="], [2, 3], [1])

        copy = p.copy()

        self.assertTrue(np.allclose(p.A, copy.A), "Should be the same coefficient matrix.")
        self.assertTrue(np.allclose(p.b, copy.b), "Should be the same constraint values.")
        self.assertTrue(np.allclose(p.c, copy.c), "Should be the same coefficient vector.")
        self.assertTrue(math.isclose(p.z, copy.z), "Should be the same constant.")
        self.assertEqual(copy.objective, "min", "Should be the same objective.")
        self.assertFalse(copy.is_sef, "Should be the same SEF state.")
        self.assertEqual(copy.inequalities, ["=", ">=", "<="], "Should be the same inequalities.")
        self.assertEqual(copy.free_variables, [2, 3], "Should be the same free variables.")
        self.assertEqual(copy.negative_variables, [1], "Should be the same negative variables.")
        self.assertEqual(copy.positive_variables, [], "Should be the same positive variables.")

        self.assertIsNot(p.A, copy.A, "Should not copy by reference.")
        self.assertIsNot(p.b, copy.b, "Should not copy by reference.")
        self.assertIsNot(p.c, copy.c, "Should not copy by reference.")
        self.assertEqual(p.inequalities, copy.inequalities, "Should not copy by reference.")
        self.assertEqual(copy.free_variables, copy.free_variables, "Should not copy by reference.")
        self.assertEqual(copy.negative_variables, copy.negative_variables, "Should not copy by reference.")
        self.assertEqual(copy.positive_variables, copy.positive_variables, "Should not copy by reference.")

if __name__ == "__main__":
    main()
