import math
import numpy as np

from ... import IntegerProgram, LinearProgram
from unittest import TestCase, main

class TestRelax(TestCase):
    def test_relax(self) -> None:
        A = np.array([
            [1, 2, 3, 4],
            [3, 5, 7, 9]
        ])
        b = np.array([-9, 7])
        c = np.array([1, 7, 1, 5])
        z = 25

        ip = IntegerProgram(A, b, c, z, "min", ["<=", ">="], [2], [1])
        lp = ip.relax()

        self.assertIsInstance(lp, LinearProgram, "Should be a linear program.")
        self.assertTrue(np.allclose(lp.A, A), "Should have the same constraint matrix.")
        self.assertTrue(np.allclose(lp.b, b), "Should have the same constraint vector.")
        self.assertTrue(np.allclose(lp.c, c), "Should have the same coefficient vector.")
        self.assertTrue(math.isclose(lp.z, z), "Should have the same constant.")
        self.assertEqual(lp.inequalities, ip.inequalities, "Should have the same inequalities.")
        self.assertEqual(lp.objective, ip.objective, "Should have the same objective.")
        self.assertEqual(lp.negative_variables, ip.negative_variables, "Should have the same negative variables.")
        self.assertEqual(lp.free_variables, ip.free_variables, "Should have the same free variables.")

if __name__ == "__main__":
    main()
