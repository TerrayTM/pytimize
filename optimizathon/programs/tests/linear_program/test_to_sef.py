import numpy as np
import math

from ... import LinearProgram
from ....enums.objective import Objective
from unittest import TestCase, main

class TestToSEF(TestCase):
    def test_to_sef(self):
        A = np.array([
            [1, 5, 3],
            [2, -1, 2],
            [1, 2, -1]
        ])
        b = np.array([5, 4, 2])
        c = np.array([1, -2, 4])
        z = 0

        p = LinearProgram(A, b, c, z, Objective.min, [">=", "<=", "="], [3])
        
        p.to_sef(in_place=True)

        self.assertTrue(np.allclose(p.A, np.array([
            [1, 5, 3, -3, -1, 0],
            [2, -1, 2, -2, 0, 1],
            [1, 2, -1, 1, 0, 0]
        ])), "Should compute correct coefficient matrix in SEF.")
        self.assertTrue(np.allclose(p.b, np.array([5, 4, 2])), "Should compute correct constraint values in SEF.")
        self.assertTrue(np.allclose(p.c, np.array([-1, 2, -4, 4, 0, 0])), "Should compute correct coefficient vector in SEF.")
        self.assertTrue(math.isclose(p.z, 0), "Should compute correct constant in SEF.")
        self.assertEqual(p.inequalities, ["=" for i in range(len(b))], "Should compute correct inequalities in SEF.")
        self.assertEqual(p.objective, Objective.max, "Should be maximizing objective function in SEF.")
        
        A = np.array([
            [1, 2, 0, 1],
            [1, -2, 16, 0],
            [8, 2, -3, 1]
        ])
        b = np.array([10, 14, -2])
        c = np.array([-2, 3, -4, 1])
        z = 0

        p = LinearProgram(A, b, c, z, Objective.max, ["<=", "<=", "="], [2])
        
        p.to_sef(in_place=True)

        self.assertTrue(np.allclose(p.A, np.array([
            [1, 2, -2, 0, 1, 1, 0],
            [1, -2, 2, 16, 0, 0, 1],
            [8, 2, -2, -3, 1, 0, 0]
        ])), "Should compute correct coefficient matrix in SEF.")
        self.assertTrue(np.allclose(p.b, np.array([10, 14, -2])), "Should compute correct constraint values in SEF.")
        self.assertTrue(np.allclose(p.c, np.array([-2, 3, -3, -4, 1, 0, 0])), "Should compute correct coefficient vector in SEF.")
        self.assertTrue(math.isclose(p.z, 0), "Should compute correct constant in SEF.")
        self.assertEqual(p.inequalities, ["=" for i in range(len(b))], "Should compute correct inequalities in SEF.")
        self.assertEqual(p.objective, Objective.max, "Should be maximizing objective function in SEF.")

if __name__ == "__main__":
    main()
