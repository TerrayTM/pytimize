import numpy as np
import math

from ... import IntegerProgram
from unittest import TestCase, main

class TestBranchAndBound(TestCase):
    def test_branch_and_bound(self):
        A = np.array([
            [1.5, 1],
            [9, 11]
        ])
        b = np.array([6, 45])
        c = np.array([60, 50])
        z = 0

        p = IntegerProgram(A, b, c, z, "max", ["<=", "<="], None, [0, 1])

        #p = p.to_sef()

        #self.assertTrue(np.allclose(p.A, copy.A), "Should be the same coefficient matrix.")
        #self.assertEqual(copy.objective, "min", "Should be the same objective.")
        #self.assertFalse(copy.is_sef, "Should be the same SEF state.")
        #self.assertNotEqual(copy.objective, p.objective, "Should not copy by reference.")
        self.assertEqual(p.branch_and_bound(), [4, 0], "Should be the same solution.")

if __name__ == "__main__":
    main()