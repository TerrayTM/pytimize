import numpy as np

from ... import LinearProgram
from unittest import TestCase, main
from itertools import product

# Test cases for verify_infeasibility (Farkas' Lemma testing)
class TestVerifyFeasibility(TestCase):
    def setUp(self):
        self.A = np.array([
            [-2, -1, 0, -1, 0],
            [1, 1, 2, 0, -1]
        ])
        self.b = np.array([1, 3])
        self.c = np.array([2, 1, 3, 0, 0])
    
    def test_verify_infeasibility(self):
        p = LinearProgram(self.A, self.b, self.c)

        y = [-1, 0]
        self.assertTrue(p.verify_infeasibility(y), "Should output true.")
        for i in range(-5, 6):
            for j in range(-5, 6):
                if j != 0 or i >= 0:
                    self.assertFalse(p.verify_infeasibility([i, j]), "Should output false.")
    
    def test_not_sef(self):
        "Testing to see if the not-in SEF is detected."
        for ineqs in product(["<=", ">=", "="], repeat=2):
            p = LinearProgram(self.A, self.b, self.c, inequalities=ineqs)
            if ineqs != ("=", "="): # actually is SEF
                with self.assertRaises(ArithmeticError, msg="Should throw error if LP is not in SEF."):
                    p.verify_infeasibility([0, 0]) # test vector
    
    def test_always_feasible(self):
        "Testing when Ax = b is feasible."
        A = np.array([
            [2, 1, 1],
            [1, 1, 1]
        ])
        b = np.array([7, 5])
        c = np.array([1, 0, 2])

        p = LinearProgram(A, b, c)
        for y in product(range(-10, 11), repeat=2):
            self.assertFalse(p.verify_infeasibility(list(y)), "Should output false.")

if __name__ == "__main__":
    main()