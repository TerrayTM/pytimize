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

        for t in range(-50, 0):
            for u in range(2, 7):
                y = [t / u, 0]
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
    
    def test_verify_infeasibility2(self):
        "Testing more certificates for Farkas infeasibility"
        A = np.array([[6, 4], [3, 0]])
        c = np.array([1, 1]) # Doesn't really matter

        for b in product(np.arange(-5, 6, 0.25), repeat=2):
            b = [float(b[0]), float(b[1])]
            p = LinearProgram(A, np.array(b), c)
            if b[1] < 0:
                for y2 in np.arange(1, 5, 0.2):
                    y = [0, float(y2)]
                    self.assertTrue(p.verify_infeasibility(y), "Should output true.")
            
            elif b[0] - 2*b[1] < 0:
                for t in np.arange(1, 6, 0.1):
                    y = [float(t), -2 * float(t)]
                    self.assertTrue(p.verify_infeasibility(y), "Should output true.")


if __name__ == "__main__":
    main()
