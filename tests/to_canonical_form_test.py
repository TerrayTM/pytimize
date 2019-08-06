import sys
import numpy as np

sys.path.append("../optimization")

from main import LinearProgram
from unittest import TestCase, main
from math import isclose

class TestToCanonicalForm(TestCase):
    def test_compute_canonical_form(self):
        A = np.array([
            [1, 1, 1, 0, 0],
            [2, 1, 0, 1, 0],
            [-1, 1, 0, 0, 1]
        ])
        b = np.array([6, 10, 4])
        c = np.array([2, 3, 0, 0, 0])
        z = 0

        expected_A = np.array([
            [1, 0, 0.5, 0, -0.5],
            [0, 1, 0.5, 0, 0.5],
            [0, 0, -1.5, 1, 0.5]
        ])
        expected_b = np.array([1, 5, 3])
        expected_c = np.array([0, 0, -2.5, 0, -0.5])
        expected_z = 17

        p = LinearProgram(A, b, c, z)
        
        result_one = p.to_canonical_form([1, 2, 4])
        
        self.assertTrue(np.allclose(result_one.A, expected_A), "Should compute correct coefficient matrix.")
        self.assertTrue(np.allclose(result_one.b, expected_b), "Should compute correct constraint values.")
        self.assertTrue(np.allclose(result_one.c, expected_c), "Should compute correct coefficient vector.")
        self.assertTrue(isclose(result_one.z, expected_z), "Should compute correct constant.")

        expected_A = np.array([
            [1, 0.5, 0, 0.5, 0],
            [0, 0.5, 1, -0.5, 0],
            [0, 1.5, 0, 0.5, 1]
        ])
        expected_b = np.array([5, 1, 9])
        expected_c = np.array([0, 2, 0, -1, 0])
        expected_z = 10

        result_two = p.to_canonical_form([1, 3, 5])
        
        self.assertTrue(np.allclose(result_two.A, expected_A), "Should compute correct coefficient matrix.")
        self.assertTrue(np.allclose(result_two.b, expected_b), "Should compute correct constraint values.")
        self.assertTrue(np.allclose(result_two.c, expected_c), "Should compute correct coefficient vector.")
        self.assertTrue(isclose(result_two.z, expected_z), "Should compute correct constant.")

    def test_invalid_basis(self):
        p = LinearProgram([[1, 2, 3], [4, 5, 6]], [1, 2], [1, 2, 3], 0)
        
        #with self.assertRaises(IndexError):
           # p.to_canonical_form([0, 1, 3]) # Check basis form square

        with self.assertRaises(IndexError):
            p.to_canonical_form([1, 5])
        
        with self.assertRaises(IndexError):
            p.to_canonical_form([0, 1])

        with self.assertRaises(IndexError):
            p.to_canonical_form([1.5, 2])

if __name__ == "__main__":
    main()
