import sys
import numpy as np

sys.path.append('../optimization')

from main import LinearProgrammingModel
from unittest import TestCase, main

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

        basis = [1, 2, 4]

        p = LinearProgrammingModel(A, b, c, z)
        p.to_canonical_form(basis, in_place=True)
        
        self.assertTrue(np.array_equal(p.A, expected_A), "Should compute correct coefficient matrix.")
        self.assertTrue(np.array_equal(p.b, expected_b), "Should compute correct constraints.")
        self.assertTrue(np.array_equal(p.c, expected_c), "Should compute correct coefficient vector.")
        self.assertTrue(p.z == expected_z, "Should compute correct constant.")

    def test_invalid_basis(self):
        p = LinearProgrammingModel([[1, 2, 3], [4, 5, 6]], [1, 2], [1, 2], 0)
        
        with self.assertRaises(IndexError):
            p.to_canonical_form([0, 1, 3])

if __name__ == "__main__":
    main()