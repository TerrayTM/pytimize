import sys
import numpy as np

sys.path.append("../optimization")

from main import LinearProgram
from unittest import TestCase, main
from math import isclose

class TestToCanonicalForm(TestCase):
    def setUp(self):
        self.A = np.array([
            [1, 1, 1, 0, 0],
            [2, 1, 0, 1, 0],
            [-1, 1, 0, 0, 1]
        ])
        self.b = np.array([6, 10, 4])
        self.c = np.array([2, 3, 0, 0, 0])
        self.z = 0

    def test_compute_canonical_form(self):
        expected_A = np.array([
            [1, 0, 0.5, 0, -0.5],
            [0, 1, 0.5, 0, 0.5],
            [0, 0, -1.5, 1, 0.5]
        ])
        expected_b = np.array([1, 5, 3])
        expected_c = np.array([0, 0, -2.5, 0, -0.5])
        expected_z = 17

        p = LinearProgram(self.A, self.b, self.c, self.z)
        
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
        
        with self.assertRaises(ValueError, msg="Should throw error for invalid basis."):
            p.to_canonical_form([0, 1, 3])

        with self.assertRaises(IndexError, msg="Should throw error for invalid basis."):
            p.to_canonical_form([1, 5])
        
        with self.assertRaises(IndexError, msg="Should throw error for invalid basis."):
            p.to_canonical_form([0, 1])

        with self.assertRaises(IndexError, msg="Should throw error for invalid basis."):
            p.to_canonical_form([1.5, 2])

        with self.assertRaises(ValueError, msg="Should throw error for invalid basis."):
            p.to_canonical_form([2, 2])

    def test_non_sef(self):
        p = LinearProgram(self.A, self.b, self.c, self.z, free_variables=[1, 3])

        with self.assertRaises(ArithmeticError, msg="Should throw error for non-SEF linear programs."):
            p.to_canonical_form([1, 2, 3])

    def test_steps(self):
        p = LinearProgram(self.A, self.b, self.c, self.z)
        p.to_canonical_form([1, 2, 4], in_place=True)
        
        expected_steps = ["1.01", "1.02", "1.03", "1.04"] #complete this

        for step in p.steps:
            if step["key"] in expected_steps:
                expected_steps.remove(step["key"])

        self.assertEqual(len(expected_steps), 0, "Should display correct steps.")

if __name__ == "__main__":
    main()
