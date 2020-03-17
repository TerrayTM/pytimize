import numpy as np

from ... import Comparator
from unittest import TestCase, main

class TestIsNegative(TestCase):
    def test_negative(self):
        self.assertTrue(Comparator.is_negative(-6), "Should test negative.")
        self.assertTrue(Comparator.is_negative(-0.1), "Should test negative.")
        self.assertTrue(Comparator.is_negative(np.array([-1, -0.01])), "Should test negative.")
        self.assertTrue(Comparator.is_negative(np.array([
            [-1, -1, -0.15],
            [-23, -0.5, -3]
        ])), "Should test negative.")

    def test_zero(self):
        self.assertFalse(Comparator.is_negative(0), "Should return false for zero.")
        self.assertFalse(Comparator.is_negative(np.array([0, -0.01])), "Should return false for zero.")
        self.assertFalse(Comparator.is_negative(np.array([
            [0, -3],
            [-5, -10]
        ])), "Should return false for zero.")

    def test_positive(self):
        self.assertFalse(Comparator.is_negative(5), "Should return false positive numbers.")
        self.assertFalse(Comparator.is_negative(0.04), "Should return false positive numbers.")
        self.assertFalse(Comparator.is_negative(np.array([3, 0.03])), "Should return false positive numbers.")
        self.assertFalse(Comparator.is_negative(np.array([
            [0.2, 3],
            [5, 1]
        ])), "Should return false positive numbers.")

    def test_close_to_zero(self):
        self.assertFalse(Comparator.is_negative(-1e-12), "Should return false for numbers close to zero.")

    def test_mixed_array(self):
        self.assertFalse(Comparator.is_negative(np.array([
            [1, -3],
            [-4, -6]
        ])), "Should return false if any element is positive")

if __name__ == "__main__":
    main()
