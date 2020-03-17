import numpy as np

from ... import Comparator
from unittest import TestCase, main

class TestIsPositive(TestCase):
    def test_negative(self):
        self.assertFalse(Comparator.is_positive(-6), "Should be false if negative.")
        self.assertFalse(Comparator.is_positive(-0.1), "Should be false if negative.")
        self.assertFalse(Comparator.is_positive(np.array([-1, -0.01])), "Should be false if negative.")
        self.assertFalse(Comparator.is_positive(np.array([
            [-1, -1, -0.15],
            [-23, -0.5, -3]
        ])), "Should be false if negative.")

    def test_zero(self):
        self.assertFalse(Comparator.is_positive(0), "Should return false for zero.")
        self.assertFalse(Comparator.is_positive(np.array([0, -0.01])), "Should return false for zero.")
        self.assertFalse(Comparator.is_positive(np.array([
            [0, -3],
            [-5, -10]
        ])), "Should return false for zero.")

    def test_positive(self):
        self.assertTrue(Comparator.is_positive(5), "Should test positive.")
        self.assertTrue(Comparator.is_positive(0.04), "Should test positive.")
        self.assertTrue(Comparator.is_positive(np.array([3, 0.03])), "Should test positive.")
        self.assertTrue(Comparator.is_positive(np.array([
            [0.2, 3],
            [5, 1]
        ])),  "Should test positive.")

    def test_close_to_zero(self):
        self.assertFalse(Comparator.is_positive(-1e-12), "Should return false for numbers close to zero.")

    def test_mixed_array(self):
        self.assertFalse(Comparator.is_positive(np.array([
            [1, -3],
            [-4, -6]
        ])), "Should return false if any element is negative")

if __name__ == "__main__":
    main()
