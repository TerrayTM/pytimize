import numpy as np

from ... import Comparator
from unittest import TestCase, main

class TestIsCloseToZero(TestCase):
    def test_zero(self):
        self.assertTrue(Comparator.is_close_to_zero(0), "Should be close to zero.")
        self.assertTrue(Comparator.is_close_to_zero(-1e-11), "Should be close to zero.")
        self.assertTrue(Comparator.is_close_to_zero(1e-11), "Should be close to zero.")
        self.assertTrue(Comparator.is_close_to_zero(np.array([0, 1e-34])).all(), "Should be close to zero.")
        self.assertTrue(Comparator.is_close_to_zero(np.array([
            [0, 1e-22],
            [2.3e-34, 1.2e-13]
        ])).all(), "Should be close to zero.")

    def test_nonzero(self):
        self.assertFalse(Comparator.is_close_to_zero(100), "Should not be close to zero.")
        self.assertFalse(Comparator.is_close_to_zero(-100), "Should not be close to zero.")
        self.assertFalse(Comparator.is_close_to_zero(np.array([4, 5])).all(), "Should not be close to zero.")
        self.assertFalse(Comparator.is_close_to_zero(np.array([
            [1e-5, 2.7e-4],
            [1e-6, 4.2e-6]
        ])).all(), "Should not be close to zero.")

if __name__ == "__main__":
    main()
