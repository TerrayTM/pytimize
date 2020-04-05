import numpy as np

from ... import Comparator
from unittest import TestCase, main

class TestIsInteger(TestCase):
    def test_greater_than_equal_true(self):
        self.assertTrue(Comparator.is_close_compare(5, ">=", 3), "Should compare properly.")
        self.assertTrue(Comparator.is_close_compare(5.5, ">=", 5), "Should compare properly.")
        self.assertTrue(Comparator.is_close_compare(4.999999999, ">=", 5), "Should compare properly.")
        self.assertTrue(Comparator.is_close_compare(5.0000001111, ">=", 5), "Should compare properly.")

    def test_greater_than_equal_false(self):
        self.assertFalse(Comparator.is_close_compare(3, ">=", 5), "Should compare properly.")
        self.assertFalse(Comparator.is_close_compare(5, ">=", 5.5), "Should compare properly.")
        self.assertFalse(Comparator.is_close_compare(5, ">=", 5.01111), "Should compare properly.")

    def test_greater_than_true(self):
        self.assertTrue(Comparator.is_close_compare(5, ">", 3), "Should compare properly.")
        self.assertTrue(Comparator.is_close_compare(5.5, ">", 5), "Should compare properly.")
        self.assertTrue(Comparator.is_close_compare(5.0001, ">", 5), "Should compare properly.")
    
    def test_greater_than_false(self):
        self.assertFalse(Comparator.is_close_compare(3, ">", 5), "Should compare properly.")
        self.assertFalse(Comparator.is_close_compare(5, ">", 5.5), "Should compare properly.")
        self.assertFalse(Comparator.is_close_compare(4.999999999, ">", 5), "Should compare properly.")
        self.assertFalse(Comparator.is_close_compare(5.00000000001, ">", 5), "Should compare properly.")

    def test_less_than_equal_true(self):
        self.assertTrue(Comparator.is_close_compare(3, "<=", 5), "Should compare properly.")
        self.assertTrue(Comparator.is_close_compare(5, "<=", 5.5), "Should compare properly.")
        self.assertTrue(Comparator.is_close_compare(5, "<=", 5.01111), "Should compare properly.")
        self.assertTrue(Comparator.is_close_compare(5, "<=", 4.999999999), "Should compare properly.")
        self.assertTrue(Comparator.is_close_compare(5, "<=", 5.0000000001), "Should compare properly.")
        self.assertTrue(Comparator.is_close_compare(4.9, "<=", 5), "Should compare properly.")

    def test_less_than_equal_false(self):
        self.assertFalse(Comparator.is_close_compare(5, "<=", 3), "Should compare properly.")
        self.assertFalse(Comparator.is_close_compare(5.5, "<=", 5), "Should compare properly.")
        self.assertFalse(Comparator.is_close_compare(5.1, "<=", 5), "Should compare properly.")

    def test_less_than_true(self):
        self.assertTrue(Comparator.is_close_compare(3, "<", 5), "Should compare properly.")
        self.assertTrue(Comparator.is_close_compare(5, "<", 5.5), "Should compare properly.")
        self.assertTrue(Comparator.is_close_compare(5.49, "<", 5.5), "Should compare properly.")
    
    def test_less_than_false(self):
        self.assertFalse(Comparator.is_close_compare(5, "<", 3), "Should compare properly.")
        self.assertFalse(Comparator.is_close_compare(5.5, "<", 5), "Should compare properly.")
        self.assertFalse(Comparator.is_close_compare(5.0001, "<", 5), "Should compare properly.")
        self.assertFalse(Comparator.is_close_compare(4.99999, "<", 5), "Should compare properly.")
        self.assertFalse(Comparator.is_close_compare(5.00000000001, "<", 5), "Should compare properly.")

    def test_greater_than_equal_array_true(self):
        self.assertTrue(Comparator.is_close_compare(np.array([6, 5]), ">=", 5))
        self.assertTrue(Comparator.is_close_compare(np.array([
            [6, 7],
            [5, 5]
        ]), ">=", 5))

    def test_greater_than_equal_array_false(self):
        self.assertFalse(Comparator.is_close_compare(np.array([4, 5]), ">=", 5))
        self.assertFalse(Comparator.is_close_compare(np.array([
            [6, 7],
            [5, 4]
        ]), ">=", 5))

    def test_greater_than_array_true(self):
        self.assertTrue(Comparator.is_close_compare(np.array([6, 6]), ">", 5))
        self.assertTrue(Comparator.is_close_compare(np.array([
            [6, 7],
            [5.5, 5.1]
        ]), ">", 5))

    def test_greater_than_array_false(self):
        self.assertFalse(Comparator.is_close_compare(np.array([6, 5.00000001]), ">", 5))
        self.assertFalse(Comparator.is_close_compare(np.array([
            [6, 0.7],
            [15, 4.99999]
        ]), ">", 5))

    def test_less_than_equal_array_true(self):
        self.assertTrue(Comparator.is_close_compare(np.array([4, 5]), "<=", 5))
        self.assertTrue(Comparator.is_close_compare(np.array([
            [3, 0.5],
            [5, 5]
        ]), "<=", 5))

    def test_less_than_equal_array_false(self):
        self.assertFalse(Comparator.is_close_compare(np.array([4, 6]), "<=", 5))
        self.assertFalse(Comparator.is_close_compare(np.array([
            [6, 7],
            [5, 4]
        ]), "<=", 5))

    def test_less_than_array_true(self):
        self.assertTrue(Comparator.is_close_compare(np.array([4.9, 4]), "<", 5))
        self.assertTrue(Comparator.is_close_compare(np.array([
            [3, 4.9],
            [4, 1e-10]
        ]), "<", 5))

    def test_less_than_array_false(self):
        self.assertFalse(Comparator.is_close_compare(np.array([3, 4.99999999999999]), "<", 5))
        self.assertFalse(Comparator.is_close_compare(np.array([
            [3, 1],
            [2, 7]
        ]), "<", 5))

    def test_invalid_comparator(self):
        with self.assertRaises(ValueError, msg="Should raise invalid comparison operator error."):
            Comparator.is_close_compare(3, "=" ,5)

        with self.assertRaises(ValueError, msg="Should raise invalid comparison operator error."):
            Comparator.is_close_compare(3, "*" ,5)

if __name__ == "__main__":
    main()
