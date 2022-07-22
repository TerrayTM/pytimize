from unittest import TestCase, main

import numpy as np

from ... import Comparator


class TestIsInteger(TestCase):
    def test_integer(self) -> None:
        self.assertTrue(Comparator.is_integer(6), "Should be close to an integer.")
        self.assertTrue(Comparator.is_integer(-5), "Should be close to an integer.")
        self.assertTrue(
            Comparator.is_integer(5.00000000001), "Should be close to an integer."
        )
        self.assertTrue(
            Comparator.is_integer(np.array([0, 1e-34])),
            "Should be close to an integer.",
        )
        self.assertTrue(
            Comparator.is_integer(np.array([[39, 9.000000000023], [2.3e-34, 1.2e-13]])),
            "Should be close to an integer.",
        )

    def test_noninteger(self) -> None:
        self.assertFalse(
            Comparator.is_integer(4.3), "Should not be close to an integer."
        )
        self.assertFalse(
            Comparator.is_integer(100000.123), "Should not be close to an integer."
        )
        self.assertFalse(
            Comparator.is_integer(100000000.003), "Should not be close to an integer."
        )
        self.assertFalse(
            Comparator.is_integer(-9.55), "Should not be close to an integer."
        )
        self.assertFalse(
            Comparator.is_integer(np.array([4.003, 5.9])),
            "Should not be close to an integer.",
        )
        self.assertFalse(
            Comparator.is_integer(np.array([100000000.1, 100.1, 10.1])),
            "Should not be close to an integer.",
        )
        self.assertFalse(
            Comparator.is_integer(np.array([[1e-6, 4.92], [1e-2, 6.99]])),
            "Should not be close to an integer.",
        )


if __name__ == "__main__":
    main()
