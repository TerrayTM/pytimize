import math
from unittest import TestCase, main

import numpy as np

from ... import LinearProgram


class TestInit(TestCase):
    def setUp(self):
        self.A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.b = np.array([6, 15, 24])
        self.c = np.array([100, 200, 300])
        self.z = 5

    def test_init(self):
        inequalities = np.array(["=", "=", "="])

        p = LinearProgram(self.A, self.b, self.c, self.z, "min", inequalities)

        self.assertTrue(
            np.allclose(p.A, self.A), "Should construct coefficient matrix."
        )
        self.assertTrue(np.allclose(p.b, self.b), "Should construct constraint values.")
        self.assertTrue(
            np.allclose(p.c, self.c), "Should construct coefficient vector."
        )
        self.assertTrue(math.isclose(p.z, self.z), "Should construct constant.")
        self.assertEqual(p.objective, "min", "Should construct objective.")
        self.assertEqual(
            p.inequalities, ["=", "=", "="], "Should construct inequalities."
        )
        self.assertFalse(p.is_sef, "Should detect non-SEF.")

        self.assertTrue(
            np.issubdtype(p.A.dtype, np.floating), "Should be of type float."
        )
        self.assertTrue(
            np.issubdtype(p.b.dtype, np.floating), "Should be of type float."
        )
        self.assertTrue(
            np.issubdtype(p.c.dtype, np.floating), "Should be of type float."
        )
        self.assertIn(type(p.z), [float, int], "Should be of type float or int.")
        self.assertIsInstance(p.objective, str, "Should be of type string.")
        self.assertIsInstance(p.inequalities, list, "Should be of type list.")

        p = LinearProgram(self.A.tolist(), self.b.tolist(), self.c.tolist(), self.z)

        self.assertTrue(
            np.allclose(p.A, self.A), "Should construct coefficient matrix."
        )
        self.assertTrue(np.allclose(p.b, self.b), "Should construct constraint values.")
        self.assertTrue(
            np.allclose(p.c, self.c), "Should construct coefficient vector."
        )
        self.assertTrue(math.isclose(p.z, self.z), "Should construct constant.")
        self.assertTrue(p.is_sef, "Should detect SEF.")

        self.assertTrue(
            np.issubdtype(p.A.dtype, np.floating), "Should be of type float."
        )
        self.assertTrue(
            np.issubdtype(p.b.dtype, np.floating), "Should be of type float."
        )
        self.assertTrue(
            np.issubdtype(p.c.dtype, np.floating), "Should be of type float."
        )
        self.assertIn(type(p.z), [float, int], "Should be of type float or int.")

    def test_invalid_dimensions(self):
        A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

        with self.assertRaises(
            ValueError,
            msg="Should throw exception if dimension mismatch between A and c.",
        ):
            p = LinearProgram(A, self.b, self.c, self.z)

        A = [[1, 2, 3, 4], [5, 6, 8], [9, 10, 11, 12]]

        with self.assertRaises(
            ValueError, msg="Should throw exception if A is a jagged array."
        ):
            p = LinearProgram(A, self.b, self.c, self.z)

        A = [[1, 2, 3, 4], [5, 6, 8, 10], [[9, 10], 10, 11, 12]]

        with self.assertRaises(
            ValueError, msg="Should throw exception if A has more than 2 dimensions."
        ):
            p = LinearProgram(A, self.b, self.c, self.z)

        A = np.array([1, 2, 3, 4])

        with self.assertRaises(
            ValueError, msg="Should throw exception if A is one dimensional."
        ):
            p = LinearProgram(A, self.b, self.c, self.z)

        b = np.array([[10, 20, 30]])

        with self.assertRaises(
            ValueError, msg="Should throw exception if b is two dimensional."
        ):
            p = LinearProgram(self.A, b, self.c, self.z)

        c = np.array([[1, 2, 3], [4, 5, 6]])

        with self.assertRaises(
            ValueError, msg="Should throw exception if c is two dimensional."
        ):
            p = LinearProgram(self.A, self.b, c, self.z)

        b = np.array([10, 20])

        with self.assertRaises(
            ValueError,
            msg="Should throw exception if dimension mismatch between A and b.",
        ):
            p = LinearProgram(self.A, b, self.c, self.z)

        with self.assertRaises(
            ValueError, msg="Should throw exception if arrays are empty."
        ):
            p = LinearProgram([], [], [], self.z)

        with self.assertRaises(
            ValueError,
            msg="Should throw exception if dimension mismatch between inequalities and b.",
        ):
            p = LinearProgram(self.A, self.b, self.c, self.z, inequalities=["="])

        with self.assertRaises(
            ValueError,
            msg="Should throw exception if dimension mismatch between inequalities and b.",
        ):
            p = LinearProgram(
                self.A, self.b, self.c, self.z, inequalities=["=", "<=", ">=", ">="]
            )

        with self.assertRaises(
            ValueError,
            msg="Should throw exception if dimension mismatch between inequalities and b.",
        ):
            p = LinearProgram(self.A, self.b, self.c, self.z, inequalities=[])

        with self.assertRaises(
            ValueError,
            msg="Should throw exception if number of free variables is more than implied.",
        ):
            p = LinearProgram(
                self.A,
                self.b,
                self.c,
                self.z,
                inequalities=None,
                free_variables=[1, 2, 3, 4],
            )

    def test_invalid_values(self):
        A = np.array([["a", 2, "b"], [5, 6, "c"], [9, "d", 12]])

        with self.assertRaises(
            ValueError, msg="Should throw exception if type of A is incorrect."
        ):
            p = LinearProgram(A, self.b, self.c, self.z)

        b = "test"

        with self.assertRaises(
            ValueError, msg="Should throw exception if type of b is incorrect."
        ):
            p = LinearProgram(self.A, b, self.c, self.z)

        c = 6

        with self.assertRaises(
            ValueError, msg="Should throw exception if type of c is incorrect."
        ):
            p = LinearProgram(self.A, self.b, c, self.z)

        with self.assertRaises(
            ValueError,
            msg="Should throw exception if type of inequalities is incorrect.",
        ):
            p = LinearProgram(self.A, self.b, self.c, self.z, inequalities="test")

        with self.assertRaises(
            ValueError,
            msg="Should throw exception if inequalities have invalid values.",
        ):
            p = LinearProgram(self.A, self.b, self.c, self.z, inequalities=["+", 23])

        with self.assertRaises(
            ValueError,
            msg="Should throw exception if free variables have invalid indices.",
        ):
            p = LinearProgram(self.A, self.b, self.c, self.z, free_variables=[0, 1])

        with self.assertRaises(
            ValueError,
            msg="Should throw exception if free variables have invalid indices.",
        ):
            p = LinearProgram(self.A, self.b, self.c, self.z, free_variables=[4])

    def test_sef_detection(self):
        p = LinearProgram(self.A, self.b, self.c, self.z)

        self.assertTrue(p.is_sef, "Should detect SEF.")

        p = LinearProgram(self.A, self.b, self.c, self.z, inequalities=["=", "=", "="])

        self.assertTrue(p.is_sef, "Should detect SEF.")

        p = LinearProgram(self.A, self.b, self.c, self.z, "min")

        self.assertFalse(p.is_sef, "Should detect non-SEF.")

        p = LinearProgram(self.A, self.b, self.c, self.z, inequalities=["<=", "=", "="])

        self.assertFalse(p.is_sef, "Should detect non-SEF.")

        p = LinearProgram(
            self.A, self.b, self.c, self.z, inequalities=None, free_variables=[1]
        )

        self.assertFalse(p.is_sef, "Should detect non-SEF.")

        p = LinearProgram(self.A, self.b, self.c, self.z, "min", None, [1])

        self.assertFalse(p.is_sef, "Should detect non-SEF.")


if __name__ == "__main__":
    main()
