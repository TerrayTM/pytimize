import sys
import numpy as np

sys.path.append("../optimization")

from main import LinearProgrammingModel
from unittest import TestCase, main
from math import isclose

# To do: add test cases for optional parameters

class TestInit(TestCase):
    def test_init(self):
        A = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        b = np.array([6, 8, 10])
        c = np.array([100, 200, 300])
        z = 5

        p = LinearProgrammingModel(A, b, c, z)

        self.assertTrue(np.allclose(p.A, A), "Should construct coefficient matrix.")
        self.assertTrue(np.allclose(p.b, b), "Should construct constraints.")
        self.assertTrue(np.allclose(p.c, c), "Should construct coefficient vector.")
        self.assertTrue(isclose(p.z, z), "Should construct constant.")
        
        self.assertTrue(np.issubdtype(p.A.dtype, np.floating), "Should be of type float.")
        self.assertTrue(np.issubdtype(p.b.dtype, np.floating), "Should be of type float.")
        self.assertTrue(np.issubdtype(p.c.dtype, np.floating), "Should be of type float.")
        self.assertTrue(type(p.z) == float or type(p.z) == int, "Should be of type float or int.")

    def test_invalid_dimensions(self):
        A = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ])
        b = np.array([6, 8, 10])
        c = np.array([100, 300])
        z = 5

        with self.assertRaises(ValueError, msg="Should throw exception if dimension mismatch between A and c."):
            p = LinearProgrammingModel(A, b, c, z)

        c = np.array([100, 200, 300, 400])
        A = np.array([1, 2, 3, 4])

        with self.assertRaises(ValueError, msg="Should throw exception if A is one dimensional."):
            p = LinearProgrammingModel(A, b, c, z)

        A = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ])
        b = np.array([
            [10, 20, 30]
        ])

        with self.assertRaises(ValueError, msg="Should throw exception if b is two dimensional."):
            p = LinearProgrammingModel(A, b, c, z)

        b = np.array([6, 8, 10])
        c = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])

        with self.assertRaises(ValueError, msg="Should throw exception if c is two dimensional."):
            p = LinearProgrammingModel(A, b, c, z)

        c = np.array([100, 200, 300, 400])
        b = np.array([10, 20])

        with self.assertRaises(ValueError, msg="Should throw exception if dimension mismatch between A and b."):
            p = LinearProgrammingModel(A, b, c, z)

        A = np.array([])
        b = np.array([])
        c = np.array([])

        with self.assertRaises(ValueError, msg="Should throw exception if arrays are empty."):
            p = LinearProgrammingModel(A, b, c, z)

    def test_invalid_types(self):
        A = np.array([
            ["a", 2, "b"],
            [5, 6, "c"],
            [9, "d", 12]
        ])
        b = np.array([6, 8, 10])
        c = np.array([100, 200, 300])
        z = 5

        with self.assertRaises(TypeError, msg="Should throw exception type of A is incorrect."):
            p = LinearProgrammingModel(A, b, c, z)

        A = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        b = "test"

        with self.assertRaises(TypeError, msg="Should throw exception type of b is incorrect."):
            p = LinearProgrammingModel(A, b, c, z)

        b = np.array([6, 8, 10])
        c = 6

        with self.assertRaises(TypeError, msg="Should throw exception type of c is incorrect."):
            p = LinearProgrammingModel(A, b, c, z)

        c = np.array([100, 200, 300])
        z = "test"

        with self.assertRaises(TypeError, msg="Should throw exception type of z is incorrect."):
            p = LinearProgrammingModel(A, b, c, z)

if __name__ == "__main__":
    main()
