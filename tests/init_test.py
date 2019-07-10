import sys
import numpy as np

sys.path.append("../optimization")
sys.path.append("../optimization/enums")

from main import LinearProgrammingModel
from unittest import TestCase, main
from objective import Objective
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
        inequalities = np.array(["=", "=", "="])

        p = LinearProgrammingModel(A, b, c, z, Objective.min, inequalities)

        self.assertTrue(np.allclose(p.A, A), "Should construct coefficient matrix.")
        self.assertTrue(np.allclose(p.b, b), "Should construct constraints.")
        self.assertTrue(np.allclose(p.c, c), "Should construct coefficient vector.")
        self.assertTrue(isclose(p.z, z), "Should construct constant.")
        self.assertTrue(p.objective == Objective.min, "Should construct objective.")
        self.assertTrue(p.inequalities == ["=", "=", "="], "Should construct inequalities.")
        self.assertTrue(p.is_sef, "Should detect SEF form.")
        
        self.assertTrue(np.issubdtype(p.A.dtype, np.floating), "Should be of type float.")
        self.assertTrue(np.issubdtype(p.b.dtype, np.floating), "Should be of type float.")
        self.assertTrue(np.issubdtype(p.c.dtype, np.floating), "Should be of type float.")
        self.assertTrue(type(p.z) == float or type(p.z) == int, "Should be of type float or int.")
        self.assertTrue(isinstance(p.objective, Objective), "Should be enum type Objective.")
        self.assertTrue(isinstance(p.inequalities, list), "Should be of type list.")

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

        A = np.array([
            [1, 2],
            [5, 6],
            [9, 10]
        ])
        b = np.array([6, 8, 10])
        c = np.array([100, 300])
        z = 5

        with self.assertRaises(ValueError, msg="Should throw exception if dimension mismatch between inequalities and b."):
            p = LinearProgrammingModel(A, b, c, z, inequalities=["="])

        with self.assertRaises(ValueError, msg="Should throw exception if dimension mismatch between inequalities and b."):
            p = LinearProgrammingModel(A, b, c, z, inequalities=["=", "<=", ">=", ">="])

    def test_invalid_values(self):
        A = np.array([
            ["a", 2, "b"],
            [5, 6, "c"],
            [9, "d", 12]
        ])
        b = np.array([6, 8, 10])
        c = np.array([100, 200, 300])
        z = 5
        objective = Objective.min

        with self.assertRaises(ValueError, msg="Should throw exception if type of A is incorrect."):
            p = LinearProgrammingModel(A, b, c, z, objective)

        A = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        b = "test"

        with self.assertRaises(ValueError, msg="Should throw exception if type of b is incorrect."):
            p = LinearProgrammingModel(A, b, c, z, objective)

        b = np.array([6, 8, 10])
        c = 6

        with self.assertRaises(ValueError, msg="Should throw exception if type of c is incorrect."):
            p = LinearProgrammingModel(A, b, c, z, objective)

        c = np.array([100, 200, 300])
        z = "test"

        with self.assertRaises(ValueError, msg="Should throw exception if type of z is incorrect."):
            p = LinearProgrammingModel(A, b, c, z, objective)
        
        z = 10
        objective = "min"

        with self.assertRaises(ValueError, msg="Should throw exception if type of objective is incorrect."):
            p = LinearProgrammingModel(A, b, c, z, objective)

        with self.assertRaises(ValueError, msg="Should throw exception if type of inequalities is incorrect."):
            p = LinearProgrammingModel(A, b, c, z, inequalities="test")

        with self.assertRaises(ValueError, msg="Should throw exception if inequalities have invalid values."):
            p = LinearProgrammingModel(A, b, c, z, inequalities=["+", 23])

    def test_sef_detection(self):
        p = LinearProgrammingModel([[1, 2, 3], [4, 5, 6]], [10, 20], [100, 300, 500], 20)

        self.assertTrue(p.is_sef, "Should detect SEF form.")

        p = LinearProgrammingModel([[1, 2, 3], [4, 5, 6]], [10, 20], [100, 300, 500], 20, inequalities=["=", "="])

        self.assertTrue(p.is_sef, "Should detect SEF form.")

        p = LinearProgrammingModel([[1, 2, 3], [4, 5, 6]], [10, 20], [100, 300, 500], 20, inequalities=["<=", "="])

        self.assertFalse(p.is_sef, "Should detect non-SEF form")

        # To do: add cases on free variables

if __name__ == "__main__":
    main()
