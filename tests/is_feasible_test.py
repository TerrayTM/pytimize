import sys
import numpy as np

sys.path.append("../optimization")
sys.path.append("../optimization/enums")

from main import LinearProgrammingModel
from unittest import TestCase, main
from objective import Objective

class IsFeasibleTest(TestCase):
    def setUp(self):
        self.A = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        self.b = np.array([6, 15, 24])
        self.c = np.array([100, 200, 300])
        self.z = 5

    def test_is_feasible(self):
        p = LinearProgrammingModel(self.A, self.b, self.c, self.z, Objective.min)

        self.assertTrue(p.is_feasible([1, 1, 1]), "Should output true.")
        self.assertFalse(p.is_feasible([-1, 0.5, 100]), "Should output false.")

        p = LinearProgrammingModel(self.A, self.b, self.c, self.z, Objective.min, [">=", ">=", ">="])

        self.assertTrue(p.is_feasible([1, 1, 1]), "Should output true.")
        self.assertTrue(p.is_feasible([2, 2, 2]), "Should output true.")
        self.assertFalse(p.is_feasible([2, 2, -2]), "Should output false.")

        # p = LinearProgrammingModel(self.A, self.b, self.c, self.z, Objective.min, [">=", "<=", ">="])

        # self.assertFalse(p.is_feasible([10, 0.25, 1]), "Should output false,")
        # self.assertFalse(p.is_feasible([2, 2, 2]), "Should output false,")
        # self.assertFalse(p.is_feasible([2, 2, -2]), "Should output false,")

        p = LinearProgrammingModel(self.A, self.b, self.c, self.z, Objective.min, ["<=", "<=", "<="], [1])
        
        self.assertTrue(p.is_feasible([1, 1, 1]), "Should output true.")
        self.assertTrue(p.is_feasible([-10, 1, 1]), "Should output true.")
        self.assertFalse(p.is_feasible([-10, 2, -2]), "Should output false.")

        p = LinearProgrammingModel(-self.A, self.b, self.c, self.z, Objective.min, [">=", ">=", ">="], [1, 3])
        
        self.assertTrue(p.is_feasible([-42, 1, -23]), "Should output true.")
        self.assertTrue(p.is_feasible([-10, 0, -1]), "Should output true.")
        self.assertFalse(p.is_feasible([-10, -2, -2]), "Should output false.")


    def test_invalid_dimension(self):
        p = LinearProgrammingModel(self.A, self.b, self.c, self.z, Objective.min)

        with self.assertRaises(ValueError, msg="Should throw error if incorrect dimension."):
            p.is_feasible(np.array([1]))

        with self.assertRaises(ValueError, msg="Should throw error if incorrect dimension."):
            p.is_feasible(np.array([1, 2, 3, 4, 5]))

    def test_invalid_value(self):
        p = LinearProgrammingModel(self.A, self.b, self.c, self.z, Objective.min)

        with self.assertRaises(ValueError, msg="Should throw error if invalid type."):
            p.is_feasible("test")

if __name__ == "__main__":
    main()
