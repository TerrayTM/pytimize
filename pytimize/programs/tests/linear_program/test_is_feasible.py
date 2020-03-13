import numpy as np

from ... import LinearProgram
from unittest import TestCase, main

# Add test for free variables and negative values

class TestIsFeasible(TestCase):
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
        p = LinearProgram(self.A, self.b, self.c, self.z, "min")

        self.assertTrue(p.is_feasible([1, 1, 1]), "Should output true.")
        self.assertFalse(p.is_feasible([-1, 0.5, 100]), "Should output false.")

        p = LinearProgram(self.A, self.b, self.c, self.z, "min", [">=", ">=", ">="])

        self.assertTrue(p.is_feasible([1, 1, 1]), "Should output true.")
        self.assertTrue(p.is_feasible([2, 2, 2]), "Should output true.")
        self.assertFalse(p.is_feasible([2, 2, -2]), "Should output false.")

        # p = LinearProgram(self.A, self.b, self.c, self.z, "min", [">=", "<=", ">="])

        # self.assertFalse(p.is_feasible([10, 0.25, 1]), "Should output false,")
        # self.assertFalse(p.is_feasible([2, 2, 2]), "Should output false,")
        # self.assertFalse(p.is_feasible([2, 2, -2]), "Should output false,")

        p = LinearProgram(self.A, self.b, self.c, self.z, "min", ["<=", "<=", "<="], [1])
        
        self.assertTrue(p.is_feasible([1, 1, 1]), "Should output true.")
        self.assertTrue(p.is_feasible([-10, 1, 1]), "Should output true.")
        self.assertFalse(p.is_feasible([-10, 2, -2]), "Should output false.")

        p = LinearProgram(-self.A, self.b, self.c, self.z, "min", [">=", ">=", ">="], [1, 3])
        
        self.assertTrue(p.is_feasible([-42, 1, -23]), "Should output true.")
        self.assertTrue(p.is_feasible([-10, 0, -1]), "Should output true.")
        self.assertFalse(p.is_feasible([-10, -2, -2]), "Should output false.")

    def test_invalid_dimension(self):
        p = LinearProgram(self.A, self.b, self.c, self.z, "min")

        with self.assertRaises(ValueError, msg="Should throw error if incorrect dimension."):
            p.is_feasible(np.array([1]))

        with self.assertRaises(ValueError, msg="Should throw error if incorrect dimension."):
            p.is_feasible(np.array([1, 2, 3, 4, 5]))

    def test_invalid_value(self):
        p = LinearProgram(self.A, self.b, self.c, self.z, "min")

        with self.assertRaises(ValueError, msg="Should throw error if invalid type."):
            p.is_feasible("test") 
    
    def test_feasible(self):
        "Testing an infeasible polyhedron"
        A = np.array([
            [1, 2],
            [-1, 0],
            [0, -1],
            [-3, -3]
        ])
        b = np.array([1, 0, 2, -6])
        c = np.array([1, 1])
        z = 0

        p = LinearProgram(A, b, c, z, "max", ["<=", "<=", "<=", "<="])
        # This polyhedron is empty, so all solutions should be infeasible

        for i in range(-5, 6):
            for j in range(-5, 6):
                self.assertFalse(p.is_feasible([i, j]), "Should output false.")
    
    def test_feasible2(self):
        A = np.array([
            [1, 2],
            [1, 1],
            [1, -1]
        ])
        b = np.array([2, 2, 0.5])
        c = np.array([2, 1])
        z = 0
        ineqs = ["<=", "<=", "<="]

        p = LinearProgram(A, b, c, z, "max", ineqs)
        self.assertTrue(p.is_feasible([0, 1]), "Should output true.")
        self.assertTrue(p.is_feasible([1, 0.5]), "Should output true.")
        self.assertFalse(p.is_feasible([2, 1]), "Should output false.")
        self.assertFalse(p.is_feasible([0.001, 2]), "Should output false.")
    
    def test_total_feasible(self):
        "Testing feasibility when the polyhedron covers all of Rn"
        A = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        b = np.array([0, 0, 0])
        c = np.array([0, 0, 0])
        z = 0

        p = LinearProgram(A, b, c, z, "max", ["=", "=", "="])
        q = LinearProgram(A, b, c, z, "max", ["<=", "<=", "<="])
        r = LinearProgram(A, b, c, z, "max", [">=", ">=", ">="])
        for i in range(-10, 11):
            for j in range(-10, 11):
                for k in range(-10, 11):
                    self.assertTrue(p.is_feasible([i, j, k]), "Should output true.")
                    self.assertTrue(q.is_feasible([i, j, k]), "Should output true.")
                    self.assertTrue(r.is_feasible([i, j, k]), "Should output true.")
    
    """
    def test_empty_feasible(self):
        "Testing feasibility with an empty matrix."
        A = np.array([[], []])
        b = np.array([])
        c = np.array([1, 1])
        z = 3

        p = LinearProgram(A, b, c, z, "max")
        self.assertTrue(p.is_feasible([1, 0]), "Should output true.")
    """

if __name__ == "__main__":
    main()
