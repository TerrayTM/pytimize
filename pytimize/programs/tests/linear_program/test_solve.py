import numpy as np

from ... import LinearProgram
from unittest import TestCase, main

class TestSolve(TestCase):
    def test_solve(self) -> None:
        A = np.array([
            [2, 1],
            [2, 3],
            [3, 1]
        ])
        b = np.array([18, 42, 24])
        c = np.array([3, 2])
        z = 0

        p = LinearProgram(A, b, c, z, inequalities=["<=", "<=", "<="])

        self.assertTrue(np.allclose(p.solve(), [3, 12]), "Should compute correct solution.")

        A = np.array([
            [-1, 2, 4, 2, -1],
            [0, 1, 0, 1, 1],
            [0, 5, 1, 2, 0]
        ])
        b = np.array([6, 1, 7])
        c = np.array([-1, 8, 5, 5, 0])
        z = -14

        p = LinearProgram(A, b, c, z, inequalities=["<=", "<=", "<="])
        
        self.assertTrue(np.allclose(p.solve(), [4, 1, 2, 0, 0]), "Should compute correct solution.")

        A = np.array([
            [-1, 2, 4, 2, -1, 1, 0, 0],
            [0, 1, 0, 1, 1, 0, 1, 0],
            [0, 5, 1, 2, 0, 0, 0, 1]
        ])
        b = np.array([6, 1, 7])
        c = np.array([1, -8, -5, -5, 0, 0, 0, 0])
        z = -14

        p = LinearProgram(A, b, c, z, "min")
        
        self.assertTrue(np.allclose(p.solve(), [4, 1, 2, 0, 0, 0, 0, 0]), "Should compute correct solution.")

    def test_solve_sef(self) -> None:
        A = np.array([
            [-1, 2, 4, 2, -1, 1, 0, 0],
            [0, 1, 0, 1, 1, 0, 1, 0],
            [0, 5, 1, 2, 0, 0, 0, 1]
        ])
        b = np.array([6, 1, 7])
        c = np.array([-1, 8, 5, 5, 0, 0, 0, 0])
        z = -14

        p = LinearProgram(A, b, c, z)
        
        self.assertTrue(np.allclose(p.solve(), [4, 1, 2, 0, 0, 0, 0, 0]), "Should compute correct solution.")

if __name__ == "__main__":
    main()
