import numpy as np

from ... import LinearProgram
from unittest import TestCase, main

class TestSimplex(TestCase):
    def test_simplex(self):
        A = np.array([
            [-1, 2, 4, 2, -1, 1, 0, 0],
            [0, 1, 0, 1, 1, 0, 1, 0],
            [0, 5, 1, 2, 0, 0, 0, 1]
        ])
        b = np.array([6, 1, 7])
        c = np.array([-1, 8, 5, 5, 0, 0, 0, 0])
        z = -14

        p = LinearProgram(A, b, c, z)

        solution, basis, certificate = p.simplex([6, 7, 8]) #TODO need work

        self.assertTrue(np.allclose(solution, [4, 1, 2, 0, 0, 0, 0, 0]), "Should compute correct solution.")
        self.assertTrue(np.allclose(basis, [1, 2, 3]), "Should compute correct optimal basis.")

        A = np.array([
            [2, 1, 1, 1, 0, 0],
            [4, 2, 3, 0, 1, 0],
            [2, 5, 5, 0, 0, 1]
        ])
        b = np.array([14, 28, 30])
        c = np.array([1, 2, -1, 0, 0, 0])
        z = 0

        p = LinearProgram(A, b, c, z)

        solution, basis, certificate = p.simplex([4, 5, 6])

        self.assertTrue(np.allclose(solution, [5, 4, 0, 0, 0, 0]), "Should compute correct solution.")
        self.assertTrue(np.allclose(basis, [1, 2, 5]), "Should compute correct optimal basis.")

        A = np.array([
            [2, 1, 1, 0, 0],
            [2, 3, 0, 1, 0],
            [3, 1, 0, 0, 1]
        ])
        b = np.array([18, 42, 24])
        c = np.array([3, 2, 0, 0, 0])
        z = 0

        p = LinearProgram(A, b, c, z)

        solution, basis, certificate = p.simplex([3, 4, 5])

        self.assertTrue(np.allclose(solution, [3, 12, 0, 0, 3]), "Should compute correct solution.")
        self.assertTrue(np.allclose(basis, [1, 2, 5]), "Should compute correct optimal basis.")

if __name__ == "__main__":
    main()
