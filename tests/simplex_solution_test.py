import sys
import numpy as np

sys.path.append("../optimization")

from main import LinearProgram
from unittest import TestCase, main

class TestSimplexSolution(TestCase):
    def test_simplex_solution(self):
        A = np.array([
            [-1, 2, 4, 2, -1, 1, 0, 0],
            [0, 1, 0, 1, 1, 0, 1, 0],
            [0, 5, 1, 2, 0, 0, 0, 1]
        ])
        b = np.array([6, 1, 7])
        c = np.array([-1, 8, 5, 5, 0, 0, 0, 0])
        z = -14

        p = LinearProgram(A, b, c, z)

        solution, basis = p.simplex_solution([6, 7, 8])

        self.assertTrue(np.allclose(solution, [4, 1, 2, 0, 0, 0, 0, 0]), "Should compute correct solution.")
        self.assertTrue(np.allclose(basis, [1, 2, 3]), "Should compute correct optimal basis.")

if __name__ == "__main__":
    main()
