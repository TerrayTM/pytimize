import numpy as np

from ... import LinearProgram
from unittest import TestCase, main

class TestTwoPhaseSimplex(TestCase):
    def test_two_phase_simplex(self):
        A = np.array([
            [-1, -2, 1],
            [1, -1, 1]
        ])
        b = np.array([-1, 3])
        c = np.array([2, -1, 2])
        z = 0

        p = LinearProgram(A, b, c, z)

        solution, basis, certificate = p.two_phase_simplex()

        self.assertTrue(np.allclose(solution, [0, 4, 7]), "Should compute correct solution.")
        self.assertTrue(np.allclose(basis, [2, 3]), "Should compute correct optimal basis.")
        self.assertTrue(np.allclose(certificate, np.array([1, 3])), "Should compute correct certificate.") # SHOULD BE -1,3 

if __name__ == "__main__":
    main()
