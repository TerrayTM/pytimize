from unittest import TestCase, main

import numpy as np

from ... import LinearProgram


class TestTwoPhaseSimplex(TestCase):
    def test_two_phase_simplex(self) -> None:
        A = np.array([[-1, -2, 1], [1, -1, 1]])
        b = np.array([-1, 3])
        c = np.array([2, -1, 2])
        z = 0

        p = LinearProgram(A, b, c, z)

        solution, basis, certificate = p.two_phase_simplex()

        self.assertTrue(
            np.allclose(solution, [0, 4, 7]), "Should compute correct solution."
        )
        self.assertTrue(
            np.allclose(basis, [2, 3]), "Should compute correct optimal basis."
        )
        self.assertTrue(
            np.allclose(certificate, np.array([-1, 3])),
            "Should compute correct certificate.",
        )
        self.assertTrue(np.allclose(p.A, A), "Should not modify program.")

        A = np.array(
            [
                [1.5, 1, 1, 0, 0, 0, 0],
                [9, 11, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, -1, 0, 0],
                [0, -1, 0, 0, 0, -1, 0],
                [-1, 0, 0, 0, 0, 0, 1],
            ]
        )
        b = np.array([6, 45, 3, -1, -4])
        c = np.array([60, 50, 0, 0, 0, 0, 0])
        z = 0

        p = LinearProgram(A, b, c, z)

        solution, basis, certificate = p.two_phase_simplex()

        self.assertTrue(
            np.allclose(solution, [4, 0, 0, 9, 1, 1, 0]),
            "Should compute correct solution.",
        )
        self.assertTrue(
            np.allclose(basis, [1, 2, 4, 5, 6]), "Should compute correct optimal basis."
        )
        self.assertTrue(
            np.allclose(certificate, np.array([50, 0, 0, 0, 15])),
            "Should compute correct certificate.",
        )
        self.assertTrue(np.allclose(p.A, A), "Should not modify program.")

        # TODO Add one more test case for optimal outcome

    def test_two_phase_simplex_infeasible(self) -> None:
        A = np.array([[2, -1, -2, 1, 0, 0], [2, -3, -1, 0, 1, 0], [-1, 1, 1, 0, 0, 1]])
        b = np.array([4, -5, -1])
        c = np.array([1, -1, 1, 0, 0, 0])
        z = 0

        p = LinearProgram(A, b, c, z)

        solution, basis, certificate = p.two_phase_simplex()

        self.assertTrue(solution is None, "Should not exist a solution.")
        self.assertTrue(basis is None, "Should not exist an optimal basis.")
        self.assertTrue(
            np.allclose(certificate, [0.25, 0.25, 1]),
            "Should compute correct certificate.",
        )
        self.assertTrue(np.allclose(p.A, A), "Should not modify program.")

        A = np.array([[4, 10, -6, -2], [-2, 2, -4, 1], [-7, -2, 0, 4]])
        b = np.array([6, 5, 3])
        c = np.array([1, 1, 1, 1])
        z = 0

        p = LinearProgram(A, b, c, z)

        solution, basis, certificate = p.two_phase_simplex()

        self.assertTrue(solution is None, "Should not exist a solution.")
        self.assertTrue(basis is None, "Should not exist an optimal basis.")
        self.assertTrue(
            np.allclose(certificate, [0.277777, -1, 0.388888]),
            "Should compute correct certificate.",
        )
        self.assertTrue(np.allclose(p.A, A), "Should not modify program.")

        A = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0]])
        b = np.array([1, 2, 3])
        c = np.array([1, 1, 1])
        z = 0

        p = LinearProgram(A, b, c, z)

        solution, basis, certificate = p.two_phase_simplex()

        self.assertTrue(solution is None, "Should not exist a solution.")
        self.assertTrue(basis is None, "Should not exist an optimal basis.")
        self.assertTrue(
            np.allclose(certificate, [1, -1, -1]), "Should compute correct certificate."
        )
        self.assertTrue(np.allclose(p.A, A), "Should not modify program.")

        A = np.array([[3, 4, 1, 0, 0], [4, 2, 0, 1, 0], [1, 1, 0, 0, -1]])
        b = np.array([12, 8, 4])
        c = np.array([4, 3, 0, 0, 0])
        z = 0

        p = LinearProgram(A, b, c, z)

        solution, basis, certificate = p.two_phase_simplex()

        self.assertTrue(solution is None, "Should not exist a solution.")
        self.assertTrue(basis is None, "Should not exist an optimal basis.")
        self.assertTrue(
            np.allclose(certificate, [0.2, 0.1, -1]),
            "Should compute correct certificate.",
        )
        self.assertTrue(np.allclose(p.A, A), "Should not modify program.")

        A = np.array([[3, 5, 2, -1], [-2, -5, -3, -1]])
        b = np.array([7, -3])
        c = np.array([1, 0, 0, 0])
        z = 0

        p = LinearProgram(A, b, c, z)

        solution, basis, certificate = p.two_phase_simplex()

        self.assertTrue(solution is None, "Should not exist a solution.")
        self.assertTrue(basis is None, "Should not exist an optimal basis.")
        self.assertTrue(
            np.allclose(certificate, [-1, -1.5]), "Should compute correct certificate."
        )
        self.assertTrue(np.allclose(p.A, A), "Should not modify program.")

    def test_two_phase_simplex_unbounded(self) -> None:
        pass  # TODO


if __name__ == "__main__":
    main()
