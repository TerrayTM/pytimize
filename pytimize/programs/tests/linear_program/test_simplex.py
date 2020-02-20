import numpy as np

from ... import LinearProgram
from unittest import TestCase, main

class TestSimplex(TestCase):
    def test_simplex(self) -> None:
        A = np.array([
            [-1, 2, 4, 2, -1, 1, 0, 0],
            [0, 1, 0, 1, 1, 0, 1, 0],
            [0, 5, 1, 2, 0, 0, 0, 1]
        ])
        b = np.array([6, 1, 7])
        c = np.array([-1, 8, 5, 5, 0, 0, 0, 0])
        z = -14

        p = LinearProgram(A, b, c, z)

        solution, basis, certificate = p.simplex([6, 7, 8])

        self.assertTrue(np.allclose(solution, [4, 1, 2, 0, 0, 0, 0, 0]), "Should compute correct solution.")
        self.assertEqual(basis, [1, 2, 3], "Should compute correct optimal basis.")
        self.assertTrue(np.allclose(certificate, [1, 1, 1]), "Should compute correct certificate.")
        self.assertTrue(np.allclose(p.A, A), "Should not modify original program.")

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
        self.assertEqual(basis, [1, 2, 5], "Should compute correct optimal basis.")
        self.assertTrue(np.allclose(certificate, [0.125, 0, 0.375]), "Should compute correct certificate.")
        self.assertTrue(np.allclose(p.A, A), "Should not modify original program.")

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
        self.assertEqual(basis, [1, 2, 5], "Should compute correct optimal basis.")
        self.assertTrue(np.allclose(certificate, [1.25, 0.25, 0]), "Should compute correct certificate.")
        self.assertTrue(np.allclose(p.A, A), "Should not modify original program.")

    def test_simplex_unbounded(self) -> None:
        A = np.array([
            [1, 1, -3, 1, 2],
            [0, 1, -2, 2, -2],
            [-2, -1, 4, 1, 0]
        ])
        b = np.array([7, -2, -3])
        c = np.array([-1, 0, 3, 7, -1])
        z = 0

        p = LinearProgram(A, b, c, z)

        solution, basis, certificate = p.simplex([3, 4, 5])

        self.assertTrue(solution is None, "Should have no optimal solution.")
        self.assertTrue(isinstance(certificate, tuple), "Should be a tuple of vectors.")
        self.assertTrue(np.allclose(certificate[0], [36, 0, 14, 13, 0]), "Should be correct feasible solution.")
        self.assertTrue(np.allclose(certificate[1], [0.5, 1, 0.5, 0, 0]), "Should be correct certifying vector.")
        self.assertTrue(basis is None, "Should return nothing.")
        self.assertTrue(np.allclose(p.A, A), "Should not modify original program.")

    def test_simplex_negative_b(self) -> None:
        pass #TODO

    def test_simplex_optimal_basis(self) -> None:
        A = np.array([
            [2, 1, 1, 0, 0],
            [2, 3, 0, 1, 0],
            [3, 1, 0, 0, 1]
        ])
        b = np.array([18, 42, 24])
        c = np.array([3, 2, 0, 0, 0])
        z = 0

        p = LinearProgram(A, b, c, z)

        solution, basis, certificate = p.simplex([1, 2, 5])

        self.assertTrue(np.allclose(solution, [3, 12, 0, 0, 3]), "Should compute correct solution.")
        self.assertEqual(basis, [1, 2, 5], "Should compute correct optimal basis.")
        self.assertTrue(np.allclose(certificate, [1.25, 0.25, 0]), "Should compute correct certificate.")
        self.assertTrue(np.allclose(p.A, A), "Should not modify original program.")
    
    def test_simplex_optimal_form(self) -> None:
        A = np.array([
            [1, 0, 0.75, -0.25, 0],
            [0, 1, -0.5, 0.5, 0],
            [0, 0, -1.75, 0.25, 1]
        ])
        b = np.array([3, 12, 3])
        c = np.array([0, 0, -1.25, -0.25, 0])
        z = 33

        p = LinearProgram(A, b, c, z)

        solution, basis, certificate = p.simplex([1, 2, 5])

        self.assertTrue(np.allclose(solution, [3, 12, 0, 0, 3]), "Should compute correct solution.")
        self.assertEqual(basis, [1, 2, 5], "Should compute correct optimal basis.")
        self.assertTrue(np.allclose(certificate, [0, 0, 0]), "Should compute correct certificate.")
        self.assertTrue(np.allclose(p.A, A), "Should not modify original program.")

    def test_simplex_infeasible(self) -> None:
        A = np.array([
            [2, -1, -2, 1, 0, 0],
            [2, -3, -1, 0, 1, 0],
            [-1, 1, 1, 0, 0, 1]
        ])
        b = np.array([4, -5, -1])
        c = np.array([1, -1, 1, 0, 0, 0])
        z = 0

        p = LinearProgram(A, b, c, z)
        
        with self.assertRaises(ArithmeticError, msg="Should throw exception if infeasible."):
            p.simplex([4, 5, 6])
        
        A = np.array([
            [4, 10, -6, -2],
            [-2, 2, -4, 1],
            [-7, -2, 0, 4]
        ])
        b = np.array([6, 5, 3])
        c = np.array([1, 1, 1, 1])
        z = 0

        p = LinearProgram(A, b, c, z)
        
        with self.assertRaises(ArithmeticError, msg="Should throw exception if infeasible."):
            p.simplex([1, 2, 3])

        A = np.array([
            [1, 1, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
        b = np.array([1, 2, 3])
        c = np.array([1, 1, 1])
        z = 0

        with self.assertRaises(ArithmeticError, msg="Should throw exception if infeasible."):
            p.simplex([1, 2, 3])

        A = np.array([
            [3, 4, 1, 0, 0],
            [4, 2, 0, 1, 0],
            [1, 1, 0, 0, -1]
        ])
        b = np.array([12, 8, 4])
        c = np.array([4, 3, 0, 0, 0])
        z = 0

        with self.assertRaises(ArithmeticError, msg="Should throw exception if infeasible."):
            p.simplex([3, 4, 5])

if __name__ == "__main__":
    main()
