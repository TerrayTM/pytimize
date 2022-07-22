from unittest import TestCase, main

import numpy as np

from ... import LinearProgram


class TestSimplexIteration(TestCase):
    def test_simplex_iteration(self) -> None:
        A_one = np.array([[1, -1, 1, -1], [2, -1, 1, 0]])
        b_one = np.array([-2, -1])
        c_one = np.array([-1, 1, -2, 4])
        z_one = 0

        A_two = np.array([[1, 0, 0, 1], [0, 1, -1, 2]])
        b_two = np.array([1, 3])
        c_two = np.array([0, 0, -1, 3])
        z_two = 2

        A_three = np.array([[-2, 1, -1, 0], [1, 0, 0, 1]])
        b_three = np.array([1, 1])
        c_three = np.array([-3, 0, -1, 0])
        z_three = 5

        lp = LinearProgram(A_one, b_one, c_one, z_one)
        solution, basis, lp = lp.simplex_iteration([1, 2])

        self.assertEqual(basis, [2, 4], "Should compute correct basis.")
        self.assertIsNone(solution, "Should compute correct solution.")
        self.assertTrue(np.allclose(A_two, lp.A), "Should compute correct iteration.")
        self.assertTrue(np.allclose(b_two, lp.b), "Should compute correct iteration.")
        self.assertTrue(np.allclose(c_two, lp.c), "Should compute correct iteration.")
        self.assertTrue(np.allclose(z_two, lp.z), "Should compute correct iteration.")

        solution, basis, lp = lp.simplex_iteration(basis)

        self.assertEqual(basis, [2, 4], "Should compute correct basis.")
        self.assertTrue(
            np.allclose(solution, [0, 1, 0, 1]), "Should compute correct solution."
        )
        self.assertTrue(np.allclose(A_three, lp.A), "Should compute correct iteration.")
        self.assertTrue(np.allclose(b_three, lp.b), "Should compute correct iteration.")
        self.assertTrue(np.allclose(c_three, lp.c), "Should compute correct iteration.")
        self.assertTrue(np.allclose(z_three, lp.z), "Should compute correct iteration.")

    def test_unsorted_basis(self) -> None:
        A_one = np.array([[1, -1, 1, -1], [2, -1, 1, 0]])
        b_one = np.array([-2, -1])
        c_one = np.array([-1, 1, -2, 4])
        z_one = 0

        A_two = np.array([[1, 0, 0, 1], [0, 1, -1, 2]])
        b_two = np.array([1, 3])
        c_two = np.array([0, 0, -1, 3])
        z_two = 2

        A_three = np.array([[-2, 1, -1, 0], [1, 0, 0, 1]])
        b_three = np.array([1, 1])
        c_three = np.array([-3, 0, -1, 0])
        z_three = 5

        lp = LinearProgram(A_one, b_one, c_one, z_one)
        solution, basis, lp = lp.simplex_iteration([2, 1])

        self.assertEqual(basis, [2, 4], "Should compute correct basis.")
        self.assertIsNone(solution, "Should compute correct solution.")
        self.assertTrue(np.allclose(A_two, lp.A), "Should compute correct iteration.")
        self.assertTrue(np.allclose(b_two, lp.b), "Should compute correct iteration.")
        self.assertTrue(np.allclose(c_two, lp.c), "Should compute correct iteration.")
        self.assertTrue(np.allclose(z_two, lp.z), "Should compute correct iteration.")

        solution, basis, lp = lp.simplex_iteration([4, 2])

        self.assertEqual(basis, [2, 4], "Should compute correct basis.")
        self.assertTrue(
            np.allclose(solution, [0, 1, 0, 1]), "Should compute correct solution."
        )
        self.assertTrue(np.allclose(A_three, lp.A), "Should compute correct iteration.")
        self.assertTrue(np.allclose(b_three, lp.b), "Should compute correct iteration.")
        self.assertTrue(np.allclose(c_three, lp.c), "Should compute correct iteration.")
        self.assertTrue(np.allclose(z_three, lp.z), "Should compute correct iteration.")

    def test_invalid_basis(self) -> None:
        A = np.array([[1, -1, 1, -1], [2, -1, 1, 0]])
        b = np.array([-2, -1])
        c = np.array([-1, 1, -2, 4])
        z = 0

        lp = LinearProgram(A, b, c, z)

        with self.assertRaises(
            ValueError, msg="Should throw exception if given invalid basis."
        ):
            lp.simplex_iteration([2, 3])

    def test_non_sef(self) -> None:
        A = np.array([[1, -1, 1, -1], [2, -1, 1, 0]])
        b = np.array([-2, -1])
        c = np.array([-1, 1, -2, 4])
        z = 0

        lp = LinearProgram(A, b, c, z, negative_variables=[1])

        with self.assertRaises(
            ArithmeticError, msg="Should throw exception if not in SEF."
        ):
            lp.simplex_iteration([1, 2])

    def test_infeasible(self) -> None:
        pass

    def test_unbounded(self) -> None:
        pass


if __name__ == "__main__":
    main()
