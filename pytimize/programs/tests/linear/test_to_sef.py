import math
from unittest import TestCase, main

import numpy as np

from ... import LinearProgram


class TestToSEF(TestCase):
    def test_to_sef(self) -> None:
        A = np.array([[1, 5, 3], [2, -1, 2], [1, 2, -1]])
        b = np.array([5, 4, 2])
        c = np.array([1, -2, 4])
        z = 0

        p = LinearProgram(A, b, c, z, "min", [">=", "<=", "="], [3])

        p.to_sef(in_place=True)

        self.assertTrue(
            np.allclose(
                p.A,
                np.array(
                    [[1, 5, 3, -3, -1, 0], [2, -1, 2, -2, 0, 1], [1, 2, -1, 1, 0, 0]]
                ),
            ),
            "Should compute correct coefficient matrix in SEF.",
        )
        self.assertTrue(
            np.allclose(p.b, np.array([5, 4, 2])),
            "Should compute correct constraint values in SEF.",
        )
        self.assertTrue(
            np.allclose(p.c, np.array([-1, 2, -4, 4, 0, 0])),
            "Should compute correct coefficient vector in SEF.",
        )
        self.assertTrue(math.isclose(p.z, 0), "Should compute correct constant in SEF.")
        self.assertEqual(
            p.inequalities,
            ["="] * len(b),
            "Should compute correct inequalities in SEF.",
        )
        self.assertEqual(
            p.objective, "max", "Should be maximizing objective function in SEF."
        )
        self.assertEqual(p.free_variables, [], "Should have no free variables.")
        self.assertEqual(p.negative_variables, [], "Should have no negative variables.")

        A = np.array([[1, 2, 0, 1], [1, -2, 16, 0], [8, 2, -3, 1]])
        b = np.array([10, 14, -2])
        c = np.array([-2, 3, -4, 1])
        z = 0

        p = LinearProgram(A, b, c, z, "max", ["<=", "<=", "="], [2])

        p.to_sef(in_place=True)

        self.assertTrue(
            np.allclose(
                p.A,
                np.array(
                    [
                        [1, 2, -2, 0, 1, 1, 0],
                        [1, -2, 2, 16, 0, 0, 1],
                        [8, 2, -2, -3, 1, 0, 0],
                    ]
                ),
            ),
            "Should compute correct coefficient matrix in SEF.",
        )
        self.assertTrue(
            np.allclose(p.b, np.array([10, 14, -2])),
            "Should compute correct constraint values in SEF.",
        )
        self.assertTrue(
            np.allclose(p.c, np.array([-2, 3, -3, -4, 1, 0, 0])),
            "Should compute correct coefficient vector in SEF.",
        )
        self.assertTrue(math.isclose(p.z, 0), "Should compute correct constant in SEF.")
        self.assertEqual(
            p.inequalities,
            ["="] * len(b),
            "Should compute correct inequalities in SEF.",
        )
        self.assertEqual(
            p.objective, "max", "Should be maximizing objective function in SEF."
        )
        self.assertEqual(p.free_variables, [], "Should have no free variables.")
        self.assertEqual(p.negative_variables, [], "Should have no negative variables.")

        A = np.array([[1, 2, 0, 1], [1, -2, 16, 0], [8, 2, -3, 1]])
        b = np.array([10, 14, -2])
        c = np.array([-2, 3, -4, 1])
        z = 0

        p = LinearProgram(A, b, c, z, "max", ["<=", "<=", "="], [2], [3, 4])

        p.to_sef(in_place=True)

        self.assertTrue(
            np.allclose(
                p.A,
                np.array(
                    [
                        [1, 2, -2, 0, -1, 1, 0],
                        [1, -2, 2, -16, 0, 0, 1],
                        [8, 2, -2, 3, -1, 0, 0],
                    ]
                ),
            ),
            "Should compute correct coefficient matrix in SEF.",
        )
        self.assertTrue(
            np.allclose(p.b, np.array([10, 14, -2])),
            "Should compute correct constraint values in SEF.",
        )
        self.assertTrue(
            np.allclose(p.c, np.array([-2, 3, -3, 4, -1, 0, 0])),
            "Should compute correct coefficient vector in SEF.",
        )
        self.assertTrue(math.isclose(p.z, 0), "Should compute correct constant in SEF.")
        self.assertEqual(
            p.inequalities,
            ["="] * len(b),
            "Should compute correct inequalities in SEF.",
        )
        self.assertEqual(
            p.objective, "max", "Should be maximizing objective function in SEF."
        )
        self.assertEqual(p.free_variables, [], "Should have no free variables.")
        self.assertEqual(p.negative_variables, [], "Should have no negative variables.")

    def test_random_order_free_variables(self) -> None:
        A = np.array([[1, 2, 0, 1], [1, -2, 16, 0], [8, 2, -3, 1]])
        b = np.array([10, 14, -2])
        c = np.array([-2, 3, -4, 1])
        z = 0

        p = LinearProgram(A, b, c, z, "max", free_variables=[2, 1])

        p.to_sef(in_place=True)

        self.assertTrue(
            np.allclose(
                p.A,
                np.array(
                    [[1, -1, 2, -2, 0, 1], [1, -1, -2, 2, 16, 0], [8, -8, 2, -2, -3, 1]]
                ),
            ),
            "Should compute correct coefficient matrix in SEF.",
        )
        self.assertTrue(
            np.allclose(p.b, np.array([10, 14, -2])),
            "Should compute correct constraint values in SEF.",
        )
        self.assertTrue(
            np.allclose(p.c, np.array([-2, 2, 3, -3, -4, 1])),
            "Should compute correct coefficient vector in SEF.",
        )
        self.assertTrue(math.isclose(p.z, 0), "Should compute correct constant in SEF.")
        self.assertEqual(
            p.inequalities,
            ["="] * len(b),
            "Should compute correct inequalities in SEF.",
        )
        self.assertEqual(
            p.objective, "max", "Should be maximizing objective function in SEF."
        )
        self.assertEqual(p.free_variables, [], "Should have no free variables.")
        self.assertEqual(p.negative_variables, [], "Should have no negative variables.")

    def test_random_order_negative_variables(self) -> None:
        A = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        b = np.array([10, 14, -2])
        c = np.array([2, 3, 4])
        z = 0

        p = LinearProgram(
            A, b, c, z, "max", free_variables=[2], negative_variables=[3, 1]
        )

        p.to_sef(in_place=True)

        self.assertTrue(
            np.allclose(p.A, np.array([[0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, -1, 0]])),
            "Should compute correct coefficient matrix in SEF.",
        )
        self.assertTrue(
            np.allclose(p.b, np.array([10, 14, -2])),
            "Should compute correct constraint values in SEF.",
        )
        self.assertTrue(
            np.allclose(p.c, np.array([-2, 3, -3, -4])),
            "Should compute correct coefficient vector in SEF.",
        )
        self.assertTrue(math.isclose(p.z, 0), "Should compute correct constant in SEF.")
        self.assertEqual(
            p.inequalities,
            ["="] * len(b),
            "Should compute correct inequalities in SEF.",
        )
        self.assertEqual(
            p.objective, "max", "Should be maximizing objective function in SEF."
        )
        self.assertEqual(p.free_variables, [], "Should have no free variables.")
        self.assertEqual(p.negative_variables, [], "Should have no negative variables.")


if __name__ == "__main__":
    main()
