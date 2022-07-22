from unittest import TestCase, main

import numpy as np

from ... import LinearProgram


class TestStr(TestCase):
    def test_str(self) -> None:
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = np.array([6, 15, 24])
        c = np.array([100, 200, 300])
        z = 5
        expected = (
            "Max [100. 200. 300.]x + 5.\n"
            "Subject To:\n"
            "\n"
            "[1.  2.  3.]     =   [ 6.]\n"
            "[4.  5.  6.]x    =   [15.]\n"
            "[7.  8.  9.]     =   [24.]\n"
            "x ≥ 0\n"
        )

        p = LinearProgram(A, b, c, z)
        self.assertEqual(str(p), expected, "Should output in correct string format.")

        A = np.array([[1.0, 2.9589, 3], [4, 0, -6.0], [7.22, -8, 9]])
        b = np.array([-6.11, 15.0, 0])
        c = np.array([0, -20.012, 300.0])
        z = 0
        expected = (
            "Max [0. -20.012 300.]x\n"
            "Subject To:\n"
            "\n"
            "[1.     2.959   3.]     ≤   [-6.11]\n"
            "[4.     0.     -6.]x    ≤   [15.  ]\n"
            "[7.22  -8.      9.]     ≤   [ 0.  ]\n"
            "x ≥ 0\n"
        )

        p = LinearProgram(A, b, c, z, inequalities=["<=", "<=", "<="])
        self.assertEqual(str(p), expected, "Should output in correct string format.")

        A = np.array([[100, 9589, -0.693], [400, 10.2, -6.056], [7, -8, 90.56]])
        b = np.array([-600, 15.0, 0.333])
        c = np.array([5, 5, 35])
        z = 8
        expected = (
            "Min [5. 5. 35.]x + 8.\n"
            "Subject To:\n"
            "\n"
            "[100.  9589.   -0.693]     ≥   [-600.   ]\n"
            "[400.    10.2  -6.056]x    ≤   [  15.   ]\n"
            "[  7.    -8.   90.56 ]     ≤   [   0.333]\n"
            "x ≥ 0\n"
        )

        p = LinearProgram(A, b, c, z, "min", [">=", "<=", "<="])
        self.assertEqual(str(p), expected, "Should output in correct string format.")

        A = np.array([[1, 1, 0], [5, 7, -3], [-7, -8, 2]])
        b = np.array([1, 15, 1])
        c = np.array([5, 5, 35])
        z = -123
        expected = (
            "Min [5. 5. 35.]x - 123.\n"
            "Subject To:\n"
            "\n"
            "[ 1.   1.   0.]     ≥   [ 1.]\n"
            "[ 5.   7.  -3.]x    =   [15.]\n"
            "[-7.  -8.   2.]     =   [ 1.]\n"
            "x ≥ 0\n"
        )

        p = LinearProgram(A, b, c, z, "min", [">=", "=", "="])
        self.assertEqual(str(p), expected, "Should output in correct string format.")

        A = np.array(
            [
                [2, -1.125, -2, 1, 0, 0.628, 1, 0, 0],
                [-2, 3.984951, 1, 0, -1, 0, 0, 1, 0],
                [1, -1.1, -1, 0, 0, -1, 0, 0, 1],
            ]
        )
        b = np.array([4, -5.1, 1.75])
        c = np.array([0, 0, 0, 0, 0, 0, -1, -1, -1])
        z = 0
        expected = (
            "Max [0. 0. 0. 0. 0. 0. -1. -1. -1.]x\n"
            "Subject To:\n"
            "\n"
            "[ 2.  -1.125  -2.  1.   0.   0.628  1.  0.  0.]     =   [ 4.  ]\n"
            "[-2.   3.985   1.  0.  -1.   0.     0.  1.  0.]x    =   [-5.1 ]\n"
            "[ 1.  -1.1    -1.  0.   0.  -1.     0.  0.  1.]     =   [ 1.75]\n"
            "x ≥ 0\n"
        )

        p = LinearProgram(A, b, c, z)
        self.assertEqual(str(p), expected, "Should output in correct string format.")

    def test_align_b(self) -> None:
        A = np.array(
            [
                [2, -1, -2, 1, 0, 0, 1, 0, 0],
                [-2, 3, 1, 0, -1, 0, 0, 1, 0],
                [1, -1, -1, 0, 0, -1, 0, 0, 1],
            ]
        )
        b = np.array([4, 5, 1])
        c = np.array([0, 0, 0, 0, 0, 0, -1, -1, -1])
        z = -2
        expected = (
            "Min [0. 0. 0. 0. 0. 0. -1. -1. -1.]x - 2.\n"
            "Subject To:\n"
            "\n"
            "[ 2.  -1.  -2.  1.   0.   0.  1.  0.  0.]     ≥   [4.]\n"
            "[-2.   3.   1.  0.  -1.   0.  0.  1.  0.]x    =   [5.]\n"
            "[ 1.  -1.  -1.  0.   0.  -1.  0.  0.  1.]     =   [1.]\n"
            "x ≥ 0\n"
        )

        p = LinearProgram(A, b, c, z, "min", [">=", "=", "="])
        self.assertEqual(str(p), expected, "Should output in correct string format.")

    def test_rounding(self) -> None:
        A = np.array(
            [
                [1, 100000.123456789, -0.0123456789],
                [0, -10.223456789, -6.00000567],
                [-1.12345, -7.9995, 90.56],
            ]
        )
        b = np.array([-600.01234567, 15.00000005, 0.999999])
        c = np.array([5, 5, 35])
        z = 8
        expected = (
            "Min [5. 5. 35.]x + 8.\n"
            "Subject To:\n"
            "\n"
            "[ 1.     100000.123  -0.01235]     ≥   [-600.012]\n"
            "[ 0.        -10.223  -6.0    ]x    ≤   [  15.0  ]\n"
            "[-1.123      -8.0    90.56   ]     ≤   [   1.0  ]\n"
            "x ≥ 0\n"
        )

        p = LinearProgram(A, b, c, z, "min", [">=", "<=", "<="])
        self.assertEqual(str(p), expected, "Should output in correct string format.")

    def test_variable_constraints(self) -> None:
        A = np.array([[1, 2, 3, 4, 5, 6], [0, 1, 0, 1, 0, 1]])
        b = np.array([1, 2])
        c = np.array([4, 5, 6, 7, 8, 9])
        z = -2

        expected_one = (
            "Min [4. 5. 6. 7. 8. 9.]x - 2.\n"
            "Subject To:\n"
            "\n"
            "[1.  2.  3.  4.  5.  6.]     =   [1.]\n"
            "[0.  1.  0.  1.  0.  1.]x    =   [2.]\n"
            "x₂, x₄, x₆ ≤ 0\n"
            "x₁, x₃, x₅ ≥ 0\n"
        )
        p = LinearProgram(A, b, c, z, "min", negative_variables=[2, 4, 6])
        self.assertEqual(
            str(p), expected_one, "Should output in correct string format."
        )

        expected_two = (
            "Min [4. 5. 6. 7. 8. 9.]x - 2.\n"
            "Subject To:\n"
            "\n"
            "[1.  2.  3.  4.  5.  6.]     =   [1.]\n"
            "[0.  1.  0.  1.  0.  1.]x    =   [2.]\n"
        )
        p = LinearProgram(A, b, c, z, "min", free_variables=[1, 2, 3, 4, 5, 6])
        self.assertEqual(
            str(p), expected_two, "Should output in correct string format."
        )

        expected_three = (
            "Min [4. 5. 6. 7. 8. 9.]x - 2.\n"
            "Subject To:\n"
            "\n"
            "[1.  2.  3.  4.  5.  6.]     =   [1.]\n"
            "[0.  1.  0.  1.  0.  1.]x    =   [2.]\n"
            "x ≤ 0\n"
        )
        p = LinearProgram(A, b, c, z, "min", negative_variables=[1, 2, 3, 4, 5, 6])
        self.assertEqual(
            str(p), expected_three, "Should output in correct string format."
        )

        expected_four = (
            "Min [4. 5. 6. 7. 8. 9.]x - 2.\n"
            "Subject To:\n"
            "\n"
            "[1.  2.  3.  4.  5.  6.]     =   [1.]\n"
            "[0.  1.  0.  1.  0.  1.]x    =   [2.]\n"
            "x ≥ 0\n"
        )
        p = LinearProgram(A, b, c, z, "min")
        self.assertEqual(
            str(p), expected_four, "Should output in correct string format."
        )

        expected_five = (
            "Min [4. 5. 6. 7. 8. 9.]x - 2.\n"
            "Subject To:\n"
            "\n"
            "[1.  2.  3.  4.  5.  6.]     =   [1.]\n"
            "[0.  1.  0.  1.  0.  1.]x    =   [2.]\n"
            "x₂, x₃, x₄, x₆ ≥ 0\n"
        )
        p = LinearProgram(A, b, c, z, "min", free_variables=[1, 5])
        self.assertEqual(
            str(p), expected_five, "Should output in correct string format."
        )

        expected_six = (
            "Min [4. 5. 6. 7. 8. 9.]x - 2.\n"
            "Subject To:\n"
            "\n"
            "[1.  2.  3.  4.  5.  6.]     =   [1.]\n"
            "[0.  1.  0.  1.  0.  1.]x    =   [2.]\n"
            "x₂, x₆ ≤ 0\n"
            "x₃, x₄ ≥ 0\n"
        )
        p = LinearProgram(
            A, b, c, z, "min", free_variables=[1, 5], negative_variables=[2, 6]
        )
        self.assertEqual(
            str(p), expected_six, "Should output in correct string format."
        )

    def test_long_variable_index(self) -> None:
        A = np.array(
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
            ]
        )
        b = np.array([1, 2])
        c = np.array(
            [
                4.4877,
                5.123456,
                6.14564,
                7.444,
                88779798,
                9,
                11,
                12,
                13e10,
                45e13,
                1.1547841e-7,
                1,
                12,
            ]
        )
        z = 125.1549879465
        expected = (
            "Min [4.488 5.123 6.146 7.444 88779798. 9. 11. 12. 1.300e+11 4.500e+14 1.155e-07 1. 12.]x + 125.155\n"
            "Subject To:\n"
            "\n"
            "[1.  2.  3.  4.  5.  6.  7.  8.  9.  10.  11.  12.  13.]     =   [1.]\n"
            "[0.  1.  0.  1.  0.  1.  0.  1.  0.   1.   0.   1.   1.]x    =   [2.]\n"
            "x₁₁, x₁₃ ≤ 0\n"
            "x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈, x₉, x₁₀, x₁₂ ≥ 0\n"
        )

        p = LinearProgram(A, b, c, z, "min", negative_variables=[11, 13])
        self.assertEqual(str(p), expected, "Should output in correct string format.")

    def test_scientific_notation(self) -> None:
        A = np.array(
            [
                [1.123e30, 456, 1.123555557894e13],
                [1e-5, 6.123e-8, -6.123456789456],
                [-1e-10, 7e17, 1.254e11],
                [1e-11, 5.499999e-4, -10.1248683e28],
            ]
        )
        b = np.array([-600e23, 19e13, 1.87e-4, 799.456])
        c = np.array([1, 2, 3])
        z = -2
        expected = (
            "Min [1. 2. 3.]x - 2.\n"
            "Subject To:\n"
            "\n"
            "[ 1.123e+30  456.          1.124e+13]     =   [ -6.000e+25]\n"
            "[ 1.000e-05    6.123e-08  -6.123    ]     =   [  1.900e+14]\n"
            "[-1.000e-10    7.000e+17   1.254e+11]x    =   [  0.000187 ]\n"
            "[ 0.           0.00055    -1.012e+29]     =   [799.456    ]\n"
            "x ≥ 0\n"
        )

        p = LinearProgram(A, b, c, z, "min")
        self.assertEqual(str(p), expected, "Should output in correct string format.")

        A = np.array(
            [
                [0.999e-4, -0.0000012335, -0.0000012336],
                [0.00000499999, 123456789.0000012336, 234567890123.4567890123],
            ]
        )
        b = np.array([1.0000012336, 0.0000012336])
        c = np.array([-123456789123456789, 0.000003456789, 0.999e-11])
        z = -0.0000000059999999999
        expected = (
            "Min [-1.235e+17 3.457e-06 0.]x - 6.000e-09\n"
            "Subject To:\n"
            "\n"
            "[9.990e-05         -1.234e-06  -1.234e-06]     =   [1.0      ]\n"
            "[5.000e-06  123456789.0         2.346e+11]x    =   [1.234e-06]\n"
            "x ≥ 0\n"
        )

        p = LinearProgram(A, b, c, z, "min")
        self.assertEqual(str(p), expected, "Should output in correct string format.")


if __name__ == "__main__":
    main()
