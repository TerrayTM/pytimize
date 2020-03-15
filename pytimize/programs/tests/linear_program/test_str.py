import numpy as np

from ... import LinearProgram
from unittest import TestCase, main

class TestStr(TestCase):
    def test_str(self):
        A = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        b = np.array([6, 15, 24])
        c = np.array([100, 200, 300])
        z = 5
        expected = (
            "Max [100. 200. 300.]x + 5\n"
            "Subject To:\n"
            "\n"
            "[1. 2. 3.]     =   [6. ]\n"
            "[4. 5. 6.]x    =   [15.]\n"
            "[7. 8. 9.]     =   [24.]\n"
            "x ≥ 0\n"
        )
        
        p = LinearProgram(A, b, c, z)
        self.assertEqual(str(p), expected, "Should output in correct string format.")

        A = np.array([
            [1.0, 2.9589, 3],
            [4, 0, -6.0],
            [7.22, -8, 9]
        ])
        b = np.array([-6.11, 15.0, 0])
        c = np.array([0, -20.012, 300.0])
        z = 0
        expected = (
            "Max [0. -20.012 300.]x\n"
            "Subject To:\n"
            "\n"
            "[1.   2.9589 3. ]     ≤   [-6.11]\n"
            "[4.   0.     -6.]x    ≤   [15.  ]\n"
            "[7.22 -8.    9. ]     ≤   [0.   ]\n"
            "x ≥ 0\n"
        )
        
        p = LinearProgram(A, b, c, z, inequalities=["<=", "<=", "<="])
        self.assertEqual(str(p), expected, "Should output in correct string format.")

        A = np.array([
            [100, 9589, -0.693],
            [400, 10.2, -6.056],
            [7, -8, 90.56]
        ])
        b = np.array([-600, 15.0, 0.333])
        c = np.array([5, 5, 35])
        z = 8
        expected = (
            "Min [5. 5. 35.]x + 8\n"
            "Subject To:\n"
            "\n"
            "[100. 9589. -0.693]     ≥   [-600.]\n"
            "[400. 10.2  -6.056]x    ≤   [15.  ]\n"
            "[7.   -8.   90.56 ]     ≤   [0.333]\n"
            "x ≥ 0\n"
        )
        
        p = LinearProgram(A, b, c, z, "min", [">=", "<=", "<="])
        self.assertEqual(str(p), expected, "Should output in correct string format.")

        A = np.array([
            [1, 1, 0],
            [5, 7, -3],
            [-7, -8, 2]
        ])
        b = np.array([1, 15, 1])
        c = np.array([5, 5, 35])
        z = -123
        expected = (
            "Min [5. 5. 35.]x - 123\n"
            "Subject To:\n"
            "\n"
            "[1.  1.  0. ]     ≥   [1. ]\n"
            "[5.  7.  -3.]x    =   [15.]\n"
            "[-7. -8. 2. ]     =   [1. ]\n"
            "x ≥ 0\n"
        )
        
        p = LinearProgram(A, b, c, z, "min", [">=", "=", "="])
        self.assertEqual(str(p), expected, "Should output in correct string format.")

    def test_align_b(self):
        A = np.array([
            [2, -1, -2, 1, 0, 0, 1, 0, 0],
            [-2, 3, 1, 0, -1, 0, 0, 1, 0],
            [1, -1, -1, 0, 0, -1, 0, 0, 1]
        ])
        b = np.array([4, 5, 1])
        c = np.array([0, 0, 0, 0, 0, 0, -1, -1, -1])
        z = -2
        expected = (
            "Min [0. 0. 0. 0. 0. 0. -1. -1. -1.]x - 2\n"
            "Subject To:\n"
            "\n"
            "[ 2.  -1.  -2.  1.   0.   0.  1.  0.  0.]     ≥   [4.]\n"
            "[-2.   3.   1.  0.  -1.   0.  0.  1.  0.]x    =   [5.]\n"
            "[ 1.  -1.  -1.  0.   0.  -1.  0.  0.  1.]     =   [1.]\n"
            "x ≥ 0\n"
        )
        
        p = LinearProgram(A, b, c, z, "min", [">=", "=", "="])
        self.assertEqual(str(p), expected, "Should output in correct string format.")

    def test_rounding(self):
        pass
        #TODO rounding test coverage

if __name__ == "__main__":
    main()
