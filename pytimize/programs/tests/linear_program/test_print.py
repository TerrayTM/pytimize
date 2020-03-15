import numpy as np

from ... import LinearProgram
from unittest import TestCase, main

# BUG formatted incorrectly (b does not line up)
    # Max [0. 0. 0. 0. 0. 0. -1. -1. -1.]x
    # Subject To:

    # [2.  -1. -2. 1.  0.  0.  1. 0. 0.]     =   [4.]
    # [-2. 3.  1.  0. -1. 0. 0. 1. 0.]x    =   [5.]
    # [1.  -1. -1. 0. 0. -1. 0. 0. 1.]     =   [1.]
    # x ≥ 0

class TestPrint(TestCase):
    def test_print(self):
        A = np.array([
            [2, -1.125, -2, 1, 0, 0.628, 1, 0, 0],
            [-2, 3.984951, 1, 0, -1, 0, 0, 1, 0],
            [1, -1.1, -1, 0, 0, -1, 0, 0, 1]
        ])
        b = np.array([4, -5.1, 1.75])
        c = np.array([0, 0, 0, 0, 0, 0, -1, -1, -1])
        z = 0

        p = LinearProgram(A, b, c, z)
        print(p)

        self.assertTrue(True, True)

if __name__ == "__main__":
    main()