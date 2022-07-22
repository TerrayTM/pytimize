from unittest import TestCase, main

import numpy as np

from ... import IntegerProgram


class TestBranchAndBound(TestCase):
    def test_branch_and_bound(self):
        A = np.array([[1.5, 1], [9, 11]])
        b = np.array([6, 45])
        c = np.array([60, 50])
        z = 0

        p = IntegerProgram(A, b, c, z, "max", ["<=", "<="])

        self.assertTrue(
            np.allclose(p.branch_and_bound(), [4, 0]), "Should be the same solution."
        )


if __name__ == "__main__":
    main()
